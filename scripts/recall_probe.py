"""Full-density blocking-recall probe.

Recall only depends on how each Source-1 query ranks the *full* S2/S3 pool of its country, so a
random sample of queries searched against the complete pool gives the exact full-density recall
at a fraction of the cost. For every retrieval channel we report its recall alone, its gain over
the baseline blocker, the recall of the union, and the recall left after truncating the union to
the top-M per query by reciprocal-rank fusion (so any truncation loss is measured, not assumed).
Base misses are dumped with a diagnosis and a per-channel recovery flag.

usage: python scripts/recall_probe.py --data DATA --work WORK [--n 10000] [--channels base,conj,...]
"""
import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ber import channels as ch  # noqa: E402
from ber.io import read_ground_truth  # noqa: E402
from ber.prepare import load_prepared  # noqa: E402

TRUNC = (30, 50, 80, 120, 200)


def _ranked(qa, rb, v, name):
    d = pd.DataFrame({"q": qa, "r": rb, "v": v}).sort_values("v", ascending=False, kind="stable")
    d = d.drop_duplicates(["q", "r"])
    d["rank"] = d.groupby("q").cumcount() + 1
    d["ch"] = name
    return d[["q", "r", "rank", "ch"]].reset_index(drop=True)


def _is_true(d, truekey):
    return np.fromiter(((a, b) in truekey for a, b in zip(d["q"].tolist(), d["r"].tolist())), bool, len(d))


def probe_country(c, q, pool, tq, tr, names, rrf_names, log):
    truekey = set(zip(tq.tolist(), tr.tolist()))
    n_true, n_q = len(tq), len(q)
    rep = {"queries": n_q, "pool": len(pool), "true_pairs": n_true}
    got = {}
    for name in names:
        t = time.time()
        qa, rb, v = ch.CHANNELS[name](q, pool)
        d = _ranked(qa, rb, v, name)
        d["y"] = _is_true(d, truekey)
        got[name] = d
        rep[name] = {"recall": float(d["y"].sum() / n_true), "pairs_per_q": len(d) / n_q,
                     "sec": round(time.time() - t, 1)}
        for k in (10, 20, 45):
            rep[name][f"recall@{k}"] = float(d.loc[d["rank"] <= k, "y"].sum() / n_true)
        log(c, name, json.dumps(rep[name]))
    keyset = {n: set(zip(d.loc[d["y"], "q"].tolist(), d.loc[d["y"], "r"].tolist())) for n, d in got.items()}
    if "base" in got:
        for n in names:
            if n != "base":
                rep[n]["gain_over_base"] = len(keyset[n] - keyset["base"]) / n_true
    union = pd.concat(got.values(), ignore_index=True)
    u = union.drop_duplicates(["q", "r"])
    rep["union"] = {"recall": float(u["y"].sum() / n_true), "pairs_per_q": len(u) / n_q}
    # reciprocal-rank fusion over the chosen channels, then keep the top-M per query
    f = union[union["ch"].isin(rrf_names)].copy()
    if len(f):
        f["s"] = 1.0 / (60 + f["rank"])
        f = f.groupby(["q", "r"], as_index=False).agg(s=("s", "sum"), y=("y", "max"))
        f = f.sort_values(["q", "s"], ascending=[True, False], kind="stable")
        f["rank"] = f.groupby("q").cumcount() + 1
        rep["rrf"] = {"channels": rrf_names, "recall_all": float(f["y"].sum() / n_true),
                      "pairs_per_q": len(f) / n_q}
        for m in TRUNC:
            rep["rrf"][f"recall@{m}"] = float(f.loc[f["rank"] <= m, "y"].sum() / n_true)
    log(c, "union", json.dumps(rep["union"]), "rrf", json.dumps(rep.get("rrf")))
    misses = []
    if "base" in got:
        d = pd.DataFrame(ch.base_diagnose(q, pool, tq, tr))
        hit = np.array([(a, b) in keyset["base"] for a, b in zip(tq.tolist(), tr.tolist())])
        m = d[~hit]
        rk = m["comb_rank"]
        rep["base_miss_taxonomy"] = {
            "n": int(len(m)),
            "no_surviving_key": float((rk == 0).mean()),
            "rank_46_100": float(((rk > 45) & (rk <= 100)).mean()),
            "rank_101_1000": float(((rk > 100) & (rk <= 1000)).mean()),
            "rank_gt_1000": float((rk > 1000).mean()),
            "no_raw_name_key": float((m["name_shared_raw"] == 0).mean()),
            "no_raw_addr_key": float((m["addr_shared_raw"] == 0).mean()),
            "no_raw_key_at_all": float(((m["name_shared_raw"] == 0) & (m["addr_shared_raw"] == 0)).mean()),
            "r_native_script": float(pool["name_native"].to_numpy()[tr[~hit]].mean()),
            "r_addr_empty": float(pool["addr_empty"].to_numpy()[tr[~hit]].mean()),
        }
        log(c, "base misses", json.dumps(rep["base_miss_taxonomy"]))
        for i in np.flatnonzero(~hit):
            j, r = int(tq[i]), int(tr[i])
            row = {"country": c, "s1": q.at[j, "entity_id"], "s1_name": q.at[j, "business_name"],
                   "s1_addr": q.at[j, "business_address"], "r": pool.at[r, "entity_id"],
                   "r_name": pool.at[r, "business_name"], "r_addr": pool.at[r, "business_address"]}
            row.update({k: d.at[i, k] for k in d.columns})
            row.update({"hit_" + n: (j, r) in keyset[n] for n in names})
            misses.append(row)
    return rep, misses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--n", type=int, default=10000, help="sampled S1 queries per country")
    ap.add_argument("--channels", default="base,base_wide,addr_only,name_noaddr,conj,rescue,bm25")
    ap.add_argument("--rrf", default="base_wide,addr_only,name_noaddr,conj,rescue,bm25,dense")
    ap.add_argument("--countries", default="")
    ap.add_argument("--out", default="reports/recall_probe")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)
    s1, s23 = load_prepared(args.data, "train", args.work)
    truth = read_ground_truth(Path(args.data) / "train" / "train_ground_truth.tsv")
    log(f"loaded {len(s1)} S1, {len(s23)} S2/S3")
    names = args.channels.split(",")
    rrf_names = [n for n in args.rrf.split(",") if n in names]
    rng = np.random.default_rng(args.seed)
    countries = args.countries.split(",") if args.countries else sorted(set(s1["country"]))
    s1 = s1[s1["country"].isin(countries)].reset_index(drop=True)
    s23 = s23[s23["country"].isin(countries)].reset_index(drop=True)
    gc.collect()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report, misses = {}, []
    if (out / "probe.json").exists():  # results of earlier invocations (e.g. one process per country)
        report = json.loads((out / "probe.json").read_text())
        old = pd.read_csv(out / "misses.tsv", sep="	", dtype=str, keep_default_na=False)
        misses = old[~old["country"].isin(countries)].to_dict("records") if len(old) else []
    for c in countries:
        ia_all = np.flatnonzero((s1["country"] == c).to_numpy())
        ia = np.sort(rng.choice(ia_all, min(args.n, len(ia_all)), replace=False))
        ib = np.flatnonzero((s23["country"] == c).to_numpy())
        q, pool = s1.iloc[ia].reset_index(drop=True), s23.iloc[ib].reset_index(drop=True)
        if len(countries) == 1:  # one country per process: the full frames are no longer needed
            del s1, s23
            gc.collect()
        pairs = [(j, x) for j, e in enumerate(q["entity_id"]) for x in truth.get(e, ())]
        tq = np.array([j for j, _ in pairs], np.int64)
        tr = pd.Index(pool["entity_id"]).get_indexer([x for _, x in pairs]).astype(np.int64)
        assert (tr >= 0).all(), "a true match is missing from its country pool (cross-country pair?)"
        report[c], m = probe_country(c, q, pool, tq, tr, names, rrf_names, log)
        misses += m
        ch.clear()
        del q, pool
        gc.collect()
        (out / "probe.json").write_text(json.dumps(report, indent=1))  # saved after every country
        pd.DataFrame(misses).to_csv(out / "misses.tsv", sep="\t", index=False)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
