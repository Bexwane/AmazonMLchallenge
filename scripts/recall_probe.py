"""Full-density retrieval-channel probe (blocking only, no model).

Recall of a blocker only depends on how each Source-1 query ranks the *full* S2/S3 pool of its
country, so a random sample of queries searched against the complete pool measures full-density
recall at a fraction of the cost. Candidate counts are the sample's pairs-per-query scaled to all
Source-1 records of the country (estimates; the chosen union is then measured on all S1 by
scripts/eval_blocking_full.py).

Step 1, per country (one process each keeps peak RAM low):
    python scripts/recall_probe.py --data D --work W --countries India --out OUT
  -> OUT/pairs_<country>.parquet  every retrieved (query, pool) pair with channel, rank, is-true
  -> OUT/probe.json               per-channel recall, gain over base, new candidates, runtime, RAM
  -> OUT/misses_<country>.parquet base misses with failure mode and which channel recovers them
Step 2, pooled over countries:
    python scripts/recall_probe.py --combine --out OUT
  -> OUT/experiments.json, OUT/e007_spec.txt   E003-E007 unions chosen greedily (no RRF, no cut)
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
from ber.buckets import assign_modes, name_frequency, pair_table  # noqa: E402
from ber.io import read_ground_truth  # noqa: E402
from ber.mem import peak_gib, rss_gib  # noqa: E402
from ber.prepare import load_prepared  # noqa: E402

DEFAULT_CHANNELS = "base,base_wide:200,addr_only:20,name_noaddr:10,conj:50,rescue,bm25:50"
DETERMINISTIC = ("addr_only", "name_noaddr", "conj", "rescue")
LEXICAL = ("bm25",)
MULTILINGUAL = ("bge_native",)
DEPTHS = (5, 10, 20, 30, 50)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def _ranked(qa, rb, v, name):
    d = pd.DataFrame({"q": qa, "r": rb, "v": v}).sort_values(["q", "v"], ascending=[True, False], kind="stable")
    d = d.drop_duplicates(["q", "r"])
    d["rank"] = (d.groupby("q").cumcount() + 1).astype(np.float32)
    d["ch"] = name
    return d[["q", "r", "rank", "ch"]].reset_index(drop=True)


def _key(q, r):
    return (np.asarray(q, np.int64) << 32) | np.asarray(r, np.int64)


def probe_country(c, q, pool, tq, tr, specs, out):
    true_keys = np.unique(_key(tq, tr))
    n_true, n_q = len(true_keys), len(q)
    rep = {"queries": n_q, "pool": len(pool), "true_pairs": n_true, "channels": {}}
    frames, keysets = [], {}
    for item, name, kw in specs:
        t = time.time()
        qa, rb, v = ch.CHANNELS[name](q, pool, **kw)
        d = _ranked(qa, rb, v, name)
        k = _key(d["q"], d["r"])
        d["y"] = np.isin(k, true_keys)
        keysets[name] = np.unique(k)
        frames.append(d)
        r = {"spec": item, "recall": float(d["y"].sum() / n_true), "pairs_per_q": len(d) / n_q,
             "sec": round(time.time() - t, 1), "rss_gib_after": round(rss_gib(), 2), "peak_gib": peak_gib()}
        for depth in (10, 20, 30, 45):
            r[f"recall@{depth}"] = float(d.loc[d["rank"] <= depth, "y"].sum() / n_true)
        rep["channels"][name] = r
        log(c, name, json.dumps(r))
    if "base" in keysets:
        base_true = np.intersect1d(keysets["base"], true_keys)
        for name, ks in keysets.items():
            if name == "base":
                continue
            new = np.setdiff1d(ks, keysets["base"])
            rc = rep["channels"][name]
            rc["gain_over_base"] = len(np.setdiff1d(np.intersect1d(ks, true_keys), base_true)) / n_true
            rc["new_pairs_per_q"] = len(new) / n_q
            rc["candidate_increase_pct"] = 100 * len(new) / len(keysets["base"])
    allp = pd.concat(frames, ignore_index=True)
    allp["q"] = allp["q"].astype(np.int32)
    allp["r"] = allp["r"].astype(np.int32)
    allp["ch"] = allp["ch"].astype("category")
    allp.to_parquet(out / f"pairs_{c}.parquet", index=False)
    union = np.unique(_key(allp["q"], allp["r"]))
    rep["union_all_channels"] = {"recall": len(np.intersect1d(union, true_keys)) / n_true,
                                 "pairs_per_q": len(union) / n_q}
    log(c, "union of all channels (no cut)", json.dumps(rep["union_all_channels"]))
    # base misses: diagnosis, failure mode, and which channel recovers each one
    if "base" in keysets:
        tk = _key(tq, tr)
        miss = ~np.isin(tk, keysets["base"])
        d = pd.DataFrame(ch.base_diagnose(q, pool, tq[miss], tr[miss]))
        d["comb"] = d["name_cos"] + d["addr_cos"]
        d["s1_idx"], d["r_idx"] = tq[miss], tr[miss]
        t = pair_table(d, q, pool, name_frequency(pool))
        t["mode"] = assign_modes(t)
        for name, ks in keysets.items():
            if name != "base":
                t["hit_" + name] = np.isin(_key(t["s1_idx"], t["r_idx"]), ks)
        t.to_parquet(out / f"misses_{c}.parquet", index=False)
        rep["base_miss_modes"] = t["mode"].value_counts(normalize=True).round(4).to_dict()
        hits = [x for x in t.columns if x.startswith("hit_")]
        rep["recovery_by_mode"] = t.groupby("mode")[hits].mean().round(3).to_dict() if hits else {}
    return rep


def run_countries(args):
    specs = ch.parse_spec(args.channels)
    s1, s23 = load_prepared(args.data, "train", args.work)
    truth = read_ground_truth(Path(args.data) / "train" / "train_ground_truth.tsv")
    log(f"loaded {len(s1)} S1, {len(s23)} S2/S3; rss {rss_gib():.1f} GiB")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pj = out / "probe.json"
    report = json.loads(pj.read_text()) if pj.exists() else {}
    countries = args.countries.split(",") if args.countries else sorted(set(s1["country"]))
    n_full = s1["country"].value_counts().to_dict()
    s1 = s1[s1["country"].isin(countries)].reset_index(drop=True)
    s23 = s23[s23["country"].isin(countries)].reset_index(drop=True)
    gc.collect()
    for c in countries:
        rng = np.random.default_rng([args.seed, sum(map(ord, c))])  # per-country stream: order-independent
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
        t = time.time()
        rep = probe_country(c, q, pool, tq, tr, specs, out)
        rep["n_s1_full"] = int(n_full[c])
        rep["sec_total"] = round(time.time() - t, 1)
        rep["peak_gib"] = peak_gib()
        rep["dense_info"] = dict(ch.DENSE_INFO)
        report[c] = rep
        ch.clear()
        ch.DENSE_INFO.clear()
        del q, pool
        gc.collect()
        pj.write_text(json.dumps(report, indent=1))  # saved after every country


# ---------------------------------------------------------------------------
# Step 2: pooled greedy union selection. No RRF and no truncation: recall is measured on the union.
# Each step adds the (channel, depth) with the best recall gain per extra candidate per S1.
# ---------------------------------------------------------------------------
def _options(P, allowed):
    opts = []
    for name in allowed:
        d = P[P["ch"] == name]
        if not len(d):
            continue
        if name == "rescue":
            opts.append((name, None, d["key"].to_numpy()))
            continue
        top = int(d["rank"].max())
        for depth in sorted({x for x in DEPTHS if x < top} | {top}):
            opts.append((name, depth, d.loc[d["rank"] <= depth, "key"].to_numpy()))
    return opts


def combine(args):
    out = Path(args.out)
    rep = json.loads((out / "probe.json").read_text())
    countries = sorted(rep)
    w = np.array([rep[c]["n_s1_full"] / rep[c]["queries"] for c in countries])
    n_s1 = sum(rep[c]["n_s1_full"] for c in countries)
    frames = []
    for i, c in enumerate(countries):
        p = pd.read_parquet(out / f"pairs_{c}.parquet")
        p["key"] = (np.int64(i) << 60) | _key(p["q"], p["r"])
        p["ch"] = p["ch"].astype(str)
        frames.append(p[["key", "ch", "rank", "y"]])
    P = pd.concat(frames, ignore_index=True)
    true_w = sum(w[i] * rep[c]["true_pairs"] for i, c in enumerate(countries))
    hit_keys = np.unique(P.loc[P["y"], "key"].to_numpy())

    def measure(keys):
        k = np.unique(keys)
        cid = (k >> 60).astype(int)
        hit = np.isin(k, hit_keys)
        per_c = {c: float(hit[cid == i].sum() / rep[c]["true_pairs"]) for i, c in enumerate(countries)}
        return float(np.sum(w[cid] * hit) / true_w), float(np.sum(w[cid])), per_c

    base_keys = np.unique(P.loc[P["ch"] == "base", "key"].to_numpy())
    b_rec, b_cand, b_pc = measure(base_keys)

    def row(label, spec, rec, cand, pc, steps=()):
        return {"experiment": label, "spec": spec, "pair_recall": rec, "recall_by_country": pc,
                "candidates_est": cand, "pairs_per_s1": cand / n_s1, "incremental_recall": rec - b_rec,
                "steps": list(steps)}

    def greedy(allowed, label):
        cur, rec, cand, pc = base_keys, b_rec, b_cand, b_pc
        steps, depth_of = [], {}
        opts = _options(P, allowed)
        while True:
            best = None
            for name, depth, keys in opts:
                if name in depth_of and (depth is None or depth <= depth_of[name]):
                    continue  # already included at this depth or deeper
                nk = np.union1d(cur, keys)
                r2, c2, pc2 = measure(nk)
                gain, extra = r2 - rec, (c2 - cand) / n_s1
                eff = gain / max(extra, 1e-9)
                if gain >= args.min_gain and eff >= args.min_eff and c2 / n_s1 <= args.max_pairs:
                    if best is None or eff > best[0]:
                        best = (eff, name, depth, nk, r2, c2, pc2, gain, extra)
            if best is None:
                break
            eff, name, depth, cur, r2, c2, pc2, gain, extra = best
            depth_of[name] = depth
            steps.append({"add": name if depth is None else f"{name}:{depth}", "recall": r2, "gain": gain,
                          "extra_pairs_per_s1": extra, "recall_per_extra_pair": eff, "pairs_per_s1": c2 / n_s1})
            rec, cand, pc = r2, c2, pc2
        spec = ",".join(["base"] + [n if d is None else f"{n}:{d}" for n, d in depth_of.items()])
        return row(label, spec, rec, cand, pc, steps)

    have = set(P["ch"].unique())
    exps = [row("E003 baseline blocker", "base", b_rec, b_cand, b_pc)]
    exps.append(greedy([x for x in DETERMINISTIC if x in have], "E004 base + deterministic recovery"))
    exps.append(greedy([x for x in LEXICAL if x in have], "E005 base + BM25"))
    if "bge_native" in have:
        exps.append(greedy(["bge_native"], "E006 base + BGE-M3 targeted (native-script pool)"))
    if "e5_native" in have:
        exps.append(greedy(["e5_native"], "E006b base + e5-small targeted (ablation)"))
    pool_all = [x for x in DETERMINISTIC + LEXICAL + MULTILINGUAL if x in have]
    e7 = greedy(pool_all, "E007 best union (greedy over all useful channels)")
    if "e5_native" in have:  # e5 only counts if it adds recall on top of everything else
        e7["e5_on_top_gain"] = greedy(pool_all + ["e5_native"], "E007+e5")["pair_recall"] - e7["pair_recall"]
    exps.append(e7)
    ref = P.loc[P["ch"] == "base_wide", "key"].to_numpy()
    if len(ref):
        exps.append(row("ref: E003 score at K=200 (what 'more K' buys)", "base,base_wide:200",
                        *measure(np.union1d(base_keys, ref))))
    exps.append(row("ref: union of every probed channel", "all", *measure(P["key"].to_numpy())))
    res = {"weights": dict(zip(countries, w.tolist())), "n_s1_full": n_s1,
           "selection": {"min_gain": args.min_gain, "min_eff": args.min_eff, "max_pairs_per_s1": args.max_pairs},
           "experiments": exps, "channels": {c: rep[c]["channels"] for c in countries},
           "base_miss_modes": {c: rep[c].get("base_miss_modes") for c in countries},
           "recovery_by_mode": {c: rep[c].get("recovery_by_mode") for c in countries},
           "dense_info": {c: rep[c].get("dense_info", {}) for c in countries},
           "peak_gib": {c: rep[c].get("peak_gib") for c in countries},
           "sec_total": {c: rep[c].get("sec_total") for c in countries}}
    (out / "experiments.json").write_text(json.dumps(res, indent=1))
    (out / "e007_spec.txt").write_text(e7["spec"])
    for e in exps:
        pc = {k: round(v, 4) for k, v in e["recall_by_country"].items()}
        log(f"{e['experiment']}: recall {e['pair_recall']:.4f} {pc} pairs/S1 {e['pairs_per_s1']:.1f} spec={e['spec']}")
    log("E007 spec ->", e7["spec"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data")
    ap.add_argument("--work")
    ap.add_argument("--n", type=int, default=10000, help="sampled S1 queries per country")
    ap.add_argument("--channels", default=DEFAULT_CHANNELS)
    ap.add_argument("--countries", default="")
    ap.add_argument("--out", default="reports/recall_probe")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--combine", action="store_true", help="pool countries and select E003-E007 unions")
    ap.add_argument("--min-gain", type=float, default=0.002, help="min pooled recall gain for a channel to join")
    ap.add_argument("--min-eff", type=float, default=0.0005, help="min recall gain per extra candidate per S1")
    ap.add_argument("--max-pairs", type=float, default=80, help="max union candidates per S1 (memory budget)")
    args = ap.parse_args()
    if args.combine:
        combine(args)
    else:
        run_countries(args)


if __name__ == "__main__":
    main()
