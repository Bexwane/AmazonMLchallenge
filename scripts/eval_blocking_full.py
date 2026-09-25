"""Blocking-only evaluation on ALL train Source-1 records against ALL S2/S3 records (no sampling).

Writes (under --exp-dir, default WORK/experiments/<exp>):
  blocking_full.json        candidate count, pairs/S1, pair recall (overall/country/source), recall@k,
                            oracle macro F0.5, present/absent split, runtime, peak RAM
  error_buckets.parquet     every absent true pair + a sample of found pairs, with bucket features
and, when --reports is given, reports/FULL_DENSITY_BASELINE.md and reports/ERROR_BUCKETS.md.
The candidate set is cached at WORK/train/cand_<tag>.parquet, the same file `ber.run` uses.

  python scripts/eval_blocking_full.py --data D --work W --exp E003-fullblock --k-comb 45 --k-name 10 --df-cap 2500
  python scripts/eval_blocking_full.py --data D --work W --exp E007-fullblock --channels "base,conj:20,..."
"""
import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ber.blocking import _weight, generate_candidates, key_matrices  # noqa: E402
from ber.buckets import name_frequency, pair_table, write_report  # noqa: E402
from ber.channels import retrieve  # noqa: E402
from ber.io import read_ground_truth  # noqa: E402
from ber.labels import blocking_report, label_pairs  # noqa: E402
from ber.mem import peak_gib, rss_gib  # noqa: E402
from ber.prepare import load_prepared  # noqa: E402
from ber.run import block_tag  # noqa: E402

T0 = time.time()


def log(*a):
    print(time.strftime("%H:%M:%S"), f"[{(time.time() - T0) / 60:.1f} min, rss {rss_gib():.1f} GiB]", *a, flush=True)


def _rowdot(A, B, ia, ib, chunk=1_000_000):
    out = np.empty(len(ia), np.float32)
    for s in range(0, len(ia), chunk):
        out[s:s + chunk] = np.asarray(A[ia[s:s + chunk]].multiply(B[ib[s:s + chunk]]).sum(1)).ravel()
    return out


class Diagnoser:
    """Scores every true pair with the E003 key matrices of its country (name_cos, addr_cos, shared raw
    keys) and, for a random sample of true pairs, its rank of name+addr among the whole country pool."""

    def __init__(self, tp, rank_sample, seed=0):
        self.tp = tp
        for c in ("name_cos", "addr_cos", "name_shared_raw", "addr_shared_raw"):
            tp[c] = np.float32(np.nan)
        tp["pool_rank"] = -1
        rng = np.random.default_rng(seed)
        self.sample = np.zeros(len(tp), bool)
        self.sample[rng.choice(len(tp), min(rank_sample, len(tp)), replace=False)] = True

    def __call__(self, country, ia, ib, M):
        tp = self.tp
        rows = np.flatnonzero((tp["country"] == country).to_numpy())
        qa = np.searchsorted(ia, tp["s1_idx"].to_numpy()[rows])
        rb = np.searchsorted(ib, tp["r_idx"].to_numpy()[rows])
        name, addr = _rowdot(M["An"], M["Bn"], qa, rb), _rowdot(M["Aa"], M["Ba"], qa, rb)
        tp.loc[rows, "name_cos"], tp.loc[rows, "addr_cos"] = name, addr
        tp.loc[rows, "name_shared_raw"] = _rowdot(M["An0"], M["Bn0"], qa, rb)
        tp.loc[rows, "addr_shared_raw"] = _rowdot(M["Aa0"], M["Ba0"], qa, rb)
        # full-pool rank for the sampled pairs
        s = self.sample[rows]
        sq, sr, sc = qa[s], rb[s], (name + addr)[s]
        Ac = sp.hstack([M["An"], M["Aa"]]).tocsr()
        uq, inv = np.unique(sq, return_inverse=True)
        rank = np.zeros(len(sq), np.int64)
        for a in range(0, len(uq), 500):
            S = (Ac[uq[a:a + 500]] @ M["Bct"]).tocsr()
            for i in np.flatnonzero((inv >= a) & (inv < a + 500)):
                if sc[i] > 0:
                    j = inv[i] - a
                    rank[i] = int((S.data[S.indptr[j]:S.indptr[j + 1]] > sc[i] + 1e-6).sum()) + 1
        tp.loc[rows[s], "pool_rank"] = rank
        log(f"  diagnosed {len(rows)} true pairs in {country} ({int(s.sum())} ranked in full pool)")


def diagnose_cached(s1, s23, diag, df_cap):
    """Rebuild the E003 per-country matrices (used when the candidate set came from the cache)."""
    An_all, Aa_all = key_matrices(s1)
    Bn_all, Ba_all = key_matrices(s23)
    for country in sorted(set(s1["country"]) | set(s23["country"])):
        ia = np.flatnonzero((s1["country"] == country).to_numpy())
        ib = np.flatnonzero((s23["country"] == country).to_numpy())
        if len(ia) == 0 or len(ib) == 0:
            continue
        M = {"An0": An_all[ia], "Aa0": Aa_all[ia], "Bn0": Bn_all[ib], "Ba0": Ba_all[ib]}
        M["An"], M["Bn"] = _weight(M["An0"], M["Bn0"], df_cap)
        M["Aa"], M["Ba"] = _weight(M["Aa0"], M["Ba0"], df_cap)
        M["Bct"] = sp.hstack([M["Bn"], M["Ba"]]).T.tocsr()
        diag(country, ia, ib, M)
        del M
        gc.collect()


def presence_table(tp):
    n = len(tp)
    r = tp["cand_rank"].to_numpy()
    rows = [("absent from candidate set", ~np.isfinite(r)),
            ("present, rank 1-10", r <= 10), ("present, rank 11-20", (r > 10) & (r <= 20)),
            ("present, rank 21-30", (r > 20) & (r <= 30)), ("present, rank > 30", np.isfinite(r) & (r > 30)),
            ("present but outside top-10", np.isfinite(r) & (r > 10)),
            ("present but outside top-20", np.isfinite(r) & (r > 20)),
            ("present but outside top-30", np.isfinite(r) & (r > 30))]
    out = []
    for label, m in rows:
        d = {"bucket": label, "true_pairs": int(m.sum()), "share": float(m.sum() / n)}
        for c in sorted(tp["country"].unique()):
            cm = (tp["country"] == c).to_numpy()
            d[f"share_{c}"] = float((m & cm).sum() / cm.sum())
        out.append(d)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--exp", required=True)
    ap.add_argument("--k-comb", type=int, default=45)
    ap.add_argument("--k-name", type=int, default=10)
    ap.add_argument("--df-cap", type=int, default=2500)
    ap.add_argument("--block-chunk", type=int, default=2000)
    ap.add_argument("--channels", default="", help="multi-channel union spec (empty = E003-style blocker)")
    ap.add_argument("--rrf-m", type=int, default=0)
    ap.add_argument("--hit-sample", type=int, default=300_000)
    ap.add_argument("--rank-sample", type=int, default=100_000)
    ap.add_argument("--reports", default="", help="folder for FULL_DENSITY_BASELINE.md / ERROR_BUCKETS.md")
    args = ap.parse_args()
    tag = block_tag(args)
    exp_dir = Path(args.work) / "experiments" / args.exp
    exp_dir.mkdir(parents=True, exist_ok=True)

    s1, s23 = load_prepared(args.data, "train", args.work)
    truth = read_ground_truth(Path(args.data) / "train" / "train_ground_truth.tsv")
    log(f"loaded {len(s1)} S1, {len(s23)} S2/S3, {sum(map(len, truth.values()))} true pairs")
    s1_pos = pd.Index(s1["entity_id"])
    r_pos = pd.Index(s23["entity_id"])
    a = [(k, x) for k, v in truth.items() for x in v]
    tp = pd.DataFrame({"s1_idx": s1_pos.get_indexer([k for k, _ in a]).astype(np.int64),
                       "r_idx": r_pos.get_indexer([x for _, x in a]).astype(np.int64)})
    del a
    assert (tp >= 0).all().all(), "ground truth id missing from train files"
    tp["country"] = s1["country"].to_numpy()[tp["s1_idx"].to_numpy()]
    diag = Diagnoser(tp, args.rank_sample)

    cache = Path(args.work) / "train" / f"cand_{tag}.parquet"
    cache.parent.mkdir(parents=True, exist_ok=True)
    t_block = time.time()
    if cache.exists():
        cand = pd.read_parquet(cache)
        block_sec, cached = None, True
        log(f"candidates from cache {cache}")
        diagnose_cached(s1, s23, diag, args.df_cap)
    else:
        cached = False
        if args.channels:
            cand = retrieve(s1, s23, args.channels, m=args.rrf_m, df_cap=args.df_cap, log=log)
            block_sec = time.time() - t_block
            diagnose_cached(s1, s23, diag, args.df_cap)
        else:
            cand = generate_candidates(s1, s23, k_comb=args.k_comb, k_name=args.k_name, df_cap=args.df_cap,
                                       chunk=args.block_chunk, log=log, on_country=diag)
            block_sec = time.time() - t_block
        cand.to_parquet(cache, index=False)
    peak_after_block = peak_gib()
    log(f"{len(cand)} candidate pairs ({len(cand) / len(s1):.1f}/S1)")

    y = label_pairs(cand, s1, s23, truth)
    ranks = []
    rep = blocking_report(cand, y, s1, s23, truth, rank_out=ranks)
    # rank (by name_cos + addr_cos within the S1's candidate list) of every true pair, NaN if absent
    tr_rows = np.flatnonzero(y == 1)
    found = pd.DataFrame({"s1_idx": cand["s1_idx"].to_numpy()[tr_rows].astype(np.int64),
                          "r_idx": cand["r_idx"].to_numpy()[tr_rows].astype(np.int64),
                          "cand_rank": ranks[0][tr_rows]})
    del ranks, y, cand
    gc.collect()
    tp = tp.merge(found, on=["s1_idx", "r_idx"], how="left")
    tp["comb"] = tp["name_cos"] + tp["addr_cos"]
    rep.update({"blocker": args.channels or f"E003-style k_comb={args.k_comb} k_name={args.k_name} df_cap={args.df_cap}",
                "cache": str(cache), "from_cache": cached,
                "block_min": None if block_sec is None else round(block_sec / 60, 1),
                "total_min": round((time.time() - T0) / 60, 1), "peak_gib_after_blocking": peak_after_block,
                "presence": presence_table(tp)})
    absent = ~np.isfinite(tp["cand_rank"].to_numpy())
    rep["absent"] = int(absent.sum())
    rep["true_pairs"] = len(tp)
    rep["absent_no_key"] = float((tp.loc[absent, "comb"] <= 0).mean())
    pr = tp.loc[absent & (tp["pool_rank"] > 0).to_numpy(), "pool_rank"]
    rep["rank_sample"] = int(len(pr))
    rep["absent_rank_quantiles"] = {str(q): float(pr.quantile(q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)} if len(pr) else {}

    # error buckets: all absent pairs + a sample of found pairs
    rng = np.random.default_rng(0)
    hit_idx = np.flatnonzero(~absent)
    hit_idx = rng.choice(hit_idx, min(args.hit_sample, len(hit_idx)), replace=False)
    nf = name_frequency(s23)
    miss_t = pair_table(tp[absent], s1, s23, nf)
    hit_t = pair_table(tp.iloc[hit_idx], s1, s23, nf)
    buckets = pd.concat([miss_t.assign(found=False), hit_t.assign(found=True)], ignore_index=True)
    buckets.to_parquet(exp_dir / "error_buckets.parquet", index=False)
    rep["peak_gib"] = peak_gib()
    rep["total_min"] = round((time.time() - T0) / 60, 1)
    (exp_dir / "blocking_full.json").write_text(json.dumps(rep, indent=1, default=float))
    log("blocking", json.dumps({k: v for k, v in rep.items() if k not in ("presence",)}, default=float))
    if args.reports:
        out = Path(args.reports)
        out.mkdir(parents=True, exist_ok=True)
        top = write_report(out / "ERROR_BUCKETS.md", rep, miss_t, hit_t)
        log("top failure modes:\n" + top[["failure_mode", "missed_pairs", "share_of_misses"]].to_string(index=False))
    log("done")


if __name__ == "__main__":
    main()
