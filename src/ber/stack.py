"""Second-stage features computed from first-stage pair probabilities.

1. Probability context within each Source-1 entity: where this pair stands among the entity's own
   candidates (rank, gap to the best, second best, total mass, confident count).
2. Sibling agreement: records of one entity in S2 and S3 resemble each other, so a candidate that looks
   like the entity's most confident *other* candidates is likely a match even when its own similarity
   to Source-1 is weak (native-script names, rewritten addresses). For every pair we compare its S2/S3
   record with the top-`top` other candidates (p >= min_p) of the same Source-1 entity.

Both only use pairs of the same Source-1 entity, so they are identical whether the entity is in a
sampled training subset or in the full test set (no dependence on other entities' predictions).
"""
import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from .features import _rank_gap, _sim


def p_context(s1_idx: np.ndarray, p: np.ndarray) -> pd.DataFrame:
    p = p.astype(np.float32)
    rank, gap = _rank_gap(p, s1_idx)
    mx = p - gap
    order = np.lexsort((-p, s1_idx))
    g, q = s1_idx[order], p[order]
    new = np.ones(len(q), bool)
    new[1:] = g[1:] != g[:-1]
    starts = np.flatnonzero(new)
    grp = np.cumsum(new) - 1
    size = np.diff(np.r_[starts, len(q)])
    second_g = np.where(size > 1, q[np.minimum(starts + 1, len(q) - 1)], 0.0)
    second = np.empty(len(p), np.float32)
    second[order] = second_g[grp]
    tot = np.empty(len(p), np.float32)
    tot[order] = np.add.reduceat(q.astype(np.float64), starts)[grp]
    c50 = np.empty(len(p), np.float32)
    c50[order] = np.add.reduceat((q >= 0.5).astype(np.float64), starts)[grp]
    c20 = np.empty(len(p), np.float32)
    c20[order] = np.add.reduceat((q >= 0.2).astype(np.float64), starts)[grp]
    return pd.DataFrame({
        "p1": p, "p1_logit": np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1)).astype(np.float32),
        "p1_rank_s1": rank, "p1_gap_s1": gap, "p1_max_s1": mx, "p1_ratio_s1": p / np.maximum(mx, 1e-6),
        "p1_second_s1": second, "p1_margin": np.where(rank == 1, p - second, p - mx).astype(np.float32),
        "p1_sum_s1": tot, "p1_sum_others": tot - p, "p1_cnt50_s1": c50, "p1_cnt20_s1": c20,
    })


def sibling_features(s1_idx: np.ndarray, r_idx: np.ndarray, p: np.ndarray, s23: pd.DataFrame,
                     top=3, min_p=0.2, workers=-1, chunk=3_000_000) -> pd.DataFrame:
    n = len(p)
    order = np.lexsort((-p, s1_idx))
    g = s1_idx[order]
    new = np.ones(n, bool)
    new[1:] = g[1:] != g[:-1]
    starts = np.flatnonzero(new)
    grp_sorted = np.cumsum(new) - 1
    pos = np.arange(n) - starts[grp_sorted]
    grp = np.empty(n, np.int64)
    grp[order] = grp_sorted
    sib_r = np.full((len(starts), top), -1, np.int64)
    sib_p = np.zeros((len(starts), top), np.float32)
    for j in range(top):
        m = (pos == j) & (p[order] >= min_p)
        sib_r[grp_sorted[m], j] = r_idx[order][m]
        sib_p[grp_sorted[m], j] = p[order][m]
    cols = {"addr": "addr_clean", "name": "name_core", "skel": "name_skel", "alpha": "addr_alpha"}
    col = {k: s23[c] for k, c in cols.items()}  # arrow-backed; rows are taken per chunk, never all at once
    hnum = s23["addr_nums"]
    feats = {f"sib_{k}_max": np.zeros(n, np.float32) for k in list(cols) + ["hnum"]}
    wsum = np.zeros(n, np.float32)
    waddr = np.zeros(n, np.float32)
    nsib = np.zeros(n, np.float32)
    top1 = {k: np.zeros(n, np.float32) for k in ("addr", "name")}
    top1_seen = np.zeros(n, bool)
    for j in range(top):
        rp = sib_r[grp, j]
        pj = sib_p[grp, j]
        valid = np.flatnonzero((rp >= 0) & (rp != r_idx))
        for a in range(0, len(valid), chunk):
            v = valid[a:a + chunk]
            x, y = r_idx[v], rp[v]
            sims = {k: _sim(col[k].iloc[x].tolist(), col[k].iloc[y].tolist(),
                            fuzz.token_set_ratio if k in ("addr", "name", "alpha") else fuzz.ratio, workers)
                    for k in cols}
            hx = [t.split(" ", 1)[0] for t in hnum.iloc[x].tolist()]
            hy = [t.split(" ", 1)[0] for t in hnum.iloc[y].tolist()]
            sims["hnum"] = np.array([float(u != "" and u == w) for u, w in zip(hx, hy)], np.float32)
            for k, s in sims.items():
                np.maximum.at(feats[f"sib_{k}_max"], v, s)
            wsum[v] += pj[v]
            waddr[v] += pj[v] * sims["addr"]
            nsib[v] += 1
            first = ~top1_seen[v]  # the most confident sibling that is not this record
            top1["addr"][v[first]] = sims["addr"][first]
            top1["name"][v[first]] = sims["name"][first]
            top1_seen[v[first]] = True
    feats["sib_addr_wmean"] = np.where(wsum > 0, waddr / np.maximum(wsum, 1e-6), 0).astype(np.float32)
    feats["sib_n"] = nsib
    feats["sib_top1_addr"], feats["sib_top1_name"] = top1["addr"], top1["name"]
    return pd.DataFrame(feats)


def stack_features(cand: pd.DataFrame, p: np.ndarray, s23: pd.DataFrame, C: pd.DataFrame) -> pd.DataFrame:
    """Stage-2 matrix: probability context + sibling agreement + the (candidate-table) context features."""
    s1_idx, r_idx = cand["s1_idx"].to_numpy(), cand["r_idx"].to_numpy()
    P = p_context(s1_idx, p)
    S = sibling_features(s1_idx, r_idx, p, s23)
    return pd.concat([P, S, C.reset_index(drop=True)], axis=1)
