"""Pairwise features for (Source-1, S2/S3) candidate pairs.

Deliberately excludes `country` so the model transfers to countries absent from training
(France in test). Context features (ranks within a Source-1 record's candidates and within a
S2/S3 record's competing Source-1 records) carry most of the precision signal.
"""
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.process import cpdist


def _sim(a, b, scorer, workers):
    return cpdist(a, b, scorer=scorer, workers=workers, dtype=np.float32) / 100.0


def _num_features(n1, n2):
    """Jaccard of number sets, first-number equality, and presence flags."""
    jac = np.zeros(len(n1), np.float32)
    first_eq = np.zeros(len(n1), np.float32)
    for i, (a, b) in enumerate(zip(n1, n2)):
        if a and b:
            sa, sb = a.split(), b.split()
            A, B = set(sa), set(sb)
            jac[i] = len(A & B) / len(A | B)
            first_eq[i] = sa[0] == sb[0] or sa[0] in B
    return jac, first_eq


def string_features(cand: pd.DataFrame, s1: pd.DataFrame, s23: pd.DataFrame, workers=-1, chunk=2_000_000) -> pd.DataFrame:
    """Name/address similarity features for every row of `cand` (index is preserved)."""
    parts = []
    for start in range(0, len(cand), chunk):
        c = cand.iloc[start:start + chunk]
        L = s1.iloc[c["s1_idx"].to_numpy()].reset_index(drop=True)
        R = s23.iloc[c["r_idx"].to_numpy()].reset_index(drop=True)
        f = {}
        ln, rn = L["name_core"].tolist(), R["name_core"].tolist()
        f["n_ratio"] = _sim(ln, rn, fuzz.ratio, workers)
        f["n_tset"] = _sim(ln, rn, fuzz.token_set_ratio, workers)
        f["n_tsort"] = _sim(ln, rn, fuzz.token_sort_ratio, workers)
        f["n_partial"] = _sim(ln, rn, fuzz.partial_ratio, workers)
        f["n_jw"] = cpdist(ln, rn, scorer=JaroWinkler.normalized_similarity, workers=workers, dtype=np.float32)
        f["n_skel_ratio"] = _sim(L["name_skel"].tolist(), R["name_skel"].tolist(), fuzz.ratio, workers)
        f["n_skel_tset"] = _sim(L["name_skel"].tolist(), R["name_skel"].tolist(), fuzz.token_set_ratio, workers)
        lc, rc = L["name_compact"].tolist(), R["name_compact"].tolist()
        f["n_compact_ratio"] = _sim(lc, rc, fuzz.ratio, workers)
        f["n_compact_partial"] = _sim(lc, rc, fuzz.partial_ratio, workers)
        f["n_clean_tset"] = _sim(L["name_clean"].tolist(), R["name_clean"].tolist(), fuzz.token_set_ratio, workers)
        f["n_core_eq"] = (L["name_core"].to_numpy() == R["name_core"].to_numpy()).astype(np.float32)
        f["n_ntok_l"] = L["name_core"].str.count(" ").to_numpy(np.float32) + 1
        f["n_ntok_r"] = R["name_core"].str.count(" ").to_numpy(np.float32) + 1
        f["r_is_web"] = R["name_is_web"].to_numpy(np.float32)
        f["r_has_alias"] = R["name_has_alias"].to_numpy(np.float32)
        f["r_native"] = R["name_native"].to_numpy(np.float32)
        la, ra = L["addr_clean"].tolist(), R["addr_clean"].tolist()
        f["a_ratio"] = _sim(la, ra, fuzz.ratio, workers)
        f["a_tset"] = _sim(la, ra, fuzz.token_set_ratio, workers)
        f["a_tsort"] = _sim(la, ra, fuzz.token_sort_ratio, workers)
        f["a_alpha_tset"] = _sim(L["addr_alpha"].tolist(), R["addr_alpha"].tolist(), fuzz.token_set_ratio, workers)
        f["a_skel_tset"] = _sim(L["addr_skel"].tolist(), R["addr_skel"].tolist(), fuzz.token_set_ratio, workers)
        f["a_num_jac"], f["a_first_num_eq"] = _num_features(L["addr_nums"].tolist(), R["addr_nums"].tolist())
        f["r_addr_empty"] = R["addr_empty"].to_numpy(np.float32)
        f["r_addr_ncomp"] = R["addr_ncomp"].to_numpy(np.float32)
        f["l_addr_ncomp"] = L["addr_ncomp"].to_numpy(np.float32)
        f["r_src"] = R["src"].to_numpy(np.float32)
        parts.append(pd.DataFrame(f))
    return pd.concat(parts, ignore_index=True).set_axis(cand.index)


def extra_features(cand: pd.DataFrame, s1: pd.DataFrame, s23: pd.DataFrame, workers=-1, chunk=2_000_000) -> pd.DataFrame:
    """v3 additions. House numbers are often truncated or padded by the noise ("5300" vs "300",
    "10824" vs "1082", "0520" vs "520"): partial_ratio catches that, ratio separates it from equality."""
    parts = []
    for start in range(0, len(cand), chunk):
        c = cand.iloc[start:start + chunk]
        ln = s1["addr_nums"].to_numpy()[c["s1_idx"].to_numpy()].tolist()
        rn = s23["addr_nums"].to_numpy()[c["r_idx"].to_numpy()].tolist()
        lf = [x.split(" ", 1)[0] for x in ln]
        rf = [x.split(" ", 1)[0] for x in rn]
        parts.append(pd.DataFrame({
            "a_num_tset": _sim(ln, rn, fuzz.token_set_ratio, workers),
            "a_hnum_partial": _sim(lf, rf, fuzz.partial_ratio, workers),
            "a_hnum_ratio": _sim(lf, rf, fuzz.ratio, workers),
        }))
    return pd.concat(parts, ignore_index=True).set_axis(cand.index)


def context_columns(cand: pd.DataFrame):
    """Names of the context features of a candidate table (computable from the table alone)."""
    return list(context_features(cand.iloc[:0]).columns)


def _rank_gap(score, group):
    """Per-group rank (1 = best, ties share the lowest rank, like pandas rank(method='min')) and
    gap to the group maximum, computed with one lexsort (much lighter than pandas groupby at 1e8 rows)."""
    order = np.lexsort((-score, group))
    g, s = group[order], score[order]
    idx = np.arange(len(order), dtype=np.int32 if len(order) < 2 ** 31 else np.int64)
    new_g = np.ones(len(order), bool)
    new_g[1:] = g[1:] != g[:-1]
    new_v = new_g.copy()
    new_v[1:] |= s[1:] != s[:-1]
    g_start = np.maximum.accumulate(np.where(new_g, idx, 0))
    v_start = np.maximum.accumulate(np.where(new_v, idx, 0))
    rank = np.empty(len(order), np.float32)
    gap = np.empty(len(order), np.float32)
    rank[order] = (v_start - g_start + 1).astype(np.float32)
    gap[order] = s - s[g_start]
    return rank, gap


def context_features(cand: pd.DataFrame, rows=None) -> pd.DataFrame:
    """Blocking scores plus rank/gap features. Ranks are computed over ALL candidate pairs of the split
    (the per-S2/S3 ranks measure competition between Source-1 records); only the rows selected by the
    boolean mask `rows` are returned, so the full-length feature matrix never exists."""
    sel = slice(None) if rows is None else np.flatnonzero(rows)
    g1, g2 = cand["s1_idx"].to_numpy(), cand["r_idx"].to_numpy()
    name, addr = cand["name_cos"].to_numpy(np.float32), cand["addr_cos"].to_numpy(np.float32)
    C = {"name_cos": name[sel], "addr_cos": addr[sel]}
    scores = {"comb": name + addr}
    extra = [c for c in cand.columns if c not in ("s1_idx", "r_idx", "name_cos", "addr_cos")]
    for c in extra:  # multi-channel blocker: fused score, channel count, per-channel ranks
        C[c] = cand[c].to_numpy(np.float32)[sel]
    if "rrf" in extra:
        scores["rrf"] = cand["rrf"].to_numpy(np.float32)
    for name_, sc in scores.items():
        if name_ == "comb":
            C["comb"] = sc[sel]
        for side, g in (("s1", g1), ("r", g2)):
            rank, gap = _rank_gap(sc, g)
            C[f"{name_}_rank_{side}"], C[f"{name_}_gap_{side}"] = rank[sel], gap[sel]
            del rank, gap
    for side, g in (("s1", g1), ("r", g2)):
        C[f"n_cand_{side}"] = np.bincount(g)[g][sel].astype(np.float32)
    index = cand.index if rows is None else cand.index[sel]
    return pd.DataFrame(C, index=index)


def pair_features(cand: pd.DataFrame, s1: pd.DataFrame, s23: pd.DataFrame, rows=None, workers=-1) -> pd.DataFrame:
    """Full feature matrix for `cand` rows selected by the boolean mask `rows` (default all)."""
    C = context_features(cand, rows)
    sub = cand if rows is None else cand[rows]
    return pd.concat([string_features(sub, s1, s23, workers), C], axis=1)
