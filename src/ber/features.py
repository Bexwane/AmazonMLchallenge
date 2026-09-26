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


def context_arrays(cols, rows=None) -> dict:
    """Context features as a dict of float32 arrays. `cols` maps column name -> full-length array of the
    candidate table (s1_idx, r_idx, name_cos, addr_cos, optional multi-channel columns). Ranks are computed
    over ALL pairs; only `rows` (boolean mask) are returned. With rows=None the pass-through columns are
    views, so no full-length copy of the candidate table is made."""
    sel = slice(None) if rows is None else np.flatnonzero(rows)
    g1, g2 = cols["s1_idx"], cols["r_idx"]
    name, addr = np.asarray(cols["name_cos"], np.float32), np.asarray(cols["addr_cos"], np.float32)
    C = {"name_cos": name[sel], "addr_cos": addr[sel]}
    scores = {"comb": name + addr}
    extra = [c for c in cols if c not in ("s1_idx", "r_idx", "name_cos", "addr_cos")]
    for c in extra:  # multi-channel blocker: fused score, channel count, per-channel ranks
        C[c] = np.asarray(cols[c], np.float32)[sel]
    if "rrf" in extra:
        scores["rrf"] = np.asarray(cols["rrf"], np.float32)
    for name_, sc in scores.items():
        if name_ == "comb":
            C["comb"] = sc[sel]
        for side, g in (("s1", g1), ("r", g2)):
            rank, gap = _rank_gap(sc, g)
            C[f"{name_}_rank_{side}"], C[f"{name_}_gap_{side}"] = rank[sel], gap[sel]
            del rank, gap
    for side, g in (("s1", g1), ("r", g2)):
        C[f"n_cand_{side}"] = np.bincount(g)[g][sel].astype(np.float32)
    return C


def context2_arrays(s1_idx, r_idx, name_cos, addr_cos, rows=None) -> dict:
    """v4 context. France (test only) has 3x more Source-1 entities sharing one address than train
    (15.5% vs 5%): there only the name separates them. For each pair: rank/gap of the name score among
    the S1's candidates and among the S2/S3 record's competing S1s, the address rank/gap on the S2/S3
    side, and how many competing S1s are tied with the best address. Stored as float16 (train and test
    alike) to keep the 1e8-row test table small; values are identical after the float32 cast."""
    sel = slice(None) if rows is None else np.flatnonzero(rows)
    name_cos, addr_cos = np.asarray(name_cos, np.float32), np.asarray(addr_cos, np.float32)
    C = {}
    for nm, sc, sides in (("name", name_cos, (("s1", s1_idx), ("r", r_idx))), ("addr", addr_cos, (("r", r_idx),))):
        for side, g in sides:
            rank, gap = _rank_gap(sc, g)
            C[f"{nm}_rank_{side}"] = np.minimum(rank[sel], 2048).astype(np.float16)
            C[f"{nm}_gap_{side}"] = gap[sel].astype(np.float16)
            if nm == "addr":
                ties = np.bincount(g, weights=(gap >= -0.02)).astype(np.float32)
                C["addr_ties_r"] = np.minimum(ties[g[sel]], 2048).astype(np.float16)
            del rank, gap
    return C


def crowd_arrays(s1: pd.DataFrame, s23: pd.DataFrame):
    """Per-record counts of Source-1 entities with exactly this normalized address, plus address codes
    for an exact-equality pair flag. Empty addresses get count 0 and code -1."""
    a1, a23 = s1["addr_clean"].to_numpy(object), s23["addr_clean"].to_numpy(object)
    codes, uniq = pd.factorize(np.concatenate([a1, a23]))
    empty = np.flatnonzero(uniq == "")
    if len(empty):
        codes[codes == empty[0]] = -1
    c1, c23 = codes[:len(a1)], codes[len(a1):]
    cnt = np.bincount(c1[c1 >= 0], minlength=len(uniq)).astype(np.float32)
    dup1 = np.where(c1 >= 0, cnt[np.maximum(c1, 0)], 0).astype(np.float32)
    dup23 = np.where(c23 >= 0, cnt[np.maximum(c23, 0)], 0).astype(np.float32)
    return {"c1": c1.astype(np.int32), "c23": c23.astype(np.int32), "dup1": dup1, "dup23": dup23}


def crowd_features(s1_idx, r_idx, A) -> pd.DataFrame:
    """l_addr_dup: S1 entities sharing this S1's address; r_addr_dup: S1 entities with the S2/S3 record's
    address; a_exact: normalized addresses identical (non-empty)."""
    c1, c23 = A["c1"][s1_idx], A["c23"][r_idx]
    return pd.DataFrame({"l_addr_dup": A["dup1"][s1_idx], "r_addr_dup": A["dup23"][r_idx],
                         "a_exact": ((c1 == c23) & (c1 >= 0)).astype(np.float32)})


def context_features(cand: pd.DataFrame, rows=None) -> pd.DataFrame:
    """DataFrame form of `context_arrays` (use only when the selected rows fit comfortably in memory)."""
    C = context_arrays({c: cand[c].to_numpy() for c in cand.columns}, rows)
    index = cand.index if rows is None else cand.index[np.flatnonzero(rows)]
    return pd.DataFrame(C, index=index)


def pair_features(cand: pd.DataFrame, s1: pd.DataFrame, s23: pd.DataFrame, rows=None, workers=-1) -> pd.DataFrame:
    """Full feature matrix for `cand` rows selected by the boolean mask `rows` (default all)."""
    C = context_features(cand, rows)
    sub = cand if rows is None else cand[rows]
    return pd.concat([string_features(sub, s1, s23, workers), C], axis=1)


def name_amb_arrays(s1: pd.DataFrame, s23: pd.DataFrame):
    """Global name ambiguity (E010). A S2/S3 record without address whose name is carried by several
    Source-1 entities cannot be attributed from the pair alone; the pair model must see how ambiguous the
    name is, including rival S1s that never reached the record's candidate list."""
    n1, n23 = s1["name_core"].to_numpy(object), s23["name_core"].to_numpy(object)
    codes, uniq = pd.factorize(np.concatenate([n1, n23]))
    empty = np.flatnonzero(uniq == "")
    if len(empty):
        codes[codes == empty[0]] = -1
    c1, c23 = codes[:len(n1)], codes[len(n1):]
    cnt1 = np.bincount(c1[c1 >= 0], minlength=len(uniq)).astype(np.float32)
    cnt23 = np.bincount(c23[c23 >= 0], minlength=len(uniq)).astype(np.float32)
    at = lambda cnt, c: np.where(c >= 0, cnt[np.maximum(c, 0)], 0).astype(np.float32)
    return {"l_name_s1dup": at(cnt1, c1), "r_name_s1cnt": at(cnt1, c23), "r_name_s23cnt": at(cnt23, c23)}


def name_amb_features(s1_idx, r_idx, A) -> pd.DataFrame:
    """l_name_s1dup: S1 entities sharing this S1's core name; r_name_s1cnt: S1 entities carrying the
    record's core name; r_name_s23cnt: S2/S3 records with that name (a twin entity adds its own)."""
    return pd.DataFrame({"l_name_s1dup": A["l_name_s1dup"][s1_idx], "r_name_s1cnt": A["r_name_s1cnt"][r_idx],
                         "r_name_s23cnt": A["r_name_s23cnt"][r_idx]})
