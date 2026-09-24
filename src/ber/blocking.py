"""Candidate generation.

Each record becomes two sparse, IDF-weighted, L2-normalized key vectors (name keys and
address keys, hashed). Keys with document frequency above `df_cap` are dropped, which keeps
the sparse products cheap and the candidates discriminative. For every Source-1 record we
keep the union of
  * top `k_comb` S2/S3 records by name_cos + addr_cos, and
  * top `k_name` S2/S3 records by name_cos alone,
searching only within the same country label (open set; no country is hard-coded).
"""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize as l2norm

N_FEATURES = 2 ** 23


def name_keys(row) -> list:
    core, skel, compact, alias = row
    keys = ["w:" + t for t in core.split()]
    keys += ["k:" + t for t in skel.split() if len(t) >= 2]
    if len(compact) >= 4:
        keys.append("c:" + compact)
        keys.append("p:" + compact[:6])
    keys += ["w:" + t for t in alias.split()]
    return keys


def addr_keys(row) -> list:
    alpha, nums, skel = row
    a = alpha.split()
    keys = ["a:" + t for t in a]
    keys += ["ak:" + t for t in skel.split() if len(t) >= 3]
    n = nums.split()
    keys += ["n:" + t for t in n]
    if n and a:
        keys.append("h:" + n[0] + "_" + a[0])
    return keys


def _hash(key_lists):
    hv = HashingVectorizer(analyzer=lambda x: x, n_features=N_FEATURES, alternate_sign=False, norm=None,
                           binary=True, dtype=np.float32)
    return hv.transform(key_lists).tocsr()


def key_matrices(df: pd.DataFrame):
    nk = [name_keys(r) for r in zip(df["name_core"], df["name_skel"], df["name_compact"], df["name_alias"])]
    ak = [addr_keys(r) for r in zip(df["addr_alpha"], df["addr_nums"], df["addr_skel"])]
    return _hash(nk), _hash(ak)


def _weight(A, B, df_cap):
    """IDF weights from the S2/S3 side, drop over-frequent keys, L2-normalize both sides."""
    df = np.bincount(B.indices, minlength=B.shape[1]).astype(np.float32)
    idf = np.log((B.shape[0] + 1) / (df + 1)).astype(np.float32)
    idf[df > df_cap] = 0
    d = sp.diags(idf)
    A2, B2 = l2norm(A @ d), l2norm(B @ d)
    A2.eliminate_zeros(); B2.eliminate_zeros()
    return A2.tocsr(), B2.tocsr()


def _topk_rows(C, k):
    """Per-row top-k (column indices, values) of a CSR matrix; rows may have < k entries."""
    rows, cols, vals = [], [], []
    ip, ix, dv = C.indptr, C.indices, C.data
    for r in range(C.shape[0]):
        a, b = ip[r], ip[r + 1]
        if a == b:
            continue
        seg = dv[a:b]
        if b - a > k:
            sel = np.argpartition(-seg, k)[:k]
        else:
            sel = np.arange(b - a)
        rows.append(np.full(len(sel), r, np.int64)); cols.append(ix[a:b][sel]); vals.append(seg[sel])
    if not rows:
        return np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, np.float32)
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)


def rowwise_dot(A, B, ia, ib, chunk=2_000_000):
    out = np.empty(len(ia), np.float32)
    for s in range(0, len(ia), chunk):
        out[s:s + chunk] = np.asarray(A[ia[s:s + chunk]].multiply(B[ib[s:s + chunk]]).sum(1)).ravel()
    return out


def generate_candidates(s1: pd.DataFrame, s23: pd.DataFrame, k_comb=40, k_name=10, df_cap=2000,
                        chunk=4000, workers=4, log=print) -> pd.DataFrame:
    """Return a frame of (s1_idx, r_idx, name_cos, addr_cos) with positional indices into s1 / s23."""
    An_all, Aa_all = key_matrices(s1)
    Bn_all, Ba_all = key_matrices(s23)
    out = []
    for country in sorted(set(s1["country"]) | set(s23["country"])):
        ia = np.flatnonzero((s1["country"] == country).to_numpy())
        ib = np.flatnonzero((s23["country"] == country).to_numpy())
        if len(ia) == 0 or len(ib) == 0:
            continue
        An, Bn = _weight(An_all[ia], Bn_all[ib], df_cap)
        Aa, Ba = _weight(Aa_all[ia], Ba_all[ib], df_cap)
        Ac = sp.hstack([An, Aa]).tocsr()
        Bct = sp.hstack([Bn, Ba]).T.tocsr()
        Bnt = Bn.T.tocsr()

        def work(start):
            sl = slice(start, min(start + chunk, len(ia)))
            r1, c1, _ = _topk_rows((Ac[sl] @ Bct).tocsr(), k_comb)
            r2, c2, _ = _topk_rows((An[sl] @ Bnt).tocsr(), k_name)
            key = np.unique(np.concatenate([(r1 + start) * len(ib) + c1, (r2 + start) * len(ib) + c2]))
            return key // len(ib), key % len(ib)

        with ThreadPoolExecutor(workers) as ex:
            parts = list(ex.map(work, range(0, len(ia), chunk)))
        la = np.concatenate([p[0] for p in parts]); lb = np.concatenate([p[1] for p in parts])
        name_cos = rowwise_dot(An, Bn, la, lb)
        addr_cos = rowwise_dot(Aa, Ba, la, lb)
        out.append(pd.DataFrame({"s1_idx": ia[la], "r_idx": ib[lb], "name_cos": name_cos, "addr_cos": addr_cos}))
        log(f"  blocking {country}: {len(ia)} S1 x {len(ib)} S2/S3 -> {len(la)} pairs ({len(la) / len(ia):.1f}/S1)")
    return pd.concat(out, ignore_index=True)
