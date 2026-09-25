"""Candidate retrieval channels for one country: (queries, pool) -> (query rows, pool rows, score).

Every channel is independent, so recall can be measured per channel, as gain over the baseline,
and in union. IDF and document frequencies always come from the pool (S2/S3 side), exactly as
in `blocking`, so a sample of queries against the full pool reproduces full-density behaviour.
"""
import os
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import normalize as l2norm

from .blocking import _hash, _topk_rows, key_matrices

_MEMO = {}  # pool-side matrices shared by channels within one country; call clear() between countries


def clear():
    _MEMO.clear()


def _memo(key, fn):
    if key not in _MEMO:
        _MEMO[key] = fn()
    return _MEMO[key]


def sparse_topk(A, Bt, k, chunk=2000, workers=4):
    """Top-k (row, column, score) of A @ B for every row of A (Bt = B.T in CSR)."""
    def work(start):
        r, c, v = _topk_rows((A[start:start + chunk] @ Bt).tocsr(), k)
        return r + start, c, v
    with ThreadPoolExecutor(workers) as ex:
        parts = list(ex.map(work, range(0, A.shape[0], chunk)))
    if not parts:
        return np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, np.float32)
    return tuple(np.concatenate([p[i] for p in parts]) for i in range(3))


def _idf(B, df_cap):
    """IDF from the pool with over-frequent keys zeroed (same as blocking._weight)."""
    df = np.bincount(B.indices, minlength=B.shape[1]).astype(np.float32)
    idf = np.log((B.shape[0] + 1) / (df + 1)).astype(np.float32)
    idf[df > df_cap] = 0
    return idf


def _apply(A, idf):
    A = l2norm(A @ sp.diags(idf))
    A.eliminate_zeros()
    return A.tocsr()


def _pool_raw(pool):
    return _memo("raw", lambda: key_matrices(pool))


def _pool_base(pool, df_cap):
    """Weighted pool matrices and their transposes, built once per df cap and shared by channels."""
    def build():
        Bn0, Ba0 = _pool_raw(pool)
        idf_n, idf_a = _idf(Bn0, df_cap), _idf(Ba0, df_cap)
        Bn, Ba = _apply(Bn0, idf_n), _apply(Ba0, idf_a)
        return {"idf_n": idf_n, "idf_a": idf_a, "Bn": Bn, "Ba": Ba, "Bnt": Bn.T.tocsr(), "Bat": Ba.T.tocsr(),
                "Bct": sp.hstack([Bn, Ba]).T.tocsr()}
    return _memo(("base", df_cap), build)


def _query_base(q, P):
    An0, Aa0 = key_matrices(q)
    return _apply(An0, P["idf_n"]), _apply(Aa0, P["idf_a"])


def base(q, pool, k_comb=45, k_name=10, df_cap=2500):
    """The E003 blocker: top k_comb by name_cos + addr_cos, union top k_name by name_cos."""
    P = _pool_base(pool, df_cap)
    An, Aa = _query_base(q, P)
    r1, c1, v1 = sparse_topk(sp.hstack([An, Aa]).tocsr(), P["Bct"], k_comb)
    r2, c2, v2 = sparse_topk(An, P["Bnt"], k_name)
    return np.concatenate([r1, r2]), np.concatenate([c1, c2]), np.concatenate([v1, v2])


def base_wide(q, pool, k=200, df_cap=2500):
    """Same score as `base`, deeper list: separates 'ranked too low' from 'never scored'."""
    P = _pool_base(pool, df_cap)
    An, Aa = _query_base(q, P)
    return sparse_topk(sp.hstack([An, Aa]).tocsr(), P["Bct"], k)


def addr_only(q, pool, k=20, df_cap=2500):
    """Address alone: rescues true matches whose name was replaced or transliterated."""
    P = _pool_base(pool, df_cap)
    return sparse_topk(_query_base(q, P)[1], P["Bat"], k)


def name_noaddr(q, pool, k=10, df_cap=2500):
    """Name search restricted to pool records with an empty address. Their name+addr score is
    capped by the missing address half, so they sink below full-address look-alikes in `base`."""
    P = _pool_base(pool, df_cap)
    idx = np.flatnonzero(pool["addr_empty"].to_numpy(bool))
    if len(idx) == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, np.float32)
    Bt = _memo(("noaddr", df_cap), lambda: P["Bn"][idx].T.tocsr())
    r, c, v = sparse_topk(_query_base(q, P)[0], Bt, k)
    return r, idx[c], v


def base_diagnose(q, pool, tq, tr, df_cap=2500, chunk=500):
    """For true pairs (tq, tr): name_cos, addr_cos, raw shared-key counts before the df cap, and
    the rank of name+addr among the whole pool (0 = no shared key survives the cap)."""
    P = _pool_base(pool, df_cap)
    An0, Aa0 = key_matrices(q)
    Bn0, Ba0 = _pool_raw(pool)
    An, Aa = _apply(An0, P["idf_n"]), _apply(Aa0, P["idf_a"])
    dot = lambda A, B: np.asarray(A[tq].multiply(B[tr]).sum(1)).ravel()
    out = {"name_cos": dot(An, P["Bn"]), "addr_cos": dot(Aa, P["Ba"]),
           "name_shared_raw": dot(An0, Bn0), "addr_shared_raw": dot(Aa0, Ba0)}
    comb = out["name_cos"] + out["addr_cos"]
    Ac = sp.hstack([An, Aa]).tocsr()
    rank = np.zeros(len(tq), np.int64)
    order = np.argsort(tq, kind="stable")
    tqs = tq[order]
    for s in range(0, Ac.shape[0], chunk):
        S = (Ac[s:s + chunk] @ P["Bct"]).tocsr()
        lo, hi = np.searchsorted(tqs, [s, s + chunk])
        for i in order[lo:hi]:
            if comb[i] > 0:
                row = S.data[S.indptr[tq[i] - s]:S.indptr[tq[i] - s + 1]]
                rank[i] = int((row > comb[i] + 1e-6).sum()) + 1
    out["comb_rank"] = rank
    return out


# ---------------------------------------------------------------------------
# Conjunction keys. At full density most single tokens are either dropped by the df cap
# ("elm", "delhi", "570", "marketing") or shared by thousands of records, so the true record ties
# with a crowd. Pairs of tokens are rare: "570|delhi", "elm|morganton", "mrktng|delhi".
# Name skeletons (not raw tokens) are paired, so romanized native-script names still meet.
# ---------------------------------------------------------------------------
def _pairs(toks, prefix):
    return [prefix + a + "|" + b for a, b in combinations(sorted(set(toks)), 2)]


def conj_keys(row):
    skel, clean_addr = row
    nt = [t for t in skel.split() if len(t) >= 2][:3]
    at = clean_addr.split()[:8]
    keys = _pairs(at, "aa:")                                    # address x address (<= 28)
    keys += ["sa:" + n + "|" + a for n in nt for a in set(at)]  # name skeleton x address (<= 24)
    keys += _pairs(nt, "nn:")                                   # name x name (<= 3)
    return keys


def conj_matrix(df, chunk=200_000):
    cols = [df[c].tolist() for c in ("name_skel", "addr_clean")]
    parts = []
    for s in range(0, len(df), chunk):  # chunked: ~50 key strings per row do not fit in RAM at once
        parts.append(_hash([conj_keys(r) for r in zip(*(c[s:s + chunk] for c in cols))]))
    return sp.vstack(parts).tocsr()


def _pool_conj(pool, df_cap):
    B = conj_matrix(pool)
    idf = _idf(B, df_cap)
    B = _apply(B, idf)
    return B.T.tocsr(), idf


def conj(q, pool, k=50, df_cap=500):
    Bt, idf = _memo(("conj", df_cap), lambda: _pool_conj(pool, df_cap))
    return sparse_topk(_apply(conj_matrix(q), idf), Bt, k)


# ---------------------------------------------------------------------------
# BM25 over the base keys with a much higher df cap: tests whether an uncapped-style lexical
# ranking (term saturation, length normalization) finds what capped IDF cosine misses.
# ---------------------------------------------------------------------------
def _bm25_pool(B, df_cap, k1=1.2, b=0.75):
    df = np.bincount(B.indices, minlength=B.shape[1]).astype(np.float32)
    idf = np.log(1 + (B.shape[0] - df + 0.5) / (df + 0.5)).astype(np.float32)
    idf[df > df_cap] = 0
    dl = np.diff(B.indptr).astype(np.float32)
    B = B.tocsr(copy=True)
    tf = B.data
    norm = np.repeat(k1 * (1 - b + b * dl / max(dl.mean(), 1)), np.diff(B.indptr))
    B.data = (tf * (k1 + 1) / (tf + norm)).astype(np.float32)
    B = B @ sp.diags(idf)
    B.eliminate_zeros()
    return B.tocsr(), idf


def bm25(q, pool, k=50, df_cap=10000):
    def build():
        B, idf = _bm25_pool(sp.hstack(_pool_raw(pool)).tocsr(), df_cap)
        return B.T.tocsr()
    Bt = _memo(("bm25", df_cap), build)
    A = sp.hstack(key_matrices(q)).tocsr()
    A.data[:] = 1
    return sparse_topk(A, Bt, k, chunk=300)


# ---------------------------------------------------------------------------
# Deterministic rescue blocks: exact keys, every pair inside a small block. No ranking, so a
# true pair in a block is never truncated. Blocks larger than `max_block` pool records are skipped.
# ---------------------------------------------------------------------------
def _rescue_keys(df):
    comp = df["name_compact"].to_numpy(object)
    skel = df["name_skel"].to_numpy(object)
    nums = df["addr_nums"].to_numpy(object)
    out = []
    for i in range(len(df)):
        ks = []
        if len(comp[i]) >= 5:
            ks.append("c:" + comp[i])
        sk = "".join(sorted(skel[i].split()))
        if len(sk) >= 4:
            ks.append("s:" + sk)
        ns = sorted(set(nums[i].split()))
        if sum(map(len, ns)) >= 4:
            ks.append("n:" + "|".join(ns))
        out.append(ks)
    s = pd.Series(out).explode().dropna()
    return pd.DataFrame({"row": s.index.to_numpy(np.int64), "key": s.to_numpy()})


def rescue(q, pool, max_block=30):
    P = _memo("rescue", lambda: _rescue_keys(pool))
    size = P["key"].value_counts()
    P = P[P["key"].map(size) <= max_block]
    m = _rescue_keys(q).merge(P, on="key", suffixes=("_q", "_p"))
    m = m.drop_duplicates(["row_q", "row_p"])
    return m["row_q"].to_numpy(), m["row_p"].to_numpy(), np.ones(len(m), np.float32)


# ---------------------------------------------------------------------------
# Dense retrieval (optional, GPU). Default multilingual-e5-small (MIT, 118M params). Pool
# embeddings are fp16 and computed once per country; top-k is an exact chunked GPU matmul.
# ---------------------------------------------------------------------------
def _texts(df):
    return ("query: " + df["business_name"].astype(str) + ", " + df["business_address"].astype(str)).tolist()


def _encode(model, texts, bs=512, chunk=200_000):
    import torch
    out = torch.empty((len(texts), model.get_sentence_embedding_dimension()), dtype=torch.float16, device="cuda")
    for s in range(0, len(texts), chunk):
        e = model.encode(texts[s:s + chunk], batch_size=bs, convert_to_tensor=True, normalize_embeddings=True,
                         show_progress_bar=False)
        out[s:s + len(e)] = e.half()
    return out


def dense(q, pool, k=50):
    try:
        import torch
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("dense: sentence-transformers/torch missing, skipped")
        return np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, np.float32)
    if not torch.cuda.is_available():
        print("dense: no GPU, skipped")
        return np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, np.float32)
    name = os.environ.get("BER_DENSE_MODEL", "intfloat/multilingual-e5-small")
    model = _memo(("dense_model", name), lambda: SentenceTransformer(name, device="cuda").half())
    model.max_seq_length = 64
    P = _memo(("dense_pool", name), lambda: _encode(model, _texts(pool)))
    Q = _encode(model, _texts(q))
    rows, cols, vals = [], [], []
    for s in range(0, len(Q), 128):
        v, c = torch.topk(Q[s:s + 128] @ P.T, k, dim=1)
        rows.append(np.repeat(np.arange(s, s + len(c)), k)); cols.append(c.cpu().numpy().ravel())
        vals.append(v.float().cpu().numpy().ravel())
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)


CHANNELS = {"base": base, "base_wide": base_wide, "addr_only": addr_only, "name_noaddr": name_noaddr, "conj": conj,
            "bm25": bm25, "rescue": rescue, "dense": dense}
