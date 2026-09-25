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
# Targeted multilingual dense retrieval (GPU). Only pool records whose name is in an Indic script are
# encoded (the measured native-script miss bucket), so no 10M-row embedding matrix is ever built.
# Queries (Latin Source-1 records) search that subset with an exact chunked fp16 GPU matmul.
#   bge_native: BAAI/bge-m3 (MIT, 568M params, 1024-d)
#   e5_native:  intfloat/multilingual-e5-small (MIT, 118M params, 384-d), ablation
# ---------------------------------------------------------------------------
DENSE_MODELS = {"bge_native": ("BAAI/bge-m3", ""), "e5_native": ("intfloat/multilingual-e5-small", "query: ")}
DENSE_INFO = {}  # filled per run: model, revision, dim, dtype, encoded counts, seconds


def _texts(df, prefix):
    return (prefix + df["business_name"].astype(str) + ", " + df["business_address"].astype(str)).tolist()


def _encode(model, texts, bs=256, chunk=100_000):
    import torch
    out = torch.empty((len(texts), model.get_sentence_embedding_dimension()), dtype=torch.float16, device="cuda")
    for s in range(0, len(texts), chunk):
        e = model.encode(texts[s:s + chunk], batch_size=bs, convert_to_tensor=True, normalize_embeddings=True,
                         show_progress_bar=False)
        out[s:s + len(e)] = e.half()
    return out


def _empty():
    return np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, np.float32)


def _dense_native(q, pool, which, k):
    import time
    try:
        import torch
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print(f"{which}: sentence-transformers/torch missing, skipped")
        return _empty()
    if not torch.cuda.is_available():
        print(f"{which}: no GPU, skipped")
        return _empty()
    idx = np.flatnonzero(pool["name_native"].to_numpy(bool))
    if len(idx) == 0:
        return _empty()
    name, prefix = DENSE_MODELS[which]
    info = DENSE_INFO.setdefault(which, {"model": name, "dtype": "float16", "index": "exact GPU matmul (chunked)",
                                         "target": "pool records with Indic-script names"})

    def load():
        m = SentenceTransformer(name, device="cuda").half()
        m.max_seq_length = 64
        try:
            from huggingface_hub import model_info
            info["revision"] = model_info(name).sha
        except Exception:
            info["revision"] = "unknown"
        info["dim"] = m.get_sentence_embedding_dimension()
        return m
    model = _memo(("dense_model", which), load)

    def enc_pool():
        t = time.time()
        e = _encode(model, _texts(pool.iloc[idx], prefix))
        info["pool_encoded"] = info.get("pool_encoded", 0) + len(idx)
        info["pool_encode_sec"] = info.get("pool_encode_sec", 0) + round(time.time() - t, 1)
        return e
    P = _memo(("dense_pool", which), enc_pool)
    t = time.time()
    Q = _encode(model, _texts(q, prefix))
    info["queries_encoded"] = info.get("queries_encoded", 0) + len(q)
    k = min(k, len(idx))
    rows, cols, vals = [], [], []
    for s in range(0, len(Q), 256):
        v, c = torch.topk(Q[s:s + 256] @ P.T, k, dim=1)
        rows.append(np.repeat(np.arange(s, s + len(c)), k)); cols.append(c.cpu().numpy().ravel())
        vals.append(v.float().cpu().numpy().ravel())
    info["query_sec"] = info.get("query_sec", 0) + round(time.time() - t, 1)
    info["gpu_peak_gib"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
    return np.concatenate(rows), idx[np.concatenate(cols)], np.concatenate(vals)


def bge_native(q, pool, k=20):
    return _dense_native(q, pool, "bge_native", k)


def e5_native(q, pool, k=20):
    return _dense_native(q, pool, "e5_native", k)


CHANNELS = {"base": base, "base_wide": base_wide, "addr_only": addr_only, "name_noaddr": name_noaddr, "conj": conj,
            "bm25": bm25, "rescue": rescue, "bge_native": bge_native, "e5_native": e5_native}


# ---------------------------------------------------------------------------
# Production blocker: UNION of channels, deduplicated. Recall is defined on the union. Reciprocal-rank
# fusion only adds ranking features (rrf, per-channel ranks); an optional cap `m` per Source-1 is a
# separate, separately measured choice (0 = keep everything). Rescue pairs are never capped.
# A channel spec is "name" or "name:depth", e.g. "base,conj:20,bm25:20,name_noaddr:5,rescue".
# ---------------------------------------------------------------------------
RRF_K = 60
NO_RANK = 999.0
_DEPTH_ARG = {"base_wide": "k", "addr_only": "k", "name_noaddr": "k", "conj": "k", "bm25": "k",
              "rescue": "max_block", "bge_native": "k", "e5_native": "k"}


def parse_spec(spec):
    """'base,conj:20' -> [('base', {}), ('conj', {'k': 20})]"""
    out = []
    for item in [x for x in spec.split(",") if x]:
        name, _, depth = item.partition(":")
        if name not in CHANNELS:
            raise ValueError(f"unknown channel {name!r}; known: {sorted(CHANNELS)}")
        out.append((item, name, {_DEPTH_ARG[name]: int(depth)} if depth else {}))
    return out


def run_channel(item, q, pool):
    _, name, kw = next(iter(parse_spec(item)))
    return CHANNELS[name](q, pool, **kw)


def _fuse(q, pool, specs, m, df_cap):
    parts = []
    for item, name, kw in specs:
        qa, rb, v = CHANNELS[name](q, pool, **kw)
        d = pd.DataFrame({"q": qa, "r": rb, "v": v}).sort_values(["q", "v"], ascending=[True, False], kind="stable")
        d = d.drop_duplicates(["q", "r"])
        d["rank"] = (d.groupby("q").cumcount() + 1).astype(np.float32)
        d["ch"] = name
        parts.append(d[["q", "r", "rank", "ch"]])
    u = pd.concat(parts, ignore_index=True)
    names = [name for _, name, _ in specs]
    u["s"] = 1.0 / (RRF_K + u["rank"])
    wide = u.pivot_table(index=["q", "r"], columns="ch", values="rank", aggfunc="min")
    wide = wide.reindex(columns=names).fillna(NO_RANK).astype(np.float32)
    wide.columns = [f"rank_{n}" for n in names]
    agg = u.groupby(["q", "r"]).agg(rrf=("s", "sum"), n_ch=("s", "size"))
    f = agg.join(wide).reset_index()
    if m:
        f = f.sort_values(["q", "rrf"], ascending=[True, False], kind="stable")
        keep = (f.groupby("q").cumcount() + 1) <= m
        if "rescue" in names:
            keep |= f["rank_rescue"] < NO_RANK
        f = f[keep.to_numpy()]
    f = f.reset_index(drop=True)
    P = _pool_base(pool, df_cap)
    An, Aa = _query_base(q, P)
    qi, ri = f["q"].to_numpy(), f["r"].to_numpy()
    f["name_cos"] = np.asarray(An[qi].multiply(P["Bn"][ri]).sum(1)).ravel().astype(np.float32)
    f["addr_cos"] = np.asarray(Aa[qi].multiply(P["Ba"][ri]).sum(1)).ravel().astype(np.float32)
    f["rrf"] = f["rrf"].astype(np.float32)
    f["n_ch"] = f["n_ch"].astype(np.float32)
    return f


def retrieve(s1, s23, spec, m=0, df_cap=2500, qchunk=50_000, log=print):
    """Candidate pairs (s1_idx, r_idx, name_cos, addr_cos, rrf, n_ch, rank_<channel>...) within each country."""
    specs = parse_spec(spec) if isinstance(spec, str) else parse_spec(",".join(spec))
    out = []
    for country in sorted(set(s1["country"]) | set(s23["country"])):
        ia = np.flatnonzero((s1["country"] == country).to_numpy())
        ib = np.flatnonzero((s23["country"] == country).to_numpy())
        if len(ia) == 0 or len(ib) == 0:
            continue
        clear()
        pool = s23.iloc[ib].reset_index(drop=True)
        n0 = sum(len(o) for o in out)
        for s in range(0, len(ia), qchunk):
            q = s1.iloc[ia[s:s + qchunk]].reset_index(drop=True)
            f = _fuse(q, pool, specs, m, df_cap)
            f.insert(0, "s1_idx", ia[s + f.pop("q").to_numpy()].astype(np.int32))
            f.insert(1, "r_idx", ib[f.pop("r").to_numpy()].astype(np.int32))
            out.append(f)
            log(f"  retrieve {country}: {min(s + qchunk, len(ia))}/{len(ia)} S1")
        n = sum(len(o) for o in out) - n0
        log(f"  retrieve {country}: {len(ia)} S1 x {len(ib)} S2/S3 -> {n} pairs ({n / len(ia):.1f}/S1)")
        del pool
    clear()
    return pd.concat(out, ignore_index=True)
