"""E010: dense multilingual similarity as pair FEATURES (not another retrieval channel).

Every candidate pair gets the cosine of two multilingual sentence embeddings of the raw text:
  d_name: business name only        (transliteration, rewritten / abbreviated names, native script)
  d_full: "name, address"           (the whole record, as a human would compare it)
Raw text (not the normalized columns) is encoded, so native-script names reach the model unchanged.

Memory: Source-1 embeddings stay on the GPU; S2/S3 records are encoded in chunks of the records that
occur in candidate pairs, and each chunk's pairs are scored immediately (pairs sorted by r_idx). No
full S2/S3 embedding matrix is ever held; the output is one float16 per pair and field.

Models (all MIT, <= 8B params):
  e5s: intfloat/multilingual-e5-small  118M, 384-d (default)
  e5b: intfloat/multilingual-e5-base   278M, 768-d
  bge: BAAI/bge-m3                     568M, 1024-d
BER_DENSE_FAKE=1 swaps in a deterministic character-trigram hashing encoder (numpy, no torch), used
only to smoke-test the plumbing on machines without a GPU.
"""
import os
import time
import zlib

import numpy as np

MODELS = {"e5s": ("intfloat/multilingual-e5-small", "query: "), "e5b": ("intfloat/multilingual-e5-base", "query: "),
          "bge": ("BAAI/bge-m3", "")}
FIELDS = {"name": 32, "full": 64}  # max tokens per field
DENSE_INFO = {}


def texts(df, rows, field, prefix=""):
    name = df["business_name"].iloc[rows].astype(str).to_numpy(object)
    if field == "name":
        t = name
    else:
        t = name + ", " + df["business_address"].iloc[rows].astype(str).to_numpy(object)
    return (prefix + t).tolist() if prefix else t.tolist()


class _Fake:
    """Deterministic trigram hashing embeddings (dim 64), numpy only."""
    dim = 64

    def encode(self, tt, max_len):
        E = np.zeros((len(tt), self.dim), np.float32)
        for i, t in enumerate(tt):
            t = f"  {t.lower()} "
            for j in range(len(t) - 2):
                E[i, zlib.crc32(t[j:j + 3].encode()) % self.dim] += 1.0
        E /= np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-6)
        return E.astype(np.float16)


class _ST:
    """sentence-transformers model in fp16 on one GPU; returns a CUDA fp16 tensor."""

    def __init__(self, key):
        import torch
        from sentence_transformers import SentenceTransformer
        assert torch.cuda.is_available(), "dense features need a GPU (Kaggle: Accelerator GPU T4/P100)"
        self.name, self.prefix = MODELS[key]
        self.m = SentenceTransformer(self.name, device="cuda").half()
        self.dim = self.m.get_sentence_embedding_dimension()
        try:
            from huggingface_hub import model_info
            rev = model_info(self.name).sha
        except Exception:
            rev = "unknown"
        DENSE_INFO.update({"model": self.name, "revision": rev, "dim": self.dim, "dtype": "float16"})

    def encode(self, tt, max_len, bs=512, chunk=200_000):
        import torch
        self.m.max_seq_length = max_len
        out = torch.empty((len(tt), self.dim), dtype=torch.float16, device="cuda")
        for s in range(0, len(tt), chunk):
            e = self.m.encode(tt[s:s + chunk], batch_size=bs, convert_to_tensor=True, normalize_embeddings=True,
                              show_progress_bar=False)
            out[s:s + len(e)] = e.half()
        return out


def load_encoder(key):
    return _Fake() if os.environ.get("BER_DENSE_FAKE") else _ST(key)


def _dot(E1, Er, qi, ri):
    """Row-wise dot products E1[qi] . Er[ri] as float32 numpy (torch or numpy inputs)."""
    if isinstance(E1, np.ndarray):
        return np.einsum("ij,ij->i", E1[qi].astype(np.float32), Er[ri].astype(np.float32))
    import torch
    q = torch.from_numpy(qi.astype(np.int64)).to(E1.device)
    r = torch.from_numpy(ri.astype(np.int64)).to(E1.device)
    return (E1[q] * Er[r]).sum(1, dtype=torch.float32).cpu().numpy()  # fp16 products, fp32 sum


def pair_dense_cos(s1, s23, s1_idx, r_idx, enc, fields=("name", "full"), prefix="", r_chunk=500_000,
                   pair_chunk=1_000_000, log=print):
    """{field: float16 cosine per pair}, aligned with (s1_idx, r_idx)."""
    order = np.argsort(r_idx, kind="stable")
    rs = r_idx[order]
    ur = np.unique(rs)
    out = {}
    for f in fields:
        t0 = time.time()
        E1 = enc.encode(texts(s1, np.arange(len(s1)), f, prefix), FIELDS[f])
        t1 = time.time()
        res = np.empty(len(r_idx), np.float16)
        for a in range(0, len(ur), r_chunk):
            u = ur[a:a + r_chunk]
            Er = enc.encode(texts(s23, u, f, prefix), FIELDS[f])
            lo, hi = np.searchsorted(rs, u[0], "left"), np.searchsorted(rs, u[-1], "right")
            loc = np.searchsorted(u, rs[lo:hi])
            for b in range(lo, hi, pair_chunk):
                e = min(b + pair_chunk, hi)
                res[order[b:e]] = _dot(E1, Er, s1_idx[order[b:e]], loc[b - lo:e - lo])
            del Er
            if (a // r_chunk) % 5 == 0:
                log(f"dense {f}: {min(a + r_chunk, len(ur))}/{len(ur)} S2/S3 records ({(time.time() - t0) / 60:.1f} min)")
        del E1
        DENSE_INFO[f"{f}_sec"] = round(time.time() - t0, 1)
        DENSE_INFO[f"{f}_s1_sec"] = round(t1 - t0, 1)
        DENSE_INFO[f"{f}_records"] = int(len(s1) + len(ur))
        out[f] = res
        log(f"dense {f}: {len(r_idx)} pairs, {len(s1) + len(ur)} records encoded in {(time.time() - t0) / 60:.1f} min")
    return out


def dense_context_arrays(s1_idx, r_idx, D, rows=None) -> dict:
    """Pair cosines plus their competition: rank/gap among the S1's candidates (both fields) and among
    the S2/S3 record's competing S1s (full text). float16, like context2_arrays."""
    from .features import _rank_gap
    sel = slice(None) if rows is None else np.flatnonzero(rows)
    C = {}
    for f, sides in (("full", (("s1", s1_idx), ("r", r_idx))), ("name", (("s1", s1_idx),))):
        sc = np.asarray(D[f], np.float32)
        C[f"d_{f}"] = D[f][sel].astype(np.float16)
        for side, g in sides:
            rank, gap = _rank_gap(sc, g)
            C[f"d_{f}_rank_{side}"] = np.minimum(rank[sel], 2048).astype(np.float16)
            C[f"d_{f}_gap_{side}"] = gap[sel].astype(np.float16)
            del rank, gap
        del sc
    return C
