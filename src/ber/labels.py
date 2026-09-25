"""Attach ground-truth labels to candidate pairs and measure blocking quality."""
import numpy as np
import pandas as pd

from .metric import macro_f05


def owner_map(truth: dict) -> dict:
    return {x: s1 for s1, ids in truth.items() for x in ids}


def owner_index(s1: pd.DataFrame, s23: pd.DataFrame, truth: dict) -> np.ndarray:
    """Positional index into s1 of the entity owning each S2/S3 record (-1 = distractor)."""
    a = [(k, x) for k, v in truth.items() for x in v]
    own = np.full(len(s23), -1, np.int64)
    r_pos = pd.Index(s23["entity_id"]).get_indexer([x for _, x in a])
    s_pos = pd.Index(s1["entity_id"]).get_indexer([k for k, _ in a])
    ok = (r_pos >= 0) & (s_pos >= 0)
    own[r_pos[ok]] = s_pos[ok]
    return own


def label_pairs(cand: pd.DataFrame, s1: pd.DataFrame, s23: pd.DataFrame, truth: dict) -> np.ndarray:
    """1 if the S2/S3 record of the pair belongs to the pair's Source-1 entity (integer arrays only)."""
    own = owner_index(s1, s23, truth)
    return (own[cand["r_idx"].to_numpy()] == cand["s1_idx"].to_numpy()).astype(np.int8)


def blocking_report(cand: pd.DataFrame, y: np.ndarray, s1: pd.DataFrame, s23: pd.DataFrame, truth: dict,
                    rank_out=None) -> dict:
    n_true = sum(len(v) for v in truth.values())
    rep = {"pairs": len(cand), "pairs_per_s1": len(cand) / len(truth), "pair_recall": float(y.sum() / n_true)}
    # recall by country and by source
    true_rows = cand[y == 1]
    s1c = s1["country"].to_numpy()
    for c in sorted(set(s1c)):
        ids = set(s1["entity_id"][s1c == c])
        tot = sum(len(truth[i]) for i in ids if i in truth)
        got = int((s1c[true_rows["s1_idx"].to_numpy()] == c).sum())
        rep[f"recall_{c}"] = got / tot if tot else float("nan")
    src = s23["src"].to_numpy()[true_rows["r_idx"].to_numpy()]
    for s in (2, 3):
        tot = sum(1 for v in truth.values() for x in v if x.startswith(f"S{s}-"))
        rep[f"recall_S{s}"] = int((src == s).sum()) / tot if tot else float("nan")
    # recall if we kept only the top-k by name_cos + addr_cos
    score = cand["name_cos"].to_numpy() + cand["addr_cos"].to_numpy()
    rank = pd.Series(-score).groupby(cand["s1_idx"].to_numpy()).rank(method="first").to_numpy(np.float32)
    if rank_out is not None:
        rank_out.append(rank)
    for k in (1, 3, 5, 10, 20, 30):
        rep[f"recall@{k}"] = float(y[rank <= k].sum() / n_true)
    # oracle: perfect classifier over the candidate set
    ids = s1["entity_id"].to_numpy()
    pred = {}
    for i, r in zip(true_rows["s1_idx"].to_numpy(), s23["entity_id"].to_numpy()[true_rows["r_idx"].to_numpy()]):
        pred.setdefault(ids[i], []).append(r)
    rep["oracle_macro_f05"] = macro_f05(pred, truth)
    return rep
