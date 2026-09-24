"""Turn pair probabilities into per-Source-1 match lists.

1. Exclusivity: in training every S2/S3 record belongs to at most one Source-1 entity, so each
   S2/S3 record is kept only for its highest-probability Source-1 candidate.
2. Selection: keep pairs whose probability clears a threshold tuned on out-of-fold data for the
   official macro F0.5.
"""
import numpy as np
import pandas as pd

from .metric import macro_f05


def exclusive_mask(p: np.ndarray, r_idx: np.ndarray) -> np.ndarray:
    order = np.lexsort((-p, r_idx))
    first = np.ones(len(order), bool)
    first[1:] = r_idx[order][1:] != r_idx[order][:-1]
    keep = np.zeros(len(p), bool)
    keep[order[first]] = True
    return keep


def select(cand: pd.DataFrame, p: np.ndarray, s1_ids: np.ndarray, r_ids: np.ndarray, thr: float, exclusive=True) -> dict:
    keep = p >= thr
    if exclusive:
        keep &= exclusive_mask(p, cand["r_idx"].to_numpy())
    sel = cand[keep]
    out = {}
    for a, b in zip(s1_ids[sel["s1_idx"].to_numpy()], r_ids[sel["r_idx"].to_numpy()]):
        out.setdefault(a, []).append(b)
    return out


def tune_threshold(cand, p, s1_ids, r_ids, truth, eval_ids, grid=None, exclusive=True):
    grid = np.round(np.arange(0.2, 0.96, 0.05), 2) if grid is None else grid
    res = {float(t): macro_f05(select(cand, p, s1_ids, r_ids, t, exclusive), truth, eval_ids) for t in grid}
    best = max(res, key=res.get)
    return best, res
