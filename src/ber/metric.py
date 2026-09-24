"""Official metric: F_0.5 computed per Source-1 entity, macro-averaged over all Source-1 entities.

Singleton rule: empty truth + empty prediction = 1.0; empty truth + any prediction = 0.0.
Non-empty truth + empty prediction = 0.0 (precision undefined, recall 0).
"""
import numpy as np

BETA2 = 0.25  # beta = 0.5


def f05(pred: set, truth: set) -> float:
    if not truth:
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0
    tp = len(pred & truth)
    if tp == 0:
        return 0.0
    p = tp / len(pred)
    r = tp / len(truth)
    return (1 + BETA2) * p * r / (BETA2 * p + r)


def macro_f05(pred: dict, truth: dict, s1_ids=None) -> float:
    """Average f05 over `s1_ids` (default: every key of `truth`). Missing predictions count as empty."""
    ids = list(truth) if s1_ids is None else list(s1_ids)
    return float(np.mean([f05(set(pred.get(s, ())), truth.get(s, set())) for s in ids])) if ids else float("nan")


def breakdown(pred: dict, truth: dict, s1_ids=None) -> dict:
    """Micro precision/recall plus macro F0.5 split by singleton vs non-singleton."""
    ids = list(truth) if s1_ids is None else list(s1_ids)
    tp = npred = ntrue = 0
    single, multi = [], []
    for s in ids:
        p, t = set(pred.get(s, ())), truth.get(s, set())
        tp += len(p & t); npred += len(p); ntrue += len(t)
        (single if not t else multi).append(f05(p, t))
    return {
        "macro_f05": float(np.mean(single + multi)),
        "micro_precision": tp / npred if npred else float("nan"),
        "micro_recall": tp / ntrue if ntrue else float("nan"),
        "f05_singletons": float(np.mean(single)) if single else float("nan"),
        "f05_nonsingletons": float(np.mean(multi)) if multi else float("nan"),
        "n_s1": len(ids), "n_singletons": len(single),
    }
