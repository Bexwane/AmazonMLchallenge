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


# ---------------------------------------------------------------------------
# Per-S1 expected-F0.5 selection. With calibrated probabilities p_i for one Source-1 entity, keeping
# its top-k candidates has expected F0.5 ~= 1.25 * sum_{i<=k} p_i / (0.25 * c * sum_i p_i + k), and
# predicting nothing scores 1 exactly when the entity is a singleton: P = prod_i (1 - p_i).
# The best k (or empty) is chosen per entity. `c` and `w0` are small corrections tuned out of fold.
# ---------------------------------------------------------------------------
def fit_calibrator(p, y):
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p, y)
    return {"x": iso.X_thresholds_.tolist(), "y": iso.y_thresholds_.tolist()}


def apply_calibrator(cal, p):
    return np.interp(p, np.asarray(cal["x"]), np.asarray(cal["y"])).astype(np.float32)


def expected_f_mask(s1_idx: np.ndarray, p: np.ndarray, c=1.0, w0=1.0) -> np.ndarray:
    order = np.lexsort((-p, s1_idx))
    g, q = s1_idx[order], p[order].astype(np.float64)
    new = np.ones(len(q), bool)
    new[1:] = g[1:] != g[:-1]
    starts = np.flatnonzero(new)
    grp = np.cumsum(new) - 1
    k = np.arange(len(q)) - starts[grp] + 1
    cum = np.cumsum(q)
    before = np.where(starts > 0, cum[starts - 1], 0.0)
    s_k = cum - before[grp]
    total = np.add.reduceat(q, starts)
    ef = 1.25 * s_k / (0.25 * c * total[grp] + k)
    ef0 = np.exp(np.add.reduceat(np.log1p(-np.clip(q, 0, 1 - 1e-7)), starts)) * w0
    best = np.maximum.reduceat(ef, starts)
    kbest = np.minimum.reduceat(np.where(ef >= best[grp], k, np.iinfo(np.int64).max), starts)
    keep_sorted = (k <= kbest[grp]) & (best > ef0)[grp]
    keep = np.zeros(len(p), bool)
    keep[order] = keep_sorted
    return keep


def rule_mask(cand: pd.DataFrame, p: np.ndarray, rule: dict) -> np.ndarray:
    """Pairs kept by a selection rule: {"method": "threshold", "thr": t} or
    {"method": "expf", "cal": calibrator, "c": c, "w0": w0, "floor": f}. Exclusivity is applied first."""
    r_idx = cand["r_idx"].to_numpy()
    if rule["method"] == "threshold":
        return (p >= rule["thr"]) & exclusive_mask(p, r_idx)
    q = apply_calibrator(rule["cal"], p) if rule.get("cal") else p
    q = np.where(exclusive_mask(p, r_idx), q, 0.0)
    keep = expected_f_mask(cand["s1_idx"].to_numpy(), q, rule.get("c", 1.0), rule.get("w0", 1.0))
    return keep & (q >= rule.get("floor", 0.0))


def select_rule(cand, p, s1_ids, r_ids, rule) -> dict:
    sel = cand[rule_mask(cand, p, rule)]
    out = {}
    for a, b in zip(s1_ids[sel["s1_idx"].to_numpy()], r_ids[sel["r_idx"].to_numpy()]):
        out.setdefault(a, []).append(b)
    return out


def selection_study(cand, p, y, s1_ids, r_ids, truth, eval_ids, folds=None, log=print):
    """Compare a global threshold with expected-F selection out of fold. The calibrator is cross-fitted
    over `folds` for the study; the returned best rule carries a calibrator fitted on all rows."""
    res = []
    thr_grid = np.round(np.arange(0.3, 0.925, 0.025), 3)
    for t in thr_grid:
        f = macro_f05(select_rule(cand, p, s1_ids, r_ids, {"method": "threshold", "thr": float(t)}), truth, eval_ids)
        res.append(({"method": "threshold", "thr": float(t)}, f))
    q = np.empty(len(p), np.float32)
    if folds is None:
        folds = np.zeros(len(p), int)
    for k in np.unique(folds):
        tr, va = folds != k, folds == k
        cal_k = fit_calibrator(p[tr] if tr.any() else p, y[tr] if tr.any() else y)
        q[va] = apply_calibrator(cal_k, p[va])
    for c in (0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.4):
        for w0 in (0.6, 0.8, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0):
            rule = {"method": "expf", "c": c, "w0": w0}
            f = macro_f05(select_rule(cand, q, s1_ids, r_ids, rule), truth, eval_ids)
            res.append((rule, f))
    res.sort(key=lambda x: -x[1])
    best = dict(res[0][0])
    if best["method"] == "expf":
        best["cal"] = fit_calibrator(p, y)
    table = [{**{k: v for k, v in r.items() if k != "cal"}, "macro_f05": f} for r, f in res]
    log(f"selection: best {({k: v for k, v in best.items() if k != 'cal'})} -> {res[0][1]:.5f}; "
        f"best threshold -> {max(f for r, f in res if r['method'] == 'threshold'):.5f}")
    return best, res[0][1], table


# ---------------------------------------------------------------------------
# Count matching for an unseen country. In train the number of matches per Source-1 entity has the
# same distribution in every country (mean 3.46, 5.6% singletons in both India and US), so a shift of
# the model's logits that makes a country's predicted matches-per-entity equal the out-of-fold value is
# a label-free recalibration. Validated on the country-holdout stress split before being used.
# ---------------------------------------------------------------------------
def shift_logit(p, delta):
    z = np.log(np.clip(p, 1e-7, 1 - 1e-7) / np.clip(1 - p, 1e-7, 1))
    return (1 / (1 + np.exp(-(z + delta)))).astype(np.float32)


def count_match_delta(cand, p, rule, target_per_s1, n_s1, lo=-8.0, hi=8.0, iters=22):
    """Logit shift so that rule-selected pairs / n_s1 == target_per_s1 (bisection; monotone in delta)."""
    for _ in range(iters):
        mid = (lo + hi) / 2
        per = rule_mask(cand, shift_logit(p, mid), rule).sum() / n_s1
        lo, hi = (mid, hi) if per < target_per_s1 else (lo, mid)
    return (lo + hi) / 2
