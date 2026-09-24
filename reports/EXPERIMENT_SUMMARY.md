# Experiment Summary

| ID | Data | Blocking | Pairs/S1 | Pair recall | CV macro F0.5 | Stress (holdout IN / US) | Thr | Decision |
|---|---|---|---|---|---|---|---|---|
| 20260925-dev-lgb-v1 | 1% universe, all S1 | k40+n10, df≤2000 | 40.9 | 0.9932 | 0.9899 (IN 0.9865, US 0.9921) | 0.9787 / 0.9910 | 0.55 | pipeline OK → scale up |
| 20260925-dev-lgb-v1c | 1% universe, 50% S1 | same | 40.9 | 0.9934 | 0.9887 | 0.9787 / 0.9904 | 0.60 | sampled-CV path OK |

Observations (dev, optimistic):
* Exclusivity is +0.0004 to +0.0012 macro F0.5, mostly on singletons (0.969 → 0.977).
* The threshold curve is flat between 0.5 and 0.75, so the threshold is robust.
* Gain concentrates in context features: comb 0.47, comb_rank_r 0.33, comb_gap_r 0.12. Competition
  between S1s is the main precision signal.
* Transfer: a model trained on US only still scores 0.979 on India, so there's no catastrophic
  country dependence.
