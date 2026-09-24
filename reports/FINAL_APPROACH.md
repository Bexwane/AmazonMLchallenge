# Final Approach (living document; no result is final until the Reproducibility Gate)

1. **Problem understanding:** Source-1 is a deduplicated reference. S2/S3 records map to at most one
   S1 (a partition), and about 26% of S2/S3 are distractors. The metric is macro F0.5 per S1, so
   singletons (5.6%) matter and precision is weighted 2×.
2. **Dataset:** see DATA_AUTOPSY.md.
3. **Metric:** `src/ber/metric.py` (unit-tested).
4. **Validation:** GroupKFold by S1 + country-holdout stress CV (VALIDATION.md).
5. **Key insights:** name-and-address blocking; Indic romanization + phonetic skeleton; precision
   from competition between S1 records (exclusivity + reverse-rank features).
6. **Feature engineering:** see `src/ber/features.py`. No country feature.
7. **Models tested:** LightGBM v1 (dev only so far).
8. **Ablations:** exclusivity on/off (dev).
9. **Error analysis:** TODO after the full-density run.
10. **Final solution:** TODO.
11. **Post-processing:** exclusive assignment + OOF-tuned threshold.
12. **Why selected:** TODO.
