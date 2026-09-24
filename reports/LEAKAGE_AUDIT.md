# Leakage Audit

| Check | Result | Evidence |
|---|---|---|
| Entity-ID numbers informative? | **No** | corr(S1 id, matched S2 id) = −0.002; median \|diff\| true 2.94e8 vs shuffled 2.93e8 (`src/eda/id_leakage.py`, 501,882 pairs) |
| File row order informative? | **No** | corr(row position S1, row position S2) = 0.001; median gap true 0.292 vs shuffled 0.292 |
| S2/S3 record matched to several S1 | No (partition) | 0 of 7.6M matched IDs repeat |
| Cross-country matches | None in sample | 0 of 76,142 pairs (1% universe) |
| Country used as feature | Not used | avoids a US/India-only mapping that cannot transfer to France |
| Validation contamination | Controlled | GroupKFold by S1 entity; all pairs of one S1 stay in one fold. Blocking and context features use no labels |
| Preprocessing fitted on labels | No | Normalization is rule-based. Blocking IDF is computed from S2/S3 inputs of the same split, with no labels |
| Test inputs used unsupervised | Yes (IDF only) | See RULES_AUDIT "Test data use" (UNKNOWN legality → minimal, documented) |

Conclusion: no exploitable ID or ordering leak. CV of about 0.99 on the 1% universe reflects the
100× lower pool density there, not leakage. Full-density CV is the next measurement.
