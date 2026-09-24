# Validation Design

**Primary CV:** GroupKFold (5 folds) over Source-1 entities, run on the candidate pairs of a random
`--cv-frac` sample of train S1s.
* Blocking and context features are computed on **all** train S1s first, so competition between S1s
  and the pool density match test conditions.
* OOF probabilities → exclusivity → threshold grid → official macro F0.5 over the sampled S1s.
  Singletons are included, and S1s with no candidates count as empty predictions.

**Stress CV:** country holdout. Train on all pairs of one country, predict the other with the primary
threshold. This proxies the unseen France in test. A change is promoted only if it does not hurt stress CV.

**Blocking diagnostics:** pair recall (overall / country / source), recall@k of the blocking score,
pairs per S1, and oracle macro F0.5 (a perfect classifier over the candidates) = the ceiling.

**Caveat:** the local dev universe (`make_subset.py 0.01`) keeps all ratios but is 100× sparser, so its
scores are optimistic. Use it for smoke tests only; decisions need full-density runs.
