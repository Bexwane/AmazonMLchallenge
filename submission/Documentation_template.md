# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Your Team Name]  
**Team Members:** [List all team members]  
**Submission Date:** 27 September 2026

---

## 1. Executive Summary
A two-stage LightGBM matcher over a high-recall, multi-channel candidate set: five retrieval channels
(hashed IDF keys, conjunction keys, BM25, bge-m3 for native-script names, name-only search for records
without address) reach 98.2% pair recall. Pair features cover string similarity, multilingual dense
cosines and, as our key innovation, **identity profiles** (legal form, business words, house number) and
**competition features** that separate the dataset's synthetic "twin" businesses from true noisy records. Selection
maximizes per-entity expected F0.5 with one S1 per record and per-country count matching for the unseen
country. Public LB 0.971723; out-of-fold CV macro F0.5 0.9801 (bagged stage 1).

---

## 2. Methodology

### 2.1 Problem Analysis
- Train: 2.21M S1, 10.3M S2/S3 records (India, US). Every S1 has 0–5+ matches (mean 3.46); 5.6% are singletons.
  **No S2/S3 record belongs to two S1s** → exclusivity (each record goes to its best S1) is exact.
- Noise: legal-suffix variants (Pvt/Private/praivet, Ltd/Limited, LLC/Co, SARL/S.A.R.L.), typos and leetspeak
  (Physíca1, 8rothers), word reorder, DBA/aka aliases, web forms, native-script names (Devanagari, Tamil,
  Telugu, Bengali, Gurmukhi…), component reordering, state abbreviations, house-number typos/truncation,
  3.3% of records with an empty address.
- **Synthetic twins**: 26% of records belong to no S1. Most false merges were twins that copy a real
  business and change its legal form (record adds a legal form the S1 lacks: twins 45.9% vs true pairs 6.6%),
  add/swap a business word (52% vs 25%) and nudge the house number's last digits.
- **Empty-address records**: 97.7% belong to some S1, but two thirds of our missed ones carry a core name shared
  by ≥2 S1s of the country; they are not attributable from the name alone.
- **Test shift**: test has 5.8 S2/S3 records per S1 vs 4.7 in train (≈1.9× the decoy density; exact-name twins
  roughly double), and a third country (France, 15% of test S1s) absent from training.

### 2.2 Solution Strategy
**Approach Type:** Blocking + two-stage classifier (LightGBM stacking) + per-entity decision rule  
**Core Innovation:** identity profiles (legal form, business-word signature, house-number digit changes) and
record-side competition (dense-cosine rank among the record's candidate S1s, name ambiguity counts, group
consensus) that turn "is this similar?" into "is this record generated from *this* S1 rather than a twin or a
same-name rival?"; learned blocking with stage 1 as a recall-controlled filter; label-free per-country count
matching for the unseen country.

---

## 3. Candidate Generation (Blocking)
- **Blocking keys used** (per country, union of channels, deduplicated):
  - `base`: hashed IDF vectors of normalized name and address tokens (+ phonetic skeletons), top-K sparse search, df cap 2500
  - `conj:20`: conjunctions of name and address keys (e.g. rare name token × house number / street)
  - `bm25:5`: BM25 over name+address with a looser df cap
  - `bge_native:5`: BAAI/bge-m3 embeddings (GPU) for records whose name is in an Indic script
  - `name_noaddr:5`: name-only search for records with an empty address
  - reciprocal-rank fusion scores, per-channel ranks and channel counts are kept as features
- **Candidate pairs generated:** train 145.2M (65.8 per S1), test 114.0M (65.8 per S1). After learned
  blocking (stage 1 as a filter, see §4) the matching model scores **8.25M test pairs (4.8 per S1)**; this is
  `candidate_pairs.tsv`.
- **How you ensured true matches were not lost:** channels were added one by one on a 10k-S1-per-country recall
  probe against the full pool, keeping those that recovered distinct miss buckets (native-script, empty address,
  no surviving key, crowds). Blocking pair recall 98.18% (India 96.8%, US 99.1%), oracle macro F0.5 0.9942.
  Learned blocking may drop only 0.1% of the true candidate pairs (those below 99.9% of all true pairs'
  stage-1 probability, never selected in practice): candidate recall 98.08%.

---

## 4. Matching Model

**Features used (stage 1, 90 features):**
- Name features: rapidfuzz ratio / token-set / token-sort / partial / Jaro-Winkler on the core name, skeleton and
  compact forms; exact-core equality; alias / web / native-script flags; token counts
- Address features: ratio / token-set / token-sort on clean, alphabetic and skeleton address; number-set Jaccard,
  first-number equality, house-number ratio and partial (typos, truncation, "5300" vs "300"); empty-address flag
- Dense: multilingual-e5-small cosine of name and of "name, address" with its rank and gap among the S1's
  candidates and among the record's candidate S1s (the strongest feature group)
- Identity profiles: legal forms added / dropped / same / Private↔Public flip, business-word signature
  added / missing, generator noise words, house-number equality, magnitude and leading digit of the difference
- Competition / context: blocker scores, per-channel ranks, rank and gap of each score within the S1's and the
  record's candidate lists, shared-address crowds, number of S1s and records carrying the same core name
  (and the same core name + legal forms)
- No country feature (France is unseen)

**Stage 2 (~110 features):** stage-1 out-of-fold probability and its per-S1 context (rank, gap, margin, mass),
sibling agreement (the record vs the S1's most confident other records), group consensus (candidates grouped by
exact address, core name, house number, legal forms and full profile: size, probability mass, share, gap), the
profile and pair features above. Trained on the stage-1 out-of-fold predictions with the same folds, on the
learned-blocking candidates. A decoy-weighted variant (negatives from records owned by no S1 ×1.9, matching test's
decoy density) is trained too; the variant and the rule are chosen on a test-like out-of-fold set.

**Model type:** LightGBM (binary, lr 0.1, early stopping), 3-fold GroupKFold by S1 on a 10% S1 sample (14.5M pairs).
Stage 1 is **bagged**: two LightGBM variants on the same features (127 leaves / seed 42 and 255 leaves, feature fraction
0.7 / seed 2027); their out-of-fold probabilities are averaged for stage 2, and all 6 fold models are averaged on test.
Stage 2 idem on the pruned pairs. Test = average of the fold models per stage.  
**Threshold selection method:** out-of-fold study of a global threshold vs per-S1 expected-F0.5 selection (with
an explicit "predict nothing" option); chosen: threshold 0.725 on stage-2 probabilities, after one-S1-per-record
exclusivity. For the test set, logits are shifted per country (bisection) so every country selects the
out-of-fold number of matches per S1 (3.327); France ×0.955 (count probes 0.94–0.97 were flat, India/US ×1.0
best: ×0.98 and ×1.02 for US scored lower).

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** 0.9801 out of fold on 220,140 train S1 (bagged stage 1; single stage 1: 0.9797,
  precision 0.9950, recall 0.9551, singletons 0.9840); public leaderboard **0.971723**.
- **Common false positives (wrong merges):** synthetic twins at the same street with a nudged house number and a
  changed legal form or extra business word (sharply reduced by the profile features); records of a different S1
  sharing an exact name (common names such as "Downtown Hair Studio"); French generic names ("Lille Club SARL")
  with many same-city entities.
- **Common false negatives (missed matches):** empty-address records whose name is carried by several S1s
  (≈55% of the misses among candidates; not attributable); true pairs never retrieved by blocking (1.8%); heavy
  name replacement at the same address; native-script names with rewritten addresses.

| experiment | change | CV macro F0.5 | public LB |
|---|---|---|---|
| E009 | v4 features, stage-2 stacking, expected-F selection, count matching | 0.9647 | 0.950 |
| E011v3 | stage 2: identity profiles, group consensus, name ambiguity, pair similarities | 0.9780 | 0.965 |
| E013 | test-like decoy weighting, France ×0.97 | 0.9779 | 0.96592 |
| E012-full | rebuilt stage 1 with profiles + dense e5 cosines, stage 2 v3 | 0.9797 | 0.971171 |
| **E015** | E012 + bagged stage 1 (second LightGBM variant, averaged) | **0.9801** | **0.971723** |

---

## 6. Conclusion
Recall-first multi-channel blocking plus a stacked LightGBM matcher reaches 0.9801 CV / 0.9717 LB. The largest gains came
from understanding how the data was generated: twin businesses differ from true records in legal form,
business words and house-number digits, and each record belongs to exactly one S1, so record-side competition
(dense-cosine rank among the record's candidate S1s) is the strongest signal. Lessons: validate on a test-like
distribution (more decoys, an unseen country) and prefer label-free calibration (count matching) for unseen
countries over country-specific tuning.

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/`: `src/ber/` (package), `run_pipeline.sh` (every command, in order, with run
times), `README.md` (environment, GPU notes, code map), `requirements.txt`, `tests/` (unit tests of the metric,
normalization, features, profiles, stacking). Entry point: `python -m ber.run <block|dense|cv|stack|predict|select>`;
`run_pipeline.sh` regenerates `output/matching_results.tsv` and `output/candidate_pairs.tsv` from the dataset.

### B. Additional Results
- Blocking (train, all 2.21M S1): pair recall 0.9818, recall@10 0.804 by fused rank, oracle macro F0.5 0.9942.
- Stage 1 alone 0.9781 CV (E009 stage 1 under the same pruning: 0.9607); the dense-cosine rank among the record's candidate S1s is the top stage-1 feature (60% of the split gain).
- Test-like CV (decoys ×1.9): 0.9784 (E013: 0.9763).
- Per-country out of fold: India 0.9735, US 0.9838; test selections per S1 after count matching: 3.327 (India, US),
  3.177 (France).
