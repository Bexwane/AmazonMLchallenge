# Business Entity Resolution: reproduction guide

Pipeline: **normalize → multi-channel blocking → dense cosines → stage-1 LightGBM → stage-2 LightGBM
(learned blocking) → exclusive, count-matched selection**. Final submission = experiment **E015**
(E012-full + bagged stage 1; public LB 0.971723; out-of-fold CV macro F0.5 0.9801 on India + US).

## Environment
- Python 3.11/3.12, `pip install -r requirements.txt`
- ~30 GB RAM, 4 CPU cores. A GPU (we used a Kaggle T4) is needed for two steps only: the `bge_native`
  blocking channel and the dense cosines. Without a GPU the `bge_native` channel is skipped silently and
  recall drops (train pair recall 0.977 instead of 0.982), so run those steps on a GPU.
- Models (downloaded from the Hugging Face hub, both MIT, well under 8B parameters):
  `BAAI/bge-m3` (568M, retrieval of native-script names in blocking) and
  `intfloat/multilingual-e5-small` (118M, revision `614241f6`, pair cosines).

## Run end to end
```bash
pip install -r requirements.txt
DATA=/path/to/student_resource/dataset WORK=/path/to/work OUT=/path/to/output bash run_pipeline.sh
python /path/to/student_resource/utils/validate_submission.py \
    --matching $OUT/matching_results.tsv --candidate $OUT/candidate_pairs.tsv --test-dir $DATA/test --check-ids
```
`run_pipeline.sh` lists every command with its run time (total ≈ 15 h: ≈ 7 h GPU, ≈ 8 h CPU). Every step
caches its result under `$WORK` (parquet / npy / LightGBM model files), so an interrupted run resumes.
`$WORK/experiments/20260927-E012-full/` then holds `cv_metrics.json`, `stack_metrics.json`, `oof.parquet`,
the fold models and `test_pairs_proba.parquet`.

Notes on reproducibility: LightGBM runs are seeded (`ber/config.py`); the GPU nearest-neighbour search of
`bge_native` can order exact ties differently between GPUs, which can change a handful of candidate pairs.

## Code map (`src/ber/`)
| module | role |
|---|---|
| `normalize.py`, `prepare.py` | name/address canonicalization (case, accents, leetspeak, legal forms, street types, Indian state and French department/region names, Indic-script romanization, phonetic skeletons), parallel, cached |
| `blocking.py`, `channels.py` | per-country candidate generation: hashed IDF key vectors (`base`), conjunction keys (`conj`), BM25 (`bm25`), bge-m3 retrieval for native-script names (`bge_native`), name-only search for records without address (`name_noaddr`); union + reciprocal-rank fusion features |
| `dense.py` | multilingual-e5-small cosines of every candidate pair (name, name+address) + rank/gap among the S1's and the record's candidates |
| `features.py` | rapidfuzz string similarities, house-number features, context (rank/gap/competition) features, crowd and name-ambiguity counts (no country feature) |
| `profile.py` | legal-form / business-word / house-number profiles to separate synthetic "twin" businesses from true records |
| `stack.py` | stage-2 features: probability context, sibling agreement, group consensus |
| `postprocess.py` | calibration, per-S1 expected-F0.5 / threshold selection, one-S1-per-record exclusivity, per-country count matching |
| `metric.py` | official macro F0.5 (unit-tested) |
| `run.py` | CLI: `block`, `dense`, `cv`, `stack`, `predict`, `select` (+ experiment options) |

Tests: `PYTHONPATH=src python -m pytest -q tests`.
