# Amazon ML Challenge 2026: Business Entity Resolution

Pipeline: **normalize → block → pair features → LightGBM → exclusive assignment + F0.5-tuned threshold**.
Every stage is cached under `--work`, so a rerun skips finished stages.

```
src/ber/
  normalize.py    name/address canonicalization, Indic-script romanization, phonetic skeleton
  prepare.py      parallel normalization of a split -> parquet cache
  blocking.py     hashed IDF key vectors (name + address), per-country top-K sparse search
  features.py     rapidfuzz string similarities + rank/gap context features (no country feature)
  postprocess.py  one-S1-per-S2/S3 exclusivity, threshold selection
  metric.py       official macro F0.5 (unit-tested)
  run.py          CLI: cv | fit | predict
scripts/          make_subset.py (miniature universe), eval_blocking.py
src/eda/          Day-1 data autopsy scripts
reports/          RULES_AUDIT, DATA_AUTOPSY, LEAKAGE_AUDIT, VALIDATION, EXPERIMENT_SUMMARY, FINAL_APPROACH
state/            machine-readable experiment state
```

## Reproduce (full data, ~30 GB RAM machine: Kaggle CPU/GPU notebook or M2 Pro)

```bash
pip install -r requirements.txt
export PYTHONPATH=src
DATA=/path/to/student_resource/dataset      # folder with train/ and test/
WORK=/path/to/work                          # cache (needs ~10 GB)
EXP=20260925-E001-lgb-v1

python -m ber.run cv      --data $DATA --work $WORK --exp $EXP --cv-frac 0.3   # OOF CV + country-holdout stress CV
python -m ber.run fit     --data $DATA --work $WORK --exp $EXP --cv-frac 0.3   # final model
python -m ber.run predict --data $DATA --work $WORK --exp $EXP --out output    # writes both TSVs
python /path/to/student_resource/utils/validate_submission.py \
  --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir $DATA/test
```

Results land in `$WORK/experiments/$EXP/` (`cv_metrics.json`, `oof.parquet`, `model.txt`,
`fit.json`, `test_pairs_proba.parquet`).

## Quick local smoke test (1% miniature universe, <2 GB RAM)

```bash
python scripts/make_subset.py 0.01                      # -> data/subsets/f0.01/train
python -m pytest -q tests
PYTHONPATH=src python -m ber.run cv --data data/subsets/f0.01 --work data/subsets/work_f0.01 --exp dev --cv-frac 0.5
```

## Blocking experiments (E004-E007, blocking only)
```
python scripts/recall_probe.py --data DATA --work WORK --countries India --out PROBE   # repeat per country
python scripts/recall_probe.py --combine --out PROBE                                    # E003-E007 unions
python scripts/eval_blocking_full.py --data DATA --work WORK --exp E003-fullblock --reports reports
python scripts/eval_blocking_full.py --data DATA --work WORK --exp E007-fullblock --channels "$(cat PROBE/e007_spec.txt)"
python scripts/make_reports.py --baseline WORK/experiments/E003-fullblock/blocking_full.json \
    --probe PROBE/experiments.json --final WORK/experiments/E007-fullblock/blocking_full.json --out reports
```
`notebooks/kaggle_E004_recall_probe.ipynb` runs all of it on Kaggle. A chosen union is used downstream with
`python -m ber.run cv|fit|predict ... --channels "<spec>"` (the candidate cache is shared).
