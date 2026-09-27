#!/usr/bin/env bash
# End-to-end reproduction of the final submission (experiment E012-full):
#   data -> normalization -> blocking -> dense cosines -> stage-1 LightGBM (3-fold CV)
#        -> stage-2 LightGBM (learned blocking) -> test prediction -> selection -> output/*.tsv
# Hardware used: Kaggle notebook, 4 CPU cores, ~30 GB RAM, one T4 GPU (blocking + dense only).
# Every step caches its result under $WORK, so re-running resumes where it stopped.
set -euo pipefail
cd "$(dirname "$0")"
DATA=${DATA:?set DATA to the dataset folder that contains train/ and test/}
WORK=${WORK:-./work}
OUT=${OUT:-../../output}
EXP=20260927-E012-full
export PYTHONPATH=src
COMMON=(--data "$DATA" --work "$WORK" --exp "$EXP" --cv-frac 0.1 --lr 0.1 --df-cap 2500
        --channels base,conj:20,bm25:5,bge_native:5,name_noaddr:5 --feat v6 --dense-model e5s --stack-feat v3)

# 1. normalization (cached as parquet)
for split in train test; do
  python -c "from ber.prepare import load_prepared; load_prepared('$DATA', '$split', '$WORK')"
done
# 2. blocking: union of 5 retrieval channels per country (GPU needed for bge_native)          ~5 h
python -m ber.run block "${COMMON[@]}" --split train
python -m ber.run block "${COMMON[@]}" --split test
# 3. dense cosines (multilingual-e5-small) for every candidate pair (GPU)                       ~2 h
python -m ber.run dense "${COMMON[@]}" --split train
python -m ber.run dense "${COMMON[@]}" --split test
# 4. stage 1: pair features on a 10% S1 sample, LightGBM, 3-fold GroupKFold by S1 (CPU)        ~3 h
python -m ber.run cv "${COMMON[@]}" --folds 3 --no-stress
# 5. stage 2 on the out-of-fold p1, with learned blocking (CPU)                                 ~1 h
python -m ber.run stack "${COMMON[@]}" --folds 3 --prune-loss 0.001 --min-cand-recall 0.98 --decoy-weight 1.9
# 6. test: stage 1 on all ~114M candidate pairs, prune, stage 2 (CPU)                           ~3.6 h
python -m ber.run predict "${COMMON[@]}" --out "$WORK/output_A"
# 7. final selection: count matching in every country, France target x0.955
python -m ber.run select "${COMMON[@]}" --count-match all --count-scale France=0.955 --out "$OUT"
echo "wrote $OUT/matching_results.tsv and $OUT/candidate_pairs.tsv"
