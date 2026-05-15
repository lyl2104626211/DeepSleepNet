#!/usr/bin/env bash
set -euo pipefail

CONFIG=${CONFIG:-configs/paper_sleep_edf_fpz_cz_v2.yaml}
RAW_DIR=${RAW_DIR:-data/raw/sleep-edf-database-expanded-1.0.0/sleep-cassette}
PROCESSED_DIR=${PROCESSED_DIR:-data/processed/sleep_edf_paper_fpz_cz_v2}
RESULT_ROOT=${RESULT_ROOT:-results/deepsleepnet_sleep_edf_paper_fpz_cz_v2}
PYTHON=${PYTHON:-python}

mkdir -p "${PROCESSED_DIR}" "${RESULT_ROOT}"

"${PYTHON}" main.py preprocess \
  --input-dir "${RAW_DIR}" \
  --output-dir "${PROCESSED_DIR}" \
  --channel Fpz-Cz \
  --trim-wake-minutes 30

for i in $(seq 0 19); do
  fold=$(printf "fold_%02d" "${i}")
  "${PYTHON}" main.py split-subjects \
    --manifest "${PROCESSED_DIR}/manifest.json" \
    --output "${PROCESSED_DIR}/${fold}.json" \
    --group-by participant \
    --n-folds 20 \
    --fold-index "${i}" \
    --seed 42
done

echo "single-channel v2 prepared at ${PROCESSED_DIR}"
