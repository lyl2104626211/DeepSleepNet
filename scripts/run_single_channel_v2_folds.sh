#!/usr/bin/env bash
set -euo pipefail

CONFIG=${CONFIG:-configs/paper_sleep_edf_fpz_cz_v2.yaml}
MANIFEST=${MANIFEST:-data/processed/sleep_edf_paper_fpz_cz_v2/manifest.json}
SPLIT_ROOT=${SPLIT_ROOT:-data/processed/sleep_edf_paper_fpz_cz_v2}
RESULT_ROOT=${RESULT_ROOT:-results/deepsleepnet_sleep_edf_paper_fpz_cz_v2}
PYTHON=${PYTHON:-python}
SKIP_DONE=${SKIP_DONE:-1}

for i in $(seq 0 19); do
  fold=$(printf "fold_%02d" "${i}")

  if [[ "${SKIP_DONE}" == "1" ]] && [[ -f "${RESULT_ROOT}/${fold}/eval_test/evaluation_test.json" ]]; then
    echo "==> ${fold}: already done, skip"
    continue
  fi

  echo "==> ${fold}: train stage1"
  "${PYTHON}" main.py train-stage1 \
    --config "${CONFIG}" \
    --manifest "${MANIFEST}" \
    --split "${SPLIT_ROOT}/${fold}.json" \
    --output-dir "${RESULT_ROOT}/${fold}/stage1"

  echo "==> ${fold}: train stage2"
  "${PYTHON}" main.py train-stage2 \
    --config "${CONFIG}" \
    --manifest "${MANIFEST}" \
    --split "${SPLIT_ROOT}/${fold}.json" \
    --stage1-checkpoint "${RESULT_ROOT}/${fold}/stage1/best_model.pt" \
    --output-dir "${RESULT_ROOT}/${fold}/stage2"

  echo "==> ${fold}: evaluate"
  "${PYTHON}" main.py evaluate-stage2 \
    --config "${CONFIG}" \
    --manifest "${MANIFEST}" \
    --split "${SPLIT_ROOT}/${fold}.json" \
    --checkpoint "${RESULT_ROOT}/${fold}/stage2/best_model.pt" \
    --subset test \
    --output-dir "${RESULT_ROOT}/${fold}/eval_test"
done
