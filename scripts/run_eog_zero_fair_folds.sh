#!/usr/bin/env bash
set -euo pipefail

CONFIG=${CONFIG:-configs/paper_sleep_edf_fpz_cz_eeg_eog.yaml}
MANIFEST=${MANIFEST:-data/processed/sleep_edf_paper_fpz_cz_eeg_eog/manifest.json}
SPLIT_ROOT=${SPLIT_ROOT:-data/processed/sleep_edf_paper_fpz_cz_eeg_eog}
RESULT_ROOT=${RESULT_ROOT:-results/deepsleepnet_sleep_edf_paper_fpz_cz_eeg_eog}
PYTHON=${PYTHON:-python}
SKIP_DONE=${SKIP_DONE:-1}

if [[ -n "${FOLDS:-}" ]]; then
  read -r -a FOLD_LIST <<< "${FOLDS}"
else
  FOLD_LIST=()
  for i in $(seq 0 19); do
    FOLD_LIST+=("$(printf "fold_%02d" "${i}")")
  done
fi

for fold in "${FOLD_LIST[@]}"; do
  if [[ "${SKIP_DONE}" == "1" ]] && [[ -f "${RESULT_ROOT}/${fold}/eval_test_eog_zero/evaluation_test.json" ]]; then
    echo "==> ${fold}: already done, skip"
    continue
  fi

  echo "==> ${fold}: evaluate with EOG zeroed"
  "${PYTHON}" main.py evaluate-stage2 \
    --config "${CONFIG}" \
    --manifest "${MANIFEST}" \
    --split "${SPLIT_ROOT}/${fold}.json" \
    --checkpoint "${RESULT_ROOT}/${fold}/stage2/best_model.pt" \
    --subset test \
    --output-dir "${RESULT_ROOT}/${fold}/eval_test_eog_zero" \
    --mask-eog
done
