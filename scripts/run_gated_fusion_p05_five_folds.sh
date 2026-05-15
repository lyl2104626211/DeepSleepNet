#!/usr/bin/env bash
set -euo pipefail

CONFIG=${CONFIG:-configs/paper_sleep_edf_fpz_cz_eeg_eog_gated_fusion_p05.yaml}
MANIFEST=${MANIFEST:-data/processed/sleep_edf_paper_fpz_cz_eeg_eog/manifest.json}
SPLIT_ROOT=${SPLIT_ROOT:-data/processed/sleep_edf_paper_fpz_cz_eeg_eog}
RESULT_ROOT=${RESULT_ROOT:-results/deepsleepnet_sleep_edf_paper_fpz_cz_eeg_eog_gated_fusion_p05}
PYTHON=${PYTHON:-python}
EOG_DROPOUT_PROB=${EOG_DROPOUT_PROB:-0.5}
SKIP_DONE=${SKIP_DONE:-1}

if [[ -n "${FOLDS:-}" ]]; then
  read -r -a FOLD_LIST <<< "${FOLDS}"
else
  FOLD_LIST=(fold_00 fold_03 fold_07 fold_13 fold_17)
fi

for fold in "${FOLD_LIST[@]}"; do
  if [[ "${SKIP_DONE}" == "1" ]] && [[ -f "${RESULT_ROOT}/${fold}/eval_test_eog_zero/evaluation_test.json" ]]; then
    echo "==> ${fold}: already done, skip"
    continue
  fi

  echo "==> ${fold}: train gated stage1 with EOG dropout p=${EOG_DROPOUT_PROB}"
  "${PYTHON}" main.py train-stage1 \
    --config "${CONFIG}" \
    --manifest "${MANIFEST}" \
    --split "${SPLIT_ROOT}/${fold}.json" \
    --output-dir "${RESULT_ROOT}/${fold}/stage1" \
    --eog-dropout-prob "${EOG_DROPOUT_PROB}"

  echo "==> ${fold}: train gated stage2 with EOG dropout p=${EOG_DROPOUT_PROB}"
  "${PYTHON}" main.py train-stage2 \
    --config "${CONFIG}" \
    --manifest "${MANIFEST}" \
    --split "${SPLIT_ROOT}/${fold}.json" \
    --stage1-checkpoint "${RESULT_ROOT}/${fold}/stage1/best_model.pt" \
    --output-dir "${RESULT_ROOT}/${fold}/stage2" \
    --eog-dropout-prob "${EOG_DROPOUT_PROB}"

  echo "==> ${fold}: evaluate normal"
  "${PYTHON}" main.py evaluate-stage2 \
    --config "${CONFIG}" \
    --manifest "${MANIFEST}" \
    --split "${SPLIT_ROOT}/${fold}.json" \
    --checkpoint "${RESULT_ROOT}/${fold}/stage2/best_model.pt" \
    --subset test \
    --output-dir "${RESULT_ROOT}/${fold}/eval_test"

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
