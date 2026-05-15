#!/usr/bin/env bash
set -euo pipefail

RESULT_ROOT=${RESULT_ROOT:-results/deepsleepnet_sleep_edf_paper_fpz_cz_v2}
EXPORT_DIR=${EXPORT_DIR:-exports}
ARCHIVE_NAME=${ARCHIVE_NAME:-$(basename "${RESULT_ROOT}")_eval_test.tar.gz}
ARCHIVE_PATH="${EXPORT_DIR}/${ARCHIVE_NAME}"

if [[ ! -d "${RESULT_ROOT}" ]]; then
  echo "result root not found: ${RESULT_ROOT}" >&2
  exit 1
fi

mkdir -p "${EXPORT_DIR}"
tmp_dir=$(mktemp -d)
trap 'rm -rf "${tmp_dir}"' EXIT

package_root="${tmp_dir}/$(basename "${RESULT_ROOT}")"
mkdir -p "${package_root}"

count=0
for fold_dir in "${RESULT_ROOT}"/fold_*; do
  [[ -d "${fold_dir}" ]] || continue

  fold=$(basename "${fold_dir}")
  eval_dir="${fold_dir}/eval_test"
  if [[ ! -d "${eval_dir}" ]]; then
    echo "skip ${fold}: eval_test not found"
    continue
  fi

  mkdir -p "${package_root}/${fold}"
  cp -a "${eval_dir}" "${package_root}/${fold}/"
  count=$((count + 1))
done

if [[ "${count}" -eq 0 ]]; then
  echo "no eval_test directories found under ${RESULT_ROOT}" >&2
  exit 1
fi

tar -czf "${ARCHIVE_PATH}" -C "${tmp_dir}" "$(basename "${RESULT_ROOT}")"

echo "packaged eval_test directories: ${count}"
echo "archive: ${ARCHIVE_PATH}"
