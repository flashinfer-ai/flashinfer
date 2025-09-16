#!/bin/bash

set -eo pipefail
set -x
: ${JUNIT_DIR:=$(realpath ./junit)}

EXIT_CODE=0

test_scripts=(
  "test_mm_fp4.py"
  "test_groupwise_scaled_gemm_fp8.py"
  "test_groupwise_scaled_gemm_mxfp4.py"
  "test_cute_dsl_blockscaled_gemm.py"
)

for test_file in "${test_scripts[@]}"; do
  xml_name="${test_file%.py}.xml"
  pytest -s "tests/${test_file}" --junit-xml="${JUNIT_DIR}/$xml_name" || EXIT_CODE=1
  if [[ -z "${RUN_TO_COMPLETION}" && $EXIT_CODE -ne 0 ]]; then
    exit $EXIT_CODE
  fi
done

exit $EXIT_CODE
