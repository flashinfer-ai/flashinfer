#!/bin/bash

set -eo pipefail
set -x
: ${JUNIT_DIR:=$(realpath ./junit)}

EXIT_CODE=0

test_scripts=(
  "test_trtllm_gen_fused_moe.py"
  "test_trtllm_cutlass_fused_moe.py"
)

for test_file in "${test_scripts[@]}"; do
  xml_name="${test_file%.py}.xml"
  pytest -s "tests/${test_file}" --junit-xml="${JUNIT_DIR}/$xml_name" || EXIT_CODE=1
  if [[ -z "${RUN_TO_COMPLETION}" && $EXIT_CODE -ne 0 ]]; then
    exit $EXIT_CODE
  fi
done

exit $EXIT_CODE
