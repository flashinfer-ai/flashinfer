#!/bin/bash

set -eo pipefail
set -x
: ${JUNIT_DIR:=$(realpath ./junit)}

pip install -e . -v

EXIT_CODE=0
scripts_to_run=(
  "run_test_blackwell_utils_kernels.sh"
  "run_test_blackwell_attention_kernels.sh"
  "run_test_blackwell_gemm_kernels.sh"
  "run_test_blackwell_moe_kernels.sh"
  "run_test_blackwell_comm_kernels.sh"
)
for script in "${scripts_to_run[@]}"; do
  bash "scripts/$script" || EXIT_CODE=1
  if [[ -z "${RUN_TO_COMPLETION}" && $EXIT_CODE -ne 0 ]]; then
    exit $EXIT_CODE
  fi
done

exit $EXIT_CODE
