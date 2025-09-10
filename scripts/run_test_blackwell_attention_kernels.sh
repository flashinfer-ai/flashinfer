#!/bin/bash

set -eo pipefail
set -x
: ${JUNIT_DIR:=$(realpath ./junit)}

EXIT_CODE=0

test_scripts=(
  "test_blackwell_fmha.py"
  "test_deepseek_mla.py"
  "test_trtllm_gen_attention.py"
  "test_trtllm_gen_mla.py"
  "test_cudnn_decode.py"
  "test_cudnn_prefill.py"
  "test_cudnn_prefill_deepseek.py"
)

for test_file in "${test_scripts[@]}"; do
  xml_name="${test_file%.py}.xml"
  pytest -s "tests/${test_file}" --junit-xml="${JUNIT_DIR}/$xml_name" || EXIT_CODE=1
done

exit $EXIT_CODE
