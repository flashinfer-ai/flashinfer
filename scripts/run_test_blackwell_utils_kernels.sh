#!/bin/bash

set -eo pipefail
set -x
: ${JUNIT_DIR:=$(realpath ./junit)}
: ${MAX_JOBS:=$(nproc)}
: ${CUDA_VISIBLE_DEVICES:=0}

pip install -e . -v
pip install --upgrade nvidia-cudnn-cu12
pip install --upgrade cuda-python==12.*

EXIT_CODE=0

test_scripts=(
    "test_activation.py"
    "test_block_sparse_indices_to_vector_sparse_offsets.py"
    "test_block_sparse.py"
    "test_create_ipc_buffer.py"
    "test_fp4_quantize.py"
    "test_fp4_tensor_torch_cute.py"
    "test_fp8_quantize.py"
    "test_green_ctx.py"
    "test_jit_example.py"
    "test_jit_warmup.py"
    "test_logits_processor.py"
    "test_norm.py"
    "test_pod_kernels.py"
    "test_quantization.py"
    "test_sampling.py"
    "test_triton_cascade.py"
)

for test_file in "${test_scripts[@]}"; do
  xml_name="${test_file%.py}.xml"
  pytest -s "tests/${test_file}" --junit-xml="${JUNIT_DIR}/$xml_name" || EXIT_CODE=1
  if [[ -z "${RUN_TO_COMPLETION}" && $EXIT_CODE -ne 0 ]]; then
    exit $EXIT_CODE
  fi
done

exit $EXIT_CODE
