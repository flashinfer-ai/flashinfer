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
    "test_batch_decode_kernels.py"
    "test_batch_prefill_kernels.py"
    "test_batch_prefill.py"
    "test_cudnn_decode.py"
    "test_cudnn_prefill_deepseek.py"
    "test_cudnn_prefill.py"
    "test_decode_fp8_calibration_scale.py"
    "test_decode_prefill_lse.py"
    "test_fp8_prefill.py"
    "test_mla_decode_kernel.py"
    "test_non_contiguous_decode.py"
    "test_non_contiguous_prefill.py"
    "test_single_prefill.py"
    "test_tensor_cores_decode.py"
    "test_trtllm_gen_attention.py"
    "test_alibi.py"
    "test_attention_sink_blackwell.py"
    "test_attention_sink.py"
    "test_batch_attention.py"
    "test_hopper_fp8_attention.py"
    "test_blackwell_fmha.py"
    "test_deepseek_mla.py"
    "test_mla_page.py"
    "test_page.py"
    "test_rope.py"
    "test_sliding_window.py"
    "test_trtllm_gen_mla.py"
    "test_hopper.py"
    "test_xqa.py"
    "test_shared_prefix_kernels.py"
    "test_logits_cap.py"
)

for test_file in "${test_scripts[@]}"; do
  xml_name="${test_file%.py}.xml"
  pytest -s "tests/${test_file}" --junit-xml="${JUNIT_DIR}/${xml_name}" || EXIT_CODE=1
  if [[ -z "${RUN_TO_COMPLETION}" && $EXIT_CODE -ne 0 ]]; then
    exit $EXIT_CODE
  fi
done

exit $EXIT_CODE
