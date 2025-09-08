#!/bin/bash

set -eo pipefail
set -x
: ${MAX_JOBS:=$(nproc)}
: ${CUDA_VISIBLE_DEVICES:=0}
: ${CI_WORKSPACE:=.}

pip install -e . -v
pip install --upgrade nvidia-cudnn-cu12
pip install --upgrade cuda-python==12.*

EXIT_CODE=0
for script in \
  run_test_blackwell_utils_kernels.sh \
  run_test_blackwell_attention_kernels.sh \
  run_test_blackwell_gemm_kernels.sh \
  run_test_blackwell_moe_kernels.sh
do
  bash scripts/$script || EXIT_CODE=1
done

exit $EXIT_CODE
