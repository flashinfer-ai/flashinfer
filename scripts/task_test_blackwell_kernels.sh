#!/bin/bash

set -eo pipefail
set -x
: ${MAX_JOBS:=$(nproc)}
: ${CUDA_VISIBLE_DEVICES:=0}

pip install -e . -v
pip install --upgrade nvidia-cudnn-cu12

# run task_blackwell_utils_kernels.sh
bash scripts/run_test_blackwell_utils_kernels.sh

# run task_blackwell_attention_kernels.sh
bash scripts/run_test_blackwell_attention_kernels.sh

# gemm kernels
bash scripts/run_test_blackwell_gemm_kernels.sh

# moe kernels
bash scripts/run_test_blackwell_moe_kernels.sh
