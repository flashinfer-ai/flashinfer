#!/bin/bash

set -eo pipefail
set -x
: ${MAX_JOBS:=$(nproc)}
: ${CUDA_VISIBLE_DEVICES:=0}
: ${SKIP_INSTALL:=0}

if [ "$SKIP_INSTALL" = "0" ]; then
  pip install -e . -v
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # avoid memory fragmentation

# Run all tests in a single pytest session for better coverage reporting
pytest -s \
  tests/attention/test_deepseek_mla.py \
  tests/gemm/test_group_gemm.py \
  tests/attention/test_batch_prefill_kernels.py
# NOTE(Zihao): need to fix tile size on KV dimension for head_dim=256 on small shared memory architecture (sm89)
# pytest -s tests/attention/test_batch_attention.py
