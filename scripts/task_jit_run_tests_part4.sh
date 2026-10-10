#!/bin/bash

set -eo pipefail
set -x
: ${MAX_JOBS:=$(nproc)}
: ${CUDA_VISIBLE_DEVICES:=0}
: ${SKIP_INSTALL:=0}

# Source test environment setup (handles package overrides like TVM-FFI)
source "$(dirname "${BASH_SOURCE[0]}")/setup_test_env.sh"

if [ "$SKIP_INSTALL" = "0" ]; then
  install_flashinfer_editable
fi

# Create JUnit XML output directories for cross-lane coverage analysis
mkdir -p junit-shard/tests/attention junit-shard/tests/utils junit-shard/tests/gemm \
  junit-shard/tests/cli junit-shard/tests/moe junit-shard/tests/experimental \
  junit-shard/tests

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # avoid memory fragmentation

# Run each test file separately to isolate CUDA memory issues
pytest -s --junitxml=junit-shard/tests/attention/test_deepseek_mla.py.xml tests/attention/test_deepseek_mla.py
pytest -s --junitxml=junit-shard/tests/gemm/test_group_gemm.py.xml tests/gemm/test_group_gemm.py
pytest -s --junitxml=junit-shard/tests/attention/test_batch_prefill_kernels.py.xml tests/attention/test_batch_prefill_kernels.py
pytest -s --junitxml=junit-shard/tests/test_artifacts.py.xml tests/test_artifacts.py
# NOTE(Zihao): need to fix tile size on KV dimension for head_dim=256 on small shared memory architecture (sm89)
# pytest -s tests/attention/test_batch_attention.py
