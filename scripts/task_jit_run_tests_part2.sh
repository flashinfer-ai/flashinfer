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

# Run each test file separately to isolate CUDA memory issues
pytest -s --junitxml=junit-shard/tests/utils/test_jit_example.py.xml tests/utils/test_jit_example.py
pytest -s --junitxml=junit-shard/tests/utils/test_jit_warmup.py.xml tests/utils/test_jit_warmup.py
pytest -s --junitxml=junit-shard/tests/utils/test_norm.py.xml tests/utils/test_norm.py
pytest -s --junitxml=junit-shard/tests/attention/test_block_sparse.py.xml tests/attention/test_block_sparse.py
pytest -s --junitxml=junit-shard/tests/attention/test_rope.py.xml tests/attention/test_rope.py
pytest -s --junitxml=junit-shard/tests/attention/test_mla_page.py.xml tests/attention/test_mla_page.py
pytest -s --junitxml=junit-shard/tests/utils/test_quantization.py.xml tests/utils/test_quantization.py
