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
pytest -s --junitxml=junit-shard/tests/utils/test_sampling.py.xml tests/utils/test_sampling.py
pytest -s --junitxml=junit-shard/tests/utils/test_topk.py.xml tests/utils/test_topk.py
