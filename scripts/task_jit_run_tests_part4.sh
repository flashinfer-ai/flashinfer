#!/bin/bash

set -eo pipefail
set -x
: ${MAX_JOBS:=$(nproc)}
: ${CUDA_VISIBLE_DEVICES:=0}
: ${SKIP_INSTALL:=0}

# Source the pre-install guards and optional dependency overrides.
source "$(dirname "${BASH_SOURCE[0]}")/setup_test_env.sh"

if [ "$SKIP_INSTALL" = "0" ]; then
  pip install -e . -v
  # shellcheck disable=SC1091
  source "$(dirname "${BASH_SOURCE[0]}")/setup_ci_test_env.sh"
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # avoid memory fragmentation

# Run each test file separately to isolate CUDA memory issues
pytest -s tests/attention/test_deepseek_mla.py
pytest -s tests/gemm/test_group_gemm.py
pytest -s tests/attention/test_batch_prefill_kernels.py
pytest -s tests/test_artifacts.py
# NOTE(Zihao): need to fix tile size on KV dimension for head_dim=256 on small shared memory architecture (sm89)
# pytest -s tests/attention/test_batch_attention.py
