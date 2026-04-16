#!/bin/bash

set -eo pipefail
set -x
: ${MAX_JOBS:=$(nproc)}
: ${CUDA_VISIBLE_DEVICES:=0}

# Source test environment setup (handles package overrides like TVM-FFI)
source "$(dirname "${BASH_SOURCE[0]}")/setup_test_env.sh"

# Clean Python bytecode cache to avoid stale imports (e.g., after module refactoring)
echo "Cleaning Python bytecode cache..."
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
find . -type f -name '*.pyc' -delete 2>/dev/null || true
echo "Cache cleaned."
echo ""

pip install -e . -v

# Single-GPU comm tests (simulate multi-rank on one GPU)
pytest -s tests/comm/test_trtllm_moe_alltoall.py

# MOE tests that don't require multi-GPU
pytest -s tests/moe/test_trtllm_cutlass_fused_moe.py

# NOTE: Multi-GPU comm tests (vllm, trtllm allreduce/fusion, nvshmem, etc.)
# have been moved to task_test_multi_gpu_comm_kernels.sh
