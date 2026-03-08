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

# vllm ar
pytest -s tests/comm/test_vllm_custom_allreduce.py
# trtllm ar + fusion
pytest -s tests/comm/test_trtllm_allreduce.py
pytest -s tests/comm/test_trtllm_allreduce_fusion.py
pytest -s tests/moe/test_trtllm_cutlass_fused_moe.py
pytest -s tests/comm/test_trtllm_moe_allreduce_fusion.py
pytest -s tests/comm/test_trtllm_moe_allreduce_fusion_finalize.py
pytest -s tests/comm/test_trtllm_moe_alltoall.py
# nvshmem ar
pytest -s tests/comm/test_nvshmem.py
pytest -s tests/comm/test_nvshmem_allreduce.py
