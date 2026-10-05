#!/bin/bash

set -eo pipefail
set -x
: ${MAX_JOBS:=$(nproc)}
# Preserve the caller's GPU selection; unset means all available devices.

# Source test environment setup (handles package overrides like TVM-FFI)
source "$(dirname "${BASH_SOURCE[0]}")/setup_test_env.sh"

# Clean Python bytecode cache to avoid stale imports (e.g., after module refactoring)
echo "Cleaning Python bytecode cache..."
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
find . -type f -name '*.pyc' -delete 2>/dev/null || true
echo "Cache cleaned."
echo ""

install_flashinfer_editable

# nvshmem4py-cu12 pins cuda-python<=12.9; letting pip resolve its deps on a
# cu13 container downgrades cuda-python/cuda-bindings and makes the next
# requirements resolution evict CUDA torch (aarch64 backtracks to the CPU-only
# wheel -> "Torch not compiled with CUDA enabled"). Install only if missing,
# and --no-deps: the image already ships the right-flavor cuda-python and
# nvidia-nvshmem libraries.
# TODO: Remove once CI container ships with nvshmem4py pre-installed.
python -c "import nvshmem.core" 2>/dev/null || pip install --no-deps nvshmem4py-cu12

# vllm ar
pytest -s tests/comm/test_vllm_custom_allreduce.py
# pcie ipc collectives (intra-node PCIe without NVLink)
# Every case here needs 2, 4 or 8 GPUs. Report insufficient visibility explicitly
# because an all-skipped pytest run still exits 0.
pcie_ipc_gpus=$(python3 -c 'import torch; print(torch.cuda.device_count())')
if [ -n "${FLASHINFER_TEST_PCIE_IPC_ORDERED_4PLUS4:-}" ] && [ "$pcie_ipc_gpus" -lt 8 ]; then
  echo "ERROR: ordered 4+4 coverage requires at least 8 visible GPUs"
  exit 1
fi
if [ "$pcie_ipc_gpus" -lt 2 ]; then
  echo "############################################################"
  echo "# SKIPPING pcie ipc collectives: $pcie_ipc_gpus GPU visible, needs >=2 (>=8 for full"
  echo "# coverage). This is NOT a pass. Set CUDA_VISIBLE_DEVICES=0,1,...,7."
  echo "############################################################"
else
  echo "pcie ipc collectives: $pcie_ipc_gpus GPUs visible (8 needed for full coverage)"
  pytest -s tests/comm/test_pcie_ipc_all_reduce.py
  if [ -n "${FLASHINFER_TEST_PCIE_IPC_ORDERED_4PLUS4:-}" ]; then
    echo "Requiring ordered 4+4 AllGather/ReduceScatter coverage for this lane"
  fi
  pytest -s \
    tests/comm/test_pcie_ipc_all_gather.py \
    tests/comm/test_pcie_ipc_reduce_scatter.py
fi
# trtllm ar + fusion
pytest -s tests/comm/test_trtllm_allreduce_fusion.py
pytest -s tests/moe/test_trtllm_cutlass_fused_moe.py
pytest -s tests/comm/test_trtllm_moe_allreduce_fusion.py
pytest -s tests/comm/test_trtllm_moe_allreduce_fusion_finalize.py
pytest -s tests/comm/test_trtllm_moe_alltoall.py
# nvshmem ar
pytest -s tests/comm/test_nvshmem.py
pytest -s tests/comm/test_nvshmem_allreduce.py
