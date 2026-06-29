#!/bin/bash
# Per-rank wrapper for bench_ep_matrix.py in the PyTorch-based FlashInfer-EP
# container (docker/Dockerfile.flashinfer-ep-pytorch / flashinfer-ep-pt2605.sqsh).
#
# Unlike run_ep_matrix_one.sh (CUDA-13.0 devel image, /opt/flashinfer-venv), this
# image uses the base image's SYSTEM python + torch + complete CUDA 13.2 toolkit
# and the matching IB-GDAKI/GPUDirect stack — the environment in which cross-node
# NCCL-EP HIGH_THROUGHPUT actually works. The HT JIT therefore just uses the base
# `nvcc` (no combined include dir): nccl_device.h is provided via CPATH from the
# nvidia-nccl wheel, exactly like ep_bench/scripts/env.sh.
set -euo pipefail

# Base image already has python+torch on PATH; resolve the wheel-installed NCCL
# + nccl.ep payloads by import (locations differ from the venv image). Fail fast
# with an actionable message instead of leaving empty LD_LIBRARY_PATH/CPATH
# entries that surface later as obscure JIT / dlopen errors.
WHEEL_EP=$(python -c "import nccl.ep,os;print(os.path.dirname(nccl.ep.__file__))") \
    || { echo "ERROR: cannot import nccl.ep — is the NCCL-EP wheel installed?" >&2; exit 1; }
NCCL_LIB=$(python -c "import nvidia.nccl,os;print(list(nvidia.nccl.__path__)[0]+'/lib')") \
    || { echo "ERROR: cannot import nvidia.nccl (nvidia-nccl-cu13 wheel missing?)" >&2; exit 1; }
NCCL_INC=$(python -c "import nvidia.nccl,os;print(list(nvidia.nccl.__path__)[0]+'/include')") \
    || { echo "ERROR: cannot resolve nvidia.nccl include dir" >&2; exit 1; }
NVCC="$(command -v nvcc)" \
    || { echo "ERROR: nvcc not found on PATH (needed for the HT JIT)" >&2; exit 1; }
for _p in "$WHEEL_EP" "$NCCL_LIB" "$NCCL_INC"; do
    [ -n "$_p" ] && [ -d "$_p" ] || { echo "ERROR: resolved path missing: '$_p'" >&2; exit 1; }
done
export LD_LIBRARY_PATH="$NCCL_LIB:$WHEEL_EP/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

# HT runtime JIT: use the base image's complete CUDA toolkit (>=13.2, matching
# NCCL's +cuda13.3 build). nccl_device.h ships in the nvidia-nccl wheel include,
# reached via CPATH (nvcc honors it) — no combined symlink dir needed.
export NCCL_EP_JIT_SOURCE_DIR="$WHEEL_EP/include/nccl_ep"
export NCCL_EP_JIT_BUILD_INCLUDE_DIR="$WHEEL_EP/include"
export NCCL_EP_JIT_CUDA_INCLUDE_DIR="${CUDA_HOME:-/usr/local/cuda}/include"
export NCCL_EP_JIT_NVCC="$NVCC"
export NCCL_EP_JIT_CACHE_DIR="/host/jitcache_pt"; mkdir -p /host/jitcache_pt || true
export CPATH="$NCCL_INC:${CPATH:-}"

export RANK="${SLURM_PROCID}"
export WORLD_SIZE="${SLURM_NTASKS}"
export LOCAL_RANK="${SLURM_LOCALID}"
export LOCAL_WORLD_SIZE="${SLURM_NTASKS_PER_NODE:-8}"
export EP_INIT_METHOD="file://${EP_SYNC:?set EP_SYNC to a shared rendezvous path}"

cd /host/flashinfer
exec python benchmarks/bench_ep_matrix.py "$@"
