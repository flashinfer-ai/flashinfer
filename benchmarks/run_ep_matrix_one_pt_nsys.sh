#!/bin/bash
# nsys-profiling variant of run_ep_matrix_one_pt.sh: wraps bench_ep_matrix.py with
# nsys (CUDA+NVTX trace) on rank 0 only; other ranks run plain. Output -> /host/prof_fi/$PROF_TAG.nsys-rep
# FI bench times with torch.cuda.Event (no CUPTI), so nsys does NOT conflict.
set -u
WHEEL_EP=$(python -c "import nccl.ep,os;print(os.path.dirname(nccl.ep.__file__))")
NCCL_LIB=$(python -c "import nvidia.nccl,os;print(list(nvidia.nccl.__path__)[0]+'/lib')")
NCCL_INC=$(python -c "import nvidia.nccl,os;print(list(nvidia.nccl.__path__)[0]+'/include')")
export LD_LIBRARY_PATH="$NCCL_LIB:$WHEEL_EP/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
export NCCL_EP_JIT_SOURCE_DIR="$WHEEL_EP/include/nccl_ep"
export NCCL_EP_JIT_BUILD_INCLUDE_DIR="$WHEEL_EP/include"
export NCCL_EP_JIT_CUDA_INCLUDE_DIR="${CUDA_HOME:-/usr/local/cuda}/include"
export NCCL_EP_JIT_NVCC="$(command -v nvcc)"
export NCCL_EP_JIT_CACHE_DIR="/host/jitcache_pt"; mkdir -p /host/jitcache_pt 2>/dev/null
export CPATH="$NCCL_INC:${CPATH:-}"
export RANK="${SLURM_PROCID}"
export WORLD_SIZE="${SLURM_NTASKS}"
export LOCAL_RANK="${SLURM_LOCALID}"
export LOCAL_WORLD_SIZE="${SLURM_NTASKS_PER_NODE:-8}"
export EP_INIT_METHOD="file://${EP_SYNC:?set EP_SYNC}"
cd /host/flashinfer
PROF_TAG="${PROF_TAG:?set PROF_TAG}"; mkdir -p /host/prof_fi 2>/dev/null
if [ "${SLURM_PROCID}" -eq 0 ]; then
  exec nsys profile -t cuda,nvtx -s none --cpuctxsw=none -f true \
       -o "/host/prof_fi/$PROF_TAG" python benchmarks/bench_ep_matrix.py "$@"
else
  exec python benchmarks/bench_ep_matrix.py "$@"
fi
