#!/bin/bash
# Per-rank wrapper for bench_ep_matrix.py under `srun --ntasks-per-node=8`.
# Wires NCCL 2.30.7 (first on LD_LIBRARY_PATH) + DOCA, derives torch.distributed
# env from SLURM, and uses file:// rendezvous (EP_SYNC, on the shared mount) so no
# MASTER_ADDR/scontrol is needed. Fabric env (NCCL_GIN_TYPE / NCCL_MNNVL_ENABLE)
# is inherited from the caller. All benchmark args are passed through ("$@").
set -u
export PATH=/opt/flashinfer-venv/bin:/root/.local/bin:$PATH
NCCLLIB=$(python - <<'PY'
import glob, ctypes, os
for p in sorted(set(glob.glob("/opt/flashinfer-venv/**/libnccl.so.2*", recursive=True))):
    try:
        lib = ctypes.CDLL(p); v = ctypes.c_int(); lib.ncclGetVersion(ctypes.byref(v))
        if v.value >= 23007:
            print(os.path.dirname(p)); break
    except Exception:
        pass
PY
)
export LD_LIBRARY_PATH="$NCCLLIB:/opt/mellanox/doca/lib/x86_64-linux-gnu:/usr/local/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

# HT runtime-JIT toolchain: cross-node HT JIT-compiles a scan kernel that
# #includes <nccl_device.h> (ships in the NCCL wheel, not the nccl.ep include).
# Build a combined include dir once per node (localid 0), others wait.
WHEEL_EP=$(python -c "import nccl.ep,os;print(os.path.dirname(nccl.ep.__file__))" 2>/dev/null)
NCCL_INC=$(python -c "import nvidia.nccl,os;print(list(nvidia.nccl.__path__)[0]+'/include')" 2>/dev/null)
JIT_INC=/host/jit_inc
if [ "${SLURM_LOCALID:-0}" = "0" ]; then
  mkdir -p "$JIT_INC"
  ln -sfn "$WHEEL_EP"/include/* "$JIT_INC"/ 2>/dev/null
  ln -sfn "$NCCL_INC"/* "$JIT_INC"/ 2>/dev/null
  touch "$JIT_INC/.ready"
else
  for _ in $(seq 1 50); do [ -f "$JIT_INC/.ready" ] && break; sleep 0.2; done
fi
export NCCL_EP_JIT_SOURCE_DIR="$WHEEL_EP/include/nccl_ep"
export NCCL_EP_JIT_BUILD_INCLUDE_DIR="$JIT_INC"
export NCCL_EP_JIT_CUDA_INCLUDE_DIR="${CUDA_HOME:-/usr/local/cuda}/include"
export NCCL_EP_JIT_NVCC="$(command -v nvcc)"
export NCCL_EP_JIT_CACHE_DIR="/host/jitcache"; mkdir -p /host/jitcache 2>/dev/null
export CPATH="$NCCL_INC:${CPATH:-}"

export RANK="${SLURM_PROCID}"
export WORLD_SIZE="${SLURM_NTASKS}"
export LOCAL_RANK="${SLURM_LOCALID}"
export LOCAL_WORLD_SIZE="${SLURM_NTASKS_PER_NODE:-8}"
export EP_INIT_METHOD="file://${EP_SYNC:?set EP_SYNC to a shared rendezvous path}"

cd /host/flashinfer
exec python benchmarks/bench_ep_matrix.py "$@"
