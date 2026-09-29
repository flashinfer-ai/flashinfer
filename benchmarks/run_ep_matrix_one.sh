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
WHEEL_EP=$(python -c "import nccl.ep,os;print(os.path.dirname(nccl.ep.__file__))" 2>/dev/null) || true
NCCL_INC=$(python -c "import nvidia.nccl,os;print(list(nvidia.nccl.__path__)[0]+'/include')" 2>/dev/null) || true
JIT_INC=/host/jit_inc
# Fail fast if include discovery failed — otherwise the symlinks below silently
# produce a broken tree and .ready still gets published, wedging the HT JIT later.
[ -n "$WHEEL_EP" ] && [ -d "$WHEEL_EP/include" ] || {
  echo "ERROR: nccl.ep include dir not found (WHEEL_EP='$WHEEL_EP')" >&2; exit 1; }
[ -n "$NCCL_INC" ] && [ -d "$NCCL_INC" ] || {
  echo "ERROR: nvidia.nccl include dir not found (NCCL_INC='$NCCL_INC')" >&2; exit 1; }
if [ "${SLURM_LOCALID:-0}" = "0" ]; then
  mkdir -p "$JIT_INC"
  ln -sfn "$WHEEL_EP"/include/* "$JIT_INC"/
  ln -sfn "$NCCL_INC"/* "$JIT_INC"/
  # The HT JIT needs nccl_device.h reachable in the combined tree; verify before
  # signalling the other ranks to proceed.
  [ -e "$JIT_INC/nccl_device.h" ] || {
    echo "ERROR: nccl_device.h missing from $JIT_INC after symlink" >&2; exit 1; }
  touch "$JIT_INC/.ready"
else
  for _ in $(seq 1 50); do [ -f "$JIT_INC/.ready" ] && break; sleep 0.2; done
  [ -f "$JIT_INC/.ready" ] || {
    echo "ERROR: timed out waiting for $JIT_INC/.ready (rank-0 include setup failed)" >&2; exit 1; }
fi
export NCCL_EP_JIT_SOURCE_DIR="$WHEEL_EP/include/nccl_ep"
export NCCL_EP_JIT_BUILD_INCLUDE_DIR="$JIT_INC"
export NCCL_EP_JIT_CUDA_INCLUDE_DIR="${CUDA_HOME:-/usr/local/cuda}/include"
NVCC="$(command -v nvcc)" || { echo "ERROR: nvcc not found on PATH (needed for the HT JIT)" >&2; exit 1; }
export NCCL_EP_JIT_NVCC="$NVCC"
export NCCL_EP_JIT_CACHE_DIR="/host/jitcache"; mkdir -p /host/jitcache 2>/dev/null
export CPATH="$NCCL_INC:${CPATH:-}"

export RANK="${SLURM_PROCID}"
export WORLD_SIZE="${SLURM_NTASKS}"
export LOCAL_RANK="${SLURM_LOCALID}"
export LOCAL_WORLD_SIZE="${SLURM_NTASKS_PER_NODE:-8}"
export EP_INIT_METHOD="file://${EP_SYNC:?set EP_SYNC to a shared rendezvous path}"

cd /host/flashinfer
exec python benchmarks/bench_ep_matrix.py "$@"
