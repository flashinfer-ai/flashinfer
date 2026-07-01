#!/usr/bin/env bash
# Fast dev install for FlashInfer + NCCL-EP (moe_ep split comm via nccl.ep).
#
# Usage (from repo root):
#   bash fast_install.sh
#
# What this builds:
#   * libnccl_ep.so  -> flashinfer/moe_ep/backends/split/comm/nccl_ep/_libs/
#   * nccl4py>=0.3.1 wheel (nccl.ep API)
#   * nccl_ep python (import nccl_ep) — ctypes helpers / ndtensor
#
# Override arch for H100:
#   NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90" bash fast_install.sh
#
# Override NCCL-EP wheel pins (match ep_bench defaults):
#   FI_NCCL_VERSION=2.30.7 FI_NCCL4PY_SPEC='nccl4py==0.3.1' bash fast_install.sh

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

# NVIDIA containers export NCCL_VERSION=2.28.3-1 (Git tag, not a PyPI wheel version).
# Do NOT inherit NCCL_VERSION here — use FI_NCCL_VERSION to override our PyPI pin.
NCCL_PYPI_VERSION="${FI_NCCL_VERSION:-2.30.7}"
# Do not use nccl4py[cu13]: its metadata can pull nvidia-nccl-cu13==2.28.3-1.
# We install NCCL/cuda wheels explicitly below.
NCCL4PY_SPEC="${FI_NCCL4PY_SPEC:-nccl4py==0.3.1}"
CUDA_CORE_VERSION="${FI_CUDA_CORE_VERSION:-1.0.1}"
CUDA_BINDINGS_VERSION="${FI_CUDA_BINDINGS_VERSION:-13.2.0}"
# FlashInfer 0.6.x imports CuTe DSL at package load when cutlass is present; the
# venv often has an older nvidia-cutlass-dsl that lacks cutlass.cute.nvgpu.OperandMajorMode.
CUTLASS_DSL_SPEC="${FI_CUTLASS_DSL_SPEC:-nvidia-cutlass-dsl[cu13]>=4.5.0}"

# Ignore NVIDIA pip constraint/config files (pins like nvidia-nccl-cu13==2.28.3-1).
pip_install() {
  env -u PIP_CONSTRAINT \
    PIP_CONSTRAINT= \
    PIP_CONFIG_FILE=/dev/null \
    python -m pip install --no-cache-dir "$@"
}

# moe_ep comm uses nccl.ep from released nccl4py (>=0.3.1). Install NCCL wheels
# before FlashInfer so `pip install -e .` does not downgrade NCCL via torch deps.
echo "Installing nvidia-nccl-cu13==${NCCL_PYPI_VERSION} (${NCCL4PY_SPEC})"
pip_install --no-deps "nvidia-nccl-cu13==${NCCL_PYPI_VERSION}"
pip_install --no-deps "${NCCL4PY_SPEC}"
pip_install --no-deps \
  "cuda-core==${CUDA_CORE_VERSION}" \
  "cuda-bindings==${CUDA_BINDINGS_VERSION}"

git submodule update --init 3rdparty/nccl

# FlashInfer editable only; runtime deps (torch, etc.) should already be in the venv.
pip_install --no-build-isolation --no-deps -e .
pip install nvshmem4py-cu13 nvidia-nvshmem-cu13

# flashinfer.jit.cubin_loader imports filelock; not always pulled transitively.
pip_install filelock

echo "Installing ${CUTLASS_DSL_SPEC} (import flashinfer requires CuTe DSL >= 4.5)"
pip_install "${CUTLASS_DSL_SPEC}"

python3 -c "
from build_backend import _synthesize_nccl_builddir
from pathlib import Path
_synthesize_nccl_builddir(Path('build_nvep/nccl'))
"

# GB200/B200 = sm_100. Override for H100: NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90"
: "${NVCC_GENCODE:=-gencode=arch=compute_100,code=sm_100}"

make -C 3rdparty/nccl/contrib/nccl_ep \
  BUILDDIR="$(pwd)/build_nvep/nccl" \
  "NVCC_GENCODE=${NVCC_GENCODE}" \
  _NCCL_EP_LSA_TEAM_SIZE_MIN=8 \
  _NCCL_EP_LSA_TEAM_SIZE_MAX=8 \
  _NCCL_EP_NUM_LSA_TEAMS_LIST="1" \
  lib -j"$(nproc)"

LIBNCCL_EP_SO="build_nvep/nccl/lib/libnccl_ep.so"
NCCL_EP_LIBS="flashinfer/moe_ep/backends/split/comm/nccl_ep/_libs"

mkdir -p "${NCCL_EP_LIBS}"
cp "${LIBNCCL_EP_SO}" "${NCCL_EP_LIBS}/"

# ctypes nccl_ep helpers (ndtensor); separate from nccl.ep API above.
pip_install --no-deps -e 3rdparty/nccl/contrib/nccl_ep/python

# Guard against any later package pulling torch's older NCCL pin.
pip_install --no-deps --force-reinstall "nvidia-nccl-cu13==${NCCL_PYPI_VERSION}"

export FLASHINFER_DISABLE_VERSION_CHECK=1

echo "=== sanity checks ==="
python -c "
import importlib.util

import nccl.ep  # noqa: F401
from nccl.core import Communicator  # noqa: F401
print('nccl.ep importable:', importlib.util.find_spec('nccl.ep') is not None)

from flashinfer.moe_ep import available_backends
print('flashinfer.moe_ep:', available_backends())
"