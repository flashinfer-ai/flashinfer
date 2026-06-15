#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# Minimal NCCL-EP install + moe_ep pytest harness for FlashInfer.
#
# Derived from the working on-cluster flow:
#   BUILD_NCCL_EP=1 pip install -e ".[nvep]"
#   pip install -e 3rdparty/nccl/contrib/nccl_ep/python
#   CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m pytest \
#       tests/moe_ep/test_moe_ep_layer_multirank.py -v -m "nvep and gpu_4" \
#       --backend=nccl_ep
#
# Env:
#   PYTHON              python executable (default: python3)
#   SKIP_INSTALL=1      run tests only (expects libnccl_ep.so already staged)
#   FAST_BUILD=1        dev shortcut: narrow nccl_ep template instantiations +
#                       single NVCC_GENCODE arch (much faster than full pip build)
#   NVCC_GENCODE        override arch flags for FAST_BUILD (default: sm_100)
#   BACKEND             nccl_ep (default) or nixl_ep — multirank test only
#   NPROC               torchrun ranks / GPUs (default: 4)
#   CUDA_VISIBLE_DEVICES  GPU list for multirank test (default: 0,1,2,3)
#   PYTEST_EXTRA        extra args forwarded to pytest

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

: "${PYTHON:=python3}"
: "${BACKEND:=nccl_ep}"
: "${NPROC:=4}"
: "${CUDA_VISIBLE_DEVICES:=0,1,2,3}"
: "${SKIP_INSTALL:=0}"
: "${FAST_BUILD:=0}"
: "${BUILD_DIR:=${REPO_ROOT}/build_nvep/nccl}"
: "${CUDA_HOME:=/usr/local/cuda}"
: "${NVCC_GENCODE:=-gencode=arch=compute_100,code=sm_100}"

cd "${REPO_ROOT}"

pip_cmd() {
    "${PYTHON}" -m pip "$@"
}

ensure_nccl_wheel() {
    if ! "${PYTHON}" -c "import nvidia.nccl" 2>/dev/null; then
        echo "=== installing nvidia-nccl-cu13 wheel ==="
        pip_cmd install --no-deps "nvidia-nccl-cu13>=2.30.4"
    fi
}

synthesize_nccl_builddir() {
    "${PYTHON}" - <<'PY'
from build_backend import _nvep_build_root, _synthesize_nccl_builddir

build = _nvep_build_root / "nccl"
_synthesize_nccl_builddir(build)
print(f"synthesized BUILDDIR={build}")
PY
}

stage_nccl_ep_so() {
    local src="${BUILD_DIR}/lib/libnccl_ep.so"
    local dst="${REPO_ROOT}/flashinfer/moe_ep/nccl_ep/_libs/libnccl_ep.so"
    if [[ ! -f "${src}" ]]; then
        echo "ERROR: expected ${src} after nccl_ep build" >&2
        exit 1
    fi
    mkdir -p "$(dirname "${dst}")"
    cp -f "${src}" "${dst}"
    echo "staged ${dst}"
}

fast_build_nccl_ep() {
    echo "=== FAST_BUILD nccl_ep (narrow template instantiations) ==="
    ensure_nccl_wheel
    synthesize_nccl_builddir
    make -C 3rdparty/nccl/contrib/nccl_ep \
        "BUILDDIR=${BUILD_DIR}" \
        "NVCC_GENCODE=${NVCC_GENCODE}" \
        _NCCL_EP_LSA_TEAM_SIZE_MIN=8 \
        _NCCL_EP_LSA_TEAM_SIZE_MAX=8 \
        _NCCL_EP_NUM_LSA_TEAMS_LIST="1" \
        lib -j"$(nproc)"
    stage_nccl_ep_so
    echo "=== editable Python install (skip native rebuild) ==="
    pip_cmd install --no-build-isolation -e .
}

full_build_nccl_ep() {
    echo "=== BUILD_NCCL_EP=1 pip install -e '.[nvep]' (slow: ~20-45 min) ==="
    ensure_nccl_wheel
    BUILD_NCCL_EP=1 pip_cmd install --no-build-isolation -e ".[nvep]"
}

install_nccl_python_wrappers() {
    echo "=== nccl_ep + nccl4py editable installs ==="
    pip_cmd install -e 3rdparty/nccl/contrib/nccl_ep/python
    CUDA_HOME="${CUDA_HOME}" pip_cmd install -e "3rdparty/nccl/bindings/nccl4py[cu13]"
}

probe_backends() {
    "${PYTHON}" - <<'PY'
from flashinfer.moe_ep import available_backends

backends = available_backends()
print("available_backends:", backends)
assert backends, "no moe_ep backends built — install step failed"
PY
}

run_pytests() {
    echo "=== pytest: CPU moe_ep config ==="
    "${PYTHON}" -m pytest tests/moe_ep/test_config.py -v

    if [[ "${BACKEND}" != "nccl_ep" ]]; then
        echo "NOTE: install path builds NCCL-EP only; skipping multirank for ${BACKEND}" >&2
        return 0
    fi

    echo "=== torchrun multirank moe_ep pytest (backend=${BACKEND}, nproc=${NPROC}) ==="
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
        torchrun --nproc_per_node="${NPROC}" \
        -m pytest tests/moe_ep/test_moe_ep_layer_multirank.py \
        -v -m "nvep and gpu_4" --backend="${BACKEND}" ${PYTEST_EXTRA:-}
}

if [[ "${SKIP_INSTALL}" != "1" ]]; then
    if [[ "${FAST_BUILD}" == "1" ]]; then
        fast_build_nccl_ep
    else
        full_build_nccl_ep
    fi
    install_nccl_python_wrappers
    probe_backends
fi

run_pytests
echo "=== install_and_test_moe_ep.sh: OK ==="
