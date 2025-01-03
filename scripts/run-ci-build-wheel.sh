#!/bin/bash
# adapted from https://github.com/punica-ai/punica/blob/591b59899f0a20760821785d06b331c8a2e5cb86/ci/run-ci-build-wheel.bash
set -e

assert_env() {
    local var_name="$1"
    if [ -z "${!var_name}" ]; then
        echo "Error: Environment variable '$var_name' is not set."
        exit 1
    fi
}

assert_env FLASHINFER_CI_CACHE
assert_env FLASHINFER_CI_PYTHON_VERSION
assert_env FLASHINFER_CI_CUDA_VERSION
assert_env FLASHINFER_CI_TORCH_VERSION
assert_env TORCH_CUDA_ARCH_LIST
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export CONDA_pkgs_dirs="${FLASHINFER_CI_CACHE}/conda-pkgs"
export XDG_CACHE_HOME="${FLASHINFER_CI_CACHE}/xdg-cache"
mkdir -p "$CONDA_pkgs_dirs" "$XDG_CACHE_HOME"
export HOME=/tmp/home
mkdir -p $HOME
export PATH="$HOME/.local/bin:$PATH"
CUDA_MAJOR="${FLASHINFER_CI_CUDA_VERSION%.*}"
CUDA_MINOR="${FLASHINFER_CI_CUDA_VERSION#*.}"
TORCH_MAJOR="${FLASHINFER_CI_TORCH_VERSION%.*}"
TORCH_MINOR="${FLASHINFER_CI_TORCH_VERSION#*.}"
PYVER="${FLASHINFER_CI_PYTHON_VERSION//./}"
export PATH="/opt/python/cp${PYVER}-cp${PYVER}/bin:$PATH"

FLASHINFER_LOCAL_VERSION="cu${CUDA_MAJOR}${CUDA_MINOR}torch${FLASHINFER_CI_TORCH_VERSION}"
if [ -n "${FLASHINFER_GIT_SHA}" ]; then
    FLASHINFER_LOCAL_VERSION="${FLASHINFER_GIT_SHA}.${FLASHINFER_LOCAL_VERSION}"
fi
# wgmma work for cuda 12.3 and above
if [ "$CUDA_MAJOR" -gt 12 ] || { [ "$CUDA_MAJOR" -eq 12 ] && [ "$CUDA_MINOR" -ge 3 ]; }; then
    FLASHINFER_ENABLE_SM90=1
else
    FLASHINFER_ENABLE_SM90=0
fi

echo "::group::Install PyTorch"
pip install torch==${FLASHINFER_CI_TORCH_VERSION}.* --index-url "https://download.pytorch.org/whl/cu${CUDA_MAJOR}${CUDA_MINOR}"
echo "::endgroup::"

echo "::group::Install build system"
pip install ninja numpy
pip install --upgrade setuptools wheel build
echo "::endgroup::"


echo "::group::Build wheel for FlashInfer"
cd "$PROJECT_ROOT"
FLASHINFER_ENABLE_AOT=1 FLASHINFER_LOCAL_VERSION=$FLASHINFER_LOCAL_VERSION FLASHINFER_ENABLE_SM90=$FLASHINFER_ENABLE_SM90 \
    python -m build --no-isolation --wheel
python -m build --no-isolation --sdist
ls -la dist/
echo "::endgroup::"
