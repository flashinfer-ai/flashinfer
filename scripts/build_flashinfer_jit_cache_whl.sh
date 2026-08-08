#!/bin/bash
set -e

# Script to build flashinfer-jit-cache wheel
# This script should be run inside the flashinfer container

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=scripts/jit_cache_build_common.sh
source "${SCRIPT_DIR}/jit_cache_build_common.sh"

echo "=========================================="
echo "Building flashinfer-jit-cache wheel"
echo "=========================================="

compute_jit_cache_parallelism

# Display build environment info
echo "CUDA Version: ${CUDA_VERSION}"
echo "CPU Architecture: ${ARCH}"
echo "CUDA Major: ${CUDA_MAJOR}"
echo "CUDA Minor: ${CUDA_MINOR}"
echo "FlashInfer Local Version: ${FLASHINFER_LOCAL_VERSION}"
echo "CUDA Architectures: ${FLASHINFER_CUDA_ARCH_LIST}"
echo "Dev Release Suffix: ${FLASHINFER_DEV_RELEASE_SUFFIX}"
echo "MAX_JOBS: ${MAX_JOBS}"
echo "NVCC_THREADS: ${FLASHINFER_NVCC_THREADS}"
echo "Memory Budget per Job: ${MEM_PER_JOB} GB"
echo "Python Version: $(python3 --version)"
echo "Git commit: $(git rev-parse HEAD 2>/dev/null || echo 'unknown')"
echo "Working directory: $(pwd)"
echo ""

# Navigate to the flashinfer-jit-cache directory
cd flashinfer-jit-cache

export CONDA_pkgs_dirs="${FLASHINFER_CI_CACHE}/conda-pkgs"
export XDG_CACHE_HOME="${FLASHINFER_CI_CACHE}/xdg-cache"
mkdir -p "$CONDA_pkgs_dirs" "$XDG_CACHE_HOME"
export HOME=/tmp/home
mkdir -p $HOME
export PATH="$HOME/.local/bin:$PATH"
export PATH="/opt/python/cp312-cp312/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda/lib64/stubs:$LD_LIBRARY_PATH"

echo "::group::Install build system"
pip install --upgrade build
echo "::endgroup::"

# Optional: set up sccache for compiler caching with S3 backend
if [ -n "$SCCACHE_BUCKET" ]; then
  echo "::group::Install sccache"
  export SCCACHE_BUCKET
  setup_sccache "cuda${CUDA_MAJOR}${CUDA_MINOR}-$(uname -m)" "$(cd .. && pwd -P)"
  echo "::endgroup::"
fi

# Clean any previous builds
echo "Cleaning previous builds..."
rm -rf dist build *.egg-info

# Build the wheel using the build module for better isolation.
# PIP_CONSTRAINT pins apache-tvm-ffi inside the isolated build env: it is a
# header-only ABI dependency, so whatever is resolved here becomes the ABI of every
# .so in this wheel. The package metadata deliberately keeps a range instead.
export PIP_CONSTRAINT="${SCRIPT_DIR}/build_constraints.txt"
echo "Building wheel (PIP_CONSTRAINT=${PIP_CONSTRAINT})..."
python -m build --wheel

echo ""
echo "✓ Build completed successfully"
echo ""
echo "Built wheels:"
ls -lh dist/

# The kernels above take their ABI from the apache-tvm-ffi resolved during the build,
# and nothing in the wheel records that. Check the artifact actually loads on the floor
# of the range it advertises. Set FLASHINFER_SKIP_ABI_VERIFY=1 for offline builds.
if [ "${FLASHINFER_SKIP_ABI_VERIFY:-0}" != "1" ]; then
  echo "::group::Verify tvm-ffi ABI against the declared range"
  python "${SCRIPT_DIR}/verify_jit_cache_abi.py" dist/*.whl
  echo "::endgroup::"
fi

# Verify version and git version
echo ""
echo "Verifying version and git version..."
pip install dist/*.whl
python -c "
import flashinfer_jit_cache
print(f'📦 Package version: {flashinfer_jit_cache.__version__}')
print(f'🔖 Git version: {flashinfer_jit_cache.__git_version__}')
"

# Copy wheels to output directory if specified
if [ -n "${OUTPUT_DIR}" ]; then
    echo ""
    echo "Copying wheels to output directory: ${OUTPUT_DIR}"
    mkdir -p "${OUTPUT_DIR}"
    cp -v dist/*.whl "${OUTPUT_DIR}/"
fi

# Print sccache stats if enabled
if [ -n "$SCCACHE_BUCKET" ]; then
  echo "::group::sccache stats"
  sccache --show-stats
  echo "::endgroup::"
fi

echo ""
echo "Build process completed!"
