#!/bin/bash
set -e

# Script to build flashinfer-jit-cache wheel
# This script should be run inside the flashinfer container

echo "=========================================="
echo "Building flashinfer-jit-cache wheel"
echo "=========================================="

# Display build environment info
echo "CUDA Version: ${CUDA_VERSION}"
echo "CPU Architecture: ${ARCH}"
echo "CUDA Major: ${CUDA_MAJOR}"
echo "CUDA Minor: ${CUDA_MINOR}"
echo "CUDA Version Suffix: ${CUDA_VERSION_SUFFIX}"
echo "CUDA Architectures: ${FLASHINFER_CUDA_ARCH_LIST}"
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

# Clean any previous builds
echo "Cleaning previous builds..."
rm -rf dist build *.egg-info

# Build the wheel using the build module for better isolation
echo "Building wheel..."
python -m build --wheel

echo ""
echo "✓ Build completed successfully"
echo ""
echo "Built wheels:"
ls -lh dist/

# Copy wheels to output directory if specified
if [ -n "${OUTPUT_DIR}" ]; then
    echo ""
    echo "Copying wheels to output directory: ${OUTPUT_DIR}"
    mkdir -p "${OUTPUT_DIR}"
    cp -v dist/*.whl "${OUTPUT_DIR}/"
fi

echo ""
echo "Build process completed!"
