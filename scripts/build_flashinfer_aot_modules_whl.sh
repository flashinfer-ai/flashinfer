#!/bin/bash
set -e

# Script to build flashinfer-aot-modules wheel
# This script should be run inside the flashinfer container

echo "=========================================="
echo "Building flashinfer-aot-modules wheel"
echo "=========================================="

# Display build environment info
echo "CUDA Version: ${CUDA_VERSION}"
echo "Architecture: ${ARCH}"
echo "CUDA Architectures: ${FLASHINFER_CUDA_ARCH_LIST}"
echo "Working directory: $(pwd)"
echo ""

# Navigate to the flashinfer-aot-modules directory
cd /workspace/flashinfer-aot-modules

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
