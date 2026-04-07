#!/bin/bash

set -eo pipefail
set -x

echo "========================================"
echo "Starting flashinfer-jit-cache test script"
echo "========================================"

# MAX_JOBS = min(nproc, max(1, MemAvailable_GB/(8 on aarch64, 4 otherwise)))
MEM_AVAILABLE_GB=$(free -g | awk '/^Mem:/ {print $7}')
NPROC=$(nproc)
MAX_JOBS=$(( MEM_AVAILABLE_GB / $([ "$(uname -m)" = "aarch64" ] && echo 8 || echo 4) ))
if (( MAX_JOBS < 1 )); then
  MAX_JOBS=1
elif (( NPROC < MAX_JOBS )); then
  MAX_JOBS=$NPROC
fi

echo "System Information:"
echo "  - Available Memory: ${MEM_AVAILABLE_GB} GB"
echo "  - Number of Processors: ${NPROC}"
echo "  - MAX_JOBS: ${MAX_JOBS}"

# Export MAX_JOBS for PyTorch's cpp_extension to use
export MAX_JOBS

if [ -z "${FLASHINFER_AOT_BUILD_PROFILE:-}" ]; then
    if [ -n "${CI:-}" ] || [ -n "${GITHUB_ACTIONS:-}" ] || [ -n "${JENKINS_HOME:-}" ] || [ -n "${JENKINS_URL:-}" ]; then
        export FLASHINFER_AOT_BUILD_PROFILE="full"
    else
        export FLASHINFER_AOT_BUILD_PROFILE="edge_fm"
    fi
fi
echo "FLASHINFER_AOT_BUILD_PROFILE: ${FLASHINFER_AOT_BUILD_PROFILE}"

: ${CUDA_VISIBLE_DEVICES:=""}
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"

# Clean Python bytecode cache to avoid stale imports (e.g., after module refactoring)
echo "Cleaning Python bytecode cache..."
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
find . -type f -name '*.pyc' -delete 2>/dev/null || true
echo "Cache cleaned."

echo ""
echo "Detecting CUDA architecture list..."
if [ -z "${FLASHINFER_CUDA_ARCH_LIST:-}" ]; then
    export FLASHINFER_CUDA_ARCH_LIST=$(python3 -c '
import torch

archs = set()
try:
    for device in range(torch.cuda.device_count()):
        major, minor = torch.cuda.get_device_capability(device)
        suffix = f"{minor}a" if major >= 9 else str(minor)
        archs.add((major, suffix))
except Exception:
    pass

if archs:
    print(" ".join(f"{major}.{minor}" for major, minor in sorted(archs)))
else:
    print("8.0")
')
    echo "FLASHINFER_CUDA_ARCH_LIST was not set; defaulting to visible GPU archs (fallback: 8.0)."
else
    echo "Using pre-set FLASHINFER_CUDA_ARCH_LIST from environment."
fi
echo "FLASHINFER_CUDA_ARCH_LIST: ${FLASHINFER_CUDA_ARCH_LIST}"

echo ""
echo "Current PyTorch version:"
python -c "import torch; print(torch.__version__)"

# Detect CUDA version from the container
CUDA_VERSION=$(python3 -c 'import torch; print(torch.version.cuda)' | cut -d'.' -f1,2 | tr -d '.')
echo "Detected CUDA version: cu${CUDA_VERSION}"

echo ""
echo "========================================"
echo "Installing flashinfer package"
echo "========================================"
pip install -e . || {
    echo "ERROR: Failed to install flashinfer package"
    exit 1
}
echo "✓ Flashinfer package installed successfully"

echo ""
echo "========================================"
echo "Building flashinfer-jit-cache wheel"
echo "========================================"
cd flashinfer-jit-cache
python -m build --wheel

# Get the built wheel file
WHEEL_FILE=$(ls dist/*.whl | head -n 1)
echo ""
echo "Built wheel: $WHEEL_FILE"
echo ""

echo ""
echo "========================================"
echo "Installing flashinfer-jit-cache wheel"
echo "========================================"
echo "Wheel file: $WHEEL_FILE"
pip install "$WHEEL_FILE" || {
    echo "ERROR: Failed to install flashinfer-jit-cache wheel"
    exit 1
}
echo "✓ Flashinfer-jit-cache wheel installed successfully"
cd ..

# Verify installation
echo ""
echo "========================================"
echo "Running verification tests"
echo "========================================"

# Test with show-config
echo "[STEP 1/2] Running 'python -m flashinfer show-config'..."
python -m flashinfer show-config || {
    echo "ERROR: Failed to run 'python -m flashinfer show-config'"
    exit 1
}
echo "✓ show-config completed successfully"

# Verify all modules are compiled
echo ""
echo "[STEP 2/2] Verifying all modules are compiled..."
python scripts/verify_all_modules_compiled.py || {
    echo "ERROR: Not all modules are compiled!"
    exit 1
}
echo "✓ All modules verified successfully"

echo ""
echo "========================================"
echo "✓✓✓ ALL TESTS PASSED! ✓✓✓"
echo "========================================"
