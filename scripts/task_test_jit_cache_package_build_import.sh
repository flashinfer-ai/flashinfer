#!/bin/bash

set -eo pipefail
set -x

# Source test environment setup (handles package overrides like TVM-FFI)
source "$(dirname "${BASH_SOURCE[0]}")/setup_test_env.sh"

echo "========================================"
echo "Starting flashinfer-jit-cache test script"
echo "========================================"

: ${CUDA_VISIBLE_DEVICES:=""}
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"

# Clean Python bytecode cache to avoid stale imports (e.g., after module refactoring)
echo "Cleaning Python bytecode cache..."
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
find . -type f -name '*.pyc' -delete 2>/dev/null || true
echo "Cache cleaned."

echo ""
echo "Detecting CUDA architecture list..."
export FLASHINFER_CUDA_ARCH_LIST=$(python3 -c '
import torch
cuda_ver = torch.version.cuda
arches = ["7.5", "8.0", "8.9", "9.0a"]
if cuda_ver is not None:
    try:
        major, minor = map(int, cuda_ver.split(".")[:2])
        if (major, minor) >= (13, 0):
            arches.append("10.0a")
            arches.append("10.3a")
            arches.append("11.0a")
            arches.append("12.0f")
        elif (major, minor) >= (12, 9):
            arches.append("10.0a")
            arches.append("10.3a")
            arches.append("12.0f")
        elif (major, minor) >= (12, 8):
            arches.append("10.0a")
            arches.append("12.0a")
    except Exception:
        pass
print(" ".join(arches))
')
echo "FLASHINFER_CUDA_ARCH_LIST: ${FLASHINFER_CUDA_ARCH_LIST}"

echo ""
echo "Current PyTorch version:"
python -c "import torch; print(torch.__version__)"

# Detect CUDA version from the container
CUDA_VERSION=$(python3 -c 'import torch; print(torch.version.cuda)' | cut -d'.' -f1,2 | tr -d '.')
echo "Detected CUDA version: cu${CUDA_VERSION}"

# Parallelism: coordinate ninja jobs (MAX_JOBS) with nvcc internal threads (FLASHINFER_NVCC_THREADS).
# Each nvcc invocation compiles multiple -gencode targets; --threads=N parallelizes them.
MEM_AVAILABLE_GB=$(free -g | awk '/^Mem:/ {print $7}')
NPROC=$(nproc)

NUM_ARCHS=$(echo "${FLASHINFER_CUDA_ARCH_LIST}" | wc -w)
NVCC_THREADS=${FLASHINFER_NVCC_THREADS:-${NUM_ARCHS}}
if (( NVCC_THREADS > 8 )); then NVCC_THREADS=8; fi
if (( NVCC_THREADS < 1 )); then NVCC_THREADS=1; fi

# Memory budget: ~2GB per nvcc thread
MEM_PER_JOB=$(( NVCC_THREADS * 2 ))
MAX_JOBS=$(( MEM_AVAILABLE_GB / MEM_PER_JOB ))
if (( MAX_JOBS < 1 )); then MAX_JOBS=1; fi

# Cap total threads at available CPUs
TOTAL_THREADS=$(( MAX_JOBS * NVCC_THREADS ))
if (( TOTAL_THREADS > NPROC )); then
  MAX_JOBS=$(( NPROC / NVCC_THREADS ))
  if (( MAX_JOBS < 1 )); then MAX_JOBS=1; fi
fi

export MAX_JOBS
export FLASHINFER_NVCC_THREADS="${NVCC_THREADS}"

echo "System Information:"
echo "  - Available Memory: ${MEM_AVAILABLE_GB} GB"
echo "  - Number of Processors: ${NPROC}"
echo "  - MAX_JOBS: ${MAX_JOBS}"
echo "  - NVCC_THREADS: ${NVCC_THREADS}"

echo ""
echo "========================================"
echo "Installing flashinfer package"
echo "========================================"
pip install -e . || {
    echo "ERROR: Failed to install flashinfer package"
    exit 1
}
echo "✓ Flashinfer package installed successfully"

# Set up sccache for compiler caching with S3 backend.
# Uses read-write mode when AWS credentials are available (nightly/release builds),
# otherwise falls back to read-only anonymous access to the public cache bucket.
SCCACHE_BUCKET="${SCCACHE_BUCKET:-flashinfer-sccache}"
SCCACHE_REGION="${SCCACHE_REGION:-us-east-2}"

echo ""
echo "========================================"
echo "Setting up sccache"
echo "========================================"
SCCACHE_VERSION="0.9.1"
SCCACHE_ARCH=$(uname -m)
curl -fsSL "https://github.com/mozilla/sccache/releases/download/v${SCCACHE_VERSION}/sccache-v${SCCACHE_VERSION}-${SCCACHE_ARCH}-unknown-linux-musl.tar.gz" | tar xz
mv "sccache-v${SCCACHE_VERSION}-${SCCACHE_ARCH}-unknown-linux-musl/sccache" /usr/local/bin/
rm -rf "sccache-v${SCCACHE_VERSION}-${SCCACHE_ARCH}-unknown-linux-musl"
chmod +x /usr/local/bin/sccache

export SCCACHE_BUCKET
export SCCACHE_REGION
export SCCACHE_S3_KEY_PREFIX="cuda${CUDA_VERSION}-${SCCACHE_ARCH}"
export SCCACHE_IDLE_TIMEOUT=0
export FLASHINFER_NVCC_LAUNCHER="sccache"
export FLASHINFER_CXX_LAUNCHER="sccache"

# If no AWS credentials, use anonymous read-only access to public bucket
if [ -z "${AWS_ACCESS_KEY_ID:-}" ]; then
  export SCCACHE_S3_NO_CREDENTIALS=true
  echo "sccache mode: read-only (public bucket, no credentials)"
else
  echo "sccache mode: read-write"
fi

sccache --start-server
echo "sccache version: $(sccache --version)"
echo "sccache bucket: ${SCCACHE_BUCKET}"
echo "sccache region: ${SCCACHE_REGION}"
echo "sccache prefix: ${SCCACHE_S3_KEY_PREFIX}"

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
echo "sccache stats"
echo "========================================"
sccache --show-stats

echo ""
echo "========================================"
echo "✓✓✓ ALL TESTS PASSED! ✓✓✓"
echo "========================================"
