#!/bin/bash

set -eo pipefail
set -x

# This script installs nightly build packages and runs tests
# Expected dist directories to be in current directory or specified via env vars

: ${TEST_SHARD:=1}
: ${CUDA_VISIBLE_DEVICES:=0}
: ${DIST_CUBIN_DIR:=dist-cubin}
: ${DIST_JIT_CACHE_DIR:=dist-jit-cache}
: ${DIST_PYTHON_DIR:=dist-python}

# Clean Python bytecode cache to avoid stale imports (e.g., after module refactoring)
echo "Cleaning Python bytecode cache..."
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
find . -type f -name '*.pyc' -delete 2>/dev/null || true
echo "Cache cleaned."
echo ""

# Display GPU information (running inside Docker container with GPU access)
echo "=== GPU Information ==="
nvidia-smi

# Install flashinfer packages
echo "Installing flashinfer-cubin from ${DIST_CUBIN_DIR}..."
pip install ${DIST_CUBIN_DIR}/*.whl

echo "Installing flashinfer-jit-cache from ${DIST_JIT_CACHE_DIR}..."
pip install ${DIST_JIT_CACHE_DIR}/*.whl

# Disable JIT to verify that jit-cache package contains all necessary
# precompiled modules for the test suite to pass without compilation
echo "Disabling JIT compilation to test with precompiled cache only..."
export FLASHINFER_DISABLE_JIT=1

echo "Installing flashinfer-python from ${DIST_PYTHON_DIR}..."
pip install ${DIST_PYTHON_DIR}/*.tar.gz

# Verify installation
echo "Verifying installation..."
# Run from /tmp to avoid importing local flashinfer/ source directory
(cd /tmp && python -m flashinfer show-config)

# Run test shard
echo "Running test shard ${TEST_SHARD}..."
export SKIP_INSTALL=1

# Pass through JIT cache report file if set
if [ -n "${FLASHINFER_JIT_CACHE_REPORT_FILE}" ]; then
  export FLASHINFER_JIT_CACHE_REPORT_FILE
fi

bash scripts/task_jit_run_tests_part${TEST_SHARD}.sh
