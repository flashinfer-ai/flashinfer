#!/bin/bash

set -eo pipefail
set -x

NIGHTLY_CUTLASS_DSL_VERSION="4.6.2"

# Source test environment setup (handles package overrides like TVM-FFI)
source "$(dirname "${BASH_SOURCE[0]}")/setup_test_env.sh"

# This script installs nightly build packages and runs tests
# Expected dist directories to be in current directory or specified via env vars

: ${TEST_SHARD:=1}
: ${CUDA_VISIBLE_DEVICES:=0}
: ${DIST_CUBIN_DIR:=dist-cubin}
: ${DIST_JIT_CACHE_DIR:=dist-jit-cache}
: ${DIST_PYTHON_DIR:=dist-python}

SOURCE_WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "${SOURCE_WORKSPACE}"
TEST_RUN_DIR="$(mktemp -d /tmp/flashinfer-nightly-tests.XXXXXX)"
trap 'rm -rf "${TEST_RUN_DIR}"' EXIT

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

# Nightly artifacts must be tested with their published 4.6 dependency even
# though the reusable CI image starts on 4.7. Reinstall the complete stack so
# this job represents a normal package consumer rather than the PrimTS CI
# source-test override.
CUDA_MAJOR=$(python -c "import torch; print(torch.version.cuda.split('.')[0])")
pip uninstall -y \
  nvidia-cutlass-dsl \
  nvidia-cutlass-dsl-libs-core \
  nvidia-cutlass-dsl-libs-base \
  nvidia-cutlass-dsl-libs-cu12 \
  nvidia-cutlass-dsl-libs-cu13 2>/dev/null || true
if [ "$CUDA_MAJOR" = "13" ]; then
  CUTLASS_DSL_PACKAGE="nvidia-cutlass-dsl[cu13]==${NIGHTLY_CUTLASS_DSL_VERSION}"
  CUTLASS_DSL_PACKAGES="nvidia-cutlass-dsl nvidia-cutlass-dsl-libs-core nvidia-cutlass-dsl-libs-base nvidia-cutlass-dsl-libs-cu12 nvidia-cutlass-dsl-libs-cu13"
else
  CUTLASS_DSL_PACKAGE="nvidia-cutlass-dsl==${NIGHTLY_CUTLASS_DSL_VERSION}"
  CUTLASS_DSL_PACKAGES="nvidia-cutlass-dsl nvidia-cutlass-dsl-libs-core nvidia-cutlass-dsl-libs-base nvidia-cutlass-dsl-libs-cu12"
fi
pip install --upgrade "$CUTLASS_DSL_PACKAGE"
python -c "
import importlib.metadata as m, sys
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

expected = '${NIGHTLY_CUTLASS_DSL_VERSION}'
for name in '${CUTLASS_DSL_PACKAGES}'.split():
    version = m.version(name)
    if version != expected:
        sys.exit(f'ERROR: {name} is {version}, expected {expected}')
requirements = [Requirement(raw) for raw in (m.distribution('flashinfer-python').requires or [])]
dsl_requirements = [
    requirement
    for requirement in requirements
    if canonicalize_name(requirement.name) == 'nvidia-cutlass-dsl'
]
if not dsl_requirements or any(str(requirement.specifier) != f'=={expected}' for requirement in dsl_requirements):
    sys.exit(f'ERROR: flashinfer-python CUTLASS DSL requirements are {dsl_requirements}, expected =={expected}')
print(f'Nightly CUTLASS DSL package check passed: {expected}')
"
pip check

# Verify installation
echo "Verifying installation..."
# Run from /tmp to avoid importing local flashinfer/ source directory
(cd /tmp && python -m flashinfer show-config)

# Copy only test sources into an isolated directory so package tests exercise the
# installed flashinfer distribution instead of shadowing it with /workspace.
cp -a "${SOURCE_WORKSPACE}/tests" "${TEST_RUN_DIR}/"
cp -a "${SOURCE_WORKSPACE}/pytest.ini" "${TEST_RUN_DIR}/"

# Run test shard
echo "Running test shard ${TEST_SHARD}..."
export SKIP_INSTALL=1

# Pass through JIT cache report file if set
if [ -n "${FLASHINFER_JIT_CACHE_REPORT_FILE}" ]; then
  export FLASHINFER_JIT_CACHE_REPORT_FILE
fi

(cd "${TEST_RUN_DIR}" && bash "${SOURCE_WORKSPACE}/scripts/task_jit_run_tests_part${TEST_SHARD}.sh")
