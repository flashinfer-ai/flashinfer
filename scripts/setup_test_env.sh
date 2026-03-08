#!/bin/bash
# Setup test environment with optional package overrides
# This script should be sourced at the beginning of CI test scripts.
#
# It reads ci/setup_python.env and installs any overridden package versions.
# This is useful for testing specific commits of dependencies (e.g., TVM-FFI)
# before they are officially released.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Source the environment override file if it exists
if [ -f "${REPO_ROOT}/ci/setup_python.env" ]; then
  source "${REPO_ROOT}/ci/setup_python.env"
fi

# Override TVM-FFI if specified
if [ -n "${TVM_FFI_REF:-}" ]; then
  echo "========================================"
  echo "Overriding TVM-FFI with ref: ${TVM_FFI_REF}"
  echo "========================================"
  pip install --force-reinstall "git+https://github.com/apache/tvm-ffi.git@${TVM_FFI_REF}"
  echo "TVM-FFI override complete."
  echo ""
fi
