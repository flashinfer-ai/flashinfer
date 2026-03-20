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

# Override nvidia-cutlass-dsl if specified
if [ -n "${CUTLASS_DSL_VERSION:-}" ]; then
  # Detect CUDA major version to select the correct extra (cu12 or cu13)
  CUDA_MAJOR=$(python -c "import torch; print(torch.version.cuda.split('.')[0])" 2>/dev/null || echo "12")
  if [ "$CUDA_MAJOR" = "13" ]; then
    CUTLASS_DSL_EXTRA="cu13"
  else
    CUTLASS_DSL_EXTRA="cu12"
  fi
  echo "========================================"
  echo "Overriding nvidia-cutlass-dsl with version: ${CUTLASS_DSL_VERSION} [${CUTLASS_DSL_EXTRA}]"
  echo "========================================"
  # Clean uninstall old packages first (recommended by NVIDIA docs)
  pip uninstall nvidia-cutlass-dsl nvidia-cutlass-dsl-libs-base nvidia-cutlass-dsl-libs-cu12 nvidia-cutlass-dsl-libs-cu13 -y 2>/dev/null || true
  pip install "nvidia-cutlass-dsl[${CUTLASS_DSL_EXTRA}]==${CUTLASS_DSL_VERSION}"
  echo "nvidia-cutlass-dsl override complete."
  echo ""
fi
