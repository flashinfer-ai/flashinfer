#!/bin/bash
# Setup test environment with optional package overrides
# This script should be sourced at the beginning of CI test scripts.
#
# It reads ci/setup_python.env and installs any overridden package versions.
# This is useful for testing specific commits of dependencies (e.g., TVM-FFI)
# before they are officially released.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

detect_cuda_major() {
  local cuda_version="${CUDA_VERSION:-}"
  cuda_version="${cuda_version#nightly/}"
  if [[ "${cuda_version}" == cu* ]]; then
    cuda_version="${cuda_version#cu}"
    echo "${cuda_version:0:2}"
    return
  fi
  if [[ "${cuda_version}" =~ ^([0-9]+)\. ]]; then
    echo "${BASH_REMATCH[1]}"
    return
  fi
  python -c "import torch; print(torch.version.cuda.split('.')[0])" 2>/dev/null || echo "12"
}

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
  # Detect CUDA major version: only CUDA 13+ needs [cu13] extra
  CUDA_MAJOR=$(detect_cuda_major)
  if [ "$CUDA_MAJOR" = "13" ]; then
    CUTLASS_DSL_PKG="nvidia-cutlass-dsl[cu13]==${CUTLASS_DSL_VERSION}"
  else
    CUTLASS_DSL_PKG="nvidia-cutlass-dsl==${CUTLASS_DSL_VERSION}"
  fi
  echo "========================================"
  echo "Overriding nvidia-cutlass-dsl with: ${CUTLASS_DSL_PKG}"
  echo "========================================"
  # Clean uninstall old packages first (recommended by NVIDIA docs)
  pip uninstall nvidia-cutlass-dsl nvidia-cutlass-dsl-libs-base nvidia-cutlass-dsl-libs-cu12 nvidia-cutlass-dsl-libs-cu13 -y 2>/dev/null || true
  pip install "${CUTLASS_DSL_PKG}"
  echo "nvidia-cutlass-dsl override complete."
  echo ""
fi
