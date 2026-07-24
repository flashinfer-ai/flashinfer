#!/bin/bash
# Setup test environment with optional package overrides
# This script should be sourced at the beginning of CI test scripts.
#
# It reads ci/setup_python.env and installs any overridden package versions.
# This is useful for testing specific commits of dependencies (e.g., TVM-FFI)
# before they are officially released.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Pin the preinstalled CUDA torch for every job-time pip install (same guard as
# test_utils.sh; idempotent — whichever is sourced first wins). Prevents a dep's
# transitive constraints from making pip re-resolve torch and silently evict the
# CUDA build (on aarch64 pip backtracks to the CPU-only PyPI wheel -> "Torch not
# compiled with CUDA enabled"); with the constraint such a resolution fails
# loudly at install time. The +cuXXX local tag is stripped: PEP 440 lets the
# installed 2.X.Y+cuNNN satisfy ==2.X.Y, but PEP-517 build envs (flashinfer-
# jit-cache's build-system.requires includes torch) inherit PIP_CONSTRAINT and
# must be able to resolve the pin from PyPI, where local-version wheels don't
# exist.
if [ -z "${PIP_CONSTRAINT:-}" ]; then
  _torch_pin=$(python -c "import torch; print('torch=='+torch.__version__.split('+')[0])" 2>/dev/null || true)
  if [ -n "${_torch_pin}" ]; then
    _constraint_file=$(mktemp /tmp/ci-torch-constraint.XXXXXX.txt)
    echo "${_torch_pin}" > "${_constraint_file}"
    export PIP_CONSTRAINT="${_constraint_file}"
    echo "Pinning for all pip installs in this job: ${_torch_pin}"
    unset _constraint_file
  fi
  unset _torch_pin
fi

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
  CUDA_MAJOR=$(python -c "import torch; print(torch.version.cuda.split('.')[0])" 2>/dev/null || echo "12")
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

# Pre-install the CUDA-13 nvidia-cutlass-dsl backend ([cu13] extra, pinned from requirements.txt)
# to prevent editable installs from leaving mismatched backend versions on CUDA 13 (sm_110a).
# No-op on CUDA 12 or when CUTLASS_DSL_VERSION is set.
if [ -z "${CUTLASS_DSL_VERSION:-}" ]; then
  CUDA_MAJOR=$(python -c "import torch; print(torch.version.cuda.split('.')[0])" 2>/dev/null || echo "12")
  if [ "$CUDA_MAJOR" = "13" ]; then
    # Lift the pinned spec from requirements.txt and add the [cu13] extra, e.g.
    # `nvidia-cutlass-dsl>=4.5.0` -> `nvidia-cutlass-dsl[cu13]>=4.5.0`.
    DSL_SPEC=$(grep -E '^nvidia-cutlass-dsl([<>=!~]|$)' "${REPO_ROOT}/requirements.txt" | head -1 | tr -d '[:space:]')
    if [ -n "$DSL_SPEC" ]; then
      DSL_CU13="${DSL_SPEC/nvidia-cutlass-dsl/nvidia-cutlass-dsl[cu13]}"
      echo "========================================"
      echo "Ensuring CUDA-13 cutlass-dsl backend: ${DSL_CU13}"
      echo "========================================"
      pip install -U "${DSL_CU13}"
      echo ""
    fi
  fi
fi
