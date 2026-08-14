#!/bin/bash
# Setup test environment with optional package overrides
# This script should be sourced at the beginning of CI test scripts.
#
# It reads ci/setup_python.env and installs any overridden package versions.
# This is useful for testing specific commits of dependencies (e.g., TVM-FFI)
# before they are officially released.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Pin the preinstalled CUDA Python stack for every job-time pip install (same
# guard as test_utils.sh; idempotent — whichever is sourced first wins). This
# prevents a branch dependency sync from replacing torch, cuda-python, or the
# cuDNN backend selected and validated when the image was built. The +cuXXX
# torch local tag is stripped because PEP-517 build environments must be able
# to resolve the constraint from PyPI, where local-version wheels do not exist.
if [ -z "${PIP_CONSTRAINT:-}" ]; then
  _cuda_stack_pins=$(python - <<'PY' 2>/dev/null || true
import importlib.metadata as metadata
import torch

print("torch==" + torch.__version__.split("+")[0])
for package in ("cuda-python", "nvidia-cudnn-cu12", "nvidia-cudnn-cu13"):
    try:
        print(f"{package}=={metadata.version(package)}")
    except metadata.PackageNotFoundError:
        pass
PY
  )
  if [ -n "${_cuda_stack_pins}" ]; then
    _constraint_file=$(mktemp /tmp/ci-cuda-stack-constraint.XXXXXX.txt)
    printf '%s\n' "${_cuda_stack_pins}" > "${_constraint_file}"
    export PIP_CONSTRAINT="${_constraint_file}"
    echo "Pinning the image CUDA stack for job-time pip installs:"
    printf '%s\n' "${_cuda_stack_pins}"
    unset _constraint_file
  fi
  unset _cuda_stack_pins
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

# Install quack-kernels for the VSA blk128 backend tests.
# quack-kernels is NOT a runtime requirement of flashinfer — only users of the
# blk128 VSA backend need it, so it is intentionally kept out of requirements.txt
# and installed here for CI only. The blk128 backend supports SM100/SM103, so we
# install quack-kernels only when such a GPU is present to avoid slowing unrelated CI jobs.
# The correct PyPI distribution name is quack-kernels (top-level package: quack).
SM_MAJOR=$(python -c "import torch; print(torch.cuda.get_device_capability()[0])" 2>/dev/null || echo "")
if [ "${SM_MAJOR}" = "10" ]; then
  echo "========================================"
  echo "Detected SM${SM_MAJOR} (SM100/SM103); installing quack-kernels for VSA blk128 tests"
  echo "========================================"
  pip install "quack-kernels==0.6.4"
  echo "quack-kernels install complete."
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
