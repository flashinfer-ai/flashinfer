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
  if ! _cuda_stack_pins=$(python - <<'PY'
import importlib.metadata as metadata
import torch

print("torch==" + torch.__version__.split("+")[0])
for package in ("cuda-python", "nvidia-cudnn-cu12", "nvidia-cudnn-cu13"):
    try:
        print(f"{package}=={metadata.version(package)}")
    except metadata.PackageNotFoundError:
        pass
PY
  ); then
    echo "ERROR: failed to inspect the image CUDA stack; refusing unpinned pip installs" >&2
    return 1
  fi
  if [ -n "${_cuda_stack_pins}" ]; then
    _constraint_file=$(mktemp /tmp/ci-cuda-stack-constraint.XXXXXX.txt)
    printf '%s\n' "${_cuda_stack_pins}" > "${_constraint_file}"
    export PIP_CONSTRAINT="${_constraint_file}"
    echo "Pinning the image CUDA stack for job-time pip installs:"
    printf '%s\n' "${_cuda_stack_pins}"
    unset _constraint_file
  else
    echo "ERROR: image CUDA stack inspection returned no package constraints" >&2
    return 1
  fi
  unset _cuda_stack_pins
fi

# Install only what this branch moved past the image. Installing the full file
# instead lets one drifted floor re-resolve every other requirement, which is how
# torch has twice been swapped for a build that does not match the image.
_reqs_output=$(python "${SCRIPT_DIR}/check_requirements.py" \
  "${REPO_ROOT}/requirements.txt") && _reqs_status=0 || _reqs_status=$?
case "${_reqs_status}" in
  0)
    echo "Requirements are satisfied by the image; nothing to install."
    ;;
  1)
    echo "Installing requirements this branch changed: ${_reqs_output//$'\n'/ }"
    _reqs_list=()
    while IFS= read -r _req; do
      [ -n "${_req}" ] && _reqs_list+=("${_req}")
    done <<< "${_reqs_output}"
    pip install "${_reqs_list[@]}"
    unset _reqs_list _req
    ;;
  *)
    echo "WARNING: requirement check failed; syncing the full requirements" >&2
    pip install -r "${REPO_ROOT}/requirements.txt"
    ;;
esac
unset _reqs_output _reqs_status

# Install using only what the image carries; the check above covered the rest.
# FLASHINFER_BUILD_NO_PIP keeps dropping isolation from activating the build
# hooks' own downloads, which isolation has always swallowed here.
install_flashinfer_editable() {
  FLASHINFER_BUILD_NO_PIP=1 \
    pip install --no-build-isolation --no-deps -e "${1:-.}" -v
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

# Install quack-kernels for the VSA blk128 backend tests.
# quack-kernels is NOT a runtime requirement of flashinfer — only users of the
# blk128 VSA backend need it, so it is intentionally kept out of requirements.txt
# and installed here for CI only. The blk128 backend supports SM100/SM103, so we
# install quack-kernels only when such a GPU is present to avoid slowing unrelated CI jobs.
# The correct PyPI distribution name is quack-kernels (top-level package: quack).
# --no-deps: quack-kernels 0.6.4 hard-pins nvidia-cutlass-dsl==4.6.2, which would
# downgrade the DSL this job just pinned and leave libs-cu13 skewed (#4555). Its
# remaining deps are either already installed (torch, apache-tvm-ffi, einops) or
# listed alongside it here.
#
# Match on the full capability, not just the major version: SM107 (Rubin) is
# also major 10, but the blk128 backend is SM100/SM103 only (see
# flashinfer/cute_dsl/sparse/sm100_blk128/, and the (10,0)/(10,3) allowlist in
# tests/attention/test_vsa_block_sparse.py), so those tests skip on Rubin and
# the install is pure cost there.  --no-deps already prevents the DSL downgrade;
# skipping the install altogether also avoids perturbing a Rubin environment
# that nothing in this package needs.
SM_CAP=$(python -c "import torch; print('%d.%d' % torch.cuda.get_device_capability())" 2>/dev/null || echo "")
if [ "${SM_CAP}" = "10.0" ] || [ "${SM_CAP}" = "10.3" ]; then
  echo "========================================"
  echo "Detected SM${SM_CAP} (SM100/SM103); installing quack-kernels for VSA blk128 tests"
  echo "========================================"
  DSL_VERSION_BEFORE=$(python -c "import importlib.metadata as m; print(m.version('nvidia-cutlass-dsl'))" 2>/dev/null || echo "")
  pip install --no-deps "quack-kernels==0.6.4" "torch-c-dlpack-ext==0.1.5"
  DSL_VERSION_AFTER=$(python -c "import importlib.metadata as m; print(m.version('nvidia-cutlass-dsl'))" 2>/dev/null || echo "")
  if [ "${DSL_VERSION_BEFORE}" != "${DSL_VERSION_AFTER}" ]; then
    echo "ERROR: quack-kernels install moved nvidia-cutlass-dsl from ${DSL_VERSION_BEFORE} to ${DSL_VERSION_AFTER}" >&2
    return 1 2>/dev/null || exit 1
  fi
  # Fail here rather than as a confusing test error if --no-deps left a gap.
  python -c "import quack"
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
