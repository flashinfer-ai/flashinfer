#!/bin/bash
set -e

# Build one provider wheel or a provider shim wheel.
# This script should be run inside the flashinfer container

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=scripts/jit_cache_build_common.sh
source "${SCRIPT_DIR}/jit_cache_build_common.sh"

finish_sccache_stats() {
  local exit_code=$?
  cleanup_jit_cache_python_build || true
  collect_sccache_stats || true
  return "${exit_code}"
}

trap finish_sccache_stats EXIT

PYTHON_VERSION_FILE="${SCRIPT_DIR}/../.python-version"
PYTHON_VERSION="$(tr -d '[:space:]' < "${PYTHON_VERSION_FILE}")"
if [[ ! "${PYTHON_VERSION}" =~ ^3\.[0-9]+$ ]]; then
  echo "Invalid Python version in ${PYTHON_VERSION_FILE}: ${PYTHON_VERSION}" >&2
  exit 2
fi
PYTHON_ABI="cp${PYTHON_VERSION//./}"

BUILD_TARGET=${FLASHINFER_JIT_CACHE_BUILD_TARGET:-}
case "${BUILD_TARGET}" in
  provider)
    : "${FLASHINFER_JIT_CACHE_PROVIDER_ARCH:?provider builds require FLASHINFER_JIT_CACHE_PROVIDER_ARCH}"
    PACKAGE_DIR=flashinfer-jit-cache-provider
    export FLASHINFER_JIT_CACHE_PROVIDER_PLATFORM_TAG="${FLASHINFER_JIT_CACHE_PROVIDER_PLATFORM_TAG:-manylinux_2_28_${ARCH}}"
    ;;
  shim)
    : "${FLASHINFER_JIT_CACHE_PROVIDER_ARCHS:?shim builds require FLASHINFER_JIT_CACHE_PROVIDER_ARCHS}"
    PACKAGE_DIR=flashinfer-jit-cache
    ;;
  *)
    echo "Invalid FLASHINFER_JIT_CACHE_BUILD_TARGET=${BUILD_TARGET}; expected provider or shim" >&2
    exit 2
    ;;
esac

echo "=========================================="
echo "Building flashinfer-jit-cache ${BUILD_TARGET} wheel"
echo "=========================================="

: "${PYTORCH_INDEX:?PYTORCH_INDEX must be set}"

compute_jit_cache_parallelism

# Display build environment info
echo "CUDA Version: ${CUDA_VERSION}"
echo "CPU Architecture: ${ARCH}"
echo "CUDA Major: ${CUDA_MAJOR}"
echo "CUDA Minor: ${CUDA_MINOR}"
echo "PyTorch Index: ${PYTORCH_INDEX}"
echo "FlashInfer Local Version: ${FLASHINFER_LOCAL_VERSION}"
echo "Provider Architecture: ${FLASHINFER_JIT_CACHE_PROVIDER_ARCH:-}"
echo "Shim Provider Architectures: ${FLASHINFER_JIT_CACHE_PROVIDER_ARCHS:-}"
echo "Dev Release Suffix: ${FLASHINFER_DEV_RELEASE_SUFFIX:-}"
echo "MAX_JOBS: ${MAX_JOBS}"
echo "NVCC_THREADS: ${FLASHINFER_NVCC_THREADS}"
echo "Memory Budget per Job: ${MEM_PER_JOB} GB"
echo "Python Version: $(python3 --version)"
echo "Git commit: $(git rev-parse HEAD 2>/dev/null || echo 'unknown')"
echo "Working directory: $(pwd)"
echo ""

# Navigate to the selected package directory.
cd "${PACKAGE_DIR}"

export CONDA_pkgs_dirs="${FLASHINFER_CI_CACHE}/conda-pkgs"
export XDG_CACHE_HOME="${FLASHINFER_CI_CACHE}/xdg-cache"
mkdir -p "$CONDA_pkgs_dirs" "$XDG_CACHE_HOME"
export HOME=/tmp/home
mkdir -p $HOME
export PATH="$HOME/.local/bin:$PATH"
export PATH="/opt/python/${PYTHON_ABI}-${PYTHON_ABI}/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda/lib64/stubs:$LD_LIBRARY_PATH"

EXPECTED_CUDA_VERSION="${CUDA_MAJOR}.${CUDA_MINOR}"
validate_jit_cache_cuda_toolchain "${EXPECTED_CUDA_VERSION}"

echo "::group::Install build system"
setup_jit_cache_python_build python3 "${EXPECTED_CUDA_VERSION}" "${PYTORCH_INDEX}"
echo "::endgroup::"

# Optional: set up sccache for compiler caching with S3 backend
if [ -n "$SCCACHE_BUCKET" ]; then
  export SCCACHE_BUCKET
  setup_sccache "cuda${CUDA_MAJOR}${CUDA_MINOR}-$(uname -m)" "$(cd .. && pwd -P)"
fi

# Clean any previous builds
echo "Cleaning previous builds..."
rm -rf -- dist build ./*.egg-info
if [ "${BUILD_TARGET}" = "provider" ]; then
  rm -rf -- flashinfer_jit_cache_provider/jit_cache
  rm -f -- \
    flashinfer_jit_cache_provider/manifest.json \
    flashinfer_jit_cache_provider/_build_meta.py
elif [ "${BUILD_TARGET}" = "shim" ]; then
  rm -rf -- flashinfer_jit_cache/jit_cache
  rm -f -- \
    flashinfer_jit_cache/_build_meta.py \
    flashinfer_jit_cache/_provider_requirements.txt
fi

# Build the wheel using the build module for better isolation
echo "Building wheel..."
python -m build --wheel

echo ""
echo "✓ Build completed successfully"
echo ""
echo "Built wheels:"
ls -lh dist/

if [ "${BUILD_TARGET}" = "provider" ]; then
  PROVIDER_TAG=${FLASHINFER_JIT_CACHE_PROVIDER_ARCH,,}
  PROVIDER_TAG=${PROVIDER_TAG#compute_}
  PROVIDER_TAG=${PROVIDER_TAG#sm_}
  PROVIDER_TAG=${PROVIDER_TAG#sm}
  PROVIDER_TAG=${PROVIDER_TAG//./}
  PROVIDER_TAG=${PROVIDER_TAG//_/}
  PROVIDER_TAG="sm${PROVIDER_TAG}"
  PACKAGE_VERSION=$(tr -d '[:space:]' < ../version.txt)
  if [ -n "${FLASHINFER_DEV_RELEASE_SUFFIX:-}" ]; then
    PACKAGE_VERSION="${PACKAGE_VERSION}.dev${FLASHINFER_DEV_RELEASE_SUFFIX}"
  fi
  PACKAGE_VERSION="${PACKAGE_VERSION}+${FLASHINFER_LOCAL_VERSION}"

  python ../scripts/verify_jit_cache_provider_artifact.py \
    --artifact-dir dist \
    --provider "${PROVIDER_TAG}" \
    --version "${PACKAGE_VERSION}" \
    --provider-platform-tag "${FLASHINFER_JIT_CACHE_PROVIDER_PLATFORM_TAG}" \
    --cuobjdump /usr/local/cuda/bin/cuobjdump \
    --cuda-architecture-policy "${CUDA_ARCHITECTURE_POLICY:-strict}"
fi

# Verify version and git version
echo ""
echo "Verifying built package metadata..."
pip install --force-reinstall --no-deps dist/*.whl
if [ "${BUILD_TARGET}" = "provider" ]; then
  python - "${FLASHINFER_JIT_CACHE_PROVIDER_ARCH}" <<'PY'
import importlib.metadata
import re
import sys

provider = re.sub(r"[-_.]+", "", sys.argv[1].lower())
provider = provider.removeprefix("compute").removeprefix("sm")
distribution = f"flashinfer-jit-cache-sm{provider}"
metadata = importlib.metadata.metadata(distribution)
entry_points = [
    entry_point
    for entry_point in importlib.metadata.entry_points().select(
        group="flashinfer.jit_cache.providers"
    )
    if entry_point.name == f"sm{provider}"
]
if len(entry_points) != 1:
    raise SystemExit(f"Expected one provider entry point, found {entry_points}")
print(f"Package version: {metadata['Version']}")
print(f"Provider entry point: {entry_points[0].value}")
PY
else
  python -c "
import flashinfer_jit_cache
print(f'Package version: {flashinfer_jit_cache.__version__}')
print(f'Git version: {flashinfer_jit_cache.__git_version__}')
"
fi

# Copy wheels to output directory if specified
if [ -n "${OUTPUT_DIR}" ]; then
    echo ""
    echo "Copying wheels to output directory: ${OUTPUT_DIR}"
    mkdir -p "${OUTPUT_DIR}"
    cp -v dist/*.whl "${OUTPUT_DIR}/"
fi

echo ""
echo "Build process completed!"
