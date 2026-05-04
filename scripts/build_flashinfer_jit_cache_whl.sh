#!/bin/bash
set -e

# Script to build flashinfer-jit-cache wheel
# This script should be run inside the flashinfer container

echo "=========================================="
echo "Building flashinfer-jit-cache wheel"
echo "=========================================="

# Parallelism: default back to one nvcc worker per compile and let ninja provide
# build-level parallelism through MAX_JOBS.
MEM_AVAILABLE_GB=$(free -g | awk '/^Mem:/ {print $7}')
NPROC=$(nproc)

NVCC_THREADS=${FLASHINFER_NVCC_THREADS:-1}
if ! [[ "$NVCC_THREADS" =~ ^[1-9][0-9]*$ ]]; then
  echo "Invalid FLASHINFER_NVCC_THREADS=${NVCC_THREADS}; using 1"
  NVCC_THREADS=1
fi
if (( NVCC_THREADS > 8 )); then NVCC_THREADS=8; fi
if (( NVCC_THREADS > NPROC )); then NVCC_THREADS=${NPROC}; fi
if (( NVCC_THREADS < 1 )); then NVCC_THREADS=1; fi

# Default to the larger of the historical 8GB/job baseline and ~2GB per nvcc
# thread when callers explicitly opt into higher nvcc threading.
ARCH_MEMORY_BUDGET_GB=8
THREAD_MEMORY_BUDGET_GB=$(( NVCC_THREADS * 2 ))
if (( THREAD_MEMORY_BUDGET_GB > ARCH_MEMORY_BUDGET_GB )); then
  MEM_PER_JOB=${THREAD_MEMORY_BUDGET_GB}
else
  MEM_PER_JOB=${ARCH_MEMORY_BUDGET_GB}
fi
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

# Display build environment info
echo "CUDA Version: ${CUDA_VERSION}"
echo "CPU Architecture: ${ARCH}"
echo "CUDA Major: ${CUDA_MAJOR}"
echo "CUDA Minor: ${CUDA_MINOR}"
echo "FlashInfer Local Version: ${FLASHINFER_LOCAL_VERSION}"
echo "CUDA Architectures: ${FLASHINFER_CUDA_ARCH_LIST}"
echo "Dev Release Suffix: ${FLASHINFER_DEV_RELEASE_SUFFIX}"
echo "MAX_JOBS: ${MAX_JOBS}"
echo "NVCC_THREADS: ${FLASHINFER_NVCC_THREADS}"
echo "Memory Budget per Job: ${MEM_PER_JOB} GB"
echo "Python Version: $(python3 --version)"
echo "Git commit: $(git rev-parse HEAD 2>/dev/null || echo 'unknown')"
echo "Working directory: $(pwd)"
echo ""

# Navigate to the flashinfer-jit-cache directory
cd flashinfer-jit-cache

export CONDA_pkgs_dirs="${FLASHINFER_CI_CACHE}/conda-pkgs"
export XDG_CACHE_HOME="${FLASHINFER_CI_CACHE}/xdg-cache"
mkdir -p "$CONDA_pkgs_dirs" "$XDG_CACHE_HOME"
export HOME=/tmp/home
mkdir -p $HOME
export PATH="$HOME/.local/bin:$PATH"
export PATH="/opt/python/cp312-cp312/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda/lib64/stubs:$LD_LIBRARY_PATH"

echo "::group::Install build system"
pip install --upgrade build
echo "::endgroup::"

install_sccache() {
  local sccache_version=$1
  local sccache_arch=$2
  local sccache_package="sccache-v${sccache_version}-${sccache_arch}-unknown-linux-musl"
  local sccache_archive="${sccache_package}.tar.gz"
  local sccache_url="https://github.com/mozilla/sccache/releases/download/v${sccache_version}/${sccache_archive}"
  local sccache_tmpdir
  local sccache_sha256

  if ! command -v sha256sum >/dev/null 2>&1; then
    echo "ERROR: sha256sum is required to verify sccache downloads"
    exit 1
  fi

  sccache_tmpdir=$(mktemp -d)
  curl -fsSL "${sccache_url}" -o "${sccache_tmpdir}/${sccache_archive}"
  curl -fsSL "${sccache_url}.sha256" -o "${sccache_tmpdir}/${sccache_archive}.sha256"
  sccache_sha256=$(awk '{print $1}' "${sccache_tmpdir}/${sccache_archive}.sha256")
  if [ -z "${sccache_sha256}" ]; then
    echo "ERROR: Missing checksum for ${sccache_archive}"
    exit 1
  fi

  printf '%s  %s\n' "${sccache_sha256}" "${sccache_tmpdir}/${sccache_archive}" | sha256sum -c -
  tar xzf "${sccache_tmpdir}/${sccache_archive}" -C "${sccache_tmpdir}"
  mv "${sccache_tmpdir}/${sccache_package}/sccache" /usr/local/bin/
  rm -rf "${sccache_tmpdir}"
  chmod +x /usr/local/bin/sccache
}

# Optional: set up sccache for compiler caching with S3 backend
if [ -n "$SCCACHE_BUCKET" ]; then
  echo "::group::Install sccache"
  SCCACHE_VERSION="0.9.1"
  SCCACHE_ARCH=$(uname -m)
  install_sccache "${SCCACHE_VERSION}" "${SCCACHE_ARCH}"

  # Namespace cache by CUDA version and CPU architecture
  export SCCACHE_BUCKET
  export SCCACHE_REGION="${SCCACHE_REGION:-us-west-2}"
  SCCACHE_SOURCE_ROOT=$(cd .. && pwd -P)
  export SCCACHE_BASEDIRS="${SCCACHE_SOURCE_ROOT}${SCCACHE_BASEDIRS:+:${SCCACHE_BASEDIRS}}"
  export SCCACHE_S3_KEY_PREFIX="cuda${CUDA_MAJOR}${CUDA_MINOR}-${SCCACHE_ARCH}"
  export SCCACHE_IDLE_TIMEOUT=0
  export FLASHINFER_NVCC_LAUNCHER="sccache"
  export FLASHINFER_CXX_LAUNCHER="sccache"

  if [ -z "${AWS_ACCESS_KEY_ID:-}" ] || [ -z "${AWS_SECRET_ACCESS_KEY:-}" ]; then
    export SCCACHE_S3_NO_CREDENTIALS=true
    echo "sccache mode: read-only (public bucket, no credentials)"
  else
    unset SCCACHE_S3_NO_CREDENTIALS
    echo "sccache mode: read-write"
  fi

  sccache --start-server
  echo "sccache version: $(sccache --version)"
  echo "sccache bucket: ${SCCACHE_BUCKET}"
  echo "sccache region: ${SCCACHE_REGION}"
  echo "sccache prefix: ${SCCACHE_S3_KEY_PREFIX}"
  echo "sccache basedirs: ${SCCACHE_BASEDIRS}"
  echo "::endgroup::"
fi

# Clean any previous builds
echo "Cleaning previous builds..."
rm -rf dist build *.egg-info

# Build the wheel using the build module for better isolation
echo "Building wheel..."
python -m build --wheel

echo ""
echo "✓ Build completed successfully"
echo ""
echo "Built wheels:"
ls -lh dist/

# Verify version and git version
echo ""
echo "Verifying version and git version..."
pip install dist/*.whl
python -c "
import flashinfer_jit_cache
print(f'📦 Package version: {flashinfer_jit_cache.__version__}')
print(f'🔖 Git version: {flashinfer_jit_cache.__git_version__}')
"

# Copy wheels to output directory if specified
if [ -n "${OUTPUT_DIR}" ]; then
    echo ""
    echo "Copying wheels to output directory: ${OUTPUT_DIR}"
    mkdir -p "${OUTPUT_DIR}"
    cp -v dist/*.whl "${OUTPUT_DIR}/"
fi

# Print sccache stats if enabled
if [ -n "$SCCACHE_BUCKET" ]; then
  echo "::group::sccache stats"
  sccache --show-stats
  echo "::endgroup::"
fi

echo ""
echo "Build process completed!"
