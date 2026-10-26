#!/bin/bash
# Run one production JIT-cache wheel build in the selected manylinux container.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

: "${ARCH:?ARCH must be set}"
: "${CUDA_VERSION:?CUDA_VERSION must be set}"
: "${DOCKER_IMAGE:?DOCKER_IMAGE must be set}"
: "${FLASHINFER_CI_CACHE:?FLASHINFER_CI_CACHE must be set}"
: "${FLASHINFER_JIT_CACHE_BUILD_TARGET:?FLASHINFER_JIT_CACHE_BUILD_TARGET must be set}"
: "${FLASHINFER_LOCAL_VERSION:?FLASHINFER_LOCAL_VERSION must be set}"
: "${PYTORCH_INDEX:?PYTORCH_INDEX must be set}"

IFS=. read -r CUDA_MAJOR CUDA_MINOR <<< "${CUDA_VERSION}"
export CUDA_MAJOR CUDA_MINOR

mkdir -p "${FLASHINFER_CI_CACHE}" "${REPO_ROOT}/sccache-stats"

# Keep docker run's default cached-image behavior, but retry transient registry
# failures before starting a build. Never retry the build itself.
if ! docker image inspect "${DOCKER_IMAGE}" >/dev/null 2>&1; then
  for attempt in 1 2 3 4; do
    if docker pull "${DOCKER_IMAGE}"; then
      break
    fi
    if (( attempt == 4 )); then
      echo "Failed to pull ${DOCKER_IMAGE} after ${attempt} attempts" >&2
      exit 1
    fi
    delay=$(( 5 * 2 ** (attempt - 1) ))
    echo "Image pull failed; retrying in ${delay}s (attempt $(( attempt + 1 ))/4)" >&2
    sleep "${delay}"
  done
fi

docker run --rm --pull=never \
  -v "${REPO_ROOT}:/workspace" \
  -v "${FLASHINFER_CI_CACHE}:/ci-cache" \
  -e AOT_MAX_JOBS_CAP="${AOT_MAX_JOBS_CAP:-0}" \
  -e AOT_MAX_JOBS_MEMORY_GB="${AOT_MAX_JOBS_MEMORY_GB:-}" \
  -e ARCH \
  -e AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID:-}" \
  -e AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY:-}" \
  -e CUDA_ARCHITECTURE_POLICY="${CUDA_ARCHITECTURE_POLICY:-strict}" \
  -e CUDA_MAJOR \
  -e CUDA_MINOR \
  -e CUDA_VERSION \
  -e FLASHINFER_CI_CACHE=/ci-cache \
  -e FLASHINFER_DEV_RELEASE_SUFFIX="${FLASHINFER_DEV_RELEASE_SUFFIX:-}" \
  -e FLASHINFER_JIT_CACHE_BUILD_TARGET \
  -e FLASHINFER_JIT_CACHE_PROVIDER_ARCH="${FLASHINFER_JIT_CACHE_PROVIDER_ARCH:-}" \
  -e FLASHINFER_JIT_CACHE_PROVIDER_ARCHS="${FLASHINFER_JIT_CACHE_PROVIDER_ARCHS:-}" \
  -e FLASHINFER_JIT_CACHE_PROVIDER_PLATFORM_TAG="${FLASHINFER_JIT_CACHE_PROVIDER_PLATFORM_TAG:-}" \
  -e FLASHINFER_LOCAL_VERSION \
  -e FLASHINFER_NVCC_THREADS="${FLASHINFER_NVCC_THREADS:-}" \
  -e JIT_CACHE_NO_OUTPUT_TIMEOUT_SECONDS="${JIT_CACHE_NO_OUTPUT_TIMEOUT_SECONDS:-5400}" \
  -e JIT_CACHE_WATCHDOG_TERM_GRACE_SECONDS="${JIT_CACHE_WATCHDOG_TERM_GRACE_SECONDS:-120}" \
  -e PYTORCH_INDEX \
  -e SCCACHE_BUCKET="${SCCACHE_BUCKET:-}" \
  -e SCCACHE_REGION="${SCCACHE_REGION:-}" \
  -e SCCACHE_STATS_DIR=/workspace/sccache-stats \
  -w /workspace \
  "${DOCKER_IMAGE}" \
  bash /workspace/scripts/build_flashinfer_jit_cache_whl.sh
