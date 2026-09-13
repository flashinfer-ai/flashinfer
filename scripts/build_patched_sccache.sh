#!/bin/bash
set -euo pipefail

# Build the pinned CUDA 13.4-compatible sccache binary in a dedicated GitHub
# Actions step. The wheel build consumes the binary from the shared CI cache.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=scripts/jit_cache_build_common.sh
source "${SCRIPT_DIR}/jit_cache_build_common.sh"

case "${CUDA_VERSION:-}" in
  13.4|134)
    ;;
  *)
    echo "ERROR: Patched sccache is only expected for CUDA 13.4, got ${CUDA_VERSION:-unset}"
    exit 1
    ;;
esac

if [ -z "${SCCACHE_PATCHED_BINARY_PATH:-}" ]; then
  echo "ERROR: SCCACHE_PATCHED_BINARY_PATH must be set"
  exit 1
fi

sccache_build_started_at=${SECONDS}
report_sccache_build_duration() {
  local exit_code=$?
  trap - EXIT
  echo "patched sccache step duration: $((SECONDS - sccache_build_started_at)) seconds"
  exit "${exit_code}"
}
trap report_sccache_build_duration EXIT

build_patched_sccache \
  "${SCCACHE_CUDA_134_REVISION}" \
  "${SCCACHE_CUDA_134_SOURCE_SHA256}" \
  "${SCCACHE_PATCHED_BINARY_PATH}"

"${SCCACHE_PATCHED_BINARY_PATH}" --version
