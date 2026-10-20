#!/bin/bash
# Run a GPU test lane script with its JIT compiles routed through sccache.
#
# Usage: scripts/run_with_test_sccache.sh <test-script> [args...]
#
# The GPU lanes JIT-compile every module from scratch on each run. This wrapper
# reuses the AOT lanes' S3 cache setup (setup_sccache), so a module compiled by
# an earlier run of any lane with the same toolchain is fetched instead of
# rebuilt. It fails open: if sccache cannot be installed, the lane still runs,
# just uncached. Cache read/write errors fall back to local compilation inside
# sccache itself.
#
# Reads (optional): SCCACHE_BUCKET, SCCACHE_REGION, SCCACHE_STATS_DIR,
# AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY (read-write when both are set,
# read-only anonymous access otherwise), FLASHINFER_WORKSPACE_BASE.

set -eo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <test-script> [args...]" >&2
    exit 2
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=scripts/jit_cache_build_common.sh
source "${SCRIPT_DIR}/jit_cache_build_common.sh"

echo "========================================"
echo "Setting up sccache for JIT test compiles"
echo "========================================"

# The subshell keeps a failed download or checksum from aborting the lane;
# setup_sccache then finds the installed binary and skips the download. It must
# not run under `!`, `if`, `&&` or `||`: bash ignores errexit there, including
# inside the subshell, and the installer relies on errexit to stop after a
# failed curl or `sha256sum -c` instead of installing an unverified binary.
if [ "$(sccache --version 2>/dev/null)" != "sccache ${SCCACHE_VERSION}" ]; then
    set +e
    (
        set -e
        install_sccache "${SCCACHE_VERSION}" "$(uname -m)"
    )
    install_status=$?
    set -e
    if [ "${install_status}" -ne 0 ] ||
        [ "$(sccache --version 2>/dev/null)" != "sccache ${SCCACHE_VERSION}" ]; then
        echo "::warning::sccache install failed; running without the compiler cache"
        exec "$@"
    fi
fi

NVCC_BIN="${CUDA_HOME:-/usr/local/cuda}/bin/nvcc"
CUDA_TAG=$(
    "${NVCC_BIN}" --version |
        sed -n 's/.*release \([0-9][0-9]*\)\.\([0-9][0-9]*\).*/\1\2/p' |
        head -n1
) || CUDA_TAG=""
if [ -z "${CUDA_TAG}" ]; then
    echo "::warning::could not read the CUDA version from ${NVCC_BIN}; running without the compiler cache"
    exec "$@"
fi

# JIT sources are generated under the workspace cache, outside the checkout.
# Strip that prefix too so cache keys do not depend on the container's HOME.
JIT_CACHE_ROOT="${FLASHINFER_WORKSPACE_BASE:-${HOME}}/.cache/flashinfer"
mkdir -p "${JIT_CACHE_ROOT}"
JIT_CACHE_ROOT=$(cd "${JIT_CACHE_ROOT}" && pwd -P)
export SCCACHE_BASEDIRS="${JIT_CACHE_ROOT}${SCCACHE_BASEDIRS:+:${SCCACHE_BASEDIRS}}"

export SCCACHE_BUCKET="${SCCACHE_BUCKET:-flashinfer-build-cache}"
# Same prefix as the AOT lanes, which build in this toolchain image.
if ! setup_sccache "cuda${CUDA_TAG}-$(uname -m)" "$(pwd -P)"; then
    # sccache checks the S3 backend at startup; an outage or denied access
    # must not fail the lane.
    echo "::warning::sccache startup failed; running without the compiler cache"
    unset FLASHINFER_CXX_LAUNCHER FLASHINFER_NVCC_LAUNCHER
    exec "$@"
fi

# Not exec: stats must be collected after the lane exits, from this shell.
trap 'collect_sccache_stats || true' EXIT
"$@"
