#!/bin/bash
# Build a local FlashInfer Python + jit-cache provider + shim wheelhouse.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

normalize_provider_tag() {
    local architecture=${1,,}
    architecture=${architecture#compute_}
    architecture=${architecture#sm_}
    architecture=${architecture#sm}
    architecture=${architecture//./}
    architecture=${architecture//_/}
    if ! [[ "${architecture}" =~ ^[0-9]{2,3}[af]?$ ]]; then
        echo "Invalid provider architecture: $1" >&2
        return 1
    fi
    printf 'sm%s\n' "${architecture}"
}

: "${FLASHINFER_JIT_CACHE_PROVIDER_ARCH:=12.1a}"
: "${FLASHINFER_LOCAL_VERSION:=cu130}"
: "${FLASHINFER_DEV_RELEASE_SUFFIX:=$(date -u +%Y%m%d)}"
: "${CUDA_VERSION:=13.0}"
: "${CUDA_MAJOR:=13}"
: "${CUDA_MINOR:=0}"
: "${ARCH:=aarch64}"
: "${DOCKER_IMAGE:=pytorch/manylinuxaarch64-builder:cuda13.0}"
: "${AOT_MAX_JOBS_CAP:=4}"
: "${AOT_MAX_JOBS_MEMORY_GB:=16}"
: "${FLASHINFER_NVCC_THREADS:=1}"
: "${CLEAN_OUTPUT:=0}"

PROVIDER_TAG=$(normalize_provider_tag "${FLASHINFER_JIT_CACHE_PROVIDER_ARCH}")
PACKAGE_VERSION=$(tr -d '[:space:]' < "${REPO_ROOT}/version.txt")
PACKAGE_VERSION="${PACKAGE_VERSION}.dev${FLASHINFER_DEV_RELEASE_SUFFIX}+${FLASHINFER_LOCAL_VERSION}"

run_on_host() {
    local output_dir cache_dir host_uid host_gid
    output_dir=${OUTPUT_DIR:-"${REPO_ROOT}/dist/jit-cache-wheelhouse-${PROVIDER_TAG}-${FLASHINFER_LOCAL_VERSION}-dev${FLASHINFER_DEV_RELEASE_SUFFIX}"}
    cache_dir=${FLASHINFER_CI_CACHE:-"${XDG_CACHE_HOME:-${HOME}/.cache}/flashinfer-wheelhouse"}

    command -v docker >/dev/null 2>&1 || {
        echo "docker is required" >&2
        exit 1
    }

    mkdir -p "${output_dir}" "${cache_dir}"
    output_dir=$(cd "${output_dir}" && pwd)
    cache_dir=$(cd "${cache_dir}" && pwd)

    if [ "${CLEAN_OUTPUT}" = "1" ]; then
        find "${output_dir}" -mindepth 1 -maxdepth 1 -type f -delete
    elif find "${output_dir}" -mindepth 1 -maxdepth 1 -type f -print -quit | grep -q .; then
        echo "Output directory is not empty: ${output_dir}" >&2
        echo "Set CLEAN_OUTPUT=1 or choose a different OUTPUT_DIR." >&2
        exit 1
    fi

    host_uid=$(id -u)
    host_gid=$(id -g)

    echo "Building ${PACKAGE_VERSION} provider ${PROVIDER_TAG}"
    echo "Container: ${DOCKER_IMAGE}"
    echo "Output: ${output_dir}"

    docker run --rm \
        --user "${host_uid}:${host_gid}" \
        -v "${REPO_ROOT}:/workspace" \
        -v "${output_dir}:/wheelhouse" \
        -v "${cache_dir}:/ci-cache" \
        -e AOT_MAX_JOBS_CAP="${AOT_MAX_JOBS_CAP}" \
        -e AOT_MAX_JOBS_MEMORY_GB="${AOT_MAX_JOBS_MEMORY_GB}" \
        -e ARCH="${ARCH}" \
        -e CUDA_MAJOR="${CUDA_MAJOR}" \
        -e CUDA_MINOR="${CUDA_MINOR}" \
        -e CUDA_VERSION="${CUDA_VERSION}" \
        -e FLASHINFER_DEV_RELEASE_SUFFIX="${FLASHINFER_DEV_RELEASE_SUFFIX}" \
        -e FLASHINFER_JIT_CACHE_PROVIDER_ARCH="${FLASHINFER_JIT_CACHE_PROVIDER_ARCH}" \
        -e FLASHINFER_LOCAL_VERSION="${FLASHINFER_LOCAL_VERSION}" \
        -e FLASHINFER_NVCC_THREADS="${FLASHINFER_NVCC_THREADS}" \
        -e HOST_GID="${host_gid}" \
        -e HOST_UID="${host_uid}" \
        -e OUTPUT_DIR=/wheelhouse \
        -e PYTHON_BIN="${PYTHON_BIN:-/opt/python/cp312-cp312/bin/python}" \
        -e FLASHINFER_CI_CACHE=/ci-cache \
        -w /workspace \
        "${DOCKER_IMAGE}" \
        bash /workspace/scripts/build_jit_cache_provider_wheelhouse.sh --inside-container
}

run_in_container() {
    local python_bin build_venv python output_dir
    python_bin=${PYTHON_BIN:-/opt/python/cp312-cp312/bin/python}
    build_venv=${BUILD_VENV:-/tmp/flashinfer-wheelhouse-venv}
    output_dir=${OUTPUT_DIR:-/wheelhouse}

    if [ "$(uname -m)" != "${ARCH}" ]; then
        echo "Container architecture $(uname -m) does not match ARCH=${ARCH}" >&2
        exit 1
    fi
    if [ ! -x "${python_bin}" ]; then
        echo "Python interpreter not found: ${python_bin}" >&2
        exit 1
    fi
    if [ ! -x /usr/local/cuda/bin/nvcc ]; then
        echo "CUDA compiler not found at /usr/local/cuda/bin/nvcc" >&2
        exit 1
    fi

    mkdir -p "${output_dir}" /tmp/flashinfer-home
    export HOME=/tmp/flashinfer-home
    export CONDA_pkgs_dirs="${FLASHINFER_CI_CACHE}/conda-pkgs"
    export XDG_CACHE_HOME="${FLASHINFER_CI_CACHE}/xdg-cache"
    export PIP_CACHE_DIR="${FLASHINFER_CI_CACHE}/pip-cache"
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH:-}"
    mkdir -p "${CONDA_pkgs_dirs}" "${XDG_CACHE_HOME}" "${PIP_CACHE_DIR}"

    "${python_bin}" -m venv "${build_venv}"
    python="${build_venv}/bin/python"
    "${python}" -m pip install --disable-pip-version-check --upgrade build

    # shellcheck source=scripts/jit_cache_build_common.sh
    source "${SCRIPT_DIR}/jit_cache_build_common.sh"
    compute_jit_cache_parallelism

    export BUILD_NVEP=0
    export FLASHINFER_DISABLE_VERSION_CHECK=1
    export FLASHINFER_CUDA_ARCH_LIST="${FLASHINFER_JIT_CACHE_PROVIDER_ARCH}"

    echo "=========================================="
    echo "Building provider wheelhouse"
    echo "=========================================="
    echo "Version: ${PACKAGE_VERSION}"
    echo "Provider: ${PROVIDER_TAG} (${FLASHINFER_CUDA_ARCH_LIST})"
    echo "CUDA: $(/usr/local/cuda/bin/nvcc --version | tail -n 1)"
    echo "Architecture: $(uname -m)"
    echo "MAX_JOBS: ${MAX_JOBS}"
    echo "NVCC_THREADS: ${FLASHINFER_NVCC_THREADS}"
    echo "Memory budget per job: ${MEM_PER_JOB} GB"

    rm -rf \
        "${REPO_ROOT}/flashinfer-jit-cache-provider/build" \
        "${REPO_ROOT}/flashinfer-jit-cache-provider/dist" \
        "${REPO_ROOT}/flashinfer-jit-cache-provider/flashinfer_jit_cache_provider/jit_cache" \
        "${REPO_ROOT}/flashinfer-jit-cache/build" \
        "${REPO_ROOT}/flashinfer-jit-cache/dist" \
        "${REPO_ROOT}/flashinfer-jit-cache/flashinfer_jit_cache/jit_cache"
    rm -f \
        "${REPO_ROOT}/flashinfer-jit-cache-provider/flashinfer_jit_cache_provider/manifest.json" \
        "${REPO_ROOT}/flashinfer-jit-cache-provider/flashinfer_jit_cache_provider/_build_meta.py" \
        "${REPO_ROOT}/flashinfer-jit-cache/flashinfer_jit_cache/_build_meta.py" \
        "${REPO_ROOT}/flashinfer-jit-cache/flashinfer_jit_cache/_provider_requirements.txt"
    find "${REPO_ROOT}/flashinfer-jit-cache-provider" \
        "${REPO_ROOT}/flashinfer-jit-cache" \
        -maxdepth 1 -type d -name '*.egg-info' -exec rm -rf {} +

    echo "Building flashinfer-python..."
    "${python}" -m build --wheel --outdir "${output_dir}" "${REPO_ROOT}"

    echo "Building ${PROVIDER_TAG} provider..."
    FLASHINFER_JIT_CACHE_PROVIDER_ARCH="${FLASHINFER_JIT_CACHE_PROVIDER_ARCH}" \
        "${python}" -m build --wheel --outdir "${output_dir}" \
        "${REPO_ROOT}/flashinfer-jit-cache-provider"

    echo "Building one-provider shim..."
    FLASHINFER_JIT_CACHE_WHEEL_KIND=shim \
    FLASHINFER_JIT_CACHE_PROVIDER_ARCHS="${FLASHINFER_JIT_CACHE_PROVIDER_ARCH}" \
        "${python}" -m build --wheel --outdir "${output_dir}" \
        "${REPO_ROOT}/flashinfer-jit-cache"

    echo "Validating wheelhouse..."
    "${python}" "${SCRIPT_DIR}/verify_jit_cache_provider_wheelhouse.py" \
        --wheelhouse "${output_dir}" \
        --provider "${PROVIDER_TAG}" \
        --version "${PACKAGE_VERSION}" \
        --cuobjdump /usr/local/cuda/bin/cuobjdump \
        --install-smoke

    echo "Built wheelhouse: ${output_dir}"
    ls -lh "${output_dir}"
}

case "${1:-}" in
    --inside-container)
        run_in_container
        ;;
    "")
        run_on_host
        ;;
    *)
        echo "Usage: $0 [--inside-container]" >&2
        exit 2
        ;;
esac
