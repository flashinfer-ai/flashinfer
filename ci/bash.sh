#!/usr/bin/env bash

if [ "$#" -lt 1 ]; then
    echo "Usage: ci/bash.sh <CONTAINER_NAME> -e key [value] -v key value [COMMAND]"
    exit -1
fi

DOCKER_IMAGE_NAME=("$1")
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(pwd)"
DOCKER_BINARY="docker"
DOCKER_ENV="-e ENV_USER_ID=$(id -u) -e ENV_GROUP_ID=$(id -g)"
DOCKER_VOLUMNS="-v ${WORKSPACE}:/workspace -v ${SCRIPT_DIR}:/docker"
USE_GPU=true

shift 1
while [[ $# -gt 0 ]]; do
    cmd="$1"
    if [[ $cmd == "-e" ]]; then
        if [[ $# -lt 2 ]]; then
            echo "ERROR: -e requires an environment variable name"
            exit -1
        fi
        env_key=$2
        if [[ $# -ge 3 && $3 != -* ]]; then
            env_value=$3
            shift 3
            DOCKER_ENV="${DOCKER_ENV} -e ${env_key}=${env_value}"
        else
            shift 2
            DOCKER_ENV="${DOCKER_ENV} -e ${env_key}"
        fi
    elif [[ $cmd == "-v" ]]; then
        volumn_key=$2
        volumn_value=$3
        shift 3
        DOCKER_VOLUMNS="${DOCKER_VOLUMNS} -v ${volumn_key}:${volumn_value}"
    elif [[ $cmd == "-j" ]]; then
        num_threads=$2
        shift 2
        DOCKER_ENV="${DOCKER_ENV} -e NUM_THREADS=${num_threads} --cpus ${num_threads}"
    elif [[ $cmd == "--no-gpu" ]]; then
        USE_GPU=false
        shift
    else
        break
    fi
done

if [ "$#" -eq 0 ]; then
    COMMAND="bash"
    if [[ $(uname) == "Darwin" ]]; then
        # Docker's host networking driver isn't supported on macOS.
        # Use default bridge network and expose port for jupyter notebook.
        DOCKER_EXTRA_PARAMS=("-it -p 8888:8888")
    else
        DOCKER_EXTRA_PARAMS=("-it --net=host")
    fi
else
    COMMAND=("$@")
fi

DOCKER_ENV="${DOCKER_ENV} -e PIP_RETRIES=10 -e PIP_DEFAULT_TIMEOUT=60"

# Share pip's downloads with the other jobs that land on this reused runner. In
# HOME because the workspace is wiped between jobs and /opt needs root.
# CI_PIP_CACHE_DIR="" opts out.
: "${CI_PIP_CACHE_DIR=${HOME:-/tmp}/.cache/flashinfer-ci/pip}"
if [ -n "${CI_PIP_CACHE_DIR}" ] && mkdir -p "${CI_PIP_CACHE_DIR}" 2>/dev/null; then
    CACHE_MB=$(du -sm "${CI_PIP_CACHE_DIR}" 2>/dev/null | cut -f1)
    if [ "${CACHE_MB:-0}" -gt "${CI_PIP_CACHE_MAX_MB:-20480}" ]; then
        echo "Clearing pip cache: ${CACHE_MB} MB exceeds ${CI_PIP_CACHE_MAX_MB:-20480} MB"
        # Containers write into it as root, so the runner user may need sudo.
        rm -rf "${CI_PIP_CACHE_DIR:?}" 2>/dev/null \
            || sudo -n rm -rf "${CI_PIP_CACHE_DIR:?}" 2>/dev/null || true
        mkdir -p "${CI_PIP_CACHE_DIR}" 2>/dev/null || true
    fi
    DOCKER_VOLUMNS="${DOCKER_VOLUMNS} -v ${CI_PIP_CACHE_DIR}:/pip-cache"
    DOCKER_ENV="${DOCKER_ENV} -e PIP_CACHE_DIR=/pip-cache"
else
    echo "Shared pip cache disabled or not writable; downloads will not be reused"
fi

# Keep content-keyed kernel compile caches across the jobs of one runner, so
# tests stop recompiling every CuTe-DSL, TIRx, KDA PTX and Triton kernel per
# job. Opt-in: a caller sets CI_COMPILE_CACHE_DIR to a directory only it uses
# (PR Test passes one under its runner's work directory), so other GPU callers
# such as the nightly package validation keep testing from a cold cache. A
# runner runs one job at a time, so no two checkouts share a directory at once.
# Each CI image gets its own subdirectory: a FlashInfer cache key does not cover
# every package in the image (torch, tvm-ffi), so an image bump starts cold.
#
# What is deliberately not kept:
# - nvcc modules: a fresh checkout makes every source newer than its .so, so
#   ninja rebuilds them anyway. They are pruned instead of taking disk.
# - Tuning results (*/autotune): tests must tune from scratch.
# - The CuTe-DSL compiler's own file cache (CUTE_DSL_CACHE_DIR), whose key
#   is not under FlashInfer's control. FlashInfer's CuTe-DSL cache is kept,
#   with FLASHINFER_CUTE_DSL_STRICT_CACHE_KEY=1 so a kernel is also keyed on
#   the in-package helpers it imports, not only on its own file.
# - The TRT-LLM DeepGEMM cache (~/.tensorrt_llm/cache), keyed on shapes only.

# Containers may write as root, so the runner user may need sudo.
remove_trees() {
    [ "$#" -gt 0 ] || return 0
    rm -rf "$@" 2>/dev/null || sudo -n rm -rf "$@" 2>/dev/null || true
}

prune_compile_cache() {
    local cache_dir=$1
    [ -d "${cache_dir}" ] || return 0
    local prunable=()
    # GNU find (CI runners); elsewhere this quietly prunes nothing.
    mapfile -d '' -t prunable < <({
        find "${cache_dir}" -path '*/cached_ops/*' -name build.ninja -printf '%h\0'
        find "${cache_dir}" -type d \( -name autotune \
            -o -path '*/.cache/flashinfer/*/*/generated' \) -prune -printf '%p\0'
        find "${cache_dir}" -name flashinfer_jit.log -printf '%p\0'
    } 2>/dev/null)
    remove_trees "${prunable[@]}"
}

COMPILE_CACHE_DIR=""
if [[ ${USE_GPU} == "true" && -n "${CI_COMPILE_CACHE_DIR:-}" ]]; then
    CACHE_MAX_MB=${CI_COMPILE_CACHE_MAX_MB:-40960}
    CACHE_MIN_FREE_MB=${CI_COMPILE_CACHE_MIN_FREE_MB:-20480}
    if ! [[ ${CACHE_MAX_MB} =~ ^[0-9]+$ && ${CACHE_MIN_FREE_MB} =~ ^[0-9]+$ ]]; then
        echo "WARNING: CI_COMPILE_CACHE_MAX_MB and CI_COMPILE_CACHE_MIN_FREE_MB must be whole MB; compile cache disabled"
    elif mkdir -p "${CI_COMPILE_CACHE_DIR}" 2>/dev/null; then
        IMAGE_KEY=$(printf '%s' "${DOCKER_IMAGE_NAME}" | tr -c 'A-Za-z0-9._-' '_')
        COMPILE_CACHE_DIR="${CI_COMPILE_CACHE_DIR}/${IMAGE_KEY}"
        prune_compile_cache "${CI_COMPILE_CACHE_DIR}"
        # Caches of images this runner has not used for a week.
        mapfile -t STALE_IMAGES < <(find "${CI_COMPILE_CACHE_DIR}" -mindepth 1 -maxdepth 1 \
            -type d ! -name "${IMAGE_KEY}" -mtime +7 2>/dev/null)
        remove_trees "${STALE_IMAGES[@]}"
        CACHE_MB=$(du -sm "${CI_COMPILE_CACHE_DIR}" 2>/dev/null | cut -f1)
        FREE_MB=$(df -Pm "${CI_COMPILE_CACHE_DIR}" 2>/dev/null | awk 'NR == 2 { print $4 }')
        echo "Compile cache: ${COMPILE_CACHE_DIR} (${CACHE_MB:-0} MB used, ${FREE_MB:-?} MB free on its volume)"
        # Both checks run only here, before the job: the job itself must also fit
        # in the free space left on this volume.
        if [ "${CACHE_MB:-0}" -gt "${CACHE_MAX_MB}" ] \
            || [ "${FREE_MB:-0}" -lt "${CACHE_MIN_FREE_MB}" ]; then
            echo "Clearing compile cache: limit ${CACHE_MAX_MB} MB, minimum free ${CACHE_MIN_FREE_MB} MB"
            remove_trees "${CI_COMPILE_CACHE_DIR:?}"
        fi
        mkdir -p "${COMPILE_CACHE_DIR}" && touch "${COMPILE_CACHE_DIR}"
        DOCKER_VOLUMNS="${DOCKER_VOLUMNS} -v ${COMPILE_CACHE_DIR}:/compile-cache"
        DOCKER_ENV="${DOCKER_ENV} -e FLASHINFER_WORKSPACE_BASE=/compile-cache/flashinfer"
        DOCKER_ENV="${DOCKER_ENV} -e FLASHINFER_CUTE_DSL_STRICT_CACHE_KEY=1"
        DOCKER_ENV="${DOCKER_ENV} -e CUTE_DSL_CACHE_DIR=/tmp/cute-dsl-cache"
        DOCKER_ENV="${DOCKER_ENV} -e TRITON_CACHE_DIR=/compile-cache/triton"
    else
        echo "WARNING: compile cache ${CI_COMPILE_CACHE_DIR} is not writable; kernels will not be reused"
    fi
fi

# Use nvidia-docker if the container is GPU.
if [[ ${USE_GPU} == "true" ]]; then
    # Keep CPU-only sharding infrastructure tests out of GPU CI collection.
    # Forward by name so Docker preserves spaces in existing pytest options.
    export PYTEST_ADDOPTS="${PYTEST_ADDOPTS:+${PYTEST_ADDOPTS} }--ignore=tests/test_sharding"
    DOCKER_ENV="${DOCKER_ENV} -e PYTEST_ADDOPTS"

    # Forward CI_GPU_NAME so summary.py can record the lane without a torch import.
    if [[ -n "${CI_GPU_NAME:-}" ]]; then
        DOCKER_ENV="${DOCKER_ENV} -e CI_GPU_NAME"
    fi
    # Forward lane-identity variables for the composite lane_id.
    if [[ -n "${TEST_SCRIPT:-}" ]]; then
        export CI_TEST_SCRIPT="${TEST_SCRIPT}"
        DOCKER_ENV="${DOCKER_ENV} -e CI_TEST_SCRIPT"
    fi
    if [[ -n "${MATRIX_CUDA:-}" ]]; then
        export CI_MATRIX_CUDA="${MATRIX_CUDA}"
        DOCKER_ENV="${DOCKER_ENV} -e CI_MATRIX_CUDA"
    fi

    # An unset host mask should leave the container's GPUs visible. Preserve
    # explicitly configured masks, including an intentionally empty one.
    if [[ -v CUDA_VISIBLE_DEVICES ]]; then
        DOCKER_ENV="${DOCKER_ENV} -e CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    fi
    if type nvidia-docker 1> /dev/null 2> /dev/null; then
        DOCKER_BINARY=nvidia-docker
    else
        DOCKER_BINARY=docker
        DOCKER_ENV="${DOCKER_ENV} --gpus all"
    fi
fi

# Print arguments.
echo "DOCKER_BINARY ${DOCKER_BINARY}"
echo "WORKSPACE: ${WORKSPACE}"
echo "IMAGE NAME: ${DOCKER_IMAGE_NAME}"
echo "ENV VARIABLES: ${DOCKER_ENV}"
echo "VOLUMES: ${DOCKER_VOLUMNS}"
echo "COMMANDS: '${COMMAND[@]}'"

# Pull the latest docker image
echo "Pulling latest image: ${DOCKER_IMAGE_NAME}"
${DOCKER_BINARY} pull ${DOCKER_IMAGE_NAME}

# By default we cleanup - remove the container once it finish running (--rm)
# and share the PID namespace (--pid=host) so the process inside does not have
# pid 1 and SIGKILL is propagated to the process inside (jenkins can kill it).

${DOCKER_BINARY} run --rm --pid=host \
    -w /workspace \
    ${DOCKER_VOLUMNS} \
    ${DOCKER_ENV} \
    ${DOCKER_EXTRA_PARAMS[@]} \
    ${DOCKER_IMAGE_NAME} \
    ${COMMAND[@]}
DOCKER_STATUS=$?

# Drop what the next job would rebuild anyway before the runner sits idle.
if [ -n "${COMPILE_CACHE_DIR}" ]; then
    prune_compile_cache "${COMPILE_CACHE_DIR}"
fi
exit "${DOCKER_STATUS}"
