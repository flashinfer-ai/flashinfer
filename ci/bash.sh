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

# Keep content-keyed kernel compile caches across the jobs of this runner, so
# tests stop recompiling every CuTe-DSL, TIRx, KDA PTX and Triton kernel per
# job. One directory per runner: a runner runs one job at a time, so no two
# checkouts ever share a directory concurrently. GPU jobs only.
# CI_COMPILE_CACHE_DIR="" opts out.
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
prune_compile_cache() {
    local cache_dir=$1
    [ -d "${cache_dir}" ] || return 0
    # Containers may write as root, so the runner user may need sudo.
    local rm_cmd=(rm -rf)
    if [ ! -w "${cache_dir}" ] || find "${cache_dir}" ! -user "$(id -u)" -print -quit 2>/dev/null | grep -q .; then
        rm_cmd=(sudo -n rm -rf)
    fi
    # GNU find/xargs (CI runners); elsewhere this quietly prunes nothing.
    {
        find "${cache_dir}" -path '*/cached_ops/*' -name build.ninja -printf '%h\0'
        find "${cache_dir}" -type d \( -name autotune \
            -o -path '*/.cache/flashinfer/*/*/generated' \) -prune -printf '%p\0'
        find "${cache_dir}" -name flashinfer_jit.log -printf '%p\0'
    } 2>/dev/null | xargs -0 -r "${rm_cmd[@]}" 2>/dev/null || true
}

COMPILE_CACHE_DIR=""
if [[ ${USE_GPU} == "true" ]]; then
    : "${CI_COMPILE_CACHE_DIR=${HOME:-/tmp}/.cache/flashinfer-ci/compile/${RUNNER_NAME:-local}}"
    if [ -n "${CI_COMPILE_CACHE_DIR}" ] && mkdir -p "${CI_COMPILE_CACHE_DIR}" 2>/dev/null; then
        COMPILE_CACHE_DIR=${CI_COMPILE_CACHE_DIR}
        prune_compile_cache "${COMPILE_CACHE_DIR}"
        CACHE_MB=$(du -sm "${COMPILE_CACHE_DIR}" 2>/dev/null | cut -f1)
        echo "Compile cache: ${COMPILE_CACHE_DIR} (${CACHE_MB:-0} MB)"
        if [ "${CACHE_MB:-0}" -gt "${CI_COMPILE_CACHE_MAX_MB:-40960}" ]; then
            echo "Clearing compile cache: ${CACHE_MB} MB exceeds ${CI_COMPILE_CACHE_MAX_MB:-40960} MB"
            rm -rf "${COMPILE_CACHE_DIR:?}" 2>/dev/null \
                || sudo -n rm -rf "${COMPILE_CACHE_DIR:?}" 2>/dev/null || true
            mkdir -p "${COMPILE_CACHE_DIR}" 2>/dev/null || true
        fi
        DOCKER_VOLUMNS="${DOCKER_VOLUMNS} -v ${COMPILE_CACHE_DIR}:/compile-cache"
        DOCKER_ENV="${DOCKER_ENV} -e FLASHINFER_WORKSPACE_BASE=/compile-cache/flashinfer"
        DOCKER_ENV="${DOCKER_ENV} -e FLASHINFER_CUTE_DSL_STRICT_CACHE_KEY=1"
        DOCKER_ENV="${DOCKER_ENV} -e CUTE_DSL_CACHE_DIR=/tmp/cute-dsl-cache"
        DOCKER_ENV="${DOCKER_ENV} -e TRITON_CACHE_DIR=/compile-cache/triton"
    else
        echo "Compile cache disabled or not writable; kernels will not be reused"
    fi
fi

# Use nvidia-docker if the container is GPU.
if [[ ${USE_GPU} == "true" ]]; then
    # Keep CPU-only sharding infrastructure tests out of GPU CI collection.
    # Forward by name so Docker preserves spaces in existing pytest options.
    export PYTEST_ADDOPTS="${PYTEST_ADDOPTS:+${PYTEST_ADDOPTS} }--ignore=tests/test_sharding"
    DOCKER_ENV="${DOCKER_ENV} -e PYTEST_ADDOPTS"

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
