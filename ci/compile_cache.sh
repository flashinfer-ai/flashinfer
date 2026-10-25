#!/usr/bin/env bash
# Sourced by bash.sh after pulling the image. Lock files live outside evicted
# trees, and each runner has its own namespace even on shared work volumes.

COMPILE_CACHE_VOLUMES=()
COMPILE_CACHE_ENV=()
COMPILE_CACHE_ENTRYPOINT=()
COMPILE_CACHE_IMAGE_DIR=""
COMPILE_CACHE_PRIVATE_DIR=""
COMPILE_CACHE_LOCKED=false

remove_cache_trees() {
    [ "$#" -gt 0 ] || return 0
    rm -rf -- "$@" 2>/dev/null || sudo -n rm -rf -- "$@" 2>/dev/null || {
        echo "WARNING: unable to remove compile cache paths" >&2
        return 1
    }
}

prune_compile_cache() {
    local directory=$1
    [ -d "${directory}" ] || return 0
    local paths=()
    mapfile -d '' -t paths < <({
        find "${directory}" -path '*/cached_ops/*' -name build.ninja -printf '%h\0'
        find "${directory}" -type d \( -name autotune \
            -o -path '*/.cache/flashinfer/*/*/generated' \) -prune -printf '%p\0'
        find "${directory}" -name flashinfer_jit.log -printf '%p\0'
    } 2>/dev/null)
    remove_cache_trees "${paths[@]}"
}

prepare_compile_cache() {
    [[ ${USE_GPU} == true && -n "${CI_COMPILE_CACHE_DIR:-}" ]] || return 0
    local mode=${CI_COMPILE_CACHE_MODE:-read-only}
    local max_mb=${CI_COMPILE_CACHE_MAX_MB:-40960}
    local min_free_mb=${CI_COMPILE_CACHE_MIN_FREE_MB:-20480}
    if ! [[ ${max_mb} =~ ^[0-9]+$ && ${min_free_mb} =~ ^[0-9]+$ ]] \
        || ! [[ ${mode} == read-only || ${mode} == read-write ]]; then
        echo "WARNING: invalid compile cache mode or limits; compile cache disabled"
        return 0
    fi
    if ! command -v flock >/dev/null; then
        echo "WARNING: flock unavailable; compile cache disabled"
        return 0
    fi
    local image_id
    image_id=$(${DOCKER_BINARY} image inspect --format '{{.Id}}' "${DOCKER_IMAGE_NAME}")
    if ! [[ ${image_id} =~ ^sha256:[0-9a-f]{64}$ ]]; then
        echo "WARNING: immutable image identity unavailable; compile cache disabled"
        return 0
    fi
    local runner_key
    runner_key=$(printf '%s' "${CI_COMPILE_CACHE_RUNNER:-${RUNNER_NAME:-local}}" | sha256sum | cut -d ' ' -f1)
    local runner_dir=${CI_COMPILE_CACHE_DIR}/runner-${runner_key}
    local images=${runner_dir}/images
    if ! mkdir -p "${images}" \
        || ! { exec 9>"${CI_COMPILE_CACHE_DIR}/.runner-${runner_key}.lock"; }; then
        echo "WARNING: compile cache is not writable; compile cache disabled"
        return 0
    fi
    if ! flock -x 9; then
        echo "WARNING: compile cache lock failed; compile cache disabled"
        exec 9>&-
        return 0
    fi
    COMPILE_CACHE_LOCKED=true
    local image_key=${image_id#sha256:}
    local used_mb free_mb
    COMPILE_CACHE_IMAGE_DIR=${images}/${image_key}
    if [[ ${mode} == read-write ]]; then
        prune_compile_cache "${images}" || return 1
        local stale=()
        mapfile -d '' -t stale < <(find "${images}" -mindepth 1 -maxdepth 1 \
            -type d ! -name "${image_key}" -mtime +7 -print0)
        remove_cache_trees "${stale[@]}" || return 1
        used_mb=$(du -sm "${images}" | cut -f1)
        free_mb=$(df -Pm "${runner_dir}" | awk 'NR == 2 { print $4 }')
        if [ "${used_mb:-0}" -gt "${max_mb}" ] || [ "${free_mb:-0}" -lt "${min_free_mb}" ]; then
            echo "Clearing compile cache: limit ${max_mb} MB, minimum free ${min_free_mb} MB"
            remove_cache_trees "${images}" || return 1
        fi
        mkdir -p "${COMPILE_CACHE_IMAGE_DIR}" || return 1
        touch "${COMPILE_CACHE_IMAGE_DIR}" || return 1
        COMPILE_CACHE_VOLUMES=(-v "${COMPILE_CACHE_IMAGE_DIR}:/compile-cache")
    else
        # PR jobs consume a read-only seed and discard their private additions.
        # A cold image has no seed; the entrypoint handles that case too.
        COMPILE_CACHE_PRIVATE_DIR=$(mktemp -d "${runner_dir}/.job.XXXXXXXX") || return 1
        COMPILE_CACHE_VOLUMES=(-v "${COMPILE_CACHE_PRIVATE_DIR}:/compile-cache")
        if [ -d "${COMPILE_CACHE_IMAGE_DIR}" ]; then
            used_mb=$(du -sm "${COMPILE_CACHE_IMAGE_DIR}" | cut -f1)
            free_mb=$(df -Pm "${runner_dir}" | awk 'NR == 2 { print $4 }')
            if [ "${used_mb}" -le "${max_mb}" ] \
                && [ "$((free_mb - used_mb))" -ge "${min_free_mb}" ]; then
                COMPILE_CACHE_VOLUMES+=(-v "${COMPILE_CACHE_IMAGE_DIR}:/compile-cache-seed:ro")
            else
                echo "Skipping compile cache seed: size or free-space limit"
            fi
        fi
        COMPILE_CACHE_ENTRYPOINT=(bash /docker/compile_cache_entrypoint.sh /compile-cache-seed /compile-cache)
    fi
    DOCKER_IMAGE_ID=${image_id}
    COMPILE_CACHE_ENV=(
        -e FLASHINFER_WORKSPACE_BASE=/compile-cache/flashinfer
        -e FLASHINFER_CUTE_DSL_STRICT_CACHE_KEY=1
        -e CUTE_DSL_CACHE_DIR=/tmp/cute-dsl-cache
        -e TRITON_CACHE_DIR=/compile-cache/triton
    )
    echo "Compile cache: mode=${mode} runner=${runner_key} image=${image_id}"
}

cleanup_compile_cache() {
    if [[ ${CI_COMPILE_CACHE_MODE:-read-only} == read-write && -n "${COMPILE_CACHE_IMAGE_DIR}" ]]; then
        prune_compile_cache "${COMPILE_CACHE_IMAGE_DIR}" || true
    fi
    if [ -n "${COMPILE_CACHE_PRIVATE_DIR}" ]; then
        remove_cache_trees "${COMPILE_CACHE_PRIVATE_DIR}" || true
    fi
    if [[ ${COMPILE_CACHE_LOCKED} == true ]]; then
        flock -u 9
        exec 9>&-
    fi
}
