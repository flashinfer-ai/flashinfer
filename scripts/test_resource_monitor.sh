#!/usr/bin/env bash
# Sourceable lifecycle shared by the standalone runner and flashinfer-ci.
# The owning wrapper stops the sampler; nested wrappers reuse its snapshot.

start_resource_monitor() {
    local monitor_enabled=${MONITOR_TEST_MEMORY:-true}
    case "${monitor_enabled,,}" in 0|false|no) return 0 ;; esac
    if [ -n "${FLASHINFER_RESOURCE_MONITOR_PID:-}" ] && kill -0 "${FLASHINFER_RESOURCE_MONITOR_PID}" 2>/dev/null; then
        return 0
    fi
    local monitor_script monitor_root monitor_dir
    monitor_script="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/test_sharding/resource_monitor.py"
    monitor_root="${JUNIT_DIR:-./junit}/resource-telemetry"
    mkdir -p "${monitor_root}"
    monitor_dir=$(mktemp -d "${monitor_root}/run-XXXXXXXX")
    monitor_dir=$(cd "${monitor_dir}" && pwd)
    export FLASHINFER_RESOURCE_SNAPSHOT="${monitor_dir}/gpu-processes.json"
    python "${monitor_script}" --output-dir "${monitor_dir}" --parent-pid "$$" \
        --interval "${RESOURCE_MONITOR_INTERVAL:-5}" > "${monitor_dir}/monitor.log" 2>&1 &
    RESOURCE_MONITOR_OWN_PID=$!
    export FLASHINFER_RESOURCE_MONITOR_PID=${RESOURCE_MONITOR_OWN_PID}
    # Let Python install its signal handlers before even a very short wrapper
    # can stop it. Metadata is written before the potentially slow GPU query.
    local attempt
    for ((attempt=0; attempt<100; attempt++)); do
        [ -f "${monitor_dir}/metadata.json" ] && return 0
        if ! kill -0 "${RESOURCE_MONITOR_OWN_PID}" 2>/dev/null; then
            echo "WARNING: resource sampler failed; see ${monitor_dir}/monitor.log" >&2
            return 0
        fi
        sleep 0.05
    done
    echo "WARNING: resource sampler startup is delayed; see ${monitor_dir}/monitor.log" >&2
}

stop_resource_monitor() {
    if [ -n "${RESOURCE_MONITOR_OWN_PID:-}" ]; then
        kill -TERM "${RESOURCE_MONITOR_OWN_PID}" 2>/dev/null || true
        wait "${RESOURCE_MONITOR_OWN_PID}" 2>/dev/null || true
        RESOURCE_MONITOR_OWN_PID=
        unset FLASHINFER_RESOURCE_MONITOR_PID FLASHINFER_RESOURCE_SNAPSHOT
    fi
}
