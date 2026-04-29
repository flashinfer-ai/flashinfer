#!/bin/bash

set -eo pipefail
set -x

# Source test environment setup (handles package overrides like TVM-FFI)
source "$(dirname "${BASH_SOURCE[0]}")/setup_test_env.sh"

: "${AOT_MEMORY_MONITOR:=true}"
: "${AOT_MEMORY_MONITOR_INTERVAL:=2}"
: "${AOT_MEMORY_REPORT_DIR:=aot-memory-reports}"

AOT_MEMORY_REPORT_FILES=()
LAST_AOT_MEMORY_MONITOR_PID=""
LAST_AOT_MEMORY_REPORT=""

mkdir -p "${AOT_MEMORY_REPORT_DIR}"
AOT_MEMORY_REPORT_DIR=$(cd "${AOT_MEMORY_REPORT_DIR}" && pwd)

sanitize_aot_memory_label() {
    local label=$1
    printf '%s' "$label" | tr -c 'A-Za-z0-9._-' '_'
}

list_process_tree_pids() {
    local root_pid=$1
    local -a queue=("$root_pid")
    local -a tree_pids=()
    local idx=0

    while [ $idx -lt ${#queue[@]} ]; do
        local pid="${queue[$idx]}"
        idx=$((idx + 1))

        if ! kill -0 "$pid" 2>/dev/null; then
            continue
        fi

        tree_pids+=("$pid")

        local child_pid
        while read -r child_pid; do
            [ -n "$child_pid" ] && queue+=("$child_pid")
        done < <(ps -o pid= --ppid "$pid" 2>/dev/null | awk '{print $1}')
    done

    printf '%s\n' "${tree_pids[@]}"
}

sum_rss_kib_for_pids() {
    local -a pid_list=("$@")

    if [ ${#pid_list[@]} -eq 0 ]; then
        echo 0
        return
    fi

    local pid_csv
    pid_csv=$(IFS=,; echo "${pid_list[*]}")
    ps -o rss= -p "$pid_csv" 2>/dev/null | awk '{sum += $1} END {print sum + 0}'
}

read_system_mem_available_kib() {
    awk '/MemAvailable/ {print $2}' /proc/meminfo 2>/dev/null || echo 0
}

read_cgroup_memory_kib() {
    local file=$1

    if [ ! -r "$file" ]; then
        echo 0
        return
    fi

    local value
    value=$(cat "$file" 2>/dev/null || echo 0)
    if [[ "$value" =~ ^[0-9]+$ ]]; then
        echo $(( (value + 1023) / 1024 ))
    else
        echo 0
    fi
}

read_cgroup_current_kib() {
    if [ -r /sys/fs/cgroup/memory.current ]; then
        read_cgroup_memory_kib /sys/fs/cgroup/memory.current
    else
        read_cgroup_memory_kib /sys/fs/cgroup/memory/memory.usage_in_bytes
    fi
}

read_cgroup_peak_kib() {
    if [ -r /sys/fs/cgroup/memory.peak ]; then
        read_cgroup_memory_kib /sys/fs/cgroup/memory.peak
    else
        read_cgroup_memory_kib /sys/fs/cgroup/memory/memory.max_usage_in_bytes
    fi
}

start_aot_memory_monitor() {
    local root_pid=$1
    local label=$2
    local safe_label
    safe_label=$(sanitize_aot_memory_label "$label")
    local report_file="${AOT_MEMORY_REPORT_DIR}/${safe_label}.memory.csv"

    LAST_AOT_MEMORY_MONITOR_PID=""
    LAST_AOT_MEMORY_REPORT="$report_file"
    AOT_MEMORY_REPORT_FILES+=("$report_file")

    if [ "$AOT_MEMORY_MONITOR" != "true" ]; then
        return
    fi

    (
        set +x

        local peak_rss_kib=0
        local peak_cgroup_current_kib=0
        local peak_cgroup_peak_kib=0
        local min_system_mem_available_kib=0
        local peak_proc_count=0
        local sample_count=0
        local start_epoch
        start_epoch=$(date +%s)

        {
            echo "# label=${label}"
            echo "# root_pid=${root_pid}"
            echo "# sample_interval_seconds=${AOT_MEMORY_MONITOR_INTERVAL}"
            echo "timestamp_utc,rss_kib,system_mem_available_kib,cgroup_current_kib,cgroup_peak_kib,process_count"
        } > "$report_file"

        while kill -0 "$root_pid" 2>/dev/null; do
            local -a tree_pids=()
            mapfile -t tree_pids < <(list_process_tree_pids "$root_pid")

            if [ ${#tree_pids[@]} -eq 0 ]; then
                sleep "$AOT_MEMORY_MONITOR_INTERVAL"
                continue
            fi

            local rss_kib system_mem_available_kib cgroup_current_kib cgroup_peak_kib proc_count timestamp
            rss_kib=$(sum_rss_kib_for_pids "${tree_pids[@]}")
            system_mem_available_kib=$(read_system_mem_available_kib)
            cgroup_current_kib=$(read_cgroup_current_kib)
            cgroup_peak_kib=$(read_cgroup_peak_kib)
            proc_count=${#tree_pids[@]}
            timestamp=$(date -u '+%Y-%m-%dT%H:%M:%SZ')

            printf '%s,%s,%s,%s,%s,%s\n' \
                "$timestamp" \
                "$rss_kib" \
                "$system_mem_available_kib" \
                "$cgroup_current_kib" \
                "$cgroup_peak_kib" \
                "$proc_count" >> "$report_file"

            [ "$rss_kib" -gt "$peak_rss_kib" ] && peak_rss_kib=$rss_kib
            [ "$cgroup_current_kib" -gt "$peak_cgroup_current_kib" ] && peak_cgroup_current_kib=$cgroup_current_kib
            [ "$cgroup_peak_kib" -gt "$peak_cgroup_peak_kib" ] && peak_cgroup_peak_kib=$cgroup_peak_kib
            [ "$proc_count" -gt "$peak_proc_count" ] && peak_proc_count=$proc_count
            if [ "$system_mem_available_kib" -gt 0 ] && { [ "$min_system_mem_available_kib" -eq 0 ] || [ "$system_mem_available_kib" -lt "$min_system_mem_available_kib" ]; }; then
                min_system_mem_available_kib=$system_mem_available_kib
            fi
            sample_count=$((sample_count + 1))

            sleep "$AOT_MEMORY_MONITOR_INTERVAL"
        done

        {
            echo "# summary"
            echo "peak_rss_kib=${peak_rss_kib}"
            echo "peak_cgroup_current_kib=${peak_cgroup_current_kib}"
            echo "peak_cgroup_peak_kib=${peak_cgroup_peak_kib}"
            echo "min_system_mem_available_kib=${min_system_mem_available_kib}"
            echo "peak_process_count=${peak_proc_count}"
            echo "sample_count=${sample_count}"
            echo "duration_seconds=$(( $(date +%s) - start_epoch ))"
        } >> "$report_file"
    ) &

    LAST_AOT_MEMORY_MONITOR_PID=$!
}

wait_for_aot_memory_monitor() {
    local monitor_pid=$1

    if [ -n "$monitor_pid" ]; then
        wait "$monitor_pid" 2>/dev/null || true
    fi
}

print_aot_memory_summary() {
    local label=$1
    local report_file=$2

    if [ "$AOT_MEMORY_MONITOR" != "true" ] || [ ! -f "$report_file" ]; then
        return
    fi

    local peak_rss_kib peak_cgroup_current_kib peak_cgroup_peak_kib min_system_mem_available_kib peak_proc_count sample_count duration_seconds
    peak_rss_kib=$(awk -F= '/^peak_rss_kib=/{print $2}' "$report_file" | tail -1)
    peak_cgroup_current_kib=$(awk -F= '/^peak_cgroup_current_kib=/{print $2}' "$report_file" | tail -1)
    peak_cgroup_peak_kib=$(awk -F= '/^peak_cgroup_peak_kib=/{print $2}' "$report_file" | tail -1)
    min_system_mem_available_kib=$(awk -F= '/^min_system_mem_available_kib=/{print $2}' "$report_file" | tail -1)
    peak_proc_count=$(awk -F= '/^peak_process_count=/{print $2}' "$report_file" | tail -1)
    sample_count=$(awk -F= '/^sample_count=/{print $2}' "$report_file" | tail -1)
    duration_seconds=$(awk -F= '/^duration_seconds=/{print $2}' "$report_file" | tail -1)

    peak_rss_kib=${peak_rss_kib:-0}
    peak_cgroup_current_kib=${peak_cgroup_current_kib:-0}
    peak_cgroup_peak_kib=${peak_cgroup_peak_kib:-0}
    min_system_mem_available_kib=${min_system_mem_available_kib:-0}
    peak_proc_count=${peak_proc_count:-0}
    sample_count=${sample_count:-0}
    duration_seconds=${duration_seconds:-0}

    echo "MEMORY: ${label}: peak RSS $(( (peak_rss_kib + 1023) / 1024 )) MiB, peak cgroup current $(( (peak_cgroup_current_kib + 1023) / 1024 )) MiB, cgroup peak $(( (peak_cgroup_peak_kib + 1023) / 1024 )) MiB, min MemAvailable $(( (min_system_mem_available_kib + 1023) / 1024 )) MiB, peak processes ${peak_proc_count}, samples ${sample_count}, duration ${duration_seconds}s"
    echo "Memory report: ${report_file}"
}

run_with_aot_memory_monitor() {
    local label=$1
    shift

    echo ""
    echo "[MEMORY] Monitoring step: ${label}"
    echo "[MEMORY] Command: $*"

    local cmd_pid monitor_pid report_file cmd_ec=0
    "$@" &
    cmd_pid=$!
    start_aot_memory_monitor "$cmd_pid" "$label"
    monitor_pid=$LAST_AOT_MEMORY_MONITOR_PID
    report_file=$LAST_AOT_MEMORY_REPORT
    wait "$cmd_pid" || cmd_ec=$?
    wait_for_aot_memory_monitor "$monitor_pid"
    print_aot_memory_summary "$label" "$report_file"

    return "$cmd_ec"
}

print_aot_memory_report_index() {
    if [ "$AOT_MEMORY_MONITOR" != "true" ] || [ ${#AOT_MEMORY_REPORT_FILES[@]} -eq 0 ]; then
        return
    fi

    echo ""
    echo "AOT memory report files:"
    printf '  %s\n' "${AOT_MEMORY_REPORT_FILES[@]}"
}

print_aot_memory_diagnostics() {
    echo ""
    echo "AOT memory diagnostics:"
    for file in \
        /sys/fs/cgroup/memory.events \
        /sys/fs/cgroup/memory.current \
        /sys/fs/cgroup/memory.peak \
        /sys/fs/cgroup/memory/memory.oom_control \
        /sys/fs/cgroup/memory/memory.failcnt \
        /sys/fs/cgroup/memory/memory.usage_in_bytes \
        /sys/fs/cgroup/memory/memory.max_usage_in_bytes; do
        if [ -r "$file" ]; then
            echo "--- ${file} ---"
            cat "$file" || true
        fi
    done

    echo "--- recent kernel OOM events ---"
    dmesg -T 2>/dev/null | grep -i "oom\|killed process\|out of memory" | tail -20 || echo "dmesg unavailable or no OOM events found"
}

finish_aot_memory_monitoring() {
    local exit_code=$?
    print_aot_memory_report_index
    if [ "$exit_code" -ne 0 ]; then
        print_aot_memory_diagnostics
    fi
}

trap finish_aot_memory_monitoring EXIT

echo "========================================"
echo "Starting flashinfer-jit-cache test script"
echo "========================================"

# MAX_JOBS = min(nproc, max(1, MemAvailable_GB/(8 on aarch64, 4 otherwise)))
MEM_AVAILABLE_GB=$(free -g | awk '/^Mem:/ {print $7}')
NPROC=$(nproc)
MAX_JOBS=$(( MEM_AVAILABLE_GB / $([ "$(uname -m)" = "aarch64" ] && echo 8 || echo 4) ))
if (( MAX_JOBS < 1 )); then
  MAX_JOBS=1
elif (( NPROC < MAX_JOBS )); then
  MAX_JOBS=$NPROC
fi

echo "System Information:"
echo "  - Available Memory: ${MEM_AVAILABLE_GB} GB"
echo "  - Number of Processors: ${NPROC}"
echo "  - MAX_JOBS: ${MAX_JOBS}"

# Export MAX_JOBS for PyTorch's cpp_extension to use
export MAX_JOBS

: ${CUDA_VISIBLE_DEVICES:=""}
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"

# Clean Python bytecode cache to avoid stale imports (e.g., after module refactoring)
echo "Cleaning Python bytecode cache..."
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
find . -type f -name '*.pyc' -delete 2>/dev/null || true
echo "Cache cleaned."

echo ""
echo "Detecting CUDA architecture list..."
export FLASHINFER_CUDA_ARCH_LIST=$(python3 -c '
import torch
cuda_ver = torch.version.cuda
arches = ["7.5", "8.0", "8.9", "9.0a"]
if cuda_ver is not None:
    try:
        major, minor = map(int, cuda_ver.split(".")[:2])
        if (major, minor) >= (13, 0):
            arches.append("10.0a")
            arches.append("10.3a")
            arches.append("11.0a")
            arches.append("12.0f")
        elif (major, minor) >= (12, 9):
            arches.append("10.0a")
            arches.append("10.3a")
            arches.append("12.0f")
        elif (major, minor) >= (12, 8):
            arches.append("10.0a")
            arches.append("12.0a")
    except Exception:
        pass
print(" ".join(arches))
')
echo "FLASHINFER_CUDA_ARCH_LIST: ${FLASHINFER_CUDA_ARCH_LIST}"

echo ""
echo "Current PyTorch version:"
python -c "import torch; print(torch.__version__)"

# Detect CUDA version from the container
CUDA_VERSION=$(python3 -c 'import torch; print(torch.version.cuda)' | cut -d'.' -f1,2 | tr -d '.')
echo "Detected CUDA version: cu${CUDA_VERSION}"

echo ""
echo "========================================"
echo "Installing flashinfer package"
echo "========================================"
run_with_aot_memory_monitor "pip_install_flashinfer_editable" pip install -e . || {
    echo "ERROR: Failed to install flashinfer package"
    exit 1
}
echo "✓ Flashinfer package installed successfully"

echo ""
echo "========================================"
echo "Building flashinfer-jit-cache wheel"
echo "========================================"
cd flashinfer-jit-cache
run_with_aot_memory_monitor "build_flashinfer_jit_cache_wheel" python -m build --wheel

# Get the built wheel file
WHEEL_FILE=$(ls dist/*.whl | head -n 1)
echo ""
echo "Built wheel: $WHEEL_FILE"
echo ""

echo ""
echo "========================================"
echo "Installing flashinfer-jit-cache wheel"
echo "========================================"
echo "Wheel file: $WHEEL_FILE"
run_with_aot_memory_monitor "pip_install_flashinfer_jit_cache_wheel" pip install "$WHEEL_FILE" || {
    echo "ERROR: Failed to install flashinfer-jit-cache wheel"
    exit 1
}
echo "✓ Flashinfer-jit-cache wheel installed successfully"
cd ..

# Verify installation
echo ""
echo "========================================"
echo "Running verification tests"
echo "========================================"

# Test with show-config
echo "[STEP 1/2] Running 'python -m flashinfer show-config'..."
run_with_aot_memory_monitor "flashinfer_show_config" python -m flashinfer show-config || {
    echo "ERROR: Failed to run 'python -m flashinfer show-config'"
    exit 1
}
echo "✓ show-config completed successfully"

# Verify all modules are compiled
echo ""
echo "[STEP 2/2] Verifying all modules are compiled..."
run_with_aot_memory_monitor "verify_all_modules_compiled" python scripts/verify_all_modules_compiled.py || {
    echo "ERROR: Not all modules are compiled!"
    exit 1
}
echo "✓ All modules verified successfully"

echo ""
echo "========================================"
echo "✓✓✓ ALL TESTS PASSED! ✓✓✓"
echo "========================================"
