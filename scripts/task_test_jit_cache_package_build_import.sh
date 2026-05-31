#!/bin/bash

set -eo pipefail
set -x

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Source test environment setup (handles package overrides like TVM-FFI)
source "${SCRIPT_DIR}/setup_test_env.sh"
# shellcheck source=scripts/jit_cache_build_common.sh
source "${SCRIPT_DIR}/jit_cache_build_common.sh"

: "${AOT_MEMORY_MONITOR:=true}"
: "${AOT_MEMORY_MONITOR_INTERVAL:=2}"
: "${AOT_MEMORY_LOG_INTERVAL:=60}"
: "${AOT_MEMORY_REPORT_DIR:=aot-memory-reports}"

AOT_MEMORY_MONITOR_SCRIPT="${SCRIPT_DIR}/aot_memory_monitor.py"
python3 "${AOT_MEMORY_MONITOR_SCRIPT}" validate-config \
    --interval "${AOT_MEMORY_MONITOR_INTERVAL}" \
    --log-interval "${AOT_MEMORY_LOG_INTERVAL}"

AOT_MEMORY_REPORT_FILES=()
LAST_AOT_MEMORY_MONITOR_PID=""
LAST_AOT_MEMORY_REPORT=""

mkdir -p "${AOT_MEMORY_REPORT_DIR}"
AOT_MEMORY_REPORT_DIR=$(cd "${AOT_MEMORY_REPORT_DIR}" && pwd)

start_aot_memory_monitor() {
    local root_pid=$1
    local label=$2
    local safe_label
    safe_label=$(python3 "${AOT_MEMORY_MONITOR_SCRIPT}" safe-label "$label")
    local report_file="${AOT_MEMORY_REPORT_DIR}/${safe_label}.memory.csv"

    LAST_AOT_MEMORY_MONITOR_PID=""
    LAST_AOT_MEMORY_REPORT="$report_file"
    AOT_MEMORY_REPORT_FILES+=("$report_file")

    if [ "$AOT_MEMORY_MONITOR" != "true" ]; then
        return
    fi

    python3 "${AOT_MEMORY_MONITOR_SCRIPT}" monitor \
        --pid "$root_pid" \
        --label "$label" \
        --report "$report_file" \
        --interval "${AOT_MEMORY_MONITOR_INTERVAL}" \
        --log-interval "${AOT_MEMORY_LOG_INTERVAL}" &

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

    python3 "${AOT_MEMORY_MONITOR_SCRIPT}" summary \
        --label "$label" \
        --report "$report_file" || true
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
    python3 "${AOT_MEMORY_MONITOR_SCRIPT}" diagnostics || true
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
import os
import platform

import torch
cuda_ver = torch.version.cuda
arches = ["7.5", "8.0", "8.9", "9.0a"]
if cuda_ver is not None:
    try:
        major, minor = map(int, cuda_ver.split(".")[:2])
        if (major, minor) >= (13, 0):
            arches.append("10.0a")
            arches.append("10.3a")
            machine = (os.environ.get("ARCH") or platform.machine()).lower()
            if machine in ("aarch64", "arm64"):
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

compute_jit_cache_parallelism

echo "System Information:"
echo "  - Available Memory: $(free -g | awk '/^Mem:/ {print $7}') GB"
echo "  - Number of Processors: $(nproc)"
echo "  - AOT MAX_JOBS Memory Budget: ${MEM_PER_JOB} GB/job"
if (( ${AOT_MAX_JOBS_CAP:-0} > 0 )); then
  echo "  - AOT MAX_JOBS Cap: ${AOT_MAX_JOBS_CAP}"
fi
echo "  - MAX_JOBS: ${MAX_JOBS}"
echo "  - NVCC_THREADS: ${FLASHINFER_NVCC_THREADS}"

echo ""
echo "========================================"
echo "Installing flashinfer package"
echo "========================================"
run_with_aot_memory_monitor "pip_install_flashinfer_editable" \
    pip install -e . -v || {
    echo "ERROR: Failed to install flashinfer package"
    exit 1
}
echo "✓ Flashinfer package installed successfully"

# Set up sccache for compiler caching with S3 backend.
# Uses read-write mode when AWS credentials are available (nightly/release builds),
# otherwise falls back to read-only anonymous access to the public cache bucket.
echo ""
echo "========================================"
echo "Setting up sccache"
echo "========================================"

export SCCACHE_BUCKET="${SCCACHE_BUCKET:-flashinfer-build-cache}"
setup_sccache "cuda${CUDA_VERSION}-$(uname -m)" "$(pwd -P)"

echo ""
echo "========================================"
echo "Building flashinfer-jit-cache wheel"
echo "========================================"
cd flashinfer-jit-cache
rm -rf dist build *.egg-info
run_with_aot_memory_monitor "build_flashinfer_jit_cache_wheel" \
    python -m build --wheel

# Get the built wheel file
WHEEL_FILE=$(ls -t dist/*.whl | head -n 1)
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
echo "sccache stats"
echo "========================================"
sccache --show-stats

echo ""
echo "========================================"
echo "✓✓✓ ALL TESTS PASSED! ✓✓✓"
echo "========================================"
