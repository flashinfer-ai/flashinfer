#!/bin/bash

set -eo pipefail
set -x

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Source test environment setup (handles package overrides like TVM-FFI)
source "${SCRIPT_DIR}/setup_test_env.sh"

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

# Parallelism: default back to one nvcc worker per compile and let ninja provide
# build-level parallelism through MAX_JOBS.
MEM_AVAILABLE_GB=$(free -g | awk '/^Mem:/ {print $7}')
NPROC=$(nproc)

NVCC_THREADS=${FLASHINFER_NVCC_THREADS:-1}
if ! [[ "$NVCC_THREADS" =~ ^[1-9][0-9]*$ ]]; then
  echo "Invalid FLASHINFER_NVCC_THREADS=${NVCC_THREADS}; using 1"
  NVCC_THREADS=1
fi
if (( NVCC_THREADS > 8 )); then NVCC_THREADS=8; fi
if (( NVCC_THREADS > NPROC )); then NVCC_THREADS=${NPROC}; fi
if (( NVCC_THREADS < 1 )); then NVCC_THREADS=1; fi

# Default to the larger of the historical 8GB/job baseline and ~2GB per nvcc
# thread when callers explicitly opt into higher nvcc threading.
# AOT_MAX_JOBS_MEMORY_GB can override this per-job budget.
ARCH_MEMORY_BUDGET_GB=8
THREAD_MEMORY_BUDGET_GB=$(( NVCC_THREADS * 2 ))
if (( THREAD_MEMORY_BUDGET_GB > ARCH_MEMORY_BUDGET_GB )); then
  DEFAULT_AOT_MAX_JOBS_MEMORY_GB=${THREAD_MEMORY_BUDGET_GB}
else
  DEFAULT_AOT_MAX_JOBS_MEMORY_GB=${ARCH_MEMORY_BUDGET_GB}
fi
: "${AOT_MAX_JOBS_MEMORY_GB:=${DEFAULT_AOT_MAX_JOBS_MEMORY_GB}}"
: "${AOT_MAX_JOBS_CAP:=0}"
if ! [[ "$AOT_MAX_JOBS_MEMORY_GB" =~ ^[1-9][0-9]*$ ]]; then
  echo "Invalid AOT_MAX_JOBS_MEMORY_GB=${AOT_MAX_JOBS_MEMORY_GB}; using ${DEFAULT_AOT_MAX_JOBS_MEMORY_GB}"
  AOT_MAX_JOBS_MEMORY_GB=${DEFAULT_AOT_MAX_JOBS_MEMORY_GB}
fi
if ! [[ "$AOT_MAX_JOBS_CAP" =~ ^[0-9]+$ ]]; then
  echo "Invalid AOT_MAX_JOBS_CAP=${AOT_MAX_JOBS_CAP}; disabling cap"
  AOT_MAX_JOBS_CAP=0
fi

MAX_JOBS=$(( MEM_AVAILABLE_GB / AOT_MAX_JOBS_MEMORY_GB ))
if (( MAX_JOBS < 1 )); then
  MAX_JOBS=1
elif (( NPROC < MAX_JOBS )); then
  MAX_JOBS=$NPROC
fi
if (( AOT_MAX_JOBS_CAP > 0 && MAX_JOBS > AOT_MAX_JOBS_CAP )); then
  MAX_JOBS=$AOT_MAX_JOBS_CAP
fi

# Cap total threads at available CPUs
TOTAL_THREADS=$(( MAX_JOBS * NVCC_THREADS ))
if (( TOTAL_THREADS > NPROC )); then
  MAX_JOBS=$(( NPROC / NVCC_THREADS ))
  if (( MAX_JOBS < 1 )); then MAX_JOBS=1; fi
fi

export MAX_JOBS
export FLASHINFER_NVCC_THREADS="${NVCC_THREADS}"

echo "System Information:"
echo "  - Available Memory: ${MEM_AVAILABLE_GB} GB"
echo "  - Number of Processors: ${NPROC}"
echo "  - AOT MAX_JOBS Memory Budget: ${AOT_MAX_JOBS_MEMORY_GB} GB/job"
if (( AOT_MAX_JOBS_CAP > 0 )); then
  echo "  - AOT MAX_JOBS Cap: ${AOT_MAX_JOBS_CAP}"
fi
echo "  - MAX_JOBS: ${MAX_JOBS}"
echo "  - NVCC_THREADS: ${NVCC_THREADS}"

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
SCCACHE_BUCKET="${SCCACHE_BUCKET:-flashinfer-build-cache}"
SCCACHE_REGION="${SCCACHE_REGION:-us-west-2}"

echo ""
echo "========================================"
echo "Setting up sccache"
echo "========================================"

install_sccache() {
  local sccache_version=$1
  local sccache_arch=$2
  local sccache_package="sccache-v${sccache_version}-${sccache_arch}-unknown-linux-musl"
  local sccache_archive="${sccache_package}.tar.gz"
  local sccache_url="https://github.com/mozilla/sccache/releases/download/v${sccache_version}/${sccache_archive}"
  local sccache_tmpdir
  local sccache_sha256

  if ! command -v sha256sum >/dev/null 2>&1; then
    echo "ERROR: sha256sum is required to verify sccache downloads"
    exit 1
  fi

  sccache_tmpdir=$(mktemp -d)
  curl -fsSL "${sccache_url}" -o "${sccache_tmpdir}/${sccache_archive}"
  curl -fsSL "${sccache_url}.sha256" -o "${sccache_tmpdir}/${sccache_archive}.sha256"
  sccache_sha256=$(awk '{print $1}' "${sccache_tmpdir}/${sccache_archive}.sha256")
  if [ -z "${sccache_sha256}" ]; then
    echo "ERROR: Missing checksum for ${sccache_archive}"
    exit 1
  fi

  printf '%s  %s\n' "${sccache_sha256}" "${sccache_tmpdir}/${sccache_archive}" | sha256sum -c -
  tar xzf "${sccache_tmpdir}/${sccache_archive}" -C "${sccache_tmpdir}"
  mv "${sccache_tmpdir}/${sccache_package}/sccache" /usr/local/bin/
  rm -rf "${sccache_tmpdir}"
  chmod +x /usr/local/bin/sccache
}

SCCACHE_VERSION="0.9.1"
SCCACHE_ARCH=$(uname -m)
install_sccache "${SCCACHE_VERSION}" "${SCCACHE_ARCH}"

export SCCACHE_BUCKET
export SCCACHE_REGION
SCCACHE_SOURCE_ROOT=$(pwd -P)
export SCCACHE_BASEDIRS="${SCCACHE_SOURCE_ROOT}${SCCACHE_BASEDIRS:+:${SCCACHE_BASEDIRS}}"
export SCCACHE_S3_KEY_PREFIX="cuda${CUDA_VERSION}-${SCCACHE_ARCH}"
export SCCACHE_IDLE_TIMEOUT=0
export FLASHINFER_NVCC_LAUNCHER="sccache"
export FLASHINFER_CXX_LAUNCHER="sccache"

# If no complete AWS credential pair is available, use anonymous read-only access
# to the public bucket.
set +x
if [ -z "${AWS_ACCESS_KEY_ID:-}" ] || [ -z "${AWS_SECRET_ACCESS_KEY:-}" ]; then
  export SCCACHE_S3_NO_CREDENTIALS=true
  echo "sccache mode: read-only (public bucket, no credentials)"
else
  unset SCCACHE_S3_NO_CREDENTIALS
  echo "sccache mode: read-write"
fi
set -x

sccache --start-server
echo "sccache version: $(sccache --version)"
echo "sccache bucket: ${SCCACHE_BUCKET}"
echo "sccache region: ${SCCACHE_REGION}"
echo "sccache prefix: ${SCCACHE_S3_KEY_PREFIX}"
echo "sccache basedirs: ${SCCACHE_BASEDIRS}"

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
