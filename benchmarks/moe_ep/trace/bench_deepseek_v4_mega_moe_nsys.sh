#!/usr/bin/env bash
# Nsight Systems profiling for DeepSeek V4 MegaMoE EP trace benchmarks
# (flashinfer.moe_ep_v2 vs vLLM DeepseekV4MegaMoEExperts, NVTX-instrumented).
#
# Run from the flashinfer repo root, inside the nsys-capable container:
#   ./open_docker.sh
#   cd /lustre/fsw/coreai_libraries_cudnn/mhoqueanik/flashinfer
#
# Examples:
#   bash benchmarks/moe_ep/trace/bench_deepseek_v4_mega_moe_nsys.sh nvtx both
#   COLD_START=off bash benchmarks/moe_ep/trace/bench_deepseek_v4_mega_moe_nsys.sh full both
#   bash benchmarks/moe_ep/trace/bench_deepseek_v4_mega_moe_nsys.sh full both --no-cold-start
#   bash benchmarks/moe_ep/trace/bench_deepseek_v4_mega_moe_nsys.sh stats both
#   bash benchmarks/moe_ep/trace/bench_deepseek_v4_mega_moe_nsys.sh gui vllm
#
# Environment overrides (optional):
#   NPROC_PER_NODE=4 NUM_TOKENS=8192 WARMUP=5 REPEAT=20
#   NVTX_CAPTURE_RANGE=cudaProfilerApi|nvtx
#   NVTX_CAPTURE_SPEC=@steady_capture   (only for nvtx capture-range)
#   OUTPUT_DIR=/path/to/profiles
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

MODE="${1:-}"
shift || true

NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
NUM_TOKENS="${NUM_TOKENS:-8192}"
NUM_MAX_TOKENS="${NUM_MAX_TOKENS:-8192}"
WARMUP="${WARMUP:-5}"
REPEAT="${REPEAT:-20}"
COLD_START="${COLD_START:-on}"
NSYS_DELAY_MS="${NSYS_DELAY_MS:-2000}"
NSYS_DURATION_MS="${NSYS_DURATION_MS:-100}"
NVTX_CAPTURE_RANGE="${NVTX_CAPTURE_RANGE:-cudaProfilerApi}"
NVTX_CAPTURE_SPEC="${NVTX_CAPTURE_SPEC:-@steady_capture}"
NSYS_TRACE_FORK="${NSYS_TRACE_FORK:-true}"
OUTPUT_DIR="${OUTPUT_DIR:-/lustre/fsw/coreai_libraries_cudnn/mhoqueanik/profiles}"
STATS_CAPTURE_MODE="${STATS_CAPTURE_MODE:-nvtx}"

BACKEND=both
BENCH_EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        flashinfer|vllm|both)
            BACKEND="$1"
            shift
            ;;
        full|nvtx|steady)
            if [[ "${MODE}" == "stats" || "${MODE}" == "gui" ]]; then
                STATS_CAPTURE_MODE="$1"
                shift
            else
                echo "Unknown argument: $1" >&2
                usage
                exit 1
            fi
            ;;
        --no-cold-start)
            COLD_START=off
            shift
            ;;
        --cold-start)
            COLD_START=on
            shift
            ;;
        -h|--help|help)
            usage
            exit 0
            ;;
        -*)
            BENCH_EXTRA_ARGS+=("$1")
            shift
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage
            exit 1
            ;;
    esac
done
if [[ ${#BENCH_EXTRA_ARGS[@]} -gt 0 ]]; then
    set -- "${BENCH_EXTRA_ARGS[@]}"
else
    set --
fi

TRACE_DIR="benchmarks/moe_ep/trace"
FLASHINFER_BENCH="${TRACE_DIR}/bench_deepseek_v4_mega_moe_flashinfer_trace.py"
VLLM_BENCH="${TRACE_DIR}/bench_deepseek_v4_mega_moe_vLLM_trace.py"

usage() {
    cat <<'EOF'
Usage: bench_deepseek_v4_mega_moe_nsys.sh <mode> [backend] [options...]

Modes:
  nvtx     Capture steady_capture via cudaProfilerStart (default) or NVTX
  full     Full timeline: setup + cold start + warmup + steady repeats
  steady   Legacy delay/duration window (no NVTX trigger)
  stats    Export nsys stats CSVs for existing .nsys-rep files
           Usage: stats [full|nvtx] [backend]   (default: nvtx)
  gui      Open a report in nsys-ui (requires GUI / X11)
           Usage: gui [full|nvtx] <backend>
  all      Run nvtx captures for both backends, then stats

Backends:
  flashinfer | vllm | both   (default: both; can appear in any order)

Script options (also accepted by the benchmark):
  --no-cold-start   Skip the first-forward cold-start pass
  --cold-start      Include cold-start pass (default)

Other flags (e.g. --warmup 3) are forwarded to the trace benchmarks.

Env:
  NPROC_PER_NODE    GPUs per node (default: 4)
  NUM_TOKENS        Token count (default: 8192)
  NUM_MAX_TOKENS    Max tokens / kernel sizing (default: 8192)
  WARMUP            Warmup iterations (default: 5)
  REPEAT            Timed iterations (default: 20)
  COLD_START        on|off for first-forward timing (default: on)
  NVTX_CAPTURE_RANGE nsys capture trigger: cudaProfilerApi (default) or nvtx
  NVTX_CAPTURE_SPEC  domain@range when NVTX_CAPTURE_RANGE=nvtx
                     (default: @steady_capture)
  NSYS_TRACE_FORK   trace torchrun workers (default: true)
  NSYS_DELAY_MS     steady-mode capture delay in ms (default: 2000)
  NSYS_DURATION_MS  steady-mode capture duration in ms (default: 100)
  OUTPUT_DIR        Profile output directory
  MASTER_ADDR       torchrun master (default: $MASTER_ADDR, else SLURM
                    if scontrol works, else hostname)
  MASTER_PORT       torchrun port (default: 29500)
EOF
}

resolve_master_addr() {
    if [[ -n "${MASTER_ADDR:-}" ]]; then
        export MASTER_ADDR
        return
    fi
    # scontrol is often bind-mounted into the container but non-functional
    # (missing Slurm plugins). Only use it when it actually works.
    if [[ -n "${SLURM_NODELIST:-}" ]] && command -v scontrol >/dev/null 2>&1; then
        local slurm_master=""
        slurm_master="$(
            scontrol show hostnames "${SLURM_NODELIST}" 2>/dev/null | head -n1
        )" || true
        if [[ -n "${slurm_master}" ]]; then
            MASTER_ADDR="${slurm_master}"
            export MASTER_ADDR
            return
        fi
    fi
    MASTER_ADDR="$(hostname -f 2>/dev/null || hostname)"
    export MASTER_ADDR
}

cold_start_flag() {
    if [[ "${COLD_START}" == "off" || "${COLD_START}" == "0" || "${COLD_START}" == "false" ]]; then
        echo --no-cold-start
    else
        echo --cold-start
    fi
}

bench_has_cold_start_flag() {
    local arg
    for arg in "$@"; do
        case "${arg}" in
            --no-cold-start|--cold-start)
                return 0
                ;;
        esac
    done
    return 1
}

default_bench_args() {
    DEFAULT_BENCH_ARGS=(
        --num-tokens "${NUM_TOKENS}"
        --num-max-tokens "${NUM_MAX_TOKENS}"
        --warmup "${WARMUP}"
        --repeat "${REPEAT}"
    )
    if ! bench_has_cold_start_flag "$@"; then
        DEFAULT_BENCH_ARGS+=("$(cold_start_flag)")
    fi
}

nsys_base() {
    local -a fork_args=()
    if [[ "${NSYS_TRACE_FORK}" == "true" || "${NSYS_TRACE_FORK}" == "1" ]]; then
        fork_args+=(--trace-fork-before-exec=true)
    fi
    nsys profile \
        --trace=cuda,nvtx,osrt,cublas,cudnn,nccl \
        --sample=none \
        --cpuctxsw=none \
        --cuda-memory-usage=true \
        --cuda-graph-trace=node \
        --force-overwrite=true \
        "${fork_args[@]}" \
        "$@"
}

profile_backend() {
    local capture_mode="$1"
    local backend="$2"
    shift 2

    local bench_script tag output_base
    case "${backend}" in
        flashinfer)
            bench_script="${FLASHINFER_BENCH}"
            tag="flashinfer_moe_ep"
            ;;
        vllm)
            bench_script="${VLLM_BENCH}"
            tag="vllm_moe_ep"
            ;;
        *)
            echo "Unknown backend: ${backend}" >&2
            exit 1
            ;;
    esac

    output_base="${OUTPUT_DIR}/${tag}_ws${NPROC_PER_NODE}_t${NUM_TOKENS}_${capture_mode}"
    mkdir -p "${OUTPUT_DIR}"

    resolve_master_addr
    export MASTER_PORT="${MASTER_PORT:-29500}"

    local -a nsys_args=(--output="${output_base}")
    case "${capture_mode}" in
        nvtx)
            case "${NVTX_CAPTURE_RANGE}" in
                cudaProfilerApi)
                    nsys_args+=(
                        --capture-range=cudaProfilerApi
                        --capture-range-end=stop
                    )
                    ;;
                nvtx)
                    nsys_args+=(
                        --capture-range=nvtx
                        --capture-range-end=stop
                        --nvtx-capture="${NVTX_CAPTURE_SPEC}"
                    )
                    ;;
                *)
                    echo "Unknown NVTX_CAPTURE_RANGE: ${NVTX_CAPTURE_RANGE}" >&2
                    exit 1
                    ;;
            esac
            ;;
        steady)
            nsys_args+=(
                --capture-range=none
                --delay="${NSYS_DELAY_MS}"
                --duration="${NSYS_DURATION_MS}"
            )
            ;;
        full)
            nsys_args+=(--capture-range=none)
            ;;
        *)
            echo "Unknown capture mode: ${capture_mode}" >&2
            exit 1
            ;;
    esac

    echo "=== nsys ${capture_mode}: ${backend} ==="
    echo "    output: ${output_base}.nsys-rep"
    echo "    bench:  ${bench_script}"
    echo "    MASTER_ADDR=${MASTER_ADDR} NPROC_PER_NODE=${NPROC_PER_NODE}"
    if [[ "${capture_mode}" == "nvtx" ]]; then
        echo "    capture_range=${NVTX_CAPTURE_RANGE}"
        if [[ "${NVTX_CAPTURE_RANGE}" == "nvtx" ]]; then
            echo "    nvtx_capture=${NVTX_CAPTURE_SPEC}"
        fi
    fi

    default_bench_args "$@"
    nsys_base "${nsys_args[@]}" \
        torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
            "${bench_script}" \
            "${DEFAULT_BENCH_ARGS[@]}" "$@"

    if [[ ! -f "${output_base}.nsys-rep" ]]; then
        echo "WARNING: no report written to ${output_base}.nsys-rep" >&2
        echo "         try full capture: bash $0 full ${backend}" >&2
        return 1
    fi
}

stats_for_backend() {
    local capture_mode="$1"
    local backend="$2"

    local tag
    case "${backend}" in
        flashinfer) tag="flashinfer_moe_ep" ;;
        vllm) tag="vllm_moe_ep" ;;
        *)
            echo "Unknown backend: ${backend}" >&2
            exit 1
            ;;
    esac

    local rep="${OUTPUT_DIR}/${tag}_ws${NPROC_PER_NODE}_t${NUM_TOKENS}_${capture_mode}.nsys-rep"
    local out_prefix="${OUTPUT_DIR}/${tag}_ws${NPROC_PER_NODE}_t${NUM_TOKENS}_${capture_mode}_stats"

    if [[ ! -f "${rep}" ]]; then
        echo "Missing report: ${rep}" >&2
        return 1
    fi

    echo "=== nsys stats: ${backend} (${capture_mode}) ==="
    nsys stats \
        --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_api_sum,osrt_sum,nvtx_sum \
        --format csv \
        --output "${out_prefix}" \
        "${rep}"

    echo "    CSV prefix: ${out_prefix}"
    echo
    nsys stats --report nvtx_sum --format table "${rep}"
    echo
    nsys stats --report cuda_gpu_kern_sum --format table "${rep}"
    echo
    nsys stats --report cuda_api_sum --format table "${rep}"
    echo
    if ! nsys stats --report nccl_sum --format table "${rep}" 2>/dev/null; then
        echo "    (nccl_sum not available in this nsys build or no NCCL trace data)"
    fi
    echo
}

gui_for_backend() {
    local capture_mode="${1:-nvtx}"
    local backend="$2"

    local tag
    case "${backend}" in
        flashinfer) tag="flashinfer_moe_ep" ;;
        vllm) tag="vllm_moe_ep" ;;
        *)
            echo "Unknown backend: ${backend}" >&2
            exit 1
            ;;
    esac

    local rep="${OUTPUT_DIR}/${tag}_ws${NPROC_PER_NODE}_t${NUM_TOKENS}_${capture_mode}.nsys-rep"
    if [[ ! -f "${rep}" ]]; then
        echo "Missing report: ${rep}" >&2
        exit 1
    fi
    exec nsys-ui "${rep}"
}

run_for_backends() {
    local fn="$1"
    shift
    case "${BACKEND}" in
        flashinfer|vllm)
            "${fn}" "$@" "${BACKEND}"
            ;;
        both)
            "${fn}" "$@" flashinfer
            "${fn}" "$@" vllm
            ;;
        *)
            echo "Unknown backend: ${BACKEND}" >&2
            usage
            exit 1
            ;;
    esac
}

if [[ -z "${MODE}" ]]; then
    usage
    exit 1
fi

case "${MODE}" in
    nvtx|full|steady)
        run_for_backends profile_backend "${MODE}" "$@"
        ;;
    stats)
        run_for_backends stats_for_backend "${STATS_CAPTURE_MODE}"
        ;;
    gui)
        if [[ "${BACKEND}" == "both" ]]; then
            echo "Pick one backend for gui: flashinfer or vllm" >&2
            exit 1
        fi
        gui_for_backend "${STATS_CAPTURE_MODE}" "${BACKEND}"
        ;;
    all)
        run_for_backends profile_backend nvtx "$@"
        run_for_backends stats_for_backend nvtx
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        echo "Unknown mode: ${MODE}" >&2
        usage
        exit 1
        ;;
esac
