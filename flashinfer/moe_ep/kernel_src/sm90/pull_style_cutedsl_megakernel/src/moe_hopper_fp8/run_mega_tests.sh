#!/usr/bin/env bash
# Functional test harness for the FP8 distributed MegaMoE fused dispatch + fc1/fc2
# + combine runner.  M01 runs single-rank with MEGA_NO_DIST=1; other cases run
# through torchrun using MEGA_NPROC/MEGA_NNODES overrides when present.
#
# Hopper FP8 tile constraints (apply to ALL tests below):
#   * Only the 1CTA cluster_shape_mnk 1,1,1 path is validated.
#   * Non-swap uses M=64 and N=128/256, selected with
#     FP8_NON_SWAP_M/N (defaults 64/128).
#   * Swap-AB uses M=128/256, K=128; FP8_SWAP_AB_M selects M (default 256)
#     and FP8_SWAP_AB_N selects token N from 16/32/64/128 (default 32).
#   * hidden must be divisible by 256 (fc2 N tile)
#   * intermediate must be divisible by 128 (fc2 K tile / fc1 N tile / 2)
#   * e5m2 element format is currently broken (fc1 epilogue quant path raises
#     "None to integer conversion"); all tests below use --kind fp8_e4m3
#
# The list keeps one representative for every existing rank/scheduler/
# expert-shape/route combination, plus topk=13, alignment, and large-shape stress.
#
# Usage:
#   bash <abs path>/run_mega_tests.sh --scale-mode per-tensor
#   bash <abs path>/run_mega_tests.sh --scale-mode blockwise --swapab
#   PYTHON=python3.11 bash .../run_mega_tests.sh
#   MEGA_NPROC=4 bash .../run_mega_tests.sh
#   bash .../run_mega_tests.sh --fail-fast
#   bash .../run_mega_tests.sh --list
#   bash .../run_mega_tests.sh --help
#
# Variant selection:
#   each invocation runs one scale mode; default is per-tensor.
#   --swapab selects swap-AB; without it, the test uses non-swap.
#   FP8_ACCUM_MODE=2xacc bash ... --scale-mode per-tensor     # legacy 2xacc
#   FP8_NON_SWAP_M=64 FP8_NON_SWAP_N=128 bash .../run_mega_tests.sh M01
#   FP8_SWAP_AB_M=256 FP8_SWAP_AB_N=32 bash .../run_mega_tests.sh --swapab M01
#
# Selective execution (positional args; substring match against test names,
# OR-combined across multiple selectors):
#   bash .../run_mega_tests.sh M01
#   bash .../run_mega_tests.sh blockwise
#   bash .../run_mega_tests.sh M01 M03 M13
#   bash .../run_mega_tests.sh balanced --fail-fast
#
# Exit code: number of failed tests (0 = all pass).

export PATH=/usr/bin:$PATH
export LD=/usr/bin/ld
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export CUDAHOSTCXX=/usr/bin/g++
export TRITON_CC=/usr/bin/gcc
export CFLAGS="-B/usr/bin"
export CXXFLAGS="-B/usr/bin"
export LDFLAGS="-B/usr/bin -fuse-ld=bfd"

set -u  # fail on undefined vars; do NOT set -e (continue on failures)

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
RUNNER="${SCRIPT_DIR}/mega_runner.py"
PYTHON="${PYTHON:-python}"
SCALE_MODE="${FP8_SCALE_MODE:-per_tensor}"
SWAP_AB=0
FAIL_FAST=0
LIST_ONLY=0
declare -a SELECTORS=()
while [ "$#" -gt 0 ]; do
    case "$1" in
        --scale-mode)
            if [ "$#" -lt 2 ]; then
                echo "ERROR: --scale-mode requires per-tensor or blockwise" >&2
                exit 2
            fi
            SCALE_MODE="$2"
            shift 2
            ;;
        --swapab)
            SWAP_AB=1
            shift
            ;;
        --fail-fast)
            FAIL_FAST=1
            shift
            ;;
        --list)
            LIST_ONLY=1
            shift
            ;;
        -h|--help)
            sed -n '2,/^# Exit code/p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
            exit 0
            ;;
        --*)
            echo "Unknown flag: $1 (use --help)" >&2
            exit 2
            ;;
        *)
            SELECTORS+=("$1")
            shift
            ;;
    esac
done
case "$SCALE_MODE" in
    per-tensor|per_tensor)
        SCALE_MODE=per_tensor
        ;;
    blockwise)
        ;;
    *)
        echo "ERROR: unsupported FP8 scale mode '$SCALE_MODE'" >&2
        exit 2
        ;;
esac
FP8_ACCUM_MODE="${FP8_ACCUM_MODE:-1xacc}"
case "$FP8_ACCUM_MODE" in
    1xacc|2xacc)
        ;;
    *)
        echo "ERROR: unsupported FP8 accumulation mode '$FP8_ACCUM_MODE'" >&2
        exit 2
        ;;
esac
TILE_ARGS=()
if [ "$SWAP_AB" -eq 1 ]; then
    FP8_SWAP_AB_M="${FP8_SWAP_AB_M:-256}"
    case "$FP8_SWAP_AB_M" in
        128|256)
            ;;
        *)
            echo "ERROR: FP8_SWAP_AB_M must be one of 128,256" >&2
            exit 2
            ;;
    esac
    FP8_SWAP_AB_N="${FP8_SWAP_AB_N:-32}"
    case "$FP8_SWAP_AB_N" in
        16|32|64|128)
            ;;
        *)
            echo "ERROR: FP8_SWAP_AB_N must be one of 16,32,64,128" >&2
            exit 2
            ;;
    esac
    TILE_ARGS=(--swap_ab --mma_tiler_mnk "${FP8_SWAP_AB_M},${FP8_SWAP_AB_N},128")
else
    FP8_NON_SWAP_M="${FP8_NON_SWAP_M:-64}"
    case "$FP8_NON_SWAP_M" in
        64)
            ;;
        *)
            echo "ERROR: FP8_NON_SWAP_M must be 64" >&2
            exit 2
            ;;
    esac
    FP8_NON_SWAP_N="${FP8_NON_SWAP_N:-128}"
    case "$FP8_NON_SWAP_N" in
        128|256)
            ;;
        *)
            echo "ERROR: FP8_NON_SWAP_N must be one of 128,256" >&2
            exit 2
            ;;
    esac
    TILE_ARGS=(--mma_tiler_mnk "${FP8_NON_SWAP_M},${FP8_NON_SWAP_N},128")
fi

resolve_mode_args() {
    local scale_mode="$1"
    local raw_args="$2"
    local m18_intermediate

    if [ "$scale_mode" = "blockwise" ]; then
        # DeepGEMM-style blockwise weight scale uses 128-wide K blocks.
        m18_intermediate=2816
    else
        m18_intermediate=2688
    fi

    raw_args="${raw_args//@M18_INTERMEDIATE@/$m18_intermediate}"
    printf '%s' "$raw_args"
}

if [ ! -f "$RUNNER" ]; then
    echo "ERROR: mega_runner.py not found at ${RUNNER}" >&2
    exit 2
fi

# Resolve the multi-rank world size.  Order of precedence:
#   1. MEGA_NPROC env override
#   2. CUDA_VISIBLE_DEVICES list length
#   3. nvidia-smi visible GPU count
#   4. fallback to 2
if [ -n "${MEGA_NPROC:-}" ]; then
    NPROC="$MEGA_NPROC"
elif [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && [ "$CUDA_VISIBLE_DEVICES" != "NoDevFiles" ]; then
    IFS=',' read -r -a _VISIBLE_DEVICES <<< "$CUDA_VISIBLE_DEVICES"
    NPROC="${#_VISIBLE_DEVICES[@]}"
elif command -v nvidia-smi >/dev/null 2>&1; then
    NPROC=$(nvidia-smi --list-gpus 2>/dev/null | wc -l | tr -d '[:space:]')
    if [ -z "$NPROC" ] || [ "$NPROC" -le 0 ]; then
        NPROC=2
    fi
else
    NPROC=2
fi

MEGA_NNODES="${MEGA_NNODES:-1}"
MEGA_NODE_RANK="${MEGA_NODE_RANK:-0}"
MEGA_MASTER_ADDR="${MEGA_MASTER_ADDR:-localhost}"
MEGA_MASTER_PORT="${MEGA_MASTER_PORT:-29500}"
WORLD_SIZE=$((NPROC * MEGA_NNODES))

test_matches_selectors() {
    local name="$1"
    if [ "${#SELECTORS[@]}" -eq 0 ]; then
        return 0
    fi
    local sel
    for sel in "${SELECTORS[@]}"; do
        if [[ "$name" == *"$sel"* ]]; then
            return 0
        fi
    done
    return 1
}

# Common Hopper FP8 tile args (the only supported config):
#   --mma_tiler_mnk 64,128,128 --cluster_shape_mnk 1,1,1
# All entries below include these explicitly for clarity.

declare -a TESTS=(
    # ── M01: single-rank sanity ──
    "M01_single_balanced_topk2       | single | --kind fp8_e4m3 --num_tokens_per_rank 192  --num_topk 2  --num_total_experts 8   --hidden 1024 --intermediate 2048 --mma_tiler_mnk 64,128,128 --cluster_shape_mnk 1,1,1 --enable_static_expert_shape --ref_compute_graph transformers"

    # ── Static expert shape: branch coverage + largest stress ──
    "M02_mr_balanced_topk2_tiny      | multi  | --kind fp8_e4m3 --num_tokens_per_rank 96   --num_topk 2  --num_total_experts 8   --hidden 1024 --intermediate 1024 --mma_tiler_mnk 64,128,128 --cluster_shape_mnk 1,1,1 --enable_static_expert_shape"
    "M03_mr_balanced_topk3_atomic    | multi  | --kind fp8_e4m3 --num_tokens_per_rank 256  --num_topk 3  --num_total_experts 24  --hidden 1536 --intermediate 2048 --mma_tiler_mnk 64,128,128 --cluster_shape_mnk 1,1,1 --enable_static_expert_shape --load_balance_mode atomic_counter"
    "M09_mr_power_law_topk13         | multi  | --kind fp8_e4m3 --num_tokens_per_rank 832  --num_topk 13 --num_total_experts 104 --hidden 2560 --intermediate 4096 --mma_tiler_mnk 64,128,128 --cluster_shape_mnk 1,1,1 --enable_static_expert_shape --route_distribution power_law"
    "M11_mr_large_topk11_atomic      | multi  | --kind fp8_e4m3 --num_tokens_per_rank 1792 --num_topk 11 --num_total_experts 88  --hidden 2816 --intermediate 4096 --mma_tiler_mnk 64,128,128 --cluster_shape_mnk 1,1,1 --enable_static_expert_shape --load_balance_mode atomic_counter"
    "M13_mr_atomic_pl_topk7         | multi  | --kind fp8_e4m3 --num_tokens_per_rank 512  --num_topk 7  --num_total_experts 56  --hidden 1792 --intermediate 1792 --mma_tiler_mnk 64,128,128 --cluster_shape_mnk 1,1,1 --enable_static_expert_shape --load_balance_mode atomic_counter --route_distribution power_law"

    # ── Dynamic expert shape: retain both existing scheduler/route combinations ──
    "M17_mr_dynshape_balanced_topk4  | multi  | --kind fp8_e4m3 --num_tokens_per_rank 384  --num_topk 4  --num_total_experts 32  --hidden 2048 --intermediate 2048 --mma_tiler_mnk 64,128,128 --cluster_shape_mnk 1,1,1"
    "M18_mr_dynshape_pl_topk7        | multi  | --kind fp8_e4m3 --num_tokens_per_rank 576  --num_topk 7  --num_total_experts 64  --hidden 1536 --intermediate @M18_INTERMEDIATE@ --mma_tiler_mnk 64,128,128 --cluster_shape_mnk 1,1,1 --route_distribution power_law --load_balance_mode atomic_counter"

)

PASS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0
declare -a FAIL_NAMES=()
TOTAL=${#TESTS[@]}
START_TIME=$SECONDS

if [ "$LIST_ONLY" -eq 1 ]; then
    for entry in "${TESTS[@]}"; do
        name="${entry%%|*}"
        name="${name%"${name##*[![:space:]]}"}"
        full_name="${SCALE_MODE}/${name}"
        if test_matches_selectors "$full_name"; then
            echo "$full_name"
        fi
    done
    exit 0
fi

echo "==========================================================================="
echo "MegaMoE FP8 functional tests"
echo "  RUNNER : ${RUNNER}"
echo "  PYTHON : ${PYTHON}"
echo "  NPROC  : ${NPROC}"
echo "  NNODES : ${MEGA_NNODES}"
echo "  WORLD  : ${WORLD_SIZE}"
echo "  MODE   : ${SCALE_MODE}"
if [ "$MEGA_NNODES" -gt 1 ]; then
    echo "  NODE   : rank ${MEGA_NODE_RANK}, master ${MEGA_MASTER_ADDR}:${MEGA_MASTER_PORT}"
fi
echo "  TOTAL  : ${TOTAL} tests"
if [ "${#SELECTORS[@]}" -gt 0 ]; then
    echo "  FILTER : ${SELECTORS[*]}"
fi
echo "==========================================================================="

for entry in "${TESTS[@]}"; do
        # Parse "name | mode | args".  Strip whitespace around each segment.
        name="${entry%%|*}"
        name="${name%"${name##*[![:space:]]}"}"
        rest="${entry#*|}"
        launch_mode="${rest%%|*}"
        launch_mode="${launch_mode#"${launch_mode%%[![:space:]]*}"}"
        launch_mode="${launch_mode%"${launch_mode##*[![:space:]]}"}"
        raw_args="${rest#*|}"
        raw_args="${raw_args#"${raw_args%%[![:space:]]*}"}"
        args="$(resolve_mode_args "$SCALE_MODE" "$raw_args")"
        full_name="${SCALE_MODE}/${name}"

        if ! test_matches_selectors "$full_name"; then
            SKIP_COUNT=$((SKIP_COUNT + 1))
            continue
        fi

        echo
        echo "==========================================================================="
        echo "[TEST] $full_name"
        echo "[MODE] $SCALE_MODE"
        case "$launch_mode" in
            single)
                echo "[CMD]  MEGA_NO_DIST=1 $PYTHON $RUNNER $args --fp8_scale_mode $SCALE_MODE --fp8_accum_mode $FP8_ACCUM_MODE ${TILE_ARGS[*]}"
                ;;
            multi)
                if [ "$MEGA_NNODES" -gt 1 ]; then
                    echo "[CMD]  torchrun --nnodes=$MEGA_NNODES --node_rank=$MEGA_NODE_RANK --nproc_per_node=$NPROC --master_addr=$MEGA_MASTER_ADDR --master_port=$MEGA_MASTER_PORT $RUNNER $args --fp8_scale_mode $SCALE_MODE --fp8_accum_mode $FP8_ACCUM_MODE ${TILE_ARGS[*]}"
                else
                    echo "[CMD]  torchrun --nproc_per_node=$NPROC $RUNNER $args --fp8_scale_mode $SCALE_MODE --fp8_accum_mode $FP8_ACCUM_MODE ${TILE_ARGS[*]}"
                fi
                ;;
            *)
                echo "[ERROR] unknown launch mode '$launch_mode' for test '$full_name'; skipping" >&2
                FAIL_COUNT=$((FAIL_COUNT + 1))
                FAIL_NAMES+=("$full_name (bad-mode)")
                continue
                ;;
        esac
        echo "==========================================================================="

        test_start=$SECONDS
        # shellcheck disable=SC2086  # intentional word-splitting on $args
        case "$launch_mode" in
            single)
                timeout 300 env MEGA_NO_DIST=1 "$PYTHON" "$RUNNER" $args --fp8_scale_mode "$SCALE_MODE" --fp8_accum_mode "$FP8_ACCUM_MODE" "${TILE_ARGS[@]}"
                rc=$?
                ;;
            multi)
                if [ "$MEGA_NNODES" -gt 1 ]; then
                    timeout 300 torchrun \
                        --nnodes="$MEGA_NNODES" --node_rank="$MEGA_NODE_RANK" \
                        --nproc_per_node="$NPROC" \
                        --master_addr="$MEGA_MASTER_ADDR" --master_port="$MEGA_MASTER_PORT" \
                        "$RUNNER" $args --fp8_scale_mode "$SCALE_MODE" --fp8_accum_mode "$FP8_ACCUM_MODE" "${TILE_ARGS[@]}"
                else
                    timeout 300 torchrun --nproc_per_node="$NPROC" "$RUNNER" $args --fp8_scale_mode "$SCALE_MODE" --fp8_accum_mode "$FP8_ACCUM_MODE" "${TILE_ARGS[@]}"
                fi
                rc=$?
                ;;
        esac
        if [ "$rc" -eq 124 ]; then
            echo "[TIMEOUT] test exceeded 300s limit — killed"
        fi
        elapsed=$((SECONDS - test_start))

        if [ "$rc" -eq 0 ]; then
            echo "[RESULT] PASS  (${elapsed}s) $full_name"
            PASS_COUNT=$((PASS_COUNT + 1))
        else
            echo "[RESULT] FAIL  (rc=${rc}, ${elapsed}s) $full_name"
            FAIL_COUNT=$((FAIL_COUNT + 1))
            FAIL_NAMES+=("$full_name")
            if [ "$FAIL_FAST" -eq 1 ]; then
                echo "[--fail-fast] aborting after first failure"
                break
            fi
        fi
done

TOTAL_ELAPSED=$((SECONDS - START_TIME))
RAN_COUNT=$((PASS_COUNT + FAIL_COUNT))
echo
echo "==========================================================================="
if [ "${#SELECTORS[@]}" -gt 0 ]; then
    echo "SUMMARY: ${PASS_COUNT}/${RAN_COUNT} passed, ${FAIL_COUNT} failed, ${SKIP_COUNT} skipped (mode: ${SCALE_MODE}; selectors: ${SELECTORS[*]}; wallclock ${TOTAL_ELAPSED}s)"
else
    echo "SUMMARY: ${PASS_COUNT}/${TOTAL} passed, ${FAIL_COUNT} failed (mode: ${SCALE_MODE}; wallclock ${TOTAL_ELAPSED}s)"
fi
echo "==========================================================================="
if [ "$FAIL_COUNT" -gt 0 ]; then
    echo "Failed tests:"
    for name in "${FAIL_NAMES[@]}"; do
        echo "  - $name"
    done
fi
if [ "${#SELECTORS[@]}" -gt 0 ] && [ "$RAN_COUNT" -eq 0 ]; then
    echo "WARNING: selectors matched 0 tests (use --list to see all available test names)"
fi

exit "$FAIL_COUNT"
