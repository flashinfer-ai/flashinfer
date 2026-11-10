#!/usr/bin/env bash
# Functional test harness for the BF16 distributed MegaMoE fused dispatch + fc1/fc2
# + combine runner.  M01 runs single-rank with MEGA_NO_DIST=1; other cases run
# through torchrun using MEGA_NPROC/MEGA_NNODES overrides when present.
#
# Hopper BF16 tile constraints (apply to ALL tests below):
#   * Cluster 1,1,1 / 2,1,1 / 1,2,1 / 2,2,1 is selected through
#     BF16_CLUSTER_SHAPE (default 1,1,1).
#   * Non-swap uses M=64 and N=128/256, selected with
#     BF16_NON_SWAP_M/N (defaults 64/128).
#   * Swap-AB uses M=128/256, K=128; BF16_SWAP_AB_M selects M (default 256)
#     and BF16_SWAP_AB_N selects token N from 8/16/32/64/128 (default 32; 8 is experimental).
#   * hidden must be divisible by 256 (fc2 N tile)
#   * intermediate must be divisible by 128 (fc2 K tile / fc1 N tile / 2)
#
# The list keeps one representative for every existing rank/scheduler/
# expert-shape/route combination, plus topk=13, alignment, and large-shape stress.
#
# Usage:
#   bash <abs path>/run_mega_tests.sh
#   bash <abs path>/run_mega_tests.sh --swapab
#   bash <abs path>/run_mega_tests.sh --pingpong
#   PYTHON=python3.11 bash .../run_mega_tests.sh
#   MEGA_NPROC=4 bash .../run_mega_tests.sh
#   bash .../run_mega_tests.sh --fail-fast
#   bash .../run_mega_tests.sh --list
#   bash .../run_mega_tests.sh --help
#
# Variant selection:
#   Activations and weights are BF16; there is no quantization knob.
#   --swapab selects swap-AB; without it, the test uses non-swap.
#   --pingpong alternates complete tasks across two WGMMA+epilogue warpgroups.
#   BF16_NON_SWAP_M=64 BF16_NON_SWAP_N=128 bash .../run_mega_tests.sh M01
#   BF16_SWAP_AB_M=256 BF16_SWAP_AB_N=32 bash .../run_mega_tests.sh --swapab M01
#   BF16_TAIL_SPLIT=1 appends --tail_split_pairs (needs a 2-CTA token cluster:
#   BF16_CLUSTER_SHAPE=1,2,1 with --swapab or 2,1,1 without), e.g.
#   BF16_TAIL_SPLIT=1 BF16_CLUSTER_SHAPE=2,1,1 BF16_NON_SWAP_N=256 bash .../run_mega_tests.sh

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
SWAP_AB=0
PINGPONG=0
FAIL_FAST=0
LIST_ONLY=0
declare -a SELECTORS=()
while [ "$#" -gt 0 ]; do
    case "$1" in
        --swapab)
            SWAP_AB=1
            shift
            ;;
        --pingpong)
            PINGPONG=1
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
TILE_ARGS=()
BF16_CLUSTER_SHAPE="${BF16_CLUSTER_SHAPE:-1,1,1}"
case "$BF16_CLUSTER_SHAPE" in
    1,1,1|2,1,1|1,2,1|2,2,1)
        ;;
    *)
        echo "ERROR: BF16_CLUSTER_SHAPE must be 1,1,1, 2,1,1, 1,2,1, or 2,2,1" >&2
        exit 2
        ;;
esac
if [ "$SWAP_AB" -eq 1 ]; then
    if [ "$PINGPONG" -eq 1 ]; then
        BF16_SWAP_AB_M="${BF16_SWAP_AB_M:-128}"
    else
        BF16_SWAP_AB_M="${BF16_SWAP_AB_M:-256}"
    fi
    case "$BF16_SWAP_AB_M" in
        128|256)
            ;;
        *)
            echo "ERROR: BF16_SWAP_AB_M must be one of 128,256" >&2
            exit 2
            ;;
    esac
    BF16_SWAP_AB_N="${BF16_SWAP_AB_N:-32}"
    case "$BF16_SWAP_AB_N" in
        8|16|32|64|128)
            ;;
        *)
            echo "ERROR: BF16_SWAP_AB_N must be one of 8,16,32,64,128" >&2
            exit 2
            ;;
    esac
    TILE_ARGS=(
        --swap_ab
        --mma_tiler_mnk "${BF16_SWAP_AB_M},${BF16_SWAP_AB_N},64"
        --cluster_shape_mnk "$BF16_CLUSTER_SHAPE"
    )
    if [ "$PINGPONG" -eq 1 ] && [ "$BF16_SWAP_AB_M" -ne 128 ]; then
        echo "ERROR: swap-AB ping-pong requires BF16_SWAP_AB_M=128" >&2
        exit 2
    fi
else
    BF16_NON_SWAP_M="${BF16_NON_SWAP_M:-64}"
    case "$BF16_NON_SWAP_M" in
        64)
            ;;
        *)
            echo "ERROR: BF16_NON_SWAP_M must be 64" >&2
            exit 2
            ;;
    esac
    BF16_NON_SWAP_N="${BF16_NON_SWAP_N:-128}"
    case "$BF16_NON_SWAP_N" in
        128|256)
            ;;
        *)
            echo "ERROR: BF16_NON_SWAP_N must be one of 128,256" >&2
            exit 2
            ;;
    esac
    TILE_ARGS=(
        --mma_tiler_mnk "${BF16_NON_SWAP_M},${BF16_NON_SWAP_N},64"
        --cluster_shape_mnk "$BF16_CLUSTER_SHAPE"
    )
    if [ "$PINGPONG" -eq 1 ] && [ "$BF16_NON_SWAP_N" -ne 128 ]; then
        echo "ERROR: non-swap ping-pong requires BF16_NON_SWAP_N=128" >&2
        exit 2
    fi
fi
if [ "$PINGPONG" -eq 1 ]; then
    TILE_ARGS=(--pingpong "${TILE_ARGS[@]}")
fi

# BF16_TAIL_SPLIT=1: the whole run shares one geometry, so reject an
# incompatible cluster shape up front.
BF16_TAIL_SPLIT="${BF16_TAIL_SPLIT:-0}"
case "$BF16_TAIL_SPLIT" in
    1)
        if { [ "$SWAP_AB" -eq 1 ] && [ "$BF16_CLUSTER_SHAPE" = "1,2,1" ]; } \
            || { [ "$SWAP_AB" -eq 0 ] && [ "$BF16_CLUSTER_SHAPE" = "2,1,1" ]; }; then
            TILE_ARGS+=(--tail_split_pairs)
        else
            echo "ERROR: BF16_TAIL_SPLIT=1 requires BF16_CLUSTER_SHAPE=1,2,1 with --swapab or BF16_CLUSTER_SHAPE=2,1,1 without it (got swap_ab=$SWAP_AB, cluster $BF16_CLUSTER_SHAPE)" >&2
            exit 2
        fi
        ;;
    0)
        ;;
    *)
        echo "ERROR: BF16_TAIL_SPLIT must be 0 or 1" >&2
        exit 2
        ;;
esac
export BF16_TAIL_SPLIT

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

# Entries retain their historical 1,1,1 arguments for readability; TILE_ARGS
# is appended last and applies BF16_CLUSTER_SHAPE to every launch.

declare -a TESTS=(
    # ── M01: single-rank sanity ──
    "M01_single_balanced_topk2       | single | --kind bf16 --num_tokens_per_rank 192  --num_topk 2  --num_total_experts 8   --hidden 1024 --intermediate 2048 --mma_tiler_mnk 64,128,64 --cluster_shape_mnk 1,1,1 --enable_static_expert_shape --ref_compute_graph transformers"

    # ── Static expert shape: branch coverage + largest stress ──
    "M02_mr_balanced_topk2_tiny      | multi  | --kind bf16 --num_tokens_per_rank 96   --num_topk 2  --num_total_experts 8   --hidden 1024 --intermediate 1024 --mma_tiler_mnk 64,128,64 --cluster_shape_mnk 1,1,1 --enable_static_expert_shape"
    "M03_mr_balanced_topk3_atomic    | multi  | --kind bf16 --num_tokens_per_rank 256  --num_topk 3  --num_total_experts 24  --hidden 1536 --intermediate 2048 --mma_tiler_mnk 64,128,64 --cluster_shape_mnk 1,1,1 --enable_static_expert_shape --load_balance_mode atomic_counter"
    "M09_mr_power_law_topk13         | multi  | --kind bf16 --num_tokens_per_rank 832  --num_topk 13 --num_total_experts 104 --hidden 2560 --intermediate 4096 --mma_tiler_mnk 64,128,64 --cluster_shape_mnk 1,1,1 --enable_static_expert_shape --route_distribution power_law"
    "M11_mr_large_topk11_atomic      | multi  | --kind bf16 --num_tokens_per_rank 1792 --num_topk 11 --num_total_experts 88  --hidden 2816 --intermediate 4096 --mma_tiler_mnk 64,128,64 --cluster_shape_mnk 1,1,1 --enable_static_expert_shape --load_balance_mode atomic_counter"
    "M13_mr_atomic_pl_topk7         | multi  | --kind bf16 --num_tokens_per_rank 512  --num_topk 7  --num_total_experts 56  --hidden 1792 --intermediate 1792 --mma_tiler_mnk 64,128,64 --cluster_shape_mnk 1,1,1 --enable_static_expert_shape --load_balance_mode atomic_counter --route_distribution power_law"

    # ── Dynamic expert shape: retain both existing scheduler/route combinations ──
    "M17_mr_dynshape_balanced_topk4  | multi  | --kind bf16 --num_tokens_per_rank 384  --num_topk 4  --num_total_experts 32  --hidden 2048 --intermediate 2048 --mma_tiler_mnk 64,128,64 --cluster_shape_mnk 1,1,1"
    "M18_mr_dynshape_pl_topk7        | multi  | --kind bf16 --num_tokens_per_rank 576  --num_topk 7  --num_total_experts 64  --hidden 1536 --intermediate 2688 --mma_tiler_mnk 64,128,64 --cluster_shape_mnk 1,1,1 --route_distribution power_law --load_balance_mode atomic_counter"

    # ── GC01: generate_c — raw pre-SwiGLU fc1 gate+up output (training
    # forward), 4-rank.  Runs under whichever operand order the script was
    # invoked with, so this one entry covers non-swap and --swapab. ──
    "GC01_mr_generate_c_topk3        | multi  | --kind bf16 --num_tokens_per_rank 256  --num_topk 3  --num_total_experts 24  --hidden 1536 --intermediate 2048 --mma_tiler_mnk 64,128,64 --cluster_shape_mnk 1,1,1 --enable_static_expert_shape --load_balance_mode atomic_counter --generate_c"

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
        full_name="${name}"
        if test_matches_selectors "$full_name"; then
            echo "$full_name"
        fi
    done
    exit 0
fi

echo "==========================================================================="
echo "MegaMoE BF16 functional tests"
echo "  RUNNER : ${RUNNER}"
echo "  PYTHON : ${PYTHON}"
echo "  NPROC  : ${NPROC}"
echo "  NNODES : ${MEGA_NNODES}"
echo "  WORLD  : ${WORLD_SIZE}"
echo "  MODE   : pingpong=${PINGPONG}"
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
        args="${rest#*|}"
        args="${args#"${args%%[![:space:]]*}"}"
        full_name="${name}"

        if ! test_matches_selectors "$full_name"; then
            SKIP_COUNT=$((SKIP_COUNT + 1))
            continue
        fi

        echo
        echo "==========================================================================="
        echo "[TEST] $full_name"
        case "$launch_mode" in
            single)
                echo "[CMD]  MEGA_NO_DIST=1 $PYTHON $RUNNER $args ${TILE_ARGS[*]}"
                ;;
            multi)
                if [ "$MEGA_NNODES" -gt 1 ]; then
                    echo "[CMD]  torchrun --nnodes=$MEGA_NNODES --node_rank=$MEGA_NODE_RANK --nproc_per_node=$NPROC --master_addr=$MEGA_MASTER_ADDR --master_port=$MEGA_MASTER_PORT $RUNNER $args ${TILE_ARGS[*]}"
                else
                    echo "[CMD]  torchrun --nproc_per_node=$NPROC $RUNNER $args ${TILE_ARGS[*]}"
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
                timeout 300 env MEGA_NO_DIST=1 "$PYTHON" "$RUNNER" $args "${TILE_ARGS[@]}"
                rc=$?
                ;;
            multi)
                if [ "$MEGA_NNODES" -gt 1 ]; then
                    timeout 300 torchrun \
                        --nnodes="$MEGA_NNODES" --node_rank="$MEGA_NODE_RANK" \
                        --nproc_per_node="$NPROC" \
                        --master_addr="$MEGA_MASTER_ADDR" --master_port="$MEGA_MASTER_PORT" \
                        "$RUNNER" $args "${TILE_ARGS[@]}"
                else
                    timeout 300 torchrun --nproc_per_node="$NPROC" "$RUNNER" $args "${TILE_ARGS[@]}"
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
    echo "SUMMARY: ${PASS_COUNT}/${RAN_COUNT} passed, ${FAIL_COUNT} failed, ${SKIP_COUNT} skipped (selectors: ${SELECTORS[*]}; wallclock ${TOTAL_ELAPSED}s)"
else
    echo "SUMMARY: ${PASS_COUNT}/${TOTAL} passed, ${FAIL_COUNT} failed (wallclock ${TOTAL_ELAPSED}s)"
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
