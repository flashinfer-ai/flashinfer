#!/usr/bin/env bash
# Functional test harness for the MXFP8 distributed MegaMoE fused dispatch +
# fc1/fc2 + combine runner.  M01 runs single-rank with MEGA_NO_DIST=1; other
# cases run through torchrun using MEGA_NPROC/MEGA_NNODES overrides when present.
#
# MXFP8 tile constraints (apply to ALL tests below):
#   * Only mma_tiler_mnk 256,256,128 with --use_2cta_instrs --cluster_shape_mnk
#     2,1,1 is validated.  Specifically:
#       - cluster_m=4 (cluster_shape_mnk 4,1,1) -> not supported
#       - tile 256,256,256 (K=256)              -> not supported
#       - tile_n=64 / 1-CTA tiles               -> not supported
#   * hidden must be divisible by 256 (fc2 N tile)
#   * intermediate must be divisible by 128 (fc2 K tile / fc1 N tile / 2)
#   * e5m2 element format is currently broken (fc1 epilogue quant path raises
#     "None to integer conversion"); all tests below use --kind mxfp8_e4m3
#
# Coverage dimensions mixed across the list below:
#   balanced/power_law routing, static/atomic_counter scheduling, topk range,
#   static/dynamic expert shape, and varied hidden/intermediate shapes (including wide
#   intermediate and large DeepSeek-ish sizes).
#
# Usage:
#   bash <abs path>/run_mega_tests.sh
#   PYTHON=python3.11 bash .../run_mega_tests.sh
#   MEGA_NPROC=4 bash .../run_mega_tests.sh
#   bash .../run_mega_tests.sh --fail-fast
#   bash .../run_mega_tests.sh --list
#   bash .../run_mega_tests.sh --help
#
# Selective execution (positional args; substring match against test names,
# OR-combined across multiple selectors):
#   bash .../run_mega_tests.sh M01
#   bash .../run_mega_tests.sh M01 M03 M12
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

FAIL_FAST=0
LIST_ONLY=0
declare -a SELECTORS=()
for arg in "$@"; do
    case "$arg" in
        --fail-fast)
            FAIL_FAST=1
            ;;
        --list)
            LIST_ONLY=1
            ;;
        -h|--help)
            sed -n '2,/^# Exit code/p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
            exit 0
            ;;
        --*)
            echo "Unknown flag: $arg (use --help)" >&2
            exit 2
            ;;
        *)
            SELECTORS+=("$arg")
            ;;
    esac
done

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

# Common MXFP8 tile args (the only supported config):
#   --mma_tiler_mnk 256,256,128 --cluster_shape_mnk 2,1,1 --use_2cta_instrs
# All entries below include these explicitly for clarity.

declare -a TESTS=(
    # ── M01: single-rank sanity ──
    "M01_single_balanced_topk2       | single | --kind mxfp8_e4m3 --num_tokens_per_rank 192  --num_topk 2  --num_total_experts 8   --hidden 1024 --intermediate 2048 --mma_tiler_mnk 256,256,128 --cluster_shape_mnk 2,1,1 --use_2cta_instrs --enable_static_expert_shape --ref_compute_graph transformers"

    # ── M02..M06: balanced routing, static expert shape ──
    "M02_mr_balanced_topk2_tiny      | multi  | --kind mxfp8_e4m3 --num_tokens_per_rank 96   --num_topk 2  --num_total_experts 8   --hidden 1024 --intermediate 1024 --mma_tiler_mnk 256,256,128 --cluster_shape_mnk 2,1,1 --use_2cta_instrs --enable_static_expert_shape"
    "M03_mr_balanced_topk3_atomic    | multi  | --kind mxfp8_e4m3 --num_tokens_per_rank 256  --num_topk 3  --num_total_experts 24  --hidden 1536 --intermediate 2048 --mma_tiler_mnk 256,256,128 --cluster_shape_mnk 2,1,1 --use_2cta_instrs --enable_static_expert_shape --load_balance_mode atomic_counter"
    "M04_mr_balanced_topk5           | multi  | --kind mxfp8_e4m3 --num_tokens_per_rank 384  --num_topk 5  --num_total_experts 40  --hidden 1792 --intermediate 2048 --mma_tiler_mnk 256,256,128 --cluster_shape_mnk 2,1,1 --use_2cta_instrs --enable_static_expert_shape"
    "M05_mr_balanced_topk7           | multi  | --kind mxfp8_e4m3 --num_tokens_per_rank 448  --num_topk 7  --num_total_experts 56  --hidden 2048 --intermediate 2304 --mma_tiler_mnk 256,256,128 --cluster_shape_mnk 2,1,1 --use_2cta_instrs --enable_static_expert_shape"
    "M06_mr_balanced_topk11          | multi  | --kind mxfp8_e4m3 --num_tokens_per_rank 880  --num_topk 11 --num_total_experts 88  --hidden 1792 --intermediate 3072 --mma_tiler_mnk 256,256,128 --cluster_shape_mnk 2,1,1 --use_2cta_instrs --enable_static_expert_shape"

    # ── M07..M09: power-law routing ──
    "M07_mr_power_law_topk3          | multi  | --kind mxfp8_e4m3 --num_tokens_per_rank 320  --num_topk 3  --num_total_experts 24  --hidden 1536 --intermediate 1792 --mma_tiler_mnk 256,256,128 --cluster_shape_mnk 2,1,1 --use_2cta_instrs --enable_static_expert_shape --route_distribution power_law"
    "M08_mr_power_law_topk7_atomic   | multi  | --kind mxfp8_e4m3 --num_tokens_per_rank 640  --num_topk 7  --num_total_experts 56  --hidden 1280 --intermediate 1920 --mma_tiler_mnk 256,256,128 --cluster_shape_mnk 2,1,1 --use_2cta_instrs --enable_static_expert_shape --route_distribution power_law --load_balance_mode atomic_counter --ref_compute_graph transformers"
    "M09_mr_power_law_topk13         | multi  | --kind mxfp8_e4m3 --num_tokens_per_rank 832  --num_topk 13 --num_total_experts 104 --hidden 2560 --intermediate 4096 --mma_tiler_mnk 256,256,128 --cluster_shape_mnk 2,1,1 --use_2cta_instrs --enable_static_expert_shape --route_distribution power_law"

    # ── M10..M11: large shapes ──
    "M10_mr_large_topk5              | multi  | --kind mxfp8_e4m3 --num_tokens_per_rank 768  --num_topk 5  --num_total_experts 40  --hidden 2304 --intermediate 3456 --mma_tiler_mnk 256,256,128 --cluster_shape_mnk 2,1,1 --use_2cta_instrs --enable_static_expert_shape"
    "M11_mr_large_topk11_atomic      | multi  | --kind mxfp8_e4m3 --num_tokens_per_rank 1792 --num_topk 11 --num_total_experts 88  --hidden 2816 --intermediate 4096 --mma_tiler_mnk 256,256,128 --cluster_shape_mnk 2,1,1 --use_2cta_instrs --enable_static_expert_shape --load_balance_mode atomic_counter"

    # ── M12..M14: atomic_counter focused ──
    "M12_mr_atomic_balanced_topk3    | multi  | --kind mxfp8_e4m3 --num_tokens_per_rank 128  --num_topk 3  --num_total_experts 24  --hidden 1280 --intermediate 2432 --mma_tiler_mnk 256,256,128 --cluster_shape_mnk 2,1,1 --use_2cta_instrs --enable_static_expert_shape --load_balance_mode atomic_counter"
    "M13_mr_atomic_pl_topk7         | multi  | --kind mxfp8_e4m3 --num_tokens_per_rank 512  --num_topk 7  --num_total_experts 56  --hidden 1792 --intermediate 1792 --mma_tiler_mnk 256,256,128 --cluster_shape_mnk 2,1,1 --use_2cta_instrs --enable_static_expert_shape --load_balance_mode atomic_counter --route_distribution power_law"
    "M14_mr_atomic_topk5             | multi  | --kind mxfp8_e4m3 --num_tokens_per_rank 1152 --num_topk 5  --num_total_experts 40  --hidden 2816 --intermediate 3584 --mma_tiler_mnk 256,256,128 --cluster_shape_mnk 2,1,1 --use_2cta_instrs --enable_static_expert_shape --load_balance_mode atomic_counter"

    # ── M15..M16: misc shape coverage ──
    "M15_mr_balanced_wide_intermed   | multi  | --kind mxfp8_e4m3 --num_tokens_per_rank 64   --num_topk 3  --num_total_experts 24  --hidden 1792 --intermediate 4864 --mma_tiler_mnk 256,256,128 --cluster_shape_mnk 2,1,1 --use_2cta_instrs --enable_static_expert_shape --load_balance_mode atomic_counter --ref_compute_graph transformers"
    "M16_mr_pl_topk13_large_atomic   | multi  | --kind mxfp8_e4m3 --num_tokens_per_rank 1280 --num_topk 13 --num_total_experts 104 --hidden 2816 --intermediate 4352 --mma_tiler_mnk 256,256,128 --cluster_shape_mnk 2,1,1 --use_2cta_instrs --enable_static_expert_shape --load_balance_mode atomic_counter --route_distribution power_law --ref_compute_graph transformers"

    # ── M17..M18: dynamic expert shape (no --enable_static_expert_shape) ──
    "M17_mr_dynshape_balanced_topk4  | multi  | --kind mxfp8_e4m3 --num_tokens_per_rank 384  --num_topk 4  --num_total_experts 32  --hidden 2048 --intermediate 2048 --mma_tiler_mnk 256,256,128 --cluster_shape_mnk 2,1,1 --use_2cta_instrs"
    "M18_mr_dynshape_pl_topk7        | multi  | --kind mxfp8_e4m3 --num_tokens_per_rank 576  --num_topk 7  --num_total_experts 64  --hidden 1536 --intermediate 2688 --mma_tiler_mnk 256,256,128 --cluster_shape_mnk 2,1,1 --use_2cta_instrs --route_distribution power_law --load_balance_mode atomic_counter"

    # ── GC01..GC03: generate_c — raw pre-SwiGLU fc1 accumulator output ──
    # Token padding is bumped to 128 (cta_tile_m) for TMA alignment; these
    # cases verify that physical row offsets are 128-aligned on every rank and
    # that the stored gate+up values match the reference fc1 accumulator.
    "GC01_single_generate_c          | single | --kind mxfp8_e4m3 --num_tokens_per_rank 192  --num_topk 2  --num_total_experts 8   --hidden 1024 --intermediate 2048 --mma_tiler_mnk 256,256,128 --cluster_shape_mnk 2,1,1 --use_2cta_instrs --enable_static_expert_shape --generate_c"
    "GC02_mr_balanced_generate_c     | multi  | --kind mxfp8_e4m3 --num_tokens_per_rank 256  --num_topk 3  --num_total_experts 24  --hidden 1536 --intermediate 2048 --mma_tiler_mnk 256,256,128 --cluster_shape_mnk 2,1,1 --use_2cta_instrs --enable_static_expert_shape --load_balance_mode atomic_counter --generate_c"
    "GC03_mr_power_law_generate_c    | multi  | --kind mxfp8_e4m3 --num_tokens_per_rank 320  --num_topk 3  --num_total_experts 24  --hidden 1536 --intermediate 1792 --mma_tiler_mnk 256,256,128 --cluster_shape_mnk 2,1,1 --use_2cta_instrs --enable_static_expert_shape --route_distribution power_law --generate_c"

    # ── CM01..CM05: combine_format 32e4m3xe8m0 — fp8 e4m3+E8M0 quantized NVLink combine ──
    # fc2 output is quantized to fp8 (e4m3) + per-32 E8M0 block scale before the
    # NVLink combine; TopkReduce dequantizes and reduces to BF16 after the kernel.
    # quantized combine_format is incompatible with --in_kernel_fc2_reduce (Form B).
    "CM01_single_combine_mxfp8          | single | --kind mxfp8_e4m3 --num_tokens_per_rank 192  --num_topk 2  --num_total_experts 8   --hidden 1024 --intermediate 2048 --mma_tiler_mnk 256,256,128 --cluster_shape_mnk 2,1,1 --use_2cta_instrs --enable_static_expert_shape --combine_format 32e4m3xe8m0"
    "CM02_mr_balanced_topk2             | multi  | --kind mxfp8_e4m3 --num_tokens_per_rank 256  --num_topk 2  --num_total_experts 16  --hidden 1024 --intermediate 1024 --mma_tiler_mnk 256,256,128 --cluster_shape_mnk 2,1,1 --use_2cta_instrs --enable_static_expert_shape --combine_format 32e4m3xe8m0"
    "CM03_mr_balanced_topk8             | multi  | --kind mxfp8_e4m3 --num_tokens_per_rank 512  --num_topk 8  --num_total_experts 64  --hidden 2048 --intermediate 2048 --mma_tiler_mnk 256,256,128 --cluster_shape_mnk 2,1,1 --use_2cta_instrs --enable_static_expert_shape --combine_format 32e4m3xe8m0"
    "CM04_mr_power_law_topk5            | multi  | --kind mxfp8_e4m3 --num_tokens_per_rank 320  --num_topk 5  --num_total_experts 40  --hidden 1280 --intermediate 1920 --mma_tiler_mnk 256,256,128 --cluster_shape_mnk 2,1,1 --use_2cta_instrs --enable_static_expert_shape --combine_format 32e4m3xe8m0 --route_distribution power_law"
    "CM05_mr_atomic_topk6               | multi  | --kind mxfp8_e4m3 --num_tokens_per_rank 384  --num_topk 6  --num_total_experts 48  --hidden 1792 --intermediate 2304 --mma_tiler_mnk 256,256,128 --cluster_shape_mnk 2,1,1 --use_2cta_instrs --enable_static_expert_shape --combine_format 32e4m3xe8m0 --load_balance_mode atomic_counter"

    # ── CM06..CM07: combine_format 32e5m2xe8m0 — fp8 e5m2+E8M0 quantized NVLink combine ──
    # Same bandwidth saving as e4m3 but using the e5m2 element format (wider dynamic
    # range, lower precision).  Uses --kind mxfp8_e5m2 so fc1/fc2 weights and
    # activations are also e5m2; the combine wire format matches the compute dtype.
    "CM06_single_combine_e5m2           | single | --kind mxfp8_e5m2 --num_tokens_per_rank 192  --num_topk 2  --num_total_experts 8   --hidden 1024 --intermediate 2048 --mma_tiler_mnk 256,256,128 --cluster_shape_mnk 2,1,1 --use_2cta_instrs --enable_static_expert_shape --combine_format 32e5m2xe8m0"
    "CM07_mr_balanced_combine_e5m2      | multi  | --kind mxfp8_e5m2 --num_tokens_per_rank 256  --num_topk 4  --num_total_experts 32  --hidden 1024 --intermediate 1024 --mma_tiler_mnk 256,256,128 --cluster_shape_mnk 2,1,1 --use_2cta_instrs --enable_static_expert_shape --combine_format 32e5m2xe8m0"
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
        if test_matches_selectors "$name"; then
            echo "$name"
        fi
    done
    exit 0
fi

echo "==========================================================================="
echo "MegaMoE MXFP8 functional tests"
echo "  RUNNER : ${RUNNER}"
echo "  PYTHON : ${PYTHON}"
echo "  NPROC  : ${NPROC}"
echo "  NNODES : ${MEGA_NNODES}"
echo "  WORLD  : ${WORLD_SIZE}"
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
    mode="${rest%%|*}"
    mode="${mode#"${mode%%[![:space:]]*}"}"
    mode="${mode%"${mode##*[![:space:]]}"}"
    args="${rest#*|}"
    args="${args#"${args%%[![:space:]]*}"}"

    if ! test_matches_selectors "$name"; then
        SKIP_COUNT=$((SKIP_COUNT + 1))
        continue
    fi

    echo
    echo "==========================================================================="
    echo "[TEST] $name"
    case "$mode" in
        single)
            echo "[CMD]  MEGA_NO_DIST=1 $PYTHON $RUNNER $args"
            ;;
        multi)
            if [ "$MEGA_NNODES" -gt 1 ]; then
                echo "[CMD]  torchrun --nnodes=$MEGA_NNODES --node_rank=$MEGA_NODE_RANK --nproc_per_node=$NPROC --master_addr=$MEGA_MASTER_ADDR --master_port=$MEGA_MASTER_PORT $RUNNER $args"
            else
                echo "[CMD]  torchrun --nproc_per_node=$NPROC $RUNNER $args"
            fi
            ;;
        *)
            echo "[ERROR] unknown launch mode '$mode' for test '$name'; skipping" >&2
            FAIL_COUNT=$((FAIL_COUNT + 1))
            FAIL_NAMES+=("$name (bad-mode)")
            continue
            ;;
    esac
    echo "==========================================================================="

    test_start=$SECONDS
    # shellcheck disable=SC2086  # intentional word-splitting on $args
    case "$mode" in
        single)
            timeout 300 env MEGA_NO_DIST=1 "$PYTHON" "$RUNNER" $args
            rc=$?
            ;;
        multi)
            if [ "$MEGA_NNODES" -gt 1 ]; then
                timeout 300 torchrun \
                    --nnodes="$MEGA_NNODES" --node_rank="$MEGA_NODE_RANK" \
                    --nproc_per_node="$NPROC" \
                    --master_addr="$MEGA_MASTER_ADDR" --master_port="$MEGA_MASTER_PORT" \
                    "$RUNNER" $args
            else
                timeout 300 torchrun --nproc_per_node="$NPROC" "$RUNNER" $args
            fi
            rc=$?
            ;;
    esac
    if [ "$rc" -eq 124 ]; then
        echo "[TIMEOUT] test exceeded 300s limit — killed"
    fi
    elapsed=$((SECONDS - test_start))

    if [ "$rc" -eq 0 ]; then
        echo "[RESULT] PASS  (${elapsed}s) $name"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        echo "[RESULT] FAIL  (rc=${rc}, ${elapsed}s) $name"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        FAIL_NAMES+=("$name")
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
