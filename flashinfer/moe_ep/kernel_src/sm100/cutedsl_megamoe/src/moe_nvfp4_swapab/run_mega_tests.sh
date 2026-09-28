#!/usr/bin/env bash
# Functional test harness for the distributed MegaMoE fused dispatch + fc1/fc2
# + combine runner.  M01 runs single-rank with MEGA_NO_DIST=1; other cases run
# through torchrun using MEGA_NPROC/MEGA_NNODES overrides when present.
#
# Coverage dimensions mixed across the list below:
#   balanced/power_law routing, static/atomic_counter scheduling, topk range,
#   1CTA/2CTA, cluster_m, transpose/UBLK fc2 output modes, SF padding
#   boundaries for hidden/intermediate shapes, and token-back mode
#   (epi_warps / standalone_warps / reuse_dispatch_warps).
#
# Test entry format (pipe-separated, optional 4th env field):
#   "name | mode | args [ | ENV=val ENV2=val2 ]"
# The 4th field, when present, is exported (via ``env``) only for that test's
# launch, e.g. the developer-only MEGA_TOKEN_BACK_ATOMIC_BATCH knob.
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
#   bash .../run_mega_tests.sh M01 M03 M19
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

# Strip leading/trailing whitespace from $1 (no external processes).
trim() {
    local s="$1"
    s="${s#"${s%%[![:space:]]*}"}"
    s="${s%"${s##*[![:space:]]}"}"
    printf '%s' "$s"
}

declare -a TESTS=(
    "M01_single_balanced_topk2       | single | --num_tokens_per_rank 192  --num_topk 2  --num_total_experts 8   --hidden 1024 --intermediate 2048 --enable_static_expert_shape --use_bulk_fc2_store --ref_compute_graph transformers"

    "M02_mr_balanced_topk2_tiny      | multi  | --num_tokens_per_rank 96   --num_topk 2  --num_total_experts 8   --hidden 1152 --intermediate 1088 --enable_static_expert_shape --use_bulk_fc2_store --in_kernel_fc2_reduce"
    "M03_mr_balanced_topk3_dual_sa   | multi  | --num_tokens_per_rank 256  --num_topk 3  --num_total_experts 24  --hidden 1568 --intermediate 2112 --enable_static_expert_shape --load_balance_mode atomic_counter --token_back_mode standalone_warps --combine_format 16e2m1xbf16"
    "M04_mr_balanced_topk5           | multi  | --num_tokens_per_rank 384  --num_topk 5  --num_total_experts 40  --hidden 1664 --intermediate 2176 --enable_static_expert_shape --use_bulk_fc2_store --in_kernel_fc2_reduce"
    "M05_mr_balanced_topk7           | multi  | --num_tokens_per_rank 448  --num_topk 7  --num_total_experts 56  --hidden 2048 --intermediate 2368 --enable_static_expert_shape --use_bulk_fc2_store --combine_format 32e4m3xe8m0"
    "M06_mr_balanced_topk11_redg_ru  | multi  | --num_tokens_per_rank 880  --num_topk 11 --num_total_experts 88  --hidden 1920 --intermediate 3136 --enable_static_expert_shape --in_kernel_fc2_reduce --token_back_mode reuse_dispatch_warps"

    "M07_mr_power_law_topk3_dual_sa  | multi  | --num_tokens_per_rank 320  --num_topk 3  --num_total_experts 24  --hidden 1632 --intermediate 1728 --enable_static_expert_shape --route_distribution power_law --token_back_mode standalone_warps --combine_format bf16"
    "M08_mr_power_law_topk7_dual_ru  | multi  | --num_tokens_per_rank 640  --num_topk 7  --num_total_experts 56  --hidden 1440 --intermediate 1920 --enable_static_expert_shape --route_distribution power_law --load_balance_mode atomic_counter --ref_compute_graph transformers --token_back_mode reuse_dispatch_warps --combine_format 32e4m3xe8m0"
    "M09_mr_power_law_topk13_clamp   | multi  | --num_tokens_per_rank 832  --num_topk 13 --num_total_experts 104 --hidden 2560 --intermediate 4160 --enable_static_expert_shape --route_distribution power_law --use_bulk_fc2_store --in_kernel_fc2_reduce --gate_up_clamp 10"

    "M10_mr_large_topk5              | multi  | --num_tokens_per_rank 768  --num_topk 5  --num_total_experts 40  --hidden 2304 --intermediate 3392 --enable_static_expert_shape --use_bulk_fc2_store --in_kernel_fc2_reduce"
    "M11_mr_large_topk11_aligned_clamp_ru | multi | --num_tokens_per_rank 1792 --num_topk 11 --num_total_experts 88  --hidden 2944 --intermediate 4096 --enable_static_expert_shape --load_balance_mode atomic_counter --gate_up_clamp 10 --token_back_mode reuse_dispatch_warps --combine_format 16e2m1xbf16"

    "M12_mr_T256_clm2_balanced_topk4_redg_sa | multi  | --num_tokens_per_rank 704  --num_topk 4  --num_total_experts 40  --hidden 2080 --intermediate 3136 --enable_static_expert_shape --mma_tiler_mnk 256,256,256 --cluster_shape_mnk 2,1,1 --use_2cta_instrs --in_kernel_fc2_reduce --token_back_mode standalone_warps"
    "M13_mr_T256_clm2_pl_topk7_dual_ru | multi  | --num_tokens_per_rank 896  --num_topk 7  --num_total_experts 56  --hidden 2272 --intermediate 2624 --enable_static_expert_shape --mma_tiler_mnk 256,256,256 --cluster_shape_mnk 2,1,1 --use_2cta_instrs --route_distribution power_law --token_back_mode reuse_dispatch_warps --combine_format 32e4m3xe8m0"
    "M14_mr_T256_clm2_balanced_topk13_sa | multi  | --num_tokens_per_rank 1280 --num_topk 13 --num_total_experts 104 --hidden 2816 --intermediate 4288 --enable_static_expert_shape --mma_tiler_mnk 256,256,256 --cluster_shape_mnk 2,1,1 --use_2cta_instrs --load_balance_mode atomic_counter --ref_compute_graph transformers --gate_up_clamp 10 --token_back_mode standalone_warps --combine_format 16e2m1xbf16"
    "M15_mr_T256_clm4_topk11_aligned | multi  | --num_tokens_per_rank 2048 --num_topk 11 --num_total_experts 88  --hidden 2176 --intermediate 3712 --enable_static_expert_shape --mma_tiler_mnk 256,256,256 --cluster_shape_mnk 4,1,1 --use_2cta_instrs --use_bulk_fc2_store --gate_up_clamp 9"
    "M16_mr_T256_clm4_pl_topk4_redg_sa | multi  | --num_tokens_per_rank 1024 --num_topk 4  --num_total_experts 40  --hidden 2432 --intermediate 3776 --enable_static_expert_shape --mma_tiler_mnk 256,256,256 --cluster_shape_mnk 4,1,1 --use_2cta_instrs --route_distribution power_law --in_kernel_fc2_reduce --load_balance_mode atomic_counter --gate_up_clamp 8 --token_back_mode standalone_warps"

    "M17_mr_atomic_balanced_topk3    | multi  | --num_tokens_per_rank 128  --num_topk 3  --num_total_experts 24  --hidden 1280 --intermediate 2432 --enable_static_expert_shape --load_balance_mode atomic_counter --use_bulk_fc2_store --in_kernel_fc2_reduce"
    "M18_mr_atomic_pl_topk7_dual_redg_ru | multi  | --num_tokens_per_rank 512  --num_topk 7  --num_total_experts 56  --hidden 1856 --intermediate 1856 --enable_static_expert_shape --load_balance_mode atomic_counter --route_distribution power_law --in_kernel_fc2_reduce --token_back_mode reuse_dispatch_warps"
    "M19_mr_atomic_T256_clm2_topk5_ru | multi  | --num_tokens_per_rank 1152 --num_topk 5  --num_total_experts 40  --hidden 2688 --intermediate 3648 --enable_static_expert_shape --load_balance_mode atomic_counter --mma_tiler_mnk 256,256,256 --cluster_shape_mnk 2,1,1 --use_2cta_instrs --gate_up_clamp 10 --token_back_mode reuse_dispatch_warps"

    "M20_mr_balanced_wide_intermed   | multi  | --num_tokens_per_rank 64   --num_topk 3  --num_total_experts 24  --hidden 1792 --intermediate 4928 --enable_static_expert_shape --load_balance_mode atomic_counter --use_bulk_fc2_store --ref_compute_graph transformers --gate_up_clamp 11 --combine_format 32e4m3xe8m0"

    # ── M21..M23: tile_n=64 ──
    "M21_mr_N64_1cta_balanced_topk3_sa    | multi  | --num_tokens_per_rank 288 --num_topk 3 --num_total_experts 48 --hidden 1408 --intermediate 2240 --enable_static_expert_shape --mma_tiler_mnk 128,64,256 --cluster_shape_mnk 1,1,1 --ref_compute_graph transformers --token_back_mode standalone_warps --combine_format 16e2m1xbf16"
    "M22_mr_N64_T256_clm2_pl_topk4_redg_sa | multi  | --num_tokens_per_rank 480 --num_topk 4 --num_total_experts 96 --hidden 1472 --intermediate 2752 --enable_static_expert_shape --mma_tiler_mnk 256,64,256 --cluster_shape_mnk 2,1,1 --use_2cta_instrs --route_distribution power_law --in_kernel_fc2_reduce --token_back_mode standalone_warps"
    "M23_mr_N64_T256_clm4_pl_atomic_topk7_ru | multi  | --num_tokens_per_rank 672 --num_topk 7 --num_total_experts 72 --hidden 1728 --intermediate 3520 --enable_static_expert_shape --mma_tiler_mnk 256,64,256 --cluster_shape_mnk 4,1,1 --use_2cta_instrs --route_distribution power_law --load_balance_mode atomic_counter --ref_compute_graph transformers --gate_up_clamp 6 --token_back_mode reuse_dispatch_warps --combine_format 32e4m3xe8m0"

    # -- Token-back modes (standalone/reuse) + in_kernel_reduce + flag > 1.
    # M24 keeps the default atomic batch (1); M25-M27 exercise the
    # developer-only MEGA_TOKEN_BACK_ATOMIC_BATCH env at 2/3 (only meaningful
    # under a non-epi token_back_mode + atomic_counter scheduling).
    "M24_mr_tbstandalone_reduce_flag8_powerlaw | multi  | --num_tokens_per_rank 3344 --num_topk 5 --num_total_experts 72 --hidden 6176 --intermediate 6080 --enable_static_expert_shape --mma_tiler_mnk 256,256,256 --cluster_shape_mnk 2,1,1 --use_2cta_instrs --route_distribution power_law --load_balance_mode atomic_counter --flag_batch 8 --epi_flag_batch 2,4 --token_back_mode standalone_warps --in_kernel_fc2_reduce"

    "M25_mr_tbreuse_reduce_atomic2_powerlaw    | multi  | --num_tokens_per_rank 1024 --num_topk 5 --num_total_experts 72 --hidden 6176 --intermediate 6080 --enable_static_expert_shape --mma_tiler_mnk 256,256,256 --cluster_shape_mnk 2,1,1 --use_2cta_instrs --route_distribution power_law --load_balance_mode atomic_counter --flag_batch 8 --epi_flag_batch 2,4 --token_back_mode reuse_dispatch_warps --in_kernel_fc2_reduce | MEGA_TOKEN_BACK_ATOMIC_BATCH=2"
    "M26_mr_tbstandalone_atomic3_powerlaw      | multi  | --num_tokens_per_rank 1024 --num_topk 5 --num_total_experts 72 --hidden 6176 --intermediate 6080 --enable_static_expert_shape --mma_tiler_mnk 256,256,256 --cluster_shape_mnk 2,1,1 --use_2cta_instrs --route_distribution power_law --load_balance_mode atomic_counter --flag_batch 8 --epi_flag_batch 2,4 --token_back_mode standalone_warps --combine_format 16e2m1xbf16 | MEGA_TOKEN_BACK_ATOMIC_BATCH=3"
    "M27_mr_tbreuse_reduce_atomic2_pl_small     | multi  | --num_tokens_per_rank 768  --num_topk 5 --num_total_experts 72 --hidden 6176 --intermediate 6080 --enable_static_expert_shape --mma_tiler_mnk 256,256,256 --cluster_shape_mnk 2,1,1 --use_2cta_instrs --route_distribution power_law --load_balance_mode atomic_counter --flag_batch 8 --epi_flag_batch 2,4 --token_back_mode reuse_dispatch_warps --in_kernel_fc2_reduce | MEGA_TOKEN_BACK_ATOMIC_BATCH=2"
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
echo "MegaMoE functional tests"
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
    # Parse "name | mode | args [ | env ]".  Strip whitespace around each
    # segment.  args carry no '|', so a 4-var IFS split is unambiguous; the
    # optional 4th field holds per-test env assignments.
    IFS='|' read -r name mode args test_env <<< "$entry"
    name="$(trim "$name")"
    mode="$(trim "$mode")"
    args="$(trim "$args")"
    test_env="$(trim "$test_env")"

    if ! test_matches_selectors "$name"; then
        SKIP_COUNT=$((SKIP_COUNT + 1))
        continue
    fi

    echo
    echo "==========================================================================="
    echo "[TEST] $name"
    case "$mode" in
        single)
            echo "[CMD]  env $test_env MEGA_NO_DIST=1 $PYTHON $RUNNER $args"
            ;;
        multi)
            if [ "$MEGA_NNODES" -gt 1 ]; then
                echo "[CMD]  env $test_env torchrun --nnodes=$MEGA_NNODES --node_rank=$MEGA_NODE_RANK --nproc_per_node=$NPROC --master_addr=$MEGA_MASTER_ADDR --master_port=$MEGA_MASTER_PORT $RUNNER $args"
            else
                echo "[CMD]  env $test_env torchrun --nproc_per_node=$NPROC $RUNNER $args"
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
    # shellcheck disable=SC2086  # intentional word-splitting on $args / $test_env
    case "$mode" in
        single)
            env $test_env MEGA_NO_DIST=1 "$PYTHON" "$RUNNER" $args
            rc=$?
            ;;
        multi)
            if [ "$MEGA_NNODES" -gt 1 ]; then
                env $test_env torchrun \
                    --nnodes="$MEGA_NNODES" --node_rank="$MEGA_NODE_RANK" \
                    --nproc_per_node="$NPROC" \
                    --master_addr="$MEGA_MASTER_ADDR" --master_port="$MEGA_MASTER_PORT" \
                    "$RUNNER" $args
            else
                env $test_env torchrun --nproc_per_node="$NPROC" "$RUNNER" $args
            fi
            rc=$?
            ;;
    esac
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
