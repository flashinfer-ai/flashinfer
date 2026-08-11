#!/usr/bin/env bash
# Functional test harness for distributed mixed MXFP8-weight/BF16 MegaMoE.
#
# The interface mirrors moe_bf16_glu/run_mega_tests.sh:
#   bash run_mega_tests.sh
#   bash run_mega_tests.sh --list
#   bash run_mega_tests.sh --fail-fast
#   bash run_mega_tests.sh reuse
#   MEGA_NPROC=4 bash run_mega_tests.sh
#   PYTHON=/path/to/python bash run_mega_tests.sh

set -u

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
RUNNER="${SCRIPT_DIR}/mega_runner.py"
PYTHON="${PYTHON:-${SCRIPT_DIR}/../../venv/bin/python}"
TORCHRUN="${TORCHRUN:-$(dirname -- "$PYTHON")/torchrun}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

if [ ! -f "$RUNNER" ]; then
    echo "ERROR: mega_runner.py not found at ${RUNNER}" >&2
    exit 2
fi
if [ ! -x "$PYTHON" ] || [ ! -x "$TORCHRUN" ]; then
    echo "ERROR: expected executable PYTHON=${PYTHON} and TORCHRUN=${TORCHRUN}" >&2
    exit 2
fi

if [ -n "${MEGA_NPROC:-}" ]; then
    NPROC="$MEGA_NPROC"
elif [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && [ "$CUDA_VISIBLE_DEVICES" != "NoDevFiles" ]; then
    IFS=',' read -r -a VISIBLE_DEVICES <<< "$CUDA_VISIBLE_DEVICES"
    NPROC=${#VISIBLE_DEVICES[@]}
elif command -v nvidia-smi >/dev/null 2>&1; then
    NPROC=$(nvidia-smi --list-gpus 2>/dev/null | wc -l | tr -d '[:space:]')
    if [ -z "$NPROC" ] || [ "$NPROC" -le 0 ]; then
        NPROC=1
    fi
else
    NPROC=1
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
        --fail-fast) FAIL_FAST=1 ;;
        --list) LIST_ONLY=1 ;;
        -h|--help)
            sed -n '2,/^$/p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
            exit 0
            ;;
        --*)
            echo "Unknown flag: $arg (use --help)" >&2
            exit 2
            ;;
        *) SELECTORS+=("$arg") ;;
    esac
done

test_matches_selectors() {
    local name="$1"
    if [ "${#SELECTORS[@]}" -eq 0 ]; then
        return 0
    fi
    local selector
    for selector in "${SELECTORS[@]}"; do
        if [[ "$name" == *"$selector"* ]]; then
            return 0
        fi
    done
    return 1
}

trim() {
    local value="$1"
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    printf '%s' "$value"
}

# Implementation details are explicit and orthogonal to the public MNK tile.
declare -a TESTS=(
    "M01_single_e4m3_n128          | single | --kind mxfp8_bf16_e4m3 --num_tokens_per_rank 192 --num_topk 2 --num_total_experts 8 --hidden 1024 --intermediate 1024 --mma_tiler_mnk 256,128,128 --transform_buffer tmem --no-accumulator_overlap --transform_k_tile 128 --cluster_shape_mnk 2,1,1 --use_2cta_instrs --enable_static_expert_shape --ref_compute_graph deepgemm"
    "M02_single_e5m2_n128          | single | --kind mxfp8_bf16_e5m2 --num_tokens_per_rank 192 --num_topk 2 --num_total_experts 8 --hidden 1024 --intermediate 1024 --mma_tiler_mnk 256,128,128 --transform_buffer tmem --no-accumulator_overlap --transform_k_tile 128 --cluster_shape_mnk 2,1,1 --use_2cta_instrs --enable_static_expert_shape --ref_compute_graph deepgemm"
    "M03_multi_epi_form_a          | multi  | --kind mxfp8_bf16_e4m3 --num_tokens_per_rank 256 --num_topk 4 --num_total_experts 32 --hidden 1536 --intermediate 2048 --mma_tiler_mnk 256,128,128 --transform_buffer tmem --no-accumulator_overlap --transform_k_tile 128 --cluster_shape_mnk 2,1,1 --use_2cta_instrs --enable_static_expert_shape --token_back_mode epi_warps --ref_compute_graph deepgemm"
    "M04_multi_reuse_form_a        | multi  | --kind mxfp8_bf16_e4m3 --num_tokens_per_rank 384 --num_topk 4 --num_total_experts 32 --hidden 1792 --intermediate 2048 --mma_tiler_mnk 256,128,128 --transform_buffer tmem --no-accumulator_overlap --transform_k_tile 128 --cluster_shape_mnk 2,1,1 --use_2cta_instrs --enable_static_expert_shape --token_back_mode reuse_dispatch_warps --ref_compute_graph deepgemm"
    "M05_multi_reuse_form_b        | multi  | --kind mxfp8_bf16_e4m3 --num_tokens_per_rank 384 --num_topk 4 --num_total_experts 32 --hidden 1792 --intermediate 2048 --mma_tiler_mnk 256,128,128 --transform_buffer tmem --no-accumulator_overlap --transform_k_tile 128 --cluster_shape_mnk 2,1,1 --use_2cta_instrs --enable_static_expert_shape --token_back_mode reuse_dispatch_warps --in_kernel_fc2_reduce --ref_compute_graph deepgemm"
    "M06_multi_n256_smem           | multi  | --kind mxfp8_bf16_e4m3 --num_tokens_per_rank 512 --num_topk 4 --num_total_experts 32 --hidden 1792 --intermediate 3072 --mma_tiler_mnk 256,256,128 --transform_buffer smem --no-accumulator_overlap --transform_k_tile 128 --cluster_shape_mnk 2,1,1 --use_2cta_instrs --enable_static_expert_shape --token_back_mode epi_warps --ref_compute_graph deepgemm"
    "M07_multi_n256_tmem_k128_overlap | multi | --kind mxfp8_bf16_e4m3 --num_tokens_per_rank 512 --num_topk 4 --num_total_experts 32 --hidden 1792 --intermediate 3072 --mma_tiler_mnk 256,256,128 --transform_buffer tmem --accumulator_overlap --transform_k_tile 64 --cluster_shape_mnk 2,1,1 --use_2cta_instrs --enable_static_expert_shape --token_back_mode epi_warps --ref_compute_graph deepgemm"
    "M08_multi_power_law_clamp     | multi  | --kind mxfp8_bf16_e5m2 --num_tokens_per_rank 512 --num_topk 4 --num_total_experts 32 --hidden 1536 --intermediate 2048 --mma_tiler_mnk 256,128,128 --transform_buffer tmem --no-accumulator_overlap --transform_k_tile 128 --cluster_shape_mnk 2,1,1 --use_2cta_instrs --enable_static_expert_shape --token_back_mode reuse_dispatch_warps --route_distribution power_law --gate_up_clamp 10 --ref_compute_graph deepgemm"
)

PASS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0
declare -a FAIL_NAMES=()
TOTAL=${#TESTS[@]}
START_TIME=$SECONDS

if [ "$LIST_ONLY" -eq 1 ]; then
    for entry in "${TESTS[@]}"; do
        IFS='|' read -r name _mode _args <<< "$entry"
        name="$(trim "$name")"
        if test_matches_selectors "$name"; then
            echo "$name"
        fi
    done
    exit 0
fi

echo "==========================================================================="
echo "MegaMoE mixed MXFP8/BF16 functional tests"
echo "  RUNNER : ${RUNNER}"
echo "  PYTHON : ${PYTHON}"
echo "  NPROC  : ${NPROC}"
echo "  NNODES : ${MEGA_NNODES}"
echo "  WORLD  : ${WORLD_SIZE}"
echo "  TOTAL  : ${TOTAL} tests"
echo "==========================================================================="

for entry in "${TESTS[@]}"; do
    IFS='|' read -r name mode args <<< "$entry"
    name="$(trim "$name")"
    mode="$(trim "$mode")"
    args="$(trim "$args")"

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
                echo "[CMD]  $TORCHRUN --nnodes=$MEGA_NNODES --node_rank=$MEGA_NODE_RANK --nproc_per_node=$NPROC --master_addr=$MEGA_MASTER_ADDR --master_port=$MEGA_MASTER_PORT $RUNNER $args"
            else
                echo "[CMD]  $TORCHRUN --nproc_per_node=$NPROC $RUNNER $args"
            fi
            ;;
        *)
            echo "ERROR: unknown launch mode '$mode'" >&2
            exit 2
            ;;
    esac
    echo "==========================================================================="

    test_start=$SECONDS
    # shellcheck disable=SC2086
    if [ "$mode" = "single" ]; then
        timeout 300 env MEGA_NO_DIST=1 "$PYTHON" "$RUNNER" $args
    elif [ "$MEGA_NNODES" -gt 1 ]; then
        timeout 300 "$TORCHRUN" \
            --nnodes="$MEGA_NNODES" \
            --node_rank="$MEGA_NODE_RANK" \
            --nproc_per_node="$NPROC" \
            --master_addr="$MEGA_MASTER_ADDR" \
            --master_port="$MEGA_MASTER_PORT" \
            "$RUNNER" $args
    else
        timeout 300 "$TORCHRUN" --nproc_per_node="$NPROC" "$RUNNER" $args
    fi
    rc=$?
    elapsed=$((SECONDS - test_start))

    if [ "$rc" -eq 0 ]; then
        echo "[RESULT] PASS  (${elapsed}s) $name"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        echo "[RESULT] FAIL  (rc=${rc}, ${elapsed}s) $name"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        FAIL_NAMES+=("$name")
        if [ "$FAIL_FAST" -eq 1 ]; then
            break
        fi
    fi
done

RAN_COUNT=$((PASS_COUNT + FAIL_COUNT))
echo
echo "==========================================================================="
echo "SUMMARY: ${PASS_COUNT}/${RAN_COUNT} passed, ${FAIL_COUNT} failed, ${SKIP_COUNT} skipped (wallclock $((SECONDS - START_TIME))s)"
echo "==========================================================================="
if [ "$FAIL_COUNT" -gt 0 ]; then
    printf '  - %s\n' "${FAIL_NAMES[@]}"
fi
if [ "${#SELECTORS[@]}" -gt 0 ] && [ "$RAN_COUNT" -eq 0 ]; then
    echo "WARNING: selectors matched 0 tests (use --list to see all names)"
fi
exit "$FAIL_COUNT"
