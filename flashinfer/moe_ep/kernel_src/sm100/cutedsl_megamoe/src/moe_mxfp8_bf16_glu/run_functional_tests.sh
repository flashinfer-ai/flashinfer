#!/usr/bin/env bash
# Functional test harness for the mixed MXFP8-weight/BF16 FC12 runner.
#
# The interface mirrors moe_bf16_glu/run_functional_tests.sh:
#   bash run_functional_tests.sh
#   bash run_functional_tests.sh --list
#   bash run_functional_tests.sh --fail-fast
#   bash run_functional_tests.sh n256
#   PYTHON=/path/to/python bash run_functional_tests.sh

set -u

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
RUNNER="${SCRIPT_DIR}/runner_fc12.py"
PYTHON="${PYTHON:-${SCRIPT_DIR}/../../venv/bin/python}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

if [ ! -f "$RUNNER" ]; then
    echo "ERROR: runner_fc12.py not found at ${RUNNER}" >&2
    exit 2
fi
if [ ! -x "$PYTHON" ]; then
    echo "ERROR: Python interpreter is not executable: ${PYTHON}" >&2
    exit 2
fi

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

# Implementation details are explicit and orthogonal to the public MNK tile.
declare -a TESTS=(
    "F01_e4m3_n128_static        | --kind mxfp8_bf16_e4m3 --tokens_after_topk 256 --experts 4 --hidden 1024 --intermediate 1024 --mma_tiler_mnk 256,128,128 --transform_buffer tmem --no-accumulator_overlap --transform_k_tile 128 --cluster_shape_mnk 2,1,1 --use_2cta_instrs --enable_static_expert_shape --balance_route --load_balance_mode static --ref_compute_graph deepgemm"
    "F02_e5m2_n128_static        | --kind mxfp8_bf16_e5m2 --tokens_after_topk 256 --experts 4 --hidden 1024 --intermediate 1024 --mma_tiler_mnk 256,128,128 --transform_buffer tmem --no-accumulator_overlap --transform_k_tile 128 --cluster_shape_mnk 2,1,1 --use_2cta_instrs --enable_static_expert_shape --balance_route --load_balance_mode static --ref_compute_graph deepgemm"
    "F03_e4m3_n128_atomic        | --kind mxfp8_bf16_e4m3 --tokens_after_topk 384 --experts 8 --hidden 1536 --intermediate 2048 --mma_tiler_mnk 256,128,128 --transform_buffer tmem --no-accumulator_overlap --transform_k_tile 128 --cluster_shape_mnk 2,1,1 --use_2cta_instrs --enable_static_expert_shape --load_balance_mode atomic_counter --ref_compute_graph deepgemm"
    "F04_e4m3_n256_smem          | --kind mxfp8_bf16_e4m3 --tokens_after_topk 512 --experts 8 --hidden 1792 --intermediate 3072 --mma_tiler_mnk 256,256,128 --transform_buffer smem --no-accumulator_overlap --transform_k_tile 128 --cluster_shape_mnk 2,1,1 --use_2cta_instrs --enable_static_expert_shape --balance_route --load_balance_mode static --ref_compute_graph deepgemm"
    "F05_e4m3_n256_tmem_k128_overlap | --kind mxfp8_bf16_e4m3 --tokens_after_topk 512 --experts 8 --hidden 1792 --intermediate 3072 --mma_tiler_mnk 256,256,128 --transform_buffer tmem --accumulator_overlap --transform_k_tile 64 --cluster_shape_mnk 2,1,1 --use_2cta_instrs --enable_static_expert_shape --balance_route --load_balance_mode static --ref_compute_graph deepgemm"
    "F06_e5m2_clamp              | --kind mxfp8_bf16_e5m2 --tokens_after_topk 320 --experts 8 --hidden 1280 --intermediate 1920 --mma_tiler_mnk 256,128,128 --transform_buffer tmem --no-accumulator_overlap --transform_k_tile 128 --cluster_shape_mnk 2,1,1 --use_2cta_instrs --enable_static_expert_shape --balance_route --load_balance_mode static --gate_up_clamp 10 --ref_compute_graph deepgemm"
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

for entry in "${TESTS[@]}"; do
    name="${entry%%|*}"
    name="${name%"${name##*[![:space:]]}"}"
    args="${entry#*|}"
    args="${args#"${args%%[![:space:]]*}"}"

    if ! test_matches_selectors "$name"; then
        SKIP_COUNT=$((SKIP_COUNT + 1))
        continue
    fi

    echo
    echo "==========================================================================="
    echo "[TEST] $name"
    echo "[CMD]  $PYTHON $RUNNER $args"
    echo "==========================================================================="

    test_start=$SECONDS
    # shellcheck disable=SC2086
    "$PYTHON" "$RUNNER" $args
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
