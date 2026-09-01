#!/usr/bin/env bash
# Functional test harness for the local fused fc1+fc2 MoE NVFP4 swap-AB runner.
# It resolves runner_fc12.py relative to this script, so CWD is irrelevant.
#
# Coverage dimensions mixed across the list below:
#   static/dynamic expert shapes, static/atomic_counter scheduling, 1CTA/2CTA,
#   cluster_m, balanced/dirichlet routing, partial-K/SF-padding boundaries, and
#   the fc2 transpose/UBLK store modes supported by the local runner.
#
# Usage:
#   bash <abs path>/run_functional_tests.sh
#   PYTHON=python3.11 bash .../run_functional_tests.sh
#   bash .../run_functional_tests.sh --fail-fast
#   bash .../run_functional_tests.sh --list
#   bash .../run_functional_tests.sh --help
#
# Selective execution (positional args; substring match against test names,
# OR-combined across multiple selectors):
#   bash .../run_functional_tests.sh A1                    # run A1_static_dynShape only
#   bash .../run_functional_tests.sh A1 A3 E15             # run those three
#   bash .../run_functional_tests.sh atomic                # run anything with "atomic"
#   bash .../run_functional_tests.sh dirichlet --fail-fast # combine with flags
#
# Exit code: number of failed tests (0 = all pass).  Per-test runner stdout /
# stderr is passed through verbatim; the [CMD] line printed before each test
# is the exact command to copy-paste for standalone debugging.

export PATH=/usr/bin:$PATH
export LD=/usr/bin/ld
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export CUDAHOSTCXX=/usr/bin/g++
export TRITON_CC=/usr/bin/gcc
export CFLAGS="-B/usr/bin"
export CXXFLAGS="-B/usr/bin"
export LDFLAGS="-B/usr/bin -fuse-ld=bfd"

set -u  # fail on undefined vars; do NOT set -e (we want to continue on failures)

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
RUNNER="${SCRIPT_DIR}/runner_fc12.py"
PYTHON="${PYTHON:-python}"

if [ ! -f "$RUNNER" ]; then
    echo "ERROR: runner_fc12.py not found at ${RUNNER}" >&2
    exit 2
fi

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
            # Positional arg: test name selector (substring match).  Multiple
            # selectors are OR-combined; empty list means "run all".
            SELECTORS+=("$arg")
            ;;
    esac
done

# Returns 0 (true) if the test name matches any selector, OR if the selector
# list is empty (default = run all).
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

# Each entry: "name | space-separated runner CLI args".
declare -a TESTS=(
    "A1_static_dynShape_transformers | --tokens_after_topk 1500 --experts 3  --hidden 1280 --intermediate 1536 --mma_tiler_mnk 128,128,256 --cluster_shape_mnk 1,1,1 --balance_route --load_balance_mode static --ref_compute_graph transformers"
    "A2_static_statShape             | --tokens_after_topk 1500 --experts 5  --hidden 1280 --intermediate 1536 --mma_tiler_mnk 128,128,256 --cluster_shape_mnk 1,1,1 --balance_route --load_balance_mode static --enable_static_expert_shape"
    "A3_atomic_dynShape              | --tokens_after_topk 1500 --experts 7  --hidden 1280 --intermediate 1536 --mma_tiler_mnk 128,128,256 --cluster_shape_mnk 1,1,1 --balance_route --load_balance_mode atomic_counter"
    "A4_atomic_statShape_ublkcp      | --tokens_after_topk 1500 --experts 11 --hidden 1280 --intermediate 1536 --mma_tiler_mnk 128,128,256 --cluster_shape_mnk 1,1,1 --balance_route --load_balance_mode atomic_counter --enable_static_expert_shape --use_bulk_fc2_store"

    "B5_cluster_m2_static_ublkcp     | --tokens_after_topk 2400 --experts 13 --hidden 1792 --intermediate 1536 --mma_tiler_mnk 256,256,256 --cluster_shape_mnk 2,1,1 --use_2cta_instrs --balance_route --load_balance_mode static --enable_static_expert_shape --use_bulk_fc2_store"
    "B6_cluster_m2_atomic_transformers | --tokens_after_topk 2400 --experts 17 --hidden 1792 --intermediate 1536 --mma_tiler_mnk 256,256,256 --cluster_shape_mnk 2,1,1 --use_2cta_instrs --balance_route --load_balance_mode atomic_counter --ref_compute_graph transformers"
    "B7_cluster_m4_atomic_statShape  | --tokens_after_topk 4500 --experts 19 --hidden 1792 --intermediate 1536 --mma_tiler_mnk 256,256,256 --cluster_shape_mnk 4,1,1 --use_2cta_instrs --balance_route --load_balance_mode atomic_counter --enable_static_expert_shape"

    "C8_dirichlet_static             | --tokens_after_topk 3000 --experts 23 --hidden 1280 --intermediate 1536 --mma_tiler_mnk 128,128,256 --cluster_shape_mnk 1,1,1 --load_balance_mode static"
    "C9_dirichlet_atomic_transformers | --tokens_after_topk 3000 --experts 29 --hidden 1280 --intermediate 1536 --mma_tiler_mnk 128,128,256 --cluster_shape_mnk 1,1,1 --load_balance_mode atomic_counter --ref_compute_graph transformers"
    "C10_dirichlet_atomic_cluster_m2 | --tokens_after_topk 9000 --experts 31 --hidden 2304 --intermediate 2560 --mma_tiler_mnk 256,256,256 --cluster_shape_mnk 2,1,1 --use_2cta_instrs --load_balance_mode atomic_counter --enable_static_expert_shape"

    "D11_large_static_balanced       | --tokens_after_topk 17000 --experts 8  --hidden 7168 --intermediate 4608 --mma_tiler_mnk 128,128,256 --cluster_shape_mnk 1,1,1 --balance_route --load_balance_mode static"
    "D12_large_atomic_balanced       | --tokens_after_topk 17000 --experts 8  --hidden 7168 --intermediate 4608 --mma_tiler_mnk 128,128,256 --cluster_shape_mnk 1,1,1 --balance_route --load_balance_mode atomic_counter"
    "D13_large_atomic_dirichlet_c4   | --tokens_after_topk 24000 --experts 37 --hidden 6912 --intermediate 4608 --mma_tiler_mnk 256,256,256 --cluster_shape_mnk 4,1,1 --use_2cta_instrs --load_balance_mode atomic_counter --enable_static_expert_shape"

    "E14_fc1_partial_k_hidden        | --tokens_after_topk 3000 --experts 41 --hidden 1312 --intermediate 1536 --mma_tiler_mnk 128,128,256 --cluster_shape_mnk 1,1,1 --balance_route --load_balance_mode atomic_counter"
    "E15_fc2_partial_k_single_tile_transformers | --tokens_after_topk 2000 --experts 43 --hidden 1280 --intermediate 192  --mma_tiler_mnk 128,128,256 --cluster_shape_mnk 1,1,1 --balance_route --load_balance_mode atomic_counter --ref_compute_graph transformers"
    "E16_tiny_double_partial_k       | --tokens_after_topk 240  --experts 47 --hidden 224  --intermediate 448  --mma_tiler_mnk 128,128,256 --cluster_shape_mnk 1,1,1 --balance_route --load_balance_mode atomic_counter"
    "E17_imbalanced_dirichlet_tiny   | --tokens_after_topk 240  --experts 53 --hidden 224  --intermediate 448  --mma_tiler_mnk 128,128,256 --cluster_shape_mnk 1,1,1 --load_balance_mode static"
    "E18_atomic_many_experts         | --tokens_after_topk 4500 --experts 32 --hidden 1280 --intermediate 1536 --mma_tiler_mnk 128,128,256 --cluster_shape_mnk 1,1,1 --load_balance_mode atomic_counter"

    "F19_T256_dirichlet_atomic_clamp | --tokens_after_topk 4500 --experts 59 --hidden 2304 --intermediate 2560 --mma_tiler_mnk 256,256,256 --cluster_shape_mnk 2,1,1 --use_2cta_instrs --load_balance_mode atomic_counter --gate_up_clamp 10"
    "F20_T256_balanced_static_stat   | --tokens_after_topk 4500 --experts 61 --hidden 2304 --intermediate 2560 --mma_tiler_mnk 256,256,256 --cluster_shape_mnk 2,1,1 --use_2cta_instrs --balance_route --load_balance_mode static --enable_static_expert_shape"
    "F21_T256_balanced_ublk_clamp3_transformers | --tokens_after_topk 4500 --experts 89 --hidden 2304 --intermediate 2560 --mma_tiler_mnk 256,256,256 --cluster_shape_mnk 2,1,1 --use_2cta_instrs --balance_route --load_balance_mode static --enable_static_expert_shape --use_bulk_fc2_store --gate_up_clamp 3 --ref_compute_graph transformers"

    # ── Group G: tile_n=64 ──
    "G22_N64_1cta_balanced_static_transformers | --tokens_after_topk 1500 --experts 67 --hidden 1280 --intermediate 1536 --mma_tiler_mnk 128,64,256 --cluster_shape_mnk 1,1,1 --balance_route --load_balance_mode static --ref_compute_graph transformers"
    "G23_N64_1cta_dirichlet_atomic   | --tokens_after_topk 3000 --experts 71 --hidden 1280 --intermediate 1536 --mma_tiler_mnk 128,64,256 --cluster_shape_mnk 1,1,1 --load_balance_mode atomic_counter"
    "G24_N64_2cta_balanced_statShape | --tokens_after_topk 2400 --experts 73 --hidden 1792 --intermediate 1536 --mma_tiler_mnk 256,64,256 --cluster_shape_mnk 2,1,1 --use_2cta_instrs --balance_route --load_balance_mode static --enable_static_expert_shape"
    "G25_N64_2cta_dirichlet_atomic   | --tokens_after_topk 4500 --experts 79 --hidden 2304 --intermediate 2560 --mma_tiler_mnk 256,64,256 --cluster_shape_mnk 2,1,1 --use_2cta_instrs --load_balance_mode atomic_counter"
    "G26_N64_cluster_m4_atomic       | --tokens_after_topk 9000 --experts 83 --hidden 2304 --intermediate 2560 --mma_tiler_mnk 256,64,256 --cluster_shape_mnk 4,1,1 --use_2cta_instrs --load_balance_mode atomic_counter"
)

PASS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0
declare -a FAIL_NAMES=()
TOTAL=${#TESTS[@]}
START_TIME=$SECONDS

# --list mode: print all test names (with selector filter applied) and exit.
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
    # Split on first '|'; trim trailing/leading whitespace on both halves.
    name="${entry%%|*}"
    name="${name%"${name##*[![:space:]]}"}"   # rstrip
    args="${entry#*|}"
    args="${args#"${args%%[![:space:]]*}"}"   # lstrip

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
    # shellcheck disable=SC2086  # intentional word-splitting on $args
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
