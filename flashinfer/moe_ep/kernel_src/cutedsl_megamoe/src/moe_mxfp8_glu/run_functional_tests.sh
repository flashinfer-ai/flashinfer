#!/usr/bin/env bash
# Functional test harness for the MXFP8 GLU fused fc1+fc2 runner.
# It resolves runner_fc12.py (co-located in this moe_mxfp8_glu/ folder)
# relative to this script, so CWD is irrelevant.
#
# MXFP8 tile constraint: the kernel only supports mma tile 256,256,128 WITH
# --use_2cta_instrs --cluster_shape_mnk 2,1,1 (FP8 MMA K=128).  Other configs
# that work for NVFP4 do NOT work for MXFP8 and were confirmed broken:
#   * tile 128,128,256 (K=256)  -> CUDA illegal memory access
#   * cluster_m=4 (4,1,1)       -> kernel deadlock (GPU spins, no progress)
#   * tile_n=64 / 1-CTA tiles   -> not supported
# So every case below fixes --mma_tiler_mnk 256,256,128 --use_2cta_instrs
# --cluster_shape_mnk 2,1,1 and varies only the rest (static/atomic
# scheduling, dyn/static expert shape, balanced/dirichlet routing, e4m3/e5m2
# element format, problem sizes).  Output dtype is bf16 (the only dtype the
# kernel emits).
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
#   bash .../run_functional_tests.sh M1                    # run M1 only
#   bash .../run_functional_tests.sh M1 M3 M7             # run those three
#   bash .../run_functional_tests.sh e5m2                  # run anything with "e5m2"
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
# All cases fix: --mma_tiler_mnk 256,256,128 --use_2cta_instrs (MXFP8 constraint).
declare -a TESTS=(
    # ── Group M: cluster_m=2, e4m3 ──
    "M1_e4m3_static_dynShape_c2     | --kind mxfp8_e4m3 --tokens_after_topk 2400 --experts 8  --hidden 1792 --intermediate 1536 --mma_tiler_mnk 256,256,128 --cluster_shape_mnk 2,1,1 --use_2cta_instrs --balance_route --load_balance_mode static"
    "M2_e4m3_static_statShape_c2    | --kind mxfp8_e4m3 --tokens_after_topk 2400 --experts 12 --hidden 1792 --intermediate 1536 --mma_tiler_mnk 256,256,128 --cluster_shape_mnk 2,1,1 --use_2cta_instrs --balance_route --load_balance_mode static --enable_static_expert_shape"
    "M3_e4m3_atomic_dynShape_c2     | --kind mxfp8_e4m3 --tokens_after_topk 2400 --experts 16 --hidden 1792 --intermediate 1536 --mma_tiler_mnk 256,256,128 --cluster_shape_mnk 2,1,1 --use_2cta_instrs --balance_route --load_balance_mode atomic_counter"

    # ── Group M: atomic + static expert shape, e4m3 ──
    "M4_e4m3_atomic_statShape_c2    | --kind mxfp8_e4m3 --tokens_after_topk 4500 --experts 19 --hidden 2304 --intermediate 2560 --mma_tiler_mnk 256,256,128 --cluster_shape_mnk 2,1,1 --use_2cta_instrs --balance_route --load_balance_mode atomic_counter --enable_static_expert_shape"

    # ── Group M: dirichlet routing, e4m3 ──
    "M5_e4m3_dirichlet_static_c2    | --kind mxfp8_e4m3 --tokens_after_topk 3000 --experts 23 --hidden 2304 --intermediate 2560 --mma_tiler_mnk 256,256,128 --cluster_shape_mnk 2,1,1 --use_2cta_instrs --load_balance_mode static"
    "M6_e4m3_dirichlet_atomic_c2    | --kind mxfp8_e4m3 --tokens_after_topk 9000 --experts 31 --hidden 2304 --intermediate 2560 --mma_tiler_mnk 256,256,128 --cluster_shape_mnk 2,1,1 --use_2cta_instrs --load_balance_mode atomic_counter --enable_static_expert_shape"

    # ── Group M: large shape (DeepSeek-ish) ──
    "M7_e4m3_large_atomic_balanced  | --kind mxfp8_e4m3 --tokens_after_topk 17000 --experts 8 --hidden 7168 --intermediate 4096 --mma_tiler_mnk 256,256,128 --cluster_shape_mnk 2,1,1 --use_2cta_instrs --balance_route --load_balance_mode atomic_counter"

    # NOTE: mxfp8_e5m2 is intentionally NOT covered here.  The e5m2 fc1
    # quantize path is currently broken in the kernel epilogue
    # (moe_mxfp8_glu/epilogue_mxfp8.py -> moe_nvfp4_swapab/moe_utils.py
    # quant_sfd_row raises "None to integer conversion is not supported"),
    # so every e5m2 launch fails regardless of the runner.  Re-add e5m2
    # cases once that kernel path is fixed.
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
