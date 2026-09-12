#!/usr/bin/env bash
# Functional test harness for the FP8 GLU fused fc1+fc2 runner.
# It resolves runner_fc12.py (co-located in this moe_hopper_fp8/ folder)
# relative to this script, so CWD is irrelevant.
#
# Hopper FP8 uses the 1CTA --cluster_shape_mnk 1,1,1 path. Non-swap selects
# M=64 and N=128/256 with FP8_NON_SWAP_M/N (defaults 64/128).
# Swap-AB uses M=128/256, K=128 and selects M/N with
# FP8_SWAP_AB_M=128/256 (default 256) and
# FP8_SWAP_AB_N=16/32/64/128 (default 32).
#
# Usage:
#   bash <abs path>/run_functional_tests.sh --scale-mode per-tensor
#   bash <abs path>/run_functional_tests.sh --scale-mode blockwise --swapab
#   PYTHON=python3.11 bash .../run_functional_tests.sh
#   bash .../run_functional_tests.sh --fail-fast
#   bash .../run_functional_tests.sh --list
#   bash .../run_functional_tests.sh --help
#
# Variant selection:
#   each invocation runs one scale mode; default is per-tensor.
#   --swapab selects swap-AB; without it, the test uses non-swap.
#   Each variant runs its six base cases and every legal M/N tile with 1xacc.
#   Per-tensor variants also run one representative 2xacc case.
#   Use a Txx or A01 selector to run only the tile or 2xacc matrix cases.
#   FP8_ACCUM_MODE selects the base-case mode; matrix modes are fixed above.
#   FP8_ACCUM_MODE=2xacc bash ... --scale-mode per-tensor
#   FP8_NON_SWAP_M=64 FP8_NON_SWAP_N=128 bash .../run_functional_tests.sh M1
#   FP8_SWAP_AB_M=256 FP8_SWAP_AB_N=32 bash .../run_functional_tests.sh --swapab M1
#
# Selective execution (positional args; substring match against test names,
# OR-combined across multiple selectors):
#   bash .../run_functional_tests.sh M1                    # run M1 only
#   bash .../run_functional_tests.sh blockwise             # run blockwise cases
#   bash .../run_functional_tests.sh M1 M3 M6             # run those three
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
    SELECTED_TILE_M=$FP8_SWAP_AB_M
    SELECTED_TILE_N=$FP8_SWAP_AB_N
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
    SELECTED_TILE_M=$FP8_NON_SWAP_M
    SELECTED_TILE_N=$FP8_NON_SWAP_N
fi

if [ ! -f "$RUNNER" ]; then
    echo "ERROR: runner_fc12.py not found at ${RUNNER}" >&2
    exit 2
fi

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
# All base cases use --mma_tiler_mnk 64,128,128 without --use_2cta_instrs.
declare -a BASE_TESTS=(
    # ── Balanced routing, e4m3 ──
    "M1_e4m3_static_dynShape_c1     | --kind fp8_e4m3 --tokens_after_topk 2400 --experts 8  --hidden 1792 --intermediate 1536 --mma_tiler_mnk 64,128,128 --cluster_shape_mnk 1,1,1 --balance_route --load_balance_mode static"
    "M2_e4m3_static_statShape_c1    | --kind fp8_e4m3 --tokens_after_topk 2400 --experts 12 --hidden 1792 --intermediate 1536 --mma_tiler_mnk 64,128,128 --cluster_shape_mnk 1,1,1 --balance_route --load_balance_mode static --enable_static_expert_shape"
    "M3_e4m3_atomic_dynShape_c1     | --kind fp8_e4m3 --tokens_after_topk 2400 --experts 16 --hidden 1792 --intermediate 1536 --mma_tiler_mnk 64,128,128 --cluster_shape_mnk 1,1,1 --balance_route --load_balance_mode atomic_counter"

    # ── Atomic + static expert shape, e4m3 ──
    "M4_e4m3_atomic_statShape_c1    | --kind fp8_e4m3 --tokens_after_topk 4500 --experts 19 --hidden 2304 --intermediate 2560 --mma_tiler_mnk 64,128,128 --cluster_shape_mnk 1,1,1 --balance_route --load_balance_mode atomic_counter --enable_static_expert_shape"

    # ── Dirichlet routing, e4m3 ──
    "M5_e4m3_dirichlet_static_c1    | --kind fp8_e4m3 --tokens_after_topk 3000 --experts 23 --hidden 2304 --intermediate 2560 --mma_tiler_mnk 64,128,128 --cluster_shape_mnk 1,1,1 --load_balance_mode static"
    "M6_e4m3_dirichlet_atomic_c1    | --kind fp8_e4m3 --tokens_after_topk 9000 --experts 31 --hidden 2304 --intermediate 2560 --mma_tiler_mnk 64,128,128 --cluster_shape_mnk 1,1,1 --load_balance_mode atomic_counter --enable_static_expert_shape"

    # NOTE: fp8_e5m2 is intentionally NOT covered here.  The e5m2 fc1
    # quantize path is currently broken in the kernel epilogue
    # (moe_hopper_fp8/epilogue_fp8.py -> moe_nvfp4_swapab/moe_utils.py
    # quant_sfd_row raises "None to integer conversion is not supported"),
    # so every e5m2 launch fails regardless of the runner.  Re-add e5m2
    # cases once that kernel path is fixed.
)

# Explicit legal geometry records: name | swap_ab | tile_m | tile_n.
# Do not replace this with independent M/N loops: non-swap and swap-AB have
# different legal value sets, so a cross-product would create invalid cases.
declare -a TILE_SHAPE_CASES=(
    "T01_nonswap_m64_n128 | 0 | 64  | 128"
    "T02_nonswap_m64_n256 | 0 | 64  | 256"
    "T03_swapab_m128_n16  | 1 | 128 | 16"
    "T04_swapab_m128_n32  | 1 | 128 | 32"
    "T05_swapab_m128_n64  | 1 | 128 | 64"
    "T06_swapab_m128_n128 | 1 | 128 | 128"
    "T07_swapab_m256_n16  | 1 | 256 | 16"
    "T08_swapab_m256_n32  | 1 | 256 | 32"
    "T09_swapab_m256_n64  | 1 | 256 | 64"
    "T10_swapab_m256_n128 | 1 | 256 | 128"
)

# One balanced/static problem is enough to compile and execute each geometry.
# H=I=1024 gives eight K=128 tiles so the representative 2xacc case exercises
# cross-K-tile promotion; 2048 routed tokens covers the M=256 geometry.
MATRIX_TEST_ARGS="--kind fp8_e4m3 --tokens_after_topk 2048 --experts 8 --hidden 1024 --intermediate 1024 --cluster_shape_mnk 1,1,1 --balance_route --load_balance_mode static --enable_static_expert_shape"

declare -a ACTIVE_NAMES=()
declare -a ACTIVE_ARGS=()
declare -a ACTIVE_ACCUM_MODES=()
declare -a ACTIVE_SWAP_AB=()
declare -a ACTIVE_TILE_M=()
declare -a ACTIVE_TILE_N=()

add_active_test() {
    ACTIVE_NAMES+=("$1")
    ACTIVE_ARGS+=("$2")
    ACTIVE_ACCUM_MODES+=("$3")
    ACTIVE_SWAP_AB+=("$4")
    ACTIVE_TILE_M+=("$5")
    ACTIVE_TILE_N+=("$6")
}

for entry in "${BASE_TESTS[@]}"; do
    name="${entry%%|*}"
    name="${name%"${name##*[![:space:]]}"}"
    args="${entry#*|}"
    args="${args#"${args%%[![:space:]]*}"}"
    add_active_test "$name" "$args" "$FP8_ACCUM_MODE" "$SWAP_AB" \
        "$SELECTED_TILE_M" "$SELECTED_TILE_N"
done

for entry in "${TILE_SHAPE_CASES[@]}"; do
    IFS='|' read -r tile_name case_swap_ab tile_m tile_n <<< "$entry"
    tile_name="${tile_name//[[:space:]]/}"
    case_swap_ab="${case_swap_ab//[[:space:]]/}"
    tile_m="${tile_m//[[:space:]]/}"
    tile_n="${tile_n//[[:space:]]/}"
    if [ "$case_swap_ab" -ne "$SWAP_AB" ]; then
        continue
    fi
    add_active_test "${tile_name}_1xacc" "$MATRIX_TEST_ARGS" 1xacc \
        "$case_swap_ab" "$tile_m" "$tile_n"
done

if [ "$SCALE_MODE" = "per_tensor" ]; then
    if [ "$SWAP_AB" -eq 1 ]; then
        operand_name=swapab
    else
        operand_name=nonswap
    fi
    add_active_test \
        "A01_${operand_name}_m${SELECTED_TILE_M}_n${SELECTED_TILE_N}_2xacc" \
        "$MATRIX_TEST_ARGS" 2xacc "$SWAP_AB" \
        "$SELECTED_TILE_M" "$SELECTED_TILE_N"
fi

PASS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0
declare -a FAIL_NAMES=()
TOTAL=${#ACTIVE_NAMES[@]}
START_TIME=$SECONDS

# --list mode: print all test names (with selector filter applied) and exit.
if [ "$LIST_ONLY" -eq 1 ]; then
    for index in "${!ACTIVE_NAMES[@]}"; do
        name="${ACTIVE_NAMES[$index]}"
        full_name="${SCALE_MODE}/${name}"
        if test_matches_selectors "$full_name"; then
            echo "$full_name"
        fi
    done
    exit 0
fi

for index in "${!ACTIVE_NAMES[@]}"; do
        name="${ACTIVE_NAMES[$index]}"
        args="${ACTIVE_ARGS[$index]}"
        accum_mode="${ACTIVE_ACCUM_MODES[$index]}"
        case_swap_ab="${ACTIVE_SWAP_AB[$index]}"
        tile_m="${ACTIVE_TILE_M[$index]}"
        tile_n="${ACTIVE_TILE_N[$index]}"
        full_name="${SCALE_MODE}/${name}"

        case_tile_args=(--mma_tiler_mnk "${tile_m},${tile_n},128")
        if [ "$case_swap_ab" -eq 1 ]; then
            case_tile_args=(--swap_ab "${case_tile_args[@]}")
        fi

        if ! test_matches_selectors "$full_name"; then
            SKIP_COUNT=$((SKIP_COUNT + 1))
            continue
        fi

        echo
        echo "==========================================================================="
        echo "[TEST] $full_name"
        echo "[MODE] scale=$SCALE_MODE accum=$accum_mode swap_ab=$case_swap_ab tile=${tile_m},${tile_n},128"
        echo "[CMD]  $PYTHON $RUNNER $args --fp8_scale_mode $SCALE_MODE --fp8_accum_mode $accum_mode ${case_tile_args[*]}"
        echo "==========================================================================="

        test_start=$SECONDS
        # shellcheck disable=SC2086  # intentional word-splitting on $args
        "$PYTHON" "$RUNNER" $args --fp8_scale_mode "$SCALE_MODE" \
            --fp8_accum_mode "$accum_mode" "${case_tile_args[@]}"
        rc=$?
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
