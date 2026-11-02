#!/usr/bin/env bash
# Functional test harness for the BF16 GLU fused fc1+fc2 runner.
# It resolves runner_fc12.py (co-located in this moe_hopper_bf16/ folder)
# relative to this script, so CWD is irrelevant.
#
# Hopper BF16 supports cluster 1,1,1 / 2,1,1 / 1,2,1 / 2,2,1, selected with
# BF16_CLUSTER_SHAPE (default 1,1,1). Non-swap selects M=64 and N=128/256
# with BF16_NON_SWAP_M/N (defaults 64/128).
# Swap-AB uses M=128/256, K=128 and selects M/N with
# BF16_SWAP_AB_M=128/256 (default 256) and
# BF16_SWAP_AB_N=8/16/32/64/128 (default 32; 8 is experimental).
#
# Usage:
#   bash <abs path>/run_functional_tests.sh
#   bash <abs path>/run_functional_tests.sh --swapab
#   bash <abs path>/run_functional_tests.sh --pingpong
#   PYTHON=python3.11 bash .../run_functional_tests.sh
#   bash .../run_functional_tests.sh --fail-fast
#   bash .../run_functional_tests.sh --list
#   bash .../run_functional_tests.sh --help
#
# Variant selection:
#   Activations and weights are BF16; there is no quantization knob.
#   --swapab selects swap-AB; without it, the test uses non-swap.
#   --pingpong alternates complete tasks across two WGMMA+epilogue warpgroups.
#   Each variant runs its six base cases plus every legal M/N tile.
#   Use a Txx selector to run only the tile matrix cases.
#   BF16_NON_SWAP_M=64 BF16_NON_SWAP_N=128 bash .../run_functional_tests.sh M1
#   BF16_SWAP_AB_M=256 BF16_SWAP_AB_N=32 bash .../run_functional_tests.sh --swapab M1
#   BF16_TAIL_SPLIT=1 appends --tail_split_pairs (needs a 2-CTA token cluster:
#   BF16_CLUSTER_SHAPE=1,2,1 with --swapab or 2,1,1 without), e.g.
#   BF16_TAIL_SPLIT=1 BF16_CLUSTER_SHAPE=1,2,1 bash .../run_functional_tests.sh --swapab

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
    SELECTED_TILE_M=$BF16_SWAP_AB_M
    SELECTED_TILE_N=$BF16_SWAP_AB_N
    if [ "$PINGPONG" -eq 1 ] && [ "$SELECTED_TILE_M" -ne 128 ]; then
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
    SELECTED_TILE_M=$BF16_NON_SWAP_M
    SELECTED_TILE_N=$BF16_NON_SWAP_N
    if [ "$PINGPONG" -eq 1 ] && [ "$SELECTED_TILE_N" -ne 128 ]; then
        echo "ERROR: non-swap ping-pong requires BF16_NON_SWAP_N=128" >&2
        exit 2
    fi
fi

if [ ! -f "$RUNNER" ]; then
    echo "ERROR: runner_fc12.py not found at ${RUNNER}" >&2
    exit 2
fi

BF16_CLUSTER_SHAPE="${BF16_CLUSTER_SHAPE:-1,1,1}"
case "$BF16_CLUSTER_SHAPE" in
    1,1,1|2,1,1|1,2,1|2,2,1)
        ;;
    *)
        echo "ERROR: BF16_CLUSTER_SHAPE must be 1,1,1, 2,1,1, 1,2,1, or 2,2,1" >&2
        exit 2
        ;;
esac

# BF16_TAIL_SPLIT=1: the whole run shares one geometry, so reject an
# incompatible cluster shape up front.
BF16_TAIL_SPLIT="${BF16_TAIL_SPLIT:-0}"
declare -a TAIL_SPLIT_ARGS=()
case "$BF16_TAIL_SPLIT" in
    1)
        if { [ "$SWAP_AB" -eq 1 ] && [ "$BF16_CLUSTER_SHAPE" = "1,2,1" ]; } \
            || { [ "$SWAP_AB" -eq 0 ] && [ "$BF16_CLUSTER_SHAPE" = "2,1,1" ]; }; then
            TAIL_SPLIT_ARGS=(--tail_split_pairs)
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
# All base cases use --mma_tiler_mnk 64,128,64 without --use_2cta_instrs.
declare -a BASE_TESTS=(
    # ── Balanced routing ──
    "M1_bf16_static_dynShape_c1     | --kind bf16 --tokens_after_topk 2400 --experts 8  --hidden 1792 --intermediate 1536 --mma_tiler_mnk 64,128,64 --cluster_shape_mnk 1,1,1 --balance_route --load_balance_mode static"
    "M2_bf16_static_statShape_c1    | --kind bf16 --tokens_after_topk 2400 --experts 12 --hidden 1792 --intermediate 1536 --mma_tiler_mnk 64,128,64 --cluster_shape_mnk 1,1,1 --balance_route --load_balance_mode static --enable_static_expert_shape"
    "M3_bf16_atomic_dynShape_c1     | --kind bf16 --tokens_after_topk 2400 --experts 16 --hidden 1792 --intermediate 1536 --mma_tiler_mnk 64,128,64 --cluster_shape_mnk 1,1,1 --balance_route --load_balance_mode atomic_counter"

    # ── Atomic + static expert shape ──
    "M4_bf16_atomic_statShape_c1    | --kind bf16 --tokens_after_topk 4500 --experts 19 --hidden 2304 --intermediate 2560 --mma_tiler_mnk 64,128,64 --cluster_shape_mnk 1,1,1 --balance_route --load_balance_mode atomic_counter --enable_static_expert_shape"

    # ── Dirichlet routing ──
    "M5_bf16_dirichlet_static_c1    | --kind bf16 --tokens_after_topk 3000 --experts 23 --hidden 2304 --intermediate 2560 --mma_tiler_mnk 64,128,64 --cluster_shape_mnk 1,1,1 --load_balance_mode static"
    "M6_bf16_dirichlet_atomic_c1    | --kind bf16 --tokens_after_topk 9000 --experts 31 --hidden 2304 --intermediate 2560 --mma_tiler_mnk 64,128,64 --cluster_shape_mnk 1,1,1 --load_balance_mode atomic_counter --enable_static_expert_shape"

    # -- generate_c: raw pre-SwiGLU fc1 gate+up output (training forward).
    # Runs under whichever operand order the script was invoked with, so the
    # single entry covers both the non-swap and the --swapab suites. --
    "M7_bf16_generate_c_statShape   | --kind bf16 --tokens_after_topk 4500 --experts 19 --hidden 2304 --intermediate 2560 --mma_tiler_mnk 64,128,64 --cluster_shape_mnk 1,1,1 --balance_route --load_balance_mode atomic_counter --enable_static_expert_shape --generate_c"
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
    "T11_swapab_m128_n8   | 1 | 128 | 8"
    "T12_swapab_m256_n8   | 1 | 256 | 8"
)

# One balanced/static problem is enough to compile and execute each geometry.
# H=I=1024 gives sixteen K=64 tiles; 2048 routed tokens covers the M=256
# geometry.
MATRIX_TEST_ARGS="--kind bf16 --tokens_after_topk 2048 --experts 8 --hidden 1024 --intermediate 1024 --cluster_shape_mnk 1,1,1 --balance_route --load_balance_mode static --enable_static_expert_shape"

declare -a ACTIVE_NAMES=()
declare -a ACTIVE_ARGS=()
declare -a ACTIVE_SWAP_AB=()
declare -a ACTIVE_TILE_M=()
declare -a ACTIVE_TILE_N=()

add_active_test() {
    ACTIVE_NAMES+=("$1")
    ACTIVE_ARGS+=("$2")
    ACTIVE_SWAP_AB+=("$3")
    ACTIVE_TILE_M+=("$4")
    ACTIVE_TILE_N+=("$5")
}

for entry in "${BASE_TESTS[@]}"; do
    name="${entry%%|*}"
    name="${name%"${name##*[![:space:]]}"}"
    args="${entry#*|}"
    args="${args#"${args%%[![:space:]]*}"}"
    add_active_test "$name" "$args" "$SWAP_AB" \
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
    if [ "$PINGPONG" -eq 1 ]; then
        if [ "$case_swap_ab" -eq 0 ] && [ "$tile_n" -ne 128 ]; then
            continue
        fi
        if [ "$case_swap_ab" -eq 1 ] && [ "$tile_m" -ne 128 ]; then
            continue
        fi
    fi
    add_active_test "$tile_name" "$MATRIX_TEST_ARGS" \
        "$case_swap_ab" "$tile_m" "$tile_n"
done

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
        full_name="${name}"
        if test_matches_selectors "$full_name"; then
            echo "$full_name"
        fi
    done
    exit 0
fi

for index in "${!ACTIVE_NAMES[@]}"; do
        name="${ACTIVE_NAMES[$index]}"
        args="${ACTIVE_ARGS[$index]}"
        case_swap_ab="${ACTIVE_SWAP_AB[$index]}"
        tile_m="${ACTIVE_TILE_M[$index]}"
        tile_n="${ACTIVE_TILE_N[$index]}"
        full_name="${name}"

        case_tile_args=(
            --mma_tiler_mnk "${tile_m},${tile_n},64"
            --cluster_shape_mnk "$BF16_CLUSTER_SHAPE"
        )
        if [ "$case_swap_ab" -eq 1 ]; then
            case_tile_args=(--swap_ab "${case_tile_args[@]}")
        fi
        if [ "$PINGPONG" -eq 1 ]; then
            case_tile_args=(--pingpong "${case_tile_args[@]}")
        fi
        if [ "${#TAIL_SPLIT_ARGS[@]}" -gt 0 ]; then
            case_tile_args+=("${TAIL_SPLIT_ARGS[@]}")
        fi

        if ! test_matches_selectors "$full_name"; then
            SKIP_COUNT=$((SKIP_COUNT + 1))
            continue
        fi

        echo
        echo "==========================================================================="
        echo "[TEST] $full_name"
        echo "[MODE] swap_ab=$case_swap_ab pingpong=$PINGPONG tile=${tile_m},${tile_n},64 tail_split=$BF16_TAIL_SPLIT"
        echo "[CMD]  $PYTHON $RUNNER $args ${case_tile_args[*]}"
        echo "==========================================================================="

        test_start=$SECONDS
        # shellcheck disable=SC2086  # intentional word-splitting on $args
        timeout 300 "$PYTHON" "$RUNNER" $args "${case_tile_args[@]}"
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
