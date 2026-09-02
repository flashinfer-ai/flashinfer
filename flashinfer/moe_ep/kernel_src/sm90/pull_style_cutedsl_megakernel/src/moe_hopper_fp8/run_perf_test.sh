#!/usr/bin/env bash
# DSV4 performance harness for moe_hopper_fp8.
#
# Config source:
#   Hugging Face DeepSeek-V4-Pro config.json plus local src/config.py DSV4:
#   hidden=7168, topk=6, total experts=384, swiglu_limit=10.0,
#   tokens/rank=1024. The fused fc12 runner uses gate+up as its
#   --intermediate value, so moe_intermediate_size=3072 maps to
#   --intermediate=6144.
#
# Tests:
#   P01: single GPU standalone runner_fc12.py, one DSV4 local shard.
#   P02: single GPU MEGA_NO_DIST=1 mega_runner.py, one DSV4 local
#        shard without distributed comm. Set DSV4_SINGLE_MEGA_TOTAL_EXPERTS=384
#        to stress a single rank that owns the full expert set.
#   P03: 4 GPU torchrun mega_runner.py, DSV4 EP view.
#
# Usage:
#   bash moe_hopper_fp8/run_perf_test.sh --scale-mode per-tensor
#   bash moe_hopper_fp8/run_perf_test.sh --scale-mode blockwise --no-heuristic --swapab
#   bash moe_hopper_fp8/run_perf_test.sh --scale-mode per-tensor --heuristic P03
#   bash moe_hopper_fp8/run_perf_test.sh --scale-mode per-tensor --no-heuristic --pingpong P03
#   bash moe_hopper_fp8/run_perf_test.sh --list
#   bash moe_hopper_fp8/run_perf_test.sh P01
#   PERF_WARMUP=3 PERF_ITERS=30 bash moe_hopper_fp8/run_perf_test.sh mega
#   bash moe_hopper_fp8/run_perf_test.sh P03
#
# Variant selection:
#   each invocation runs one scale mode; default is per-tensor.
#   MegaMoE uses token/scale heuristics by default. --no-heuristic restores
#   manual geometry; --swapab or --pingpong also imply manual geometry.
#   FP8_ACCUM_MODE=2xacc bash ... --scale-mode per-tensor    # compare accumulation
#   FP8_CLUSTER_SHAPE=2,2,1 bash ... --scale-mode per-tensor P01 P02
#   FP8_NON_SWAP_M=64 FP8_NON_SWAP_N=128 bash .../run_perf_test.sh P01 P02
#   FP8_SWAP_AB_M=256 FP8_SWAP_AB_N=32 bash .../run_perf_test.sh --swapab P01

set -uo pipefail

export PATH=/usr/bin:$PATH
export LD=/usr/bin/ld
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export CUDAHOSTCXX=/usr/bin/g++
export TRITON_CC=/usr/bin/gcc
export CFLAGS="-B/usr/bin"
export CXXFLAGS="-B/usr/bin"
export LDFLAGS="-B/usr/bin -fuse-ld=bfd"
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PYTHON="${PYTHON:-python}"
FC12_RUNNER="${SCRIPT_DIR}/runner_fc12.py"
MEGA_RUNNER="${SCRIPT_DIR}/mega_runner.py"

if [ ! -f "$FC12_RUNNER" ]; then
    echo "ERROR: runner_fc12.py not found at ${FC12_RUNNER}" >&2
    exit 2
fi
if [ ! -f "$MEGA_RUNNER" ]; then
    echo "ERROR: mega_runner.py not found at ${MEGA_RUNNER}" >&2
    exit 2
fi

SCALE_MODE="${FP8_SCALE_MODE:-per_tensor}"
SWAP_AB=0
PINGPONG=0
USE_HEURISTIC=1
HEURISTIC_EXPLICIT=0
MANUAL_CONFIG_REQUESTED=0
for config_var in \
    FP8_CLUSTER_SHAPE \
    FP8_NON_SWAP_M \
    FP8_NON_SWAP_N \
    FP8_SWAP_AB_M \
    FP8_SWAP_AB_N; do
    if [[ -v "$config_var" ]]; then
        MANUAL_CONFIG_REQUESTED=1
        break
    fi
done
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
            MANUAL_CONFIG_REQUESTED=1
            shift
            ;;
        --pingpong)
            PINGPONG=1
            MANUAL_CONFIG_REQUESTED=1
            shift
            ;;
        --heuristic)
            USE_HEURISTIC=1
            HEURISTIC_EXPLICIT=1
            shift
            ;;
        --no-heuristic)
            USE_HEURISTIC=0
            HEURISTIC_EXPLICIT=1
            shift
            ;;
        --list)
            LIST_ONLY=1
            shift
            ;;
        -h|--help)
            sed -n '2,34p' "${BASH_SOURCE[0]}"
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

if [ "$USE_HEURISTIC" -eq 1 ] && [ "$MANUAL_CONFIG_REQUESTED" -eq 1 ]; then
    if [ "$HEURISTIC_EXPLICIT" -eq 1 ]; then
        echo "ERROR: --heuristic cannot be combined with manual launch settings" >&2
        exit 2
    fi
    USE_HEURISTIC=0
fi

# DSV4 defaults from src/config.py plus the fused-fc12 gate+up convention.
DSV4_TOKENS_PER_RANK="${DSV4_TOKENS_PER_RANK:-1024}"
DSV4_TOPK="${DSV4_TOPK:-6}"
DSV4_TOTAL_EXPERTS="${DSV4_TOTAL_EXPERTS:-384}"
DSV4_EP_SIZE=4
if [ $((DSV4_TOTAL_EXPERTS % DSV4_EP_SIZE)) -ne 0 ]; then
    echo "ERROR: DSV4_TOTAL_EXPERTS must be divisible by DSV4_EP_SIZE" >&2
    exit 2
fi
DSV4_LOCAL_EXPERTS="${DSV4_LOCAL_EXPERTS:-$((DSV4_TOTAL_EXPERTS / DSV4_EP_SIZE))}"
DSV4_HIDDEN="${DSV4_HIDDEN:-7168}"
DSV4_INTERMEDIATE_DOWNPROJ="${DSV4_INTERMEDIATE_DOWNPROJ:-3072}"
DSV4_INTERMEDIATE_GATEUP="${DSV4_INTERMEDIATE_GATEUP:-$((DSV4_INTERMEDIATE_DOWNPROJ * 2))}"
DSV4_ROUTE_ROWS="${DSV4_ROUTE_ROWS:-$((DSV4_TOKENS_PER_RANK * DSV4_TOPK))}"
DSV4_GATE_UP_CLAMP="${DSV4_GATE_UP_CLAMP:-10.0}"
GATE_UP_CLAMP_ARGS=""
if [ -n "$DSV4_GATE_UP_CLAMP" ]; then
    GATE_UP_CLAMP_ARGS="--gate_up_clamp $DSV4_GATE_UP_CLAMP"
fi

# P02 defaults to the same local expert shard as P01/P03. A true distributed
# DSV4 run still uses DSV4_TOTAL_EXPERTS in P03; single-rank cannot express
# world_size=4 with total_experts=384 without launching 4 ranks.
DSV4_SINGLE_MEGA_TOTAL_EXPERTS="${DSV4_SINGLE_MEGA_TOTAL_EXPERTS:-$DSV4_LOCAL_EXPERTS}"

FP8_CLUSTER_SHAPE="${FP8_CLUSTER_SHAPE:-1,1,1}"
case "$FP8_CLUSTER_SHAPE" in
    1,1,1|2,1,1|1,2,1|2,2,1)
        ;;
    *)
        echo "ERROR: FP8_CLUSTER_SHAPE must be 1,1,1, 2,1,1, 1,2,1, or 2,2,1" >&2
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
    TILE_ARGS="--swap_ab --mma_tiler_mnk ${FP8_SWAP_AB_M},${FP8_SWAP_AB_N},128 --cluster_shape_mnk ${FP8_CLUSTER_SHAPE}"
    if [ "$PINGPONG" -eq 1 ] && [ "$FP8_SWAP_AB_M" -ne 128 ]; then
        echo "ERROR: swap-AB ping-pong requires FP8_SWAP_AB_M=128" >&2
        exit 2
    fi
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
    TILE_ARGS="--mma_tiler_mnk ${FP8_NON_SWAP_M},${FP8_NON_SWAP_N},128 --cluster_shape_mnk ${FP8_CLUSTER_SHAPE}"
    if [ "$PINGPONG" -eq 1 ] && [ "$FP8_NON_SWAP_N" -ne 128 ]; then
        echo "ERROR: non-swap ping-pong requires FP8_NON_SWAP_N=128" >&2
        exit 2
    fi
fi
if [ "$PINGPONG" -eq 1 ]; then
    TILE_ARGS="--pingpong $TILE_ARGS"
fi
FC12_TILE_ARGS="$TILE_ARGS"
MEGA_TILE_ARGS="$TILE_ARGS"
if [ "$USE_HEURISTIC" -eq 1 ]; then
    # Leaving MegaMoE geometry unspecified lets mega_runner.py select the
    # token/scale-dependent launch configuration from heuristic_config.py.
    MEGA_TILE_ARGS=""
fi
COMMON_KIND_ARGS="--kind fp8_e4m3"
COMMON_PERF_ARGS="--perf_run --skip_ref_check"
FP8_ACCUM_MODE="${FP8_ACCUM_MODE:-1xacc}"
case "$FP8_ACCUM_MODE" in
    1xacc|2xacc)
        ;;
    *)
        echo "ERROR: unsupported FP8 accumulation mode '$FP8_ACCUM_MODE'" >&2
        exit 2
        ;;
esac

# MegaMoE combine (fc2 output) surface, mirroring the NVFP4 naming.  P01 is
# the standalone fc12 runner and has no token communication, so it ignores
# both knobs and is combine-mode invariant.
#   FP8_TOKEN_BACK_MODE : who performs the cross-rank fc2 write-back.
#     epi_warps            - fc2 epilogue STG/REDG straight to the source rank
#     reuse_dispatch_warps - stage locally, dispatch warps push
#     standalone_warps     - stage locally, four dedicated warps push
#     unset (default)      - the token-bucket heuristic table's per-bucket
#                            winner (heuristic_config.py)
#   FP8_IN_KERNEL_REDUCE : 1 collapses topk in kernel (epi_warps REDG or the
#     dispatch modes' cp.reduce bulk push); 0 runs the separate TopkReduce.
FP8_TOKEN_BACK_MODE="${FP8_TOKEN_BACK_MODE:-heuristic}"
MEGA_COMBINE_ARGS=""
case "$FP8_TOKEN_BACK_MODE" in
    heuristic)
        ;;
    epi_warps|standalone_warps|reuse_dispatch_warps)
        MEGA_COMBINE_ARGS="--token_back_mode $FP8_TOKEN_BACK_MODE"
        ;;
    *)
        echo "ERROR: unsupported FP8 token-back mode '$FP8_TOKEN_BACK_MODE'" >&2
        exit 2
        ;;
esac
FP8_IN_KERNEL_REDUCE="${FP8_IN_KERNEL_REDUCE:-0}"
case "$FP8_IN_KERNEL_REDUCE" in
    1)
        MEGA_COMBINE_ARGS="$MEGA_COMBINE_ARGS --in_kernel_fc2_reduce"
        ;;
    0)
        ;;
    *)
        echo "ERROR: FP8_IN_KERNEL_REDUCE must be 0 or 1" >&2
        exit 2
        ;;
esac

PERF_WARMUP="${PERF_WARMUP:-3}"
PERF_ITERS="${PERF_ITERS:-20}"
FC12_WARMUP="${FC12_WARMUP:-$PERF_WARMUP}"
FC12_ITERS="${FC12_ITERS:-$PERF_ITERS}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-3600}"

# Current H200 multi-rank path needs these disabled unless the environment has
# working NCCL/NVSHMEM NVLS setup. Users can override either variable.
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export NVSHMEM_DISABLE_NVLS="${NVSHMEM_DISABLE_NVLS:-1}"

FC12_DSV4_ARGS="$COMMON_KIND_ARGS \
  --tokens_after_topk $DSV4_ROUTE_ROWS \
  --experts $DSV4_LOCAL_EXPERTS \
  --hidden $DSV4_HIDDEN \
  --intermediate $DSV4_INTERMEDIATE_GATEUP \
  $FC12_TILE_ARGS \
  --balance_route \
  --load_balance_mode atomic_counter \
  --enable_static_expert_shape \
  $GATE_UP_CLAMP_ARGS \
  --perf_warmup $FC12_WARMUP \
  --perf_iters $FC12_ITERS \
  $COMMON_PERF_ARGS"

MEGA_DSV4_SINGLE_ARGS="$COMMON_KIND_ARGS \
  --num_tokens_per_rank $DSV4_TOKENS_PER_RANK \
  --num_topk $DSV4_TOPK \
  --num_total_experts $DSV4_SINGLE_MEGA_TOTAL_EXPERTS \
  --hidden $DSV4_HIDDEN \
  --intermediate $DSV4_INTERMEDIATE_GATEUP \
  $MEGA_TILE_ARGS \
  --route_distribution balanced \
  --load_balance_mode atomic_counter \
  --enable_static_expert_shape \
  $GATE_UP_CLAMP_ARGS \
  $COMMON_PERF_ARGS \
  --use_torch_profiler \
  --perf_warmup $PERF_WARMUP \
  --perf_iters $PERF_ITERS"

MEGA_DSV4_MULTI_ARGS="$COMMON_KIND_ARGS \
  --num_tokens_per_rank $DSV4_TOKENS_PER_RANK \
  --num_topk $DSV4_TOPK \
  --num_total_experts $DSV4_TOTAL_EXPERTS \
  --hidden $DSV4_HIDDEN \
  --intermediate $DSV4_INTERMEDIATE_GATEUP \
  $MEGA_TILE_ARGS \
  --route_distribution balanced \
  --load_balance_mode atomic_counter \
  --enable_static_expert_shape \
  $GATE_UP_CLAMP_ARGS \
  $COMMON_PERF_ARGS \
  --use_torch_profiler \
  --perf_warmup $PERF_WARMUP \
  --perf_iters $PERF_ITERS"

declare -a TESTS=(
    "P01_fc12_single_dsv4_local | fc12        | $FC12_DSV4_ARGS"
    "P02_mega_single_dsv4_local | mega_single | $MEGA_DSV4_SINGLE_ARGS"
    "P03_mega_4rank_dsv4        | mega_multi  | $MEGA_DSV4_MULTI_ARGS"
)

test_matches_selectors() {
    local name="$1"
    local run_type="$2"
    if [ "${#SELECTORS[@]}" -eq 0 ]; then
        return 0
    fi
    local sel
    for sel in "${SELECTORS[@]}"; do
        if [[ "$name" == *"$sel"* || "$run_type" == *"$sel"* ]]; then
            return 0
        fi
    done
    return 1
}

print_config() {
    echo "==========================================================================="
    echo "Hopper FP8 DSV4 performance tests"
    echo "  PYTHON                 : $PYTHON"
    echo "  FC12_RUNNER            : $FC12_RUNNER"
    echo "  MEGA_RUNNER            : $MEGA_RUNNER"
    echo "  tokens_per_rank        : $DSV4_TOKENS_PER_RANK"
    echo "  topk                   : $DSV4_TOPK"
    echo "  total_experts          : $DSV4_TOTAL_EXPERTS"
    echo "  ep_size                : $DSV4_EP_SIZE"
    echo "  local experts          : $DSV4_LOCAL_EXPERTS"
    echo "  single mega total exp  : $DSV4_SINGLE_MEGA_TOTAL_EXPERTS"
    echo "  hidden                 : $DSV4_HIDDEN"
    echo "  intermediate_gateup    : $DSV4_INTERMEDIATE_GATEUP"
    echo "  intermediate_downproj  : $DSV4_INTERMEDIATE_DOWNPROJ"
    echo "  gate_up_clamp          : ${DSV4_GATE_UP_CLAMP:-off}"
    echo "  fp8_scale_mode         : $SCALE_MODE"
    echo "  mega config source     : $([ "$USE_HEURISTIC" -eq 1 ] && echo heuristic || echo manual)"
    echo "  token_back_mode        : $FP8_TOKEN_BACK_MODE"
    echo "  in_kernel_reduce       : $FP8_IN_KERNEL_REDUCE"
    echo "  swap_ab                : $([ "$USE_HEURISTIC" -eq 1 ] && echo heuristic || echo "$SWAP_AB")"
    echo "  fp8_accum_mode         : $FP8_ACCUM_MODE"
    echo "  cluster_shape_mnk      : $([ "$USE_HEURISTIC" -eq 1 ] && echo heuristic || echo "$FP8_CLUSTER_SHAPE")"
    echo "  route_rows             : $DSV4_ROUTE_ROWS"
    echo "  mega perf warmup/iters : $PERF_WARMUP / $PERF_ITERS"
    echo "  fc12 perf warmup/iters: $FC12_WARMUP / $FC12_ITERS"
    echo "  timeout seconds        : $TIMEOUT_SECONDS"
    echo "  NCCL_NVLS_ENABLE       : $NCCL_NVLS_ENABLE"
    echo "  NVSHMEM_DISABLE_NVLS   : $NVSHMEM_DISABLE_NVLS"
    echo "==========================================================================="
}

run_fc12_case() {
    local args="$1"
    local scale_mode="$2"
    echo "[CMD] timeout $TIMEOUT_SECONDS $PYTHON $FC12_RUNNER $args --fp8_scale_mode $scale_mode --fp8_accum_mode $FP8_ACCUM_MODE"
    # shellcheck disable=SC2086
    timeout "$TIMEOUT_SECONDS" "$PYTHON" "$FC12_RUNNER" $args --fp8_scale_mode "$scale_mode" --fp8_accum_mode "$FP8_ACCUM_MODE"
}

run_mega_single_case() {
    local args="$1"
    local scale_mode="$2"
    echo "[CMD] timeout $TIMEOUT_SECONDS env MEGA_NO_DIST=1 $PYTHON $MEGA_RUNNER $args --fp8_scale_mode $scale_mode --fp8_accum_mode $FP8_ACCUM_MODE $MEGA_COMBINE_ARGS"
    # shellcheck disable=SC2086
    timeout "$TIMEOUT_SECONDS" env MEGA_NO_DIST=1 "$PYTHON" "$MEGA_RUNNER" $args --fp8_scale_mode "$scale_mode" --fp8_accum_mode "$FP8_ACCUM_MODE" $MEGA_COMBINE_ARGS
}

run_mega_multi_case() {
    local args="$1"
    local scale_mode="$2"
    local nproc="${MEGA_NPROC:-$DSV4_EP_SIZE}"
    local visible_gpus=""

    if [ "$nproc" -ne "$DSV4_EP_SIZE" ]; then
        echo "ERROR: P03 is fixed to ${DSV4_EP_SIZE} GPUs; got MEGA_NPROC=$nproc" >&2
        return 2
    fi

    if command -v nvidia-smi >/dev/null 2>&1; then
        visible_gpus="$(nvidia-smi --list-gpus 2>/dev/null | wc -l | tr -d '[:space:]')"
        if [ -n "$visible_gpus" ] && [ "$visible_gpus" -lt "$nproc" ]; then
            echo "ERROR: P03 requires $nproc visible GPUs, got $visible_gpus" >&2
            return 2
        fi
    fi

    echo "[CMD] timeout $TIMEOUT_SECONDS torchrun --standalone --nproc_per_node=$nproc $MEGA_RUNNER $args --fp8_scale_mode $scale_mode --fp8_accum_mode $FP8_ACCUM_MODE $MEGA_COMBINE_ARGS"
    # shellcheck disable=SC2086
    timeout "$TIMEOUT_SECONDS" torchrun --standalone --nproc_per_node="$nproc" "$MEGA_RUNNER" $args --fp8_scale_mode "$scale_mode" --fp8_accum_mode "$FP8_ACCUM_MODE" $MEGA_COMBINE_ARGS
}

if [ "$LIST_ONLY" -eq 1 ]; then
    for entry in "${TESTS[@]}"; do
        name="${entry%%|*}"
        name="${name%"${name##*[![:space:]]}"}"
        rest="${entry#*|}"
        run_type="${rest%%|*}"
        run_type="${run_type#"${run_type%%[![:space:]]*}"}"
        run_type="${run_type%"${run_type##*[![:space:]]}"}"
        full_name="${SCALE_MODE}/${name}"
        if test_matches_selectors "$full_name" "$run_type"; then
            echo "$full_name [$run_type]"
        fi
    done
    exit 0
fi

print_config

PASS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0
declare -a FAIL_NAMES=()
START_TIME=$SECONDS

for entry in "${TESTS[@]}"; do
        name="${entry%%|*}"
        name="${name%"${name##*[![:space:]]}"}"
        rest="${entry#*|}"
        run_type="${rest%%|*}"
        run_type="${run_type#"${run_type%%[![:space:]]*}"}"
        run_type="${run_type%"${run_type##*[![:space:]]}"}"
        args="${rest#*|}"
        args="${args#"${args%%[![:space:]]*}"}"
        full_name="${SCALE_MODE}/${name}"

        if ! test_matches_selectors "$full_name" "$run_type"; then
            SKIP_COUNT=$((SKIP_COUNT + 1))
            continue
        fi

        echo
        echo "==========================================================================="
        echo "[TEST] $full_name"
        echo "[MODE] $SCALE_MODE"
        echo "[TYPE] $run_type"
        echo "==========================================================================="

        test_start=$SECONDS
        case "$run_type" in
            fc12)
                run_fc12_case "$args" "$SCALE_MODE"
                rc=$?
                ;;
            mega_single)
                run_mega_single_case "$args" "$SCALE_MODE"
                rc=$?
                ;;
            mega_multi)
                run_mega_multi_case "$args" "$SCALE_MODE"
                rc=$?
                ;;
            *)
                echo "ERROR: unknown run type '$run_type'" >&2
                rc=2
                ;;
        esac
        elapsed=$((SECONDS - test_start))

        if [ "$rc" -eq 0 ]; then
            echo "[RESULT] PASS (${elapsed}s) $full_name"
            PASS_COUNT=$((PASS_COUNT + 1))
        else
            if [ "$rc" -eq 124 ]; then
                echo "[TIMEOUT] $full_name exceeded ${TIMEOUT_SECONDS}s"
            fi
            echo "[RESULT] FAIL (rc=$rc, ${elapsed}s) $full_name"
            FAIL_COUNT=$((FAIL_COUNT + 1))
            FAIL_NAMES+=("$full_name")
        fi
done

TOTAL_ELAPSED=$((SECONDS - START_TIME))
RAN_COUNT=$((PASS_COUNT + FAIL_COUNT))
TOTAL=${#TESTS[@]}

echo
echo "==========================================================================="
echo "SUMMARY: ${PASS_COUNT}/${RAN_COUNT} passed, ${FAIL_COUNT} failed, ${SKIP_COUNT}/${TOTAL} skipped (mode: ${SCALE_MODE}; wallclock ${TOTAL_ELAPSED}s)"
if [ "$FAIL_COUNT" -gt 0 ]; then
    echo "Failed tests:"
    for name in "${FAIL_NAMES[@]}"; do
        echo "  - $name"
    done
fi
if [ "$RAN_COUNT" -eq 0 ]; then
    echo "WARNING: selectors matched 0 tests (use --list to see available tests)"
fi
echo "==========================================================================="

exit "$FAIL_COUNT"
