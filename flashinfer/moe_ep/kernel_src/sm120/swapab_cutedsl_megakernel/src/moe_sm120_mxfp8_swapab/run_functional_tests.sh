#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_PERF="${RUN_PERF:-0}"

run_case() {
  local name="$1"
  shift
  printf '\n===== single-rank: %s =====\n' "$name"
  "$PYTHON_BIN" -m moe_sm120_mxfp8_swapab.runner_fc12 "$@"
}

common=(
  --experts 8
  --hidden 1024
  --intermediate 2048
  --cluster_shape_mnk 1,1,1
  --enable_static_expert_shape
)

# Cover every supported SM120 N tile using deterministic E4M3 inputs.
run_case "E4M3 balanced N32" \
  --kind mxfp8_e4m3 --tokens_after_topk 256 --balance_route \
  --mma_tiler_mnk 64,32,128 "${common[@]}"

run_case "E4M3 balanced N64" \
  --kind mxfp8_e4m3 --tokens_after_topk 512 --balance_route \
  --mma_tiler_mnk 64,64,128 "${common[@]}"

# 517 routed rows force both scheduler and FC2 tail handling.
run_case "E4M3 skewed tail N128" \
  --kind mxfp8_e4m3 --tokens_after_topk 517 \
  --mma_tiler_mnk 64,128,128 "${common[@]}"

# E5M2 shares the QMMA path but exercises a distinct data type and reference.
run_case "E5M2 balanced N64" \
  --kind mxfp8_e5m2 --tokens_after_topk 256 --balance_route \
  --mma_tiler_mnk 64,64,128 "${common[@]}"

if [[ "$RUN_PERF" == "1" ]]; then
  run_case "DSV4-equivalent performance" \
    --kind mxfp8_e4m3 --tokens_after_topk "${ROUTED_ROWS:-122880}" \
    --experts "${LOCAL_EXPERTS:-48}" --balance_route \
    --hidden 7168 --intermediate 6144 \
    --mma_tiler_mnk 64,128,128 --cluster_shape_mnk 1,1,1 \
    --enable_static_expert_shape --perf_run --skip_ref_check
fi

printf '\nAll single-rank SM120 tests passed.\n'
