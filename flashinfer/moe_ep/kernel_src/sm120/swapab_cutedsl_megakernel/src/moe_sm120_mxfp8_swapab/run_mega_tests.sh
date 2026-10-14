#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
RUN_PERF="${RUN_PERF:-0}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NVSHMEM_HEAP_KIND="${NVSHMEM_HEAP_KIND:-SYSMEM}"
export NVSHMEM_SYMMETRIC_SIZE="${NVSHMEM_SYMMETRIC_SIZE:-1G}"
export MEGA_STRICT_TORCH_REF="${MEGA_STRICT_TORCH_REF:-1}"

run_case() {
  local name="$1"
  shift
  printf '\n===== %s-rank: %s =====\n' "$NPROC_PER_NODE" "$name"
  "$PYTHON_BIN" -m torch.distributed.run \
    --standalone --nproc_per_node="$NPROC_PER_NODE" \
    -m moe_sm120_mxfp8_swapab.mega_runner "$@"
}

common=(
  --hidden 1024
  --intermediate 2048
  --cluster_shape_mnk 1,1,1
  --enable_static_expert_shape
  --token_back_mode epi_warps
)

# Multi-rank pool publication is 64-token aligned, so N64 is its minimum tile.
run_case "balanced topk4 N64" \
  --num_tokens_per_rank 128 --num_topk 4 --num_total_experts 64 \
  --route_distribution balanced --mma_tiler_mnk 64,64,128 \
  "${common[@]}"

# Production top-k and an intentionally skewed route stress dispatch tails.
run_case "power-law topk6 N64" \
  --num_tokens_per_rank 131 --num_topk 6 --num_total_experts 64 \
  --route_distribution power_law --power_law_exponent 1.2 \
  --mma_tiler_mnk 64,64,128 "${common[@]}"

# The production kernel uses DeepGEMM form-A top-k weighting before quantize.
run_case "balanced topk8 N128" \
  --num_tokens_per_rank 128 --num_topk 8 --num_total_experts 64 \
  --route_distribution balanced --ref_compute_graph deepgemm \
  --mma_tiler_mnk 64,128,128 "${common[@]}"

if [[ "$RUN_PERF" == "1" ]]; then
  export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
  export NVSHMEM_SYMMETRIC_SIZE="${PERF_NVSHMEM_SYMMETRIC_SIZE:-8G}"
  run_case "DSV4 performance" \
    --num_tokens_per_rank "${TOKENS_PER_RANK:-20480}" \
    --num_topk 6 --num_total_experts 384 \
    --hidden 7168 --intermediate 6144 --route_distribution balanced \
    --mma_tiler_mnk 64,128,128 --cluster_shape_mnk 1,1,1 \
    --enable_static_expert_shape --token_back_mode epi_warps \
    --perf_run --skip_ref_check --use_cuda_events \
    --perf_warmup "${PERF_WARMUP:-10}" --perf_iters "${PERF_ITERS:-30}"
fi

printf '\nAll multi-rank SM120 tests passed.\n'
