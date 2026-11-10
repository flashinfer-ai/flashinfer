#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
RUN_PERF="${RUN_PERF:-0}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export NVSHMEM_HEAP_KIND="${NVSHMEM_HEAP_KIND:-VIDMEM}"
export NVSHMEM_SYMMETRIC_SIZE="${NVSHMEM_SYMMETRIC_SIZE:-2G}"
export MEGA_STRICT_TORCH_REF="${MEGA_STRICT_TORCH_REF:-1}"

run_case() {
  local name="$1"
  shift
  printf '\n===== %s-rank: %s =====\n' "$NPROC_PER_NODE" "$name"
  "$PYTHON_BIN" -m torch.distributed.run \
    --standalone --nproc_per_node="$NPROC_PER_NODE" \
    -m moe_sm120_nvfp4_split.mega_runner "$@"
}

common=(
  --hidden 4096 --intermediate 4096
  --num_topk 6 --num_total_experts 256
  --gate_up_clamp 10.0
  --data_parallel_size 1 --tensor_parallel_size 1
  --enable_static_expert_shape
  --comm_backend p2p_direct
  --split_launch green_graph
)

run_case "DSV4-flash balanced reference" \
  --num_tokens_per_rank 2048 --route_distribution balanced "${common[@]}"
run_case "DSV4-flash power-law reference" \
  --num_tokens_per_rank 2048 --route_distribution power_law \
  --power_law_exponent 1.2 "${common[@]}"

if [[ "$RUN_PERF" == "1" ]]; then
  run_case "DSV4-flash performance" \
    --num_tokens_per_rank "${TOKENS_PER_RANK:-2048}" \
    --route_distribution balanced "${common[@]}" \
    --perf_run --skip_ref_check --use_cuda_events \
    --perf_warmup "${PERF_WARMUP:-50}" --perf_iters "${PERF_ITERS:-200}"
fi

printf '\nAll EP4 SM120 tests passed.\n'
