#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
REPLAYS="${REPLAYS:-100}"

if [[ "$NPROC_PER_NODE" -ne 8 ]]; then
  echo "EP8 hybrid validation requires NPROC_PER_NODE=8" >&2
  exit 2
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export NVSHMEM_HEAP_KIND=VIDMEM
export NVSHMEM_SYMMETRIC_SIZE="${NVSHMEM_SYMMETRIC_SIZE:-2G}"
export NVSHMEM_DEBUG="${NVSHMEM_DEBUG:-WARN}"
# Transport correctness is gated by the byte-exact replay checks below.
export MEGA_STRICT_TORCH_REF="${MEGA_STRICT_TORCH_REF:-0}"
export MEGA_SPLIT_INCLUDE_K3=1

run_case() {
  local name="$1"
  shift
  printf '\n===== EP8 hybrid: %s =====\n' "$name"
  "$PYTHON_BIN" -m torch.distributed.run \
    --standalone --nproc_per_node=8 \
    -m moe_sm120_nvfp4_split.mega_runner \
    --data_parallel_size 1 --tensor_parallel_size 1 \
    --hidden 1024 --intermediate 2048 \
    --num_topk 4 --num_total_experts 64 \
    --enable_static_expert_shape \
    --comm_backend nvshmem_hybrid "$@"
}

# Independent Torch-reference coverage for representative route distributions.
run_case "balanced reference" \
  --num_tokens_per_rank 128 --route_distribution balanced
run_case "power-law reference" \
  --num_tokens_per_rank 131 --route_distribution power_law \
  --power_law_exponent 1.2
run_case "zero-count peer reference" \
  --num_tokens_per_rank 128 --route_distribution zero_count_peer

# Dynamic GPU-resident routes exercise epoch/slot reuse.  The final replay is
# reconstructed byte-for-byte from every owner's FC2 staging metadata.
run_case "dynamic-route replay byte-exact" \
  --num_tokens_per_rank 128 --route_distribution balanced \
  --perf_run --skip_ref_check --use_cuda_events --enable_debug_checks \
  --dynamic_route_replays "$REPLAYS" --transport_replay_validation

# A fixed route over the same number of replays must be bit deterministic;
# terminal ACK/credit reuse is still exercised on every epoch.
run_case "static-route replay determinism" \
  --num_tokens_per_rank 128 --route_distribution balanced \
  --perf_run --skip_ref_check --use_cuda_events --enable_debug_checks \
  --perf_warmup 0 --perf_iters "$REPLAYS" \
  --transport_replay_validation --verify_replay_bit_determinism

printf '\nAll EP8 hybrid correctness/replay tests passed.\n'
