#!/usr/bin/env bash
# Unified NVFP4 MoE benchmark sweep — checkmate shapes for today's MVP.
#
# Runs the 8 shapes enumerated in unified_moe_mvp_plan.md §4:
#   EP=1 sweep:  tokens = 1, 16, 1024, 4096, 16384
#   EP=16 sweep: tokens = 1, 16, 4096
#
# DeepSeek-V3 geometry: hidden=7168, intermediate=2048, experts=256, top_k=8,
# n_group=8, topk_group=4, routed_scaling_factor=2.5.
#
# Pass criterion: MoELayer winner must match the expected-winner column in the
# plan's shape table for every shape; per-backend latency within ±5% of the
# reference numbers.
set -euo pipefail

CSV=/tmp/unified_moe_today.csv
: > "$CSV"  # truncate

BASE=(
    --routine unified_nvfp4_moe
    --hidden_size 7168 --intermediate_size 2048
    --num_experts 256 --top_k 8
    --n_group 8 --topk_group 4
    --routing_method deepseek_v3 --routed_scaling_factor 2.5
    --output_path "$CSV"
    -v
)

# EP=1 sweep (local_num_experts defaults to num_experts=256)
for M in 1 16 1024 4096 16384; do
    echo "=== S_EP1 num_tokens=$M ==="
    python benchmarks/flashinfer_benchmark.py "${BASE[@]}" --num_tokens "$M"
done

# EP=16 sweep (local_num_experts=16)
for M in 1 16 4096; do
    echo "=== S_EP16 num_tokens=$M ==="
    python benchmarks/flashinfer_benchmark.py "${BASE[@]}" \
        --num_tokens "$M" --local_num_experts 16
done

echo "Results: $CSV"
