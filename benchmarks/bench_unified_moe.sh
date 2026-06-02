#!/usr/bin/env bash
# Unified NVFP4 MoE benchmark sweep (DeepSeek-V3 geometry).
#
# Run from the repo root:  bash benchmarks/bench_unified_moe.sh
#
# Shapes (see docs/design_docs/flashinfer_moe_api.md §11 for the reference
# winner/latency tables this sweep reproduces):
#   EP=1 sweep:  tokens = 1, 16, 1024, 4096, 16384
#   EP=16 sweep: tokens = 1, 16, 4096
#
# DeepSeek-V3 geometry: hidden=7168, intermediate=2048, experts=256, top_k=8,
# n_group=8, topk_group=4, routed_scaling_factor=2.5.
#
# Expectation: the MoELayer winner and per-backend latencies track the
# cross-backend tables in the design doc (TRTLLM-gen wins ≤512 tokens, CuteDSL
# wins ≥1024). Requires an SM100 (Blackwell) GPU.
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
