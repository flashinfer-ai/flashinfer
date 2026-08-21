"""Benchmark SM120 direct BF16 fused MoE for decode batches M=1..8.

This benchmark models one EP=4 rank: 64 local experts out of 256 global
experts, top-k=8, with two local and six remote routes per token.

Examples
--------
python benchmarks/bench_sm120_direct_fused_moe.py --preset qwen
python benchmarks/bench_sm120_direct_fused_moe.py --preset joyai --csv result.csv
"""

from __future__ import annotations

import argparse
import csv
import gc
import tempfile
from pathlib import Path

import torch

from flashinfer.jit import env as jit_env
from flashinfer.fused_moe import (
    cutlass_fused_moe,
    sm120_direct_fused_moe,
    sm120_direct_fused_moe_workspace,
)


PRESETS = {
    "qwen": {"hidden": 2048, "intermediate": 512},
    "joyai": {"hidden": 2048, "intermediate": 768},
}


def _bench(fn, warmup: int, iterations: int) -> float:
    """Measure one CUDA Graph replay path and return latency in microseconds."""
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        graph.replay()
    end.record()
    end.synchronize()
    latency_us = start.elapsed_time(end) * 1000.0 / iterations
    del graph
    gc.collect()
    return latency_us


def main() -> None:
    """Run the direct-kernel and CUTLASS comparison for all supported M."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=PRESETS, default="qwen")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--csv", type=Path)
    parser.add_argument(
        "--force-jit-baseline",
        action="store_true",
        help="build CUTLASS from this checkout instead of loading an installed AOT cache",
    )
    args = parser.parse_args()
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.iterations <= 0:
        parser.error("--iterations must be positive")

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("bench_sm120_direct_fused_moe requires SM120")
    force_jit_aot_dir = None
    if args.force_jit_baseline:
        force_jit_aot_dir = tempfile.TemporaryDirectory(prefix="flashinfer-empty-aot-")
        jit_env.FLASHINFER_AOT_DIR = Path(force_jit_aot_dir.name)
    torch.manual_seed(args.seed)
    geometry = PRESETS[args.preset]
    hidden = geometry["hidden"]
    intermediate = geometry["intermediate"]
    max_tokens, topk = 8, 8
    local_experts, global_experts = 64, 256

    hidden_states = torch.randn(max_tokens, hidden, dtype=torch.bfloat16, device="cuda")
    gemm1_weights = torch.randn(
        local_experts,
        2 * intermediate,
        hidden,
        dtype=torch.bfloat16,
        device="cuda",
    )
    gemm2_weights = torch.randn(
        local_experts,
        hidden,
        intermediate,
        dtype=torch.bfloat16,
        device="cuda",
    )
    expert_map = torch.full((global_experts,), -1, dtype=torch.int32, device="cuda")
    expert_map[:local_experts] = torch.arange(
        local_experts, dtype=torch.int32, device="cuda"
    )
    topk_ids = []
    for _ in range(max_tokens):
        local = torch.randperm(local_experts, device="cuda")[:2].to(torch.int32)
        remote = (
            torch.randperm(global_experts - local_experts, device="cuda")[:6]
            + local_experts
        ).to(torch.int32)
        topk_ids.append(torch.cat((local, remote)))
    topk_ids = torch.stack(topk_ids).contiguous()
    topk_weights = torch.softmax(
        torch.randn(max_tokens, topk, dtype=torch.float32, device="cuda"), dim=-1
    ).contiguous()
    output = torch.empty(max_tokens, hidden, dtype=torch.bfloat16, device="cuda")
    cutlass_output = torch.empty_like(output)
    workspace = sm120_direct_fused_moe_workspace(
        max_tokens, topk, intermediate, device="cuda"
    )

    rows = []
    for num_tokens in range(1, 9):

        def run_direct():
            """Run the direct SM120 kernel for the current token count."""
            return sm120_direct_fused_moe(
                hidden_states[:num_tokens],
                topk_ids[:num_tokens],
                topk_weights[:num_tokens],
                gemm1_weights,
                gemm2_weights,
                expert_map,
                output=output[:num_tokens],
                workspace=workspace[: num_tokens * topk],
                skip_check=True,
            )

        def run_cutlass():
            """Run the CUTLASS baseline for the current token count."""
            return cutlass_fused_moe(
                input=hidden_states[:num_tokens],
                token_selected_experts=topk_ids[:num_tokens],
                token_final_scales=topk_weights[:num_tokens],
                fc1_expert_weights=gemm1_weights,
                fc2_expert_weights=gemm2_weights,
                output_dtype=torch.bfloat16,
                quant_scales=[],
                tp_size=1,
                tp_rank=0,
                ep_size=4,
                ep_rank=0,
                output=cutlass_output[:num_tokens],
                tune_max_num_tokens=max_tokens,
            )

        direct_us = _bench(run_direct, args.warmup, args.iterations)
        cutlass_us = _bench(run_cutlass, args.warmup, args.iterations)
        row = {
            "preset": args.preset,
            "num_tokens": num_tokens,
            "hidden_size": hidden,
            "intermediate_size": intermediate,
            "topk": topk,
            "local_experts": local_experts,
            "cutlass_us": cutlass_us,
            "direct_us": direct_us,
            "speedup": cutlass_us / direct_us,
        }
        rows.append(row)
        print(
            f"{args.preset} M={num_tokens}: CUTLASS {cutlass_us:.3f} us, "
            f"direct {direct_us:.3f} us, {cutlass_us / direct_us:.3f}x"
        )

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
