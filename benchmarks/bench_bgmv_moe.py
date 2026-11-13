"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Multi-LoRA MoE BGMV Kernel Benchmark.

Compares the BGMV MoE CUDA kernel against FlashInfer's grouped_mm_bf16 baseline
across multiple model configurations and token counts.

Usage:
    FLASHINFER_DISABLE_VERSION_CHECK=1 python benchmarks/bench_bgmv_moe.py
"""

import os

os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")

import time
from dataclasses import dataclass
from typing import Callable

import torch


@dataclass
class BenchmarkConfig:
    """Configuration for a single benchmark run."""

    name: str
    num_tokens: int
    hidden_size: int
    rank: int
    num_experts: int
    top_k: int
    num_loras: int
    num_slices: int
    dtype: torch.dtype


# Model configurations to benchmark
CONFIGS = [
    # Large MoE (hidden=3072, rank=32, 128 experts)
    BenchmarkConfig("Decode-1tok-LargeMoE", 1, 3072, 32, 128, 2, 8, 1, torch.bfloat16),
    BenchmarkConfig("Decode-4tok-LargeMoE", 4, 3072, 32, 128, 2, 8, 1, torch.bfloat16),
    BenchmarkConfig("Decode-8tok-LargeMoE", 8, 3072, 32, 128, 2, 8, 1, torch.bfloat16),
    BenchmarkConfig(
        "Decode-32tok-LargeMoE", 32, 3072, 32, 128, 2, 8, 1, torch.bfloat16
    ),
    BenchmarkConfig(
        "Prefill-256tok-LargeMoE", 256, 3072, 32, 128, 2, 8, 1, torch.bfloat16
    ),
    BenchmarkConfig(
        "Prefill-512tok-LargeMoE", 512, 3072, 32, 128, 2, 8, 1, torch.bfloat16
    ),
    BenchmarkConfig(
        "Prefill-1024tok-LargeMoE", 1024, 3072, 32, 128, 2, 8, 1, torch.bfloat16
    ),
    # Nemotron-Nano-3-30B-A3B (hidden=2688, rank=32, 128 experts)
    BenchmarkConfig("Decode-1tok-Nemotron", 1, 2688, 32, 128, 2, 8, 1, torch.bfloat16),
    BenchmarkConfig("Decode-4tok-Nemotron", 4, 2688, 32, 128, 2, 8, 1, torch.bfloat16),
    BenchmarkConfig("Decode-8tok-Nemotron", 8, 2688, 32, 128, 2, 8, 1, torch.bfloat16),
    BenchmarkConfig(
        "Decode-32tok-Nemotron", 32, 2688, 32, 128, 2, 8, 1, torch.bfloat16
    ),
    BenchmarkConfig(
        "Prefill-256tok-Nemotron", 256, 2688, 32, 128, 2, 8, 1, torch.bfloat16
    ),
    BenchmarkConfig(
        "Prefill-512tok-Nemotron", 512, 2688, 32, 128, 2, 8, 1, torch.bfloat16
    ),
    BenchmarkConfig(
        "Prefill-1024tok-Nemotron", 1024, 2688, 32, 128, 2, 8, 1, torch.bfloat16
    ),
]


def generate_test_data(config: BenchmarkConfig, device: str = "cuda"):
    """Generate random test data for a benchmark configuration."""
    num_tokens = config.num_tokens
    hidden_size = config.hidden_size
    rank = config.rank
    num_experts = config.num_experts
    top_k = config.top_k
    num_loras = config.num_loras
    num_slices = config.num_slices
    dtype = config.dtype
    num_pairs = num_tokens * top_k
    feat_out = hidden_size

    x = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device) * 0.1
    lora_a_weights = [
        torch.randn(
            num_loras, num_experts, rank, hidden_size, dtype=dtype, device=device
        )
        * 0.01
        for _ in range(num_slices)
    ]
    lora_b_weights = [
        torch.randn(num_loras, num_experts, feat_out, rank, dtype=dtype, device=device)
        * 0.01
        for _ in range(num_slices)
    ]
    sorted_token_ids = torch.arange(
        num_tokens, device=device, dtype=torch.int64
    ).repeat_interleave(top_k)
    expert_ids = torch.randint(
        0, num_experts, (num_pairs,), device=device, dtype=torch.int64
    )
    topk_weights = (
        torch.softmax(torch.randn(num_tokens, top_k, device=device), dim=-1)
        .view(-1)
        .to(torch.float32)
    )
    lora_indices = torch.randint(
        0, num_loras, (num_tokens,), device=device, dtype=torch.int64
    )

    return {
        "x": x,
        "lora_a_weights": lora_a_weights,
        "lora_b_weights": lora_b_weights,
        "sorted_token_ids": sorted_token_ids,
        "expert_ids": expert_ids,
        "topk_weights": topk_weights,
        "lora_indices": lora_indices,
        "num_pairs": num_pairs,
        "feat_out": feat_out,
    }


def benchmark_fn(fn: Callable, warmup: int = 10, repeat: int = 100) -> float:
    """Benchmark a function, return median time in microseconds."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(repeat):
        torch.cuda.synchronize()
        start = time.perf_counter_ns()
        fn()
        torch.cuda.synchronize()
        end = time.perf_counter_ns()
        times.append((end - start) / 1000.0)

    times.sort()
    return times[len(times) // 2]


def run_benchmark(config: BenchmarkConfig):
    """Run benchmark for a single configuration."""
    from flashinfer.fused_moe.bgmv_moe import (
        bgmv_moe_shrink,
        bgmv_moe_expand,
        fill_w_ptr,
    )

    data = generate_test_data(config)
    results = {"config": config.name}

    num_tokens = config.num_tokens
    num_pairs = data["num_pairs"]
    rank = config.rank
    num_experts = config.num_experts
    num_slices = config.num_slices
    num_loras = config.num_loras
    feat_out = data["feat_out"]
    hidden_size = config.hidden_size
    dtype = config.dtype
    device = "cuda"

    # === BGMV MoE kernel ===
    w_ptr_a = torch.zeros(num_slices, num_experts, dtype=torch.int64, device=device)
    lora_stride_a = 0
    for s in range(num_slices):
        lora_stride_a = fill_w_ptr(w_ptr_a, data["lora_a_weights"][s], num_experts, s)

    w_ptr_b = torch.zeros(num_slices, num_experts, dtype=torch.int64, device=device)
    lora_stride_b = 0
    for s in range(num_slices):
        lora_stride_b = fill_w_ptr(w_ptr_b, data["lora_b_weights"][s], num_experts, s)

    shrink_out = torch.zeros(num_slices, num_pairs, rank, dtype=dtype, device=device)
    slice_start_loc = torch.zeros(num_slices, dtype=torch.int64, device=device)
    for s in range(num_slices):
        slice_start_loc[s] = s * feat_out
    output_slices = [feat_out] * num_slices
    y_accum = torch.zeros(
        num_tokens, feat_out * num_slices, dtype=torch.float32, device=device
    )

    def cuda_fn():
        shrink_out.zero_()
        y_accum.zero_()
        bgmv_moe_shrink(
            shrink_out,
            data["x"],
            w_ptr_a,
            data["sorted_token_ids"],
            data["expert_ids"],
            data["lora_indices"],
            lora_stride_a,
        )
        bgmv_moe_expand(
            y_accum,
            shrink_out,
            w_ptr_b,
            data["sorted_token_ids"],
            data["expert_ids"],
            data["topk_weights"],
            data["lora_indices"],
            slice_start_loc,
            output_slices,
            lora_stride_b,
        )

    cuda_time = benchmark_fn(cuda_fn)
    results["bgmv_moe_us"] = cuda_time

    # === grouped_mm_bf16 baseline ===
    try:
        from flashinfer.grouped_mm import grouped_mm_bf16

        num_groups = num_loras * num_experts
        lora_ids_expanded = data["lora_indices"][data["sorted_token_ids"]]
        group_ids = lora_ids_expanded * num_experts + data["expert_ids"]
        group_ids[lora_ids_expanded < 0] = num_groups

        sorted_indices = torch.argsort(group_ids)
        sorted_group_ids = group_ids[sorted_indices]
        valid_mask = sorted_group_ids < num_groups
        num_valid = valid_mask.sum().item()

        sorted_token_indices = data["sorted_token_ids"][sorted_indices[:num_valid]]
        g_input = data["x"][sorted_token_indices]

        counts = torch.zeros(num_groups + 1, dtype=torch.int32, device=device)
        valid_groups = sorted_group_ids[:num_valid]
        for g in range(num_groups):
            counts[g + 1] = counts[g] + (valid_groups == g).sum().to(torch.int32)
        g_m_indptr = counts

        g_lora_a = data["lora_a_weights"][0].view(
            num_loras * num_experts, rank, hidden_size
        )
        g_lora_b = data["lora_b_weights"][0].view(
            num_loras * num_experts, feat_out, rank
        )
        g_shrink_out = torch.zeros(num_valid, rank, dtype=dtype, device=device)
        g_expand_out = torch.zeros(num_valid, feat_out, dtype=dtype, device=device)

        # Warmup
        grouped_mm_bf16(g_input, g_lora_a, g_m_indptr, out=g_shrink_out)
        grouped_mm_bf16(g_shrink_out, g_lora_b, g_m_indptr, out=g_expand_out)
        torch.cuda.synchronize()

        # Kernel only
        def gg_kernel_fn():
            g_shrink_out.zero_()
            g_expand_out.zero_()
            grouped_mm_bf16(g_input, g_lora_a, g_m_indptr, out=g_shrink_out)
            grouped_mm_bf16(g_shrink_out, g_lora_b, g_m_indptr, out=g_expand_out)

        gg_kernel_time = benchmark_fn(gg_kernel_fn)
        results["gg_kernel_us"] = gg_kernel_time

        # Sort + kernel
        def gg_full_fn():
            _sorted_indices = torch.argsort(group_ids)
            _sorted_token_indices = data["sorted_token_ids"][
                _sorted_indices[:num_valid]
            ]
            _g_input = data["x"][_sorted_token_indices]
            g_shrink_out.zero_()
            g_expand_out.zero_()
            grouped_mm_bf16(_g_input, g_lora_a, g_m_indptr, out=g_shrink_out)
            grouped_mm_bf16(g_shrink_out, g_lora_b, g_m_indptr, out=g_expand_out)

        gg_full_time = benchmark_fn(gg_full_fn)
        results["gg_full_us"] = gg_full_time
    except (ImportError, RuntimeError) as e:
        print(f"  [SKIP] grouped_mm_bf16 baseline: {e}")
        results["gg_kernel_us"] = float("nan")
        results["gg_full_us"] = float("nan")

    return results


def main():
    """Run all benchmarks and print results table."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmarks.")
        return

    device_name = torch.cuda.get_device_name(0)

    # Trigger JIT compilation before printing the table
    from flashinfer.fused_moe.bgmv_moe import _get_bgmv_moe_module

    _get_bgmv_moe_module()

    print(f"\n{'=' * 100}")
    print("Multi-LoRA MoE BGMV Kernel Benchmark")
    print(f"Device: {device_name}")
    print(f"{'=' * 100}\n")

    # Header
    print(
        f"{'Config':<28} {'GG-kern (μs)':>13} {'GG-sort+kern (μs)':>18} "
        f"{'BGMV MoE (μs)':>14} {'vs GG-kern':>11} {'vs GG-sort+kern':>16}"
    )
    print(f"{'-' * 28} {'-' * 13} {'-' * 18} {'-' * 14} {'-' * 11} {'-' * 16}")

    for config in CONFIGS:
        results = run_benchmark(config)

        def fmt(v):
            return f"{v:.1f}" if v == v else "N/A"

        def fmt_speedup(baseline, kernel):
            if baseline != baseline or kernel != kernel or kernel == 0:
                return "N/A"
            return f"{baseline / kernel:.2f}x"

        bgmv = results["bgmv_moe_us"]
        gg_k = results.get("gg_kernel_us", float("nan"))
        gg_f = results.get("gg_full_us", float("nan"))

        print(
            f"{results['config']:<28} "
            f"{fmt(gg_k):>13} "
            f"{fmt(gg_f):>18} "
            f"{fmt(bgmv):>14} "
            f"{fmt_speedup(gg_k, bgmv):>11} "
            f"{fmt_speedup(gg_f, bgmv):>16}"
        )

    print(f"\n{'=' * 100}")
    print("\nNotes:")
    print(
        "  - 'GG-kern' = FlashInfer grouped_mm_bf16 kernel only (pre-sorted, no sort overhead)"
    )
    print(
        "  - 'GG-sort+kern' = FlashInfer grouped_mm_bf16 with token sorting (sort + kernel)"
    )
    print("  - 'BGMV MoE' = BGMV MoE CUDA kernel (this PR)")
    print("  - 'LargeMoE' = hidden=3072, rank=32, 128 experts (large MoE model config)")
    print(
        "  - 'Nemotron' = hidden=2688, rank=32, 128 experts (Nemotron-Nano-3-30B-A3B)"
    )
    print("  - All times are median of 100 runs after 10 warmup iterations")


if __name__ == "__main__":
    main()
