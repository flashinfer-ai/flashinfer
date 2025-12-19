"""
Benchmark comparison of MoE Top-K implementations:
1. PyTorch torch.topk (baseline)
2. FlashInfer moe_top_k (warp-reduction, N<=512, K<=16)
3. Sonic-MoE TopK_Softmax (bitonic sort, N<=4096, K<=128)

Uses CUPTI backend for accurate kernel timing.
"""

import argparse
import sys
from dataclasses import dataclass

import numpy as np
import torch

from flashinfer.testing.utils import bench_gpu_time

# Import FlashInfer implementations
try:
    from flashinfer import moe_top_k as flashinfer_moe_topk

    HAS_FLASHINFER_MOE = True
except ImportError:
    HAS_FLASHINFER_MOE = False

# Import Sonic-MoE
try:
    sys.path.insert(0, "/home/zihaoye2/data/sonic-moe")
    from sonicmoe.functional.forward import _topk_fwd

    HAS_SONIC_MOE = True
except ImportError:
    HAS_SONIC_MOE = False


@dataclass
class BenchConfig:
    """Benchmark configuration."""

    N: int  # num_experts
    T: int  # num_tokens
    K: int  # top-k
    dtype: torch.dtype = torch.bfloat16
    name: str = ""


def sonic_moe_topk(logits: torch.Tensor, k: int):
    """Sonic-MoE bitonic sort top-k."""
    T, N = logits.shape
    values = torch.empty(T, k, dtype=torch.float32, device=logits.device)
    indices = torch.empty(T, k, dtype=torch.int32, device=logits.device)
    _topk_fwd(logits, k, values, indices, require_softmax_fusion=False)
    return values, indices


def pytorch_topk(logits: torch.Tensor, k: int):
    """PyTorch baseline."""
    return logits.topk(k, dim=-1)


def bench_single_config(
    config: BenchConfig, dry_run_iters: int = 10, repeat_iters: int = 100
):
    """Benchmark a single configuration."""
    logits = torch.randn(config.T, config.N, dtype=config.dtype, device="cuda")

    results = {"config": config}

    # PyTorch baseline
    measurements = bench_gpu_time(
        lambda: pytorch_topk(logits, config.K),
        enable_cupti=True,
        dry_run_iters=dry_run_iters,
        repeat_iters=repeat_iters,
    )
    results["pytorch_us"] = np.median(measurements) * 1e3

    # FlashInfer warp-reduction (N<=512, K<=16)
    if HAS_FLASHINFER_MOE and config.N <= 512 and config.K <= 16:
        measurements = bench_gpu_time(
            lambda: flashinfer_moe_topk(logits, config.K),
            enable_cupti=True,
            dry_run_iters=dry_run_iters,
            repeat_iters=repeat_iters,
        )
        results["flashinfer_warp_us"] = np.median(measurements) * 1e3

    # Sonic-MoE bitonic (N<=4096, K<=128)
    if HAS_SONIC_MOE and config.N <= 4096 and config.K <= 128 and config.N % 8 == 0:
        measurements = bench_gpu_time(
            lambda: sonic_moe_topk(logits, config.K),
            enable_cupti=True,
            dry_run_iters=dry_run_iters,
            repeat_iters=repeat_iters,
        )
        results["sonic_moe_us"] = np.median(measurements) * 1e3

    return results


def print_results_table(results_list, title: str):
    """Print results in a formatted table."""
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)

    # Header
    header = f"{'Config':<25} | {'PyTorch':>12} | {'FI-Warp':>12} | {'Sonic-MoE':>12} | {'Best':>12} | {'Speedup':>10}"
    print(header)
    print("-" * 100)

    for res in results_list:
        cfg = res["config"]
        config_str = f"N={cfg.N},T={cfg.T},K={cfg.K}"
        if cfg.name:
            config_str = f"{cfg.name}"

        pytorch_us = res.get("pytorch_us", float("nan"))
        fi_warp_us = res.get("flashinfer_warp_us", float("nan"))
        sonic_us = res.get("sonic_moe_us", float("nan"))

        # Find best
        times = {"PyTorch": pytorch_us}
        if not np.isnan(fi_warp_us):
            times["FI-Warp"] = fi_warp_us
        if not np.isnan(sonic_us):
            times["Sonic-MoE"] = sonic_us

        best_name = min(times, key=times.get)
        best_time = times[best_name]
        speedup = pytorch_us / best_time if best_time > 0 else 0

        def fmt(val):
            return f"{val:>10.2f}us" if not np.isnan(val) else f"{'N/A':>12}"

        print(
            f"{config_str:<25} | {fmt(pytorch_us)} | {fmt(fi_warp_us)} | {fmt(sonic_us)} | {best_name:>12} | {speedup:>8.2f}x"
        )


def run_moe_routing_benchmark():
    """Benchmark MoE routing scenarios (small N, small K)."""
    configs = [
        # Standard MoE models
        BenchConfig(N=8, T=4096, K=2, name="Mixtral-8x7B"),
        BenchConfig(N=64, T=4096, K=4, name="Qwen-MoE"),
        BenchConfig(N=160, T=4096, K=6, name="DeepSeek-V2"),
        BenchConfig(N=256, T=4096, K=8, name="DeepSeek-V3"),
        # Varying batch sizes
        BenchConfig(N=256, T=64, K=8, name="N=256,T=64,K=8"),
        BenchConfig(N=256, T=256, K=8, name="N=256,T=256,K=8"),
        BenchConfig(N=256, T=1024, K=8, name="N=256,T=1024,K=8"),
        BenchConfig(N=256, T=4096, K=8, name="N=256,T=4096,K=8"),
        BenchConfig(N=256, T=16384, K=8, name="N=256,T=16384,K=8"),
        BenchConfig(N=256, T=65536, K=8, name="N=256,T=65536,K=8"),
        # Varying K
        BenchConfig(N=256, T=4096, K=1, name="N=256,K=1"),
        BenchConfig(N=256, T=4096, K=2, name="N=256,K=2"),
        BenchConfig(N=256, T=4096, K=4, name="N=256,K=4"),
        BenchConfig(N=256, T=4096, K=8, name="N=256,K=8"),
        BenchConfig(N=256, T=4096, K=16, name="N=256,K=16"),
    ]

    results = [bench_single_config(cfg) for cfg in configs]
    print_results_table(
        results,
        "MoE Routing Benchmark (FlashInfer warp-reduction territory: N<=512, K<=16)",
    )


def run_extended_moe_benchmark():
    """Benchmark extended MoE scenarios (larger N and K, Sonic-MoE territory)."""
    configs = [
        # Large N (beyond FlashInfer warp support)
        BenchConfig(N=512, T=4096, K=8, name="N=512,K=8"),
        BenchConfig(N=1024, T=4096, K=8, name="N=1024,K=8"),
        BenchConfig(N=2048, T=4096, K=8, name="N=2048,K=8"),
        BenchConfig(N=4096, T=4096, K=8, name="N=4096,K=8"),
        # Large K (beyond FlashInfer warp support)
        BenchConfig(N=256, T=4096, K=32, name="N=256,K=32"),
        BenchConfig(N=256, T=4096, K=64, name="N=256,K=64"),
        BenchConfig(N=256, T=4096, K=128, name="N=256,K=128"),
        # Large N and K
        BenchConfig(N=1024, T=4096, K=32, name="N=1024,K=32"),
        BenchConfig(N=2048, T=4096, K=64, name="N=2048,K=64"),
    ]

    results = [bench_single_config(cfg) for cfg in configs]
    print_results_table(
        results, "Extended MoE Benchmark (Sonic-MoE territory: N<=4096, K<=128)"
    )


def run_all_benchmarks():
    """Run all benchmark suites."""
    print("\n" + "#" * 100)
    print("# MoE Top-K Implementation Comparison Benchmark")
    print("# Using CUPTI backend for accurate kernel timing")
    print("#" * 100)
    print("\nImplementations available:")
    print("  - PyTorch torch.topk: Always available")
    print(f"  - FlashInfer moe_top_k (warp): {HAS_FLASHINFER_MOE} (N<=512, K<=16)")
    print(f"  - Sonic-MoE (bitonic): {HAS_SONIC_MOE} (N<=4096, K<=128)")
    print(f"\nDevice: {torch.cuda.get_device_name()}")

    run_moe_routing_benchmark()
    run_extended_moe_benchmark()


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description="MoE Top-K Benchmark Comparison")
    parser.add_argument(
        "--suite",
        choices=["all", "moe", "extended"],
        default="all",
        help="Which benchmark suite to run",
    )
    parser.add_argument(
        "--dry-run-iters",
        type=int,
        default=10,
        help="Number of warmup iterations",
    )
    parser.add_argument(
        "--repeat-iters",
        type=int,
        default=100,
        help="Number of measured iterations",
    )
    args = parser.parse_args()

    if args.suite == "all":
        run_all_benchmarks()
    elif args.suite == "moe":
        run_moe_routing_benchmark()
    elif args.suite == "extended":
        run_extended_moe_benchmark()


if __name__ == "__main__":
    main()
