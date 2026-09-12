#!/usr/bin/env python3
"""
Benchmark script comparing torch.softmax vs flashinfer.softmax performance.
Creates a heatmap showing speedup across different batch sizes and hidden dimensions.
"""

import numpy as np
import torch
from typing import List, Tuple
import flashinfer
from flashinfer.testing.utils import bench_gpu_time


@torch.inference_mode()
def benchmark_torch_softmax(logits: torch.Tensor) -> float:
    """Benchmark torch's native softmax."""
    measurements = bench_gpu_time(
        lambda: torch.softmax(logits, dim=-1),
        dry_run_time_ms=100,
        repeat_time_ms=1000,
    )
    return np.median(measurements)


@torch.inference_mode()
def benchmark_flashinfer_softmax(
    logits: torch.Tensor,
    *,
    temperature: float | None = None,
    enable_pdl: bool = False,
) -> float:
    """Benchmark flashinfer's softmax."""
    measurements = bench_gpu_time(
        lambda: flashinfer.sampling.softmax(
            logits, temperature=temperature, enable_pdl=enable_pdl
        ),
        dry_run_time_ms=100,
        repeat_time_ms=1000,
    )
    return np.median(measurements)


@torch.inference_mode()
def benchmark_vibecuda_softmax(
    logits: torch.Tensor,
    *,
    temperature: float | None = None,
    enable_pdl: bool = False,
) -> float:
    """Benchmark flashinfer's vibecuda-backend softmax (SM100-class only)."""
    measurements = bench_gpu_time(
        lambda: flashinfer.sampling.softmax(
            logits,
            temperature=temperature,
            enable_pdl=enable_pdl,
            backend="vibecuda",
        ),
        dry_run_time_ms=100,
        repeat_time_ms=1000,
    )
    return np.median(measurements)


def run_benchmark(
    batch_sizes: List[int], hidden_sizes: List[int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run benchmarks for all combinations of batch_size and hidden_size.

    Returns:
        torch_times: 2D array of torch softmax times (ms)
        flashinfer_times: 2D array of flashinfer (upstream backend) times (ms)
        speedups: 2D array of speedup ratios (torch_time / flashinfer_time vs vibecuda)
    """
    n_batch = len(batch_sizes)
    n_hidden = len(hidden_sizes)

    has_vibecuda = (
        flashinfer.utils.get_compute_capability(torch.device("cuda"))[0] >= 10
    )

    torch_times = np.zeros((n_batch, n_hidden))
    flashinfer_times = np.zeros((n_batch, n_hidden))
    vibecuda_times = np.zeros((n_batch, n_hidden))
    speedups = np.zeros((n_batch, n_hidden))

    print("Running benchmarks...")
    print("=" * 120)
    print(
        f"{'Batch Size':<12} {'Hidden Size':<12} {'Torch (ms)':<15} "
        f"{'FlashInfer (ms)':<18} {'VibeCUDA (ms)':<16} "
        f"{'FI/VibeCUDA':<12} {'Bandwidth (GB/s)':<18}"
    )
    print("=" * 120)

    for i, batch_size in enumerate(batch_sizes):
        for j, hidden_size in enumerate(hidden_sizes):
            # Generate random logits
            torch.manual_seed(42)
            logits = torch.randn(
                batch_size, hidden_size, device="cuda", dtype=torch.float32
            )

            # Benchmark torch softmax
            torch_time_ms = benchmark_torch_softmax(logits)
            torch_times[i, j] = torch_time_ms

            # Benchmark flashinfer softmax (upstream backend)
            flashinfer_time_ms = benchmark_flashinfer_softmax(logits)
            flashinfer_times[i, j] = flashinfer_time_ms

            # Benchmark vibecuda-backend softmax where supported and record
            # speedup vs the upstream flashinfer backend on this workload
            if has_vibecuda:
                vibecuda_time_ms = benchmark_vibecuda_softmax(logits)
                vibecuda_times[i, j] = vibecuda_time_ms
                speedup = flashinfer_time_ms / vibecuda_time_ms
            else:
                vibecuda_time_ms = float("nan")
                speedup = float("nan")
            speedups[i, j] = speedup

            # Calculate effective bandwidth (read + write) of the best backend
            io_bytes = logits.numel() * logits.element_size() * 2
            best_ms = vibecuda_time_ms if has_vibecuda else flashinfer_time_ms
            bandwidth_gb_s = io_bytes * 1e-6 / best_ms

            print(
                f"{batch_size:<12} {hidden_size:<12} {torch_time_ms:<15.4f} "
                f"{flashinfer_time_ms:<18.4f} {vibecuda_time_ms:<16.4f} "
                f"{speedup:<12.2f}x {bandwidth_gb_s:<18.2f}"
            )

    print("=" * 120)
    return torch_times, flashinfer_times, speedups


def run_pdl_benchmark() -> float:
    """Run PR 4282's scalar-temperature PDL performance case."""
    batch_size = 64
    hidden_size = 32000
    temperature = 1.0
    torch.manual_seed(42)
    logits = torch.randn(batch_size, hidden_size, device="cuda", dtype=torch.float32)

    reference = torch.softmax(logits / temperature, dim=-1)
    baseline = flashinfer.sampling.softmax(
        logits, temperature=temperature, enable_pdl=True
    )
    candidate = flashinfer.sampling.softmax(
        logits, temperature=temperature, enable_pdl=True, backend="vibecuda"
    )
    torch.testing.assert_close(baseline, reference, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(candidate, reference, rtol=1e-3, atol=1e-3)

    flashinfer_time_ms = benchmark_flashinfer_softmax(
        logits, temperature=temperature, enable_pdl=True
    )
    vibecuda_time_ms = benchmark_vibecuda_softmax(
        logits, temperature=temperature, enable_pdl=True
    )
    speedup = flashinfer_time_ms / vibecuda_time_ms

    print("\nPR 4282 scalar-temperature PDL case:")
    print("=" * 100)
    print(
        f"batch={batch_size}, vocab={hidden_size}, temperature={temperature}, "
        f"PDL=on: FlashInfer={flashinfer_time_ms:.6f} ms, "
        f"VibeCUDA={vibecuda_time_ms:.6f} ms, speedup={speedup:.4f}x"
    )
    print("=" * 100)
    return speedup


def plot_heatmap(
    speedups: np.ndarray,
    batch_sizes: List[int],
    hidden_sizes: List[int],
    save_path: str = "softmax_speedup_heatmap.png",
):
    """Create and save a heatmap of speedup values."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("matplotlib/seaborn not installed; skipping heatmap generation")
        return
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create heatmap
    sns.heatmap(
        speedups,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        center=1.0,
        cbar_kws={"label": "Speedup (x)"},
        xticklabels=[f"{h // 1000}K" for h in hidden_sizes],
        yticklabels=batch_sizes,
        ax=ax,
        vmin=0.5,  # Adjust color scale
        vmax=max(3.0, speedups.max()),  # Dynamic upper bound
    )

    ax.set_xlabel("Hidden Size", fontsize=12, fontweight="bold")
    ax.set_ylabel("Batch Size", fontsize=12, fontweight="bold")
    ax.set_title(
        "FlashInfer VibeCUDA Softmax Speedup vs Upstream Backend (Higher is Better)",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"\nHeatmap saved to: {save_path}")

    # Also create a performance comparison plot
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 2: Speedup trends across batch sizes
    for j, hidden_size in enumerate(hidden_sizes):
        ax2.plot(
            batch_sizes,
            speedups[:, j],
            marker="o",
            label=f"Hidden={hidden_size // 1000}K",
            linewidth=2,
        )

    ax2.set_xlabel("Batch Size", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Speedup (x)", fontsize=12, fontweight="bold")
    ax2.set_title("Speedup vs Batch Size", fontsize=13, fontweight="bold")
    ax2.set_xscale("log", base=2)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)
    ax2.axhline(y=1.0, color="red", linestyle="--", alpha=0.5, label="No speedup")

    # Plot 1: Speedup trends across hidden sizes
    for i, batch_size in enumerate(batch_sizes[::2]):  # Sample every other batch size
        idx = i * 2
        ax1.plot(
            [h // 1000 for h in hidden_sizes],
            speedups[idx, :],
            marker="s",
            label=f"Batch={batch_size}",
            linewidth=2,
        )

    ax1.set_xlabel("Hidden Size (K)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Speedup (x)", fontsize=12, fontweight="bold")
    ax1.set_title("Speedup vs Hidden Size", fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)
    ax1.axhline(y=1.0, color="red", linestyle="--", alpha=0.5)

    plt.tight_layout()
    comparison_path = save_path.replace(".png", "_trends.png")
    plt.savefig(comparison_path, dpi=300, bbox_inches="tight")
    print(f"Trend plots saved to: {comparison_path}")


def main():
    """Main benchmark execution."""
    # Configuration
    batch_sizes = [1, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    hidden_sizes = [32000, 64000, 128000, 256000]

    print("=" * 100)
    print("FlashInfer Softmax Benchmark: torch vs upstream backend vs vibecuda backend")
    print("=" * 100)
    print(f"Batch sizes: {batch_sizes}")
    print(f"Hidden sizes: {hidden_sizes}")
    print(f"Device: {torch.cuda.get_device_name()}")
    print("=" * 100)
    print()

    # Run benchmarks
    _, _, speedups = run_benchmark(batch_sizes, hidden_sizes)
    pdl_speedup = run_pdl_benchmark()
    all_speedups = np.append(speedups.ravel(), pdl_speedup)

    # Print summary statistics
    print("\nSummary Statistics:")
    print("=" * 100)
    print("Workloads: 41 (40 temperature=None/PDL-off + 1 scalar-temperature PDL)")
    print(f"Average speedup: {np.mean(all_speedups):.2f}x")
    print(f"Median speedup: {np.median(all_speedups):.2f}x")
    print(f"Min speedup: {np.min(all_speedups):.2f}x")
    print(f"Max speedup: {np.max(all_speedups):.2f}x")
    print("=" * 100)

    # Generate heatmap
    plot_heatmap(speedups, batch_sizes, hidden_sizes)

    print("\nBenchmark complete!")


if __name__ == "__main__":
    main()
