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

Benchmark: MXFP8 Quantization Backend Comparison (CUDA vs CuTe-DSL)

Compares the performance of CUDA and CuTe-DSL backends for MXFP8 quantization
across different M and K dimensions. Generates heatmaps showing relative
performance (speedup of CuTe-DSL over CUDA).

Can also measure achieved memory bandwidth in TB/s for the CuTe-DSL backend.

Usage:
    # Speedup comparison mode (default)
    python bench_mxfp8_quantize_backend_comparison.py

    # Bandwidth measurement mode (cute-dsl only)
    python bench_mxfp8_quantize_backend_comparison.py --bandwidth

Requirements:
    - Blackwell GPU (SM100+) for CuTe-DSL backend
    - matplotlib for visualization
"""

import argparse
import numpy as np
import torch
from typing import Dict, List, Tuple

from flashinfer.testing.utils import bench_gpu_time

# Constants for bandwidth calculation
SF_VEC_SIZE = 32  # Scale factor vector size for MXFP8


def get_cc():
    """Get CUDA compute capability."""
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor


def bench_mxfp8_quantize(
    m: int,
    k: int,
    dtype: torch.dtype,
    is_sf_swizzled_layout: bool,
    backend: str,
) -> float:
    """
    Benchmark MXFP8 quantization for a specific configuration.

    Args:
        m: Number of rows
        k: Number of columns
        dtype: Input dtype (torch.float16 or torch.bfloat16)
        is_sf_swizzled_layout: Whether to use swizzled scale factor layout
        backend: "cuda" or "cute-dsl"

    Returns:
        Median execution time in milliseconds
    """
    import flashinfer

    # Create input tensor
    x = torch.randn(m, k, device="cuda", dtype=dtype)

    # Warmup and get output shapes
    _ = flashinfer.mxfp8_quantize(
        x,
        is_sf_swizzled_layout=is_sf_swizzled_layout,
        backend=backend,
    )

    # Benchmark
    def run_kernel():
        flashinfer.mxfp8_quantize(
            x,
            is_sf_swizzled_layout=is_sf_swizzled_layout,
            backend=backend,
        )

    times = bench_gpu_time(
        fn=run_kernel,
        enable_cupti=True,
        dry_run_iters=5,
        repeat_iters=30,
        cold_l2_cache=True,
        use_cuda_graph=False,
    )

    return np.median(times)


def compute_bandwidth_tb_per_sec(
    m: int, k: int, dtype: torch.dtype, time_ms: float
) -> float:
    """
    Compute achieved memory bandwidth in TB/s.

    Memory bandwidth calculation for mxfp8_quantize:
    - Read: input tensor (2 bytes per element for fp16/bf16)
    - Write: quantized tensor (1 byte per element, fp8)
    - Write: scale factors (1 byte per scale factor)

    Args:
        m: Number of rows
        k: Number of columns
        dtype: Input dtype (determines bytes per element)
        time_ms: Execution time in milliseconds

    Returns:
        Achieved bandwidth in TB/s
    """
    input_dtype_bytes = 2  # fp16 or bf16

    num_elements = m * k
    num_scale_factors = num_elements // SF_VEC_SIZE

    # Total bytes transferred
    problem_bytes = (
        num_elements * input_dtype_bytes  # input read
        + num_elements * 1  # fp8 output write
        + num_scale_factors * 1  # scale factors write
    )

    # Convert ms to seconds, bytes to TB
    tb_per_sec = problem_bytes / (1e9 * time_ms)  # 1e9 = 10^12 bytes/TB / 10^3 ms/s
    return tb_per_sec


def run_bandwidth_sweep(
    m_values: List[int],
    k_values: List[int],
    dtype: torch.dtype,
    is_sf_swizzled_layout: bool,
) -> Dict[Tuple[int, int], float]:
    """
    Run bandwidth benchmark sweep for CuTe-DSL backend only.

    Returns:
        Dictionary mapping (m, k) to achieved bandwidth in TB/s
    """
    bandwidth_results = {}

    total = len(m_values) * len(k_values)
    current = 0

    layout_str = "swizzled" if is_sf_swizzled_layout else "linear"
    print(f"\nBenchmarking {layout_str} layout, dtype={dtype} (CuTe-DSL bandwidth)")
    print("=" * 60)

    for m in m_values:
        for k in k_values:
            current += 1
            print(f"[{current}/{total}] M={m:5d}, K={k:5d} ... ", end="", flush=True)

            # Benchmark CuTe-DSL backend only
            time_ms = bench_mxfp8_quantize(
                m, k, dtype, is_sf_swizzled_layout, backend="cute-dsl"
            )

            # Compute bandwidth
            bandwidth = compute_bandwidth_tb_per_sec(m, k, dtype, time_ms)
            bandwidth_results[(m, k)] = bandwidth

            print(f"time={time_ms:.3f}ms, bandwidth={bandwidth:.2f} TB/s")

    return bandwidth_results


def run_benchmark_sweep(
    m_values: List[int],
    k_values: List[int],
    dtype: torch.dtype,
    is_sf_swizzled_layout: bool,
) -> Tuple[Dict[Tuple[int, int], float], Dict[Tuple[int, int], float]]:
    """
    Run benchmark sweep for both backends.

    Returns:
        Tuple of (cuda_times, cute_dsl_times) dictionaries
    """
    cuda_times = {}
    cute_dsl_times = {}

    total = len(m_values) * len(k_values)
    current = 0

    layout_str = "swizzled" if is_sf_swizzled_layout else "linear"
    print(f"\nBenchmarking {layout_str} layout, dtype={dtype}")
    print("=" * 60)

    for m in m_values:
        for k in k_values:
            current += 1
            print(f"[{current}/{total}] M={m:5d}, K={k:5d} ... ", end="", flush=True)

            # Benchmark CUDA backend
            cuda_time = bench_mxfp8_quantize(
                m, k, dtype, is_sf_swizzled_layout, backend="cuda"
            )
            cuda_times[(m, k)] = cuda_time

            # Benchmark CuTe-DSL backend
            cute_dsl_time = bench_mxfp8_quantize(
                m, k, dtype, is_sf_swizzled_layout, backend="cute-dsl"
            )
            cute_dsl_times[(m, k)] = cute_dsl_time

            # Compute speedup
            speedup = cuda_time / cute_dsl_time
            speedup_str = (
                f"{speedup:.2f}x" if speedup >= 1 else f"{1 / speedup:.2f}x slower"
            )
            print(
                f"CUDA={cuda_time:.3f}ms, CuTe-DSL={cute_dsl_time:.3f}ms, "
                f"Speedup={speedup_str}"
            )

    return cuda_times, cute_dsl_times


def create_heatmap(
    m_values: List[int],
    k_values: List[int],
    cuda_times: Dict[Tuple[int, int], float],
    cute_dsl_times: Dict[Tuple[int, int], float],
    title: str,
    output_file: str,
):
    """
    Create a heatmap showing relative performance (CuTe-DSL speedup over CUDA).

    Values > 1.0 mean CuTe-DSL is faster, < 1.0 means CUDA is faster.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError:
        print("matplotlib not installed, skipping heatmap generation")
        return

    # Create speedup matrix (CUDA time / CuTe-DSL time)
    # > 1.0 means CuTe-DSL is faster
    speedup_matrix = np.zeros((len(m_values), len(k_values)))

    for i, m in enumerate(m_values):
        for j, k in enumerate(k_values):
            cuda_time = cuda_times.get((m, k), float("nan"))
            cute_dsl_time = cute_dsl_times.get((m, k), float("nan"))
            if cute_dsl_time > 0:
                speedup_matrix[i, j] = cuda_time / cute_dsl_time
            else:
                speedup_matrix[i, j] = float("nan")

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Create diverging colormap centered at 1.0
    # Green = CuTe-DSL faster (>1), Red = CUDA faster (<1)
    vmin = min(0.5, np.nanmin(speedup_matrix))
    vmax = max(2.0, np.nanmax(speedup_matrix))

    # Use log scale centered at 1.0 for better visualization
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=1.0, vmax=vmax)

    # Create heatmap
    im = ax.imshow(
        speedup_matrix,
        cmap="RdYlGn",  # Red-Yellow-Green: red=CUDA faster, green=CuTe-DSL faster
        norm=norm,
        aspect="auto",
    )

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel("Speedup (CUDA time / CuTe-DSL time)", rotation=-90, va="bottom")

    # Set ticks and labels
    ax.set_xticks(np.arange(len(k_values)))
    ax.set_yticks(np.arange(len(m_values)))
    ax.set_xticklabels([str(k) for k in k_values])
    ax.set_yticklabels([str(m) for m in m_values])

    # Rotate x-axis labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add value annotations
    for i in range(len(m_values)):
        for j in range(len(k_values)):
            value = speedup_matrix[i, j]
            if not np.isnan(value):
                # Choose text color based on background
                text_color = "white" if value < 0.7 or value > 1.5 else "black"
                ax.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=8,
                )

    # Labels and title
    ax.set_xlabel("K (columns)")
    ax.set_ylabel("M (rows)")
    ax.set_title(title + "\n(>1.0 = CuTe-DSL faster, <1.0 = CUDA faster)")

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Saved heatmap to {output_file}")
    plt.close()


def create_bandwidth_heatmap(
    m_values: List[int],
    k_values: List[int],
    bandwidth_results: Dict[Tuple[int, int], float],
    title: str,
    output_file: str,
):
    """
    Create a heatmap showing achieved memory bandwidth in TB/s.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping heatmap generation")
        return

    # Create bandwidth matrix
    bandwidth_matrix = np.zeros((len(m_values), len(k_values)))

    for i, m in enumerate(m_values):
        for j, k in enumerate(k_values):
            bandwidth_matrix[i, j] = bandwidth_results.get((m, k), float("nan"))

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Use sequential colormap (higher bandwidth = better = greener)
    vmin = np.nanmin(bandwidth_matrix)
    vmax = np.nanmax(bandwidth_matrix)

    # Create heatmap with viridis colormap (good for sequential data)
    im = ax.imshow(
        bandwidth_matrix,
        cmap="YlGn",  # Yellow-Green: darker green = higher bandwidth
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
    )

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel("Achieved Bandwidth (TB/s)", rotation=-90, va="bottom")

    # Set ticks and labels
    ax.set_xticks(np.arange(len(k_values)))
    ax.set_yticks(np.arange(len(m_values)))
    ax.set_xticklabels([str(k) for k in k_values])
    ax.set_yticklabels([str(m) for m in m_values])

    # Rotate x-axis labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add value annotations
    for i in range(len(m_values)):
        for j in range(len(k_values)):
            value = bandwidth_matrix[i, j]
            if not np.isnan(value):
                # Choose text color based on background brightness
                normalized = (value - vmin) / (vmax - vmin) if vmax > vmin else 0.5
                text_color = "white" if normalized > 0.6 else "black"
                ax.text(
                    j,
                    i,
                    f"{value:.1f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=8,
                )

    # Labels and title
    ax.set_xlabel("K (columns)")
    ax.set_ylabel("M (rows)")
    ax.set_title(title)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Saved heatmap to {output_file}")
    plt.close()


def print_bandwidth_summary_table(
    m_values: List[int],
    k_values: List[int],
    bandwidth_results: Dict[Tuple[int, int], float],
    layout_name: str,
):
    """Print a summary table of bandwidth results."""
    print(f"\n{'=' * 80}")
    print(f"Bandwidth Summary: {layout_name} (TB/s)")
    print(f"{'=' * 80}")

    # Header
    header = "M\\K".ljust(8)
    for k in k_values:
        header += f"{k:>8}"
    print(header)
    print("-" * (8 + 8 * len(k_values)))

    # Data rows
    for m in m_values:
        row = f"{m:<8}"
        for k in k_values:
            bandwidth = bandwidth_results.get((m, k), float("nan"))
            if not np.isnan(bandwidth):
                row += f"{bandwidth:>8.1f}"
            else:
                row += f"{'N/A':>8}"
        print(row)

    # Compute overall statistics
    bandwidths = [b for b in bandwidth_results.values() if not np.isnan(b)]

    if bandwidths:
        print("\nStatistics:")
        print(f"  Mean bandwidth: {np.mean(bandwidths):.2f} TB/s")
        print(f"  Min bandwidth:  {min(bandwidths):.2f} TB/s")
        print(f"  Max bandwidth:  {max(bandwidths):.2f} TB/s")
        print(f"  Std deviation:  {np.std(bandwidths):.2f} TB/s")


def print_summary_table(
    m_values: List[int],
    k_values: List[int],
    cuda_times: Dict[Tuple[int, int], float],
    cute_dsl_times: Dict[Tuple[int, int], float],
    layout_name: str,
):
    """Print a summary table of results."""
    print(f"\n{'=' * 80}")
    print(f"Summary: {layout_name}")
    print(f"{'=' * 80}")

    # Header
    header = "M\\K".ljust(8)
    for k in k_values:
        header += f"{k:>8}"
    print(header)
    print("-" * (8 + 8 * len(k_values)))

    # Data rows
    for m in m_values:
        row = f"{m:<8}"
        for k in k_values:
            cuda_time = cuda_times.get((m, k), float("nan"))
            cute_dsl_time = cute_dsl_times.get((m, k), float("nan"))
            if cute_dsl_time > 0:
                speedup = cuda_time / cute_dsl_time
                row += f"{speedup:>8.2f}"
            else:
                row += f"{'N/A':>8}"
        print(row)

    # Compute overall statistics
    speedups = []
    for m in m_values:
        for k in k_values:
            cuda_time = cuda_times.get((m, k))
            cute_dsl_time = cute_dsl_times.get((m, k))
            if cuda_time and cute_dsl_time and cute_dsl_time > 0:
                speedups.append(cuda_time / cute_dsl_time)

    if speedups:
        print("\nStatistics:")
        print(f"  Geometric mean speedup: {np.exp(np.mean(np.log(speedups))):.2f}x")
        print(f"  Min speedup: {min(speedups):.2f}x")
        print(f"  Max speedup: {max(speedups):.2f}x")
        print(
            f"  Cases where CuTe-DSL faster: {sum(1 for s in speedups if s > 1)}/{len(speedups)}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark MXFP8 Quantization: CUDA vs CuTe-DSL"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16"],
        help="Input data type",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="mxfp8_backend_comparison",
        help="Prefix for output files",
    )
    parser.add_argument(
        "--bandwidth",
        action="store_true",
        help="Measure achieved memory bandwidth (TB/s) for CuTe-DSL backend only, "
        "instead of comparing speedup between CUDA and CuTe-DSL",
    )
    args = parser.parse_args()

    # Check GPU capability
    cc = get_cc()
    print(f"GPU Compute Capability: SM{cc}")

    if cc < 100:
        print("ERROR: CuTe-DSL backend requires Blackwell GPU (SM100+)")
        return

    # Set dtype
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    print(f"Data type: {dtype}")

    # Define sweep ranges (powers of 2 + common transformer hidden dimensions)
    m_values = [
        128,
        256,
        384,
        512,
        768,
        1024,
        1536,
        2048,
        3072,
        4096,
        6144,
        8192,
        12288,
        16384,
        32768,
    ]
    k_values = [
        128,
        256,
        384,
        512,
        768,
        1024,
        1536,
        2048,
        3072,
        4096,
        5120,
        6144,
        8192,
        12288,
        16384,
    ]

    print(f"\nM values: {m_values}")
    print(f"K values: {k_values}")

    if args.bandwidth:
        # Bandwidth measurement mode: CuTe-DSL only
        print("\n" + "=" * 80)
        print("BANDWIDTH MEASUREMENT MODE (CuTe-DSL only)")
        print("=" * 80)

        # Benchmark linear layout (non-swizzled)
        print("\n" + "=" * 80)
        print("BENCHMARKING LINEAR (NON-SWIZZLED) LAYOUT - BANDWIDTH")
        print("=" * 80)

        bandwidth_linear = run_bandwidth_sweep(
            m_values, k_values, dtype, is_sf_swizzled_layout=False
        )

        print_bandwidth_summary_table(
            m_values, k_values, bandwidth_linear, "Linear Layout"
        )

        create_bandwidth_heatmap(
            m_values,
            k_values,
            bandwidth_linear,
            f"MXFP8 Quantization Bandwidth (CuTe-DSL) - Linear Layout - {args.dtype}",
            f"{args.output_prefix}_bandwidth_linear_{args.dtype}.png",
        )

        # Benchmark swizzled layout
        print("\n" + "=" * 80)
        print("BENCHMARKING SWIZZLED LAYOUT - BANDWIDTH")
        print("=" * 80)

        bandwidth_swizzled = run_bandwidth_sweep(
            m_values, k_values, dtype, is_sf_swizzled_layout=True
        )

        print_bandwidth_summary_table(
            m_values, k_values, bandwidth_swizzled, "Swizzled Layout"
        )

        create_bandwidth_heatmap(
            m_values,
            k_values,
            bandwidth_swizzled,
            f"MXFP8 Quantization Bandwidth (CuTe-DSL) - Swizzled Layout - {args.dtype}",
            f"{args.output_prefix}_bandwidth_swizzled_{args.dtype}.png",
        )

    else:
        # Speedup comparison mode: CUDA vs CuTe-DSL
        # Benchmark linear layout (non-swizzled)
        print("\n" + "=" * 80)
        print("BENCHMARKING LINEAR (NON-SWIZZLED) LAYOUT")
        print("=" * 80)

        cuda_times_linear, cute_dsl_times_linear = run_benchmark_sweep(
            m_values, k_values, dtype, is_sf_swizzled_layout=False
        )

        print_summary_table(
            m_values,
            k_values,
            cuda_times_linear,
            cute_dsl_times_linear,
            "Linear Layout",
        )

        create_heatmap(
            m_values,
            k_values,
            cuda_times_linear,
            cute_dsl_times_linear,
            f"MXFP8 Quantization Speedup (CuTe-DSL vs CUDA) - Linear Layout - {args.dtype}",
            f"{args.output_prefix}_linear_{args.dtype}.png",
        )

        # Benchmark swizzled layout
        print("\n" + "=" * 80)
        print("BENCHMARKING SWIZZLED LAYOUT")
        print("=" * 80)

        cuda_times_swizzled, cute_dsl_times_swizzled = run_benchmark_sweep(
            m_values, k_values, dtype, is_sf_swizzled_layout=True
        )

        print_summary_table(
            m_values,
            k_values,
            cuda_times_swizzled,
            cute_dsl_times_swizzled,
            "Swizzled Layout",
        )

        create_heatmap(
            m_values,
            k_values,
            cuda_times_swizzled,
            cute_dsl_times_swizzled,
            f"MXFP8 Quantization Speedup (CuTe-DSL vs CUDA) - Swizzled Layout - {args.dtype}",
            f"{args.output_prefix}_swizzled_{args.dtype}.png",
        )

    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
