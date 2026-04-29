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

Benchmark: NVFP4 Quantization Backend Comparison (CUDA vs CuTe-DSL)

Compares the performance of CUDA and CuTe-DSL backends for NVFP4 quantization
across different M and K dimensions. Supports both swizzled 128x4 and linear
scale factor layouts. Each configuration is verified for correctness before
timing. Generates heatmaps showing relative performance (speedup of CuTe-DSL
over CUDA).

Can also measure achieved memory bandwidth in TB/s for the CuTe-DSL backend.

Usage:
    # Speedup comparison mode (default, includes correctness verification)
    python bench_nvfp4_quantize_backend_comparison.py

    # Bandwidth measurement mode (cute-dsl only)
    python bench_nvfp4_quantize_backend_comparison.py --bandwidth

Requirements:
    - Blackwell GPU (SM100+) for CuTe-DSL backend
    - matplotlib for visualization
"""

import argparse
import numpy as np
import torch
from typing import Dict, List, Tuple

from flashinfer.testing.utils import bench_gpu_time

# Constants for NVFP4
NVFP4_SF_VEC_SIZE = 16
FLOAT4_E2M1_MAX = 6.0
FLOAT8_E4M3_MAX = float(torch.finfo(torch.float8_e4m3fn).max)


def get_cc():
    """Get CUDA compute capability."""
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor


def verify_nvfp4_correctness(
    m: int,
    k: int,
    dtype: torch.dtype,
    is_sf_swizzled_layout: bool,
) -> Tuple[bool, str, float, float]:
    """
    Verify that both backends produce correct outputs via roundtrip test.

    Returns:
        Tuple of (success, message, quant_match_pct, scale_match_pct)
        On failure, quant_match_pct and scale_match_pct are 0.0
    """
    from flashinfer.quantization.fp4_quantization import (
        e2m1_and_ufp8sf_scale_to_float,
        fp4_quantize,
    )

    torch.manual_seed(42)
    x = torch.randn(m, k, device="cuda", dtype=dtype)
    amax = x.abs().max().to(torch.float32)
    global_sf = (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / amax).cuda()

    try:
        # Test CUDA backend
        quant_cuda, scale_cuda = fp4_quantize(
            x,
            global_sf,
            sf_vec_size=16,
            sf_use_ue8m0=False,
            is_sf_swizzled_layout=is_sf_swizzled_layout,
            backend="cuda",
        )
        dq_cuda = e2m1_and_ufp8sf_scale_to_float(
            quant_cuda.cpu().view(torch.uint8),
            scale_cuda.cpu().view(torch.uint8).reshape(-1),
            torch.tensor([1.0]),
            16,
            1,
            is_sf_swizzled_layout,
        )

        # Test CuTe-DSL backend
        quant_cute, scale_cute = fp4_quantize(
            x,
            global_sf,
            sf_vec_size=16,
            sf_use_ue8m0=False,
            is_sf_swizzled_layout=is_sf_swizzled_layout,
            backend="cute-dsl",
        )
        dq_cute = e2m1_and_ufp8sf_scale_to_float(
            quant_cute.cpu().view(torch.uint8),
            scale_cute.cpu().view(torch.uint8).reshape(-1),
            torch.tensor([1.0]),
            16,
            1,
            is_sf_swizzled_layout,
        )

        # Check shapes match
        if quant_cuda.shape != quant_cute.shape:
            return (
                False,
                f"Quant shape mismatch: CUDA={quant_cuda.shape}, CuTe={quant_cute.shape}",
                0.0,
                0.0,
            )
        if scale_cuda.shape != scale_cute.shape:
            return (
                False,
                f"Scale shape mismatch: CUDA={scale_cuda.shape}, CuTe={scale_cute.shape}",
                0.0,
                0.0,
            )

        # Check roundtrip quality for both backends (cosine similarity)
        x_f32 = x.cpu().to(torch.float32).view(1, -1)
        dq_cuda_f32 = dq_cuda.cpu().to(torch.float32).view(1, -1)
        dq_cute_f32 = dq_cute.cpu().to(torch.float32).view(1, -1)

        cos_sim_cuda = torch.nn.functional.cosine_similarity(x_f32, dq_cuda_f32).item()
        cos_sim_cute = torch.nn.functional.cosine_similarity(x_f32, dq_cute_f32).item()

        # Check backend agreement
        quant_match_pct = (quant_cuda == quant_cute).float().mean().item() * 100
        scale_match_pct = (scale_cuda == scale_cute).float().mean().item() * 100

        # FP4 quantization should have cosine similarity > 0.9
        if cos_sim_cuda < 0.9:
            return (
                False,
                f"CUDA roundtrip quality too low: cos_sim={cos_sim_cuda:.4f}",
                quant_match_pct,
                scale_match_pct,
            )
        if cos_sim_cute < 0.9:
            return (
                False,
                f"CuTe-DSL roundtrip quality too low: cos_sim={cos_sim_cute:.4f}",
                quant_match_pct,
                scale_match_pct,
            )

        return True, "OK", quant_match_pct, scale_match_pct

    except Exception as e:
        return False, f"Exception: {e}", 0.0, 0.0


def bench_nvfp4_quantize(
    m: int,
    k: int,
    dtype: torch.dtype,
    is_sf_swizzled_layout: bool,
    backend: str,
) -> float:
    """
    Benchmark NVFP4 quantization for a specific configuration.

    Returns:
        Median execution time in milliseconds
    """
    from flashinfer.quantization.fp4_quantization import fp4_quantize

    x = torch.randn(m, k, device="cuda", dtype=dtype)
    amax = x.abs().max().to(torch.float32)
    global_sf = (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / amax).cuda()

    # Warmup
    _ = fp4_quantize(
        x,
        global_sf,
        sf_vec_size=16,
        sf_use_ue8m0=False,
        is_sf_swizzled_layout=is_sf_swizzled_layout,
        backend=backend,
    )

    def run_kernel():
        fp4_quantize(
            x,
            global_sf,
            sf_vec_size=16,
            sf_use_ue8m0=False,
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

    Memory bandwidth calculation for nvfp4_quantize:
    - Read: input tensor (2 bytes per element for fp16/bf16)
    - Write: quantized tensor (0.5 bytes per element, since fp4 = 4 bits)
    - Write: scale factors (1 byte per scale factor)
    """
    input_dtype_bytes = 2  # fp16 or bf16

    num_elements = m * k
    num_scale_factors = num_elements // NVFP4_SF_VEC_SIZE

    problem_bytes = (
        num_elements * input_dtype_bytes  # input read
        + num_elements // 2  # fp4 output write
        + num_scale_factors * 1  # scale factors write
    )

    tb_per_sec = problem_bytes / (1e9 * time_ms)
    return tb_per_sec


def run_bandwidth_sweep(
    m_values: List[int],
    k_values: List[int],
    dtype: torch.dtype,
    is_sf_swizzled_layout: bool,
) -> Dict[Tuple[int, int], float]:
    """Run bandwidth benchmark sweep for CuTe-DSL backend only."""
    bandwidth_results = {}

    total = len(m_values) * len(k_values)
    current = 0

    layout_str = "swizzled" if is_sf_swizzled_layout else "linear"
    print(
        f"\nBenchmarking NVFP4 {layout_str} layout, dtype={dtype} (CuTe-DSL bandwidth)"
    )
    print("=" * 60)

    for m in m_values:
        for k in k_values:
            current += 1
            print(f"[{current}/{total}] M={m:5d}, K={k:5d} ... ", end="", flush=True)

            time_ms = bench_nvfp4_quantize(
                m, k, dtype, is_sf_swizzled_layout, backend="cute-dsl"
            )

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
    """Run benchmark sweep for both backends with inline correctness verification."""
    cuda_times = {}
    cute_dsl_times = {}
    failures = []

    total = len(m_values) * len(k_values)
    current = 0

    layout_str = "swizzled" if is_sf_swizzled_layout else "linear"
    print(f"\nBenchmarking NVFP4 {layout_str} layout, dtype={dtype}")
    print("=" * 95)
    print(
        f"{'Progress':<12} {'M':>5}  {'K':>5}  | "
        f"{'--Match--':^14} | "
        f"{'-------Timing-------':^28}"
    )
    print(
        f"{'':12} {'':>5}  {'':>5}  | "
        f"{'quant':>6} {'scale':>6} | "
        f"{'CUDA':>8} {'CuTe':>8} {'Speedup':>10}"
    )
    print("-" * 95)

    for m in m_values:
        for k in k_values:
            current += 1

            # Verify correctness first
            success, verify_msg, quant_match, scale_match = verify_nvfp4_correctness(
                m, k, dtype, is_sf_swizzled_layout
            )
            if not success:
                failures.append((m, k, verify_msg))
                print(f"[{current:3d}/{total}]  {m:5d}  {k:5d}  | FAIL: {verify_msg}")
                continue

            # Benchmark CUDA backend
            cuda_time = bench_nvfp4_quantize(
                m, k, dtype, is_sf_swizzled_layout, backend="cuda"
            )
            cuda_times[(m, k)] = cuda_time

            # Benchmark CuTe-DSL backend
            cute_dsl_time = bench_nvfp4_quantize(
                m, k, dtype, is_sf_swizzled_layout, backend="cute-dsl"
            )
            cute_dsl_times[(m, k)] = cute_dsl_time

            # Compute speedup
            speedup = cuda_time / cute_dsl_time
            speedup_str = (
                f"{speedup:.2f}x" if speedup >= 1 else f"{1 / speedup:.2f}x slower"
            )
            print(
                f"[{current:3d}/{total}]  {m:5d}  {k:5d}  | "
                f"{quant_match:5.1f}% {scale_match:6.1f}% | "
                f"{cuda_time:7.3f}ms {cute_dsl_time:7.3f}ms {speedup_str:>10}"
            )

    if failures:
        print(f"\nWARNING: {len(failures)}/{total} configurations failed verification:")
        for m, k, msg in failures:
            print(f"  - M={m}, K={k}: {msg}")

    return cuda_times, cute_dsl_times


def create_heatmap(
    m_values: List[int],
    k_values: List[int],
    cuda_times: Dict[Tuple[int, int], float],
    cute_dsl_times: Dict[Tuple[int, int], float],
    title: str,
    output_file: str,
):
    """Create a heatmap showing relative performance (CuTe-DSL speedup over CUDA)."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError:
        print("matplotlib not installed, skipping heatmap generation")
        return

    speedup_matrix = np.zeros((len(m_values), len(k_values)))

    for i, m in enumerate(m_values):
        for j, k in enumerate(k_values):
            cuda_time = cuda_times.get((m, k), float("nan"))
            cute_dsl_time = cute_dsl_times.get((m, k), float("nan"))
            if cute_dsl_time > 0:
                speedup_matrix[i, j] = cuda_time / cute_dsl_time
            else:
                speedup_matrix[i, j] = float("nan")

    fig, ax = plt.subplots(figsize=(12, 10))

    vmin = min(0.5, np.nanmin(speedup_matrix))
    vmax = max(2.0, np.nanmax(speedup_matrix))
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=1.0, vmax=vmax)

    im = ax.imshow(speedup_matrix, cmap="RdYlGn", norm=norm, aspect="auto")

    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel("Speedup (CUDA time / CuTe-DSL time)", rotation=-90, va="bottom")

    ax.set_xticks(np.arange(len(k_values)))
    ax.set_yticks(np.arange(len(m_values)))
    ax.set_xticklabels([str(k) for k in k_values])
    ax.set_yticklabels([str(m) for m in m_values])

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(len(m_values)):
        for j in range(len(k_values)):
            value = speedup_matrix[i, j]
            if not np.isnan(value):
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
    """Create a heatmap showing achieved memory bandwidth in TB/s."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping heatmap generation")
        return

    bandwidth_matrix = np.zeros((len(m_values), len(k_values)))

    for i, m in enumerate(m_values):
        for j, k in enumerate(k_values):
            bandwidth_matrix[i, j] = bandwidth_results.get((m, k), float("nan"))

    fig, ax = plt.subplots(figsize=(12, 10))

    vmin = np.nanmin(bandwidth_matrix)
    vmax = np.nanmax(bandwidth_matrix)

    im = ax.imshow(bandwidth_matrix, cmap="YlGn", vmin=vmin, vmax=vmax, aspect="auto")

    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel("Achieved Bandwidth (TB/s)", rotation=-90, va="bottom")

    ax.set_xticks(np.arange(len(k_values)))
    ax.set_yticks(np.arange(len(m_values)))
    ax.set_xticklabels([str(k) for k in k_values])
    ax.set_yticklabels([str(m) for m in m_values])

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(len(m_values)):
        for j in range(len(k_values)):
            value = bandwidth_matrix[i, j]
            if not np.isnan(value):
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
    layout_name: str = "Swizzled Layout",
):
    """Print a summary table of bandwidth results."""
    print(f"\n{'=' * 80}")
    print(f"Bandwidth Summary: NVFP4 {layout_name} (TB/s)")
    print(f"{'=' * 80}")

    header = "M\\K".ljust(8)
    for k in k_values:
        header += f"{k:>8}"
    print(header)
    print("-" * (8 + 8 * len(k_values)))

    for m in m_values:
        row = f"{m:<8}"
        for k in k_values:
            bandwidth = bandwidth_results.get((m, k), float("nan"))
            if not np.isnan(bandwidth):
                row += f"{bandwidth:>8.1f}"
            else:
                row += f"{'N/A':>8}"
        print(row)

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
    layout_name: str = "Swizzled Layout",
):
    """Print a summary table of results."""
    print(f"\n{'=' * 80}")
    print(f"Summary: NVFP4 {layout_name} (Speedup: CUDA time / CuTe-DSL time)")
    print(f"{'=' * 80}")

    header = "M\\K".ljust(8)
    for k in k_values:
        header += f"{k:>8}"
    print(header)
    print("-" * (8 + 8 * len(k_values)))

    for m in m_values:
        row = f"{m:<8}"
        for k in k_values:
            cuda_time = cuda_times.get((m, k), float("nan"))
            cute_dsl_time = cute_dsl_times.get((m, k), float("nan"))
            if cute_dsl_time > 0 and not np.isnan(cuda_time):
                speedup = cuda_time / cute_dsl_time
                row += f"{speedup:>8.2f}"
            else:
                row += f"{'N/A':>8}"
        print(row)

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
        description="Benchmark NVFP4 quantization backends"
    )
    parser.add_argument(
        "--bandwidth",
        action="store_true",
        help="Run bandwidth benchmark (CuTe-DSL only) instead of comparison",
    )
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16"],
        default="bfloat16",
        help="Input data type (default: bfloat16)",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="nvfp4_quantize_backend",
        help="Output file prefix for heatmaps",
    )
    args = parser.parse_args()

    # Check compute capability
    cc = get_cc()
    print(f"GPU Compute Capability: SM{cc}")

    if cc < 100:
        print("ERROR: CuTe-DSL backend requires Blackwell GPU (SM100+)")
        return

    # Get dtype
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    print(f"Data type: {dtype}")

    # Define sweep ranges
    # K constraints:
    # - Linear layout: K must be a multiple of 16 (NVFP4_SF_VEC_SIZE)
    # - Swizzled layout: K must be a multiple of 64 because K/16 (SF blocks
    #   per row) must be a multiple of 4 for the swizzled padding
    # We use K values that satisfy both constraints (multiples of 64)
    m_values = [
        1,
        2,
        4,
        8,
        16,
        32,
        64,
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
        print("\n" + "=" * 80)
        print("BANDWIDTH MEASUREMENT MODE (CuTe-DSL only)")
        print("=" * 80)

        # Benchmark linear layout
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
            f"NVFP4 Quantization Bandwidth (CuTe-DSL) - Linear Layout - {args.dtype}",
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
            f"NVFP4 Quantization Bandwidth (CuTe-DSL) - Swizzled Layout - {args.dtype}",
            f"{args.output_prefix}_bandwidth_swizzled_{args.dtype}.png",
        )
    else:
        # Speedup comparison mode: CUDA vs CuTe-DSL
        # Benchmark linear layout
        print("\n" + "=" * 80)
        print("BENCHMARKING LINEAR (NON-SWIZZLED) LAYOUT")
        print("=" * 80)

        cuda_times_linear, cute_dsl_times_linear = run_benchmark_sweep(
            m_values,
            k_values,
            dtype,
            is_sf_swizzled_layout=False,
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
            f"NVFP4 Quantization Speedup (CuTe-DSL vs CUDA) - Linear Layout - {args.dtype}",
            f"{args.output_prefix}_comparison_linear_{args.dtype}.png",
        )

        # Benchmark swizzled layout
        print("\n" + "=" * 80)
        print("BENCHMARKING SWIZZLED LAYOUT")
        print("=" * 80)

        cuda_times_swizzled, cute_dsl_times_swizzled = run_benchmark_sweep(
            m_values,
            k_values,
            dtype,
            is_sf_swizzled_layout=True,
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
            f"NVFP4 Quantization Speedup (CuTe-DSL vs CUDA) - Swizzled Layout - {args.dtype}",
            f"{args.output_prefix}_comparison_swizzled_{args.dtype}.png",
        )

    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
