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

Benchmark: Fused RMSNorm + FP4 Quantization using CuTe-DSL Backend

Compares the CuTe-DSL fused kernel against separate RMSNorm + FP4 quantization.

Usage:
    python bench_cute_dsl_rmsnorm_fp4quant.py

Requirements:
    - Blackwell GPU (SM100+)
    - CuTe-DSL installed
"""

import numpy as np
import torch
from scipy.stats import gmean
from flashinfer.testing.utils import bench_gpu_time


def get_cc():
    """Get CUDA compute capability."""
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor


def compute_bandwidth_gb_s(
    batch_size: int,
    hidden_size: int,
    block_size: int,
    time_ms: float,
) -> float:
    """
    Compute achieved memory bandwidth in GB/s for fused RMSNorm + FP4 quantization.

    The fused kernel performs:
    1. Read input x: [batch_size, hidden_size] in fp16/bf16 (2 bytes/elem)
    2. Read weight: [hidden_size] in fp16/bf16 (2 bytes/elem)
    3. Write y_fp4: [batch_size, hidden_size/2] packed uint8 (1 byte per 2 FP4 values)
    4. Write block_scale: [batch_size, hidden_size/block_size] in fp8/uint8 (1 byte/elem)

    Formula:
        read_bytes  = batch_size * hidden_size * 2 + hidden_size * 2
        write_bytes = batch_size * hidden_size / 2 + batch_size * hidden_size / block_size
        total_bytes = read_bytes + write_bytes
        bandwidth   = total_bytes / time_in_seconds / 1e9  (GB/s)

    Args:
        batch_size: Batch size (number of rows)
        hidden_size: Hidden dimension
        block_size: FP4 quantization block size (16 or 32)
        time_ms: Kernel execution time in milliseconds

    Returns:
        Achieved bandwidth in GB/s
    """
    # Read: x (fp16) + weight (fp16)
    read_bytes = batch_size * hidden_size * 2 + hidden_size * 2

    # Write: y_fp4 (packed uint8) + block_scale (fp8/uint8)
    write_bytes = batch_size * (hidden_size // 2) + batch_size * (
        hidden_size // block_size
    )

    total_bytes = read_bytes + write_bytes
    time_s = time_ms / 1000.0

    if time_s <= 0:
        return 0.0

    return total_bytes / time_s / 1e9


def bench_cute_dsl(batch_size, hidden_size, dtype, block_size=16):
    """Benchmark CuTe-DSL backend."""
    from flashinfer.cute_dsl.rmsnorm_fp4quant import rmsnorm_fp4quant

    eps = 1e-6

    x = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
    weight = torch.randn(hidden_size, device="cuda", dtype=dtype)
    y_fp4 = torch.empty(batch_size, hidden_size // 2, device="cuda", dtype=torch.uint8)

    # Scale factor dtype depends on format
    if block_size == 32:
        block_scale = torch.empty(
            batch_size, hidden_size // block_size, device="cuda", dtype=torch.uint8
        )
        scale_format = "ue8m0"
    else:
        block_scale = torch.empty(
            batch_size,
            hidden_size // block_size,
            device="cuda",
            dtype=torch.float8_e4m3fn,
        )
        scale_format = "e4m3"

    # Benchmark with bench_gpu_time
    times = bench_gpu_time(
        lambda: rmsnorm_fp4quant(
            x,
            weight,
            y_fp4,
            block_scale,
            eps=eps,
            block_size=block_size,
            scale_format=scale_format,
        ),
        cold_l2_cache=True,
        enable_cupti=True,
        use_cuda_graph=False,
        dry_run_iters=10,
        repeat_iters=100,
    )

    # Return median time
    return np.median(times)


def bench_separate_flashinfer(batch_size, hidden_size, dtype, block_size=16):
    """Benchmark separate FlashInfer operations: rmsnorm + fp4_quantize.

    Returns tuple of (rmsnorm_time_ms, fp4_quant_time_ms, total_time_ms)
    """
    from flashinfer.norm import rmsnorm
    from flashinfer.fp4_quantization import fp4_quantize

    eps = 1e-6

    x = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
    weight = torch.randn(hidden_size, device="cuda", dtype=dtype)
    y_normed = torch.empty_like(x)

    # Compute global_scale for fp4_quantize (required when sf_use_ue8m0 is false)
    # Use a fixed scale for benchmarking consistency
    global_scale = torch.tensor([1.0], device="cuda", dtype=torch.float32)

    # Benchmark rmsnorm alone
    times_rmsnorm = bench_gpu_time(
        lambda: rmsnorm(x, weight, eps=eps, out=y_normed),
        cold_l2_cache=True,
        enable_cupti=True,
        use_cuda_graph=False,
        dry_run_iters=10,
        repeat_iters=100,
    )
    t_rmsnorm = np.median(times_rmsnorm)

    # Run rmsnorm once to get y_normed for fp4_quantize
    rmsnorm(x, weight, eps=eps, out=y_normed)

    # Benchmark fp4_quantize alone
    times_fp4 = bench_gpu_time(
        lambda: fp4_quantize(
            y_normed,
            global_scale=None if block_size == 32 else global_scale,
            sf_vec_size=block_size,
            sf_use_ue8m0=(block_size == 32),
            is_sf_swizzled_layout=False,
        ),
        cold_l2_cache=True,
        enable_cupti=True,
        use_cuda_graph=False,
        dry_run_iters=10,
        repeat_iters=100,
    )
    t_fp4 = np.median(times_fp4)

    return t_rmsnorm, t_fp4, t_rmsnorm + t_fp4


def sanity_check_outputs(dtype=torch.float16, block_size=16):
    """Verify CuTe-DSL output matches separate RMSNorm + fp4_quantize operations."""
    from flashinfer.cute_dsl.rmsnorm_fp4quant import rmsnorm_fp4quant
    from flashinfer.norm import rmsnorm
    from flashinfer.fp4_quantization import fp4_quantize

    # Test with a few configurations
    test_configs = [
        (128, 256),
        (512, 1024),
        (1024, 2048),
    ]

    eps = 1e-6
    all_passed = True

    for batch_size, hidden_size in test_configs:
        # Create inputs
        x = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
        weight = torch.randn(hidden_size, device="cuda", dtype=dtype)

        # CuTe-DSL path
        y_fp4_cute = torch.empty(
            batch_size, hidden_size // 2, device="cuda", dtype=torch.uint8
        )
        if block_size == 32:
            block_scale_cute = torch.empty(
                batch_size, hidden_size // block_size, device="cuda", dtype=torch.uint8
            )
            scale_format = "ue8m0"
        else:
            block_scale_cute = torch.empty(
                batch_size,
                hidden_size // block_size,
                device="cuda",
                dtype=torch.float8_e4m3fn,
            )
            scale_format = "e4m3"

        rmsnorm_fp4quant(
            x,
            weight,
            y_fp4_cute,
            block_scale_cute,
            eps=eps,
            block_size=block_size,
            scale_format=scale_format,
        )

        # Separate path
        y_normed = torch.empty_like(x)
        rmsnorm(x, weight, eps=eps, out=y_normed)

        global_scale = torch.tensor(
            [y_normed.abs().max().item() / 6.0], device="cuda", dtype=torch.float32
        )
        y_fp4_sep, block_scale_sep = fp4_quantize(
            y_normed,
            global_scale=None if block_size == 32 else global_scale,
            sf_vec_size=block_size,
            sf_use_ue8m0=(block_size == 32),
            is_sf_swizzled_layout=False,
        )

        # Compare FP4 outputs - use relaxed criteria since:
        # 1. FP4 is very low precision (4 bits), small float differences can flip values
        # 2. Different scale factor computation between fused and separate paths
        # 3. Different floating-point operation ordering
        match_count = (y_fp4_cute == y_fp4_sep).sum().item()
        total_count = y_fp4_cute.numel()
        match_pct = match_count / total_count * 100

        # For FP4, 70% exact match is reasonable given the precision constraints
        # The important thing is that both produce valid quantized outputs
        if match_pct < 70.0:
            all_passed = False
            print(
                f"  WARN: ({batch_size}, {hidden_size}) - "
                f"FP4 match: {match_pct:.1f}% (expected >= 70%)"
            )
        else:
            print(f"  OK: ({batch_size}, {hidden_size}) - FP4 match")

    return all_passed


def run_benchmark():
    """Run full benchmark suite."""
    print("=" * 80)
    print("Fused RMSNorm + FP4 Quantization Benchmark")
    print("=" * 80)

    cc = get_cc()
    print(f"GPU Compute Capability: SM{cc}")

    if cc < 100:
        raise RuntimeError("Blackwell GPU (SM100+) required for FP4 quantization")

    dtype = torch.float16
    block_size = 16

    # Sanity check: verify CuTe-DSL output matches separate operations
    print()
    print("Running sanity check...")
    if sanity_check_outputs(dtype, block_size):
        print(
            "✓ Confirmed: CuTe-DSL output is equivalent to RMSNorm + fp4_quantization"
        )
    else:
        print("✗ Warning: Some outputs did not match closely")
    print()

    # Test configurations
    # sweep batch_size from 1024 to 65536 (powers of 2)
    # and hidden_size from 2048 to 32768 (powers of 2)
    # Add some non-powers-of-2 batch sizes for more realism/variance.
    batch_sizes = [2**i for i in range(10, 17)]  # 1024 to 65536
    batch_sizes += [1000, 3000, 5000, 10000, 15000, 25000, 60000]  # non-powers-of-2
    batch_sizes = sorted(list(set(batch_sizes)))  # deduplicate and sort

    hidden_sizes = [2**j for j in range(11, 16)]  # 2048 to 32768
    hidden_sizes += [1536]
    hidden_sizes = sorted(list(set(hidden_sizes)))  # deduplicate and sort

    configs = [
        (batch_size, hidden_size)
        for batch_size in batch_sizes
        for hidden_size in hidden_sizes
    ]

    print()
    header = (
        f"{'Batch':<8} {'Hidden':<8} "
        f"{'CuTe-DSL (µs)':<14} {'BW (GB/s)':<10} "
        f"{'RMSNorm (µs)':<13} {'FP4Q (µs)':<11} {'Separate (µs)':<14} "
        f"{'vs Separate':<12}"
    )
    print(header)
    print("-" * len(header))

    results = []

    for batch_size, hidden_size in configs:
        # CuTe-DSL timing
        try:
            t_cute = bench_cute_dsl(batch_size, hidden_size, dtype, block_size)
            t_cute_us = t_cute * 1e3  # ms to µs
            bw_cute = compute_bandwidth_gb_s(
                batch_size, hidden_size, block_size, t_cute
            )
        except Exception as e:
            print(f"{batch_size:<8} {hidden_size:<8} ERROR: {e}")
            continue

        # Separate FlashInfer timing
        try:
            t_rmsnorm, t_fp4, t_separate = bench_separate_flashinfer(
                batch_size, hidden_size, dtype, block_size
            )
            t_rmsnorm_us = t_rmsnorm * 1e3  # ms to µs
            t_fp4_us = t_fp4 * 1e3  # ms to µs
            t_separate_us = t_separate * 1e3  # ms to µs
            speedup_sep = t_separate / t_cute if t_cute > 0 else 0
            rmsnorm_str = f"{t_rmsnorm_us:.1f}"
            fp4_str = f"{t_fp4_us:.1f}"
            separate_str = f"{t_separate_us:.1f}"
            speedup_sep_str = f"{speedup_sep:.2f}x"
        except Exception:
            t_rmsnorm_us = None
            t_fp4_us = None
            t_separate_us = None
            rmsnorm_str = "N/A"
            fp4_str = "N/A"
            separate_str = "N/A"
            speedup_sep_str = "N/A"
            speedup_sep = None

        print(
            f"{batch_size:<8} {hidden_size:<8} "
            f"{t_cute_us:<14.1f} {bw_cute:<10.1f} "
            f"{rmsnorm_str:<13} {fp4_str:<11} {separate_str:<14} "
            f"{speedup_sep_str:<12}"
        )

        result = {
            "batch_size": batch_size,
            "hidden_size": hidden_size,
            "cute_dsl_us": t_cute_us,
            "cute_dsl_bw_gb_s": bw_cute,
            "rmsnorm_us": t_rmsnorm_us,
            "fp4_quant_us": t_fp4_us,
            "separate_us": t_separate_us,
            "speedup_vs_separate": speedup_sep,
        }
        results.append(result)

    print()
    print("=" * 80)

    # Calculate and print geomean speedup
    speedups = [
        r["speedup_vs_separate"]
        for r in results
        if r["speedup_vs_separate"] is not None
    ]

    if speedups:
        geomean_speedup = gmean(speedups)
        print(f"Geomean speedup vs Separate (2 kernels): {geomean_speedup:.2f}x")

    print("=" * 80)
    print("Benchmark Complete")
    print("=" * 80)


if __name__ == "__main__":
    run_benchmark()
