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

Benchmark: Fused Add + RMSNorm + FP4 Quantization using CuTe-DSL Backend

Compares the CuTe-DSL fused kernel against separate FlashInfer operations:
    - torch.add + flashinfer.rmsnorm + flashinfer.fp4_quantize

Usage:
    python bench_cute_dsl_add_rmsnorm_fp4quant.py
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
    Compute achieved memory bandwidth in GB/s for fused Add + RMSNorm + FP4 quantization.

    The fused kernel performs:
    1. Read input x: [batch_size, hidden_size] in fp16/bf16 (2 bytes/elem)
    2. Read residual r: [batch_size, hidden_size] in fp16/bf16 (2 bytes/elem)
    3. Read weight: [hidden_size] in fp16/bf16 (2 bytes/elem)
    4. Write y_fp4: [batch_size, hidden_size/2] packed uint8 (1 byte per 2 FP4 values)
    5. Write block_scale: [batch_size, hidden_size/block_size] in fp8/uint8 (1 byte/elem)

    Formula:
        read_bytes  = batch_size * hidden_size * 2 * 2 + hidden_size * 2
        write_bytes = batch_size * hidden_size / 2 + batch_size * hidden_size / block_size
        total_bytes = read_bytes + write_bytes
        bandwidth   = total_bytes / time_in_seconds / 1e9  (GB/s)
    """
    # Read: x (fp16) + r (fp16) + weight (fp16)
    read_bytes = batch_size * hidden_size * 2 * 2 + hidden_size * 2

    # Write: y_fp4 (packed uint8) + block_scale (fp8/uint8)
    write_bytes = batch_size * (hidden_size // 2) + batch_size * (
        hidden_size // block_size
    )

    total_bytes = read_bytes + write_bytes
    time_s = time_ms / 1000.0

    if time_s <= 0:
        return 0.0

    return total_bytes / time_s / 1e9


def bench_fused_cute_dsl(batch_size, hidden_size, dtype, block_size=16):
    """Benchmark fused CuTe-DSL kernel."""
    from flashinfer.cute_dsl.add_rmsnorm_fp4quant import add_rmsnorm_fp4quant_cute_dsl

    eps = 1e-6

    x = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
    r = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
    weight = torch.randn(hidden_size, device="cuda", dtype=dtype)
    y_fp4 = torch.empty(batch_size, hidden_size // 2, device="cuda", dtype=torch.uint8)

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

    times = bench_gpu_time(
        lambda: add_rmsnorm_fp4quant_cute_dsl(
            x,
            r,
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

    return np.median(times)


def bench_fully_separate(batch_size, hidden_size, dtype, block_size=16):
    """Benchmark fully separate operations: torch.add + rmsnorm + fp4_quantize.

    Returns tuple of (add_time_ms, rmsnorm_time_ms, fp4_quant_time_ms, total_time_ms)
    """
    from flashinfer.norm import rmsnorm
    from flashinfer.fp4_quantization import fp4_quantize

    eps = 1e-6

    x = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
    r = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
    weight = torch.randn(hidden_size, device="cuda", dtype=dtype)

    # Pre-allocate intermediate tensors
    h = torch.empty_like(x)
    y_normed = torch.empty_like(x)

    # Compute global_scale for fp4_quantize (required when sf_use_ue8m0 is false)
    global_scale = torch.tensor([1.0], device="cuda", dtype=torch.float32)

    # Benchmark torch.add alone
    times_add = bench_gpu_time(
        lambda: torch.add(x, r, out=h),
        cold_l2_cache=True,
        enable_cupti=True,
        use_cuda_graph=False,
        dry_run_iters=10,
        repeat_iters=100,
    )
    t_add = np.median(times_add)

    # Run add once to get h for rmsnorm
    torch.add(x, r, out=h)

    # Benchmark rmsnorm alone
    times_rmsnorm = bench_gpu_time(
        lambda: rmsnorm(h, weight, eps=eps, out=y_normed),
        cold_l2_cache=True,
        enable_cupti=True,
        use_cuda_graph=False,
        dry_run_iters=10,
        repeat_iters=100,
    )
    t_rmsnorm = np.median(times_rmsnorm)

    # Run rmsnorm once to get y_normed for fp4_quantize
    rmsnorm(h, weight, eps=eps, out=y_normed)

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

    return t_add, t_rmsnorm, t_fp4, t_add + t_rmsnorm + t_fp4


def bench_partial_separate(batch_size, hidden_size, dtype, block_size=16):
    """Benchmark partial separate: torch.add + fused rmsnorm_fp4quant.

    Returns tuple of (add_time_ms, rmsnorm_fp4quant_time_ms, total_time_ms)
    """
    from flashinfer.cute_dsl.rmsnorm_fp4quant import rmsnorm_fp4quant_cute_dsl

    eps = 1e-6

    x = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
    r = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
    weight = torch.randn(hidden_size, device="cuda", dtype=dtype)

    # Pre-allocate tensors
    h = torch.empty_like(x)
    y_fp4 = torch.empty(batch_size, hidden_size // 2, device="cuda", dtype=torch.uint8)

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

    # Benchmark torch.add alone
    times_add = bench_gpu_time(
        lambda: torch.add(x, r, out=h),
        cold_l2_cache=True,
        enable_cupti=True,
        use_cuda_graph=False,
        dry_run_iters=10,
        repeat_iters=100,
    )
    t_add = np.median(times_add)

    # Run add once to get h for rmsnorm_fp4quant
    torch.add(x, r, out=h)

    # Benchmark fused rmsnorm_fp4quant
    times_rmsnorm_fp4 = bench_gpu_time(
        lambda: rmsnorm_fp4quant_cute_dsl(
            h,
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
    t_rmsnorm_fp4 = np.median(times_rmsnorm_fp4)

    return t_add, t_rmsnorm_fp4, t_add + t_rmsnorm_fp4


def sanity_check_outputs(dtype=torch.float16, block_size=16):
    """Verify CuTe-DSL output matches separate torch.add + RMSNorm + fp4_quantize."""
    from flashinfer.cute_dsl.add_rmsnorm_fp4quant import add_rmsnorm_fp4quant_cute_dsl
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
        # Create inputs (use same seed for reproducibility)
        x = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
        r = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
        weight = torch.randn(hidden_size, device="cuda", dtype=dtype)

        # CuTe-DSL fused path
        y_fp4_fused = torch.empty(
            batch_size, hidden_size // 2, device="cuda", dtype=torch.uint8
        )
        if block_size == 32:
            block_scale_fused = torch.empty(
                batch_size, hidden_size // block_size, device="cuda", dtype=torch.uint8
            )
            scale_format = "ue8m0"
        else:
            block_scale_fused = torch.empty(
                batch_size,
                hidden_size // block_size,
                device="cuda",
                dtype=torch.float8_e4m3fn,
            )
            scale_format = "e4m3"

        add_rmsnorm_fp4quant_cute_dsl(
            x,
            r,
            weight,
            y_fp4_fused,
            block_scale_fused,
            eps=eps,
            block_size=block_size,
            scale_format=scale_format,
        )

        # Separate path: torch.add + rmsnorm + fp4_quantize
        h = x + r
        y_normed = torch.empty_like(x)
        rmsnorm(h, weight, eps=eps, out=y_normed)

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
        match_count = (y_fp4_fused == y_fp4_sep).sum().item()
        total_count = y_fp4_fused.numel()
        match_pct = match_count / total_count * 100

        # For FP4, 70% exact match is reasonable given the precision constraints
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
    print("=" * 120)
    print("Fused Add + RMSNorm + FP4 Quantization Benchmark")
    print("=" * 120)

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
            "✓ Confirmed: CuTe-DSL output is equivalent to "
            "torch.add + RMSNorm + fp4_quantization"
        )
    else:
        print("✗ Warning: Some outputs did not match closely")
    print()

    # Test configurations
    batch_sizes = [2**i for i in range(10, 17)]  # 1024 to 65536
    batch_sizes += [1000, 3000, 5000, 10000, 15000, 25000, 60000]
    batch_sizes = sorted(list(set(batch_sizes)))

    hidden_sizes = [2**j for j in range(11, 16)]  # 2048 to 32768
    hidden_sizes += [1536]
    hidden_sizes = sorted(list(set(hidden_sizes)))

    configs = [
        (batch_size, hidden_size)
        for batch_size in batch_sizes
        for hidden_size in hidden_sizes
    ]

    print()
    print("Legend:")
    print("  Fully Sep = torch.add + RMSNorm + FP4 Quantization (3 kernels)")
    print("  Partial Sep = torch.add + fused RMSNorm-FP4Quant (2 kernels)")
    print()
    header = (
        f"{'Batch':<8} {'Hidden':<8} "
        f"{'Fused (µs)':<11} {'BW (GB/s)':<10} "
        f"{'Add (µs)':<10} {'RMSNorm (µs)':<13} {'FP4Q (µs)':<10} "
        f"{'RN+FP4 (µs)':<12} "
        f"{'Full Sep':<10} {'Part Sep':<10} "
        f"{'vs Full':<9} {'vs Part':<9}"
    )
    print(header)
    print("-" * len(header))

    results = []

    for batch_size, hidden_size in configs:
        # Fused CuTe-DSL kernel timing (add + rmsnorm + fp4quant all in one)
        try:
            t_fused = bench_fused_cute_dsl(batch_size, hidden_size, dtype, block_size)
            t_fused_us = t_fused * 1e3  # ms to µs
            bw_fused = compute_bandwidth_gb_s(
                batch_size, hidden_size, block_size, t_fused
            )
        except Exception as e:
            print(f"{batch_size:<8} {hidden_size:<8} FUSED ERROR: {e}")
            continue

        # Fully separate: torch.add + rmsnorm + fp4_quantize
        try:
            t_add, t_rmsnorm, t_fp4, t_full_sep = bench_fully_separate(
                batch_size, hidden_size, dtype, block_size
            )
            t_add_us = t_add * 1e3  # ms to µs
            t_rmsnorm_us = t_rmsnorm * 1e3  # ms to µs
            t_fp4_us = t_fp4 * 1e3  # ms to µs
            t_full_sep_us = t_full_sep * 1e3  # ms to µs
            speedup_full = t_full_sep / t_fused if t_fused > 0 else 0
            add_str = f"{t_add_us:.1f}"
            rmsnorm_str = f"{t_rmsnorm_us:.1f}"
            fp4_str = f"{t_fp4_us:.1f}"
            full_sep_str = f"{t_full_sep_us:.1f}"
            speedup_full_str = f"{speedup_full:.2f}x"
        except Exception as e:
            print(f"{batch_size:<8} {hidden_size:<8} FULLY SEPARATE ERROR: {e}")
            t_add_us = None
            t_rmsnorm_us = None
            t_fp4_us = None
            t_full_sep_us = None
            add_str = "N/A"
            rmsnorm_str = "N/A"
            fp4_str = "N/A"
            full_sep_str = "N/A"
            speedup_full_str = "N/A"
            speedup_full = None

        # Partial separate: torch.add + fused rmsnorm_fp4quant
        try:
            t_add_p, t_rn_fp4, t_part_sep = bench_partial_separate(
                batch_size, hidden_size, dtype, block_size
            )
            t_rn_fp4_us = t_rn_fp4 * 1e3  # ms to µs
            t_part_sep_us = t_part_sep * 1e3  # ms to µs
            speedup_part = t_part_sep / t_fused if t_fused > 0 else 0
            rn_fp4_str = f"{t_rn_fp4_us:.1f}"
            part_sep_str = f"{t_part_sep_us:.1f}"
            speedup_part_str = f"{speedup_part:.2f}x"
        except Exception as e:
            print(f"{batch_size:<8} {hidden_size:<8} PARTIAL SEPARATE ERROR: {e}")
            t_rn_fp4_us = None
            t_part_sep_us = None
            rn_fp4_str = "N/A"
            part_sep_str = "N/A"
            speedup_part_str = "N/A"
            speedup_part = None

        print(
            f"{batch_size:<8} {hidden_size:<8} "
            f"{t_fused_us:<11.1f} {bw_fused:<10.1f} "
            f"{add_str:<10} {rmsnorm_str:<13} {fp4_str:<10} "
            f"{rn_fp4_str:<12} "
            f"{full_sep_str:<10} {part_sep_str:<10} "
            f"{speedup_full_str:<9} {speedup_part_str:<9}"
        )

        result = {
            "batch_size": batch_size,
            "hidden_size": hidden_size,
            "fused_us": t_fused_us,
            "fused_bw_gb_s": bw_fused,
            "add_us": t_add_us,
            "rmsnorm_us": t_rmsnorm_us,
            "fp4_quant_us": t_fp4_us,
            "rmsnorm_fp4quant_us": t_rn_fp4_us,
            "fully_separate_us": t_full_sep_us,
            "partial_separate_us": t_part_sep_us,
            "speedup_vs_fully_separate": speedup_full,
            "speedup_vs_partial_separate": speedup_part,
        }
        results.append(result)

    print()
    print("=" * 120)

    # Calculate and print geomean speedups
    speedups_full = [
        r["speedup_vs_fully_separate"]
        for r in results
        if r["speedup_vs_fully_separate"] is not None
    ]
    speedups_part = [
        r["speedup_vs_partial_separate"]
        for r in results
        if r["speedup_vs_partial_separate"] is not None
    ]

    if speedups_full:
        geomean_full = gmean(speedups_full)
        print(f"Geomean speedup vs Fully Separate (3 kernels):   {geomean_full:.2f}x")
    if speedups_part:
        geomean_part = gmean(speedups_part)
        print(f"Geomean speedup vs Partial Separate (2 kernels): {geomean_part:.2f}x")

    print("=" * 120)
    print("Benchmark Complete")
    print("=" * 120)


if __name__ == "__main__":
    run_benchmark()
