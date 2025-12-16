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

Compares the CuTe-DSL backend against the cuDNN backend for fused
RMSNorm + FP4 quantization.

Usage:
    python bench_cute_dsl_rmsnorm_fp4quant.py

Requirements:
    - Blackwell GPU (SM100+)
    - CuTe-DSL installed
    - cuDNN >= 9.18.0 (for comparison)
"""

import json
import numpy as np
import torch
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
    from flashinfer.cute_dsl.rmsnorm_fp4quant import rmsnorm_fp4quant_cute_dsl

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
        lambda: rmsnorm_fp4quant_cute_dsl(
            x,
            weight,
            y_fp4,
            block_scale,
            eps=eps,
            block_size=block_size,
            scale_format=scale_format,
        ),
        l2_flush=True,
        enable_cupti=True,
        use_cuda_graph=False,
    )

    # Return median time
    return np.median(times)


def bench_cudnn(batch_size, hidden_size, dtype, block_size=16):
    """Benchmark cuDNN backend."""
    from flashinfer.norm import rmsnorm_fp4quant, CUDNN_AVAILABLE

    if not CUDNN_AVAILABLE:
        return None

    try:
        import cudnn

        if cudnn.backend_version() < 91800:
            return None
    except Exception:
        return None

    eps = 1e-6

    x = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
    weight = torch.randn(hidden_size, device="cuda", dtype=dtype)
    y_fp4 = torch.empty(batch_size, hidden_size // 2, device="cuda", dtype=torch.uint8)
    block_scale = torch.empty(
        batch_size, hidden_size // block_size, device="cuda", dtype=torch.float8_e4m3fn
    )

    # Benchmark with bench_gpu_time
    times = bench_gpu_time(
        lambda: rmsnorm_fp4quant(
            x, weight, y_fp4, block_scale, eps=eps, block_size=block_size
        ),
        l2_flush=True,
        enable_cupti=True,
        use_cuda_graph=False,
    )

    # Return median time
    return np.median(times)


def run_benchmark():
    """Run full benchmark suite."""
    print("=" * 80)
    print("Fused RMSNorm + FP4 Quantization Benchmark")
    print("=" * 80)

    cc = get_cc()
    print(f"GPU Compute Capability: SM{cc}")

    if cc < 100:
        print("ERROR: Blackwell GPU (SM100+) required for FP4 quantization")
        return

    dtype = torch.float16
    block_size = 16

    # Test configurations
    # sweep batch_size from 512 to 65536 (powers of 2)
    # and hidden_size from 2048 to 32768 (powers of 2)
    # Add some non-powers-of-2 batch sizes for more realism/variance.
    batch_sizes = [2**i for i in range(9, 17)]  # 512 to 65536
    batch_sizes += [1000, 3000, 5000, 10000, 15000, 25000, 60000]  # non-powers-of-2
    batch_sizes = sorted(list(set(batch_sizes)))  # deduplicate and sort

    hidden_sizes = [2**j for j in range(11, 16)]  # 2048 to 32768

    configs = [
        (batch_size, hidden_size)
        for batch_size in batch_sizes
        for hidden_size in hidden_sizes
    ]

    print()
    header = (
        f"{'Batch':<8} {'Hidden':<8} "
        f"{'CuTe-DSL (µs)':<14} {'BW (GB/s)':<12} "
        f"{'cuDNN (µs)':<12} {'BW (GB/s)':<12} "
        f"{'Rel Perf':<10}"
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

        # cuDNN timing
        try:
            t_cudnn = bench_cudnn(batch_size, hidden_size, dtype, block_size)
            if t_cudnn is not None:
                t_cudnn_us = t_cudnn * 1e3  # ms to µs
                bw_cudnn = compute_bandwidth_gb_s(
                    batch_size, hidden_size, block_size, t_cudnn
                )
                speedup = t_cudnn / t_cute if t_cute > 0 else 0
                speedup_str = f"{speedup:.2f}x"
                cudnn_str = f"{t_cudnn_us:.1f}"
                bw_cudnn_str = f"{bw_cudnn:.1f}"
            else:
                t_cudnn_us = None
                bw_cudnn = None
                speedup_str = "N/A"
                cudnn_str = "N/A"
                bw_cudnn_str = "N/A"
        except Exception:
            t_cudnn_us = None
            bw_cudnn = None
            speedup_str = "N/A"
            cudnn_str = "N/A"
            bw_cudnn_str = "N/A"

        print(
            f"{batch_size:<8} {hidden_size:<8} "
            f"{t_cute_us:<14.1f} {bw_cute:<12.1f} "
            f"{cudnn_str:<12} {bw_cudnn_str:<12} "
            f"{speedup_str:<10}"
        )

        result = {
            "batch_size": batch_size,
            "hidden_size": hidden_size,
            "cute_dsl_us": t_cute_us,
            "cute_dsl_bw_gb_s": bw_cute,
            "cudnn_us": t_cudnn_us,
            "cudnn_bw_gb_s": bw_cudnn,
            "speedup": speedup if t_cudnn_us else None,
        }
        results.append(result)

    print()
    print("=" * 80)
    print("Benchmark Complete")
    print("=" * 80)

    # Output JSON summary
    summary = {
        "gpu_cc": cc,
        "dtype": str(dtype),
        "block_size": block_size,
        "results": results,
    }
    print(f"MAIN_OUTPUT={json.dumps(summary)}")


if __name__ == "__main__":
    run_benchmark()
