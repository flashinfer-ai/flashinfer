"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Benchmark: Fused SwiGLU + NVFP4 Quantization

Compares the fused CuTe-DSL kernel with the equivalent unfused CuTe-DSL path
(silu_and_mul followed by fp4_quantize).

Usage:
    python bench_silu_and_mul_nvfp4_quantize.py

Requirements:
    - Blackwell GPU (SM100+)
"""

import numpy as np
import torch

from flashinfer import fp4_quantize, silu_and_mul, silu_and_mul_nvfp4_quantize
from flashinfer.testing.utils import bench_gpu_time

FLOAT4_E2M1_MAX = 6.0
FLOAT8_E4M3_MAX = float(torch.finfo(torch.float8_e4m3fn).max)
SF_VEC_SIZE = 16


def _gmean(values):
    """Geometric mean of a non-empty sequence of positive values."""
    return float(np.exp(np.mean(np.log(values))))


def get_cc():
    """Get CUDA compute capability as a single integer, for example 100 for SM100."""
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor


def compute_fused_bandwidth_gb_s(m: int, k: int, dtype: torch.dtype, time_ms: float):
    """Achieved memory bandwidth in GB/s for the fused SwiGLU + NVFP4 kernel."""
    elem_bytes = torch.finfo(dtype).bits // 8
    read_bytes = m * (2 * k) * elem_bytes
    write_bytes = m * (k // 2) + m * (k // SF_VEC_SIZE)
    total_bytes = read_bytes + write_bytes
    time_s = time_ms / 1000.0
    if time_s <= 0:
        return 0.0
    return total_bytes / time_s / 1e9


def _global_scale(amax: float = 3.0):
    # Use a fixed calibration scale for timing.
    return torch.tensor(
        [FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / amax],
        device="cuda",
        dtype=torch.float32,
    )


def bench_fused(m, k, dtype, is_swizzled, global_scale):
    """Return median fused-kernel latency in milliseconds."""
    x = torch.randn(m, 2 * k, device="cuda", dtype=dtype)
    times = bench_gpu_time(
        lambda: silu_and_mul_nvfp4_quantize(x, global_scale, SF_VEC_SIZE, is_swizzled),
        cold_l2_cache=True,
        enable_cupti=True,
        use_cuda_graph=False,
        dry_run_iters=10,
        repeat_iters=100,
    )
    return np.median(times)


def bench_unfused(m, k, dtype, is_swizzled, global_scale):
    """Return median unfused-path latency in milliseconds."""
    x = torch.randn(m, 2 * k, device="cuda", dtype=dtype)

    def unfused_operation():
        y = silu_and_mul(x)
        fp4_quantize(
            y,
            global_scale=global_scale,
            sf_vec_size=SF_VEC_SIZE,
            sf_use_ue8m0=False,
            is_sf_swizzled_layout=is_swizzled,
            backend="cute-dsl",
        )

    times = bench_gpu_time(
        unfused_operation,
        cold_l2_cache=True,
        enable_cupti=True,
        use_cuda_graph=False,
        dry_run_iters=10,
        repeat_iters=100,
    )
    return np.median(times)


def sanity_check_outputs(dtype, global_scale, is_swizzled):
    """Verify the fused kernel matches unfused silu_and_mul + fp4_quantize."""
    test_configs = [(128, 512), (1024, 2048), (2048, 8192)]
    all_passed = True
    for m, k in test_configs:
        x = torch.randn(m, 2 * k, device="cuda", dtype=dtype)

        out_fused, _ = silu_and_mul_nvfp4_quantize(
            x, global_scale, SF_VEC_SIZE, is_swizzled
        )
        y = silu_and_mul(x)
        out_ref, _ = fp4_quantize(
            y,
            global_scale=global_scale,
            sf_vec_size=SF_VEC_SIZE,
            sf_use_ue8m0=False,
            is_sf_swizzled_layout=is_swizzled,
            backend="cute-dsl",
        )

        # Allow small numerical differences from the unfused path. Unit tests provide
        # more comprehensive correctness coverage.
        match = (out_fused.view(torch.uint8) == out_ref.view(torch.uint8)).sum().item()
        pct = match / out_fused.numel() * 100
        if pct < 99.0:
            all_passed = False
            print(f"  WARN: ({m}, {k}) - FP4 match {pct:.2f}% (expected >= 99%)")
        else:
            print(f"  OK:   ({m}, {k}) - FP4 match {pct:.2f}%")
    return all_passed


def run_benchmark():
    print("=" * 84)
    print("Fused SwiGLU + NVFP4 Quantization Benchmark")
    print("=" * 84)

    cc = get_cc()
    print(f"GPU Compute Capability: SM{cc}")
    if cc < 100:
        raise RuntimeError("Blackwell GPU (SM100+) required for NVFP4 quantization")

    dtype = torch.float16
    is_swizzled = True
    global_scale = _global_scale()

    print()
    print("Running sanity check...")
    if sanity_check_outputs(dtype, global_scale, is_swizzled):
        print("Confirmed: fused output matches silu_and_mul + fp4_quantize")
    else:
        print("Warning: some outputs did not match closely")
    print()

    batch_sizes = [2**i for i in range(10, 17)]  # 1024 to 65536
    batch_sizes += [1536, 4096 + 512]
    batch_sizes = sorted(set(batch_sizes))
    hidden_sizes = [2048, 4096, 8192, 16384]

    configs = [(m, k) for m in batch_sizes for k in hidden_sizes]

    header = (
        f"{'M':<8} {'K':<8} "
        f"{'Fused (us)':<12} {'BW (GB/s)':<12} "
        f"{'Unfused (us)':<14} {'Speedup':<10}"
    )
    print(header)
    print("-" * len(header))

    results = []
    for m, k in configs:
        try:
            t_fused = bench_fused(m, k, dtype, is_swizzled, global_scale)
            t_fused_us = t_fused * 1e3
            bw = compute_fused_bandwidth_gb_s(m, k, dtype, t_fused)
        except Exception as e:
            print(f"{m:<8} {k:<8} FUSED ERROR: {e}")
            continue

        try:
            t_unfused = bench_unfused(m, k, dtype, is_swizzled, global_scale)
            t_unfused_us = t_unfused * 1e3
            speedup = t_unfused / t_fused if t_fused > 0 else 0.0
            unfused_str = f"{t_unfused_us:.1f}"
            speedup_str = f"{speedup:.2f}x"
        except Exception:
            t_unfused_us = None
            unfused_str = "N/A"
            speedup_str = "N/A"
            speedup = None

        print(
            f"{m:<8} {k:<8} "
            f"{t_fused_us:<12.1f} {bw:<12.1f} "
            f"{unfused_str:<14} {speedup_str:<10}"
        )
        results.append(
            {
                "m": m,
                "k": k,
                "speedup": speedup,
            }
        )

    print()
    print("=" * 84)
    speedups = [r["speedup"] for r in results if r["speedup"] is not None]
    if speedups:
        print(
            f"Geomean speedup vs unfused (silu_and_mul + fp4_quantize): "
            f"{_gmean(speedups):.2f}x"
        )
    print("=" * 84)
    print("Benchmark Complete")
    print("=" * 84)


if __name__ == "__main__":
    run_benchmark()
