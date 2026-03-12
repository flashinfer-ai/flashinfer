"""
Benchmark for flashinfer.fused_rmsnorm_silu() — fused RMSNorm + SiLU activation.

Measures kernel execution time across all WAN VAE problem sizes using
flashinfer's bench_gpu_time utility with CUPTI for accurate GPU timing.

Usage:
    python benchmarks/bench_fused_rmsnorm_silu.py
"""

import numpy as np
import torch

import flashinfer
from flashinfer.testing.utils import bench_gpu_time

C_VALUES = [64, 128, 160, 256, 320, 512, 640, 1024]
TOKEN_VALUES = [1560, 6240, 24960, 99840, 399360]


def main():
    device = torch.cuda.current_device()
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"SM: {torch.cuda.get_device_capability(device)}")
    print(f"Timing: CUPTI, dry_run=10, repeat=30")
    print()
    print(f"{'C':>6}  {'tokens':>8}  {'elements':>12}  {'median_ms':>10}  {'min_ms':>10}  {'GB/s':>8}")
    print("-" * 65)

    for C in C_VALUES:
        for num_tokens in TOKEN_VALUES:
            x = torch.randn(num_tokens, C, dtype=torch.bfloat16, device="cuda") * 5.0 + 5.0
            w = torch.rand(C, dtype=torch.bfloat16, device="cuda") * 1.5 + 0.5
            out = torch.empty_like(x)

            # Warmup (triggers JIT compile on first call)
            flashinfer.fused_rmsnorm_silu(x, w, 1e-6, out=out)
            torch.cuda.synchronize()

            times = bench_gpu_time(
                fn=lambda: flashinfer.fused_rmsnorm_silu(x, w, 1e-6, out=out),
                enable_cupti=True,
                dry_run_iters=10,
                repeat_iters=30,
            )

            median_ms = np.median(times)
            min_ms = np.min(times)
            bytes_transferred = num_tokens * C * 2 * 2 + C * 2
            gb_per_s = bytes_transferred / (median_ms * 1e-3) / 1e9

            print(f"{C:>6}  {num_tokens:>8}  {num_tokens * C:>12}  {median_ms:>10.4f}  {min_ms:>10.4f}  {gb_per_s:>8.1f}")

    print()


if __name__ == "__main__":
    main()
