"""
Benchmark for flashinfer.fused_rmsnorm_silu() — fused RMSNorm + SiLU activation.

Measures kernel execution time across all WAN VAE problem sizes using
flashinfer's bench_gpu_time utility with CUPTI for accurate GPU timing.

Usage:
    python benchmarks/bench_fused_rmsnorm_silu.py
    python benchmarks/bench_fused_rmsnorm_silu.py --enable-cupti
    python benchmarks/bench_fused_rmsnorm_silu.py --use-cuda-graph
"""

import argparse

import numpy as np
import torch

import flashinfer
from flashinfer.testing.utils import bench_gpu_time

C_VALUES = [64, 128, 160, 256, 320, 512, 640, 1024]
TOKEN_VALUES = [1560, 6240, 24960, 99840, 399360]


def main():
    parser = argparse.ArgumentParser(description="Benchmark fused_rmsnorm_silu")
    parser.add_argument("--enable-cupti", action="store_true", help="Use CUPTI for accurate GPU timing")
    parser.add_argument("--use-cuda-graph", action="store_true", help="Use CUDA graph for reduced launch overhead")
    parser.add_argument("--C", type=int, nargs="+", default=None, help="Hidden dims to test (default: all)")
    parser.add_argument("--tokens", type=int, nargs="+", default=None, help="Token counts to test (default: all)")
    args = parser.parse_args()

    c_values = args.C or C_VALUES
    token_values = args.tokens or TOKEN_VALUES

    device = torch.cuda.current_device()
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"SM: {torch.cuda.get_device_capability(device)}")
    print(f"Timing: {'CUPTI' if args.enable_cupti else 'CUDA graph' if args.use_cuda_graph else 'CUDA events'}")
    print()
    print(f"{'C':>6}  {'tokens':>8}  {'elements':>12}  {'median_ms':>10}  {'min_ms':>10}  {'GB/s':>8}")
    print("-" * 65)

    for C in c_values:
        for num_tokens in token_values:
            x = torch.randn(num_tokens, C, dtype=torch.bfloat16, device="cuda") * 5.0 + 5.0
            w = torch.rand(C, dtype=torch.bfloat16, device="cuda") * 1.5 + 0.5
            out = torch.empty_like(x)

            # Warmup (triggers NVRTC compile on first call)
            flashinfer.fused_rmsnorm_silu(x, w, 1e-6, out=out)
            torch.cuda.synchronize()

            times = bench_gpu_time(
                fn=lambda: flashinfer.fused_rmsnorm_silu(x, w, 1e-6, out=out),
                enable_cupti=args.enable_cupti,
                use_cuda_graph=args.use_cuda_graph,
            )

            median_ms = np.median(times)
            min_ms = np.min(times)
            # Bandwidth: read input (bf16) + weight (bf16, broadcast) + write output (bf16)
            bytes_transferred = num_tokens * C * 2 * 2 + C * 2  # read + write + weight
            gb_per_s = bytes_transferred / (median_ms * 1e-3) / 1e9

            print(f"{C:>6}  {num_tokens:>8}  {num_tokens * C:>12}  {median_ms:>10.4f}  {min_ms:>10.4f}  {gb_per_s:>8.1f}")

    print()


if __name__ == "__main__":
    main()
