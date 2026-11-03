"""
Benchmark silu_and_mul, gelu_and_mul and gelu_tanh_and_mul.

Each call is timed inside a CUDA graph so launch overhead does not hide kernel
time. By default the inputs rotate across enough buffers that every call reads
from HBM; pass --hot to reuse one L2-resident buffer instead.

Usage:
    python benchmarks/bench_act_and_mul.py
    python benchmarks/bench_act_and_mul.py --acts silu --num-tokens 1 16 --hidden-sizes 14336 --hot
"""

import argparse

import numpy as np
import torch

import flashinfer
from flashinfer.testing.utils import bench_gpu_time


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--acts",
        nargs="+",
        choices=["silu", "gelu", "gelu_tanh"],
        default=["silu", "gelu", "gelu_tanh"],
    )
    parser.add_argument(
        "--num-tokens",
        nargs="+",
        type=int,
        default=[1, 16, 64, 128, 256, 1024, 4096, 16384],
    )
    # hidden size is the output width d; the input is (num_tokens, 2 * d).
    parser.add_argument(
        "--hidden-sizes", nargs="+", type=int, default=[4096, 14336, 28672]
    )
    parser.add_argument(
        "--dtypes", nargs="+", choices=["float16", "bfloat16"], default=["bfloat16"]
    )
    parser.add_argument(
        "--hot", action="store_true", help="Keep the working set in L2."
    )
    args = parser.parse_args()

    for act in args.acts:
        fn = getattr(flashinfer, f"{act}_and_mul")
        for dtype_str in args.dtypes:
            dtype = getattr(torch, dtype_str)
            for hidden_size in args.hidden_sizes:
                for num_tokens in args.num_tokens:
                    x = torch.randn(
                        (num_tokens, 2 * hidden_size), dtype=dtype, device="cuda"
                    )
                    out = torch.empty(
                        (num_tokens, hidden_size), dtype=dtype, device="cuda"
                    )
                    measurements = bench_gpu_time(
                        fn,
                        input_args=(x, out),
                        use_cuda_graph=True,
                        num_iters_within_graph=100,
                        cold_l2_cache=not args.hot,
                    )
                    latency_ms = np.median(measurements)
                    num_bytes = (x.numel() + out.numel()) * x.element_size()
                    print(
                        f"act: {act:9},",
                        f"dtype: {dtype_str:8},",
                        f"num_tokens: {num_tokens:5},",
                        f"hidden_size: {hidden_size:5},",
                        f"latency: {latency_ms * 1e3:8.2f}us,",
                        f"throughput: {num_bytes / (latency_ms * 1e-3) * 1e-9:7.1f}GB/s",
                    )
        print("---")


if __name__ == "__main__":
    main()
