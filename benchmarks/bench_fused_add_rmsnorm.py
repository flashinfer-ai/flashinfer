import argparse
from typing import cast

import torch
from triton.testing import do_bench

import flashinfer

@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", nargs='+', type=int, default=[1, 19, 99, 989])
    parser.add_argument("--hidden-sizes", nargs='+', type=int, default=[111, 500, 1024, 3072, 4096, 8192])
    parser.add_argument("--dtypes", nargs='+', choices=["float16", "bfloat16"], default=["float16"])
    args = parser.parse_args()

    eps = 1e-6

    # Loop over each combination of batch_size, hidden_size, and dtype
    for batch_size in args.batch_sizes:
        for hidden_size in args.hidden_sizes:
            for dtype_str in args.dtypes:
                dtype = getattr(torch, dtype_str)

                # Define tensors with the correct dtype
                x = torch.randn((batch_size, hidden_size), dtype=dtype, device="cuda")
                residual = torch.randn_like(x)
                weight = torch.randn(hidden_size, dtype=dtype, device="cuda")

                @torch.cuda.nvtx.range(f"fused_add_rmsnorm batch_size={batch_size}, hidden_size={hidden_size}, dtype={dtype_str}")
                def fn() -> None:
                    flashinfer.fused_add_rmsnorm(x, residual, weight, eps)

                # Run benchmarking
                latency_ms = cast(float, do_bench(fn))
                throughput = (
                    (x.numel() * x.element_size() * 2
                     + residual.numel() * residual.element_size() * 2
                     + weight.numel() * weight.element_size())
                    / (latency_ms * 1e-3)
                )
                print(
                    f"batch_size: {batch_size:3},",
                    f"hidden_size: {hidden_size:5},",
                    f"dtype: {dtype_str:8},",
                    f"latency: {latency_ms*1e3:2.0f}us,",
                    f"throughput: {throughput*1e-9:7.3f}GB/s",
                )

        print("---")

    torch.cuda.profiler.stop()

if __name__ == "__main__":
    main()
