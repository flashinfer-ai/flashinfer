"""Benchmark the Triton RMSNorm single-load fast path against the general
two-pass kernel.

`flashinfer.triton.norm.rms_norm` dispatches the plain (no-residual, no-scale,
contiguous, hidden <= 8192) path to `rms_norm_single_pass_kernel`, which loads
each row once and reuses it for both the sum-of-squares reduction and the
normalize step (1 read + 1 write) instead of the general kernel's 2 reads + 1
write. This script times both over a sweep of hidden sizes and token counts and
reports effective memory bandwidth so the speedup is reproducible.

    python benchmarks/bench_triton_rmsnorm.py
    python benchmarks/bench_triton_rmsnorm.py --hidden-sizes 4096 8192 --dtypes bfloat16
"""

import argparse

import numpy as np
import torch
import triton

from flashinfer.testing.utils import bench_gpu_time
from flashinfer.triton.kernels.norm import rms_norm_kernel
from flashinfer.triton.norm import rms_norm


def _general_two_pass(x, weight, out, eps):
    """Invoke the general two-pass kernel directly (the pre-fast-path path)."""
    b, n = x.shape
    block_size = triton.next_power_of_2(n)
    num_warps = max(8, min(32, block_size // 256))
    rms_norm_kernel[(b,)](
        n=n,
        b=b,
        x_ptr=x,
        x_stride=x.stride(0),
        x_scale_ptr=None,
        r_ptr=None,
        r_stride=0,
        w_ptr=weight,
        o_ptr=out,
        o_stride=out.stride(0),
        o_scale_ptr=None,
        EPS=eps,
        BLOCK_SIZE=block_size,
        HAS_IN_SCALE=False,
        HAS_OUT_SCALE=False,
        HAS_OUTPUT=True,
        HAS_RESIDUAL=False,
        num_warps=num_warps,
    )


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--token-counts", nargs="+", type=int, default=[1, 128, 2048, 8192]
    )
    parser.add_argument(
        "--hidden-sizes",
        nargs="+",
        type=int,
        default=[1024, 2048, 3072, 4096, 5120, 7168, 8192],
    )
    parser.add_argument(
        "--dtypes",
        nargs="+",
        choices=["float16", "bfloat16"],
        default=["float16", "bfloat16"],
    )
    parser.add_argument(
        "--repeat", type=int, default=5, help="outer timing runs, aggregated by mean"
    )
    parser.add_argument(
        "--iters", type=int, default=100, help="measured iterations per run"
    )
    args = parser.parse_args()

    def median_ms(fn):
        # Fixed iteration count so the protocol is deterministic ("repeat R x
        # iters I") rather than the adaptive default; median over the run.
        return np.median(bench_gpu_time(fn, repeat_iters=args.iters))

    eps = 1e-6
    all_speedups = []
    for dtype_str in args.dtypes:
        dtype = getattr(torch, dtype_str)
        for hidden_size in args.hidden_sizes:
            for tokens in args.token_counts:
                x = torch.randn((tokens, hidden_size), dtype=dtype, device="cuda")
                weight = torch.randn(hidden_size, dtype=dtype, device="cuda")
                out = torch.empty_like(x)

                # Correctness: the fast path is bit-identical to the general one.
                out_fast = torch.empty_like(x)
                rms_norm(x, weight, out_fast, eps)
                out_general = torch.empty_like(x)
                _general_two_pass(x, weight, out_general, eps)
                assert torch.equal(out_fast, out_general), "fast != general"

                row_bytes = tokens * hidden_size * x.element_size()
                w_bytes = hidden_size * x.element_size()

                # repeat outer runs, then average (mirrors the reported table).
                before_runs = [
                    median_ms(lambda: _general_two_pass(x, weight, out, eps))
                    for _ in range(args.repeat)
                ]
                after_runs = [
                    median_ms(lambda: rms_norm(x, weight, out, eps))
                    for _ in range(args.repeat)
                ]
                before_ms = float(np.mean(before_runs))
                after_ms = float(np.mean(after_runs))
                # AFTER moves 2 units (1 read + 1 write); BEFORE moves 3.
                after_bw = (2 * row_bytes + w_bytes) / (after_ms * 1e-3)
                speedup = before_ms / after_ms if after_ms else float("nan")
                all_speedups.append(speedup)
                print(
                    f"tokens: {tokens:5}, hidden: {hidden_size:5}, "
                    f"dtype: {dtype_str:8}, before: {before_ms * 1e3:6.1f}us, "
                    f"after: {after_ms * 1e3:6.1f}us, speedup: {speedup:5.3f}x, "
                    f"after_bw: {after_bw * 1e-9:7.1f}GB/s"
                )
        print("---")

    if all_speedups:
        geomean = float(np.exp(np.mean(np.log(all_speedups))))
        print(
            f"speedup geomean: {geomean:.3f}x "
            f"(range {min(all_speedups):.3f}-{max(all_speedups):.3f}x) "
            f"over {len(all_speedups)} shapes, repeat {args.repeat} x {args.iters} iters"
        )


if __name__ == "__main__":
    main()
