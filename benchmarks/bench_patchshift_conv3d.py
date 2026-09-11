"""
Copyright (c) 2026 by the PatchShift Conv3d contributors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""Benchmark PatchShift Conv3d against ``torch.nn.functional.conv3d``."""

import argparse
import statistics

import torch
import torch.nn.functional as F

import flashinfer
from flashinfer.testing import bench_gpu_time


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--d", type=int, default=4)
    parser.add_argument("--h", type=int, default=128)
    parser.add_argument("--w", type=int, default=120)
    parser.add_argument("--c", type=int, default=128)
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument("--dry-run-iters", type=int, default=10)
    parser.add_argument("--repeat-iters", type=int, default=50)
    parser.add_argument(
        "--use-cuda-events",
        action="store_true",
        help="disable preferred CUPTI timing",
    )
    parser.add_argument(
        "--cold-l2-cache",
        action="store_true",
        help="flush L2 between iterations (default models reused inference weights)",
    )
    parser.add_argument("--skip-refcheck", action="store_true")
    return parser.parse_args()


def _measure(fn, args: argparse.Namespace) -> tuple[float, float]:
    times = bench_gpu_time(
        fn=fn,
        dry_run_iters=args.dry_run_iters,
        repeat_iters=args.repeat_iters,
        enable_cupti=not args.use_cuda_events,
        cold_l2_cache=args.cold_l2_cache,
    )
    return statistics.median(times), statistics.pstdev(times)


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        raise RuntimeError("PatchShift Conv3d benchmark requires SM100a/B200")
    if args.c <= 0 or args.c % 8 != 0 or args.k <= 0:
        raise ValueError("C must be positive and divisible by 8; K must be positive")

    torch.manual_seed(0)
    input_ndhwc = (
        torch.randn(
            args.n,
            args.d,
            args.h,
            args.w,
            args.c,
            dtype=torch.bfloat16,
            device="cuda",
        )
        * 0.125
    )
    input_ncdhw = input_ndhwc.permute(0, 4, 1, 2, 3).contiguous()
    weight = (
        torch.randn(
            args.k,
            args.c,
            3,
            3,
            3,
            dtype=torch.bfloat16,
            device="cuda",
        )
        * 0.0625
    )
    packed_weight = flashinfer.pack_patchshift_conv3d_weight(weight)
    workspace = flashinfer.prepare_patchshift_conv3d(input_ndhwc, packed_weight, args.k)

    def run_patchshift():
        return flashinfer.patchshift_conv3d(
            input_ndhwc, packed_weight, workspace, args.k
        )

    def run_torch():
        return F.conv3d(input_ncdhw, weight, padding=1)

    if not args.skip_refcheck:
        actual = run_patchshift()
        expected = run_torch().permute(0, 2, 3, 4, 1)
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)

    patchshift_ms, patchshift_std = _measure(run_patchshift, args)
    torch_ms, torch_std = _measure(run_torch, args)
    flops = 2 * args.n * args.d * args.h * args.w * args.c * args.k * 27
    patchshift_tflops = flops / (patchshift_ms * 1e9)
    torch_tflops = flops / (torch_ms * 1e9)
    speedup = torch_ms / patchshift_ms

    print(
        "shape,backend,median_ms,std_ms,tflops,speedup_vs_torch\n"
        f"{args.n}x{args.d}x{args.h}x{args.w}x{args.c}x{args.k},"
        f"patchshift,{patchshift_ms:.6f},{patchshift_std:.6f},"
        f"{patchshift_tflops:.3f},{speedup:.3f}\n"
        f"{args.n}x{args.d}x{args.h}x{args.w}x{args.c}x{args.k},"
        f"torch,{torch_ms:.6f},{torch_std:.6f},{torch_tflops:.3f},1.000"
    )


if __name__ == "__main__":
    main()
