"""
Benchmark: fused epilogues of the masked grouped GEMM (``grouped_gemm_nt_masked``)

Times MoE GEMM1 on SM100/SM103 with NVFP4 inputs, producing either the BF16
activation or the NVFP4 GEMM2 input:

  gemm          GEMM1 with a full-width BF16 ``(m, 2I, l)`` output
  gemm+act      gemm + ``silu_and_mul_scaled_nvfp4_experts_quantize``
  swiglu        GEMM1 with the fused SwiGLU epilogue, BF16 ``(m, I, l)`` output
  swiglu+quant  swiglu + ``scaled_fp4_grouped_quantize``

Each expert has ``capacity_factor * tokens`` rows, ``rows_per_token * tokens``
of them valid. Timing uses CUDA graphs with a warm L2: the weights exceed L2
anyway, and the activation kernels read what the GEMM has just written, as
they would in a MoE layer.

Usage:
    python benchmarks/bench_grouped_gemm_masked_fused_epilogue.py
    python benchmarks/bench_grouped_gemm_masked_fused_epilogue.py --tokens 128 512 --tile-n 128
"""

import argparse

import numpy as np
import torch

from flashinfer import (
    scaled_fp4_grouped_quantize,
    silu_and_mul_scaled_nvfp4_experts_quantize,
)
from flashinfer.gemm import grouped_gemm_nt_masked
from flashinfer.testing.utils import bench_gpu_time
from flashinfer.tllm_enums import ActivationType

SF_VEC = 16


def nvfp4_operand(l: int, rows: int, k: int, device):
    """Random NVFP4 operand: packed ``(rows, k/2, l)`` and its scales ``(32, 4, rm, 4, rk, l)``."""
    packed = torch.randint(0, 256, (l, rows, k // 2), dtype=torch.uint8, device=device)
    sf = (
        torch.rand(l, rows // 128, k // SF_VEC // 4, 32, 4, 4, device=device) * 1.5
        + 0.25
    ).to(torch.float8_e4m3fn)
    return packed.permute(1, 2, 0), sf.permute(3, 4, 1, 5, 2, 0)


def median_us(fn) -> float:
    fn()
    torch.cuda.synchronize()
    times = bench_gpu_time(
        fn, use_cuda_graph=True, num_iters_within_graph=20, cold_l2_cache=False
    )
    return float(np.median(times)) * 1e3


def bench(args, tokens: int, b, sfb) -> dict:
    device = torch.device("cuda")
    l, inter = args.num_experts, args.intermediate
    m = (args.capacity_factor * tokens + 127) // 128 * 128
    a, sfa = nvfp4_operand(l, m, args.hidden, device)
    masked_m = torch.full(
        (l,), int(tokens * args.rows_per_token), dtype=torch.int32, device=device
    )
    alpha = torch.full((l,), 1e-3, device=device)
    global_scale = torch.full((l,), 100.0, device=device)
    full = torch.empty(l, m, 2 * inter, dtype=torch.bfloat16, device=device)
    half = torch.empty(l, m, inter, dtype=torch.bfloat16, device=device)

    def gemm(out, **kwargs):
        grouped_gemm_nt_masked(
            (a, sfa),
            (b, sfb),
            out.permute(1, 2, 0),
            masked_m,
            ab_dtype="float4_e2m1fn",
            sf_dtype="float8_e4m3fn",
            c_dtype="bfloat16",
            sf_vec_size=SF_VEC,
            mma_tiler_mn=(128, args.tile_n),
            alpha=alpha,
            alpha_dtype="float32",
            **kwargs,
        )

    def swiglu():
        gemm(half, activation_type=ActivationType.Swiglu)

    def gemm_act():
        gemm(full)
        silu_and_mul_scaled_nvfp4_experts_quantize(
            full, masked_m, global_scale, nvfp4_4over6=None
        )

    def swiglu_quant():
        swiglu()
        scaled_fp4_grouped_quantize(half, masked_m, global_scale, nvfp4_4over6=None)

    return {
        "gemm": median_us(lambda: gemm(full)),
        "gemm+act": median_us(gemm_act),
        "swiglu": median_us(swiglu),
        "swiglu+quant": median_us(swiglu_quant),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--num-experts", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=6144)
    parser.add_argument("--intermediate", type=int, default=3072)
    parser.add_argument("--tokens", type=int, nargs="+", default=[32, 128, 256, 512])
    parser.add_argument("--capacity-factor", type=int, default=8)
    parser.add_argument("--rows-per-token", type=float, default=0.25)
    parser.add_argument("--tile-n", type=int, choices=[128, 256], default=256)
    args = parser.parse_args()

    major, minor = torch.cuda.get_device_capability()
    if (major, minor) not in ((10, 0), (10, 3)):
        raise SystemExit("The fused epilogues require SM100/SM103")

    torch.manual_seed(0)
    b, sfb = nvfp4_operand(
        args.num_experts, 2 * args.intermediate, args.hidden, torch.device("cuda")
    )
    columns = ["gemm", "gemm+act", "swiglu", "swiglu+quant"]
    print(
        "| tokens | rows/expert | "
        + " | ".join(f"{c} (us)" for c in columns)
        + " | swiglu+quant vs gemm+act |"
    )
    print("|" + "---:|" * (len(columns) + 3))
    for tokens in args.tokens:
        t = bench(args, tokens, b, sfb)
        speedup = (1 - t["swiglu+quant"] / t["gemm+act"]) * 100
        rows = int(tokens * args.rows_per_token)
        print(
            f"| {tokens} | {rows} | "
            + " | ".join(f"{t[c]:.1f}" for c in columns)
            + f" | {speedup:.1f}% |"
        )


if __name__ == "__main__":
    main()
