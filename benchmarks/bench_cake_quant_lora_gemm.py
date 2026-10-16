"""Benchmark ``mm_nvfp4_lora`` / ``mm_fp8_lora`` (Cake generated programs) with CUPTI GPU time.

Times the fused quantized-base + runtime BF16 LoRA GEMM on one problem and, for
reference, the unfused chain of the plain base GEMM (``mm_fp4`` / ``mm_fp8``) plus
an in-place ``addmm_`` LoRA correction.  Timing uses
:func:`flashinfer.testing.bench_gpu_time` with CUPTI activity tracing and a cold
L2 cache, so the numbers are GPU kernel time only.

Example::

    python benchmarks/bench_cake_quant_lora_gemm.py --family fp8 --m 4096 --n 3072 --k 3072 --rank 64 --bias
"""

from __future__ import annotations

import argparse

import torch

from flashinfer import SfLayout, nvfp4_quantize
from flashinfer.gemm import mm_fp4, mm_fp8_lora, mm_nvfp4_lora
from flashinfer.testing.utils import bench_gpu_time

E4M3_MAX = 448.0


def _quantize_fp8(source: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = (source.float().abs().max() / E4M3_MAX).clamp(min=1e-12).reshape(1)
    return (source.float() / scale).clamp(-E4M3_MAX, E4M3_MAX).to(
        torch.float8_e4m3fn
    ).contiguous(), scale


def _quantize_nvfp4(
    source: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """FlashInfer's stock quantizer in the 128x4 swizzled layout (the ``mm_fp4`` / ``mm_nvfp4_svdquant`` convention)."""
    global_scale = (
        (E4M3_MAX * 6.0 / source.float().abs().nan_to_num().max())
        .reshape(1)
        .float()
        .contiguous()
    )
    packed, scale_factors = nvfp4_quantize(
        source, global_scale, sfLayout=SfLayout.layout_128x4, do_shuffle=False
    )
    return packed.view(torch.uint8), scale_factors.view(torch.uint8), global_scale


def _problem(args: argparse.Namespace, device: torch.device) -> dict:
    generator = torch.Generator(device=device).manual_seed(args.seed)

    def randn(*shape, scale: float):
        return (
            torch.randn(shape, generator=generator, dtype=torch.float32, device=device)
            * scale
        ).to(torch.bfloat16)

    m, n, k, rank = args.m, args.n, args.k, args.rank
    x = randn(m, k, scale=1.0 / k**0.25)
    w = randn(n, k, scale=1.0 / k**0.25)
    lora_mag = (args.alpha_base / args.lora_scale) ** 0.5
    p = dict(
        d=randn(m, rank, scale=lora_mag / rank**0.25).contiguous(),
        lora_b=randn(n, rank, scale=lora_mag / rank**0.25).contiguous(),
        lora_scale=torch.tensor([args.lora_scale], dtype=torch.float32, device=device),
        bias=randn(n, scale=1.0).contiguous() if args.bias else None,
        out=torch.empty((m, n), dtype=torch.bfloat16, device=device),
    )
    if args.family == "nvfp4":
        a, a_sf, a_global = _quantize_nvfp4(x)
        b, b_sf, b_global = _quantize_nvfp4(w)
        p.update(
            a=a,
            b=b,
            a_sf=a_sf.reshape(-1),
            b_sf=b_sf.reshape(-1),
            a_sf_2d=a_sf,
            b_sf_2d=b_sf,
            alpha=(args.alpha_base / (a_global * b_global)).reshape(1).float(),
        )
    else:
        a, a_scale = _quantize_fp8(x)
        b, b_scale = _quantize_fp8(w)
        p.update(
            a=a,
            b=b,
            a_scale=a_scale,
            b_scale=(b_scale * args.alpha_base).reshape(1).float(),
        )
    return p


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--family", choices=("nvfp4", "fp8"), required=True)
    parser.add_argument("--m", type=int, default=4096)
    parser.add_argument("--n", type=int, default=3072)
    parser.add_argument("--k", type=int, default=3072)
    parser.add_argument("--rank", type=int, default=64, choices=(32, 64, 128))
    parser.add_argument("--bias", action="store_true")
    parser.add_argument("--alpha-base", type=float, default=0.37)
    parser.add_argument("--lora-scale", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--no-unfused",
        action="store_true",
        help="skip the base GEMM + addmm_ reference chain",
    )
    args = parser.parse_args()

    device = torch.device("cuda", torch.cuda.current_device())
    p = _problem(args, device)
    flops = 2.0 * args.m * args.n * args.k + 2.0 * args.m * args.n * args.rank

    if args.family == "nvfp4":

        def fused():
            mm_nvfp4_lora(
                p["a"],
                p["b"],
                p["a_sf"],
                p["b_sf"],
                p["alpha"],
                p["d"],
                p["lora_b"],
                p["lora_scale"],
                bias=p["bias"],
                out=p["out"],
            )

        def unfused():
            base = mm_fp4(
                p["a"],
                p["b"].T,
                p["a_sf_2d"],
                p["b_sf_2d"].T,
                p["alpha"],
                torch.bfloat16,
                None,
                block_size=16,
                use_8x4_sf_layout=False,
                backend="auto",
                use_nvfp4=True,
            )
            base.addmm_(p["d"], p["lora_b"].t(), alpha=args.lora_scale)
            if p["bias"] is not None:
                base.add_(p["bias"])
            return base

    else:

        def fused():
            mm_fp8_lora(
                p["a"],
                p["b"],
                p["a_scale"],
                p["b_scale"],
                p["d"],
                p["lora_b"],
                p["lora_scale"],
                bias=p["bias"],
                out=p["out"],
            )

        def unfused():
            # cuBLASLt per-tensor FP8 GEMM (the Goal's FP8 baseline chain); bias is added in fp32 by addmm_/add_.
            base = torch._scaled_mm(
                p["a"],
                p["b"].t(),
                scale_a=p["a_scale"],
                scale_b=p["b_scale"],
                out_dtype=torch.bfloat16,
            )
            base.addmm_(p["d"], p["lora_b"].t(), alpha=args.lora_scale)
            if p["bias"] is not None:
                base.add_(p["bias"])
            return base

    fused()
    torch.cuda.synchronize()
    times = bench_gpu_time(fused, enable_cupti=True, cold_l2_cache=True)
    median_ms = float(torch.tensor(times).median())
    print(
        f"{args.family} M={args.m} N={args.n} K={args.k} rank={args.rank} bias={args.bias}: "
        f"fused {median_ms * 1e3:.1f} us  ({flops / median_ms / 1e9:.1f} TFLOP/s)"
    )
    if not args.no_unfused:
        try:
            unfused()
            torch.cuda.synchronize()
            times = bench_gpu_time(unfused, enable_cupti=True, cold_l2_cache=True)
        except (
            Exception
        ) as exc:  # the plain base GEMM may not support this shape or device
            print(f"unfused chain unavailable: {exc}")
            return
        unfused_ms = float(torch.tensor(times).median())
        print(
            f"unfused base GEMM + addmm_ chain: {unfused_ms * 1e3:.1f} us  (fused / unfused = {median_ms / unfused_ms:.3f})"
        )


if __name__ == "__main__":
    main()
