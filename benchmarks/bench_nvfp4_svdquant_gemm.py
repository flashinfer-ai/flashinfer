#!/usr/bin/env python3
"""Benchmark NVFP4 SVDQuant on SM100/SM103 and SM120/SM121 GPUs.

For every (n, k) x m problem this script times six things after autotuning:
  1. fused             : the fused residual + LoRA-up SVDQuant kernel
  2. unfused           : the same operation composed from separate SM120 kernels
  3. svdquant_linear   : the full chain (nvfp4_quantize_smooth -> bf16 LoRA-down
                         GEMM -> selected fused/unfused output implementation)
  4. residual_fp4      : the stock NVFP4 GEMM on the same residual operands
                         (no LoRA correction), as the lower-bound baseline
  5. BF16 GEMM         : dense BF16 residual GEMM + bias; timed but hidden
  6. FP8 per-tensor    : dense FP8 residual GEMM; timed but hidden

The reported algorithmic TFLOPS/s count matmul operations only:
  * fused GEMM:      2*m*n*k + 2*m*n*rank
  * svdquant_linear: fused GEMM + 2*m*k*rank (LoRA-down)
  * residual_fp4:    2*m*n*k

Quantization, alpha scaling, and bias addition are timed where applicable but
are not included in the operation count.

The LoRA rank defaults to 32; pass e.g. --ranks 32,64,96,128 to sweep.
The SVDQuant backend defaults to auto; pass --svdquant-backend fused or
--svdquant-backend unfused to override implementation selection.

Timing uses flashinfer.testing bench_gpu_time (CUPTI preferred, automatic
fallback to CUDA events). CUDA Graph replay is enabled by default; pass
--no-cuda-graph to measure eager launches instead.
"""

import argparse
import sys

import numpy as np
import torch

from flashinfer import (
    SfLayout,
    autotune,
    bmm_fp8,
    mm_fp4,
    mm_nvfp4_svdquant,
    nvfp4_quantize,
    svdquant_linear,
)
from flashinfer.gemm.gemm_svdquant import SVDQUANT_LORA_RANK_GRANULARITY
from flashinfer.testing.utils import bench_gpu_time
from flashinfer.utils import get_compute_capability

# Qwen-Image DiT linear shapes: (n, k) per layer type, m image-token counts.
NK_SHAPES = [(3072, 3072), (12288, 3072), (3072, 12288)]
M_VALUES = [4096, 6889, 9216, 16384]


def _to_float8(x, dtype=torch.float8_e4m3fn):
    """Quantize one tensor with a single scale, outside the timed region."""
    finfo = torch.finfo(dtype)
    min_value, max_value = x.aminmax()
    amax = torch.maximum(min_value.abs(), max_value.abs()).clamp(min=1e-12)
    scale = finfo.max / amax
    quantized = (x * scale).clamp(min=finfo.min, max=finfo.max).to(dtype)
    return quantized, scale.reciprocal().float()


def _build_case(m, n, k, rank, device):
    """Build all operands for one problem once (outside the timed region)."""
    quantize_backend = "cute-dsl" if get_compute_capability(device)[0] == 12 else "cuda"
    x = torch.randn(m, k, dtype=torch.bfloat16, device=device) / (k**0.25)
    pqs = (
        (1.0 + 0.3 * torch.randn(k, dtype=torch.bfloat16, device=device))
        .abs()
        .contiguous()
    )
    smoothed = (x * pqs).to(torch.bfloat16)
    global_sf = (
        ((448.0 * 6.0) / smoothed.float().abs().nan_to_num().max())
        .reshape(1)
        .contiguous()
    )

    w = torch.randn(n, k, dtype=torch.bfloat16, device=device) / (k**0.25)
    gw = ((448.0 * 6.0) / w.float().abs().nan_to_num().max()).reshape(1)
    wq, w_sf = nvfp4_quantize(
        w,
        gw,
        sfLayout=SfLayout.layout_128x4,
        do_shuffle=False,
        backend=quantize_backend,
    )
    wq = wq.view(torch.uint8)
    w_sf = w_sf.view(torch.uint8)
    alpha = (1.0 / (global_sf * gw)).reshape(1).float()

    # Quantized activation (byte-identical to nvfp4_quantize_smooth(x, pqs, gs)).
    xq, x_sf = nvfp4_quantize(
        smoothed,
        global_sf,
        sfLayout=SfLayout.layout_128x4,
        do_shuffle=False,
        backend=quantize_backend,
    )
    xq = xq.view(torch.uint8)
    x_sf = x_sf.view(torch.uint8)

    lora_a = torch.randn(rank, k, dtype=torch.bfloat16, device=device) / (k**0.25)
    l2t_smoothed = (pqs.unsqueeze(1) * lora_a.t()).contiguous()  # [k, rank]
    lora_b = torch.randn(n, rank, dtype=torch.bfloat16, device=device) / (rank**0.25)
    l1_scaled = (lora_b.float() / alpha).to(torch.bfloat16).contiguous()
    d = torch.mm(x, l2t_smoothed)  # LoRA-down output for the fused-GEMM-only path
    bias = torch.randn(n, dtype=torch.bfloat16, device=device).contiguous()
    x_fp8, x_fp8_scale = _to_float8(x)
    # bmm_fp8 expects B as column-major [batch, k, n]. Quantizing W.T retains
    # its column-major stride, while the singleton batch dimension is a view.
    w_fp8, w_fp8_scale = _to_float8(w.T)
    assert w_fp8.stride(0) == 1, (
        f"bmm_fp8 requires a column-major B; got strides {w_fp8.stride()}"
    )

    return {
        "x": x,
        "pqs": pqs,
        "global_sf": global_sf,
        "xq": xq,
        "x_sf": x_sf,  # 2-D swizzled layout (mm_fp4 convention)
        "x_sf_flat": x_sf.reshape(-1),  # 1-D buffer (fused-kernel convention)
        "wq": wq,
        "w": w,
        "w_sf": w_sf,
        "w_sf_flat": w_sf.reshape(-1),
        "alpha": alpha,
        "l2t_smoothed": l2t_smoothed,
        "l1_scaled": l1_scaled,
        "d": d,
        "bias": bias,
        "out_fused": torch.empty(m, n, dtype=torch.bfloat16, device=device),
        "out_fp4": torch.empty(m, n, dtype=torch.bfloat16, device=device),
        "out_bf16": torch.empty(m, n, dtype=torch.bfloat16, device=device),
        "x_fp8": x_fp8.unsqueeze(0),
        "x_fp8_scale": x_fp8_scale,
        "w_fp8": w_fp8.unsqueeze(0),
        "w_fp8_scale": w_fp8_scale,
        "out_fp8": torch.empty(1, m, n, dtype=torch.bfloat16, device=device),
    }


def _median_us(times_ms):
    return float(np.median(times_ms) * 1000.0)


def _matmul_flops(m, n, k, rank):
    """Return (fused, full-linear, residual-only) algorithmic FLOP counts."""
    residual_flops = 2 * m * n * k
    lora_up_flops = 2 * m * n * rank
    lora_down_flops = 2 * m * k * rank
    fused_flops = residual_flops + lora_up_flops
    return fused_flops, fused_flops + lora_down_flops, residual_flops


def _tflops_per_sec(flops, latency_us):
    """Convert an operation count and latency in microseconds to TFLOPS/s."""
    return flops / (latency_us * 1e6) if latency_us > 0 else float("nan")


def bench_one(
    m,
    n,
    k,
    rank,
    device,
    mm_fp4_backend,
    svdquant_backend,
    unfused_backend=None,
    use_cuda_graph=True,
    cold_l2_cache=False,
):
    c = _build_case(m, n, k, rank, device)

    fused_backend = "cute-dsl" if get_compute_capability(device)[0] == 12 else "cutlass"

    def run_fused(xq, wq, x_sf, w_sf, alpha, d, l1, bias, out):
        mm_nvfp4_svdquant(
            xq,
            wq,
            x_sf,
            w_sf,
            alpha,
            d,
            l1,
            bias=bias,
            out=out,
            backend=fused_backend,
        )

    def run_svdquant_linear(
        x, wq, w_sf, alpha, pqs, l2t_smoothed, l1_scaled, global_sf, bias
    ):
        svdquant_linear(
            x,
            wq,
            w_sf,
            alpha,
            pqs,
            l2t_smoothed,
            l1_scaled,
            global_sf,
            bias=bias,
            backend=svdquant_backend,
        )

    def run_unfused(xq, wq, x_sf, w_sf, alpha, d, l1, bias, out):
        mm_nvfp4_svdquant(
            xq,
            wq,
            x_sf,
            w_sf,
            alpha,
            d,
            l1,
            bias=bias,
            out=out,
            backend=unfused_backend,
        )

    def run_mm_fp4(xq, wq, x_sf, w_sf, alpha, out):
        mm_fp4(
            xq,
            wq.T,
            x_sf,
            w_sf.T,
            alpha,
            torch.bfloat16,
            out,
            block_size=16,
            use_8x4_sf_layout=False,
            backend=mm_fp4_backend,
            use_nvfp4=True,
        )

    def run_residual_gemm(x, w, bias, out):
        torch.addmm(
            bias,
            x,
            w.T,
            out=out,
        )

    def run_fp8_per_tensor(x, w, x_scale, w_scale, out):
        bmm_fp8(
            x,
            w,
            x_scale,
            w_scale,
            torch.bfloat16,
            out=out,
            backend="auto",
        )

    fused_args = (
        c["xq"],
        c["wq"],
        c["x_sf_flat"],
        c["w_sf_flat"],
        c["alpha"],
        c["d"],
        c["l1_scaled"],
        c["bias"],
        c["out_fused"],
    )
    svdquant_linear_args = (
        c["x"],
        c["wq"],
        c["w_sf_flat"],
        c["alpha"],
        c["pqs"],
        c["l2t_smoothed"],
        c["l1_scaled"],
        c["global_sf"],
        c["bias"],
    )
    mm_fp4_args = (
        c["xq"],
        c["wq"],
        c["x_sf"],
        c["w_sf"],
        c["alpha"],
        c["out_fp4"],
    )
    residual_gemm_args = (c["x"], c["w"], c["bias"], c["out_bf16"])
    fp8_args = (
        c["x_fp8"],
        c["w_fp8"],
        c["x_fp8_scale"],
        c["w_fp8_scale"],
        c["out_fp8"],
    )

    # Tune once; subsequent calls replay the best tactic from the tuner cache.
    with autotune(True):
        for _ in range(3):
            run_fused(*fused_args)
            if unfused_backend is not None:
                run_unfused(*fused_args)
            run_svdquant_linear(*svdquant_linear_args)
            run_mm_fp4(*mm_fp4_args)
            run_residual_gemm(*residual_gemm_args)
            run_fp8_per_tensor(*fp8_args)
    torch.cuda.synchronize()

    bench_kwargs = dict(
        dry_run_time_ms=100,
        repeat_time_ms=500,
        use_cuda_graph=use_cuda_graph,
        enable_cupti=True,
        cold_l2_cache=cold_l2_cache,
    )
    fused_us = _median_us(
        bench_gpu_time(run_fused, input_args=fused_args, **bench_kwargs)
    )
    unfused_us = (
        _median_us(bench_gpu_time(run_unfused, input_args=fused_args, **bench_kwargs))
        if unfused_backend is not None
        else float("nan")
    )
    svdquant_linear_us = _median_us(
        bench_gpu_time(
            run_svdquant_linear, input_args=svdquant_linear_args, **bench_kwargs
        )
    )
    mm_fp4_us = _median_us(
        bench_gpu_time(run_mm_fp4, input_args=mm_fp4_args, **bench_kwargs)
    )
    residual_gemm_us = _median_us(
        bench_gpu_time(run_residual_gemm, input_args=residual_gemm_args, **bench_kwargs)
    )
    fp8_per_tensor_us = _median_us(
        bench_gpu_time(run_fp8_per_tensor, input_args=fp8_args, **bench_kwargs)
    )

    return (
        fused_us,
        unfused_us,
        svdquant_linear_us,
        mm_fp4_us,
        residual_gemm_us,
        fp8_per_tensor_us,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--nk-shapes",
        type=lambda s: [tuple(map(int, shape.split("x"))) for shape in s.split(",")],
        default=NK_SHAPES,
        help="comma-separated NxK shapes (default: 3072x3072,12288x3072,3072x12288)",
    )
    parser.add_argument(
        "--m-values",
        type=lambda s: [int(m) for m in s.split(",")],
        default=M_VALUES,
        help="comma-separated M values (default: 4096,6889,9216,16384)",
    )
    parser.add_argument(
        "--ranks",
        type=lambda s: [int(r) for r in s.split(",")],
        default=[SVDQUANT_LORA_RANK_GRANULARITY],
        help="comma-separated LoRA ranks to sweep (positive multiples of 32)",
    )
    parser.add_argument(
        "--svdquant-backend",
        choices=("auto", "fused", "unfused"),
        default="auto",
        help=(
            "SVDQuant implementation policy: auto tunes fused versus unfused on "
            "SM120; fused or unfused forces that implementation (default: auto)"
        ),
    )
    parser.add_argument(
        "--cuda-graph",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="benchmark CUDA Graph replay (default); use --no-cuda-graph for eager",
    )
    parser.add_argument(
        "--cold-l2-cache",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="request cold-L2 timing (default: warm L2)",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is not available; this benchmark requires a Blackwell GPU.")
        sys.exit(1)
    major, minor = get_compute_capability(torch.device(device="cuda"))
    if (major, minor) not in ((10, 0), (10, 3), (12, 0), (12, 1)):
        print(
            "NVFP4 SVDQuant kernels require SM100/SM103 or SM120/SM121; "
            f"got SM{major}{minor}. Exiting."
        )
        sys.exit(1)

    torch.manual_seed(0)
    device = torch.device("cuda")
    mm_fp4_backend = "cutlass" if major == 10 else "b12x"
    unfused_backend = "cute-dsl-unfused" if major == 12 else None
    if major == 12:
        svdquant_backend = {
            "auto": "auto",
            "fused": "cute-dsl",
            "unfused": "cute-dsl-unfused",
        }[args.svdquant_backend]
    else:
        if args.svdquant_backend == "unfused":
            parser.error("--svdquant-backend unfused is only available on SM120/SM121")
        svdquant_backend = "cutlass" if args.svdquant_backend == "fused" else "auto"
    print(f"Device: {torch.cuda.get_device_name(device)} (SM{major}{minor})")
    print(f"mm_fp4 baseline backend: {mm_fp4_backend}")
    print(f"SVDQuant implementation policy: {args.svdquant_backend}")
    print(
        "BF16 residual_gemm baseline: torch.addmm (PyTorch-selected CUDA backend; "
        "residual + bias only, no LoRA-down or LoRA-up correction)"
    )
    print(
        "FP8 per-tensor baseline: bmm_fp8 backend=auto "
        "(residual only; scales, quantization, LoRA-down, and LoRA-up excluded)"
    )
    print(f"unfused oracle backend: {unfused_backend or 'not available'}")
    print(f"execution mode: {'CUDA graph' if args.cuda_graph else 'eager'}")
    print(f"L2 mode: {'cold' if args.cold_l2_cache else 'warm'}")
    print("Timing: median GPU time in us (CUPTI preferred, CUDA-event fallback)")
    print("TFLOPS/s: algorithmic matmul operations; see module docstring\n")

    header = (
        f"{'n':>6} {'k':>6} {'m':>6} {'R':>5} | "
        f"{'svdquant_linear us':>18} {'gain vs BF16':>13} {'gain vs FP8':>12} | "
        f"{'residual_fp4/fused':>20} {'residual_fp4 TF/s/us':>22} | "
        f"{'fusion gain':>11} {'fused TF/s/us':>17} {'unfused TF/s/us':>19}"
    )
    print(header)
    print("-" * len(header))

    for rank in args.ranks:
        for n, k in args.nk_shapes:
            for m in args.m_values:
                (
                    fused_us,
                    unfused_us,
                    svdquant_linear_us,
                    mm_fp4_us,
                    residual_gemm_us,
                    fp8_per_tensor_us,
                ) = bench_one(
                    m,
                    n,
                    k,
                    rank,
                    device,
                    mm_fp4_backend,
                    svdquant_backend,
                    unfused_backend,
                    args.cuda_graph,
                    args.cold_l2_cache,
                )
                fused_flops, _, mm_fp4_flops = _matmul_flops(m, n, k, rank)
                fused_tflops = _tflops_per_sec(fused_flops, fused_us)
                unfused_tflops = _tflops_per_sec(fused_flops, unfused_us)
                mm_fp4_tflops = _tflops_per_sec(mm_fp4_flops, mm_fp4_us)
                gain_over_bf16 = (
                    (residual_gemm_us / svdquant_linear_us - 1.0) * 100.0
                    if svdquant_linear_us > 0
                    else float("nan")
                )
                gain_over_fp8 = (
                    (fp8_per_tensor_us / svdquant_linear_us - 1.0) * 100.0
                    if svdquant_linear_us > 0
                    else float("nan")
                )
                fusion_efficiency = (
                    mm_fp4_us / fused_us if fused_us > 0 else float("nan")
                )
                fusion_gain = (
                    (unfused_us / fused_us - 1.0) * 100.0
                    if fused_us > 0
                    else float("nan")
                )
                print(
                    f"{n:>6} {k:>6} {m:>6} {rank:>5} | "
                    f"{svdquant_linear_us:>18.2f} {gain_over_bf16:>12.1f}% "
                    f"{gain_over_fp8:>11.1f}% | "
                    f"{fusion_efficiency:>20.3f} "
                    f"{mm_fp4_tflops:>10.2f}/{mm_fp4_us:<9.2f} | "
                    f"{fusion_gain:>10.1f}% "
                    f"{fused_tflops:>8.2f}/{fused_us:<8.2f} "
                    f"{unfused_tflops:>9.2f}/{unfused_us:<9.2f}"
                )
            print("-" * len(header))


if __name__ == "__main__":
    main()
