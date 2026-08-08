#!/usr/bin/env python3
"""Benchmark NVFP4 SVDQuant on SM100/SM103 and SM120/SM121 GPUs.

For every (n, k) x m problem this script times five things after autotuning:
  1. mm_nvfp4_svdquant : selected SVDQuant implementation; auto tunes fused
                         versus unfused on SM120, or can be explicitly overridden
  2. unfused oracle    : the same operation composed from separate SM120 kernels
  3. svdquant_linear   : the full chain (nvfp4_quantize_smooth -> bf16 LoRA-down
                         GEMM -> fused GEMM)
  4. mm_fp4            : the stock NVFP4 GEMM on the same residual operands
                         (no LoRA correction), as the lower-bound baseline
  5. bf16 linear       : conventional dense BF16 linear GEMM + bias on the
                         unquantized activation and weight

The reported algorithmic TFLOPS/s count matmul operations only:
  * fused GEMM:      2*m*n*k + 2*m*n*rank
  * svdquant_linear: fused GEMM + 2*m*k*rank (LoRA-down)
  * mm_fp4:          2*m*n*k
  * bf16 linear:     2*m*n*k

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

    def run_selected():
        mm_nvfp4_svdquant(
            c["xq"],
            c["wq"],
            c["x_sf_flat"],
            c["w_sf_flat"],
            c["alpha"],
            c["d"],
            c["l1_scaled"],
            bias=c["bias"],
            out=c["out_fused"],
            backend=svdquant_backend,
        )

    def run_linear():
        svdquant_linear(
            c["x"],
            c["wq"],
            c["w_sf_flat"],
            c["alpha"],
            c["pqs"],
            c["l2t_smoothed"],
            c["l1_scaled"],
            c["global_sf"],
            bias=c["bias"],
            backend=svdquant_backend,
        )

    def run_unfused():
        mm_nvfp4_svdquant(
            c["xq"],
            c["wq"],
            c["x_sf_flat"],
            c["w_sf_flat"],
            c["alpha"],
            c["d"],
            c["l1_scaled"],
            bias=c["bias"],
            out=c["out_fused"],
            backend=unfused_backend,
        )

    def run_mm_fp4():
        mm_fp4(
            c["xq"],
            c["wq"].T,
            c["x_sf"],
            c["w_sf"].T,
            c["alpha"],
            torch.bfloat16,
            c["out_fp4"],
            block_size=16,
            use_8x4_sf_layout=False,
            backend=mm_fp4_backend,
            use_nvfp4=True,
        )

    def run_bf16_linear():
        torch.addmm(
            c["bias"],
            c["x"],
            c["w"].T,
            out=c["out_bf16"],
        )

    # Tune once; subsequent calls replay the best tactic from the tuner cache.
    with autotune(True):
        for _ in range(3):
            run_selected()
            if unfused_backend is not None:
                run_unfused()
            run_linear()
            run_mm_fp4()
            run_bf16_linear()
    torch.cuda.synchronize()

    bench_kwargs = dict(
        dry_run_time_ms=100,
        repeat_time_ms=500,
        use_cuda_graph=use_cuda_graph,
        enable_cupti=True,
        cold_l2_cache=cold_l2_cache,
    )
    selected_us = _median_us(bench_gpu_time(run_selected, **bench_kwargs))
    unfused_us = (
        _median_us(bench_gpu_time(run_unfused, **bench_kwargs))
        if unfused_backend is not None
        else float("nan")
    )
    linear_us = _median_us(bench_gpu_time(run_linear, **bench_kwargs))
    mm_fp4_us = _median_us(bench_gpu_time(run_mm_fp4, **bench_kwargs))
    bf16_linear_us = _median_us(bench_gpu_time(run_bf16_linear, **bench_kwargs))

    return selected_us, unfused_us, linear_us, mm_fp4_us, bf16_linear_us


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
    print("BF16 linear baseline: torch.addmm (PyTorch-selected CUDA backend)")
    print(f"unfused oracle backend: {unfused_backend or 'not available'}")
    print(f"execution mode: {'CUDA graph' if args.cuda_graph else 'eager'}")
    print(f"L2 mode: {'cold' if args.cold_l2_cache else 'warm'}")
    print("Timing: median GPU time in us (CUPTI preferred, CUDA-event fallback)")
    print("TFLOPS/s: algorithmic matmul operations; see module docstring\n")

    header = (
        f"{'n':>6} {'k':>6} {'m':>6} {'rank':>5} | "
        f"{'selected us':>11} {'selected TF/s':>13} | "
        f"{'unfused us':>11} {'unfused TF/s':>13} {'unfused/selected':>17} | "
        f"{'linear us':>10} {'linear TF/s':>11} | "
        f"{'mm_fp4 us':>10} {'mm_fp4 TF/s':>11} | {'mm_fp4/selected':>15}"
        f" | {'bf16 us':>9} {'bf16 TF/s':>10}"
    )
    print(header)
    print("-" * len(header))

    for rank in args.ranks:
        for n, k in args.nk_shapes:
            for m in args.m_values:
                selected_us, unfused_us, linear_us, mm_fp4_us, bf16_linear_us = (
                    bench_one(
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
                )
                fused_flops, linear_flops, mm_fp4_flops = _matmul_flops(m, n, k, rank)
                selected_tflops = _tflops_per_sec(fused_flops, selected_us)
                unfused_tflops = _tflops_per_sec(fused_flops, unfused_us)
                linear_tflops = _tflops_per_sec(linear_flops, linear_us)
                mm_fp4_tflops = _tflops_per_sec(mm_fp4_flops, mm_fp4_us)
                bf16_linear_tflops = _tflops_per_sec(mm_fp4_flops, bf16_linear_us)
                unfused_to_selected = (
                    unfused_us / selected_us if selected_us > 0 else float("nan")
                )
                mm_fp4_to_selected = (
                    mm_fp4_us / selected_us if selected_us > 0 else float("nan")
                )
                print(
                    f"{n:>6} {k:>6} {m:>6} {rank:>5} | "
                    f"{selected_us:>11.2f} {selected_tflops:>13.2f} | "
                    f"{unfused_us:>11.2f} {unfused_tflops:>13.2f} "
                    f"{unfused_to_selected:>17.3f} | "
                    f"{linear_us:>10.2f} {linear_tflops:>11.2f} | "
                    f"{mm_fp4_us:>10.2f} {mm_fp4_tflops:>11.2f} | "
                    f"{mm_fp4_to_selected:>15.3f} | "
                    f"{bf16_linear_us:>9.2f} {bf16_linear_tflops:>10.2f}"
                )
            print("-" * len(header))


if __name__ == "__main__":
    main()
