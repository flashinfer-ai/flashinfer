#!/usr/bin/env python3
"""Benchmark for the SM100 NVFP4 SVDQuant fused GEMM (Qwen-Image linear shapes).

For every (n, k) x m problem this script times three things after autotuning:
  1. mm_nvfp4_svdquant : fused residual NVFP4 GEMM + rank-r BF16 LoRA-up + bias
  2. svdquant_linear   : the full chain (nvfp4_quantize_smooth -> bf16 LoRA-down
                         GEMM -> fused GEMM)
  3. mm_fp4 (cutlass)  : the stock NVFP4 GEMM on the same residual operands
                         (no LoRA correction), as the lower-bound baseline

The LoRA rank defaults to 32; pass e.g. --ranks 32,64,96,128 to sweep.

Timing uses flashinfer.testing bench_gpu_time (CUPTI preferred, automatic
fallback to CUDA events).
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
    wq, w_sf = nvfp4_quantize(w, gw, sfLayout=SfLayout.layout_128x4, do_shuffle=False)
    wq = wq.view(torch.uint8)
    w_sf = w_sf.view(torch.uint8)
    alpha = (1.0 / (global_sf * gw)).reshape(1).float()

    # Quantized activation (byte-identical to nvfp4_quantize_smooth(x, pqs, gs)).
    xq, x_sf = nvfp4_quantize(
        smoothed, global_sf, sfLayout=SfLayout.layout_128x4, do_shuffle=False
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
        "w_sf": w_sf,
        "w_sf_flat": w_sf.reshape(-1),
        "alpha": alpha,
        "l2t_smoothed": l2t_smoothed,
        "l1_scaled": l1_scaled,
        "d": d,
        "bias": bias,
        "out_fused": torch.empty(m, n, dtype=torch.bfloat16, device=device),
        "out_fp4": torch.empty(m, n, dtype=torch.bfloat16, device=device),
    }


def _median_us(times_ms):
    return float(np.median(times_ms) * 1000.0)


def bench_one(m, n, k, rank, device):
    c = _build_case(m, n, k, rank, device)

    def run_fused():
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
            backend="cutlass",
            use_nvfp4=True,
        )

    # Tune once; subsequent calls replay the best tactic from the tuner cache.
    with autotune(True):
        for _ in range(3):
            run_fused()
            run_linear()
            run_mm_fp4()
    torch.cuda.synchronize()

    bench_kwargs = dict(
        dry_run_time_ms=100,
        repeat_time_ms=500,
        use_cuda_graph=True,
        enable_cupti=True,
        cold_l2_cache=True,
    )
    fused_us = _median_us(bench_gpu_time(run_fused, **bench_kwargs))
    linear_us = _median_us(bench_gpu_time(run_linear, **bench_kwargs))
    mm_fp4_us = _median_us(bench_gpu_time(run_mm_fp4, **bench_kwargs))

    return fused_us, linear_us, mm_fp4_us


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ranks",
        type=lambda s: [int(r) for r in s.split(",")],
        default=[SVDQUANT_LORA_RANK_GRANULARITY],
        help="comma-separated LoRA ranks to sweep (positive multiples of 32)",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is not available; this benchmark requires an SM100-class GPU.")
        sys.exit(1)
    major, minor = get_compute_capability(torch.device(device="cuda"))
    if major != 10:
        print(
            "NVFP4 SVDQuant kernels require SM100-class GPUs (Blackwell); "
            f"got SM{major}{minor}. Exiting."
        )
        sys.exit(1)

    torch.manual_seed(0)
    device = torch.device("cuda")
    print(f"Device: {torch.cuda.get_device_name(device)} (SM{major}{minor})")
    print("Timing: median GPU time in us (CUPTI preferred, CUDA-event fallback)\n")

    header = (
        f"{'n':>6} {'k':>6} {'m':>6} {'rank':>5} | {'fused GEMM':>12} "
        f"{'svdq linear':>12} {'mm_fp4':>12} | {'fused/mm_fp4':>12}"
    )
    print(header)
    print("-" * len(header))

    for rank in args.ranks:
        for n, k in NK_SHAPES:
            for m in M_VALUES:
                fused_us, linear_us, mm_fp4_us = bench_one(m, n, k, rank, device)
                ratio = fused_us / mm_fp4_us if mm_fp4_us > 0 else float("nan")
                print(
                    f"{n:>6} {k:>6} {m:>6} {rank:>5} | {fused_us:>12.2f} "
                    f"{linear_us:>12.2f} {mm_fp4_us:>12.2f} | {ratio:>12.3f}"
                )
            print("-" * len(header))


if __name__ == "__main__":
    main()
