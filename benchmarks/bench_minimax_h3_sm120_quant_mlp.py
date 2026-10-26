"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Benchmark the SM120 FP8 / NVFP4 fused MiniMax-H3 MLP (RMSNorm + AdaLN + FC1 + SwiGLU + FC2 +
gated residual) against a segmented torch / FlashInfer chain (rmsnorm -> modulate -> quantize ->
GEMM -> silu_and_mul -> quantize -> GEMM -> gate * o + residual).

    python benchmarks/bench_minimax_h3_sm120_quant_mlp.py --variant nvfp4 --m 4097 33472 38592

Timing uses CUDA events around whole operator calls (median of ``--iters`` after ``--warmup``
launches); it is a convenience comparison, not a kernel-level measurement.
"""

from __future__ import annotations

import argparse
import json
from typing import Callable

import torch

from flashinfer import mm_fp4, silu_and_mul
from flashinfer.diffusion_ops import (
    minimax_h3_mlp_fp8_sm120,
    minimax_h3_mlp_nvfp4_sm120,
    prepare_minimax_h3_fc1_weight_fp8,
    prepare_minimax_h3_fc1_weight_nvfp4,
    prepare_minimax_h3_fc2_weight_fp8,
    prepare_minimax_h3_fc2_weight_nvfp4_sm120,
)
from flashinfer.diffusion_ops.cake_minimax_h3_sm120_quant_fc1_swiglu import (
    E4M3_MAX,
    MINIMAX_H3_ADALN_ROWS,
    MINIMAX_H3_EPS,
    MINIMAX_H3_FC1_ROWS,
    MINIMAX_H3_FFN,
    MINIMAX_H3_HIDDEN,
    MINIMAX_H3_SF_BLOCK,
    fp8_scale_from_amax,
)
from flashinfer.diffusion_ops.minimax_h3_fc1_swiglu import (
    minimax_h3_nvfp4_alpha,
    minimax_h3_nvfp4_global_scale,
)
from flashinfer.quantization import fp4_quantize

# Representative MiniMax-H3 token counts (one 128/256-row tail case, two video DiT SP1 frame counts).
DEFAULT_ROWS = (4097, 33472, 38592)
FLOPS_PER_ROW = 2.0 * MINIMAX_H3_HIDDEN * (MINIMAX_H3_FC1_ROWS + MINIMAX_H3_FFN)


def synthetic_model(device: torch.device, generator: torch.Generator) -> dict:
    bf16 = torch.bfloat16

    def uniform(shape, low, high):
        return torch.empty(shape, dtype=bf16, device=device).uniform_(
            low, high, generator=generator
        )

    def normal(shape, std):
        t = torch.empty(shape, dtype=bf16, device=device)
        return t.normal_(mean=0.0, std=std, generator=generator)

    return {
        "fc1_weight": normal((MINIMAX_H3_FC1_ROWS, MINIMAX_H3_HIDDEN), 0.02),
        "fc2_weight": normal((MINIMAX_H3_HIDDEN, MINIMAX_H3_FFN), 0.02),
        "x_norm_weight": uniform((MINIMAX_H3_HIDDEN,), 0.9, 1.1),
        "adaln_scale": uniform((MINIMAX_H3_ADALN_ROWS, MINIMAX_H3_HIDDEN), -0.05, 0.05),
        "adaln_shift": uniform((MINIMAX_H3_ADALN_ROWS, MINIMAX_H3_HIDDEN), -0.05, 0.05),
        "gate": uniform((MINIMAX_H3_ADALN_ROWS, MINIMAX_H3_HIDDEN), -1.0, 1.0),
    }


def synthetic_inputs(
    rows: int, device: torch.device, generator: torch.Generator
) -> dict:
    x = torch.empty((rows, MINIMAX_H3_HIDDEN), dtype=torch.bfloat16, device=device)
    x.normal_(mean=0.0, std=0.5, generator=generator)
    residual = torch.empty(
        (rows, MINIMAX_H3_HIDDEN), dtype=torch.bfloat16, device=device
    )
    residual.normal_(mean=0.0, std=1.0, generator=generator)
    positions = torch.arange(rows, device=device, dtype=torch.int64)
    adaln_index = torch.div(
        positions * MINIMAX_H3_ADALN_ROWS, max(rows, 1), rounding_mode="floor"
    ).clamp_max(MINIMAX_H3_ADALN_ROWS - 1)
    return {"x": x, "residual": residual, "adaln_index": adaln_index}


def torch_pre_norm(x, model, adaln_index):
    norm = torch.nn.functional.rms_norm(
        x, (MINIMAX_H3_HIDDEN,), model["x_norm_weight"], eps=MINIMAX_H3_EPS
    )
    scale = model["adaln_scale"].index_select(0, adaln_index)
    shift = model["adaln_shift"].index_select(0, adaln_index)
    return torch.addcmul(
        shift, norm.to(torch.bfloat16), (scale + 1.0).to(torch.bfloat16)
    ).to(torch.bfloat16)


def gated_residual(o, model, adaln_index, residual):
    gate = model["gate"].index_select(0, adaln_index)
    return (residual + (gate * o).to(torch.bfloat16)).to(torch.bfloat16)


def quantize_fp8_rows(t: torch.Tensor, chunk_rows: int = 2048):
    """Per-row E4M3 (``scale = RN(amax / 448)``) in the natural row order for the chain."""
    q = torch.empty(t.shape, dtype=torch.float8_e4m3fn, device=t.device)
    scale = torch.empty((t.shape[0],), dtype=torch.float32, device=t.device)
    for start in range(0, t.shape[0], chunk_rows):
        stop = min(start + chunk_rows, t.shape[0])
        rows = t[start:stop].float()
        s = fp8_scale_from_amax(rows.abs().amax(dim=1).clamp_min(1e-12))
        scale[start:stop] = s
        q[start:stop] = (
            (rows / s[:, None]).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
        )
    return q, scale


def quantize_fp8_tokens(a: torch.Tensor):
    scale = fp8_scale_from_amax(a.float().abs().amax(dim=1).clamp_min(1e-12))
    q = (a.float() / scale[:, None]).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    return q, scale


def chain_fp8(inputs, model, weights):
    w1_q, w1_scale, w2_q, w2_scale = weights
    a = torch_pre_norm(inputs["x"], model, inputs["adaln_index"])
    a_q, a_scale = quantize_fp8_tokens(a)
    h = torch._scaled_mm(
        a_q,
        w1_q.t(),
        scale_a=a_scale[:, None],
        scale_b=w1_scale[None, :],
        out_dtype=torch.bfloat16,
    )
    y = silu_and_mul(h)
    y_q, y_scale = quantize_fp8_tokens(y)
    o = torch._scaled_mm(
        y_q,
        w2_q.t(),
        scale_a=y_scale[:, None],
        scale_b=w2_scale[None, :],
        out_dtype=torch.bfloat16,
    )
    return gated_residual(o, model, inputs["adaln_index"], inputs["residual"])


def _fp4(t, g):
    return fp4_quantize(
        t,
        g,
        sf_vec_size=MINIMAX_H3_SF_BLOCK,
        sf_use_ue8m0=False,
        is_sf_swizzled_layout=True,
    )


def chain_nvfp4(inputs, model, weights, g_a, alpha1, g_y, alpha2):
    w1_q, w1_sf, w2_q, w2_sf = weights
    a = torch_pre_norm(inputs["x"], model, inputs["adaln_index"])
    a_q, a_sf = _fp4(a, g_a)
    h = mm_fp4(
        a_q,
        w1_q.t(),
        a_sf,
        w1_sf.t() if w1_sf.ndim == 2 else w1_sf,
        alpha1,
        torch.bfloat16,
        backend="cutlass",
    )
    y = silu_and_mul(h)
    y_q, y_sf = _fp4(y, g_y)
    o = mm_fp4(
        y_q,
        w2_q.t(),
        y_sf,
        w2_sf.t() if w2_sf.ndim == 2 else w2_sf,
        alpha2,
        torch.bfloat16,
        backend="cutlass",
    )
    return gated_residual(o, model, inputs["adaln_index"], inputs["residual"])


def time_ms(fn: Callable[[], object], warmup: int, iters: int) -> float:
    """Median wall time of ``fn`` on the current stream measured with CUDA events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        stop.record()
        stop.synchronize()
        samples.append(start.elapsed_time(stop))
    samples.sort()
    return float(samples[len(samples) // 2])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--variant", choices=("fp8", "nvfp4"), default="nvfp4")
    parser.add_argument("--m", type=int, nargs="+", default=list(DEFAULT_ROWS))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--fp8-mma-form", type=int, default=-1, choices=(-1, 0, 2))
    parser.add_argument("--no-baseline", action="store_true")
    parser.add_argument("--json", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda:0")
    generator = torch.Generator(device=device).manual_seed(4618)
    model = synthetic_model(device, generator)
    if args.variant == "fp8":
        w1 = prepare_minimax_h3_fc1_weight_fp8(model["fc1_weight"])
        w2 = prepare_minimax_h3_fc2_weight_fp8(model["fc2_weight"])
        chain_weights = quantize_fp8_rows(model["fc1_weight"]) + w2
        g_a = g_w1 = g_w2 = alpha1 = None
    else:
        g_w1 = minimax_h3_nvfp4_global_scale(model["fc1_weight"])
        g_w2 = minimax_h3_nvfp4_global_scale(model["fc2_weight"])
        w1 = prepare_minimax_h3_fc1_weight_nvfp4(
            model["fc1_weight"], g_w1
        )  # SM120 layout here
        w2 = prepare_minimax_h3_fc2_weight_nvfp4_sm120(model["fc2_weight"], g_w2)
        chain_weights = _fp4(model["fc1_weight"], g_w1) + _fp4(
            model["fc2_weight"], g_w2
        )
        # Static activation global scale for a calibrated |a| <= 8 (synthetic model).
        g_a = torch.tensor([E4M3_MAX * 6.0 / 8.0], dtype=torch.float32, device=device)
        alpha1 = minimax_h3_nvfp4_alpha(g_a, g_w1)

    rows_out = []
    for rows in args.m:
        inputs = synthetic_inputs(rows, device, generator)
        if args.variant == "fp8":
            fn = lambda: minimax_h3_mlp_fp8_sm120(  # noqa: E731
                inputs["x"],
                model["x_norm_weight"],
                model["adaln_scale"],
                model["adaln_shift"],
                inputs["adaln_index"],
                model["gate"],
                inputs["residual"],
                *w1,
                *w2,
                fp8_mma_form=args.fp8_mma_form,
            )
            base = lambda: chain_fp8(inputs, model, chain_weights)  # noqa: E731
        else:
            # Static y global scale calibrated once from the BF16 chain's y of this shape.
            with torch.no_grad():
                a = torch_pre_norm(inputs["x"], model, inputs["adaln_index"])
                a_q, a_sf = _fp4(a, g_a)
                h = mm_fp4(
                    a_q,
                    chain_weights[0].t(),
                    a_sf,
                    chain_weights[1].t()
                    if chain_weights[1].ndim == 2
                    else chain_weights[1],
                    alpha1,
                    torch.bfloat16,
                    backend="cutlass",
                )
                g_y = minimax_h3_nvfp4_global_scale(silu_and_mul(h))
                del a, a_q, a_sf, h
            alpha2 = minimax_h3_nvfp4_alpha(g_y, g_w2)
            a1, a2 = float(alpha1.item()), float(alpha2.item())
            fn = lambda: minimax_h3_mlp_nvfp4_sm120(  # noqa: E731
                inputs["x"],
                model["x_norm_weight"],
                model["adaln_scale"],
                model["adaln_shift"],
                inputs["adaln_index"],
                model["gate"],
                inputs["residual"],
                g_a,
                *w1,
                a1,
                g_y,
                *w2,
                a2,
            )
            base = lambda: chain_nvfp4(
                inputs, model, chain_weights, g_a, alpha1, g_y, alpha2
            )  # noqa: E731
        fn()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        fused_ms = time_ms(fn, args.warmup, args.iters)
        row = {
            "variant": args.variant,
            "M": rows,
            "fused_ms": fused_ms,
            "fused_tflops": rows * FLOPS_PER_ROW / fused_ms / 1.0e9,
            "fused_peak_gib": torch.cuda.max_memory_allocated() / 2**30,
        }
        if not args.no_baseline:
            try:
                base()
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                row["chain_ms"] = time_ms(base, args.warmup, args.iters)
                row["chain_peak_gib"] = torch.cuda.max_memory_allocated() / 2**30
                row["speedup"] = row["chain_ms"] / fused_ms
            except torch.OutOfMemoryError:
                row["chain_ms"] = None
                row["chain_status"] = "memory_limited"
            torch.cuda.empty_cache()
        print(json.dumps(row), flush=True)
        rows_out.append(row)
    if args.json:
        with open(args.json, "w") as stream:
            json.dump(
                {"device": torch.cuda.get_device_name(device), "rows": rows_out},
                stream,
                indent=1,
            )


if __name__ == "__main__":
    main()
