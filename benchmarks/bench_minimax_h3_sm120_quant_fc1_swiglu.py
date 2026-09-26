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

Benchmark the SM120 FP8 / NVFP4 fused MiniMax-H3 RMSNorm + AdaLN + FC1 + SwiGLU against a
segmented torch / FlashInfer chain (rmsnorm -> modulate -> quantize -> GEMM -> silu_and_mul).

    python benchmarks/bench_minimax_h3_sm120_quant_fc1_swiglu.py --variant nvfp4 --m 4097 33472 38592

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
    minimax_h3_fc1_swiglu_fp8,
    minimax_h3_fc1_swiglu_nvfp4,
    prepare_minimax_h3_fc1_weight_fp8,
    prepare_minimax_h3_fc1_weight_nvfp4,
)
from flashinfer.diffusion_ops.cake_minimax_h3_sm120_quant_fc1_swiglu import (
    E4M3_MAX,
    MINIMAX_H3_ADALN_ROWS,
    MINIMAX_H3_EPS,
    MINIMAX_H3_FC1_ROWS,
    MINIMAX_H3_HIDDEN,
    MINIMAX_H3_SF_BLOCK,
    fp8_scale_from_amax,
)
from flashinfer.diffusion_ops.minimax_h3_fc1_swiglu import (
    minimax_h3_nvfp4_alpha,
    minimax_h3_nvfp4_global_scale,
)
from flashinfer.quantization import fp4_quantize

# Representative MiniMax-H3 token counts (one 128-row tail case, two video DiT SP1 frame counts).
DEFAULT_ROWS = (4097, 33472, 38592)


def synthetic_model(device: torch.device, generator: torch.Generator) -> dict:
    bf16 = torch.bfloat16

    def uniform(shape, low, high):
        return torch.empty(shape, dtype=bf16, device=device).uniform_(
            low, high, generator=generator
        )

    fc1_weight = torch.empty(
        (MINIMAX_H3_FC1_ROWS, MINIMAX_H3_HIDDEN), dtype=bf16, device=device
    )
    fc1_weight.normal_(mean=0.0, std=0.02, generator=generator)
    return {
        "fc1_weight": fc1_weight,
        "x_norm_weight": uniform((MINIMAX_H3_HIDDEN,), 0.9, 1.1),
        "adaln_scale": uniform((MINIMAX_H3_ADALN_ROWS, MINIMAX_H3_HIDDEN), -0.05, 0.05),
        "adaln_shift": uniform((MINIMAX_H3_ADALN_ROWS, MINIMAX_H3_HIDDEN), -0.05, 0.05),
    }


def synthetic_inputs(
    rows: int, device: torch.device, generator: torch.Generator
) -> dict:
    x = torch.empty((rows, MINIMAX_H3_HIDDEN), dtype=torch.bfloat16, device=device)
    x.normal_(mean=0.0, std=0.5, generator=generator)
    positions = torch.arange(rows, device=device, dtype=torch.int64)
    adaln_index = (
        torch.div(
            positions * MINIMAX_H3_ADALN_ROWS, max(rows, 1), rounding_mode="floor"
        )
        .clamp_max(MINIMAX_H3_ADALN_ROWS - 1)
        .to(torch.int32)
    )
    return {"x": x, "adaln_index": adaln_index}


def torch_pre_norm(x, model, adaln_index):
    norm = torch.nn.functional.rms_norm(
        x, (MINIMAX_H3_HIDDEN,), model["x_norm_weight"], eps=MINIMAX_H3_EPS
    )
    index = adaln_index.to(torch.int64)
    scale = model["adaln_scale"].index_select(0, index)
    shift = model["adaln_shift"].index_select(0, index)
    return torch.addcmul(
        shift, norm.to(torch.bfloat16), (scale + 1.0).to(torch.bfloat16)
    ).to(torch.bfloat16)


def quantize_fp8_rows_linear(weight: torch.Tensor, chunk_rows: int = 2048):
    """Per-output-channel E4M3 weight in the natural (gate rows; up rows) order for the baseline."""
    w_q = torch.empty(weight.shape, dtype=torch.float8_e4m3fn, device=weight.device)
    scale = torch.empty((weight.shape[0],), dtype=torch.float32, device=weight.device)
    for start in range(0, weight.shape[0], chunk_rows):
        stop = min(start + chunk_rows, weight.shape[0])
        rows = weight[start:stop].float()
        s = fp8_scale_from_amax(rows.abs().amax(dim=1).clamp_min(1e-12))
        scale[start:stop] = s
        w_q[start:stop] = (
            (rows / s[:, None]).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
        )
    return w_q, scale


def baseline_fp8(inputs, model, weights):
    w_q, w_scale = weights
    a = torch_pre_norm(inputs["x"], model, inputs["adaln_index"])
    scale = fp8_scale_from_amax(a.float().abs().amax(dim=1).clamp_min(1e-12))
    a_q = (
        (a.float() / scale[:, None]).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    )
    h = torch._scaled_mm(
        a_q,
        w_q.t(),
        scale_a=scale[:, None],
        scale_b=w_scale[None, :],
        out_dtype=torch.bfloat16,
    )
    return silu_and_mul(h)


def baseline_nvfp4(inputs, model, weights, act_global_scale, alpha):
    w_q, w_sf = weights
    a = torch_pre_norm(inputs["x"], model, inputs["adaln_index"])
    a_q, a_sf = fp4_quantize(
        a,
        act_global_scale,
        sf_vec_size=MINIMAX_H3_SF_BLOCK,
        sf_use_ue8m0=False,
        is_sf_swizzled_layout=True,
    )
    h = mm_fp4(
        a_q,
        w_q.t(),
        a_sf,
        w_sf.t() if w_sf.ndim == 2 else w_sf,
        alpha,
        torch.bfloat16,
        backend="cutlass",
    )
    return silu_and_mul(h)


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
    parser.add_argument("--no-baseline", action="store_true")
    parser.add_argument("--json", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda:0")
    generator = torch.Generator(device=device).manual_seed(4532)
    model = synthetic_model(device, generator)
    if args.variant == "fp8":
        fused_weights = prepare_minimax_h3_fc1_weight_fp8(model["fc1_weight"])
        baseline_weights = quantize_fp8_rows_linear(model["fc1_weight"])
        g_w = act_global_scale = alpha = None
    else:
        g_w = minimax_h3_nvfp4_global_scale(model["fc1_weight"])
        # Dispatches to the SM120 layout on a compute capability 12.x device.
        fused_weights = prepare_minimax_h3_fc1_weight_nvfp4(model["fc1_weight"], g_w)
        baseline_weights = fp4_quantize(
            model["fc1_weight"],
            g_w,
            sf_vec_size=MINIMAX_H3_SF_BLOCK,
            sf_use_ue8m0=False,
            is_sf_swizzled_layout=True,
        )
        # Static activation global scale for a calibrated |a| <= 8 (synthetic model).
        act_global_scale = torch.tensor(
            [E4M3_MAX * 6.0 / 8.0], dtype=torch.float32, device=device
        )
        alpha = minimax_h3_nvfp4_alpha(act_global_scale, g_w)

    rows_out = []
    for rows in args.m:
        inputs = synthetic_inputs(rows, device, generator)
        if args.variant == "fp8":
            fn = lambda: minimax_h3_fc1_swiglu_fp8(  # noqa: E731
                inputs["x"],
                model["x_norm_weight"],
                model["adaln_scale"],
                model["adaln_shift"],
                inputs["adaln_index"],
                fused_weights[0],
                fused_weights[1],
            )
        else:
            alpha_value = float(alpha.item())
            fn = lambda: minimax_h3_fc1_swiglu_nvfp4(  # noqa: E731
                inputs["x"],
                model["x_norm_weight"],
                model["adaln_scale"],
                model["adaln_shift"],
                inputs["adaln_index"],
                act_global_scale,
                fused_weights[0],
                fused_weights[1],
                alpha_value,
            )
        fn()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        fused_ms = time_ms(fn, args.warmup, args.iters)
        flops = 2.0 * rows * MINIMAX_H3_HIDDEN * MINIMAX_H3_FC1_ROWS
        row = {
            "variant": args.variant,
            "M": rows,
            "fused_ms": fused_ms,
            "fused_tflops": flops / fused_ms / 1.0e9,
            "fused_peak_gib": torch.cuda.max_memory_allocated() / 2**30,
        }
        if not args.no_baseline:
            try:
                if args.variant == "fp8":
                    base = lambda: baseline_fp8(inputs, model, baseline_weights)  # noqa: E731
                else:
                    base = lambda: baseline_nvfp4(  # noqa: E731
                        inputs, model, baseline_weights, act_global_scale, alpha
                    )
                base()
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                row["baseline_ms"] = time_ms(base, args.warmup, args.iters)
                row["baseline_peak_gib"] = torch.cuda.max_memory_allocated() / 2**30
                row["speedup"] = row["baseline_ms"] / fused_ms
            except torch.OutOfMemoryError:
                row["baseline_ms"] = None
                row["baseline_status"] = "memory_limited"
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
