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

Benchmark the SM120 FP8 / NVFP4 fused MiniMax-H3 output projection against a segmented
torch / FlashInfer chain (quantization -> GEMM -> torch gated residual).

    python benchmarks/bench_minimax_h3_sm120_quant_out_proj.py --variant nvfp4 --m 33472 38592
"""

from __future__ import annotations

import argparse
import json

import torch

from flashinfer import mm_fp4
from flashinfer.diffusion_ops import (
    minimax_h3_fp8_out_proj,
    minimax_h3_nvfp4_out_proj,
    quantize_minimax_h3_o_weight_fp8,
    quantize_minimax_h3_o_weight_nvfp4,
)
from flashinfer.diffusion_ops.cake_minimax_h3_sm120_quant_out_proj import (
    MINIMAX_H3_ATTN_DIM,
    MINIMAX_H3_GATE_ROWS,
    MINIMAX_H3_HIDDEN,
    MINIMAX_H3_SF_BLOCK,
    fp8_scale_from_amax,
    nvfp4_global_scale_from_amax,
)
from flashinfer.quantization import fp4_quantize
from flashinfer.testing.utils import bench_gpu_time

# Representative MiniMax-H3 token counts (video DiT frames x 64 x 64 latents, batch 1, SP1).
DEFAULT_ROWS = (33472, 38592, 48768, 58944, 74240, 109952)


def synthetic_model(device: torch.device, generator: torch.Generator) -> dict:
    o_weight = torch.empty(
        (MINIMAX_H3_HIDDEN, MINIMAX_H3_ATTN_DIM), dtype=torch.bfloat16, device=device
    )
    o_weight.normal_(mean=0.0, std=0.01, generator=generator)
    gate = torch.empty(
        (MINIMAX_H3_GATE_ROWS, MINIMAX_H3_HIDDEN), dtype=torch.bfloat16, device=device
    )
    gate.uniform_(-1.0, 1.0, generator=generator)
    return {"o_weight": o_weight, "gate": gate}


def synthetic_inputs(
    rows: int, device: torch.device, generator: torch.Generator
) -> dict:
    attn_out = torch.empty(
        (rows, MINIMAX_H3_ATTN_DIM), dtype=torch.bfloat16, device=device
    )
    attn_out.normal_(mean=0.0, std=0.5, generator=generator)
    residual = torch.empty(
        (rows, MINIMAX_H3_HIDDEN), dtype=torch.bfloat16, device=device
    )
    residual.normal_(mean=0.0, std=1.0, generator=generator)
    positions = torch.arange(rows, device=device, dtype=torch.int64)
    gate_index = (
        torch.div(positions * MINIMAX_H3_GATE_ROWS, max(rows, 1), rounding_mode="floor")
        .clamp_max(MINIMAX_H3_GATE_ROWS - 1)
        .to(torch.int32)
    )
    return {"attn_out": attn_out, "residual": residual, "gate_index": gate_index}


def torch_gate_residual(inputs: dict, model: dict, o: torch.Tensor) -> torch.Tensor:
    """SGLang ``_modulate_gate`` torch fallback: ``residual + gate[idx] * o`` in BF16."""

    index = inputs["gate_index"].to(torch.int64)
    return inputs["residual"] + model["gate"].index_select(0, index) * o


def baseline_fp8(inputs, model, weights):
    w_q, w_scale = weights
    a = inputs["attn_out"]
    scale = fp8_scale_from_amax(a.float().abs().amax(dim=1).clamp_min(1e-12))
    a_q = (a.float() / scale[:, None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    o = torch._scaled_mm(
        a_q,
        w_q.t(),
        scale_a=scale[:, None],
        scale_b=w_scale[None, :],
        out_dtype=torch.bfloat16,
    )
    return torch_gate_residual(inputs, model, o)


def baseline_nvfp4(inputs, model, weights, act_global_scale, alpha):
    w_q, w_sf, _w_gs = weights
    a_q, a_sf = fp4_quantize(
        inputs["attn_out"],
        act_global_scale,
        sf_vec_size=MINIMAX_H3_SF_BLOCK,
        sf_use_ue8m0=False,
        is_sf_swizzled_layout=True,
    )
    o = mm_fp4(
        a_q,
        w_q.t(),
        a_sf,
        w_sf.t() if w_sf.ndim == 2 else w_sf,
        alpha,
        torch.bfloat16,
        backend="cutlass",
    )
    return torch_gate_residual(inputs, model, o)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--variant", choices=("fp8", "nvfp4"), default="nvfp4")
    parser.add_argument("--m", type=int, nargs="+", default=list(DEFAULT_ROWS))
    parser.add_argument("--no-baseline", action="store_true")
    parser.add_argument("--json", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda:0")
    generator = torch.Generator(device=device).manual_seed(4616)
    model = synthetic_model(device, generator)
    if args.variant == "fp8":
        weights = quantize_minimax_h3_o_weight_fp8(model["o_weight"])
    else:
        weights = quantize_minimax_h3_o_weight_nvfp4(model["o_weight"])
    act_global_scale = nvfp4_global_scale_from_amax(4.0).to(device)

    rows_out = []
    for rows in args.m:
        inputs = synthetic_inputs(rows, device, generator)
        if args.variant == "fp8":
            fn = lambda: minimax_h3_fp8_out_proj(  # noqa: E731
                inputs["attn_out"],
                weights[0],
                weights[1],
                model["gate"],
                inputs["gate_index"],
                inputs["residual"],
            )
        else:
            alpha = 1.0 / (float(act_global_scale.item()) * float(weights[2].item()))
            fn = lambda: minimax_h3_nvfp4_out_proj(  # noqa: E731
                inputs["attn_out"],
                weights[0],
                weights[1],
                weights[2],
                act_global_scale,
                model["gate"],
                inputs["gate_index"],
                inputs["residual"],
            )
        fn()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        times = bench_gpu_time(fn, cold_l2_cache=True)
        kernel_ms = float(sorted(times)[len(times) // 2])
        peak_gib = torch.cuda.max_memory_allocated() / 2**30
        row = {
            "variant": args.variant,
            "M": rows,
            "fused_ms": kernel_ms,
            "fused_tflops": 2.0
            * rows
            * MINIMAX_H3_HIDDEN
            * MINIMAX_H3_ATTN_DIM
            / kernel_ms
            / 1e9,
            "fused_peak_gib": peak_gib,
        }
        if not args.no_baseline:
            try:
                if args.variant == "fp8":
                    base = lambda: baseline_fp8(inputs, model, weights)  # noqa: E731
                else:
                    alpha_t = torch.tensor([alpha], dtype=torch.float32, device=device)
                    base = lambda: baseline_nvfp4(  # noqa: E731
                        inputs, model, weights, act_global_scale, alpha_t
                    )
                base()
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                base_times = bench_gpu_time(base, cold_l2_cache=True)
                row["baseline_ms"] = float(sorted(base_times)[len(base_times) // 2])
                row["baseline_peak_gib"] = torch.cuda.max_memory_allocated() / 2**30
                row["speedup"] = row["baseline_ms"] / kernel_ms
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
