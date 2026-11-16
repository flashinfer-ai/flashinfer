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

Benchmark the SM120 FP8 / NVFP4 fused MiniMax-H3 pre-attention against a segmented
torch / FlashInfer chain (norm + AdaLN + quantization -> GEMM -> Q/K RMSNorm + RoPE).

    python benchmarks/bench_minimax_h3_sm120_quant_pre_attention.py --variant nvfp4 --m 33472 38592
"""

from __future__ import annotations

import argparse
import json

import torch

from flashinfer import mm_fp4
from flashinfer.diffusion_ops import (
    minimax_h3_fp8_pre_attention,
    minimax_h3_nvfp4_pre_attention,
    quantize_minimax_h3_qkv_weight_fp8,
    quantize_minimax_h3_qkv_weight_nvfp4,
)
from flashinfer.diffusion_ops.cake_minimax_h3_sm120_quant_pre_attention import (
    MINIMAX_H3_DEFAULT_EPS,
    MINIMAX_H3_HEAD_DIM,
    MINIMAX_H3_HIDDEN,
    MINIMAX_H3_NUM_HEADS,
    MINIMAX_H3_QKV_WIDTH,
    MINIMAX_H3_ROPE_DIM,
    MINIMAX_H3_SF_BLOCK,
    fp8_scale_from_amax,
    nvfp4_global_scale_from_amax,
)
from flashinfer.quantization import fp4_quantize
from flashinfer.testing.utils import bench_gpu_time

ADALN_ROWS = 9
# Representative MiniMax-H3 token counts (video DiT frames x 64 x 64 latents, batch 1, SP1).
DEFAULT_ROWS = (33472, 38592, 48768, 58944, 74240, 109952)


def synthetic_model(device: torch.device, generator: torch.Generator) -> dict:
    bf16 = torch.bfloat16

    def uniform(shape, low, high):
        return torch.empty(shape, dtype=bf16, device=device).uniform_(
            low, high, generator=generator
        )

    qkv_weight = torch.empty(
        (MINIMAX_H3_QKV_WIDTH, MINIMAX_H3_HIDDEN), dtype=bf16, device=device
    )
    qkv_weight.normal_(mean=0.0, std=0.01, generator=generator)
    return {
        "qkv_weight": qkv_weight,
        "x_norm_weight": uniform((MINIMAX_H3_HIDDEN,), 0.9, 1.1),
        "adaln_scale": uniform((ADALN_ROWS, MINIMAX_H3_HIDDEN), -0.05, 0.05),
        "adaln_shift": uniform((ADALN_ROWS, MINIMAX_H3_HIDDEN), -0.05, 0.05),
        "q_norm_weight": uniform((MINIMAX_H3_HEAD_DIM,), 0.9, 1.1),
        "k_norm_weight": uniform((MINIMAX_H3_HEAD_DIM,), 0.9, 1.1),
    }


def synthetic_inputs(
    rows: int, device: torch.device, generator: torch.Generator
) -> dict:
    x = torch.empty((rows, MINIMAX_H3_HIDDEN), dtype=torch.bfloat16, device=device)
    x.normal_(mean=0.0, std=0.5, generator=generator)
    positions = torch.arange(rows, device=device, dtype=torch.int64)
    adaln_index = (
        torch.div(positions * ADALN_ROWS, max(rows, 1), rounding_mode="floor")
        .clamp_max(ADALN_ROWS - 1)
        .to(torch.int32)
    )
    angles = (
        torch.rand((rows, MINIMAX_H3_ROPE_DIM // 2), device=device, generator=generator)
        * 6.283185
    )
    rope_cos_sin = (
        torch.cat((angles.cos(), angles.sin()), dim=-1).to(torch.bfloat16).contiguous()
    )
    return {"x": x, "adaln_index": adaln_index, "rope_cos_sin": rope_cos_sin}


def torch_pre_norm(x, model, adaln_index):
    norm = torch.nn.functional.rms_norm(
        x, (MINIMAX_H3_HIDDEN,), model["x_norm_weight"], eps=MINIMAX_H3_DEFAULT_EPS
    )
    index = adaln_index.to(torch.int64)
    scale = model["adaln_scale"].index_select(0, index)
    shift = model["adaln_shift"].index_select(0, index)
    return torch.addcmul(
        shift, norm.to(torch.bfloat16), (scale + 1.0).to(torch.bfloat16)
    ).to(torch.bfloat16)


def torch_post(qkv, model, rope_cos_sin):
    rows = qkv.shape[0]
    grouped = qkv.view(rows, MINIMAX_H3_NUM_HEADS, 3, MINIMAX_H3_HEAD_DIM)
    q = torch.nn.functional.rms_norm(
        grouped[:, :, 0, :],
        (MINIMAX_H3_HEAD_DIM,),
        model["q_norm_weight"],
        eps=MINIMAX_H3_DEFAULT_EPS,
    )
    k = torch.nn.functional.rms_norm(
        grouped[:, :, 1, :],
        (MINIMAX_H3_HEAD_DIM,),
        model["k_norm_weight"],
        eps=MINIMAX_H3_DEFAULT_EPS,
    )
    half = MINIMAX_H3_ROPE_DIM // 2
    cos = torch.cat((rope_cos_sin[:, :half], rope_cos_sin[:, :half]), dim=-1)[
        :, None, :
    ].float()
    sin = torch.cat((rope_cos_sin[:, half:], rope_cos_sin[:, half:]), dim=-1)[
        :, None, :
    ].float()

    def rope(t):
        rotary = t[..., :MINIMAX_H3_ROPE_DIM].float()
        rotated_half = torch.cat((-rotary[..., half:], rotary[..., :half]), dim=-1)
        return torch.cat(
            (
                (rotary * cos + rotated_half * sin).to(torch.bfloat16),
                t[..., MINIMAX_H3_ROPE_DIM:],
            ),
            dim=-1,
        )

    return (
        rope(q.to(torch.bfloat16)).contiguous(),
        rope(k.to(torch.bfloat16)).contiguous(),
        grouped[:, :, 2, :].contiguous(),
    )


def baseline_fp8(inputs, model, weights):
    w_q, w_scale = weights
    a = torch_pre_norm(inputs["x"], model, inputs["adaln_index"])
    scale = fp8_scale_from_amax(a.float().abs().amax(dim=1).clamp_min(1e-12))
    a_q = (a.float() / scale[:, None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    qkv = torch._scaled_mm(
        a_q,
        w_q.t(),
        scale_a=scale[:, None],
        scale_b=w_scale[None, :],
        out_dtype=torch.bfloat16,
    )
    return torch_post(qkv, model, inputs["rope_cos_sin"])


def baseline_nvfp4(inputs, model, weights, act_global_scale, alpha):
    w_q, w_sf, _w_gs = weights
    a = torch_pre_norm(inputs["x"], model, inputs["adaln_index"])
    a_q, a_sf = fp4_quantize(
        a,
        act_global_scale,
        sf_vec_size=MINIMAX_H3_SF_BLOCK,
        sf_use_ue8m0=False,
        is_sf_swizzled_layout=True,
    )
    qkv = mm_fp4(
        a_q,
        w_q.t(),
        a_sf,
        w_sf.t() if w_sf.ndim == 2 else w_sf,
        alpha,
        torch.bfloat16,
        backend="cutlass",
    )
    return torch_post(qkv, model, inputs["rope_cos_sin"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--variant", choices=("fp8", "nvfp4"), default="nvfp4")
    parser.add_argument("--m", type=int, nargs="+", default=list(DEFAULT_ROWS))
    parser.add_argument("--out-mode", choices=("bf16", "e4m3", "nvfp4"), default="bf16")
    parser.add_argument("--no-baseline", action="store_true")
    parser.add_argument("--json", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda:0")
    generator = torch.Generator(device=device).manual_seed(4532)
    model = synthetic_model(device, generator)
    if args.variant == "fp8":
        weights = quantize_minimax_h3_qkv_weight_fp8(model["qkv_weight"])
    else:
        weights = quantize_minimax_h3_qkv_weight_nvfp4(model["qkv_weight"])
    act_global_scale = nvfp4_global_scale_from_amax(8.0).to(device)
    out_scales = {}
    if args.out_mode == "e4m3":
        out_scales = {f"{n}_descale": 8.0 / 448.0 for n in "qkv"}
    elif args.out_mode == "nvfp4":
        out_scales = {
            f"{n}_global_scale": nvfp4_global_scale_from_amax(8.0).to(device)
            for n in "qkv"
        }

    rows_out = []
    for rows in args.m:
        inputs = synthetic_inputs(rows, device, generator)
        if args.variant == "fp8":
            fn = lambda: minimax_h3_fp8_pre_attention(  # noqa: E731
                inputs["x"],
                model["x_norm_weight"],
                model["adaln_scale"],
                model["adaln_shift"],
                inputs["adaln_index"],
                weights[0],
                weights[1],
                model["q_norm_weight"],
                model["k_norm_weight"],
                inputs["rope_cos_sin"],
                out_mode=args.out_mode,
                **out_scales,
            )
        else:
            alpha = 1.0 / (float(act_global_scale.item()) * float(weights[2].item()))
            fn = lambda: minimax_h3_nvfp4_pre_attention(  # noqa: E731
                inputs["x"],
                model["x_norm_weight"],
                model["adaln_scale"],
                model["adaln_shift"],
                inputs["adaln_index"],
                weights[0],
                weights[1],
                weights[2],
                act_global_scale,
                model["q_norm_weight"],
                model["k_norm_weight"],
                inputs["rope_cos_sin"],
                out_mode=args.out_mode,
                alpha=alpha,
                **out_scales,
            )
        fn()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        times = bench_gpu_time(fn, cold_l2_cache=True)
        kernel_ms = float(sorted(times)[len(times) // 2])
        peak_gib = torch.cuda.max_memory_allocated() / 2**30
        row = {
            "variant": args.variant,
            "out_mode": args.out_mode,
            "M": rows,
            "fused_ms": kernel_ms,
            "fused_peak_gib": peak_gib,
        }
        if not args.no_baseline and args.out_mode == "bf16":
            try:
                if args.variant == "fp8":
                    base = lambda: baseline_fp8(inputs, model, weights)  # noqa: E731
                else:
                    alpha_t = torch.tensor([alpha], dtype=torch.float32, device=device)
                    base = lambda: baseline_nvfp4(
                        inputs, model, weights, act_global_scale, alpha_t
                    )  # noqa: E731
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
