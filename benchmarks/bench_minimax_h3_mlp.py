# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Benchmark the MiniMax-H3 full MLP block (BF16 / MXFP8 / NVFP4) on SM100a / SM103a.

Candidate: ``minimax_h3_mlp`` / ``minimax_h3_mlp_mxfp8`` / ``minimax_h3_mlp_nvfp4`` (three
launches: norm + AdaLN, FC1 + SwiGLU with the FC2-ready epilogue, FC2 + gate + residual).

Baseline: the segmented chain built from FlashInfer's own pieces -- ``minimax_h3_fc1_swiglu*``
(norm + FC1 + SwiGLU), for the quantized variants ``mxfp8_quantize`` / ``nvfp4_quantize`` of
``y``, the FC2 GEMM through ``torch.nn.functional.linear`` (cuBLAS, BF16) or
``flashinfer.mm_mxfp8`` / ``flashinfer.mm_fp4`` (``--fc2-backend``), and the indexed gate +
residual epilogue in torch (two BF16 round points).  The candidate's output is validated
against that chain with the operator test's budget rule before timing.

The default suite measures the 5-second P8 production center. ``--suite final`` measures all
24 duration/parallelism centers. JIT compilation, tensor creation, weight preparation and the
static NVFP4 global-scale calibration are outside the timing boundary.

``--operand-layout contract`` (default) uses contiguous ``[9, 5376]`` tables and int64 indices.
``--operand-layout engine`` uses the engine's own operands: the shift / scale / gate tables are
the column chunks 3 / 4 / 5 of ONE ``[rows, 6 * 5376]`` modulation projection (row stride
``6 * 5376``; ``--adaln-rows``, default 6) addressed by one int64 ``combined_indices`` table.
``--cuda-graph`` replays each arm through a CUDA graph inside the CUPTI span so the host launch
gaps of the multi-kernel arms do not count.
"""

import argparse
import math

import numpy as np
import torch
import torch.nn.functional as F

from flashinfer import SfLayout, mm_fp4, mm_mxfp8, mxfp8_quantize, nvfp4_quantize
from flashinfer.diffusion_ops import (
    minimax_h3_fc1_swiglu,
    minimax_h3_fc1_swiglu_mxfp8,
    minimax_h3_fc1_swiglu_nvfp4,
    minimax_h3_mlp,
    minimax_h3_mlp_mxfp8,
    minimax_h3_mlp_nvfp4,
    prepare_minimax_h3_fc1_weight_mxfp8,
    prepare_minimax_h3_fc1_weight_nvfp4,
    prepare_minimax_h3_fc2_weight_mxfp8,
    prepare_minimax_h3_fc2_weight_nvfp4,
)
from flashinfer.diffusion_ops.minimax_h3_fc1_swiglu import (
    minimax_h3_nvfp4_alpha,
    minimax_h3_nvfp4_global_scale,
    mxfp8_activation_scale_workspace_bytes,
    nvfp4_activation_scale_workspace_bytes,
)
from flashinfer.diffusion_ops.minimax_h3_mlp import (
    NVFP4_A_PACKED_COLS,
    NVFP4_Y_PACKED_COLS,
    mxfp8_y_scale_workspace_bytes,
    nvfp4_y_scale_workspace_bytes,
)
from flashinfer.testing.utils import bench_gpu_time

HIDDEN = 5376
FFN = 14336
FC1_ROWS = 2 * FFN
ADALN_ROWS = 9
ENGINE_TABLE_CHUNKS = 6
ENGINE_SHIFT_CHUNK, ENGINE_SCALE_CHUNK, ENGINE_GATE_CHUNK = 3, 4, 5
EPS = 1.0e-5
VARIANTS = ("bf16", "mxfp8", "nvfp4")

# Operator test rule: elements outside atol + rtol * |ref| are budgeted (accumulation-order flips
# of the three BF16 round points), a wrong tile produces thousands.
ATOL = 1e-2
RTOL = 1e-2
MAX_VIOLATION_FRACTION = 2.0e-7
MAX_VIOLATIONS_FLOOR = 4

CENTER_SHAPES = [
    (4, 33472, 1),
    (4, 16736, 2),
    (4, 8368, 4),
    (4, 4184, 8),
    (5, 38592, 1),
    (5, 19296, 2),
    (5, 9648, 4),
    (5, 4824, 8),
    (6, 48768, 1),
    (6, 24384, 2),
    (6, 12192, 4),
    (6, 6096, 8),
    (8, 58944, 1),
    (8, 29472, 2),
    (8, 14736, 4),
    (8, 7368, 8),
    (10, 74240, 1),
    (10, 37120, 2),
    (10, 18560, 4),
    (10, 9280, 8),
    (15, 109952, 1),
    (15, 54976, 2),
    (15, 27488, 4),
    (15, 13744, 8),
]
ACTIVE_SHAPES = [(5, 4824, 8)]


def _make_model(device: torch.device, *, layout: str, adaln_rows: int):
    generator = torch.Generator(device=device)
    generator.manual_seed(4612)

    def normal(shape, std):
        out = torch.empty(shape, dtype=torch.bfloat16, device=device)
        return out.normal_(0.0, std, generator=generator)

    def uniform(shape, low, high):
        out = torch.empty(shape, dtype=torch.bfloat16, device=device)
        return out.uniform_(low, high, generator=generator)

    if layout == "contract":
        adaln_scale = uniform((adaln_rows, HIDDEN), -0.05, 0.05)
        adaln_shift = uniform((adaln_rows, HIDDEN), -0.05, 0.05)
        gate = uniform((adaln_rows, HIDDEN), -1.0, 1.0)
        backing = None
    else:
        # Column chunks 3 (shift_mlp), 4 (scale_mlp) and 5 (gate_mlp) of the modulation projection.
        backing = uniform((adaln_rows, ENGINE_TABLE_CHUNKS * HIDDEN), -0.05, 0.05)

        def chunk(index):
            return backing[:, index * HIDDEN : (index + 1) * HIDDEN]

        adaln_shift = chunk(ENGINE_SHIFT_CHUNK)
        adaln_scale = chunk(ENGINE_SCALE_CHUNK)
        gate = chunk(ENGINE_GATE_CHUNK)
        gate.uniform_(-1.0, 1.0, generator=generator)
        assert gate.stride() == (ENGINE_TABLE_CHUNKS * HIDDEN, 1)

    return {
        "x_norm_weight": uniform((HIDDEN,), 0.9, 1.1),
        "adaln_scale": adaln_scale,
        "adaln_shift": adaln_shift,
        "gate": gate,
        "fc1_weight": normal((FC1_ROWS, HIDDEN), 0.02),
        "fc2_weight": normal((HIDDEN, FFN), 0.01),
        "_backing": backing,
    }


def _prepare_weights(model, variant: str):
    """Offline weight preparation of the candidate (combined scale tiles) and of the baseline's
    library FC2 (FlashInfer's own quantizers, 128x4 swizzled scales)."""
    if variant == "bf16":
        return {}
    if variant == "mxfp8":
        w1_q, w1_tiles = prepare_minimax_h3_fc1_weight_mxfp8(model["fc1_weight"])
        w2_q, w2_tiles = prepare_minimax_h3_fc2_weight_mxfp8(model["fc2_weight"])
        w2_mx, w2_sf = mxfp8_quantize(model["fc2_weight"], is_sf_swizzled_layout=True)
        return {
            "fc1_weight_q": w1_q,
            "fc1_scale_tiles": w1_tiles,
            "fc2_weight_q": w2_q,
            "fc2_scale_tiles": w2_tiles,
            "fc2_library_q": w2_mx,
            "fc2_library_sf": w2_sf,
        }
    g_w1 = minimax_h3_nvfp4_global_scale(model["fc1_weight"])
    g_w2 = minimax_h3_nvfp4_global_scale(model["fc2_weight"])
    w1_q, w1_tiles = prepare_minimax_h3_fc1_weight_nvfp4(model["fc1_weight"], g_w1)
    w2_q, w2_tiles = prepare_minimax_h3_fc2_weight_nvfp4(model["fc2_weight"], g_w2)
    w2_fp4, w2_sf = nvfp4_quantize(
        model["fc2_weight"], g_w2, sfLayout=SfLayout.layout_128x4, do_shuffle=False
    )
    return {
        "fc1_weight_q": w1_q,
        "fc1_scale_tiles": w1_tiles,
        "fc2_weight_q": w2_q,
        "fc2_scale_tiles": w2_tiles,
        "fc2_library_q": w2_fp4,
        "fc2_library_sf": w2_sf,
        "w1_global_scale": g_w1,
        "w2_global_scale": g_w2,
    }


def _reference_modulated(x, x_norm_weight, adaln_scale, adaln_shift, adaln_index):
    norm = F.rms_norm(x, (HIDDEN,), x_norm_weight, eps=EPS).to(torch.bfloat16)
    rows = int(adaln_scale.shape[0])
    idx = adaln_index.long()
    valid = (idx >= 0) & (idx < rows)
    safe = idx.clamp(0, rows - 1)
    a = torch.addcmul(
        adaln_shift.index_select(0, safe),
        norm,
        (adaln_scale.index_select(0, safe) + 1.0).to(torch.bfloat16),
    ).to(torch.bfloat16)
    return torch.where(valid[:, None], a, torch.zeros_like(a))


def _make_case(m: int, p: int, model, weights, variant: str, device, *, backend):
    generator = torch.Generator(device=device)
    generator.manual_seed(4612 + m + p)
    x = torch.empty((m, HIDDEN), dtype=torch.bfloat16, device=device)
    x.normal_(0.0, 0.5, generator=generator)
    residual = torch.empty((m, HIDDEN), dtype=torch.bfloat16, device=device)
    residual.normal_(0.0, 1.0, generator=generator)
    adaln_rows = model["adaln_scale"].shape[0]
    rows = torch.arange(m, dtype=torch.int64, device=device)
    adaln_index = torch.div(rows * adaln_rows, m, rounding_mode="floor").clamp_max(
        adaln_rows - 1
    )
    case = {
        **model,
        **weights,
        "variant": variant,
        "backend": backend,
        "x": x,
        "adaln_index": adaln_index,
        "residual": residual,
        "out": torch.empty((m, HIDDEN), dtype=torch.bfloat16, device=device),
        "baseline_out": torch.empty((m, HIDDEN), dtype=torch.bfloat16, device=device),
        "baseline_y": torch.empty((m, FFN), dtype=torch.bfloat16, device=device),
    }
    if variant == "bf16":
        case["workspace_a"] = torch.empty(
            (m, HIDDEN), dtype=torch.bfloat16, device=device
        )
        case["workspace_y"] = torch.empty((m, FFN), dtype=torch.bfloat16, device=device)
        case["baseline_a"] = torch.empty(
            (m, HIDDEN), dtype=torch.bfloat16, device=device
        )
    elif variant == "mxfp8":
        a_sf_bytes = mxfp8_activation_scale_workspace_bytes(m)
        case["workspace_a_q"] = torch.empty(
            (m, HIDDEN), dtype=torch.float8_e4m3fn, device=device
        )
        case["workspace_a_sf"] = torch.zeros(
            (a_sf_bytes,), dtype=torch.uint8, device=device
        )
        case["workspace_y_q"] = torch.empty(
            (m, FFN), dtype=torch.float8_e4m3fn, device=device
        )
        case["workspace_y_sf"] = torch.zeros(
            (mxfp8_y_scale_workspace_bytes(m),), dtype=torch.uint8, device=device
        )
        case["baseline_a_q"] = torch.empty(
            (m, HIDDEN), dtype=torch.float8_e4m3fn, device=device
        )
        case["baseline_a_sf"] = torch.zeros(
            (a_sf_bytes,), dtype=torch.uint8, device=device
        )
    else:
        a_sf_bytes = nvfp4_activation_scale_workspace_bytes(m)
        case["workspace_a_q"] = torch.empty(
            (m, NVFP4_A_PACKED_COLS), dtype=torch.uint8, device=device
        )
        case["workspace_a_sf"] = torch.zeros(
            (a_sf_bytes,), dtype=torch.uint8, device=device
        )
        case["workspace_y_q"] = torch.empty(
            (m, NVFP4_Y_PACKED_COLS), dtype=torch.uint8, device=device
        )
        case["workspace_y_sf"] = torch.zeros(
            (nvfp4_y_scale_workspace_bytes(m),), dtype=torch.uint8, device=device
        )
        case["baseline_a_q"] = torch.empty(
            (m, NVFP4_A_PACKED_COLS), dtype=torch.uint8, device=device
        )
        case["baseline_a_sf"] = torch.zeros(
            (a_sf_bytes,), dtype=torch.uint8, device=device
        )
        # Static global scales calibrated outside the timed region (as the engine would): a from
        # the reference modulated activation, y from the FC1 operator's output under that scale.
        a_ref = _reference_modulated(
            x,
            model["x_norm_weight"],
            model["adaln_scale"],
            model["adaln_shift"],
            adaln_index,
        )
        g_a = minimax_h3_nvfp4_global_scale(a_ref)
        del a_ref
        alpha1 = minimax_h3_nvfp4_alpha(g_a, weights["w1_global_scale"])
        y = minimax_h3_fc1_swiglu_nvfp4(
            x,
            model["x_norm_weight"],
            model["adaln_scale"],
            model["adaln_shift"],
            adaln_index,
            g_a,
            weights["fc1_weight_q"],
            weights["fc1_scale_tiles"],
            alpha1,
            out=case["baseline_y"],
            workspace_q=case["baseline_a_q"],
            workspace_sf=case["baseline_a_sf"],
        )
        g_y = minimax_h3_nvfp4_global_scale(y)
        case.update(
            a_global_scale=g_a,
            alpha1=alpha1,
            y_global_scale=g_y,
            alpha2=minimax_h3_nvfp4_alpha(g_y, weights["w2_global_scale"]),
        )
    return case


def _torch_epilogue(o, gate, adaln_index, residual, out):
    """``out = BF16(residual + BF16(gate[idx] * o))`` with ``gate = 0`` for an index outside the table."""
    rows = int(gate.shape[0])
    valid = (adaln_index >= 0) & (adaln_index < rows)
    g = gate.index_select(0, adaln_index.clamp(0, rows - 1))
    g = torch.where(valid[:, None], g, torch.zeros_like(g))
    p = (g * o).to(torch.bfloat16)
    torch.add(residual, p, out=out)
    return out


def _segmented_baseline(case):
    variant = case["variant"]
    norm_args = (
        case["x"],
        case["x_norm_weight"],
        case["adaln_scale"],
        case["adaln_shift"],
        case["adaln_index"],
    )
    if variant == "bf16":
        y = minimax_h3_fc1_swiglu(
            *norm_args,
            case["fc1_weight"],
            out=case["baseline_y"],
            workspace=case["baseline_a"],
            eps=EPS,
        )
        o = F.linear(y, case["fc2_weight"])
    elif variant == "mxfp8":
        y = minimax_h3_fc1_swiglu_mxfp8(
            *norm_args,
            case["fc1_weight_q"],
            case["fc1_scale_tiles"],
            out=case["baseline_y"],
            workspace_q=case["baseline_a_q"],
            workspace_sf=case["baseline_a_sf"],
            eps=EPS,
        )
        y_q, y_sf = mxfp8_quantize(y, is_sf_swizzled_layout=True)
        o = mm_mxfp8(
            y_q,
            case["fc2_library_q"].t(),
            y_sf,
            case["fc2_library_sf"],
            out_dtype=torch.bfloat16,
            backend=case["backend"],
        )
    else:
        y = minimax_h3_fc1_swiglu_nvfp4(
            *norm_args,
            case["a_global_scale"],
            case["fc1_weight_q"],
            case["fc1_scale_tiles"],
            case["alpha1"],
            out=case["baseline_y"],
            workspace_q=case["baseline_a_q"],
            workspace_sf=case["baseline_a_sf"],
            eps=EPS,
        )
        y_q, y_sf = nvfp4_quantize(
            y,
            case["y_global_scale"],
            sfLayout=SfLayout.layout_128x4,
            do_shuffle=False,
        )
        o = mm_fp4(
            y_q,
            case["fc2_library_q"].T,
            y_sf,
            case["fc2_library_sf"].T,
            case["alpha2"],
            torch.bfloat16,
            None,
            block_size=16,
            backend=case["backend"],
        )
    return _torch_epilogue(
        o, case["gate"], case["adaln_index"], case["residual"], case["baseline_out"]
    )


def _run_candidate(case):
    variant = case["variant"]
    norm_args = (
        case["x"],
        case["x_norm_weight"],
        case["adaln_scale"],
        case["adaln_shift"],
        case["adaln_index"],
    )
    if variant == "bf16":
        return minimax_h3_mlp(
            *norm_args,
            case["fc1_weight"],
            case["fc2_weight"],
            case["gate"],
            case["residual"],
            out=case["out"],
            workspace_a=case["workspace_a"],
            workspace_y=case["workspace_y"],
            eps=EPS,
        )
    if variant == "mxfp8":
        return minimax_h3_mlp_mxfp8(
            *norm_args,
            case["fc1_weight_q"],
            case["fc1_scale_tiles"],
            case["fc2_weight_q"],
            case["fc2_scale_tiles"],
            case["gate"],
            case["residual"],
            out=case["out"],
            workspace_a_q=case["workspace_a_q"],
            workspace_a_sf=case["workspace_a_sf"],
            workspace_y_q=case["workspace_y_q"],
            workspace_y_sf=case["workspace_y_sf"],
            eps=EPS,
        )
    return minimax_h3_mlp_nvfp4(
        *norm_args,
        case["a_global_scale"],
        case["fc1_weight_q"],
        case["fc1_scale_tiles"],
        case["alpha1"],
        case["y_global_scale"],
        case["fc2_weight_q"],
        case["fc2_scale_tiles"],
        case["alpha2"],
        case["gate"],
        case["residual"],
        out=case["out"],
        workspace_a_q=case["workspace_a_q"],
        workspace_a_sf=case["workspace_a_sf"],
        workspace_y_q=case["workspace_y_q"],
        workspace_y_sf=case["workspace_y_sf"],
        eps=EPS,
    )


def _validate(actual, expected, label):
    diff = (actual.float() - expected.float()).abs()
    bad = int((diff > (ATOL + RTOL * expected.float().abs())).sum().item())
    budget = max(
        MAX_VIOLATIONS_FLOOR, int(math.ceil(MAX_VIOLATION_FRACTION * diff.numel()))
    )
    if not torch.isfinite(actual.float()).all():
        raise RuntimeError(f"{label}: non-finite candidate output")
    if bad > budget:
        raise RuntimeError(
            f"{label}: {bad} elements outside atol={ATOL} rtol={RTOL} (budget {budget}), "
            f"max |err| {float(diff.max().item()):.4g}"
        )
    return bad


def _median_ms(fn, *, cuda_graph: bool):
    times = bench_gpu_time(
        fn,
        enable_cupti=True,
        dry_run_iters=10,
        repeat_iters=100,
        use_cuda_graph=cuda_graph,
    )
    return float(np.median(times))


def _bench_shape(
    duration, m, p, model, weights, variant, device, *, backend, cuda_graph
):
    case = _make_case(m, p, model, weights, variant, device, backend=backend)
    expected = _segmented_baseline(case)
    actual = _run_candidate(case)
    torch.cuda.synchronize()
    violations = _validate(actual, expected, f"{variant} M={m} P={p}")

    baseline_ms = _median_ms(lambda: _segmented_baseline(case), cuda_graph=cuda_graph)
    candidate_ms = _median_ms(lambda: _run_candidate(case), cuda_graph=cuda_graph)
    gemm_tflops = (
        2.0 * m * (HIDDEN * FC1_ROWS + FFN * HIDDEN) / (candidate_ms * 1e-3) / 1e12
    )
    return {
        "duration": duration,
        "M": m,
        "P": p,
        "variant": variant,
        "baseline_ms": baseline_ms,
        "candidate_ms": candidate_ms,
        "speedup": baseline_ms / candidate_ms,
        "tflops": gemm_tflops,
        "violations": violations,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark the SM100a / SM103a MiniMax-H3 full MLP block"
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--suite", choices=("active", "final"), default="active")
    parser.add_argument(
        "--variant",
        choices=VARIANTS + ("all",),
        default="bf16",
        help="operator variant",
    )
    parser.add_argument(
        "--operand-layout",
        choices=("contract", "engine"),
        default="contract",
        help="contract: contiguous tables; engine: shift / scale / gate column chunks of one "
        "[rows, 6 * 5376] projection with one int64 index",
    )
    parser.add_argument(
        "--adaln-rows",
        type=int,
        default=None,
        help="table rows (default 9 for contract, 6 for engine)",
    )
    parser.add_argument(
        "--fc2-backend",
        default="auto",
        help="flashinfer.mm_mxfp8 / mm_fp4 backend of the segmented baseline's FC2 (quantized variants)",
    )
    parser.add_argument(
        "--cuda-graph",
        action="store_true",
        help="replay each arm through a CUDA graph inside the CUPTI span (no host launch gaps)",
    )
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    if torch.cuda.get_device_capability(device) not in {(10, 0), (10, 3)}:
        raise RuntimeError("This benchmark requires compute capability 10.0 or 10.3")

    layout = args.operand_layout
    adaln_rows = args.adaln_rows or (ADALN_ROWS if layout == "contract" else 6)
    shapes = ACTIVE_SHAPES if args.suite == "active" else CENTER_SHAPES
    variants = VARIANTS if args.variant == "all" else (args.variant,)
    model = _make_model(device, layout=layout, adaln_rows=adaln_rows)
    print(
        f"GPU: {torch.cuda.get_device_name(device)}, operand layout: {layout}, "
        f"table rows: {adaln_rows}, FC2 library backend: {args.fc2_backend}, "
        f"CUDA graph: {args.cuda_graph}"
    )
    print(
        f"{'variant':>7} {'duration':>8} {'M':>8} {'P':>3} {'baseline ms':>13} {'fused ms':>10} "
        f"{'speedup':>9} {'TFLOPS':>8} {'viol':>5}"
    )
    for variant in variants:
        weights = _prepare_weights(model, variant)
        for duration, m, p in shapes:
            result = _bench_shape(
                duration,
                m,
                p,
                model,
                weights,
                variant,
                device,
                backend=args.fc2_backend,
                cuda_graph=args.cuda_graph,
            )
            print(
                f"{variant:>7} {duration:>7}s {m:>8} {p:>3} "
                f"{result['baseline_ms']:>13.6f} {result['candidate_ms']:>10.6f} "
                f"{result['speedup']:>8.4f}x {result['tflops']:>8.1f} {result['violations']:>5}"
            )
        del weights
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
