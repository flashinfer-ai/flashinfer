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
"""SM120 (GB202) FP8 / NVFP4 fused MiniMax-H3 pre-attention against an exact torch emulation.

The reference quantizes the BF16 normalized activation exactly like the kernel's stage 1
(per-token E4M3 with ``RN(amax / 448)`` or FlashInfer ``fp4_quantize`` block-16 NVFP4), runs the
dequantized GEMM in FP32, rounds to BF16, then applies the per-head Q/K RMSNorm and the partial
NeoX RoPE in FP32 with BF16 round points.  Kernel and emulation reduce the RMS statistics in
different orders, so single activations (and, rarely, a token's FP8 scale) may differ by one BF16
ulp; the tolerances below cover those flips.
"""

import pytest
import torch

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
from flashinfer.utils import get_compute_capability

ADALN_ROWS = 9
E4M3_MAX = 448.0
E2M1_MAX = 6.0
# Calibrated activation / output absmax used for the NVFP4 global scales (synthetic model).
ACT_AMAX = 8.0
OUT_AMAX = 8.0
# Tail lengths around the 128-row tile, plus multi-tile shapes.
ROWS = [1, 127, 128, 129, 257, 4097]


def _supported() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _minor = get_compute_capability(torch.device("cuda:0"))
    return major == 12


requires_sm120 = pytest.mark.skipif(
    not _supported(), reason="requires a CUDA GPU with compute capability 12.x (GB202)"
)


# --------------------------------------------------------------------------------------------
# Synthetic model + inputs
# --------------------------------------------------------------------------------------------
_MODEL = {}


def model_tensors(device: torch.device) -> dict:
    key = str(device)
    if key in _MODEL:
        return _MODEL[key]
    generator = torch.Generator(device=device).manual_seed(4532)
    bf16 = torch.bfloat16

    def uniform(shape, low, high):
        return torch.empty(shape, dtype=bf16, device=device).uniform_(
            low, high, generator=generator
        )

    qkv_weight = torch.empty(
        (MINIMAX_H3_QKV_WIDTH, MINIMAX_H3_HIDDEN), dtype=bf16, device=device
    )
    qkv_weight.normal_(mean=0.0, std=0.01, generator=generator)
    model = {
        "qkv_weight": qkv_weight,
        "x_norm_weight": uniform((MINIMAX_H3_HIDDEN,), 0.9, 1.1),
        "adaln_scale": uniform((ADALN_ROWS, MINIMAX_H3_HIDDEN), -0.05, 0.05),
        "adaln_shift": uniform((ADALN_ROWS, MINIMAX_H3_HIDDEN), -0.05, 0.05),
        "q_norm_weight": uniform((MINIMAX_H3_HEAD_DIM,), 0.9, 1.1),
        "k_norm_weight": uniform((MINIMAX_H3_HEAD_DIM,), 0.9, 1.1),
    }
    model["fp8"] = quantize_minimax_h3_qkv_weight_fp8(qkv_weight)
    model["nvfp4"] = quantize_minimax_h3_qkv_weight_nvfp4(qkv_weight)
    _MODEL[key] = model
    return model


def rope_cache(rows: int, device: torch.device) -> torch.Tensor:
    """3-D RoPE table: 16 frequencies per axis (frame / height / width), [cos(48), sin(48)]."""

    positions = torch.arange(rows, dtype=torch.float32, device=device)
    axes = (
        torch.div(positions, 4096, rounding_mode="floor"),
        torch.div(positions, 64, rounding_mode="floor").remainder(64),
        positions.remainder(64),
    )
    per_axis = MINIMAX_H3_ROPE_DIM // 6  # 16 frequencies per axis
    inv_freq = 1.0 / (
        10000.0
        ** (torch.arange(per_axis, dtype=torch.float32, device=device) / per_axis)
    )
    angles = torch.cat(
        [axis[:, None] * inv_freq[None, :] for axis in axes], dim=-1
    )  # [rows, 48]
    return (
        torch.cat((angles.cos(), angles.sin()), dim=-1).to(torch.bfloat16).contiguous()
    )


def make_inputs(rows: int, seed: int, device: torch.device) -> dict:
    model = model_tensors(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    x = torch.empty((rows, MINIMAX_H3_HIDDEN), dtype=torch.bfloat16, device=device)
    x.normal_(mean=0.0, std=0.5, generator=generator)
    positions = torch.arange(rows, device=device, dtype=torch.int64)
    adaln_index = (
        torch.div(positions * ADALN_ROWS, max(rows, 1), rounding_mode="floor")
        .clamp_max(ADALN_ROWS - 1)
        .to(torch.int32)
    )
    return {
        "x": x,
        "x_norm_weight": model["x_norm_weight"],
        "adaln_scale": model["adaln_scale"],
        "adaln_shift": model["adaln_shift"],
        "adaln_index": adaln_index,
        "q_norm_weight": model["q_norm_weight"],
        "k_norm_weight": model["k_norm_weight"],
        "rope_cos_sin": rope_cache(rows, device),
    }


# --------------------------------------------------------------------------------------------
# Exact torch emulation
# --------------------------------------------------------------------------------------------
def normalized_activation(inputs: dict) -> torch.Tensor:
    """BF16 ``a = bf16(shift + bf16(rmsnorm(x) * w) * bf16(1 + scale))`` (the kernel's round points)."""

    norm = torch.nn.functional.rms_norm(
        inputs["x"],
        (MINIMAX_H3_HIDDEN,),
        inputs["x_norm_weight"],
        eps=MINIMAX_H3_DEFAULT_EPS,
    ).to(torch.bfloat16)
    index = inputs["adaln_index"].to(torch.int64)
    scale = inputs["adaln_scale"].index_select(0, index)
    shift = inputs["adaln_shift"].index_select(0, index)
    return torch.addcmul(shift, norm, (scale + 1.0).to(torch.bfloat16)).to(
        torch.bfloat16
    )


def dequantized_fp8_operands(
    a: torch.Tensor, model: dict
) -> tuple[torch.Tensor, torch.Tensor]:
    scale = fp8_scale_from_amax(a.float().abs().amax(dim=1).clamp_min(1e-12))
    a_q = (
        (a.float() / scale[:, None]).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    )
    w_q, w_scale = model["fp8"]
    return a_q.float() * scale[:, None], w_q.float() * w_scale[:, None]


def dequantize_nvfp4(
    packed: torch.Tensor, sf: torch.Tensor, global_scale: torch.Tensor
) -> torch.Tensor:
    """Row-major ``[rows, K/2]`` E2M1x2 codes + ``[rows, K/16]`` UE4M3 scales -> FP32 ``[rows, K]``."""

    low = packed & 0x0F
    high = packed >> 4
    codes = torch.stack((low, high), dim=-1).reshape(packed.shape[0], -1)
    table = torch.tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ],
        dtype=torch.float32,
        device=packed.device,
    )
    values = table[codes.to(torch.int64)]
    scales = (
        sf.view(torch.float8_e4m3fn)
        .float()
        .repeat_interleave(MINIMAX_H3_SF_BLOCK, dim=1)
    )
    return values * scales / global_scale.float()


def make_workspaces(variant: str, rows: int, device: torch.device) -> dict:
    """Stage-1 activation buffers, handed to the operator so the test can read them back."""

    if variant == "fp8":
        return {
            "act_q": torch.empty(
                (rows, MINIMAX_H3_HIDDEN), dtype=torch.float8_e4m3fn, device=device
            ),
            "act_scale": torch.empty((rows,), dtype=torch.float32, device=device),
        }
    return {
        "act_q": torch.empty(
            (rows, MINIMAX_H3_HIDDEN // 2), dtype=torch.uint8, device=device
        ),
        "act_sf": torch.empty(
            (rows, MINIMAX_H3_HIDDEN // MINIMAX_H3_SF_BLOCK),
            dtype=torch.uint8,
            device=device,
        ),
    }


def e4m3_ulp(x: torch.Tensor) -> torch.Tensor:
    """Spacing of the E4M3 grid at |x| (3 mantissa bits, subnormal step 2^-9)."""

    magnitude = x.abs().clamp_min(2.0**-6)
    return torch.exp2(torch.floor(torch.log2(magnitude)) - 3)


def e2m1_spacing(values: torch.Tensor) -> torch.Tensor:
    """Distance to the next E2M1 code at each decoded magnitude (0.5 below 2, 1 below 4, else 2)."""

    magnitude = values.abs()
    return torch.where(
        magnitude < 2.0,
        torch.full_like(magnitude, 0.5),
        torch.where(
            magnitude < 4.0,
            torch.full_like(magnitude, 1.0),
            torch.full_like(magnitude, 2.0),
        ),
    )


def kernel_stage1_activation(
    variant: str, workspaces: dict, act_global_scale
) -> torch.Tensor:
    """FP32 dequantization of the operator's own stage-1 output (the fused GEMM's exact A operand)."""

    if variant == "fp8":
        return workspaces["act_q"].float() * workspaces["act_scale"][:, None]
    return dequantize_nvfp4(workspaces["act_q"], workspaces["act_sf"], act_global_scale)


def assert_stage1(
    variant: str, a: torch.Tensor, workspaces: dict, act_global_scale
) -> None:
    """Stage 1 (norm + AdaLN + quantization) against its definition, at quantization resolution.

    FP8: the per-token scale is ``RN(amax / 448)`` (one BF16 ulp of the row absmax is the only
    freedom) and every dequantized value lies within half an E4M3 step of the BF16 activation.
    NVFP4: every block scale is within one UE4M3 code of ``amax_block * global_scale / 6`` and every
    dequantized value lies within half an E2M1 step (at that block scale) of the BF16 activation;
    values saturated at |6| may fall short of larger activations.  Mirrors the Cake contract rules.
    """

    a32 = a.float()
    bf16_ulp = torch.exp2(torch.floor(torch.log2(a32.abs().clamp_min(2.0**-126))) - 7)
    deq = kernel_stage1_activation(variant, workspaces, act_global_scale)
    if variant == "fp8":
        scale = workspaces["act_scale"]
        expected_scale = fp8_scale_from_amax(a32.abs().amax(dim=1).clamp_min(1e-12))
        torch.testing.assert_close(scale, expected_scale, atol=0.0, rtol=2**-7)
        bound = 0.5 * e4m3_ulp(workspaces["act_q"].float()) * scale[:, None] + bf16_ulp
        saturated = torch.zeros_like(bound, dtype=torch.bool)
    else:
        gs = act_global_scale.float()
        blocks = a32.view(a32.shape[0], -1, MINIMAX_H3_SF_BLOCK)
        sf_ref = (
            (blocks.abs().amax(dim=-1) * (gs / E2M1_MAX))
            .to(torch.float8_e4m3fn)
            .float()
        )
        sf_kernel = workspaces["act_sf"].view(torch.float8_e4m3fn).float()
        sf_bound = e4m3_ulp(torch.maximum(sf_ref, sf_kernel)) * (1.0 + 2**-20)
        assert bool(((sf_kernel - sf_ref).abs() <= sf_bound).all()), (
            "act_sf beyond one UE4M3 code"
        )
        block_scale = (sf_kernel / gs).repeat_interleave(MINIMAX_H3_SF_BLOCK, dim=1)
        low = workspaces["act_q"] & 0x0F
        high = workspaces["act_q"] >> 4
        codes = torch.stack((low, high), dim=-1).reshape(a32.shape[0], -1)
        values = dequantize_nvfp4(
            workspaces["act_q"],
            torch.full_like(workspaces["act_sf"], 0x38),  # scale 1.0 -> raw E2M1 values
            torch.ones((), device=a.device),
        )
        del codes
        bound = 0.5 * e2m1_spacing(values) * block_scale + bf16_ulp
        saturated = (values.abs() >= E2M1_MAX) & (a32.abs() > deq.abs())
    err = (deq - a32).abs()
    ok = (err <= bound) | saturated
    assert torch.isfinite(deq).all(), "non-finite dequantized activation"
    assert bool(ok.all()), (
        f"stage-1 {variant} activation off by more than half a quantization step: "
        f"max err {float(err.max()):.4g}, violations {int((~ok).sum())}/{ok.numel()}"
    )


def dequantized_nvfp4_operands(
    a: torch.Tensor, act_global_scale: torch.Tensor, model: dict
):
    a_q, a_sf = fp4_quantize(
        a,
        act_global_scale,
        sf_vec_size=MINIMAX_H3_SF_BLOCK,
        sf_use_ue8m0=False,
        is_sf_swizzled_layout=False,
    )
    a_deq = dequantize_nvfp4(a_q, a_sf.reshape(a.shape[0], -1), act_global_scale)
    w_q, _w_sf_swizzled, w_global_scale = model["nvfp4"]
    _w_q_linear, w_sf_linear = fp4_quantize(
        model["qkv_weight"],
        w_global_scale,
        sf_vec_size=MINIMAX_H3_SF_BLOCK,
        sf_use_ue8m0=False,
        is_sf_swizzled_layout=False,
    )
    w_deq = dequantize_nvfp4(
        w_q, w_sf_linear.reshape(MINIMAX_H3_QKV_WIDTH, -1), w_global_scale
    )
    return a_deq, w_deq


def partial_neox_rope(x: torch.Tensor, rope_cos_sin: torch.Tensor) -> torch.Tensor:
    half = MINIMAX_H3_ROPE_DIM // 2
    rotary = x[..., :MINIMAX_H3_ROPE_DIM].float()
    tail = x[..., MINIMAX_H3_ROPE_DIM:]
    cos = torch.cat((rope_cos_sin[:, :half], rope_cos_sin[:, :half]), dim=-1).float()[
        :, None, :
    ]
    sin = torch.cat((rope_cos_sin[:, half:], rope_cos_sin[:, half:]), dim=-1).float()[
        :, None, :
    ]
    rotated_half = torch.cat((-rotary[..., half:], rotary[..., :half]), dim=-1)
    return torch.cat(
        ((rotary * cos + rotated_half * sin).to(torch.bfloat16), tail), dim=-1
    )


def reference_qkv(
    inputs: dict, variant: str, act_global_scale=None, workspaces: dict | None = None
):
    """Exact emulation of the operator in BF16: ``(q, k, v)`` each ``[rows, 56, 128]``.

    With ``workspaces`` (the operator's own stage-1 buffers, validated by ``assert_stage1``), the
    reference GEMM consumes exactly the quantized activation the fused GEMM consumed, so the
    comparison isolates the GEMM + epilogue; otherwise the activation is re-quantized in torch.
    """

    model = model_tensors(inputs["x"].device)
    a = normalized_activation(inputs)
    if variant == "fp8":
        a_deq, w_deq = dequantized_fp8_operands(a, model)
    else:
        a_deq, w_deq = dequantized_nvfp4_operands(a, act_global_scale, model)
    if workspaces is not None:
        a_deq = kernel_stage1_activation(variant, workspaces, act_global_scale)
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        y = torch.empty(
            (a.shape[0], MINIMAX_H3_QKV_WIDTH), dtype=torch.bfloat16, device=a.device
        )
        for start in range(0, a.shape[0], 1024):
            stop = min(start + 1024, a.shape[0])
            y[start:stop] = (a_deq[start:stop] @ w_deq.t()).to(torch.bfloat16)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
    grouped = y.view(a.shape[0], MINIMAX_H3_NUM_HEADS, 3, MINIMAX_H3_HEAD_DIM)
    q = torch.nn.functional.rms_norm(
        grouped[:, :, 0, :],
        (MINIMAX_H3_HEAD_DIM,),
        inputs["q_norm_weight"],
        eps=MINIMAX_H3_DEFAULT_EPS,
    ).to(torch.bfloat16)
    k = torch.nn.functional.rms_norm(
        grouped[:, :, 1, :],
        (MINIMAX_H3_HEAD_DIM,),
        inputs["k_norm_weight"],
        eps=MINIMAX_H3_DEFAULT_EPS,
    ).to(torch.bfloat16)
    return (
        partial_neox_rope(q, inputs["rope_cos_sin"]).contiguous(),
        partial_neox_rope(k, inputs["rope_cos_sin"]).contiguous(),
        grouped[:, :, 2, :].contiguous(),
    )


def dequantize_output(
    out, variant_out_mode: str, name: str, scales: dict
) -> torch.Tensor:
    value = getattr(out, name)
    if variant_out_mode == "bf16":
        return value.float()
    if variant_out_mode == "e4m3":
        return value.float() * scales[f"{name}_descale"]
    rows = value.shape[0]
    packed = value.reshape(rows * MINIMAX_H3_NUM_HEADS, MINIMAX_H3_HEAD_DIM // 2)
    sf = getattr(out, f"{name}_sf").reshape(
        rows * MINIMAX_H3_NUM_HEADS, MINIMAX_H3_HEAD_DIM // MINIMAX_H3_SF_BLOCK
    )
    return dequantize_nvfp4(packed, sf, scales[f"{name}_global_scale"]).reshape(
        rows, MINIMAX_H3_NUM_HEADS, MINIMAX_H3_HEAD_DIM
    )


def run_operator(
    inputs: dict,
    variant: str,
    out_mode: str,
    scales: dict,
    act_global_scale=None,
    workspaces: dict | None = None,
):
    model = model_tensors(inputs["x"].device)
    common = dict(
        **(workspaces or {}),
        eps=MINIMAX_H3_DEFAULT_EPS,
        out_mode=out_mode,
        q_descale=scales.get("q_descale"),
        k_descale=scales.get("k_descale"),
        v_descale=scales.get("v_descale"),
        q_global_scale=scales.get("q_global_scale"),
        k_global_scale=scales.get("k_global_scale"),
        v_global_scale=scales.get("v_global_scale"),
    )
    if variant == "fp8":
        w_q, w_scale = model["fp8"]
        return minimax_h3_fp8_pre_attention(
            inputs["x"],
            inputs["x_norm_weight"],
            inputs["adaln_scale"],
            inputs["adaln_shift"],
            inputs["adaln_index"],
            w_q,
            w_scale,
            inputs["q_norm_weight"],
            inputs["k_norm_weight"],
            inputs["rope_cos_sin"],
            **common,
        )
    w_q, w_sf, w_global_scale = model["nvfp4"]
    return minimax_h3_nvfp4_pre_attention(
        inputs["x"],
        inputs["x_norm_weight"],
        inputs["adaln_scale"],
        inputs["adaln_shift"],
        inputs["adaln_index"],
        w_q,
        w_sf,
        w_global_scale,
        act_global_scale,
        inputs["q_norm_weight"],
        inputs["k_norm_weight"],
        inputs["rope_cos_sin"],
        **common,
    )


def output_scales(out_mode: str, device: torch.device) -> dict:
    if out_mode == "e4m3":
        return {f"{name}_descale": OUT_AMAX / E4M3_MAX for name in "qkv"}
    if out_mode == "nvfp4":
        gs = nvfp4_global_scale_from_amax(OUT_AMAX).to(device)
        return {f"{name}_global_scale": gs for name in "qkv"}
    return {}


def bf16_ulp(magnitude: torch.Tensor) -> torch.Tensor:
    return torch.exp2(torch.floor(torch.log2(magnitude.clamp_min(2.0**-126))) - 7)


def rope_pair_magnitude(mag: torch.Tensor) -> torch.Tensor:
    """``max(|ref_i|, |ref_partner_i|)`` over the RoPE-rotated dims ``[0, 96)`` (partner ``i +- 48``);
    the pass-through dims ``[96, 128)`` keep their own magnitude."""

    half = MINIMAX_H3_ROPE_DIM // 2
    rotary = mag[..., :MINIMAX_H3_ROPE_DIM]
    partner = torch.cat((rotary[..., half:], rotary[..., :half]), dim=-1)
    return torch.cat(
        (torch.maximum(rotary, partner), mag[..., MINIMAX_H3_ROPE_DIM:]), dim=-1
    )


def assert_outputs(out, reference, out_mode: str, scales: dict) -> None:
    """``|actual - ref| <= atol + rtol * mag + 2 * bf16_ulp(mag) + step`` per element.

    The kernel's FP32 accumulation order differs from the reference GEMM, so each pre-normalization
    QKV value may round to the neighbouring BF16 code; RoPE mixes each ``(i, i +- 48)`` pair, so for
    Q/K the magnitude is the pair maximum (rotation-invariant relative bound) and two BF16 steps of
    slack cover the pre-norm round point.  ``step`` is one representable step of the E4M3 / NVFP4
    output format at the element's magnitude (a one-ulp BF16 difference may flip the stored code).
    Mirrors the Cake contract's ``compare_outputs`` rule.
    """

    atol, rtol = {"bf16": (1e-2, 1e-2), "e4m3": (1e-2, 1e-2), "nvfp4": (1e-2, 1e-2)}[
        out_mode
    ]
    for name, expected in zip("qkv", reference, strict=True):
        actual = dequantize_output(out, out_mode, name, scales)
        expected = expected.float()
        assert actual.shape == expected.shape, name
        assert torch.isfinite(actual).all(), name
        mag = expected.abs()
        if name in ("q", "k"):
            mag = rope_pair_magnitude(mag)
        bound = atol + rtol * mag + 2.0 * bf16_ulp(mag)
        if out_mode == "e4m3":
            # One E4M3 step of the *stored* code (value / descale), scaled back to the output domain.
            descale = scales[f"{name}_descale"]
            stored = torch.maximum(actual.abs(), expected.abs()) / descale
            bound = bound + e4m3_ulp(stored) * descale
        elif out_mode == "nvfp4":
            gs = scales[f"{name}_global_scale"].float()
            sf = getattr(out, f"{name}_sf").view(torch.float8_e4m3fn).float()
            block_scale = (sf / gs).repeat_interleave(MINIMAX_H3_SF_BLOCK, dim=-1)
            raw = (
                actual
                * gs
                / sf.repeat_interleave(MINIMAX_H3_SF_BLOCK, dim=-1).clamp_min(2.0**-9)
            )
            bound = bound + e2m1_spacing(raw) * block_scale
        err = (actual - expected).abs()
        bad = ~(err <= bound)
        assert not bool(bad.any()), (
            f"{name}: {int(bad.sum())}/{bad.numel()} elements outside the bound; "
            f"max abs err {float(err.max()):.4g}, max excess {float((err - bound).max()):.4g}"
        )


@requires_sm120
@pytest.mark.parametrize("variant", ["fp8", "nvfp4"])
@pytest.mark.parametrize("rows", ROWS)
def test_minimax_h3_quant_pre_attention_bf16_output(variant: str, rows: int) -> None:
    device = torch.device("cuda:0")
    inputs = make_inputs(rows, seed=rows, device=device)
    act_global_scale = nvfp4_global_scale_from_amax(ACT_AMAX).to(device)
    workspaces = make_workspaces(variant, rows, device)
    out = run_operator(inputs, variant, "bf16", {}, act_global_scale, workspaces)
    torch.cuda.synchronize()
    assert_stage1(variant, normalized_activation(inputs), workspaces, act_global_scale)
    reference = reference_qkv(inputs, variant, act_global_scale, workspaces)
    assert out.q.dtype == torch.bfloat16 and out.q.shape == (
        rows,
        MINIMAX_H3_NUM_HEADS,
        MINIMAX_H3_HEAD_DIM,
    )
    assert out.q_sf is None
    assert_outputs(out, reference, "bf16", {})


@requires_sm120
@pytest.mark.parametrize("variant", ["fp8", "nvfp4"])
@pytest.mark.parametrize("out_mode", ["e4m3", "nvfp4"])
def test_minimax_h3_quant_pre_attention_quantized_outputs(
    variant: str, out_mode: str
) -> None:
    device = torch.device("cuda:0")
    rows = 257
    inputs = make_inputs(rows, seed=7, device=device)
    act_global_scale = nvfp4_global_scale_from_amax(ACT_AMAX).to(device)
    scales = output_scales(out_mode, device)
    workspaces = make_workspaces(variant, rows, device)
    out = run_operator(inputs, variant, out_mode, scales, act_global_scale, workspaces)
    torch.cuda.synchronize()
    assert_stage1(variant, normalized_activation(inputs), workspaces, act_global_scale)
    reference = reference_qkv(inputs, variant, act_global_scale, workspaces)
    if out_mode == "e4m3":
        assert out.q.dtype == torch.float8_e4m3fn and out.q_sf is None
    else:
        assert out.q.dtype == torch.uint8 and out.q.shape == (
            rows,
            MINIMAX_H3_NUM_HEADS,
            MINIMAX_H3_HEAD_DIM // 2,
        )
        assert out.q_sf is not None and out.q_sf.shape == (
            rows,
            MINIMAX_H3_NUM_HEADS,
            MINIMAX_H3_HEAD_DIM // 16,
        )
    assert_outputs(out, reference, out_mode, scales)


@requires_sm120
def test_minimax_h3_quant_pre_attention_preallocated_outputs_and_workspaces() -> None:
    device = torch.device("cuda:0")
    rows = 300
    inputs = make_inputs(rows, seed=3, device=device)
    model = model_tensors(device)
    w_q, w_scale = model["fp8"]
    q = torch.empty(
        (rows, MINIMAX_H3_NUM_HEADS, MINIMAX_H3_HEAD_DIM),
        dtype=torch.bfloat16,
        device=device,
    )
    k = torch.empty_like(q)
    v = torch.empty_like(q)
    act_q = torch.empty(
        (rows, MINIMAX_H3_HIDDEN), dtype=torch.float8_e4m3fn, device=device
    )
    act_scale = torch.empty((rows,), dtype=torch.float32, device=device)
    out = minimax_h3_fp8_pre_attention(
        inputs["x"],
        inputs["x_norm_weight"],
        inputs["adaln_scale"],
        inputs["adaln_shift"],
        inputs["adaln_index"],
        w_q,
        w_scale,
        inputs["q_norm_weight"],
        inputs["k_norm_weight"],
        inputs["rope_cos_sin"],
        q=q,
        k=k,
        v=v,
        act_q=act_q,
        act_scale=act_scale,
    )
    torch.cuda.synchronize()
    assert (
        out.q.data_ptr() == q.data_ptr()
        and out.k.data_ptr() == k.data_ptr()
        and out.v.data_ptr() == v.data_ptr()
    )
    reference = reference_qkv(inputs, "fp8")
    assert_outputs(out, reference, "bf16", {})
    # The stage-1 workspace holds the per-token E4M3 activation and its scale.
    a = normalized_activation(inputs)
    expected_scale = fp8_scale_from_amax(a.float().abs().amax(dim=1).clamp_min(1e-12))
    torch.testing.assert_close(act_scale, expected_scale, atol=0.0, rtol=2**-7)


@requires_sm120
def test_minimax_h3_quant_pre_attention_rejects_bad_inputs() -> None:
    device = torch.device("cuda:0")
    inputs = make_inputs(64, seed=1, device=device)
    model = model_tensors(device)
    w_q, w_scale = model["fp8"]
    good = (
        inputs["x"],
        inputs["x_norm_weight"],
        inputs["adaln_scale"],
        inputs["adaln_shift"],
        inputs["adaln_index"],
        w_q,
        w_scale,
        inputs["q_norm_weight"],
        inputs["k_norm_weight"],
        inputs["rope_cos_sin"],
    )
    with pytest.raises(ValueError):
        minimax_h3_fp8_pre_attention(inputs["x"].float(), *good[1:])
    with pytest.raises(ValueError):
        minimax_h3_fp8_pre_attention(inputs["x"][:, :64], *good[1:])
    with pytest.raises(ValueError):
        minimax_h3_fp8_pre_attention(*good, out_mode="int8")
    with pytest.raises(ValueError):
        minimax_h3_fp8_pre_attention(*good, out_mode="e4m3")  # descales are required
    with pytest.raises(ValueError):
        minimax_h3_fp8_pre_attention(*good[:5], w_q.view(torch.uint8), *good[6:])
