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
"""SM120 (GB202) FP8 / NVFP4 fused MiniMax-H3 output projection against an exact torch emulation.

The reference quantizes ``attn_out`` exactly like the operator's stage 1 (per-token E4M3 with
``RN(amax / 448)`` or FlashInfer ``fp4_quantize`` block-16 NVFP4), runs the dequantized GEMM in
FP32, rounds the projection ``o`` to BF16, then applies ``out = bf16(residual + bf16(gate[idx] * o))``
(the SGLang ``indexed_gate_bf16`` round points).  Kernel and emulation accumulate the GEMM in a
different order, so ``o`` may round to the neighbouring BF16 code; the tolerance below covers that
flip through the gate multiply and the two final roundings.
"""

import pytest
import torch

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
from flashinfer.utils import get_compute_capability

E4M3_MAX = 448.0
# Calibrated activation absmax used for the NVFP4 global scale (synthetic attention output N(0, 0.5)).
ACT_AMAX = 4.0
# Tail lengths around the 128-row tile, plus multi-tile shapes.
ROWS = [1, 127, 128, 129, 257, 4097]


def _supported() -> bool:
    if not torch.cuda.is_available():
        return False
    # The public functions are gated to compute capability 12.0 (GB202); 12.1 would raise, not run.
    return get_compute_capability(torch.device("cuda:0")) == (12, 0)


requires_sm120 = pytest.mark.skipif(
    not _supported(), reason="requires a CUDA GPU with compute capability 12.0 (GB202)"
)


# --------------------------------------------------------------------------------------------
# Synthetic model + inputs
# --------------------------------------------------------------------------------------------
_MODEL = {}


def model_tensors(device: torch.device) -> dict:
    key = str(device)
    if key in _MODEL:
        return _MODEL[key]
    generator = torch.Generator(device=device).manual_seed(4616)
    o_weight = torch.empty(
        (MINIMAX_H3_HIDDEN, MINIMAX_H3_ATTN_DIM), dtype=torch.bfloat16, device=device
    )
    o_weight.normal_(mean=0.0, std=0.01, generator=generator)
    gate = torch.empty(
        (MINIMAX_H3_GATE_ROWS, MINIMAX_H3_HIDDEN), dtype=torch.bfloat16, device=device
    )
    gate.uniform_(-1.0, 1.0, generator=generator)
    model = {"o_weight": o_weight, "gate": gate}
    model["fp8"] = quantize_minimax_h3_o_weight_fp8(o_weight)
    model["nvfp4"] = quantize_minimax_h3_o_weight_nvfp4(o_weight)
    _MODEL[key] = model
    return model


def make_inputs(
    rows: int, seed: int, device: torch.device, guard_rows: bool = False
) -> dict:
    generator = torch.Generator(device=device).manual_seed(seed)
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
    if guard_rows:
        # Out-of-range indices must contribute a zero gate (out = residual).
        gate_index = torch.where(
            positions % 101 == 50, torch.full_like(gate_index, -1), gate_index
        )
        gate_index = torch.where(
            positions % 103 == 60,
            torch.full_like(gate_index, MINIMAX_H3_GATE_ROWS),
            gate_index,
        )
    return {
        "attn_out": attn_out,
        "residual": residual,
        "gate_index": gate_index.contiguous(),
    }


# --------------------------------------------------------------------------------------------
# Exact torch emulation
# --------------------------------------------------------------------------------------------
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


def unswizzle_sf_128x4(swizzled: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """Inverse of the FlashInfer 128x4 swizzled scale layout (``get_sf_out_offset_128x4``)."""

    padded_rows = (rows + 127) // 128 * 128
    padded_cols = (cols + 3) // 4 * 4
    flat = swizzled.reshape(-1)
    m = torch.arange(rows, device=swizzled.device)[:, None]
    k = torch.arange(cols, device=swizzled.device)[None, :]
    num_k_tiles = padded_cols // 4
    offset = (
        (m // 128) * num_k_tiles * 512
        + (k // 4) * 512
        + (m % 32) * 16
        + ((m % 128) // 32) * 4
        + (k % 4)
    )
    assert int(offset.max()) < padded_rows * padded_cols
    return flat[offset]


def dequantized_operands(
    variant: str, attn_out: torch.Tensor, model: dict, act_global_scale
):
    """FP32 dequantized (A, W) exactly as the operator's stage 1 quantizes them."""

    if variant == "fp8":
        scale = fp8_scale_from_amax(attn_out.float().abs().amax(dim=1).clamp_min(1e-12))
        a_q = (
            (attn_out.float() / scale[:, None])
            .clamp(-E4M3_MAX, E4M3_MAX)
            .to(torch.float8_e4m3fn)
        )
        w_q, w_scale = model["fp8"]
        return a_q.float() * scale[:, None], w_q.float() * w_scale[:, None]
    w_q, w_sf, w_gs = model["nvfp4"]
    a_q, a_sf = fp4_quantize(
        attn_out,
        act_global_scale,
        sf_vec_size=MINIMAX_H3_SF_BLOCK,
        sf_use_ue8m0=False,
        is_sf_swizzled_layout=False,
    )
    a_deq = dequantize_nvfp4(a_q, a_sf.reshape(attn_out.shape[0], -1), act_global_scale)
    w_sf_linear = unswizzle_sf_128x4(
        w_sf, MINIMAX_H3_HIDDEN, MINIMAX_H3_ATTN_DIM // MINIMAX_H3_SF_BLOCK
    )
    return a_deq, dequantize_nvfp4(w_q, w_sf_linear, w_gs)


def reference(variant: str, inputs: dict, model: dict, act_global_scale):
    """Returns ``(out, p)``: the exact operator output and the BF16 gated projection."""

    a_deq, w_deq = dequantized_operands(
        variant, inputs["attn_out"], model, act_global_scale
    )
    rows = inputs["attn_out"].shape[0]
    o = torch.empty(
        (rows, MINIMAX_H3_HIDDEN), dtype=torch.bfloat16, device=a_deq.device
    )
    saved = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for start in range(0, rows, 4096):
            stop = min(rows, start + 4096)
            o[start:stop] = (a_deq[start:stop] @ w_deq.t()).to(torch.bfloat16)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = saved
    index = inputs["gate_index"].to(torch.int64)
    valid = (index >= 0) & (index < MINIMAX_H3_GATE_ROWS)
    g = model["gate"].index_select(0, index.clamp(0, MINIMAX_H3_GATE_ROWS - 1))
    g = torch.where(valid[:, None], g, torch.zeros_like(g))
    p = (g * o).to(torch.bfloat16)
    out = (inputs["residual"] + p).to(torch.bfloat16)
    return out, p


def bf16_ulp(x: torch.Tensor) -> torch.Tensor:
    magnitude = x.abs().clamp_min(2.0**-126)
    return torch.exp2(torch.floor(torch.log2(magnitude)) - 7)


def assert_matches(
    out: torch.Tensor, expected: torch.Tensor, p: torch.Tensor, inputs: dict
) -> None:
    """``|out - ref| <= atol + rtol * mag + 2 bf16 steps``, ``mag = max(|ref|, |gate * o|)``: one BF16
    flip of the projection (accumulation order) moves ``gate * o`` by at most two steps at
    ``|p|``, its rounding adds one, and the final ``residual + p`` rounding one more."""

    mag = torch.maximum(expected.float().abs(), p.float().abs())
    bound = 1e-2 + 1e-2 * mag + 2.0 * bf16_ulp(mag)
    err = (out.float() - expected.float()).abs()
    assert torch.isfinite(out.float()).all(), "non-finite output"
    bad = err > bound
    assert not bool(bad.any()), (
        f"{int(bad.sum())} / {bad.numel()} elements outside the bound; max err {float(err.max())}"
    )
    invalid = (inputs["gate_index"] < 0) | (
        inputs["gate_index"] >= MINIMAX_H3_GATE_ROWS
    )
    if bool(invalid.any()):
        assert torch.equal(out[invalid], inputs["residual"][invalid]), (
            "out-of-range gate rows must reproduce the residual exactly"
        )


def run_operator(
    variant: str, inputs: dict, model: dict, act_global_scale, workspaces=None
):
    workspaces = workspaces or {}
    if variant == "fp8":
        w_q, w_scale = model["fp8"]
        return minimax_h3_fp8_out_proj(
            inputs["attn_out"],
            w_q,
            w_scale,
            model["gate"],
            inputs["gate_index"],
            inputs["residual"],
            **workspaces,
        )
    w_q, w_sf, w_gs = model["nvfp4"]
    return minimax_h3_nvfp4_out_proj(
        inputs["attn_out"],
        w_q,
        w_sf,
        w_gs,
        act_global_scale,
        model["gate"],
        inputs["gate_index"],
        inputs["residual"],
        **workspaces,
    )


# --------------------------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------------------------
@requires_sm120
@pytest.mark.parametrize("variant", ["fp8", "nvfp4"])
@pytest.mark.parametrize("rows", ROWS)
def test_out_proj_matches_exact_emulation(variant: str, rows: int) -> None:
    device = torch.device("cuda:0")
    model = model_tensors(device)
    inputs = make_inputs(
        rows, seed=616000 + rows, device=device, guard_rows=(rows >= 129)
    )
    act_global_scale = nvfp4_global_scale_from_amax(ACT_AMAX).to(device)
    out = run_operator(variant, inputs, model, act_global_scale)
    torch.cuda.synchronize()
    expected, p = reference(variant, inputs, model, act_global_scale)
    assert_matches(out, expected, p, inputs)


@requires_sm120
@pytest.mark.parametrize("variant", ["fp8", "nvfp4"])
def test_out_proj_stage1_workspaces_are_exact(variant: str) -> None:
    """The caller-owned stage-1 buffers hold the exact quantization of ``attn_out``."""

    device = torch.device("cuda:0")
    model = model_tensors(device)
    rows = 257
    inputs = make_inputs(rows, seed=616500, device=device)
    act_global_scale = nvfp4_global_scale_from_amax(ACT_AMAX).to(device)
    out = torch.empty((rows, MINIMAX_H3_HIDDEN), dtype=torch.bfloat16, device=device)
    if variant == "fp8":
        workspaces = {
            "out": out,
            "act_q": torch.empty(
                (rows, MINIMAX_H3_ATTN_DIM), dtype=torch.float8_e4m3fn, device=device
            ),
            "act_scale": torch.empty((rows,), dtype=torch.float32, device=device),
        }
        returned = run_operator(variant, inputs, model, act_global_scale, workspaces)
        torch.cuda.synchronize()
        scale = fp8_scale_from_amax(
            inputs["attn_out"].float().abs().amax(dim=1).clamp_min(1e-12)
        )
        codes = (
            (inputs["attn_out"].float() / scale[:, None])
            .clamp(-E4M3_MAX, E4M3_MAX)
            .to(torch.float8_e4m3fn)
        )
        assert torch.equal(workspaces["act_scale"], scale)
        assert torch.equal(
            workspaces["act_q"].view(torch.uint8), codes.view(torch.uint8)
        )
    else:
        workspaces = {
            "out": out,
            "act_q": torch.empty(
                (rows, MINIMAX_H3_ATTN_DIM // 2), dtype=torch.uint8, device=device
            ),
            "act_sf": torch.empty(
                (rows, MINIMAX_H3_ATTN_DIM // MINIMAX_H3_SF_BLOCK),
                dtype=torch.uint8,
                device=device,
            ),
        }
        returned = run_operator(variant, inputs, model, act_global_scale, workspaces)
        torch.cuda.synchronize()
        a_q, a_sf = fp4_quantize(
            inputs["attn_out"],
            act_global_scale,
            sf_vec_size=MINIMAX_H3_SF_BLOCK,
            sf_use_ue8m0=False,
            is_sf_swizzled_layout=False,
        )
        assert torch.equal(workspaces["act_sf"], a_sf.reshape(rows, -1))
        assert torch.equal(workspaces["act_q"], a_q.reshape(rows, -1))
    assert returned.data_ptr() == out.data_ptr()
    expected, p = reference(variant, inputs, model, act_global_scale)
    assert_matches(out, expected, p, inputs)


@requires_sm120
def test_out_proj_rejects_bad_inputs() -> None:
    device = torch.device("cuda:0")
    model = model_tensors(device)
    inputs = make_inputs(129, seed=616900, device=device)
    w_q, w_scale = model["fp8"]
    with pytest.raises(ValueError):
        minimax_h3_fp8_out_proj(
            inputs["attn_out"][:, :-16].contiguous(),
            w_q,
            w_scale,
            model["gate"],
            inputs["gate_index"],
            inputs["residual"],
        )
    with pytest.raises(ValueError):
        minimax_h3_fp8_out_proj(
            inputs["attn_out"],
            w_q,
            w_scale,
            model["gate"],
            inputs["gate_index"].to(torch.int64),
            inputs["residual"],
        )
    with pytest.raises(ValueError):
        minimax_h3_fp8_out_proj(
            inputs["attn_out"],
            w_q,
            w_scale,
            model["gate"][:4].contiguous(),
            inputs["gate_index"],
            inputs["residual"],
        )
    with pytest.raises(ValueError):
        minimax_h3_fp8_out_proj(
            inputs["attn_out"],
            w_q,
            w_scale,
            model["gate"],
            inputs["gate_index"],
            inputs["residual"].float(),
        )
