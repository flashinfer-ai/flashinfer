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
"""

import functools
from typing import Optional, Tuple, Union

import torch

from ..api_logging import flashinfer_api
from ..jit.cake_minimax_h3_sm120_quant_out_proj import (
    gen_minimax_h3_sm120_quant_out_proj_module,
)
from ..utils import register_custom_op, register_fake_op, supported_compute_capability

MINIMAX_H3_HIDDEN = 5376
MINIMAX_H3_ATTN_DIM = 7168  # 56 heads x 128
MINIMAX_H3_GATE_ROWS = 9
MINIMAX_H3_SF_BLOCK = 16
MINIMAX_H3_MAX_ROWS = 1 << 24
E4M3_MAX = 448.0
E2M1_MAX = 6.0

_ATTN_SF = MINIMAX_H3_ATTN_DIM // MINIMAX_H3_SF_BLOCK  # 448

Scalar = Union[float, torch.Tensor]


@functools.cache
def _get_module():
    return gen_minimax_h3_sm120_quant_out_proj_module().build_and_load()


def _require(
    name: str,
    tensor: torch.Tensor,
    dtype: torch.dtype,
    shape: Tuple[int, ...],
    device: torch.device,
) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"{name} must be a torch.Tensor")
    if tensor.dtype != dtype:
        raise ValueError(f"{name} must be {dtype}, got {tensor.dtype}")
    if tuple(tensor.shape) != tuple(shape):
        raise ValueError(
            f"{name} must have shape {tuple(shape)}, got {tuple(tensor.shape)}"
        )
    if not tensor.is_cuda or tensor.device != device:
        raise ValueError(f"{name} must be a CUDA tensor on {device}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    return tensor


def _check_common(
    attn_out: torch.Tensor,
    gate: torch.Tensor,
    gate_index: torch.Tensor,
    residual: torch.Tensor,
    out: Optional[torch.Tensor],
) -> Tuple[int, torch.device, torch.Tensor]:
    if (
        not isinstance(attn_out, torch.Tensor)
        or attn_out.ndim != 2
        or attn_out.shape[1] != MINIMAX_H3_ATTN_DIM
    ):
        raise ValueError(f"attn_out must be [M, {MINIMAX_H3_ATTN_DIM}]")
    rows = int(attn_out.shape[0])
    if not 1 <= rows <= MINIMAX_H3_MAX_ROWS:
        raise ValueError(f"M must lie in [1, {MINIMAX_H3_MAX_ROWS}], got {rows}")
    device = attn_out.device
    _require("attn_out", attn_out, torch.bfloat16, (rows, MINIMAX_H3_ATTN_DIM), device)
    _require(
        "gate", gate, torch.bfloat16, (MINIMAX_H3_GATE_ROWS, MINIMAX_H3_HIDDEN), device
    )
    _require("gate_index", gate_index, torch.int32, (rows,), device)
    _require("residual", residual, torch.bfloat16, (rows, MINIMAX_H3_HIDDEN), device)
    if out is None:
        out = torch.empty(
            (rows, MINIMAX_H3_HIDDEN), dtype=torch.bfloat16, device=device
        )
    _require("out", out, torch.bfloat16, (rows, MINIMAX_H3_HIDDEN), device)
    return rows, device, out


@register_custom_op(
    "flashinfer::minimax_h3_sm120_fp8_out_proj",
    mutates_args=("act_q", "act_scale", "out"),
)
def _fp8_impl(
    attn_out: torch.Tensor,
    o_weight_q: torch.Tensor,
    o_weight_scale: torch.Tensor,
    gate: torch.Tensor,
    gate_index: torch.Tensor,
    residual: torch.Tensor,
    act_q: torch.Tensor,
    act_scale: torch.Tensor,
    out: torch.Tensor,
) -> None:
    _get_module().minimax_h3_sm120_fp8_out_proj(
        attn_out,
        o_weight_q,
        o_weight_scale,
        gate,
        gate_index,
        residual,
        act_q,
        act_scale,
        out,
    )


@register_fake_op("flashinfer::minimax_h3_sm120_fp8_out_proj")
def _fp8_fake(
    attn_out,
    o_weight_q,
    o_weight_scale,
    gate,
    gate_index,
    residual,
    act_q,
    act_scale,
    out,
) -> None:
    pass


@register_custom_op(
    "flashinfer::minimax_h3_sm120_nvfp4_out_proj",
    mutates_args=("act_q", "act_sf", "out"),
)
def _nvfp4_impl(
    attn_out: torch.Tensor,
    o_weight_q: torch.Tensor,
    o_weight_sf: torch.Tensor,
    act_global_scale: torch.Tensor,
    gate: torch.Tensor,
    gate_index: torch.Tensor,
    residual: torch.Tensor,
    act_q: torch.Tensor,
    act_sf: torch.Tensor,
    out: torch.Tensor,
    alpha: float,
) -> None:
    _get_module().minimax_h3_sm120_nvfp4_out_proj(
        attn_out,
        o_weight_q,
        o_weight_sf,
        act_global_scale,
        gate,
        gate_index,
        residual,
        act_q,
        act_sf,
        out,
        alpha,
    )


@register_fake_op("flashinfer::minimax_h3_sm120_nvfp4_out_proj")
def _nvfp4_fake(
    attn_out,
    o_weight_q,
    o_weight_sf,
    act_global_scale,
    gate,
    gate_index,
    residual,
    act_q,
    act_sf,
    out,
    alpha,
) -> None:
    pass


def fp8_scale_from_amax(amax: torch.Tensor) -> torch.Tensor:
    r"""``RN(amax / 448)`` with a true IEEE division (``tensor / 448.0`` multiplies by a reciprocal
    and differs in the last FP32 bit, which flips E4M3 rounding ties)."""

    return amax / torch.full((), E4M3_MAX, dtype=torch.float32, device=amax.device)


def nvfp4_global_scale_from_amax(amax: Union[float, torch.Tensor]) -> torch.Tensor:
    r"""FlashInfer NVFP4 global scale ``448 * 6 / amax`` as an FP32 ``[1]`` CUDA tensor."""

    if isinstance(amax, torch.Tensor):
        return (E4M3_MAX * E2M1_MAX / amax.float()).reshape(1).contiguous()
    return torch.tensor(
        [E4M3_MAX * E2M1_MAX / float(amax)], dtype=torch.float32, device="cuda"
    )


def quantize_minimax_h3_o_weight_fp8(
    o_weight: torch.Tensor, chunk_rows: int = 1792
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Per-output-channel E4M3 quantization of the BF16 ``[5376, 7168]`` attention output weight.

    Returns ``(o_weight_q float8_e4m3fn [5376, 7168], o_weight_scale float32 [5376])`` with
    ``o_weight_scale[n] = RN(amax(row n) / 448)`` and ``o_weight_q = RN(row / scale)``.
    """

    if tuple(o_weight.shape) != (MINIMAX_H3_HIDDEN, MINIMAX_H3_ATTN_DIM):
        raise ValueError(
            f"o_weight must be [{MINIMAX_H3_HIDDEN}, {MINIMAX_H3_ATTN_DIM}]"
        )
    weight_q = torch.empty(
        o_weight.shape, dtype=torch.float8_e4m3fn, device=o_weight.device
    )
    scale = torch.empty(
        (MINIMAX_H3_HIDDEN,), dtype=torch.float32, device=o_weight.device
    )
    for start in range(0, MINIMAX_H3_HIDDEN, chunk_rows):
        stop = min(start + chunk_rows, MINIMAX_H3_HIDDEN)
        rows = o_weight[start:stop].float()
        row_scale = fp8_scale_from_amax(rows.abs().amax(dim=1).clamp_min(1e-12))
        scale[start:stop] = row_scale
        weight_q[start:stop] = (
            (rows / row_scale[:, None])
            .clamp(-E4M3_MAX, E4M3_MAX)
            .to(torch.float8_e4m3fn)
        )
    return weight_q, scale


def quantize_minimax_h3_o_weight_nvfp4(
    o_weight: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""FlashInfer NVFP4 quantization of the BF16 ``[5376, 7168]`` attention output weight.

    Returns ``(o_weight_q uint8 [5376, 3584], o_weight_sf uint8 (128x4 swizzled layout),
    o_weight_global_scale float32 [1])`` exactly as :func:`flashinfer.fp4_quantize` with
    ``sf_vec_size=16`` and ``is_sf_swizzled_layout=True`` produces them.
    """

    from ..quantization import fp4_quantize

    if tuple(o_weight.shape) != (MINIMAX_H3_HIDDEN, MINIMAX_H3_ATTN_DIM):
        raise ValueError(
            f"o_weight must be [{MINIMAX_H3_HIDDEN}, {MINIMAX_H3_ATTN_DIM}]"
        )
    global_scale = nvfp4_global_scale_from_amax(o_weight.float().abs().amax())
    weight_q, weight_sf = fp4_quantize(
        o_weight.contiguous(),
        global_scale,
        sf_vec_size=MINIMAX_H3_SF_BLOCK,
        sf_use_ue8m0=False,
        is_sf_swizzled_layout=True,
    )
    return weight_q.contiguous(), weight_sf.contiguous(), global_scale


@supported_compute_capability([120])
@flashinfer_api
def minimax_h3_fp8_out_proj(
    attn_out: torch.Tensor,
    o_weight_q: torch.Tensor,
    o_weight_scale: torch.Tensor,
    gate: torch.Tensor,
    gate_index: torch.Tensor,
    residual: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
    act_q: Optional[torch.Tensor] = None,
    act_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""MiniMax-H3 FP8 (W8A8) attention output projection with the fused gated residual for
    SM120 (GB202: RTX 5090 / RTX PRO 6000 Blackwell).

    One launch: a per-token E4M3 quantization of ``attn_out`` (``scale = RN(amax / 448)``) run by
    the spare warps of the persistent kernel's producer warpgroup and overlapped, through per-tile
    ready flags, with the ``mma.sync`` GEMM ``[M, 7168] x [7168, 5376]`` whose epilogue applies the
    per-token x per-channel dequant scales, rounds the projection ``o`` to BF16 once and writes
    ``out = BF16(residual + BF16(gate[gate_index[m]] * o))``.  This is the SGLang
    ``indexed_gate_bf16`` round-point convention.  Rows whose ``gate_index`` lies outside
    ``[0, 9)`` contribute ``gate = 0`` (``out = residual``).

    Parameters
    ----------
    attn_out : torch.Tensor
        BF16 ``[M, 7168]`` packed attention output (the ``[M, 56, 128]`` NHD output viewed row-major).
    o_weight_q : torch.Tensor
        E4M3 ``[5376, 7168]`` per-output-channel quantized weight (see
        :func:`quantize_minimax_h3_o_weight_fp8`).
    o_weight_scale : torch.Tensor
        FP32 ``[5376]`` dequant multipliers.
    gate : torch.Tensor
        BF16 ``[9, 5376]`` per-index ``gate_msa`` rows of the AdaLN plan.
    gate_index : torch.Tensor
        int32 ``[M]`` per-row table index.
    residual : torch.Tensor
        BF16 ``[M, 5376]`` residual stream.
    out : Optional[torch.Tensor]
        BF16 ``[M, 5376]`` output (allocated when ``None``).
    act_q, act_scale : Optional[torch.Tensor]
        Optional caller-owned quantization workspaces (E4M3 ``[M, 7168]``, FP32 ``[M]``).

    Returns
    -------
    torch.Tensor
        The BF16 ``[M, 5376]`` post-attention hidden state.
    """

    rows, device, out = _check_common(attn_out, gate, gate_index, residual, out)
    _require(
        "o_weight_q",
        o_weight_q,
        torch.float8_e4m3fn,
        (MINIMAX_H3_HIDDEN, MINIMAX_H3_ATTN_DIM),
        device,
    )
    _require(
        "o_weight_scale", o_weight_scale, torch.float32, (MINIMAX_H3_HIDDEN,), device
    )
    if act_q is None:
        act_q = torch.empty(
            (rows, MINIMAX_H3_ATTN_DIM), dtype=torch.float8_e4m3fn, device=device
        )
    if act_scale is None:
        act_scale = torch.empty((rows,), dtype=torch.float32, device=device)
    _require("act_q", act_q, torch.float8_e4m3fn, (rows, MINIMAX_H3_ATTN_DIM), device)
    _require("act_scale", act_scale, torch.float32, (rows,), device)
    _fp8_impl(
        attn_out,
        o_weight_q,
        o_weight_scale,
        gate,
        gate_index,
        residual,
        act_q,
        act_scale,
        out,
    )
    return out


@supported_compute_capability([120])
@flashinfer_api
def minimax_h3_nvfp4_out_proj(
    attn_out: torch.Tensor,
    o_weight_q: torch.Tensor,
    o_weight_sf: torch.Tensor,
    o_weight_global_scale: Scalar,
    act_global_scale: torch.Tensor,
    gate: torch.Tensor,
    gate_index: torch.Tensor,
    residual: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
    act_q: Optional[torch.Tensor] = None,
    act_sf: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""MiniMax-H3 NVFP4 (W4A4) attention output projection with the fused gated residual for
    SM120 (GB202: RTX 5090 / RTX PRO 6000 Blackwell).

    One launch: FlashInfer-convention block-16 NVFP4 quantization of ``attn_out``
    (``sf = UE4M3(block_amax * act_global_scale / 6)``, ``code = RN(x * act_global_scale / sf)``)
    run by the spare warps of the persistent kernel's producer warpgroup and overlapped, through
    per-tile ready flags, with the ``kind::mxf4nvf4`` block-scaled ``mma.sync`` GEMM with the fused
    ``out = BF16(residual + BF16(gate[gate_index[m]] * o))`` epilogue.  Rows whose ``gate_index``
    lies outside ``[0, 9)`` contribute ``gate = 0`` (``out = residual``).

    Parameters
    ----------
    attn_out : torch.Tensor
        BF16 ``[M, 7168]`` packed attention output.
    o_weight_q : torch.Tensor
        uint8 ``[5376, 3584]`` E2M1x2 weight (see :func:`quantize_minimax_h3_o_weight_nvfp4`).
    o_weight_sf : torch.Tensor
        uint8 UE4M3 block-16 weight scales in the FlashInfer 128x4 swizzled layout.
    o_weight_global_scale : float or torch.Tensor
        FP32 per-tensor weight global scale ``448 * 6 / amax(W)``.
    act_global_scale : torch.Tensor
        FP32 ``[1]`` CUDA tensor, the calibrated activation global scale ``448 * 6 / amax``.
    gate, gate_index, residual, out :
        As in :func:`minimax_h3_fp8_out_proj`.
    act_q, act_sf : Optional[torch.Tensor]
        Optional caller-owned quantization workspaces (uint8 ``[M, 3584]``, uint8 ``[M, 448]``).

    Returns
    -------
    torch.Tensor
        The BF16 ``[M, 5376]`` post-attention hidden state.
    """

    rows, device, out = _check_common(attn_out, gate, gate_index, residual, out)
    _require(
        "o_weight_q",
        o_weight_q,
        torch.uint8,
        (MINIMAX_H3_HIDDEN, MINIMAX_H3_ATTN_DIM // 2),
        device,
    )
    if (
        not isinstance(o_weight_sf, torch.Tensor)
        or o_weight_sf.dtype != torch.uint8
        or o_weight_sf.numel() != MINIMAX_H3_HIDDEN * _ATTN_SF
        or not o_weight_sf.is_cuda
        or o_weight_sf.device != device
        or not o_weight_sf.is_contiguous()
    ):
        raise ValueError(
            "o_weight_sf must be a contiguous uint8 CUDA tensor with 5376 * 448 entries "
            "(FlashInfer 128x4 swizzled layout)"
        )
    if isinstance(o_weight_global_scale, torch.Tensor):
        if o_weight_global_scale.numel() != 1:
            raise ValueError("o_weight_global_scale must be a single value")
        w_gs = float(o_weight_global_scale.item())
    else:
        w_gs = float(o_weight_global_scale)
    act_gs = act_global_scale.reshape(1).to(torch.float32).contiguous()
    _require("act_global_scale", act_gs, torch.float32, (1,), device)
    alpha = 1.0 / (float(act_gs.item()) * w_gs)
    if act_q is None:
        act_q = torch.empty(
            (rows, MINIMAX_H3_ATTN_DIM // 2), dtype=torch.uint8, device=device
        )
    if act_sf is None:
        act_sf = torch.empty((rows, _ATTN_SF), dtype=torch.uint8, device=device)
    _require("act_q", act_q, torch.uint8, (rows, MINIMAX_H3_ATTN_DIM // 2), device)
    _require("act_sf", act_sf, torch.uint8, (rows, _ATTN_SF), device)
    _nvfp4_impl(
        attn_out,
        o_weight_q,
        o_weight_sf,
        act_gs,
        gate,
        gate_index,
        residual,
        act_q,
        act_sf,
        out,
        alpha,
    )
    return out


__all__ = [
    "MINIMAX_H3_ATTN_DIM",
    "MINIMAX_H3_GATE_ROWS",
    "MINIMAX_H3_HIDDEN",
    "MINIMAX_H3_SF_BLOCK",
    "fp8_scale_from_amax",
    "minimax_h3_fp8_out_proj",
    "minimax_h3_nvfp4_out_proj",
    "nvfp4_global_scale_from_amax",
    "quantize_minimax_h3_o_weight_fp8",
    "quantize_minimax_h3_o_weight_nvfp4",
]
