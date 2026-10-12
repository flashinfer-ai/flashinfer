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

"""SM120 (GB202: RTX 5090 / RTX PRO 6000 Blackwell) FP8 / NVFP4 fused MiniMax-H3 MLP:
RMSNorm + indexed AdaLN + FC1 + SwiGLU + FC2 + gated residual."""

import functools
import math
from typing import Optional, Tuple, Union

import torch

from ..api_logging import flashinfer_api
from ..jit.cake_minimax_h3_sm120_quant_mlp import gen_minimax_h3_sm120_quant_mlp_module
from ..utils import register_custom_op, register_fake_op, supported_compute_capability
from .cake_minimax_h3_sm120_quant_fc1_swiglu import (
    E4M3_MAX,
    MINIMAX_H3_EPS,
    MINIMAX_H3_FC1_ROWS,
    MINIMAX_H3_FFN,
    MINIMAX_H3_HIDDEN,
    MINIMAX_H3_SF_BLOCK,
    NVFP4_PACKED_COLS,
    NVFP4_SF_COLS,
    fp8_scale_from_amax,
)

Scalar = Union[float, torch.Tensor]

# The FC2 quantization warps address y with 32-bit word indices (row * 7168 words).
MINIMAX_H3_MLP_MAX_ROWS = (2**31 - 1) // MINIMAX_H3_FFN
MINIMAX_H3_FC2_BLOCK_M = 256
FFN_PACKED_COLS = MINIMAX_H3_FFN // 2  # 7168 E2M1 nibble pairs per y row
FFN_SF_COLS = MINIMAX_H3_FFN // MINIMAX_H3_SF_BLOCK  # 896 UE4M3 scales per y row

FP8_MMA_FORM_AUTO = -1
FP8_MMA_FORM_LEGACY = 0
FP8_MMA_FORM_MXF8F6F4 = 2


@functools.cache
def _get_module():
    return gen_minimax_h3_sm120_quant_mlp_module().build_and_load()


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


def _check_rows(x: torch.Tensor) -> int:
    if (
        not isinstance(x, torch.Tensor)
        or x.ndim != 2
        or x.shape[1] != MINIMAX_H3_HIDDEN
    ):
        raise ValueError(f"x must be [M, {MINIMAX_H3_HIDDEN}]")
    rows = int(x.shape[0])
    if not 1 <= rows <= MINIMAX_H3_MLP_MAX_ROWS:
        raise ValueError(f"M must lie in [1, {MINIMAX_H3_MLP_MAX_ROWS}], got {rows}")
    return rows


def _check_table(name: str, table: torch.Tensor, device: torch.device) -> int:
    """A ``[rows, 5376]`` BF16 table view: unit column stride, 16-byte-aligned base and row pitch
    (contiguous tables and column chunks of a wider ``[rows, k * 5376]`` projection both qualify)."""
    if (
        not isinstance(table, torch.Tensor)
        or table.ndim != 2
        or table.shape[1] != MINIMAX_H3_HIDDEN
        or table.dtype != torch.bfloat16
    ):
        raise ValueError(f"{name} must be a bfloat16 [rows, {MINIMAX_H3_HIDDEN}] view")
    if not table.is_cuda or table.device != device:
        raise ValueError(f"{name} must be a CUDA tensor on {device}")
    rows = int(table.shape[0])
    if rows < 1:
        raise ValueError(f"{name} needs at least one row")
    if table.stride(1) != 1 or table.stride(0) % 8 != 0 or table.data_ptr() % 16 != 0:
        raise ValueError(
            f"{name} needs a unit column stride and a 16-byte-aligned base and row pitch"
        )
    return rows


def _check_common(
    x, x_norm_weight, adaln_scale, adaln_shift, adaln_index, gate, residual, eps
) -> Tuple[int, torch.device]:
    rows = _check_rows(x)
    device = x.device
    _require("x", x, torch.bfloat16, (rows, MINIMAX_H3_HIDDEN), device)
    _require(
        "x_norm_weight", x_norm_weight, torch.bfloat16, (MINIMAX_H3_HIDDEN,), device
    )
    table_rows = _check_table("adaln_scale", adaln_scale, device)
    if _check_table("adaln_shift", adaln_shift, device) != table_rows:
        raise ValueError("adaln_scale and adaln_shift must share one row count")
    if adaln_shift.stride(0) != adaln_scale.stride(0):
        raise ValueError("adaln_scale and adaln_shift must share one row stride")
    if _check_table("gate", gate, device) != table_rows:
        raise ValueError("gate must have the row count of adaln_scale")
    _require("adaln_index", adaln_index, torch.int64, (rows,), device)
    _require("residual", residual, torch.bfloat16, (rows, MINIMAX_H3_HIDDEN), device)
    eps = float(eps)
    if not (math.isfinite(eps) and eps > 0.0):
        raise ValueError(f"eps must be a positive finite float, got {eps}")
    return rows, device


def _workspace(
    name: str,
    tensor: Optional[torch.Tensor],
    dtype: torch.dtype,
    shape: Tuple[int, ...],
    device: torch.device,
) -> torch.Tensor:
    if tensor is None:
        return torch.empty(shape, dtype=dtype, device=device)
    return _require(name, tensor, dtype, shape, device)


def _flags(workspace_flags: Optional[torch.Tensor], rows: int, device) -> torch.Tensor:
    num_m_tiles = -(-rows // MINIMAX_H3_FC2_BLOCK_M)
    return _workspace(
        "workspace_flags", workspace_flags, torch.int32, (num_m_tiles,), device
    )


def _output(
    out: Optional[torch.Tensor], rows: int, device: torch.device
) -> torch.Tensor:
    if out is None:
        return torch.empty(
            (rows, MINIMAX_H3_HIDDEN), dtype=torch.bfloat16, device=device
        )
    return _require("out", out, torch.bfloat16, (rows, MINIMAX_H3_HIDDEN), device)


def _scalar_f32(value: Scalar, name: str, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"{name} must hold exactly one element")
        return value.to(device=device, dtype=torch.float32).reshape(1).contiguous()
    return torch.tensor([float(value)], dtype=torch.float32, device=device)


def _scalar_float(value: Scalar, name: str) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"{name} must hold exactly one element")
        return float(value.item())
    return float(value)


def _check_fc2_weight(fc2_weight: torch.Tensor) -> None:
    if (
        not isinstance(fc2_weight, torch.Tensor)
        or tuple(fc2_weight.shape) != (MINIMAX_H3_HIDDEN, MINIMAX_H3_FFN)
        or fc2_weight.dtype != torch.bfloat16
    ):
        raise ValueError(
            f"fc2_weight must be bfloat16 [{MINIMAX_H3_HIDDEN}, {MINIMAX_H3_FFN}]"
        )
    if not fc2_weight.is_cuda:
        raise ValueError("fc2_weight must be a CUDA tensor")


# --------------------------------------------------------------------------------------------
# Weight preparation (offline)
# --------------------------------------------------------------------------------------------


@flashinfer_api
def prepare_minimax_h3_fc2_weight_fp8(
    fc2_weight: torch.Tensor, chunk_rows: int = 1024
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Quantize the FC2 weight for :func:`minimax_h3_mlp_fp8_sm120`.

    ``fc2_weight`` is the BF16 ``nn.Linear`` matrix ``[5376, 14336]`` (output channels x FFN).  Each
    output channel (row) is quantized to E4M3 with ``scale = RN(max(amax(row), 1e-12) / 448)`` (true
    IEEE division) and ``q = RN_sat(row / scale)``; the row order is unchanged.

    Returns ``(fc2_weight_q, fc2_weight_scale)``: ``float8_e4m3fn`` ``[5376, 14336]`` and ``float32``
    ``[5376]``.
    """
    _check_fc2_weight(fc2_weight)
    weight_q = torch.empty(
        fc2_weight.shape, dtype=torch.float8_e4m3fn, device=fc2_weight.device
    )
    scale = torch.empty(
        (MINIMAX_H3_HIDDEN,), dtype=torch.float32, device=fc2_weight.device
    )
    for start in range(0, MINIMAX_H3_HIDDEN, int(chunk_rows)):
        stop = min(start + int(chunk_rows), MINIMAX_H3_HIDDEN)
        rows = fc2_weight[start:stop].float()
        row_scale = fp8_scale_from_amax(rows.abs().amax(dim=1).clamp_min(1e-12))
        scale[start:stop] = row_scale
        weight_q[start:stop] = (
            (rows / row_scale[:, None])
            .clamp(-E4M3_MAX, E4M3_MAX)
            .to(torch.float8_e4m3fn)
        )
    return weight_q, scale


@flashinfer_api
def prepare_minimax_h3_fc2_weight_nvfp4_sm120(
    fc2_weight: torch.Tensor, w_global_scale: Scalar
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Quantize the FC2 weight for :func:`minimax_h3_mlp_nvfp4_sm120`.

    ``fc2_weight`` is the BF16 ``[5376, 14336]`` matrix and ``w_global_scale`` its float32 global
    scale (``448 * 6 / absmax``).  The rows are quantized with FlashInfer's
    :func:`~flashinfer.nvfp4_quantize` (``sfLayout=SfLayout.layout_128x4, do_shuffle=False``; per 16
    consecutive K elements: UE4M3 scale = ``E4M3_RN(g * absmax / 6)`` saturating at 448, E2M1
    round-to-nearest with saturation of ``w * g / scale``).

    Returns ``(fc2_weight_q, fc2_weight_sf)``: packed E2M1 ``uint8`` ``[5376, 7168]`` weights (even
    element in the low nibble) and a flat ``uint8`` tensor of ``5376 * 896`` bytes holding the block
    scales in the FlashInfer 128x4 swizzled layout (the GEMM streams them as 42 output tiles of 448
    rows x 256 bytes).
    """
    from ..quantization.fp4_quantization import nvfp4_quantize
    from ..tllm_enums import SfLayout

    _check_fc2_weight(fc2_weight)
    g_w = _scalar_f32(w_global_scale, "w_global_scale", fc2_weight.device)
    w_q, w_sf = nvfp4_quantize(
        fc2_weight.contiguous(), g_w, sfLayout=SfLayout.layout_128x4, do_shuffle=False
    )
    w_q = w_q.view(torch.uint8).reshape(MINIMAX_H3_HIDDEN, FFN_PACKED_COLS).contiguous()
    w_sf = w_sf.view(torch.uint8).reshape(-1).contiguous()
    expected = MINIMAX_H3_HIDDEN * FFN_SF_COLS
    if w_sf.numel() != expected:
        raise RuntimeError(
            f"nvfp4_quantize returned {w_sf.numel()} scale bytes, expected {expected}"
        )
    return w_q, w_sf


def minimax_h3_mlp_fc2_flags_sm120(rows: int) -> int:
    r"""Elements of the ``int32`` FC2 tile-flag workspace the SM120 MLP needs for ``rows`` rows
    (one per 256-row tile)."""
    return -(-int(rows) // MINIMAX_H3_FC2_BLOCK_M)


# --------------------------------------------------------------------------------------------
# Custom ops
# --------------------------------------------------------------------------------------------


@register_custom_op(
    "flashinfer::minimax_h3_sm120_fp8_mlp",
    mutates_args=(
        "workspace_a_q",
        "workspace_a_scale",
        "workspace_y",
        "workspace_y_q",
        "workspace_y_scale",
        "workspace_flags",
        "out",
    ),
)
def _fp8_impl(
    x: torch.Tensor,
    x_norm_weight: torch.Tensor,
    adaln_scale: torch.Tensor,
    adaln_shift: torch.Tensor,
    adaln_index: torch.Tensor,
    gate: torch.Tensor,
    residual: torch.Tensor,
    fc1_weight_q: torch.Tensor,
    fc1_weight_scale: torch.Tensor,
    fc2_weight_q: torch.Tensor,
    fc2_weight_scale: torch.Tensor,
    workspace_a_q: torch.Tensor,
    workspace_a_scale: torch.Tensor,
    workspace_y: torch.Tensor,
    workspace_y_q: torch.Tensor,
    workspace_y_scale: torch.Tensor,
    workspace_flags: torch.Tensor,
    out: torch.Tensor,
    eps: float,
    fp8_mma_form: int,
) -> None:
    _get_module().minimax_h3_sm120_fp8_mlp(
        x,
        x_norm_weight,
        adaln_scale,
        adaln_shift,
        adaln_index,
        gate,
        residual,
        fc1_weight_q,
        fc1_weight_scale,
        fc2_weight_q,
        fc2_weight_scale,
        workspace_a_q,
        workspace_a_scale,
        workspace_y,
        workspace_y_q,
        workspace_y_scale,
        workspace_flags,
        out,
        eps,
        fp8_mma_form,
    )


@register_fake_op("flashinfer::minimax_h3_sm120_fp8_mlp")
def _fp8_fake(
    x,
    x_norm_weight,
    adaln_scale,
    adaln_shift,
    adaln_index,
    gate,
    residual,
    fc1_weight_q,
    fc1_weight_scale,
    fc2_weight_q,
    fc2_weight_scale,
    workspace_a_q,
    workspace_a_scale,
    workspace_y,
    workspace_y_q,
    workspace_y_scale,
    workspace_flags,
    out,
    eps,
    fp8_mma_form,
) -> None:
    pass


@register_custom_op(
    "flashinfer::minimax_h3_sm120_nvfp4_mlp",
    mutates_args=(
        "workspace_a_q",
        "workspace_a_sf",
        "workspace_y_q",
        "workspace_y_sf",
        "workspace_flags",
        "out",
    ),
)
def _nvfp4_impl(
    x: torch.Tensor,
    x_norm_weight: torch.Tensor,
    adaln_scale: torch.Tensor,
    adaln_shift: torch.Tensor,
    adaln_index: torch.Tensor,
    gate: torch.Tensor,
    residual: torch.Tensor,
    a_global_scale: torch.Tensor,
    fc1_weight_q: torch.Tensor,
    fc1_scale_tiles: torch.Tensor,
    y_global_scale: torch.Tensor,
    fc2_weight_q: torch.Tensor,
    fc2_weight_sf: torch.Tensor,
    workspace_a_q: torch.Tensor,
    workspace_a_sf: torch.Tensor,
    workspace_y_q: torch.Tensor,
    workspace_y_sf: torch.Tensor,
    workspace_flags: torch.Tensor,
    out: torch.Tensor,
    eps: float,
    fc1_alpha: float,
    fc2_alpha: float,
) -> None:
    _get_module().minimax_h3_sm120_nvfp4_mlp(
        x,
        x_norm_weight,
        adaln_scale,
        adaln_shift,
        adaln_index,
        gate,
        residual,
        a_global_scale,
        fc1_weight_q,
        fc1_scale_tiles,
        y_global_scale,
        fc2_weight_q,
        fc2_weight_sf,
        workspace_a_q,
        workspace_a_sf,
        workspace_y_q,
        workspace_y_sf,
        workspace_flags,
        out,
        eps,
        fc1_alpha,
        fc2_alpha,
    )


@register_fake_op("flashinfer::minimax_h3_sm120_nvfp4_mlp")
def _nvfp4_fake(
    x,
    x_norm_weight,
    adaln_scale,
    adaln_shift,
    adaln_index,
    gate,
    residual,
    a_global_scale,
    fc1_weight_q,
    fc1_scale_tiles,
    y_global_scale,
    fc2_weight_q,
    fc2_weight_sf,
    workspace_a_q,
    workspace_a_sf,
    workspace_y_q,
    workspace_y_sf,
    workspace_flags,
    out,
    eps,
    fc1_alpha,
    fc2_alpha,
) -> None:
    pass


# --------------------------------------------------------------------------------------------
# Operators
# --------------------------------------------------------------------------------------------


@supported_compute_capability([120])
@flashinfer_api
def minimax_h3_mlp_fp8_sm120(
    x: torch.Tensor,
    x_norm_weight: torch.Tensor,
    adaln_scale: torch.Tensor,
    adaln_shift: torch.Tensor,
    adaln_index: torch.Tensor,
    gate: torch.Tensor,
    residual: torch.Tensor,
    fc1_weight_q: torch.Tensor,
    fc1_weight_scale: torch.Tensor,
    fc2_weight_q: torch.Tensor,
    fc2_weight_scale: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
    workspace_a_q: Optional[torch.Tensor] = None,
    workspace_a_scale: Optional[torch.Tensor] = None,
    workspace_y: Optional[torch.Tensor] = None,
    workspace_y_q: Optional[torch.Tensor] = None,
    workspace_y_scale: Optional[torch.Tensor] = None,
    workspace_flags: Optional[torch.Tensor] = None,
    eps: float = MINIMAX_H3_EPS,
    fp8_mma_form: int = FP8_MMA_FORM_AUTO,
) -> torch.Tensor:
    r"""Fused FP8 (W8A8) MiniMax-H3 MLP block for SM120 (RTX 5090 / RTX PRO 6000 Blackwell):
    RMSNorm + indexed AdaLN + FC1 + SwiGLU + FC2 + gated residual.

    Computes, for each of the ``M`` rows (batch 1, no sequence parallelism)::

        n   = bf16(rmsnorm(x, eps) * x_norm_weight)                        # FP32 sum of squares, rsqrt
        a   = bf16(adaln_shift[i] + n * bf16(1 + adaln_scale[i]))         # i = adaln_index[row]
        a[i outside [0, rows)] = 0                                          # device-side guard
        a_q = e4m3(a / s_a),  s_a = RN(amax_row(|a|) / 448)                # per-token activation scale
        h   = bf16(a_q @ fc1_weight_q^T * s_a * fc1_weight_scale)          # FP32 accumulation, [M, 28672]
        y   = bf16(bf16(silu(h_gate)) * h_up)                               # [M, 14336]
        y_q = e4m3(y / s_y),  s_y = RN(amax_row(|y|) / 448)                # per-token, inside the FC2 launch
        o   = bf16(y_q @ fc2_weight_q^T * s_y * fc2_weight_scale)          # FP32 accumulation, [M, 5376]
        out = bf16(residual + bf16(gate[i] * o))

    Three kernels run on the current stream (plus one memset of the FC2 tile flags): the one-CTA-per-row
    norm/AdaLN/quantization kernel, the persistent FC1 GEMM with the SwiGLU epilogue, and the persistent
    FC2 GEMM whose producer warps quantize ``y`` tile by tile and whose epilogue applies the gate and the
    residual.  On GeForce GB202 boards the FP8 ``mma.sync`` is issued in the ``kind::mxf8f6f4`` form with
    unit UE8M0 scales (twice the FP32-accumulating issue rate of the dense form there); the result is
    bitwise identical to the legacy form used on the RTX PRO 6000.

    Parameters
    ----------
    x : torch.Tensor
        Contiguous ``bfloat16`` ``[M, 5376]`` hidden states, ``1 <= M <= 149796``.
    x_norm_weight : torch.Tensor
        ``bfloat16`` ``[5376]`` RMSNorm weight.
    adaln_scale, adaln_shift, gate : torch.Tensor
        ``bfloat16`` ``[rows, 5376]`` table views sharing one row count (unit column stride, 16-byte-aligned
        base and row pitch; ``adaln_scale`` and ``adaln_shift`` share one row stride).  Column chunks of a
        wider ``[rows, k * 5376]`` modulation projection qualify without a copy.
    adaln_index : torch.Tensor
        ``int64`` ``[M]`` table row per activation row; rows with an index outside ``[0, rows)`` get a
        zero activation and a zero gate.
    residual : torch.Tensor
        Contiguous ``bfloat16`` ``[M, 5376]`` residual stream.
    fc1_weight_q, fc1_weight_scale : torch.Tensor
        ``float8_e4m3fn`` ``[28672, 5376]`` and ``float32`` ``[28672]`` from
        :func:`prepare_minimax_h3_fc1_weight_fp8` (SM120 prepacked row order).
    fc2_weight_q, fc2_weight_scale : torch.Tensor
        ``float8_e4m3fn`` ``[5376, 14336]`` and ``float32`` ``[5376]`` from
        :func:`prepare_minimax_h3_fc2_weight_fp8`.
    out : Optional[torch.Tensor]
        Optional ``bfloat16`` ``[M, 5376]`` output (allocated when omitted); it may be ``residual`` or ``x``
        (the epilogue reads a row's residual before writing it).
    workspace_a_q, workspace_a_scale : Optional[torch.Tensor]
        Optional caller-owned ``float8_e4m3fn`` ``[M, 5376]`` / ``float32`` ``[M]`` buffers that receive
        the quantized activation.
    workspace_y, workspace_y_q, workspace_y_scale : Optional[torch.Tensor]
        Optional caller-owned ``bfloat16`` ``[M, 14336]`` / ``float8_e4m3fn`` ``[M, 14336]`` / ``float32``
        ``[M]`` buffers for ``y`` and its quantization.
    workspace_flags : Optional[torch.Tensor]
        Optional caller-owned ``int32`` buffer of :func:`minimax_h3_mlp_fc2_flags_sm120` ``(M)`` elements
        (the FC2 tile ready flags; zeroed by the operator).
    eps : float
        RMSNorm epsilon (positive; the operator is validated at ``1e-5``).
    fp8_mma_form : int
        ``-1`` selects the board's FP8 instruction form, ``0`` forces the legacy form, ``2`` the
        ``kind::mxf8f6f4`` form with unit scales.

    Returns
    -------
    torch.Tensor
        ``bfloat16`` ``[M, 5376]`` (``out``).
    """
    rows, device = _check_common(
        x, x_norm_weight, adaln_scale, adaln_shift, adaln_index, gate, residual, eps
    )
    _require(
        "fc1_weight_q",
        fc1_weight_q,
        torch.float8_e4m3fn,
        (MINIMAX_H3_FC1_ROWS, MINIMAX_H3_HIDDEN),
        device,
    )
    _require(
        "fc1_weight_scale",
        fc1_weight_scale,
        torch.float32,
        (MINIMAX_H3_FC1_ROWS,),
        device,
    )
    _require(
        "fc2_weight_q",
        fc2_weight_q,
        torch.float8_e4m3fn,
        (MINIMAX_H3_HIDDEN, MINIMAX_H3_FFN),
        device,
    )
    _require(
        "fc2_weight_scale",
        fc2_weight_scale,
        torch.float32,
        (MINIMAX_H3_HIDDEN,),
        device,
    )
    if int(fp8_mma_form) not in (
        FP8_MMA_FORM_AUTO,
        FP8_MMA_FORM_LEGACY,
        FP8_MMA_FORM_MXF8F6F4,
    ):
        raise ValueError(
            "fp8_mma_form must be -1 (per board), 0 (legacy) or 2 (kind::mxf8f6f4)"
        )
    workspace_a_q = _workspace(
        "workspace_a_q",
        workspace_a_q,
        torch.float8_e4m3fn,
        (rows, MINIMAX_H3_HIDDEN),
        device,
    )
    workspace_a_scale = _workspace(
        "workspace_a_scale", workspace_a_scale, torch.float32, (rows,), device
    )
    workspace_y = _workspace(
        "workspace_y", workspace_y, torch.bfloat16, (rows, MINIMAX_H3_FFN), device
    )
    workspace_y_q = _workspace(
        "workspace_y_q",
        workspace_y_q,
        torch.float8_e4m3fn,
        (rows, MINIMAX_H3_FFN),
        device,
    )
    workspace_y_scale = _workspace(
        "workspace_y_scale", workspace_y_scale, torch.float32, (rows,), device
    )
    workspace_flags = _flags(workspace_flags, rows, device)
    out = _output(out, rows, device)
    _fp8_impl(
        x,
        x_norm_weight,
        adaln_scale,
        adaln_shift,
        adaln_index,
        gate,
        residual,
        fc1_weight_q,
        fc1_weight_scale,
        fc2_weight_q,
        fc2_weight_scale,
        workspace_a_q,
        workspace_a_scale,
        workspace_y,
        workspace_y_q,
        workspace_y_scale,
        workspace_flags,
        out,
        float(eps),
        int(fp8_mma_form),
    )
    return out


@supported_compute_capability([120])
@flashinfer_api
def minimax_h3_mlp_nvfp4_sm120(
    x: torch.Tensor,
    x_norm_weight: torch.Tensor,
    adaln_scale: torch.Tensor,
    adaln_shift: torch.Tensor,
    adaln_index: torch.Tensor,
    gate: torch.Tensor,
    residual: torch.Tensor,
    a_global_scale: torch.Tensor,
    fc1_weight_q: torch.Tensor,
    fc1_scale_tiles: torch.Tensor,
    fc1_alpha: Scalar,
    y_global_scale: torch.Tensor,
    fc2_weight_q: torch.Tensor,
    fc2_weight_sf: torch.Tensor,
    fc2_alpha: Scalar,
    *,
    out: Optional[torch.Tensor] = None,
    workspace_a_q: Optional[torch.Tensor] = None,
    workspace_a_sf: Optional[torch.Tensor] = None,
    workspace_y_q: Optional[torch.Tensor] = None,
    workspace_y_sf: Optional[torch.Tensor] = None,
    workspace_flags: Optional[torch.Tensor] = None,
    eps: float = MINIMAX_H3_EPS,
) -> torch.Tensor:
    r"""Fused NVFP4 (W4A4) MiniMax-H3 MLP block for SM120 (RTX 5090 / RTX PRO 6000 Blackwell).

    The norm kernel computes the BF16 modulated activation ``a`` (as in :func:`minimax_h3_mlp_fp8_sm120`)
    and quantizes it with FlashInfer's :func:`~flashinfer.nvfp4_quantize` recipe (per 16 consecutive K
    elements ``sf = E4M3_RN(g * absmax * rcp(6))`` saturating at 448 with ``g = a_global_scale``, codes
    ``E2M1_RN_saturate(a * rcp(sf * rcp(g)))``; an all-zero block writes ``sf = 0`` and codes 0).  The FC1
    GEMM accumulates ``(a_q * a_sf) . (w_q * w_sf)`` in FP32, applies ``fc1_alpha`` before the BF16 round
    of ``h``, forms ``y = bf16(bf16(silu(h_gate)) * h_up)`` in registers and quantizes ``y`` in the same
    epilogue with the same recipe under ``g = y_global_scale`` (block-16 E2M1 codes + dense UE4M3 scales);
    the FC2 GEMM consumes those operands directly::

        h   = bf16(fc1_alpha * ((a_q * a_sf) @ (fc1_weight_q * fc1_scale_tiles)^T))
        y   = bf16(bf16(silu(h_gate)) * h_up)
        o   = bf16(fc2_alpha * ((y_q * y_sf) @ (fc2_weight_q * fc2_weight_sf)^T))
        out = bf16(residual + bf16(gate[i] * o))

    Parameters
    ----------
    x, x_norm_weight, adaln_scale, adaln_shift, adaln_index, gate, residual, eps
        As in :func:`minimax_h3_mlp_fp8_sm120`.
    a_global_scale, y_global_scale : torch.Tensor
        ``float32`` ``[1]`` CUDA tensors: the activation global scales of ``a`` and ``y``
        (``448 * 6 / absmax`` convention, typically calibrated); read on the device.
    fc1_weight_q, fc1_scale_tiles : torch.Tensor
        Packed E2M1 ``uint8`` ``[28672, 2688]`` weights and the flat ``uint8`` ``[28672 * 336]`` swizzled
        block scales from :func:`prepare_minimax_h3_fc1_weight_nvfp4_sm120`.
    fc1_alpha, fc2_alpha : float or torch.Tensor
        ``1 / (a_global_scale * w1_global_scale)`` and ``1 / (y_global_scale * w2_global_scale)`` (a Python
        float or a one-element tensor; a device tensor is read with a host synchronization).
    fc2_weight_q, fc2_weight_sf : torch.Tensor
        Packed E2M1 ``uint8`` ``[5376, 7168]`` weights and the flat ``uint8`` ``[5376 * 896]`` swizzled block
        scales from :func:`prepare_minimax_h3_fc2_weight_nvfp4_sm120`.
    out : Optional[torch.Tensor]
        Optional ``bfloat16`` ``[M, 5376]`` output; may be ``residual`` or ``x``.
    workspace_a_q, workspace_a_sf : Optional[torch.Tensor]
        Optional caller-owned ``uint8`` ``[M, 2688]`` / ``uint8`` ``[M, 336]`` buffers (packed codes and
        dense row-major block scales of ``a``).
    workspace_y_q, workspace_y_sf : Optional[torch.Tensor]
        Optional caller-owned ``uint8`` ``[M, 7168]`` / ``uint8`` ``[M, 896]`` buffers (packed codes and
        dense row-major block scales of ``y``).
    workspace_flags : Optional[torch.Tensor]
        Optional caller-owned ``int32`` buffer of :func:`minimax_h3_mlp_fc2_flags_sm120` ``(M)`` elements.

    Returns
    -------
    torch.Tensor
        ``bfloat16`` ``[M, 5376]`` (``out``).
    """
    rows, device = _check_common(
        x, x_norm_weight, adaln_scale, adaln_shift, adaln_index, gate, residual, eps
    )
    if not isinstance(a_global_scale, torch.Tensor) or not isinstance(
        y_global_scale, torch.Tensor
    ):
        raise ValueError(
            "a_global_scale and y_global_scale must be float32 [1] CUDA tensors"
        )
    a_global_scale = _require(
        "a_global_scale", a_global_scale.reshape(1), torch.float32, (1,), device
    )
    y_global_scale = _require(
        "y_global_scale", y_global_scale.reshape(1), torch.float32, (1,), device
    )
    fc1_alpha_f = _scalar_float(fc1_alpha, "fc1_alpha")
    fc2_alpha_f = _scalar_float(fc2_alpha, "fc2_alpha")
    _require(
        "fc1_weight_q",
        fc1_weight_q,
        torch.uint8,
        (MINIMAX_H3_FC1_ROWS, NVFP4_PACKED_COLS),
        device,
    )
    fc1_scale_tiles = _flat_u8(
        "fc1_scale_tiles", fc1_scale_tiles, MINIMAX_H3_FC1_ROWS * NVFP4_SF_COLS, device
    )
    _require(
        "fc2_weight_q",
        fc2_weight_q,
        torch.uint8,
        (MINIMAX_H3_HIDDEN, FFN_PACKED_COLS),
        device,
    )
    fc2_weight_sf = _flat_u8(
        "fc2_weight_sf", fc2_weight_sf, MINIMAX_H3_HIDDEN * FFN_SF_COLS, device
    )
    workspace_a_q = _workspace(
        "workspace_a_q", workspace_a_q, torch.uint8, (rows, NVFP4_PACKED_COLS), device
    )
    workspace_a_sf = _workspace(
        "workspace_a_sf", workspace_a_sf, torch.uint8, (rows, NVFP4_SF_COLS), device
    )
    workspace_y_q = _workspace(
        "workspace_y_q", workspace_y_q, torch.uint8, (rows, FFN_PACKED_COLS), device
    )
    workspace_y_sf = _workspace(
        "workspace_y_sf", workspace_y_sf, torch.uint8, (rows, FFN_SF_COLS), device
    )
    workspace_flags = _flags(workspace_flags, rows, device)
    out = _output(out, rows, device)
    _nvfp4_impl(
        x,
        x_norm_weight,
        adaln_scale,
        adaln_shift,
        adaln_index,
        gate,
        residual,
        a_global_scale,
        fc1_weight_q,
        fc1_scale_tiles,
        y_global_scale,
        fc2_weight_q,
        fc2_weight_sf,
        workspace_a_q,
        workspace_a_sf,
        workspace_y_q,
        workspace_y_sf,
        workspace_flags,
        out,
        float(eps),
        fc1_alpha_f,
        fc2_alpha_f,
    )
    return out


def _flat_u8(
    name: str, tensor: torch.Tensor, numel: int, device: torch.device
) -> torch.Tensor:
    if (
        not isinstance(tensor, torch.Tensor)
        or tensor.dtype != torch.uint8
        or tensor.numel() != numel
        or not tensor.is_cuda
        or tensor.device != device
        or not tensor.is_contiguous()
    ):
        raise ValueError(
            f"{name} must be a contiguous uint8 CUDA tensor with {numel} entries (128x4 swizzled layout)"
        )
    return tensor.reshape(-1)


__all__ = [
    "FFN_PACKED_COLS",
    "FFN_SF_COLS",
    "FP8_MMA_FORM_AUTO",
    "FP8_MMA_FORM_LEGACY",
    "FP8_MMA_FORM_MXF8F6F4",
    "MINIMAX_H3_FC2_BLOCK_M",
    "MINIMAX_H3_MLP_MAX_ROWS",
    "minimax_h3_mlp_fc2_flags_sm120",
    "minimax_h3_mlp_fp8_sm120",
    "minimax_h3_mlp_nvfp4_sm120",
    "prepare_minimax_h3_fc2_weight_fp8",
    "prepare_minimax_h3_fc2_weight_nvfp4_sm120",
]
