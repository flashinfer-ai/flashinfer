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
import math
from typing import Optional, Tuple, Union

import torch

from ..api_logging import flashinfer_api
from ..jit.cake_minimax_h3_sm120_quant_fc1_swiglu import (
    gen_minimax_h3_sm120_quant_fc1_swiglu_module,
)
from ..utils import register_custom_op, register_fake_op, supported_compute_capability

# MiniMax-H3 video DiT block dimensions served by the generated kernels.
MINIMAX_H3_HIDDEN = 5376
MINIMAX_H3_FFN = 14336
MINIMAX_H3_FC1_ROWS = 2 * MINIMAX_H3_FFN  # [gate rows; up rows] of the fused FC1 weight
MINIMAX_H3_ADALN_ROWS = 9
MINIMAX_H3_EPS = 1.0e-5
MINIMAX_H3_MAX_ROWS = 1 << 24
MINIMAX_H3_SF_BLOCK = 16
NVFP4_PACKED_COLS = MINIMAX_H3_HIDDEN // 2  # 2688 E2M1 nibble pairs per row
NVFP4_SF_COLS = MINIMAX_H3_HIDDEN // MINIMAX_H3_SF_BLOCK  # 336 UE4M3 scales per row
# SM120 prepacked FC1 row order: 8 gate rows, then the 8 up rows of the same output columns.
SM120_FC1_INTERLEAVE = 8
E4M3_MAX = 448.0
E2M1_MAX = 6.0

Scalar = Union[float, torch.Tensor]


@functools.cache
def _get_module():
    return gen_minimax_h3_sm120_quant_fc1_swiglu_module().build_and_load()


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
    if not 1 <= rows <= MINIMAX_H3_MAX_ROWS:
        raise ValueError(f"M must lie in [1, {MINIMAX_H3_MAX_ROWS}], got {rows}")
    return rows


def _check_common(
    x, x_norm_weight, adaln_scale, adaln_shift, adaln_index, eps
) -> Tuple[int, torch.device]:
    rows = _check_rows(x)
    device = x.device
    _require("x", x, torch.bfloat16, (rows, MINIMAX_H3_HIDDEN), device)
    _require(
        "x_norm_weight", x_norm_weight, torch.bfloat16, (MINIMAX_H3_HIDDEN,), device
    )
    _require(
        "adaln_scale",
        adaln_scale,
        torch.bfloat16,
        (MINIMAX_H3_ADALN_ROWS, MINIMAX_H3_HIDDEN),
        device,
    )
    _require(
        "adaln_shift",
        adaln_shift,
        torch.bfloat16,
        (MINIMAX_H3_ADALN_ROWS, MINIMAX_H3_HIDDEN),
        device,
    )
    _require("adaln_index", adaln_index, torch.int32, (rows,), device)
    eps = float(eps)
    if not (math.isfinite(eps) and eps > 0.0):
        raise ValueError(f"eps must be a positive finite float, got {eps}")
    return rows, device


def _output(
    out: Optional[torch.Tensor], rows: int, device: torch.device
) -> torch.Tensor:
    if out is None:
        return torch.empty((rows, MINIMAX_H3_FFN), dtype=torch.bfloat16, device=device)
    return _require("out", out, torch.bfloat16, (rows, MINIMAX_H3_FFN), device)


def _scalar_f32(value: Scalar, name: str, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"{name} must hold exactly one element")
        return value.to(device=device, dtype=torch.float32).reshape(1).contiguous()
    return torch.tensor([float(value)], dtype=torch.float32, device=device)


# --------------------------------------------------------------------------------------------
# Weight preparation (offline)
# --------------------------------------------------------------------------------------------


def interleave_minimax_h3_fc1_rows_sm120(t: torch.Tensor) -> torch.Tensor:
    r"""``[28672, ...]`` gate-rows-then-up-rows -> the SM120 prepacked row order.

    Prepacked row ``16 * (c // 8) + (c % 8) + 8 * is_up`` holds output column ``c`` (eight gate
    rows, then the eight up rows of the same columns), so each ``mma.sync`` n8 atom pair of the
    SM120 GEMM holds gate and up for the same output columns in the same thread.
    """
    if int(t.shape[0]) != MINIMAX_H3_FC1_ROWS:
        raise ValueError(f"expected {MINIMAX_H3_FC1_ROWS} rows, got {int(t.shape[0])}")
    groups = MINIMAX_H3_FFN // SM120_FC1_INTERLEAVE
    view = t.reshape(2, groups, SM120_FC1_INTERLEAVE, *t.shape[1:])
    return (
        view.permute(1, 0, 2, *range(3, view.ndim))
        .reshape(MINIMAX_H3_FC1_ROWS, *t.shape[1:])
        .contiguous()
    )


def deinterleave_minimax_h3_fc1_rows_sm120(t: torch.Tensor) -> torch.Tensor:
    r"""Inverse of :func:`interleave_minimax_h3_fc1_rows_sm120`."""
    if int(t.shape[0]) != MINIMAX_H3_FC1_ROWS:
        raise ValueError(f"expected {MINIMAX_H3_FC1_ROWS} rows, got {int(t.shape[0])}")
    groups = MINIMAX_H3_FFN // SM120_FC1_INTERLEAVE
    view = t.reshape(groups, 2, SM120_FC1_INTERLEAVE, *t.shape[1:])
    return (
        view.permute(1, 0, 2, *range(3, view.ndim))
        .reshape(MINIMAX_H3_FC1_ROWS, *t.shape[1:])
        .contiguous()
    )


def fp8_scale_from_amax(amax: torch.Tensor) -> torch.Tensor:
    r"""``RN(amax / 448)`` with a true IEEE division (``tensor / 448.0`` multiplies by a reciprocal
    and differs in the last FP32 bit, which flips E4M3 rounding ties)."""

    return amax / torch.full((), E4M3_MAX, dtype=torch.float32, device=amax.device)


def _check_fc1_weight(fc1_weight: torch.Tensor) -> None:
    if (
        not isinstance(fc1_weight, torch.Tensor)
        or tuple(fc1_weight.shape) != (MINIMAX_H3_FC1_ROWS, MINIMAX_H3_HIDDEN)
        or fc1_weight.dtype != torch.bfloat16
    ):
        raise ValueError(
            f"fc1_weight must be bfloat16 [{MINIMAX_H3_FC1_ROWS}, {MINIMAX_H3_HIDDEN}]"
        )
    if not fc1_weight.is_cuda:
        raise ValueError("fc1_weight must be a CUDA tensor")


def prepare_minimax_h3_fc1_weight_fp8(
    fc1_weight: torch.Tensor, chunk_rows: int = 2048
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Quantize the fused FC1 weight for :func:`minimax_h3_fc1_swiglu_fp8` (SM120).

    ``fc1_weight`` is the BF16 ``[28672, 5376]`` matrix with gate rows ``[0, 14336)`` followed by
    up rows ``[14336, 28672)``.  Each output channel (row) is quantized to E4M3 with
    ``scale = RN(max(amax(row), 1e-12) / 448)`` (true IEEE division) and ``q = RN_sat(row / scale)``,
    then the rows are permuted into the SM120 prepacked order
    (:func:`interleave_minimax_h3_fc1_rows_sm120`: eight gate rows, then the eight up rows of the
    same output columns).

    Returns ``(fc1_weight_q, fc1_weight_scale)``: ``float8_e4m3fn`` ``[28672, 5376]`` and
    ``float32`` ``[28672]`` in the prepacked row order.  The layout is specific to the SM120
    operator; it is not interchangeable with the SM100/SM103 ``prepare_minimax_h3_fc1_weight_*``
    outputs.
    """
    _check_fc1_weight(fc1_weight)
    weight_q = torch.empty(
        fc1_weight.shape, dtype=torch.float8_e4m3fn, device=fc1_weight.device
    )
    scale = torch.empty(
        (MINIMAX_H3_FC1_ROWS,), dtype=torch.float32, device=fc1_weight.device
    )
    for start in range(0, MINIMAX_H3_FC1_ROWS, int(chunk_rows)):
        stop = min(start + int(chunk_rows), MINIMAX_H3_FC1_ROWS)
        rows = fc1_weight[start:stop].float()
        row_scale = fp8_scale_from_amax(rows.abs().amax(dim=1).clamp_min(1e-12))
        scale[start:stop] = row_scale
        weight_q[start:stop] = (
            (rows / row_scale[:, None])
            .clamp(-E4M3_MAX, E4M3_MAX)
            .to(torch.float8_e4m3fn)
        )
    return (
        interleave_minimax_h3_fc1_rows_sm120(weight_q),
        interleave_minimax_h3_fc1_rows_sm120(scale),
    )


def prepare_minimax_h3_fc1_weight_nvfp4_sm120(
    fc1_weight: torch.Tensor, w_global_scale: Scalar
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Quantize the fused FC1 weight for the SM120 NVFP4 FC1+SwiGLU operator.

    ``fc1_weight`` is the BF16 ``[28672, 5376]`` matrix (gate rows then up rows) and
    ``w_global_scale`` its float32 global scale (``448 * 6 / absmax``).  The rows are first
    permuted into the SM120 prepacked order (:func:`interleave_minimax_h3_fc1_rows_sm120`) and
    then quantized with FlashInfer's :func:`~flashinfer.nvfp4_quantize`
    (``sfLayout=SfLayout.layout_128x4, do_shuffle=False``; per 16 consecutive K elements: UE4M3
    scale = ``E4M3_RN(g * absmax / 6)`` saturating at 448, E2M1 round-to-nearest with saturation
    of ``w * g / scale``).  Because the quantization is row-local, the result equals the row
    permutation of the linear quantization.

    Returns ``(fc1_weight_q, fc1_scale_tiles)``: packed E2M1 ``uint8`` ``[28672, 2688]`` weights
    (even element in the low nibble) in the prepacked row order and a flat ``uint8`` tensor of
    ``28672 * 336`` bytes holding the block scales of the prepacked rows in the FlashInfer 128x4
    swizzled layout (the GEMM streams them as 224 tiles of 168 rows x 256 bytes).  The layout is
    specific to the SM120 operator; it is not interchangeable with the SM100/SM103
    ``prepare_minimax_h3_fc1_weight_nvfp4`` output.
    """
    from ..quantization.fp4_quantization import nvfp4_quantize
    from ..tllm_enums import SfLayout

    _check_fc1_weight(fc1_weight)
    g_w = _scalar_f32(w_global_scale, "w_global_scale", fc1_weight.device)
    prepacked = interleave_minimax_h3_fc1_rows_sm120(fc1_weight)
    w_q, w_sf = nvfp4_quantize(
        prepacked, g_w, sfLayout=SfLayout.layout_128x4, do_shuffle=False
    )
    w_q = (
        w_q.view(torch.uint8)
        .reshape(MINIMAX_H3_FC1_ROWS, NVFP4_PACKED_COLS)
        .contiguous()
    )
    w_sf = w_sf.view(torch.uint8).reshape(-1).contiguous()
    expected = MINIMAX_H3_FC1_ROWS * NVFP4_SF_COLS
    if w_sf.numel() != expected:
        raise RuntimeError(
            f"nvfp4_quantize returned {w_sf.numel()} scale bytes, expected {expected}"
        )
    return w_q, w_sf


def nvfp4_activation_scale_bytes_sm120(rows: int) -> int:
    r"""Bytes of the dense row-major ``[M, 336]`` NVFP4 activation scale workspace the SM120
    operator writes (the first ``M * 336`` bytes of the caller's ``workspace_sf`` buffer)."""
    return int(rows) * NVFP4_SF_COLS


# --------------------------------------------------------------------------------------------
# Custom ops
# --------------------------------------------------------------------------------------------


@register_custom_op(
    "flashinfer::minimax_h3_sm120_fp8_fc1_swiglu",
    mutates_args=("workspace_q", "workspace_scale", "out"),
)
def _fp8_impl(
    x: torch.Tensor,
    x_norm_weight: torch.Tensor,
    adaln_scale: torch.Tensor,
    adaln_shift: torch.Tensor,
    adaln_index: torch.Tensor,
    fc1_weight_q: torch.Tensor,
    fc1_weight_scale: torch.Tensor,
    workspace_q: torch.Tensor,
    workspace_scale: torch.Tensor,
    out: torch.Tensor,
    eps: float,
) -> None:
    _get_module().minimax_h3_sm120_fp8_fc1_swiglu(
        x,
        x_norm_weight,
        adaln_scale,
        adaln_shift,
        adaln_index,
        fc1_weight_q,
        fc1_weight_scale,
        workspace_q,
        workspace_scale,
        out,
        eps,
    )


@register_fake_op("flashinfer::minimax_h3_sm120_fp8_fc1_swiglu")
def _fp8_fake(
    x,
    x_norm_weight,
    adaln_scale,
    adaln_shift,
    adaln_index,
    fc1_weight_q,
    fc1_weight_scale,
    workspace_q,
    workspace_scale,
    out,
    eps,
) -> None:
    pass


@register_custom_op(
    "flashinfer::minimax_h3_sm120_nvfp4_fc1_swiglu",
    mutates_args=("workspace_q", "workspace_sf", "out"),
)
def _nvfp4_impl(
    x: torch.Tensor,
    x_norm_weight: torch.Tensor,
    adaln_scale: torch.Tensor,
    adaln_shift: torch.Tensor,
    adaln_index: torch.Tensor,
    act_global_scale: torch.Tensor,
    fc1_weight_q: torch.Tensor,
    fc1_scale_tiles: torch.Tensor,
    workspace_q: torch.Tensor,
    workspace_sf: torch.Tensor,
    out: torch.Tensor,
    eps: float,
    alpha: float,
) -> None:
    _get_module().minimax_h3_sm120_nvfp4_fc1_swiglu(
        x,
        x_norm_weight,
        adaln_scale,
        adaln_shift,
        adaln_index,
        act_global_scale,
        fc1_weight_q,
        fc1_scale_tiles,
        workspace_q,
        workspace_sf,
        out,
        eps,
        alpha,
    )


@register_fake_op("flashinfer::minimax_h3_sm120_nvfp4_fc1_swiglu")
def _nvfp4_fake(
    x,
    x_norm_weight,
    adaln_scale,
    adaln_shift,
    adaln_index,
    act_global_scale,
    fc1_weight_q,
    fc1_scale_tiles,
    workspace_q,
    workspace_sf,
    out,
    eps,
    alpha,
) -> None:
    pass


# --------------------------------------------------------------------------------------------
# Operators
# --------------------------------------------------------------------------------------------


@supported_compute_capability([120])
@flashinfer_api
def minimax_h3_fc1_swiglu_fp8(
    x: torch.Tensor,
    x_norm_weight: torch.Tensor,
    adaln_scale: torch.Tensor,
    adaln_shift: torch.Tensor,
    adaln_index: torch.Tensor,
    fc1_weight_q: torch.Tensor,
    fc1_weight_scale: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
    workspace_q: Optional[torch.Tensor] = None,
    workspace_scale: Optional[torch.Tensor] = None,
    eps: float = MINIMAX_H3_EPS,
) -> torch.Tensor:
    r"""Fused FP8 (W8A8) RMSNorm + indexed AdaLN + FC1 GEMM + SwiGLU of the MiniMax-H3 video DiT
    block for SM120 (RTX 5090 / RTX PRO 6000 Blackwell).

    Computes, for each of the ``M`` rows (batch 1, no sequence parallelism)::

        n   = bf16(rmsnorm(x, eps) * x_norm_weight)                       # FP32 sum of squares, rsqrt
        a   = bf16(adaln_shift[i] + n * bf16(1 + adaln_scale[i]))        # i = adaln_index[row]
        a[i outside [0, 9)] = 0                                            # device-side guard
        a_q = e4m3(a / s_row),  s_row = RN(amax_row(|a|) / 448)           # per-token activation scale
        h   = bf16(a_q @ fc1_weight_q^T * s_row * fc1_weight_scale)       # FP32 accumulation, [M, 28672]
        y   = bf16(bf16(silu(h_gate)) * h_up)                              # [M, 14336]

    Two kernels run on the current stream: a one-CTA-per-row norm/AdaLN/quantization kernel that
    writes ``a_q`` and ``s_row`` into the workspaces, and a persistent ``mma.sync`` GEMM whose
    epilogue applies the dequantization scales and the SwiGLU.

    Parameters
    ----------
    x : torch.Tensor
        Contiguous ``bfloat16`` ``[M, 5376]`` hidden states, ``1 <= M <= 2**24``.
    x_norm_weight : torch.Tensor
        ``bfloat16`` ``[5376]`` RMSNorm weight.
    adaln_scale, adaln_shift : torch.Tensor
        ``bfloat16`` ``[9, 5376]`` AdaLN tables.
    adaln_index : torch.Tensor
        ``int32`` ``[M]`` table row per activation row.
    fc1_weight_q, fc1_weight_scale : torch.Tensor
        ``float8_e4m3fn`` ``[28672, 5376]`` and ``float32`` ``[28672]`` from
        :func:`prepare_minimax_h3_fc1_weight_fp8` (SM120 prepacked row order).
    out : Optional[torch.Tensor]
        Optional ``bfloat16`` ``[M, 14336]`` output (allocated when omitted).
    workspace_q : Optional[torch.Tensor]
        Optional caller-owned ``float8_e4m3fn`` ``[M, 5376]`` buffer that receives ``a_q``.
    workspace_scale : Optional[torch.Tensor]
        Optional caller-owned ``float32`` ``[M]`` buffer that receives the per-token scales.
    eps : float
        RMSNorm epsilon (positive; the operator is validated at ``1e-5``).

    Returns
    -------
    torch.Tensor
        ``bfloat16`` ``[M, 14336]``.
    """
    rows, device = _check_common(
        x, x_norm_weight, adaln_scale, adaln_shift, adaln_index, eps
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
    if workspace_q is None:
        workspace_q = torch.empty(
            (rows, MINIMAX_H3_HIDDEN), dtype=torch.float8_e4m3fn, device=device
        )
    if workspace_scale is None:
        workspace_scale = torch.empty((rows,), dtype=torch.float32, device=device)
    _require(
        "workspace_q",
        workspace_q,
        torch.float8_e4m3fn,
        (rows, MINIMAX_H3_HIDDEN),
        device,
    )
    _require("workspace_scale", workspace_scale, torch.float32, (rows,), device)
    out = _output(out, rows, device)
    _fp8_impl(
        x,
        x_norm_weight,
        adaln_scale,
        adaln_shift,
        adaln_index,
        fc1_weight_q,
        fc1_weight_scale,
        workspace_q,
        workspace_scale,
        out,
        float(eps),
    )
    return out


@supported_compute_capability([120])
@flashinfer_api
def _minimax_h3_fc1_swiglu_nvfp4_sm120(
    x: torch.Tensor,
    x_norm_weight: torch.Tensor,
    adaln_scale: torch.Tensor,
    adaln_shift: torch.Tensor,
    adaln_index: torch.Tensor,
    a_global_scale: torch.Tensor,
    fc1_weight_q: torch.Tensor,
    fc1_scale_tiles: torch.Tensor,
    alpha: Scalar,
    *,
    out: Optional[torch.Tensor] = None,
    workspace_q: Optional[torch.Tensor] = None,
    workspace_sf: Optional[torch.Tensor] = None,
    eps: float = MINIMAX_H3_EPS,
) -> torch.Tensor:
    r"""SM120 route of :func:`flashinfer.diffusion_ops.minimax_h3_fc1_swiglu_nvfp4` (same signature).

    The norm kernel computes the BF16 modulated activation ``a`` (as in
    :func:`minimax_h3_fc1_swiglu_fp8`) and quantizes it with FlashInfer's
    :func:`~flashinfer.nvfp4_quantize` recipe: per 16 consecutive K elements
    ``sf = E4M3_RN(g * absmax * rcp(6))`` saturating at 448 with ``g = a_global_scale``, codes
    ``E2M1_RN_saturate(a * rcp(sf * rcp(g)))`` (an all-zero block writes ``sf = 0`` and codes 0),
    written row-major as ``uint8`` ``[M, 2688]`` codes and ``uint8`` ``[M, 336]`` scales.  The
    block-scaled ``mma.sync`` GEMM accumulates ``(a_q * a_sf) . (w_q * w_sf)`` in FP32 and the
    epilogue applies ``alpha = 1 / (a_global_scale * w_global_scale)`` before the BF16 round::

        h = bf16(alpha * ((a_q * a_sf) @ (fc1_weight_q * fc1_scale_tiles)^T))
        y = bf16(bf16(silu(h_gate)) * h_up)

    Parameters
    ----------
    x, x_norm_weight, adaln_scale, adaln_shift, adaln_index, eps
        As in :func:`minimax_h3_fc1_swiglu_fp8`.
    a_global_scale : torch.Tensor
        ``float32`` ``[1]`` CUDA tensor, the activation global scale (``448 * 6 / absmax(a)``
        convention, typically calibrated); read on the device.
    fc1_weight_q, fc1_scale_tiles : torch.Tensor
        Packed E2M1 ``uint8`` ``[28672, 2688]`` weights and the flat ``uint8`` ``[28672 * 336]``
        swizzled block scales from :func:`prepare_minimax_h3_fc1_weight_nvfp4_sm120`.
    alpha : float or torch.Tensor
        ``1 / (a_global_scale * w_global_scale)`` (a Python float or a one-element tensor; a
        device tensor is read with a host synchronization).
    out : Optional[torch.Tensor]
        Optional ``bfloat16`` ``[M, 14336]`` output.
    workspace_q : Optional[torch.Tensor]
        Optional caller-owned ``uint8`` ``[M, 2688]`` buffer that receives the packed E2M1 activation.
    workspace_sf : Optional[torch.Tensor]
        Optional caller-owned contiguous ``uint8`` CUDA buffer of at least ``M * 336`` bytes.  On SM120
        the activation scales are **dense row-major** ``[M, 336]`` and occupy the first ``M * 336``
        bytes of the buffer (a buffer sized by
        :func:`~flashinfer.diffusion_ops.minimax_h3_fc1_swiglu.nvfp4_activation_scale_workspace_bytes`
        is always large enough); the remaining bytes are untouched.  Allocated when omitted.

    Returns
    -------
    torch.Tensor
        ``bfloat16`` ``[M, 14336]``.
    """
    rows, device = _check_common(
        x, x_norm_weight, adaln_scale, adaln_shift, adaln_index, eps
    )
    if not isinstance(a_global_scale, torch.Tensor):
        raise ValueError("a_global_scale must be a float32 [1] CUDA tensor")
    a_global_scale = _require(
        "a_global_scale", a_global_scale.reshape(1), torch.float32, (1,), device
    )
    if isinstance(alpha, torch.Tensor):
        if alpha.numel() != 1:
            raise ValueError("alpha must hold exactly one element")
        alpha = float(alpha.item())
    alpha = float(alpha)
    _require(
        "fc1_weight_q",
        fc1_weight_q,
        torch.uint8,
        (MINIMAX_H3_FC1_ROWS, NVFP4_PACKED_COLS),
        device,
    )
    if (
        not isinstance(fc1_scale_tiles, torch.Tensor)
        or fc1_scale_tiles.dtype != torch.uint8
        or fc1_scale_tiles.numel() != MINIMAX_H3_FC1_ROWS * NVFP4_SF_COLS
        or not fc1_scale_tiles.is_cuda
        or fc1_scale_tiles.device != device
        or not fc1_scale_tiles.is_contiguous()
    ):
        raise ValueError(
            f"fc1_scale_tiles must be a contiguous uint8 CUDA tensor with "
            f"{MINIMAX_H3_FC1_ROWS * NVFP4_SF_COLS} entries (SM120 prepacked rows, 128x4 swizzled layout)"
        )
    fc1_scale_tiles = fc1_scale_tiles.reshape(-1)
    if workspace_q is None:
        workspace_q = torch.empty(
            (rows, NVFP4_PACKED_COLS), dtype=torch.uint8, device=device
        )
    _require("workspace_q", workspace_q, torch.uint8, (rows, NVFP4_PACKED_COLS), device)
    sf_bytes = nvfp4_activation_scale_bytes_sm120(rows)
    if workspace_sf is None:
        workspace_sf = torch.empty(
            (rows, NVFP4_SF_COLS), dtype=torch.uint8, device=device
        )
    else:
        if (
            not isinstance(workspace_sf, torch.Tensor)
            or workspace_sf.dtype != torch.uint8
        ):
            raise ValueError("workspace_sf must be a uint8 tensor")
        if (
            not workspace_sf.is_cuda
            or workspace_sf.device != device
            or not workspace_sf.is_contiguous()
        ):
            raise ValueError(
                f"workspace_sf must be a contiguous CUDA tensor on {device}"
            )
        if workspace_sf.numel() < sf_bytes:
            raise ValueError(
                f"workspace_sf must hold at least {sf_bytes} bytes, got {workspace_sf.numel()}"
            )
        # Dense [M, 336] scales in the leading bytes of the caller's buffer (same storage).
        workspace_sf = workspace_sf.reshape(-1)[:sf_bytes].view(rows, NVFP4_SF_COLS)
    out = _output(out, rows, device)
    _nvfp4_impl(
        x,
        x_norm_weight,
        adaln_scale,
        adaln_shift,
        adaln_index,
        a_global_scale,
        fc1_weight_q,
        fc1_scale_tiles,
        workspace_q,
        workspace_sf,
        out,
        float(eps),
        alpha,
    )
    return out


__all__ = [
    "E2M1_MAX",
    "E4M3_MAX",
    "MINIMAX_H3_ADALN_ROWS",
    "MINIMAX_H3_EPS",
    "MINIMAX_H3_FC1_ROWS",
    "MINIMAX_H3_FFN",
    "MINIMAX_H3_HIDDEN",
    "MINIMAX_H3_MAX_ROWS",
    "MINIMAX_H3_SF_BLOCK",
    "NVFP4_PACKED_COLS",
    "NVFP4_SF_COLS",
    "SM120_FC1_INTERLEAVE",
    "deinterleave_minimax_h3_fc1_rows_sm120",
    "fp8_scale_from_amax",
    "interleave_minimax_h3_fc1_rows_sm120",
    "minimax_h3_fc1_swiglu_fp8",
    "nvfp4_activation_scale_bytes_sm120",
    "prepare_minimax_h3_fc1_weight_fp8",
    "prepare_minimax_h3_fc1_weight_nvfp4_sm120",
]
