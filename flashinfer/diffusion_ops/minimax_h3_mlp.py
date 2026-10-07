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
from typing import Dict, Optional, Tuple, Union

import torch

from ..jit.minimax_h3_mlp import (
    MiniMaxH3MlpTarget,
    gen_minimax_h3_mlp_module,
    minimax_h3_mlp_target,
)
from ..utils import get_compute_capability, register_custom_op, register_fake_op
from .minimax_h3_fc1_swiglu import (
    MINIMAX_H3_ADALN_ROWS,
    MINIMAX_H3_EPS,
    MINIMAX_H3_FC1_ROWS,
    MINIMAX_H3_FFN,
    MINIMAX_H3_HIDDEN,
    MXFP8_BLOCK,
    MXFP8_FC1_SCALE_TILE_BYTES,
    NVFP4_BLOCK,
    NVFP4_FC1_SCALE_TILE_BYTES,
    NVFP4_PACKED_COLS,
    _check_norm_inputs,
    _check_table,
    _check_tensor,
    _scalar_f32,
    _scale_tiles,
    _scale_workspace,
    _unswizzle_sf_128x4,
    mxfp8_activation_scale_workspace_bytes,
    nvfp4_activation_scale_workspace_bytes,
)
from .minimax_h3_out_proj import _weight_scale_tiles

# Production default table row count (3 modulation rows x 3 timestep groups) used by the tests
# and benchmarks.  It is NOT a validation bound: the operators accept ``[rows, 5376]`` tables
# with any ``rows >= 1`` and read the row count and row stride from the tensors.
MINIMAX_H3_GATE_ROWS = MINIMAX_H3_ADALN_ROWS
# The sequence-parallel (Ulysses) degree only labels a row configuration of the MLP: the
# operator consumes the local ``[M, 5376]`` rows of one rank and needs no receive layout.
MINIMAX_H3_SEQUENCE_PARALLEL_DEGREES = (1, 2, 4, 8)

# GEMM tiling facts the host-side workspace sizing depends on.
_BLOCK_M = 128
_CTA_GROUP = 2
_FC2_BLOCK_N = 256  # output columns per CTA pair of the FC2 GEMM
_FC2_N_TILES = MINIMAX_H3_HIDDEN // _FC2_BLOCK_N  # 21

# FlashInfer 128x4 swizzled scale-factor tiles: 512 bytes = 128 rows x 4 K-blocks, byte offset
# (row % 32) * 16 + (row // 32) * 4 + kblock inside the tile; tiles ordered (row block, K set).
_SF_TILE_ROWS = 128
_SF_TILE_BYTES = 512
# FC1 activation ``a`` (K = 5376): the CAKE-611 norm kernels' layout (see minimax_h3_fc1_swiglu).
MXFP8_A_SF_COLS = MINIMAX_H3_HIDDEN // MXFP8_BLOCK  # 168 UE8M0 scales per row
MXFP8_A_SF_K_TILES = MXFP8_A_SF_COLS // 4  # 42 tiles per 128-row block
NVFP4_A_PACKED_COLS = NVFP4_PACKED_COLS  # 2688 E2M1 nibble pairs per row
NVFP4_A_SF_COLS = MINIMAX_H3_HIDDEN // NVFP4_BLOCK  # 336 UE4M3 scales per row
NVFP4_A_SF_K_TILES = NVFP4_A_SF_COLS // 4  # 84 tiles per 128-row block
# FC2 activation ``y`` (K = 14336): written by the FC1 epilogues, streamed by the FC2 GEMM.
MXFP8_Y_SF_COLS = MINIMAX_H3_FFN // MXFP8_BLOCK  # 448 UE8M0 scales per row
MXFP8_Y_SF_K_TILES = MXFP8_Y_SF_COLS // 4  # 112 tiles per 128-row block
NVFP4_Y_PACKED_COLS = MINIMAX_H3_FFN // 2  # 7168 E2M1 nibble pairs per row
NVFP4_Y_SF_COLS = MINIMAX_H3_FFN // NVFP4_BLOCK  # 896 UE4M3 scales per row
NVFP4_Y_SF_K_TILES = NVFP4_Y_SF_COLS // 4  # 224 tiles per 128-row block
# Combined 256-row FC2 weight scale tiles: [n_tile][k_set][half][512 bytes] (21 column tiles).
MXFP8_FC2_SCALE_TILE_BYTES = _FC2_N_TILES * MXFP8_Y_SF_K_TILES * 2 * _SF_TILE_BYTES
NVFP4_FC2_SCALE_TILE_BYTES = _FC2_N_TILES * NVFP4_Y_SF_K_TILES * 2 * _SF_TILE_BYTES

# FC2 tail split-K workspace: one slot per split tile, [slot][2 CTAs][128 rows][256 cols] FP32
# partial sums and [slot][2 CTAs] hand-off counters; slots = (SMs // 2) // 2 + 1 per device.
_FC2_PARTIAL_FLOATS_PER_SLOT = _CTA_GROUP * _BLOCK_M * _FC2_BLOCK_N
_FC2_FLAGS_PER_SLOT = _CTA_GROUP


@functools.lru_cache(maxsize=None)
def _get_module(target: MiniMaxH3MlpTarget):
    return gen_minimax_h3_mlp_module(target).build_and_load()


def _module_for(device: torch.device):
    return _get_module(minimax_h3_mlp_target(get_compute_capability(device)))


def _m_tiles(rows: int) -> int:
    tiles = (rows + _BLOCK_M - 1) // _BLOCK_M
    return tiles + tiles % _CTA_GROUP


def mxfp8_a_scale_workspace_bytes(rows: int) -> int:
    """Bytes of the swizzled MXFP8 scale workspace of the modulated activation ``a`` (K = 5376)
    for ``rows`` activation rows (128-row tiles padded to the even tile count the paired GEMM
    consumes)."""
    return mxfp8_activation_scale_workspace_bytes(rows)


def nvfp4_a_scale_workspace_bytes(rows: int) -> int:
    """Bytes of the swizzled NVFP4 scale workspace of the modulated activation ``a`` (K = 5376)."""
    return nvfp4_activation_scale_workspace_bytes(rows)


def mxfp8_y_scale_workspace_bytes(rows: int) -> int:
    """Bytes of the swizzled MXFP8 scale workspace of the SwiGLU output ``y`` (K = 14336) for
    ``rows`` activation rows (row tiles padded to the even count the paired FC2 GEMM consumes)."""
    return _m_tiles(int(rows)) * MXFP8_Y_SF_K_TILES * _SF_TILE_BYTES


def nvfp4_y_scale_workspace_bytes(rows: int) -> int:
    """Bytes of the swizzled NVFP4 scale workspace of the SwiGLU output ``y`` (K = 14336)."""
    return _m_tiles(int(rows)) * NVFP4_Y_SF_K_TILES * _SF_TILE_BYTES


# --------------------------------------------------------------------------------------------
# FC2 tail split-K workspace (per device)
# --------------------------------------------------------------------------------------------

_FC2_TAIL_WORKSPACE: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}


def _fc2_tail_workspace(device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-device FP32 partial-sum buffer and hand-off counters of the FC2 tail split-K, shared
    by the three variants.  The counters start at zero and every launch that uses them re-arms
    them to zero (CUDA-graph replay safe); FC2 launches of one device must not run concurrently."""
    index = device.index if device.index is not None else torch.cuda.current_device()
    workspace = _FC2_TAIL_WORKSPACE.get(index)
    if workspace is None:
        clusters = (
            torch.cuda.get_device_properties(index).multi_processor_count // _CTA_GROUP
        )
        slots = clusters // 2 + 1
        where = torch.device("cuda", index)
        partial = torch.empty(
            (slots * _FC2_PARTIAL_FLOATS_PER_SLOT,), dtype=torch.float32, device=where
        )
        # The kernel treats the counters as unsigned 32-bit words (int32 storage: DLPack-portable).
        flags = torch.zeros(
            (slots * _FC2_FLAGS_PER_SLOT,), dtype=torch.int32, device=where
        )
        workspace = (partial, flags)
        _FC2_TAIL_WORKSPACE[index] = workspace
    return workspace


# --------------------------------------------------------------------------------------------
# Argument validation
# --------------------------------------------------------------------------------------------


def _check_gate(
    gate: torch.Tensor, adaln_scale: torch.Tensor, device: torch.device
) -> None:
    """The gate table shares the row count and the row stride of the AdaLN tables: one int64
    index addresses all three (the engine's combined index over the shift / scale / gate column
    chunks of one modulation projection, or three contiguous ``[rows, 5376]`` tables)."""
    _check_table(gate, "gate", device)
    if gate.shape[0] != adaln_scale.shape[0] or gate.stride(0) != adaln_scale.stride(0):
        raise ValueError(
            "gate must have the row count and row stride of the AdaLN tables (one index "
            f"addresses all three), got rows {gate.shape[0]} / stride {gate.stride(0)} vs "
            f"rows {adaln_scale.shape[0]} / stride {adaln_scale.stride(0)}"
        )
    # The FC2 epilogue forms the table offset as a 32x32->64-bit multiply of the row index and
    # the row pitch: rows must not overlap and the pitch must fit 32 bits.
    if not MINIMAX_H3_HIDDEN <= gate.stride(0) < 2**32:
        raise ValueError(
            f"gate row stride must be in [{MINIMAX_H3_HIDDEN}, 2^32) elements, got {gate.stride(0)}"
        )


def _check_operands(
    x, x_norm_weight, adaln_scale, adaln_shift, adaln_index, gate, residual, eps
) -> Tuple[int, torch.device]:
    rows, device = _check_norm_inputs(
        x, x_norm_weight, adaln_scale, adaln_shift, adaln_index, eps
    )
    if adaln_scale.stride(0) != adaln_shift.stride(0):
        raise ValueError(
            "adaln_scale and adaln_shift must have the same row stride, got "
            f"{adaln_scale.stride(0)} and {adaln_shift.stride(0)}"
        )
    _check_gate(gate, adaln_scale, device)
    _check_tensor(
        residual, "residual", (rows, MINIMAX_H3_HIDDEN), torch.bfloat16, device
    )
    return rows, device


def _output(
    out: Optional[torch.Tensor], rows: int, device: torch.device
) -> torch.Tensor:
    if out is None:
        return torch.empty(
            (rows, MINIMAX_H3_HIDDEN), dtype=torch.bfloat16, device=device
        )
    _check_tensor(out, "out", (rows, MINIMAX_H3_HIDDEN), torch.bfloat16, device)
    return out


def _dense_workspace(
    value: Optional[torch.Tensor],
    name: str,
    shape: Tuple[int, int],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if value is None:
        return torch.empty(shape, dtype=dtype, device=device)
    _check_tensor(value, name, shape, dtype, device)
    return value


# --------------------------------------------------------------------------------------------
# Weight preparation (offline, torch + FlashInfer quantizers)
# --------------------------------------------------------------------------------------------


def _check_fc2_weight(fc2_weight: torch.Tensor) -> None:
    if (
        tuple(fc2_weight.shape) != (MINIMAX_H3_HIDDEN, MINIMAX_H3_FFN)
        or fc2_weight.dtype != torch.bfloat16
    ):
        raise ValueError(
            f"fc2_weight must be bfloat16 [{MINIMAX_H3_HIDDEN}, {MINIMAX_H3_FFN}]"
        )
    if not fc2_weight.is_cuda:
        raise ValueError("fc2_weight must be a CUDA tensor")


def prepare_minimax_h3_fc2_weight_mxfp8(
    fc2_weight: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Quantize the FC2 weight for :func:`minimax_h3_mlp_mxfp8`.

    ``fc2_weight`` is the BF16 ``[5376, 14336]`` ``nn.Linear`` weight.  Quantization is
    FlashInfer's :func:`~flashinfer.mxfp8_quantize` (per 32 consecutive K elements: UE8M0 scale =
    ``absmax / 448`` rounded up to a power of two, E4M3 round-to-nearest-even of ``w / scale``; an
    all-zero block keeps scale byte 0).

    Returns ``(fc2_weight_q, fc2_scale_tiles)``: the ``float8_e4m3fn`` ``[5376, 14336]`` weights
    and a flat ``uint8`` tensor of ``21 * 112 * 1024`` bytes holding the weight scales in the
    combined 256-row tile order the fused GEMM streams (output tile ``t`` covers weight rows
    ``[256 t, 256 (t + 1))``; each 4-column K set of that block is two 512-byte 128x4 tiles, rows
    0-127 then 128-255; see :func:`flashinfer.diffusion_ops.minimax_h3_out_proj._weight_scale_tiles`).
    The FC1 weight is prepared by
    :func:`~flashinfer.diffusion_ops.minimax_h3_fc1_swiglu.prepare_minimax_h3_fc1_weight_mxfp8`.
    """
    from ..quantization.fp8_quantization import mxfp8_quantize

    _check_fc2_weight(fc2_weight)
    w_q, w_sf = mxfp8_quantize(fc2_weight.contiguous(), is_sf_swizzled_layout=True)
    w_sf = w_sf.view(torch.uint8).reshape(-1)
    expected = MINIMAX_H3_HIDDEN * MXFP8_Y_SF_COLS
    if w_sf.numel() != expected:
        raise RuntimeError(
            f"mxfp8_quantize returned {w_sf.numel()} scale bytes, expected {expected}"
        )
    sf_linear = _unswizzle_sf_128x4(w_sf, MINIMAX_H3_HIDDEN, MXFP8_Y_SF_COLS)
    return w_q.view(torch.float8_e4m3fn).contiguous(), _weight_scale_tiles(sf_linear)


def prepare_minimax_h3_fc2_weight_nvfp4(
    fc2_weight: torch.Tensor, w_global_scale: Union[torch.Tensor, float]
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Quantize the FC2 weight for :func:`minimax_h3_mlp_nvfp4`.

    ``fc2_weight`` is the BF16 ``[5376, 14336]`` weight and ``w_global_scale`` its float32 global
    scale (:func:`~flashinfer.diffusion_ops.minimax_h3_fc1_swiglu.minimax_h3_nvfp4_global_scale`).
    Quantization is FlashInfer's :func:`~flashinfer.nvfp4_quantize` (per 16 consecutive K
    elements: UE4M3 scale = ``E4M3_RN(g * absmax / 6)`` saturating at 448, E2M1 round-to-nearest
    with saturation of ``w * g / scale``; an all-zero block writes scale 0 and codes 0).

    Returns ``(fc2_weight_q, fc2_scale_tiles)``: packed E2M1 ``uint8`` ``[5376, 7168]`` weights
    (even element in the low nibble) and a flat ``uint8`` tensor of ``21 * 224 * 1024`` bytes with
    the weight scales in the combined 256-row tile order.  The FC1 weight is prepared by
    :func:`~flashinfer.diffusion_ops.minimax_h3_fc1_swiglu.prepare_minimax_h3_fc1_weight_nvfp4`
    (on the SM100 / SM103 device that runs the operator).
    """
    from ..quantization.fp4_quantization import nvfp4_quantize
    from ..tllm_enums import SfLayout

    _check_fc2_weight(fc2_weight)
    g_w = _scalar_f32(w_global_scale, "w_global_scale", fc2_weight.device)
    w_q, w_sf = nvfp4_quantize(
        fc2_weight.contiguous(), g_w, sfLayout=SfLayout.layout_128x4, do_shuffle=False
    )
    w_q = (
        w_q.view(torch.uint8)
        .reshape(MINIMAX_H3_HIDDEN, NVFP4_Y_PACKED_COLS)
        .contiguous()
    )
    w_sf = w_sf.view(torch.uint8).reshape(-1)
    expected = MINIMAX_H3_HIDDEN * NVFP4_Y_SF_COLS
    if w_sf.numel() != expected:
        raise RuntimeError(
            f"nvfp4_quantize returned {w_sf.numel()} scale bytes, expected {expected}"
        )
    sf_linear = _unswizzle_sf_128x4(w_sf, MINIMAX_H3_HIDDEN, NVFP4_Y_SF_COLS)
    return w_q, _weight_scale_tiles(sf_linear)


# --------------------------------------------------------------------------------------------
# Operators
# --------------------------------------------------------------------------------------------


@register_custom_op(
    "flashinfer::minimax_h3_mlp",
    mutates_args=("out", "workspace_a", "workspace_y", "fc2_partial", "fc2_flags"),
)
def _minimax_h3_mlp_impl(
    x: torch.Tensor,
    x_norm_weight: torch.Tensor,
    adaln_scale: torch.Tensor,
    adaln_shift: torch.Tensor,
    adaln_index: torch.Tensor,
    fc1_weight: torch.Tensor,
    fc2_weight: torch.Tensor,
    gate: torch.Tensor,
    residual: torch.Tensor,
    out: torch.Tensor,
    workspace_a: torch.Tensor,
    workspace_y: torch.Tensor,
    fc2_partial: torch.Tensor,
    fc2_flags: torch.Tensor,
    eps: float,
) -> None:
    _module_for(x.device).minimax_h3_mlp(
        x,
        x_norm_weight,
        adaln_scale,
        adaln_shift,
        adaln_index,
        fc1_weight,
        fc2_weight,
        gate,
        residual,
        out,
        workspace_a,
        workspace_y,
        fc2_partial,
        fc2_flags,
        eps,
    )


@register_fake_op("flashinfer::minimax_h3_mlp")
def _minimax_h3_mlp_fake(
    x: torch.Tensor,
    x_norm_weight: torch.Tensor,
    adaln_scale: torch.Tensor,
    adaln_shift: torch.Tensor,
    adaln_index: torch.Tensor,
    fc1_weight: torch.Tensor,
    fc2_weight: torch.Tensor,
    gate: torch.Tensor,
    residual: torch.Tensor,
    out: torch.Tensor,
    workspace_a: torch.Tensor,
    workspace_y: torch.Tensor,
    fc2_partial: torch.Tensor,
    fc2_flags: torch.Tensor,
    eps: float,
) -> None:
    pass


@register_custom_op(
    "flashinfer::minimax_h3_mlp_mxfp8",
    mutates_args=(
        "out",
        "workspace_a_q",
        "workspace_a_sf",
        "workspace_y_q",
        "workspace_y_sf",
        "fc2_partial",
        "fc2_flags",
    ),
)
def _minimax_h3_mlp_mxfp8_impl(
    x: torch.Tensor,
    x_norm_weight: torch.Tensor,
    adaln_scale: torch.Tensor,
    adaln_shift: torch.Tensor,
    adaln_index: torch.Tensor,
    fc1_weight_q: torch.Tensor,
    fc1_scale_tiles: torch.Tensor,
    fc2_weight_q: torch.Tensor,
    fc2_scale_tiles: torch.Tensor,
    gate: torch.Tensor,
    residual: torch.Tensor,
    out: torch.Tensor,
    workspace_a_q: torch.Tensor,
    workspace_a_sf: torch.Tensor,
    workspace_y_q: torch.Tensor,
    workspace_y_sf: torch.Tensor,
    fc2_partial: torch.Tensor,
    fc2_flags: torch.Tensor,
    eps: float,
) -> None:
    _module_for(x.device).minimax_h3_mlp_mxfp8(
        x,
        x_norm_weight,
        adaln_scale,
        adaln_shift,
        adaln_index,
        fc1_weight_q,
        fc1_scale_tiles,
        fc2_weight_q,
        fc2_scale_tiles,
        gate,
        residual,
        out,
        workspace_a_q,
        workspace_a_sf,
        workspace_y_q,
        workspace_y_sf,
        fc2_partial,
        fc2_flags,
        eps,
    )


@register_fake_op("flashinfer::minimax_h3_mlp_mxfp8")
def _minimax_h3_mlp_mxfp8_fake(
    x: torch.Tensor,
    x_norm_weight: torch.Tensor,
    adaln_scale: torch.Tensor,
    adaln_shift: torch.Tensor,
    adaln_index: torch.Tensor,
    fc1_weight_q: torch.Tensor,
    fc1_scale_tiles: torch.Tensor,
    fc2_weight_q: torch.Tensor,
    fc2_scale_tiles: torch.Tensor,
    gate: torch.Tensor,
    residual: torch.Tensor,
    out: torch.Tensor,
    workspace_a_q: torch.Tensor,
    workspace_a_sf: torch.Tensor,
    workspace_y_q: torch.Tensor,
    workspace_y_sf: torch.Tensor,
    fc2_partial: torch.Tensor,
    fc2_flags: torch.Tensor,
    eps: float,
) -> None:
    pass


@register_custom_op(
    "flashinfer::minimax_h3_mlp_nvfp4",
    mutates_args=(
        "out",
        "workspace_a_q",
        "workspace_a_sf",
        "workspace_y_q",
        "workspace_y_sf",
        "fc2_partial",
        "fc2_flags",
    ),
)
def _minimax_h3_mlp_nvfp4_impl(
    x: torch.Tensor,
    x_norm_weight: torch.Tensor,
    adaln_scale: torch.Tensor,
    adaln_shift: torch.Tensor,
    adaln_index: torch.Tensor,
    a_global_scale: torch.Tensor,
    fc1_weight_q: torch.Tensor,
    fc1_scale_tiles: torch.Tensor,
    alpha1: torch.Tensor,
    y_global_scale: torch.Tensor,
    fc2_weight_q: torch.Tensor,
    fc2_scale_tiles: torch.Tensor,
    alpha2: torch.Tensor,
    gate: torch.Tensor,
    residual: torch.Tensor,
    out: torch.Tensor,
    workspace_a_q: torch.Tensor,
    workspace_a_sf: torch.Tensor,
    workspace_y_q: torch.Tensor,
    workspace_y_sf: torch.Tensor,
    fc2_partial: torch.Tensor,
    fc2_flags: torch.Tensor,
    eps: float,
) -> None:
    _module_for(x.device).minimax_h3_mlp_nvfp4(
        x,
        x_norm_weight,
        adaln_scale,
        adaln_shift,
        adaln_index,
        a_global_scale,
        fc1_weight_q,
        fc1_scale_tiles,
        alpha1,
        y_global_scale,
        fc2_weight_q,
        fc2_scale_tiles,
        alpha2,
        gate,
        residual,
        out,
        workspace_a_q,
        workspace_a_sf,
        workspace_y_q,
        workspace_y_sf,
        fc2_partial,
        fc2_flags,
        eps,
    )


@register_fake_op("flashinfer::minimax_h3_mlp_nvfp4")
def _minimax_h3_mlp_nvfp4_fake(
    x: torch.Tensor,
    x_norm_weight: torch.Tensor,
    adaln_scale: torch.Tensor,
    adaln_shift: torch.Tensor,
    adaln_index: torch.Tensor,
    a_global_scale: torch.Tensor,
    fc1_weight_q: torch.Tensor,
    fc1_scale_tiles: torch.Tensor,
    alpha1: torch.Tensor,
    y_global_scale: torch.Tensor,
    fc2_weight_q: torch.Tensor,
    fc2_scale_tiles: torch.Tensor,
    alpha2: torch.Tensor,
    gate: torch.Tensor,
    residual: torch.Tensor,
    out: torch.Tensor,
    workspace_a_q: torch.Tensor,
    workspace_a_sf: torch.Tensor,
    workspace_y_q: torch.Tensor,
    workspace_y_sf: torch.Tensor,
    fc2_partial: torch.Tensor,
    fc2_flags: torch.Tensor,
    eps: float,
) -> None:
    pass


def minimax_h3_mlp(
    x: torch.Tensor,
    x_norm_weight: torch.Tensor,
    adaln_scale: torch.Tensor,
    adaln_shift: torch.Tensor,
    adaln_index: torch.Tensor,
    fc1_weight: torch.Tensor,
    fc2_weight: torch.Tensor,
    gate: torch.Tensor,
    residual: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
    workspace_a: Optional[torch.Tensor] = None,
    workspace_y: Optional[torch.Tensor] = None,
    eps: float = MINIMAX_H3_EPS,
) -> torch.Tensor:
    r"""Fused BF16 MLP block of the MiniMax-H3 video DiT: RMSNorm + indexed AdaLN + FC1 +
    SwiGLU + FC2 + indexed gate + residual (SM100 / SM103).

    Computes, with every intermediate rounded to BF16 exactly where the PyTorch module graph
    does::

        n   = BF16(RMSNorm_fp32(x, x_norm_weight, eps))                   # FP32 sum of squares, rsqrt
        a   = BF16(n * BF16(1 + adaln_scale[idx]) + adaln_shift[idx])     # idx = adaln_index[row]
        h   = BF16(a @ fc1_weight^T)                                       # FP32 accumulation
        y   = BF16(BF16(silu(h[:, :14336])) * h[:, 14336:])
        o   = BF16(y @ fc2_weight^T)                                       # FP32 accumulation
        p   = BF16(gate[idx] * o)
        out = BF16(residual + p)

    A row whose ``idx`` lies outside ``[0, rows)`` has ``a = 0`` and ``gate = 0``, so ``out ==
    residual`` bit-exactly for that row.  Three kernels run on the current stream: a
    one-warp-per-row norm kernel writing ``a`` into ``workspace_a``, a persistent 2-CTA tcgen05
    FC1 GEMM with the fused SwiGLU epilogue writing ``y`` into ``workspace_y``, and a persistent
    2-CTA FC2 GEMM (launched with programmatic dependent launch so its prologue overlaps the FC1
    tail) with the fused gate + residual epilogue.  The modulation operands are the engine's own
    tensors (column-chunk table views, int64 indices); nothing is copied on the host.

    Parameters
    ----------
    x : torch.Tensor
        Contiguous ``bfloat16`` ``[M, 5376]`` activations of this rank, ``1 <= M <= 2**24`` (any
        sequence-parallel degree; the operator sees only the local rows).
    x_norm_weight : torch.Tensor
        ``bfloat16`` ``[5376]`` RMSNorm weight.
    adaln_scale, adaln_shift, gate : torch.Tensor
        ``bfloat16`` ``[rows, 5376]`` tables with the same ``rows >= 1`` and the same row
        stride.  ``stride(1)`` must be ``1``; ``stride(0)`` may exceed ``5376`` (for example
        ``6 * 5376`` for the shift / scale / gate column chunks of the engine's
        ``[rows, 6 * 5376]`` modulation projection) and must be a multiple of 8 elements (16
        bytes) in ``[5376, 2**32)``; the data pointers must be 16-byte aligned.  The tensors
        are passed to the kernels as they are, with their row count and row stride.
    adaln_index : torch.Tensor
        Contiguous ``int64`` ``[M]`` table row per activation row, shared by the AdaLN
        modulation and the output gate.  Values in ``[0, rows)`` select a table row; any other
        int64 value makes the modulated activation row zero and the gate zero
        (``out = residual``).
    fc1_weight : torch.Tensor
        Contiguous ``bfloat16`` ``[28672, 5376]`` fused FC1 weight (gate rows ``[0, 14336)``
        then up rows ``[14336, 28672)``).
    fc2_weight : torch.Tensor
        Contiguous ``bfloat16`` ``[5376, 14336]`` FC2 weight (``nn.Linear`` layout).
    residual : torch.Tensor
        Contiguous ``bfloat16`` ``[M, 5376]`` residual stream.
    out : Optional[torch.Tensor]
        Optional ``bfloat16`` ``[M, 5376]`` output (allocated when omitted).  ``out`` may be
        ``residual`` itself (in-place hidden-state update).
    workspace_a : Optional[torch.Tensor]
        Optional caller-owned ``bfloat16`` ``[M, 5376]`` scratch that receives ``a``.
    workspace_y : Optional[torch.Tensor]
        Optional caller-owned ``bfloat16`` ``[M, 14336]`` scratch that receives ``y``.
    eps : float
        RMSNorm epsilon (runtime argument).

    Returns
    -------
    torch.Tensor
        ``bfloat16`` ``[M, 5376]``.
    """
    rows, device = _check_operands(
        x, x_norm_weight, adaln_scale, adaln_shift, adaln_index, gate, residual, eps
    )
    _check_tensor(
        fc1_weight,
        "fc1_weight",
        (MINIMAX_H3_FC1_ROWS, MINIMAX_H3_HIDDEN),
        torch.bfloat16,
        device,
    )
    _check_tensor(
        fc2_weight,
        "fc2_weight",
        (MINIMAX_H3_HIDDEN, MINIMAX_H3_FFN),
        torch.bfloat16,
        device,
    )
    workspace_a = _dense_workspace(
        workspace_a, "workspace_a", (rows, MINIMAX_H3_HIDDEN), torch.bfloat16, device
    )
    workspace_y = _dense_workspace(
        workspace_y, "workspace_y", (rows, MINIMAX_H3_FFN), torch.bfloat16, device
    )
    out = _output(out, rows, device)
    fc2_partial, fc2_flags = _fc2_tail_workspace(device)
    _minimax_h3_mlp_impl(
        x,
        x_norm_weight,
        adaln_scale,
        adaln_shift,
        adaln_index,
        fc1_weight,
        fc2_weight,
        gate,
        residual,
        out,
        workspace_a,
        workspace_y,
        fc2_partial,
        fc2_flags,
        float(eps),
    )
    return out


def minimax_h3_mlp_mxfp8(
    x: torch.Tensor,
    x_norm_weight: torch.Tensor,
    adaln_scale: torch.Tensor,
    adaln_shift: torch.Tensor,
    adaln_index: torch.Tensor,
    fc1_weight_q: torch.Tensor,
    fc1_scale_tiles: torch.Tensor,
    fc2_weight_q: torch.Tensor,
    fc2_scale_tiles: torch.Tensor,
    gate: torch.Tensor,
    residual: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
    workspace_a_q: Optional[torch.Tensor] = None,
    workspace_a_sf: Optional[torch.Tensor] = None,
    workspace_y_q: Optional[torch.Tensor] = None,
    workspace_y_sf: Optional[torch.Tensor] = None,
    eps: float = MINIMAX_H3_EPS,
) -> torch.Tensor:
    r"""MXFP8 (W8A8, E4M3 + UE8M0 per-32 block scales) variant of :func:`minimax_h3_mlp`.

    The norm kernel computes the same BF16 modulated activation ``a`` and quantizes it exactly as
    FlashInfer's :func:`~flashinfer.mxfp8_quantize` does (per 32 consecutive K elements: UE8M0
    scale = ``absmax / 448`` rounded **up** to a power of two, E4M3 round-to-nearest-even; an
    all-zero block keeps scale byte 0).  The FC1 GEMM accumulates the block-scaled products in
    FP32, applies the SwiGLU round points and quantizes ``y`` in registers with the same recipe
    (no standalone quantization pass), and the FC2 GEMM consumes that operand::

        h   = BF16(dequant(a_q, a_sf) @ dequant(fc1_weight_q, fc1_scale_tiles)^T)
        y   = BF16(BF16(silu(h[:, :14336])) * h[:, 14336:])
        y_q, y_sf = mxfp8_quantize(y)
        o   = BF16(dequant(y_q, y_sf) @ dequant(fc2_weight_q, fc2_scale_tiles)^T)
        out = BF16(residual + BF16(gate[idx] * o))

    Parameters
    ----------
    x, x_norm_weight, adaln_scale, adaln_shift, adaln_index, gate, residual, out, eps
        As in :func:`minimax_h3_mlp`.
    fc1_weight_q : torch.Tensor
        ``float8_e4m3fn`` ``[28672, 5376]`` weights from
        :func:`~flashinfer.diffusion_ops.minimax_h3_fc1_swiglu.prepare_minimax_h3_fc1_weight_mxfp8`
        (gate rows then up rows).
    fc1_scale_tiles : torch.Tensor
        Flat ``uint8`` ``[112 * 42 * 1024]`` FC1 weight scale tiles from the same call.
    fc2_weight_q : torch.Tensor
        ``float8_e4m3fn`` ``[5376, 14336]`` weights from :func:`prepare_minimax_h3_fc2_weight_mxfp8`.
    fc2_scale_tiles : torch.Tensor
        Flat ``uint8`` ``[21 * 112 * 1024]`` FC2 weight scale tiles from the same call.
    workspace_a_q : Optional[torch.Tensor]
        Optional caller-owned ``float8_e4m3fn`` ``[M, 5376]`` buffer that receives ``a_q``.
    workspace_a_sf : Optional[torch.Tensor]
        Optional caller-owned ``uint8`` buffer of at least :func:`mxfp8_a_scale_workspace_bytes`
        ``(M)`` bytes that receives ``a_sf`` in the 128x4 swizzled layout (row tiles padded to an
        even count).  Allocated zeroed when omitted.
    workspace_y_q : Optional[torch.Tensor]
        Optional caller-owned ``float8_e4m3fn`` ``[M, 14336]`` buffer that receives ``y_q``.
    workspace_y_sf : Optional[torch.Tensor]
        Optional caller-owned ``uint8`` buffer of at least :func:`mxfp8_y_scale_workspace_bytes`
        ``(M)`` bytes that receives ``y_sf`` in the 128x4 swizzled layout.  Allocated zeroed when
        omitted.

    Returns
    -------
    torch.Tensor
        ``bfloat16`` ``[M, 5376]``.
    """
    rows, device = _check_operands(
        x, x_norm_weight, adaln_scale, adaln_shift, adaln_index, gate, residual, eps
    )
    _check_tensor(
        fc1_weight_q,
        "fc1_weight_q",
        (MINIMAX_H3_FC1_ROWS, MINIMAX_H3_HIDDEN),
        torch.float8_e4m3fn,
        device,
    )
    fc1_scale_tiles = _scale_tiles(
        fc1_scale_tiles, "fc1_scale_tiles", MXFP8_FC1_SCALE_TILE_BYTES, device
    )
    _check_tensor(
        fc2_weight_q,
        "fc2_weight_q",
        (MINIMAX_H3_HIDDEN, MINIMAX_H3_FFN),
        torch.float8_e4m3fn,
        device,
    )
    fc2_scale_tiles = _scale_tiles(
        fc2_scale_tiles, "fc2_scale_tiles", MXFP8_FC2_SCALE_TILE_BYTES, device
    )
    workspace_a_q = _dense_workspace(
        workspace_a_q,
        "workspace_a_q",
        (rows, MINIMAX_H3_HIDDEN),
        torch.float8_e4m3fn,
        device,
    )
    workspace_a_sf = _scale_workspace(
        workspace_a_sf, "workspace_a_sf", mxfp8_a_scale_workspace_bytes(rows), device
    )
    workspace_y_q = _dense_workspace(
        workspace_y_q,
        "workspace_y_q",
        (rows, MINIMAX_H3_FFN),
        torch.float8_e4m3fn,
        device,
    )
    workspace_y_sf = _scale_workspace(
        workspace_y_sf, "workspace_y_sf", mxfp8_y_scale_workspace_bytes(rows), device
    )
    out = _output(out, rows, device)
    fc2_partial, fc2_flags = _fc2_tail_workspace(device)
    _minimax_h3_mlp_mxfp8_impl(
        x,
        x_norm_weight,
        adaln_scale,
        adaln_shift,
        adaln_index,
        fc1_weight_q,
        fc1_scale_tiles,
        fc2_weight_q,
        fc2_scale_tiles,
        gate,
        residual,
        out,
        workspace_a_q,
        workspace_a_sf,
        workspace_y_q,
        workspace_y_sf,
        fc2_partial,
        fc2_flags,
        float(eps),
    )
    return out


def minimax_h3_mlp_nvfp4(
    x: torch.Tensor,
    x_norm_weight: torch.Tensor,
    adaln_scale: torch.Tensor,
    adaln_shift: torch.Tensor,
    adaln_index: torch.Tensor,
    a_global_scale: torch.Tensor,
    fc1_weight_q: torch.Tensor,
    fc1_scale_tiles: torch.Tensor,
    alpha1: torch.Tensor,
    y_global_scale: torch.Tensor,
    fc2_weight_q: torch.Tensor,
    fc2_scale_tiles: torch.Tensor,
    alpha2: torch.Tensor,
    gate: torch.Tensor,
    residual: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
    workspace_a_q: Optional[torch.Tensor] = None,
    workspace_a_sf: Optional[torch.Tensor] = None,
    workspace_y_q: Optional[torch.Tensor] = None,
    workspace_y_sf: Optional[torch.Tensor] = None,
    eps: float = MINIMAX_H3_EPS,
) -> torch.Tensor:
    r"""NVFP4 (W4A4, E2M1 + UE4M3 per-16 block scales + FP32 global scales) variant of
    :func:`minimax_h3_mlp`.

    The norm kernel computes the BF16 modulated activation ``a`` and quantizes it with
    FlashInfer's :func:`~flashinfer.nvfp4_quantize` recipe under the static global scale
    ``a_global_scale`` (per 16 consecutive K elements ``sf = E4M3_RN(g * absmax / 6)``
    saturating at 448, codes ``E2M1_RN_saturate(a / (sf / g))`` with the same approximate
    reciprocals FlashInfer uses; an all-zero block writes ``sf = 0`` and codes 0).  The FC1 GEMM
    accumulates ``(a_q * a_sf) . (w1_q * w1_sf)`` in FP32, applies ``alpha1`` before the BF16
    round of ``h`` as ``mm_fp4`` does, applies the SwiGLU round points and quantizes ``y`` in
    registers with the same recipe under ``y_global_scale``; the FC2 GEMM consumes that operand
    and applies ``alpha2`` before the BF16 round of ``o``::

        h   = BF16(alpha1 * ((a_q * a_sf) @ (fc1_weight_q * fc1_scale_tiles)^T))
        y   = BF16(BF16(silu(h[:, :14336])) * h[:, 14336:])
        y_q, y_sf = nvfp4_quantize(y, y_global_scale)
        o   = BF16(alpha2 * ((y_q * y_sf) @ (fc2_weight_q * fc2_scale_tiles)^T))
        out = BF16(residual + BF16(gate[idx] * o))

    Parameters
    ----------
    x, x_norm_weight, adaln_scale, adaln_shift, adaln_index, gate, residual, out, eps
        As in :func:`minimax_h3_mlp`.
    a_global_scale, y_global_scale : torch.Tensor
        ``float32`` ``[1]`` static activation global scales of ``a`` and ``y`` (``448 * 6 /
        absmax`` convention, typically calibrated; see
        :func:`~flashinfer.diffusion_ops.minimax_h3_fc1_swiglu.minimax_h3_nvfp4_global_scale`).
    fc1_weight_q : torch.Tensor
        Packed E2M1 ``uint8`` ``[28672, 2688]`` weights from
        :func:`~flashinfer.diffusion_ops.minimax_h3_fc1_swiglu.prepare_minimax_h3_fc1_weight_nvfp4`
        prepared on an SM100 / SM103 device (gate rows then up rows).
    fc1_scale_tiles : torch.Tensor
        Flat ``uint8`` ``[128 * 84 * 1024]`` FC1 weight scale tiles from the same call.
    alpha1 : torch.Tensor
        ``float32`` ``[1]`` = ``1 / (a_global_scale * w1_global_scale)``
        (:func:`~flashinfer.diffusion_ops.minimax_h3_fc1_swiglu.minimax_h3_nvfp4_alpha`).
    fc2_weight_q : torch.Tensor
        Packed E2M1 ``uint8`` ``[5376, 7168]`` weights from :func:`prepare_minimax_h3_fc2_weight_nvfp4`.
    fc2_scale_tiles : torch.Tensor
        Flat ``uint8`` ``[21 * 224 * 1024]`` FC2 weight scale tiles from the same call.
    alpha2 : torch.Tensor
        ``float32`` ``[1]`` = ``1 / (y_global_scale * w2_global_scale)``.
    workspace_a_q : Optional[torch.Tensor]
        Optional caller-owned ``uint8`` ``[M, 2688]`` buffer that receives the packed E2M1 ``a``.
    workspace_a_sf : Optional[torch.Tensor]
        Optional caller-owned ``uint8`` buffer of at least :func:`nvfp4_a_scale_workspace_bytes`
        ``(M)`` bytes that receives ``a_sf`` in the 128x4 swizzled layout.  Allocated zeroed when
        omitted.
    workspace_y_q : Optional[torch.Tensor]
        Optional caller-owned ``uint8`` ``[M, 7168]`` buffer that receives the packed E2M1 ``y``.
    workspace_y_sf : Optional[torch.Tensor]
        Optional caller-owned ``uint8`` buffer of at least :func:`nvfp4_y_scale_workspace_bytes`
        ``(M)`` bytes that receives ``y_sf`` in the 128x4 swizzled layout.  Allocated zeroed when
        omitted.

    Returns
    -------
    torch.Tensor
        ``bfloat16`` ``[M, 5376]``.
    """
    rows, device = _check_operands(
        x, x_norm_weight, adaln_scale, adaln_shift, adaln_index, gate, residual, eps
    )
    a_global_scale = _scalar_f32(a_global_scale, "a_global_scale", device)
    alpha1 = _scalar_f32(alpha1, "alpha1", device)
    y_global_scale = _scalar_f32(y_global_scale, "y_global_scale", device)
    alpha2 = _scalar_f32(alpha2, "alpha2", device)
    _check_tensor(
        fc1_weight_q,
        "fc1_weight_q",
        (MINIMAX_H3_FC1_ROWS, NVFP4_A_PACKED_COLS),
        torch.uint8,
        device,
    )
    fc1_scale_tiles = _scale_tiles(
        fc1_scale_tiles, "fc1_scale_tiles", NVFP4_FC1_SCALE_TILE_BYTES, device
    )
    _check_tensor(
        fc2_weight_q,
        "fc2_weight_q",
        (MINIMAX_H3_HIDDEN, NVFP4_Y_PACKED_COLS),
        torch.uint8,
        device,
    )
    fc2_scale_tiles = _scale_tiles(
        fc2_scale_tiles, "fc2_scale_tiles", NVFP4_FC2_SCALE_TILE_BYTES, device
    )
    workspace_a_q = _dense_workspace(
        workspace_a_q, "workspace_a_q", (rows, NVFP4_A_PACKED_COLS), torch.uint8, device
    )
    workspace_a_sf = _scale_workspace(
        workspace_a_sf, "workspace_a_sf", nvfp4_a_scale_workspace_bytes(rows), device
    )
    workspace_y_q = _dense_workspace(
        workspace_y_q, "workspace_y_q", (rows, NVFP4_Y_PACKED_COLS), torch.uint8, device
    )
    workspace_y_sf = _scale_workspace(
        workspace_y_sf, "workspace_y_sf", nvfp4_y_scale_workspace_bytes(rows), device
    )
    out = _output(out, rows, device)
    fc2_partial, fc2_flags = _fc2_tail_workspace(device)
    _minimax_h3_mlp_nvfp4_impl(
        x,
        x_norm_weight,
        adaln_scale,
        adaln_shift,
        adaln_index,
        a_global_scale,
        fc1_weight_q,
        fc1_scale_tiles,
        alpha1,
        y_global_scale,
        fc2_weight_q,
        fc2_scale_tiles,
        alpha2,
        gate,
        residual,
        out,
        workspace_a_q,
        workspace_a_sf,
        workspace_y_q,
        workspace_y_sf,
        fc2_partial,
        fc2_flags,
        float(eps),
    )
    return out


__all__ = [
    "MINIMAX_H3_GATE_ROWS",
    "MINIMAX_H3_SEQUENCE_PARALLEL_DEGREES",
    "MXFP8_FC2_SCALE_TILE_BYTES",
    "NVFP4_FC2_SCALE_TILE_BYTES",
    "minimax_h3_mlp",
    "minimax_h3_mlp_mxfp8",
    "minimax_h3_mlp_nvfp4",
    "mxfp8_a_scale_workspace_bytes",
    "mxfp8_y_scale_workspace_bytes",
    "nvfp4_a_scale_workspace_bytes",
    "nvfp4_y_scale_workspace_bytes",
    "prepare_minimax_h3_fc2_weight_mxfp8",
    "prepare_minimax_h3_fc2_weight_nvfp4",
]
