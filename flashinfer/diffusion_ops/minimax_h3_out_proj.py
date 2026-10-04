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

from ..jit.minimax_h3_out_proj import (
    MiniMaxH3OutProjTarget,
    gen_minimax_h3_out_proj_module,
    minimax_h3_out_proj_target,
)
from ..utils import get_compute_capability, register_custom_op, register_fake_op
from .minimax_h3_fc1_swiglu import (
    _scalar_f32,
    _swizzle_sf_128x4,
    _unswizzle_sf_128x4,
    minimax_h3_nvfp4_alpha,
    minimax_h3_nvfp4_global_scale,
)

# MiniMax-H3 video DiT attention block dimensions served by the generated kernels.
MINIMAX_H3_HIDDEN = 5376  # output width (o_weight rows)
MINIMAX_H3_NUM_HEADS = 56
MINIMAX_H3_HEAD_DIM = 128
MINIMAX_H3_ATTN_DIM = (
    MINIMAX_H3_NUM_HEADS * MINIMAX_H3_HEAD_DIM
)  # 7168, the GEMM reduction
MINIMAX_H3_GATE_ROWS = 9
MINIMAX_H3_SEQUENCE_PARALLEL_DEGREES = (1, 2, 4, 8)
MINIMAX_H3_MAX_ROWS = 1 << 24

# GEMM tiling facts the host-side workspace sizing depends on.
_BLOCK_M = 128
_CTA_GROUP = 2
_N_TILE_ROWS = 256  # output columns (= weight rows) per CTA pair
_N_TILES = MINIMAX_H3_HIDDEN // _N_TILE_ROWS  # 21

# FlashInfer 128x4 swizzled scale-factor tiles: 512 bytes = 128 rows x 4 K-blocks, byte offset
# (row % 32) * 16 + (row // 32) * 4 + kblock inside the tile; tiles ordered (row block, K set).
_SF_TILE_ROWS = 128
_SF_TILE_BYTES = 512
MXFP8_BLOCK = 32
MXFP8_SF_COLS = MINIMAX_H3_ATTN_DIM // MXFP8_BLOCK  # 224 UE8M0 scales per row
MXFP8_SF_K_TILES = MXFP8_SF_COLS // 4  # 56 tiles per 128-row block
NVFP4_BLOCK = 16
NVFP4_PACKED_COLS = MINIMAX_H3_ATTN_DIM // 2  # 3584 E2M1 nibble pairs per row
NVFP4_SF_COLS = MINIMAX_H3_ATTN_DIM // NVFP4_BLOCK  # 448 UE4M3 scales per row
NVFP4_SF_K_TILES = NVFP4_SF_COLS // 4  # 112 tiles per 128-row block
# Combined 256-row weight scale tiles: [n_tile][k_set][half][512 bytes].
MXFP8_O_SCALE_TILE_BYTES = _N_TILES * MXFP8_SF_K_TILES * 2 * _SF_TILE_BYTES
NVFP4_O_SCALE_TILE_BYTES = _N_TILES * NVFP4_SF_K_TILES * 2 * _SF_TILE_BYTES

_REFERENCE_CHUNK_ROWS = 1024


@functools.lru_cache(maxsize=None)
def _get_module(target: MiniMaxH3OutProjTarget):
    return gen_minimax_h3_out_proj_module(target).build_and_load()


def _module_for(device: torch.device):
    return _get_module(minimax_h3_out_proj_target(get_compute_capability(device)))


def _m_tiles(rows: int) -> int:
    tiles = (rows + _BLOCK_M - 1) // _BLOCK_M
    return tiles + tiles % _CTA_GROUP


def mxfp8_activation_scale_workspace_bytes(rows: int) -> int:
    """Bytes of the swizzled MXFP8 activation scale workspace for ``rows`` activation rows (128-row
    tiles padded to the even tile count the paired GEMM consumes)."""
    return _m_tiles(int(rows)) * MXFP8_SF_K_TILES * _SF_TILE_BYTES


def nvfp4_activation_scale_workspace_bytes(rows: int) -> int:
    """Bytes of the swizzled NVFP4 activation scale workspace for ``rows`` activation rows."""
    return _m_tiles(int(rows)) * NVFP4_SF_K_TILES * _SF_TILE_BYTES


# --------------------------------------------------------------------------------------------
# Argument validation
# --------------------------------------------------------------------------------------------


def _check_tensor(
    value: torch.Tensor,
    name: str,
    shape: Tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.dtype != dtype:
        raise ValueError(f"{name} must be {dtype}, got {value.dtype}")
    if tuple(value.shape) != tuple(shape):
        raise ValueError(
            f"{name} must have shape {tuple(shape)}, got {tuple(value.shape)}"
        )
    if not value.is_cuda or value.device != device:
        raise ValueError(f"{name} must be a CUDA tensor on {device}")
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _check_attn_out(attn_out: torch.Tensor) -> Tuple[int, int, torch.device]:
    """Validate the ``[P, M, 56 // P, 128]`` receive layout; returns ``(P, M, device)``."""
    if not isinstance(attn_out, torch.Tensor) or attn_out.ndim != 4:
        raise ValueError(
            "attn_out must be a rank-4 tensor [P, M, 56 // P, 128] with P in "
            f"{MINIMAX_H3_SEQUENCE_PARALLEL_DEGREES}"
        )
    degree, rows = int(attn_out.shape[0]), int(attn_out.shape[1])
    if degree not in MINIMAX_H3_SEQUENCE_PARALLEL_DEGREES:
        raise ValueError(
            f"attn_out.shape[0] (the sequence-parallel degree P) must be one of "
            f"{MINIMAX_H3_SEQUENCE_PARALLEL_DEGREES}, got {degree}"
        )
    if not 1 <= rows <= MINIMAX_H3_MAX_ROWS:
        raise ValueError(f"M must satisfy 1 <= M <= {MINIMAX_H3_MAX_ROWS}, got {rows}")
    if not attn_out.is_cuda:
        raise ValueError("attn_out must be a CUDA tensor")
    _check_tensor(
        attn_out,
        "attn_out",
        (degree, rows, MINIMAX_H3_NUM_HEADS // degree, MINIMAX_H3_HEAD_DIM),
        torch.bfloat16,
        attn_out.device,
    )
    return degree, rows, attn_out.device


def _check_epilogue_inputs(
    gate: torch.Tensor,
    gate_index: torch.Tensor,
    residual: torch.Tensor,
    rows: int,
    device: torch.device,
) -> None:
    _check_tensor(
        gate, "gate", (MINIMAX_H3_GATE_ROWS, MINIMAX_H3_HIDDEN), torch.bfloat16, device
    )
    _check_tensor(gate_index, "gate_index", (rows,), torch.int32, device)
    _check_tensor(
        residual, "residual", (rows, MINIMAX_H3_HIDDEN), torch.bfloat16, device
    )


def _output(
    out: Optional[torch.Tensor], rows: int, device: torch.device
) -> torch.Tensor:
    if out is None:
        return torch.empty(
            (rows, MINIMAX_H3_HIDDEN), dtype=torch.bfloat16, device=device
        )
    _check_tensor(out, "out", (rows, MINIMAX_H3_HIDDEN), torch.bfloat16, device)
    return out


def _scale_workspace(
    value: Optional[torch.Tensor], name: str, min_bytes: int, device: torch.device
) -> torch.Tensor:
    if value is None:
        # Rows beyond M inside the last (padded) tile are never written by the kernel; a zeroed
        # buffer keeps those scale bytes finite.
        return torch.zeros((min_bytes,), dtype=torch.uint8, device=device)
    if (
        not isinstance(value, torch.Tensor)
        or value.ndim != 1
        or value.dtype != torch.uint8
    ):
        raise ValueError(f"{name} must be a 1-D uint8 tensor")
    if not value.is_cuda or value.device != device or not value.is_contiguous():
        raise ValueError(f"{name} must be a contiguous CUDA tensor on {device}")
    if value.numel() < min_bytes:
        raise ValueError(
            f"{name} must hold at least {min_bytes} bytes, got {value.numel()}"
        )
    return value


def _scale_tiles(
    value: torch.Tensor, name: str, num_bytes: int, device: torch.device
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.dtype != torch.uint8:
        raise ValueError(f"{name} must be a uint8 tensor")
    if value.numel() != num_bytes:
        raise ValueError(
            f"{name} must hold exactly {num_bytes} bytes, got {value.numel()}"
        )
    if not value.is_cuda or value.device != device or not value.is_contiguous():
        raise ValueError(f"{name} must be a contiguous CUDA tensor on {device}")
    return value.reshape(-1)


# --------------------------------------------------------------------------------------------
# Layout helpers and pure-torch reference
# --------------------------------------------------------------------------------------------


def minimax_h3_unpack_attn_out(attn_out: torch.Tensor) -> torch.Tensor:
    r"""``[P, M, 56 // P, 128]`` receive layout -> the logical activation ``A`` ``[M, 7168]``
    with ``A[m, h * 128 + d] = attn_out[h // (56 // P), m, h % (56 // P), d]`` (a copy; the
    fused kernels never materialise it)."""
    degree, rows = int(attn_out.shape[0]), int(attn_out.shape[1])
    return (
        attn_out.reshape(degree, rows, MINIMAX_H3_ATTN_DIM // degree)
        .permute(1, 0, 2)
        .reshape(rows, MINIMAX_H3_ATTN_DIM)
    )


def _gate_residual_epilogue(
    o: torch.Tensor,
    gate: torch.Tensor,
    gate_index: torch.Tensor,
    residual: torch.Tensor,
) -> torch.Tensor:
    idx = gate_index.to(torch.int64)
    valid = (idx >= 0) & (idx < MINIMAX_H3_GATE_ROWS)
    g = gate.index_select(0, idx.clamp(0, MINIMAX_H3_GATE_ROWS - 1))
    g = torch.where(valid[:, None], g, torch.zeros_like(g))
    p = (g * o).to(torch.bfloat16)  # BF16 x BF16: FP32 product, one rounding
    return (residual + p).to(torch.bfloat16)


def _reference_from_operands(
    a: torch.Tensor,
    w: torch.Tensor,
    gate: torch.Tensor,
    gate_index: torch.Tensor,
    residual: torch.Tensor,
    alpha: Optional[float] = None,
) -> torch.Tensor:
    """``out = BF16(residual + BF16(gate[idx] * BF16(alpha * a @ w^T)))`` with an FP32 GEMM
    (TF32 disabled) over any float-convertible operands ``a`` ``[M, K]`` / ``w`` ``[N, K]``."""
    allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        w_t = w.float().t()
        o = torch.empty(
            (a.shape[0], MINIMAX_H3_HIDDEN), dtype=torch.bfloat16, device=a.device
        )
        for r0 in range(0, a.shape[0], _REFERENCE_CHUNK_ROWS):
            r1 = min(a.shape[0], r0 + _REFERENCE_CHUNK_ROWS)
            h = a[r0:r1].float() @ w_t
            if alpha is not None:
                h = h * float(alpha)
            o[r0:r1] = h.to(torch.bfloat16)
        return _gate_residual_epilogue(o, gate, gate_index, residual)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32


def minimax_h3_out_proj_reference(
    attn_out: torch.Tensor,
    o_weight: torch.Tensor,
    gate: torch.Tensor,
    gate_index: torch.Tensor,
    residual: torch.Tensor,
) -> torch.Tensor:
    r"""Pure-torch BF16 reference of :func:`minimax_h3_out_proj`: unpack the receive layout,
    FP32 GEMM (TF32 disabled) rounded to BF16, indexed gate product rounded to BF16, residual sum
    rounded to BF16."""
    return _reference_from_operands(
        minimax_h3_unpack_attn_out(attn_out), o_weight, gate, gate_index, residual
    )


def _weight_scale_tiles(sf_linear: torch.Tensor) -> torch.Tensor:
    """``[5376, C]`` linear weight scales -> flat combined tiles ``[n_tile][k_set][half][512]``.

    Output tile ``t`` (21 tiles of 256 output columns) covers weight rows ``[256 t, 256 (t + 1))``;
    each 4-column K set of that block is two 512-byte 128x4 tiles (rows 0-127, then rows 128-255),
    the order the 2-CTA block-scaled MMA reads its B scale factors in.  Byte offset of row ``n``,
    K block ``c``::

        n_tile = n // 256, half = (n % 256) // 128, r = n % 128, k_set = c // 4
        ((n_tile * k_sets + k_set) * 2 + half) * 512 + (r % 32) * 16 + (r // 32) * 4 + c % 4
    """
    if tuple(sf_linear.shape[:1]) != (MINIMAX_H3_HIDDEN,) or sf_linear.shape[1] % 4:
        raise ValueError(
            f"weight scales must be [{MINIMAX_H3_HIDDEN}, 4k], got {tuple(sf_linear.shape)}"
        )
    k_sets = int(sf_linear.shape[1]) // 4
    tiles = _swizzle_sf_128x4(
        sf_linear
    )  # (128-row tile = n_tile * 2 + half, k_set, 512 B)
    return (
        tiles.reshape(_N_TILES, 2, k_sets, _SF_TILE_BYTES)
        .permute(0, 2, 1, 3)
        .contiguous()
        .reshape(-1)
    )


# --------------------------------------------------------------------------------------------
# Weight preparation (offline, torch + FlashInfer quantizers)
# --------------------------------------------------------------------------------------------


def _check_o_weight(o_weight: torch.Tensor) -> None:
    if (
        tuple(o_weight.shape) != (MINIMAX_H3_HIDDEN, MINIMAX_H3_ATTN_DIM)
        or o_weight.dtype != torch.bfloat16
    ):
        raise ValueError(
            f"o_weight must be bfloat16 [{MINIMAX_H3_HIDDEN}, {MINIMAX_H3_ATTN_DIM}]"
        )
    if not o_weight.is_cuda:
        raise ValueError("o_weight must be a CUDA tensor")


def prepare_minimax_h3_o_weight_mxfp8(
    o_weight: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Quantize the attention output-projection weight for :func:`minimax_h3_out_proj_mxfp8`.

    ``o_weight`` is the BF16 ``[5376, 7168]`` ``nn.Linear`` weight.  Quantization is FlashInfer's
    :func:`~flashinfer.mxfp8_quantize` (per 32 consecutive K elements: UE8M0 scale = ``absmax /
    448`` rounded up to a power of two, E4M3 round-to-nearest-even of ``w / scale``; an all-zero
    block keeps scale byte 0).

    Returns ``(o_weight_q, o_scale_tiles)``: the ``float8_e4m3fn`` ``[5376, 7168]`` weights and a
    flat ``uint8`` tensor of ``21 * 56 * 1024`` bytes holding the weight scales in the combined
    256-row tile order the fused GEMM streams (see :func:`_weight_scale_tiles`).  Relies on
    ``mxfp8_quantize(..., is_sf_swizzled_layout=True)`` returning the scales in the 128x4 swizzled
    layout with rows padded to 128 and columns to a multiple of 4 (both exact for this shape).
    """
    from ..quantization.fp8_quantization import mxfp8_quantize

    _check_o_weight(o_weight)
    w_q, w_sf = mxfp8_quantize(o_weight.contiguous(), is_sf_swizzled_layout=True)
    w_sf = w_sf.view(torch.uint8).reshape(-1)
    expected = MINIMAX_H3_HIDDEN * MXFP8_SF_COLS
    if w_sf.numel() != expected:
        raise RuntimeError(
            f"mxfp8_quantize returned {w_sf.numel()} scale bytes, expected {expected}"
        )
    sf_linear = _unswizzle_sf_128x4(w_sf, MINIMAX_H3_HIDDEN, MXFP8_SF_COLS)
    return w_q.view(torch.float8_e4m3fn).contiguous(), _weight_scale_tiles(sf_linear)


def prepare_minimax_h3_o_weight_nvfp4(
    o_weight: torch.Tensor, w_global_scale: Union[torch.Tensor, float]
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Quantize the attention output-projection weight for :func:`minimax_h3_out_proj_nvfp4`.

    ``o_weight`` is the BF16 ``[5376, 7168]`` weight and ``w_global_scale`` its float32 global
    scale (:func:`~flashinfer.diffusion_ops.minimax_h3_fc1_swiglu.minimax_h3_nvfp4_global_scale`).
    Quantization is FlashInfer's :func:`~flashinfer.nvfp4_quantize` (per 16 consecutive K
    elements: UE4M3 scale = ``E4M3_RN(g * absmax / 6)`` saturating at 448, E2M1 round-to-nearest
    with saturation of ``w * g / scale``; an all-zero block writes scale 0 and codes 0).

    Returns ``(o_weight_q, o_scale_tiles)``: packed E2M1 ``uint8`` ``[5376, 3584]`` weights (even
    element in the low nibble) and a flat ``uint8`` tensor of ``21 * 112 * 1024`` bytes with the
    weight scales in the combined 256-row tile order.  Relies on
    ``nvfp4_quantize(..., sfLayout=SfLayout.layout_128x4, do_shuffle=False)`` returning the scales
    in the 128x4 swizzled layout with rows padded to 128 and columns to a multiple of 4.
    """
    from ..quantization.fp4_quantization import nvfp4_quantize
    from ..tllm_enums import SfLayout

    _check_o_weight(o_weight)
    g_w = _scalar_f32(w_global_scale, "w_global_scale", o_weight.device)
    w_q, w_sf = nvfp4_quantize(
        o_weight.contiguous(), g_w, sfLayout=SfLayout.layout_128x4, do_shuffle=False
    )
    w_q = (
        w_q.view(torch.uint8).reshape(MINIMAX_H3_HIDDEN, NVFP4_PACKED_COLS).contiguous()
    )
    w_sf = w_sf.view(torch.uint8).reshape(-1)
    expected = MINIMAX_H3_HIDDEN * NVFP4_SF_COLS
    if w_sf.numel() != expected:
        raise RuntimeError(
            f"nvfp4_quantize returned {w_sf.numel()} scale bytes, expected {expected}"
        )
    sf_linear = _unswizzle_sf_128x4(w_sf, MINIMAX_H3_HIDDEN, NVFP4_SF_COLS)
    return w_q, _weight_scale_tiles(sf_linear)


# --------------------------------------------------------------------------------------------
# Operators
# --------------------------------------------------------------------------------------------


@register_custom_op("flashinfer::minimax_h3_out_proj", mutates_args=("out",))
def _minimax_h3_out_proj_impl(
    attn_out: torch.Tensor,
    o_weight: torch.Tensor,
    gate: torch.Tensor,
    gate_index: torch.Tensor,
    residual: torch.Tensor,
    out: torch.Tensor,
) -> None:
    _module_for(attn_out.device).minimax_h3_out_proj(
        attn_out, o_weight, gate, gate_index, residual, out
    )


@register_fake_op("flashinfer::minimax_h3_out_proj")
def _minimax_h3_out_proj_fake(
    attn_out: torch.Tensor,
    o_weight: torch.Tensor,
    gate: torch.Tensor,
    gate_index: torch.Tensor,
    residual: torch.Tensor,
    out: torch.Tensor,
) -> None:
    pass


@register_custom_op(
    "flashinfer::minimax_h3_out_proj_mxfp8",
    mutates_args=("workspace_q", "workspace_sf", "out"),
)
def _minimax_h3_out_proj_mxfp8_impl(
    attn_out: torch.Tensor,
    o_weight_q: torch.Tensor,
    o_scale_tiles: torch.Tensor,
    gate: torch.Tensor,
    gate_index: torch.Tensor,
    residual: torch.Tensor,
    out: torch.Tensor,
    workspace_q: torch.Tensor,
    workspace_sf: torch.Tensor,
) -> None:
    _module_for(attn_out.device).minimax_h3_out_proj_mxfp8(
        attn_out,
        o_weight_q,
        o_scale_tiles,
        gate,
        gate_index,
        residual,
        out,
        workspace_q,
        workspace_sf,
    )


@register_fake_op("flashinfer::minimax_h3_out_proj_mxfp8")
def _minimax_h3_out_proj_mxfp8_fake(
    attn_out: torch.Tensor,
    o_weight_q: torch.Tensor,
    o_scale_tiles: torch.Tensor,
    gate: torch.Tensor,
    gate_index: torch.Tensor,
    residual: torch.Tensor,
    out: torch.Tensor,
    workspace_q: torch.Tensor,
    workspace_sf: torch.Tensor,
) -> None:
    pass


@register_custom_op(
    "flashinfer::minimax_h3_out_proj_nvfp4",
    mutates_args=("workspace_q", "workspace_sf", "out"),
)
def _minimax_h3_out_proj_nvfp4_impl(
    attn_out: torch.Tensor,
    a_global_scale: torch.Tensor,
    o_weight_q: torch.Tensor,
    o_scale_tiles: torch.Tensor,
    alpha: torch.Tensor,
    gate: torch.Tensor,
    gate_index: torch.Tensor,
    residual: torch.Tensor,
    out: torch.Tensor,
    workspace_q: torch.Tensor,
    workspace_sf: torch.Tensor,
) -> None:
    _module_for(attn_out.device).minimax_h3_out_proj_nvfp4(
        attn_out,
        a_global_scale,
        o_weight_q,
        o_scale_tiles,
        alpha,
        gate,
        gate_index,
        residual,
        out,
        workspace_q,
        workspace_sf,
    )


@register_fake_op("flashinfer::minimax_h3_out_proj_nvfp4")
def _minimax_h3_out_proj_nvfp4_fake(
    attn_out: torch.Tensor,
    a_global_scale: torch.Tensor,
    o_weight_q: torch.Tensor,
    o_scale_tiles: torch.Tensor,
    alpha: torch.Tensor,
    gate: torch.Tensor,
    gate_index: torch.Tensor,
    residual: torch.Tensor,
    out: torch.Tensor,
    workspace_q: torch.Tensor,
    workspace_sf: torch.Tensor,
) -> None:
    pass


def minimax_h3_out_proj(
    attn_out: torch.Tensor,
    o_weight: torch.Tensor,
    gate: torch.Tensor,
    gate_index: torch.Tensor,
    residual: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""Fused BF16 attention output projection + indexed gate + residual of the MiniMax-H3 video
    DiT block, consuming the sequence-parallel (Ulysses) receive layout in place (SM100 / SM103).

    ``attn_out`` is ``[P, M, 56 // P, 128]``: block ``p`` holds heads ``[p * 56 // P, (p + 1) *
    56 // P)`` of every row, so the logical activation is ``A[m, h * 128 + d] =
    attn_out[h // (56 // P), m, h % (56 // P), d]``.  The kernel's activation tensor map spans
    that layout directly; nothing is re-laid out on the host.  With every intermediate rounded to
    BF16 exactly where the PyTorch module graph does::

        o   = BF16(A @ o_weight^T)                 # FP32 accumulation
        p   = BF16(gate[gate_index[m]] * o)        # gate = 0 for an index outside [0, 9)
        out = BF16(residual + p)

    One persistent 2-CTA tcgen05 GEMM with the fused epilogue runs on the current stream.

    Parameters
    ----------
    attn_out : torch.Tensor
        Contiguous ``bfloat16`` ``[P, M, 56 // P, 128]`` with ``P`` in ``{1, 2, 4, 8}`` and
        ``1 <= M <= 2**24``.
    o_weight : torch.Tensor
        Contiguous ``bfloat16`` ``[5376, 7168]`` output-projection weight (``nn.Linear`` layout).
    gate : torch.Tensor
        ``bfloat16`` ``[9, 5376]`` gate table.
    gate_index : torch.Tensor
        ``int32`` ``[M]`` table row per activation row.
    residual : torch.Tensor
        Contiguous ``bfloat16`` ``[M, 5376]`` residual stream.
    out : Optional[torch.Tensor]
        Optional ``bfloat16`` ``[M, 5376]`` output (allocated when omitted).

    Returns
    -------
    torch.Tensor
        ``bfloat16`` ``[M, 5376]``.
    """
    _degree, rows, device = _check_attn_out(attn_out)
    _check_tensor(
        o_weight,
        "o_weight",
        (MINIMAX_H3_HIDDEN, MINIMAX_H3_ATTN_DIM),
        torch.bfloat16,
        device,
    )
    _check_epilogue_inputs(gate, gate_index, residual, rows, device)
    out = _output(out, rows, device)
    _minimax_h3_out_proj_impl(attn_out, o_weight, gate, gate_index, residual, out)
    return out


def minimax_h3_out_proj_mxfp8(
    attn_out: torch.Tensor,
    o_weight_q: torch.Tensor,
    o_scale_tiles: torch.Tensor,
    gate: torch.Tensor,
    gate_index: torch.Tensor,
    residual: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
    workspace_q: Optional[torch.Tensor] = None,
    workspace_sf: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""MXFP8 (W8A8, E4M3 + UE8M0 per-32 block scales) variant of :func:`minimax_h3_out_proj`.

    A one-warp-per-row pass reads the receive layout and quantizes the logical activation exactly
    as FlashInfer's :func:`~flashinfer.mxfp8_quantize` does (per 32 consecutive K elements: UE8M0
    scale = ``absmax / 448`` rounded **up** to a power of two, E4M3 round-to-nearest-even; an
    all-zero block keeps scale byte 0), writing the dense ``[M, 7168]`` E4M3 activation and its
    128x4 swizzled scales to the workspaces.  The GEMM accumulates the block-scaled products in
    FP32 and applies the same epilogue::

        o = BF16(dequant(a_q, a_sf) @ dequant(o_weight_q, o_scale_tiles)^T)
        out = BF16(residual + BF16(gate[gate_index[m]] * o))

    Parameters
    ----------
    attn_out, gate, gate_index, residual, out
        As in :func:`minimax_h3_out_proj`.
    o_weight_q : torch.Tensor
        ``float8_e4m3fn`` ``[5376, 7168]`` weights from :func:`prepare_minimax_h3_o_weight_mxfp8`.
    o_scale_tiles : torch.Tensor
        Flat ``uint8`` ``[21 * 56 * 1024]`` weight scale tiles from the same call.
    workspace_q : Optional[torch.Tensor]
        Optional caller-owned ``float8_e4m3fn`` ``[M, 7168]`` buffer that receives ``a_q``.
    workspace_sf : Optional[torch.Tensor]
        Optional caller-owned ``uint8`` buffer of at least
        :func:`mxfp8_activation_scale_workspace_bytes` ``(M)`` bytes that receives ``a_sf`` in the
        128x4 swizzled layout (row tiles padded to an even count).  Allocated zeroed when omitted.

    Returns
    -------
    torch.Tensor
        ``bfloat16`` ``[M, 5376]``.
    """
    _degree, rows, device = _check_attn_out(attn_out)
    _check_tensor(
        o_weight_q,
        "o_weight_q",
        (MINIMAX_H3_HIDDEN, MINIMAX_H3_ATTN_DIM),
        torch.float8_e4m3fn,
        device,
    )
    o_scale_tiles = _scale_tiles(
        o_scale_tiles, "o_scale_tiles", MXFP8_O_SCALE_TILE_BYTES, device
    )
    _check_epilogue_inputs(gate, gate_index, residual, rows, device)
    if workspace_q is None:
        workspace_q = torch.empty(
            (rows, MINIMAX_H3_ATTN_DIM), dtype=torch.float8_e4m3fn, device=device
        )
    else:
        _check_tensor(
            workspace_q,
            "workspace_q",
            (rows, MINIMAX_H3_ATTN_DIM),
            torch.float8_e4m3fn,
            device,
        )
    workspace_sf = _scale_workspace(
        workspace_sf,
        "workspace_sf",
        mxfp8_activation_scale_workspace_bytes(rows),
        device,
    )
    out = _output(out, rows, device)
    _minimax_h3_out_proj_mxfp8_impl(
        attn_out,
        o_weight_q,
        o_scale_tiles,
        gate,
        gate_index,
        residual,
        out,
        workspace_q,
        workspace_sf,
    )
    return out


def minimax_h3_out_proj_nvfp4(
    attn_out: torch.Tensor,
    a_global_scale: torch.Tensor,
    o_weight_q: torch.Tensor,
    o_scale_tiles: torch.Tensor,
    alpha: torch.Tensor,
    gate: torch.Tensor,
    gate_index: torch.Tensor,
    residual: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
    workspace_q: Optional[torch.Tensor] = None,
    workspace_sf: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""NVFP4 (W4A4, E2M1 + UE4M3 per-16 block scales + FP32 global scales) variant of
    :func:`minimax_h3_out_proj`.

    A one-warp-per-row pass reads the receive layout and quantizes the logical activation with
    FlashInfer's :func:`~flashinfer.nvfp4_quantize` recipe (``cvt_warp_fp16_to_fp4``): per 16
    consecutive K elements ``sf = E4M3_RN(g * absmax * rcp(6))`` saturating at 448 with ``g =
    a_global_scale``, codes ``E2M1_RN_saturate(a * rcp(sf * rcp(g)))`` (an all-zero block writes
    ``sf = 0`` and codes 0); the ``rcp`` are the same approximate reciprocals FlashInfer uses, so
    the quantized activation is bit-identical to ``nvfp4_quantize(A, a_global_scale,
    SfLayout.layout_128x4)``.  The GEMM accumulates ``(a_q * a_sf) . (w_q * w_sf)`` in FP32 and
    the epilogue applies ``alpha = 1 / (a_global_scale * w_global_scale)`` before the BF16 round,
    as ``mm_fp4`` does::

        o = BF16(alpha * ((a_q * a_sf) @ (o_weight_q * o_scale_tiles)^T))
        out = BF16(residual + BF16(gate[gate_index[m]] * o))

    Parameters
    ----------
    attn_out, gate, gate_index, residual, out
        As in :func:`minimax_h3_out_proj`.
    a_global_scale : torch.Tensor
        ``float32`` ``[1]`` activation global scale (``448 * 6 / absmax(A)`` convention, typically
        a calibrated static scale; see
        :func:`~flashinfer.diffusion_ops.minimax_h3_fc1_swiglu.minimax_h3_nvfp4_global_scale`).
    o_weight_q : torch.Tensor
        Packed E2M1 ``uint8`` ``[5376, 3584]`` weights from :func:`prepare_minimax_h3_o_weight_nvfp4`.
    o_scale_tiles : torch.Tensor
        Flat ``uint8`` ``[21 * 112 * 1024]`` weight scale tiles from the same call.
    alpha : torch.Tensor
        ``float32`` ``[1]`` = ``1 / (a_global_scale * w_global_scale)``
        (:func:`~flashinfer.diffusion_ops.minimax_h3_fc1_swiglu.minimax_h3_nvfp4_alpha`).
    workspace_q : Optional[torch.Tensor]
        Optional caller-owned ``uint8`` ``[M, 3584]`` buffer that receives the packed E2M1 activation.
    workspace_sf : Optional[torch.Tensor]
        Optional caller-owned ``uint8`` buffer of at least
        :func:`nvfp4_activation_scale_workspace_bytes` ``(M)`` bytes that receives the activation
        scales in the 128x4 swizzled layout.  Allocated zeroed when omitted.

    Returns
    -------
    torch.Tensor
        ``bfloat16`` ``[M, 5376]``.
    """
    _degree, rows, device = _check_attn_out(attn_out)
    a_global_scale = _scalar_f32(a_global_scale, "a_global_scale", device)
    alpha = _scalar_f32(alpha, "alpha", device)
    _check_tensor(
        o_weight_q,
        "o_weight_q",
        (MINIMAX_H3_HIDDEN, NVFP4_PACKED_COLS),
        torch.uint8,
        device,
    )
    o_scale_tiles = _scale_tiles(
        o_scale_tiles, "o_scale_tiles", NVFP4_O_SCALE_TILE_BYTES, device
    )
    _check_epilogue_inputs(gate, gate_index, residual, rows, device)
    if workspace_q is None:
        workspace_q = torch.empty(
            (rows, NVFP4_PACKED_COLS), dtype=torch.uint8, device=device
        )
    else:
        _check_tensor(
            workspace_q, "workspace_q", (rows, NVFP4_PACKED_COLS), torch.uint8, device
        )
    workspace_sf = _scale_workspace(
        workspace_sf,
        "workspace_sf",
        nvfp4_activation_scale_workspace_bytes(rows),
        device,
    )
    out = _output(out, rows, device)
    _minimax_h3_out_proj_nvfp4_impl(
        attn_out,
        a_global_scale,
        o_weight_q,
        o_scale_tiles,
        alpha,
        gate,
        gate_index,
        residual,
        out,
        workspace_q,
        workspace_sf,
    )
    return out


__all__ = [
    "MINIMAX_H3_ATTN_DIM",
    "MINIMAX_H3_GATE_ROWS",
    "MINIMAX_H3_HEAD_DIM",
    "MINIMAX_H3_HIDDEN",
    "MINIMAX_H3_MAX_ROWS",
    "MINIMAX_H3_NUM_HEADS",
    "MINIMAX_H3_SEQUENCE_PARALLEL_DEGREES",
    "MXFP8_O_SCALE_TILE_BYTES",
    "NVFP4_O_SCALE_TILE_BYTES",
    "minimax_h3_nvfp4_alpha",
    "minimax_h3_nvfp4_global_scale",
    "minimax_h3_out_proj",
    "minimax_h3_out_proj_mxfp8",
    "minimax_h3_out_proj_nvfp4",
    "minimax_h3_out_proj_reference",
    "minimax_h3_unpack_attn_out",
    "mxfp8_activation_scale_workspace_bytes",
    "nvfp4_activation_scale_workspace_bytes",
    "prepare_minimax_h3_o_weight_mxfp8",
    "prepare_minimax_h3_o_weight_nvfp4",
]
