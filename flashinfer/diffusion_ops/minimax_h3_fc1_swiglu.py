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

from ..jit.minimax_h3_fc1_swiglu import (
    MiniMaxH3Fc1SwigluTarget,
    gen_minimax_h3_fc1_swiglu_module,
    minimax_h3_fc1_swiglu_target,
)
from ..utils import get_compute_capability, register_custom_op, register_fake_op

# MiniMax-H3 video DiT block dimensions served by the generated kernels.
MINIMAX_H3_HIDDEN = 5376
MINIMAX_H3_FFN = 14336
MINIMAX_H3_FC1_ROWS = 2 * MINIMAX_H3_FFN  # [gate rows; up rows]
MINIMAX_H3_ADALN_ROWS = 9
MINIMAX_H3_EPS = 1.0e-5
MINIMAX_H3_MAX_ROWS = 1 << 24

# GEMM tiling facts the host-side workspace sizing depends on.
_BLOCK_M = 128
_CTA_GROUP = 2
_N_TILE_ROWS = 112  # gate/up weight rows per CTA of the quantized GEMMs
_QUANT_N_TILES = MINIMAX_H3_FFN // _N_TILE_ROWS  # 128 output tiles

# FlashInfer 128x4 swizzled scale-factor tiles: 512 bytes = 128 rows x 4 K-blocks, byte offset
# (row % 32) * 16 + (row // 32) * 4 + kblock inside the tile; tiles ordered (row block, K set).
_SF_TILE_ROWS = 128
_SF_TILE_BYTES = 512
MXFP8_BLOCK = 32
MXFP8_SF_COLS = MINIMAX_H3_HIDDEN // MXFP8_BLOCK  # 168 UE8M0 scales per row
MXFP8_SF_K_TILES = MXFP8_SF_COLS // 4  # 42 tiles per 128-row block
NVFP4_BLOCK = 16
NVFP4_PACKED_COLS = MINIMAX_H3_HIDDEN // 2  # 2688 E2M1 nibble pairs per row
NVFP4_SF_COLS = MINIMAX_H3_HIDDEN // NVFP4_BLOCK  # 336 UE4M3 scales per row
NVFP4_SF_K_TILES = NVFP4_SF_COLS // 4  # 84 tiles per 128-row block
# Combined 224-row (padded to 256) gate/up weight scale tiles: [tile][k_set][half][512 bytes].
MXFP8_FC1_SCALE_TILE_BYTES = _QUANT_N_TILES * MXFP8_SF_K_TILES * 2 * _SF_TILE_BYTES
NVFP4_FC1_SCALE_TILE_BYTES = _QUANT_N_TILES * NVFP4_SF_K_TILES * 2 * _SF_TILE_BYTES

_E4M3_MAX = 448.0
_E2M1_MAX = 6.0


@functools.lru_cache(maxsize=None)
def _get_module(target: MiniMaxH3Fc1SwigluTarget):
    return gen_minimax_h3_fc1_swiglu_module(target).build_and_load()


def _module_for(device: torch.device):
    return _get_module(minimax_h3_fc1_swiglu_target(get_compute_capability(device)))


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


def _check_rows(x: torch.Tensor) -> int:
    if not isinstance(x, torch.Tensor) or x.ndim != 2:
        raise ValueError(f"x must be a rank-2 tensor [M, {MINIMAX_H3_HIDDEN}]")
    rows = int(x.shape[0])
    if not 1 <= rows <= MINIMAX_H3_MAX_ROWS:
        raise ValueError(f"M must satisfy 1 <= M <= {MINIMAX_H3_MAX_ROWS}, got {rows}")
    return rows


def _check_norm_inputs(
    x, x_norm_weight, adaln_scale, adaln_shift, adaln_index, eps
) -> Tuple[int, torch.device]:
    rows = _check_rows(x)
    device = x.device
    _check_tensor(x, "x", (rows, MINIMAX_H3_HIDDEN), torch.bfloat16, device)
    _check_tensor(
        x_norm_weight, "x_norm_weight", (MINIMAX_H3_HIDDEN,), torch.bfloat16, device
    )
    _check_tensor(
        adaln_scale,
        "adaln_scale",
        (MINIMAX_H3_ADALN_ROWS, MINIMAX_H3_HIDDEN),
        torch.bfloat16,
        device,
    )
    _check_tensor(
        adaln_shift,
        "adaln_shift",
        (MINIMAX_H3_ADALN_ROWS, MINIMAX_H3_HIDDEN),
        torch.bfloat16,
        device,
    )
    _check_tensor(adaln_index, "adaln_index", (rows,), torch.int32, device)
    if float(eps) != MINIMAX_H3_EPS:
        raise ValueError(
            f"eps must be {MINIMAX_H3_EPS} (the validated MiniMax-H3 contract), got {eps}"
        )
    return rows, device


def _output(
    out: Optional[torch.Tensor], rows: int, device: torch.device
) -> torch.Tensor:
    if out is None:
        return torch.empty((rows, MINIMAX_H3_FFN), dtype=torch.bfloat16, device=device)
    _check_tensor(out, "out", (rows, MINIMAX_H3_FFN), torch.bfloat16, device)
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


def _scalar_f32(
    value: Union[torch.Tensor, float], name: str, device: torch.device
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"{name} must hold exactly one element")
        return value.to(device=device, dtype=torch.float32).reshape(1).contiguous()
    return torch.tensor([float(value)], dtype=torch.float32, device=device)


# --------------------------------------------------------------------------------------------
# Scale-factor layout helpers (pure torch)
# --------------------------------------------------------------------------------------------


def _swizzle_sf_128x4(sf_linear: torch.Tensor) -> torch.Tensor:
    """``[R, C]`` scale bytes -> flat FlashInfer 128x4 swizzled layout (rows padded to 128, columns
    to a multiple of 4).  Tile ``(r // 128, c // 4)`` holds byte ``(r % 32) * 16 + (r % 128 // 32)
    * 4 + c % 4``; tiles are ordered row-block major, K-set minor."""
    rows, cols = sf_linear.shape
    padded_rows = -(-rows // _SF_TILE_ROWS) * _SF_TILE_ROWS
    padded_cols = -(-cols // 4) * 4
    padded = torch.zeros(
        (padded_rows, padded_cols), dtype=torch.uint8, device=sf_linear.device
    )
    padded[:rows, :cols] = sf_linear
    view = padded.reshape(padded_rows // _SF_TILE_ROWS, 4, 32, padded_cols // 4, 4)
    # (tile, r_hi, r_lo, c_hi, c_lo) -> (tile, c_hi, r_lo, r_hi, c_lo)
    return view.permute(0, 3, 2, 1, 4).reshape(-1).contiguous()


def _unswizzle_sf_128x4(
    sf_swizzled: torch.Tensor, rows: int, cols: int
) -> torch.Tensor:
    """Inverse of :func:`_swizzle_sf_128x4`: flat swizzled bytes for ``rows`` (padded to 128) x
    ``cols`` (padded to 4) -> ``[rows, cols]``.  This is the layout FlashInfer's ``mxfp8_quantize``
    (``is_sf_swizzled_layout=True``) and ``nvfp4_quantize`` (``SfLayout.layout_128x4``) emit."""
    padded_rows = -(-rows // _SF_TILE_ROWS) * _SF_TILE_ROWS
    padded_cols = -(-cols // 4) * 4
    view = sf_swizzled.reshape(padded_rows // _SF_TILE_ROWS, padded_cols // 4, 32, 4, 4)
    return (
        view.permute(0, 3, 2, 1, 4)
        .reshape(padded_rows, padded_cols)[:rows, :cols]
        .contiguous()
    )


def _combined_weight_scale_tiles(sf_linear: torch.Tensor) -> torch.Tensor:
    """``[28672, C]`` linear weight scales -> flat combined tiles ``[tile][k_set][half][512]``.

    Output tile ``t`` (128 tiles of 112 output columns) pairs gate rows ``[112 t, 112 (t + 1))``
    with the matching up rows into one 224-row block (rows 224..255 zero); each 128-K set of that
    block is two 512-byte 128x4 tiles (combined rows 0-127, then 128-255), the order the 2-CTA
    block-scaled MMA reads its B scale factors in."""
    if tuple(sf_linear.shape[:1]) != (MINIMAX_H3_FC1_ROWS,):
        raise ValueError(
            f"weight scales must have {MINIMAX_H3_FC1_ROWS} rows, got {tuple(sf_linear.shape)}"
        )
    cols = int(sf_linear.shape[1])
    gate = sf_linear[:MINIMAX_H3_FFN].reshape(_QUANT_N_TILES, _N_TILE_ROWS, cols)
    up = sf_linear[MINIMAX_H3_FFN:].reshape(_QUANT_N_TILES, _N_TILE_ROWS, cols)
    combined = torch.zeros(
        (_QUANT_N_TILES, 2 * _SF_TILE_ROWS, cols),
        dtype=torch.uint8,
        device=sf_linear.device,
    )
    combined[:, :_N_TILE_ROWS] = gate
    combined[:, _N_TILE_ROWS : 2 * _N_TILE_ROWS] = up
    tiles = _swizzle_sf_128x4(
        combined.reshape(_QUANT_N_TILES * 2 * _SF_TILE_ROWS, cols)
    )
    k_sets = -(-cols // 4)
    # swizzle order is (row tile = n_tile * 2 + half, k_set) -> (n_tile, k_set, half)
    return (
        tiles.reshape(_QUANT_N_TILES, 2, k_sets, _SF_TILE_BYTES)
        .permute(0, 2, 1, 3)
        .contiguous()
        .reshape(-1)
    )


# --------------------------------------------------------------------------------------------
# Weight preparation (offline, torch + FlashInfer quantizers)
# --------------------------------------------------------------------------------------------


def prepare_minimax_h3_fc1_weight_mxfp8(
    fc1_weight: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Quantize the fused FC1 weight for :func:`minimax_h3_fc1_swiglu_mxfp8`.

    ``fc1_weight`` is the BF16 ``[28672, 5376]`` matrix with gate rows ``[0, 14336)`` followed by up
    rows ``[14336, 28672)``.  Quantization is FlashInfer's :func:`~flashinfer.mxfp8_quantize`
    (per 32 consecutive K elements: UE8M0 scale = ``absmax / 448`` rounded up to a power of two,
    E4M3 round-to-nearest-even of ``w / scale``; an all-zero block keeps scale byte 0).

    Returns ``(fc1_weight_q, fc1_scale_tiles)``: the ``float8_e4m3fn`` ``[28672, 5376]`` weights
    and a flat ``uint8`` tensor of ``128 * 42 * 1024`` bytes holding the weight scales in the
    combined 224-row gate/up tile order the fused GEMM streams (see
    :func:`_combined_weight_scale_tiles`).  Relies on ``mxfp8_quantize(..., is_sf_swizzled_layout=True)``
    returning the scales in the 128x4 swizzled layout with rows padded to 128 and columns to a
    multiple of 4 (both exact for this shape), which is unswizzled here before re-tiling.
    """
    from ..quantization.fp8_quantization import mxfp8_quantize

    if (
        tuple(fc1_weight.shape) != (MINIMAX_H3_FC1_ROWS, MINIMAX_H3_HIDDEN)
        or fc1_weight.dtype != torch.bfloat16
    ):
        raise ValueError(
            f"fc1_weight must be bfloat16 [{MINIMAX_H3_FC1_ROWS}, {MINIMAX_H3_HIDDEN}]"
        )
    if not fc1_weight.is_cuda:
        raise ValueError("fc1_weight must be a CUDA tensor")
    w_q, w_sf = mxfp8_quantize(fc1_weight.contiguous(), is_sf_swizzled_layout=True)
    w_sf = w_sf.view(torch.uint8).reshape(-1)
    expected = MINIMAX_H3_FC1_ROWS * MXFP8_SF_COLS
    if w_sf.numel() != expected:
        raise RuntimeError(
            f"mxfp8_quantize returned {w_sf.numel()} scale bytes, expected {expected}"
        )
    sf_linear = _unswizzle_sf_128x4(w_sf, MINIMAX_H3_FC1_ROWS, MXFP8_SF_COLS)
    return w_q.view(torch.float8_e4m3fn).contiguous(), _combined_weight_scale_tiles(
        sf_linear
    )


def minimax_h3_nvfp4_global_scale(tensor: torch.Tensor) -> torch.Tensor:
    r"""FlashInfer / TensorRT-LLM per-tensor NVFP4 global scale ``448 * 6 / absmax`` as a float32
    ``[1]`` tensor on the tensor's device."""
    absmax = tensor.detach().float().abs().nan_to_num().max()
    return ((_E4M3_MAX * _E2M1_MAX) / absmax).reshape(1).to(torch.float32)


def minimax_h3_nvfp4_alpha(
    a_global_scale: torch.Tensor, w_global_scale: torch.Tensor
) -> torch.Tensor:
    r"""``alpha = 1 / (a_global_scale * w_global_scale)`` as a float32 ``[1]`` tensor."""
    g_a = a_global_scale.to(torch.float32).reshape(())
    g_w = w_global_scale.to(device=g_a.device, dtype=torch.float32).reshape(())
    return (1.0 / (g_a * g_w)).reshape(1)


def prepare_minimax_h3_fc1_weight_nvfp4(
    fc1_weight: torch.Tensor, w_global_scale: Union[torch.Tensor, float]
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Quantize the fused FC1 weight for :func:`minimax_h3_fc1_swiglu_nvfp4`.

    ``fc1_weight`` is the BF16 ``[28672, 5376]`` matrix (gate rows then up rows) and
    ``w_global_scale`` its float32 global scale (:func:`minimax_h3_nvfp4_global_scale`).
    Quantization is FlashInfer's :func:`~flashinfer.nvfp4_quantize` (per 16 consecutive K elements:
    UE4M3 scale = ``E4M3_RN(g * absmax / 6)`` saturating at 448, E2M1 round-to-nearest with
    saturation of ``w * g / scale``; an all-zero block writes scale 0 and codes 0).

    Returns ``(fc1_weight_q, fc1_scale_tiles)``: packed E2M1 ``uint8`` ``[28672, 2688]`` weights
    (even element in the low nibble) and a flat ``uint8`` tensor of ``128 * 84 * 1024`` bytes with
    the weight scales in the combined 224-row gate/up tile order.  Relies on
    ``nvfp4_quantize(..., sfLayout=SfLayout.layout_128x4, do_shuffle=False)`` returning the scales
    in the 128x4 swizzled layout with rows padded to 128 and columns to a multiple of 4.
    """
    from ..quantization.fp4_quantization import nvfp4_quantize
    from ..tllm_enums import SfLayout

    if (
        tuple(fc1_weight.shape) != (MINIMAX_H3_FC1_ROWS, MINIMAX_H3_HIDDEN)
        or fc1_weight.dtype != torch.bfloat16
    ):
        raise ValueError(
            f"fc1_weight must be bfloat16 [{MINIMAX_H3_FC1_ROWS}, {MINIMAX_H3_HIDDEN}]"
        )
    if not fc1_weight.is_cuda:
        raise ValueError("fc1_weight must be a CUDA tensor")
    g_w = _scalar_f32(w_global_scale, "w_global_scale", fc1_weight.device)
    w_q, w_sf = nvfp4_quantize(
        fc1_weight.contiguous(), g_w, sfLayout=SfLayout.layout_128x4, do_shuffle=False
    )
    w_q = (
        w_q.view(torch.uint8)
        .reshape(MINIMAX_H3_FC1_ROWS, NVFP4_PACKED_COLS)
        .contiguous()
    )
    w_sf = w_sf.view(torch.uint8).reshape(-1)
    expected = MINIMAX_H3_FC1_ROWS * NVFP4_SF_COLS
    if w_sf.numel() != expected:
        raise RuntimeError(
            f"nvfp4_quantize returned {w_sf.numel()} scale bytes, expected {expected}"
        )
    sf_linear = _unswizzle_sf_128x4(w_sf, MINIMAX_H3_FC1_ROWS, NVFP4_SF_COLS)
    return w_q, _combined_weight_scale_tiles(sf_linear)


# --------------------------------------------------------------------------------------------
# Operators
# --------------------------------------------------------------------------------------------


@register_custom_op(
    "flashinfer::minimax_h3_fc1_swiglu", mutates_args=("workspace", "out")
)
def _minimax_h3_fc1_swiglu_impl(
    x: torch.Tensor,
    x_norm_weight: torch.Tensor,
    adaln_scale: torch.Tensor,
    adaln_shift: torch.Tensor,
    adaln_index: torch.Tensor,
    fc1_weight: torch.Tensor,
    workspace: torch.Tensor,
    out: torch.Tensor,
    eps: float,
) -> None:
    _module_for(x.device).minimax_h3_fc1_swiglu(
        x,
        x_norm_weight,
        adaln_scale,
        adaln_shift,
        adaln_index,
        fc1_weight,
        workspace,
        out,
        eps,
    )


@register_fake_op("flashinfer::minimax_h3_fc1_swiglu")
def _minimax_h3_fc1_swiglu_fake(
    x: torch.Tensor,
    x_norm_weight: torch.Tensor,
    adaln_scale: torch.Tensor,
    adaln_shift: torch.Tensor,
    adaln_index: torch.Tensor,
    fc1_weight: torch.Tensor,
    workspace: torch.Tensor,
    out: torch.Tensor,
    eps: float,
) -> None:
    pass


@register_custom_op(
    "flashinfer::minimax_h3_fc1_swiglu_mxfp8",
    mutates_args=("workspace_q", "workspace_sf", "out"),
)
def _minimax_h3_fc1_swiglu_mxfp8_impl(
    x: torch.Tensor,
    x_norm_weight: torch.Tensor,
    adaln_scale: torch.Tensor,
    adaln_shift: torch.Tensor,
    adaln_index: torch.Tensor,
    fc1_weight_q: torch.Tensor,
    fc1_scale_tiles: torch.Tensor,
    workspace_q: torch.Tensor,
    workspace_sf: torch.Tensor,
    out: torch.Tensor,
    eps: float,
) -> None:
    _module_for(x.device).minimax_h3_fc1_swiglu_mxfp8(
        x,
        x_norm_weight,
        adaln_scale,
        adaln_shift,
        adaln_index,
        fc1_weight_q,
        fc1_scale_tiles,
        workspace_q,
        workspace_sf,
        out,
        eps,
    )


@register_fake_op("flashinfer::minimax_h3_fc1_swiglu_mxfp8")
def _minimax_h3_fc1_swiglu_mxfp8_fake(
    x: torch.Tensor,
    x_norm_weight: torch.Tensor,
    adaln_scale: torch.Tensor,
    adaln_shift: torch.Tensor,
    adaln_index: torch.Tensor,
    fc1_weight_q: torch.Tensor,
    fc1_scale_tiles: torch.Tensor,
    workspace_q: torch.Tensor,
    workspace_sf: torch.Tensor,
    out: torch.Tensor,
    eps: float,
) -> None:
    pass


@register_custom_op(
    "flashinfer::minimax_h3_fc1_swiglu_nvfp4",
    mutates_args=("workspace_q", "workspace_sf", "out"),
)
def _minimax_h3_fc1_swiglu_nvfp4_impl(
    x: torch.Tensor,
    x_norm_weight: torch.Tensor,
    adaln_scale: torch.Tensor,
    adaln_shift: torch.Tensor,
    adaln_index: torch.Tensor,
    a_global_scale: torch.Tensor,
    fc1_weight_q: torch.Tensor,
    fc1_scale_tiles: torch.Tensor,
    alpha: torch.Tensor,
    workspace_q: torch.Tensor,
    workspace_sf: torch.Tensor,
    out: torch.Tensor,
    eps: float,
) -> None:
    _module_for(x.device).minimax_h3_fc1_swiglu_nvfp4(
        x,
        x_norm_weight,
        adaln_scale,
        adaln_shift,
        adaln_index,
        a_global_scale,
        fc1_weight_q,
        fc1_scale_tiles,
        alpha,
        workspace_q,
        workspace_sf,
        out,
        eps,
    )


@register_fake_op("flashinfer::minimax_h3_fc1_swiglu_nvfp4")
def _minimax_h3_fc1_swiglu_nvfp4_fake(
    x: torch.Tensor,
    x_norm_weight: torch.Tensor,
    adaln_scale: torch.Tensor,
    adaln_shift: torch.Tensor,
    adaln_index: torch.Tensor,
    a_global_scale: torch.Tensor,
    fc1_weight_q: torch.Tensor,
    fc1_scale_tiles: torch.Tensor,
    alpha: torch.Tensor,
    workspace_q: torch.Tensor,
    workspace_sf: torch.Tensor,
    out: torch.Tensor,
    eps: float,
) -> None:
    pass


def minimax_h3_fc1_swiglu(
    x: torch.Tensor,
    x_norm_weight: torch.Tensor,
    adaln_scale: torch.Tensor,
    adaln_shift: torch.Tensor,
    adaln_index: torch.Tensor,
    fc1_weight: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
    workspace: Optional[torch.Tensor] = None,
    eps: float = MINIMAX_H3_EPS,
) -> torch.Tensor:
    r"""Fused BF16 RMSNorm + indexed AdaLN + FC1 GEMM + SwiGLU of the MiniMax-H3 video DiT block
    (SM100 / SM103).

    Computes, with every intermediate rounded to BF16 exactly where the PyTorch module graph does::

        n = BF16(RMSNorm_fp32(x, x_norm_weight, eps))                     # FP32 sum of squares, rsqrt
        a = BF16(n * BF16(1 + adaln_scale[idx]) + adaln_shift[idx])       # idx = adaln_index[row]
        a[idx outside [0, 9)] = 0                                          # device-side guard
        h = BF16(a @ fc1_weight^T)                                         # FP32 accumulation
        y = BF16(BF16(silu(h[:, :14336])) * h[:, 14336:])

    ``fc1_weight`` is ``[28672, 5376]`` with the **gate rows first** (``[0, 14336)``) and the up
    rows second (``[14336, 28672)``).  Two kernels run on the current stream: a one-warp-per-row
    norm kernel that writes ``a`` into ``workspace`` and a persistent 2-CTA tcgen05 GEMM with the
    fused SwiGLU epilogue.

    Parameters
    ----------
    x : torch.Tensor
        Contiguous ``bfloat16`` ``[M, 5376]`` activations, ``1 <= M <= 2**24``.
    x_norm_weight : torch.Tensor
        ``bfloat16`` ``[5376]`` RMSNorm weight.
    adaln_scale, adaln_shift : torch.Tensor
        ``bfloat16`` ``[9, 5376]`` AdaLN tables.
    adaln_index : torch.Tensor
        ``int32`` ``[M]`` table row per activation row.
    fc1_weight : torch.Tensor
        Contiguous ``bfloat16`` ``[28672, 5376]`` fused FC1 weight (gate rows then up rows).
    out : Optional[torch.Tensor]
        Optional ``bfloat16`` ``[M, 14336]`` output (allocated when omitted).
    workspace : Optional[torch.Tensor]
        Optional caller-owned ``bfloat16`` ``[M, 5376]`` scratch that receives ``a`` (allocated
        when omitted).
    eps : float
        RMSNorm epsilon; must equal ``1e-5`` (the validated contract).

    Returns
    -------
    torch.Tensor
        ``bfloat16`` ``[M, 14336]``.
    """
    rows, device = _check_norm_inputs(
        x, x_norm_weight, adaln_scale, adaln_shift, adaln_index, eps
    )
    _check_tensor(
        fc1_weight,
        "fc1_weight",
        (MINIMAX_H3_FC1_ROWS, MINIMAX_H3_HIDDEN),
        torch.bfloat16,
        device,
    )
    if workspace is None:
        workspace = torch.empty(
            (rows, MINIMAX_H3_HIDDEN), dtype=torch.bfloat16, device=device
        )
    else:
        _check_tensor(
            workspace, "workspace", (rows, MINIMAX_H3_HIDDEN), torch.bfloat16, device
        )
    out = _output(out, rows, device)
    _minimax_h3_fc1_swiglu_impl(
        x,
        x_norm_weight,
        adaln_scale,
        adaln_shift,
        adaln_index,
        fc1_weight,
        workspace,
        out,
        float(eps),
    )
    return out


def minimax_h3_fc1_swiglu_mxfp8(
    x: torch.Tensor,
    x_norm_weight: torch.Tensor,
    adaln_scale: torch.Tensor,
    adaln_shift: torch.Tensor,
    adaln_index: torch.Tensor,
    fc1_weight_q: torch.Tensor,
    fc1_scale_tiles: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
    workspace_q: Optional[torch.Tensor] = None,
    workspace_sf: Optional[torch.Tensor] = None,
    eps: float = MINIMAX_H3_EPS,
) -> torch.Tensor:
    r"""MXFP8 (W8A8, E4M3 + UE8M0 per-32 block scales) variant of :func:`minimax_h3_fc1_swiglu`.

    The norm kernel computes the same BF16 modulated activation ``a`` and quantizes it exactly as
    FlashInfer's :func:`~flashinfer.mxfp8_quantize` does (per 32 consecutive K elements: UE8M0
    scale = ``absmax / 448`` rounded **up** to a power of two, E4M3 round-to-nearest-even; an
    all-zero block keeps scale byte 0).  The GEMM accumulates the block-scaled products in FP32::

        h = BF16(dequant(a_q, a_sf) @ dequant(fc1_weight_q, fc1_scale_tiles)^T)
        y = BF16(BF16(silu(h[:, :14336])) * h[:, 14336:])

    Parameters
    ----------
    x, x_norm_weight, adaln_scale, adaln_shift, adaln_index, eps
        As in :func:`minimax_h3_fc1_swiglu`.
    fc1_weight_q : torch.Tensor
        ``float8_e4m3fn`` ``[28672, 5376]`` weights from :func:`prepare_minimax_h3_fc1_weight_mxfp8`
        (gate rows then up rows).
    fc1_scale_tiles : torch.Tensor
        Flat ``uint8`` ``[128 * 42 * 1024]`` weight scale tiles from the same call.
    out : Optional[torch.Tensor]
        Optional ``bfloat16`` ``[M, 14336]`` output.
    workspace_q : Optional[torch.Tensor]
        Optional caller-owned ``float8_e4m3fn`` ``[M, 5376]`` buffer that receives ``a_q``.
    workspace_sf : Optional[torch.Tensor]
        Optional caller-owned ``uint8`` buffer of at least
        :func:`mxfp8_activation_scale_workspace_bytes` ``(M)`` bytes that receives ``a_sf`` in the
        128x4 swizzled layout (row tiles padded to an even count).  Allocated zeroed when omitted.

    Returns
    -------
    torch.Tensor
        ``bfloat16`` ``[M, 14336]``.
    """
    rows, device = _check_norm_inputs(
        x, x_norm_weight, adaln_scale, adaln_shift, adaln_index, eps
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
    if workspace_q is None:
        workspace_q = torch.empty(
            (rows, MINIMAX_H3_HIDDEN), dtype=torch.float8_e4m3fn, device=device
        )
    else:
        _check_tensor(
            workspace_q,
            "workspace_q",
            (rows, MINIMAX_H3_HIDDEN),
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
    _minimax_h3_fc1_swiglu_mxfp8_impl(
        x,
        x_norm_weight,
        adaln_scale,
        adaln_shift,
        adaln_index,
        fc1_weight_q,
        fc1_scale_tiles,
        workspace_q,
        workspace_sf,
        out,
        float(eps),
    )
    return out


def minimax_h3_fc1_swiglu_nvfp4(
    x: torch.Tensor,
    x_norm_weight: torch.Tensor,
    adaln_scale: torch.Tensor,
    adaln_shift: torch.Tensor,
    adaln_index: torch.Tensor,
    a_global_scale: torch.Tensor,
    fc1_weight_q: torch.Tensor,
    fc1_scale_tiles: torch.Tensor,
    alpha: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
    workspace_q: Optional[torch.Tensor] = None,
    workspace_sf: Optional[torch.Tensor] = None,
    eps: float = MINIMAX_H3_EPS,
) -> torch.Tensor:
    r"""NVFP4 (W4A4, E2M1 + UE4M3 per-16 block scales + FP32 global scales) variant of
    :func:`minimax_h3_fc1_swiglu`.

    The norm kernel computes the BF16 modulated activation ``a`` and quantizes it with FlashInfer's
    :func:`~flashinfer.nvfp4_quantize` recipe (``cvt_warp_fp16_to_fp4``): per 16 consecutive K
    elements ``sf = E4M3_RN(g * absmax * rcp(6))`` saturating at 448 with ``g = a_global_scale``,
    codes ``E2M1_RN_saturate(a * rcp(sf * rcp(g)))`` (an all-zero block writes ``sf = 0`` and
    codes 0); the ``rcp`` are the same approximate reciprocals FlashInfer uses, so the quantized
    activation is bit-identical to ``nvfp4_quantize(a, a_global_scale, SfLayout.layout_128x4)``.
    The GEMM accumulates ``(a_q * a_sf) . (w_q * w_sf)`` in FP32 and the epilogue applies
    ``alpha = 1 / (a_global_scale * w_global_scale)`` before the BF16 round, as ``mm_fp4`` does::

        h = BF16(alpha * ((a_q * a_sf) @ (fc1_weight_q * fc1_scale_tiles)^T))
        y = BF16(BF16(silu(h[:, :14336])) * h[:, 14336:])

    Parameters
    ----------
    x, x_norm_weight, adaln_scale, adaln_shift, adaln_index, eps
        As in :func:`minimax_h3_fc1_swiglu`.
    a_global_scale : torch.Tensor
        ``float32`` ``[1]`` activation global scale (``448 * 6 / absmax(a)`` convention, typically a
        calibrated static scale; see :func:`minimax_h3_nvfp4_global_scale`).
    fc1_weight_q : torch.Tensor
        Packed E2M1 ``uint8`` ``[28672, 2688]`` weights from :func:`prepare_minimax_h3_fc1_weight_nvfp4`
        (gate rows then up rows).
    fc1_scale_tiles : torch.Tensor
        Flat ``uint8`` ``[128 * 84 * 1024]`` weight scale tiles from the same call.
    alpha : torch.Tensor
        ``float32`` ``[1]`` = ``1 / (a_global_scale * w_global_scale)`` (:func:`minimax_h3_nvfp4_alpha`).
    out : Optional[torch.Tensor]
        Optional ``bfloat16`` ``[M, 14336]`` output.
    workspace_q : Optional[torch.Tensor]
        Optional caller-owned ``uint8`` ``[M, 2688]`` buffer that receives the packed E2M1 activation.
    workspace_sf : Optional[torch.Tensor]
        Optional caller-owned ``uint8`` buffer of at least
        :func:`nvfp4_activation_scale_workspace_bytes` ``(M)`` bytes that receives the activation
        scales in the 128x4 swizzled layout.  Allocated zeroed when omitted.

    Returns
    -------
    torch.Tensor
        ``bfloat16`` ``[M, 14336]``.
    """
    rows, device = _check_norm_inputs(
        x, x_norm_weight, adaln_scale, adaln_shift, adaln_index, eps
    )
    a_global_scale = _scalar_f32(a_global_scale, "a_global_scale", device)
    alpha = _scalar_f32(alpha, "alpha", device)
    _check_tensor(
        fc1_weight_q,
        "fc1_weight_q",
        (MINIMAX_H3_FC1_ROWS, NVFP4_PACKED_COLS),
        torch.uint8,
        device,
    )
    fc1_scale_tiles = _scale_tiles(
        fc1_scale_tiles, "fc1_scale_tiles", NVFP4_FC1_SCALE_TILE_BYTES, device
    )
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
    _minimax_h3_fc1_swiglu_nvfp4_impl(
        x,
        x_norm_weight,
        adaln_scale,
        adaln_shift,
        adaln_index,
        a_global_scale,
        fc1_weight_q,
        fc1_scale_tiles,
        alpha,
        workspace_q,
        workspace_sf,
        out,
        float(eps),
    )
    return out


__all__ = [
    "MINIMAX_H3_ADALN_ROWS",
    "MINIMAX_H3_EPS",
    "MINIMAX_H3_FC1_ROWS",
    "MINIMAX_H3_FFN",
    "MINIMAX_H3_HIDDEN",
    "MINIMAX_H3_MAX_ROWS",
    "MXFP8_FC1_SCALE_TILE_BYTES",
    "NVFP4_FC1_SCALE_TILE_BYTES",
    "minimax_h3_fc1_swiglu",
    "minimax_h3_fc1_swiglu_mxfp8",
    "minimax_h3_fc1_swiglu_nvfp4",
    "minimax_h3_nvfp4_alpha",
    "minimax_h3_nvfp4_global_scale",
    "mxfp8_activation_scale_workspace_bytes",
    "nvfp4_activation_scale_workspace_bytes",
    "prepare_minimax_h3_fc1_weight_mxfp8",
    "prepare_minimax_h3_fc1_weight_nvfp4",
]
