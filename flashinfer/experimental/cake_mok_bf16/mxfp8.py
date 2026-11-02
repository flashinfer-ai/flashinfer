# Copyright 2026 Cursor Research
# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""PyTorch MoK MXFP8 quantization for caller-prequantized routed weights.

Recipe (MoK ``mxfp8_quantize``): 32 consecutive elements along the quantized
(K, last) dimension share one UE8M0 scale byte,
``round_up(max(amax * 0.002232142857f, 1e-12f))`` saturated to 254; data is
``cvt.rn.satfinite.e4m3(x * 2**(127 - byte))``. Scale tiles cover 128 rows x
128 K, stored as ``[rows // 128, K // 128, 32, 16]`` bytes with byte
``(row % 32) * 16 + (row // 32) * 4 + k_block`` (tensor-core scale layout);
experts are concatenated along the first dimension.
"""

import functools

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # the PyTorch path below quantizes the same bytes
    triton = None

_BLOCK = 32
# The FP32 ``1 / 448`` factor of the recipe.
_SCALE_FACTOR = 0.002232142857
# Float32 temporaries of one quantization pass stay below this many bytes:
# experts are quantized together in chunks of this size.
_CHUNK_BYTES = 1 << 30


@functools.cache
def _scale_factor(device):
    """The FP32 ``1 / 448`` factor of the recipe, created once per device."""
    return torch.tensor(_SCALE_FACTOR, dtype=torch.float32, device=device)


def _e8m0_round_up(scale):
    """FP32 -> UE8M0 byte, rounding toward +inf and saturating to 254."""
    bits = scale.contiguous().view(torch.int32)
    exponent = (bits >> 23) & 0xFF
    mantissa = bits & 0x7FFFFF
    byte = exponent + (mantissa != 0).to(torch.int32)
    return byte.clamp(max=254).to(torch.uint8)


def _quantize_rows(x):
    """Quantize ``[E, rows, cols]`` along ``cols``: ``(E4M3 data, [E, rows, cols / 32] bytes)``."""
    experts, rows, cols = x.shape
    blocks = x.float().reshape(experts, rows, cols // _BLOCK, _BLOCK)
    amax = blocks.abs().amax(-1)
    scale = torch.clamp(amax * _scale_factor(x.device), min=1e-12)
    byte = _e8m0_round_up(scale)
    inverse = ((254 - byte.to(torch.int32)) << 23).view(torch.float32)
    data = (blocks * inverse[..., None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return data.reshape(experts, rows, cols), byte


def _scale_tiles(byte):
    """[E, R, K/32] bytes -> [E * R/128, K/128, 32, 16] tensor-core scale tiles."""
    experts, rows, kb = byte.shape
    tiles = byte.reshape(
        experts, rows // 128, 4, 32, kb // 4, 4
    )  # [e, rb, row//32, row%32, kt, kb%4]
    return tiles.permute(0, 1, 4, 3, 2, 5).reshape(
        experts * rows // 128, kb // 4, 32, 16
    )


def _quantize_layout_torch(matrices, transposed):
    """One layout of ``[E, M, N]`` matrices with PyTorch operations: experts are
    quantized together in chunks that bound the FP32 temporaries to
    ``_CHUNK_BYTES``; the result does not depend on the chunking."""
    experts, rows, cols = matrices.shape
    chunk = max(1, _CHUNK_BYTES // (rows * cols * 4))
    shape = (experts, cols, rows) if transposed else (experts, rows, cols)
    data = torch.empty(shape, dtype=torch.float8_e4m3fn, device=matrices.device)
    scales = []
    for start in range(0, experts, chunk):
        part = matrices[start : start + chunk]
        source = part.transpose(1, 2).contiguous() if transposed else part
        data[start : start + chunk], byte = _quantize_rows(source)
        scales.append(_scale_tiles(byte))
    return data, torch.cat(scales)


if triton is not None:

    @triton.jit
    def _quantize_tile_kernel(
        x_ptr,
        data_ptr,
        scale_ptr,
        stride_e,
        stride_r,
        stride_c,
        row_tiles,
        col_tiles,
        scale_factor,
    ):
        """One 128 x 128 tile of the quantized layout (source element ``(e, r, c)``
        at ``e * stride_e + r * stride_r + c * stride_c``): E4M3 data and the
        tile's 512 scale bytes. NaN propagates through the block maximum and the
        clamps as in :func:`_quantize_rows`."""
        tile = tl.program_id(0)
        expert = (tile // row_tiles).to(tl.int64)
        row_tile = tile % row_tiles
        col_tile = tl.program_id(1)
        rows = row_tile * 128 + tl.arange(0, 128)
        cols = col_tile * 128 + tl.arange(0, 128)
        source = (
            expert * stride_e
            + rows[:, None].to(tl.int64) * stride_r
            + cols[None, :].to(tl.int64) * stride_c
        )
        values = tl.reshape(tl.load(x_ptr + source).to(tl.float32), (128, 4, 32))
        amax = tl.max(tl.abs(values), axis=2)
        has_nan = tl.max((values != values).to(tl.int32), axis=2) > 0
        amax = tl.where(has_nan, float("nan"), amax)
        scale = amax * scale_factor
        scale = tl.where(scale != scale, scale, tl.maximum(scale, 1e-12))
        bits = scale.to(tl.int32, bitcast=True)
        byte = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
        byte = tl.minimum(byte, 254)
        inverse = ((254 - byte) << 23).to(tl.float32, bitcast=True)
        scaled = values * inverse[:, :, None]
        scaled = tl.where(
            scaled != scaled, scaled, tl.minimum(tl.maximum(scaled, -448.0), 448.0)
        )
        out_cols = col_tiles * 128
        target = (
            expert * row_tiles * 128 * out_cols
            + rows[:, None].to(tl.int64) * out_cols
            + cols[None, :]
        )
        tl.store(
            data_ptr + target,
            tl.reshape(scaled.to(tl.float8e4nv), (128, 128)),
        )
        local = tl.arange(0, 128)
        tile_base = ((expert * row_tiles + row_tile) * col_tiles + col_tile) * 512
        scale_offsets = (
            tile_base
            + ((local % 32) * 16 + (local // 32) * 4)[:, None]
            + tl.arange(0, 4)[None, :]
        )
        tl.store(scale_ptr + scale_offsets, byte.to(tl.uint8))


def _quantize_layout_triton(matrices, transposed):
    """One layout of ``[E, M, N]`` matrices in one fused kernel launch."""
    experts, rows, cols = matrices.shape
    stride_e, stride_m, stride_n = matrices.stride()
    if transposed:
        rows, cols = cols, rows
        stride_m, stride_n = stride_n, stride_m
    data = torch.empty(
        (experts, rows, cols), dtype=torch.float8_e4m3fn, device=matrices.device
    )
    scales = torch.empty(
        (experts * rows // 128, cols // 128, 32, 16),
        dtype=torch.uint8,
        device=matrices.device,
    )
    grid = (experts * (rows // 128), cols // 128)
    _quantize_tile_kernel[grid](
        matrices,
        data,
        scales,
        stride_e,
        stride_m,
        stride_n,
        rows // 128,
        cols // 128,
        _SCALE_FACTOR,
        num_warps=8,
    )
    return data, scales


def mxfp8_quantize(x, return_normal=True, return_transposed=True):
    """MoK ``mxfp8_quantize`` for BF16 ``[M, N]`` or ``[E, M, N]`` matrices.

    Returns ``(fp8, scales, fp8_t, scales_t)``; entries not requested are
    ``None``. ``fp8`` is E4M3 with the input shape and ``scales`` is uint8
    ``[E * M // 128, N // 128, 32, 16]``; the transposed pair quantizes
    ``x.transpose(-1, -2)`` the same way. M and N must be multiples of 128.
    CUDA tensors are quantized by one fused Triton kernel per layout; without
    Triton (or off CUDA) PyTorch operations produce the same bytes.
    """
    if not isinstance(x, torch.Tensor) or x.dtype != torch.bfloat16:
        raise ValueError("x must be a BF16 tensor")
    if x.ndim not in (2, 3) or any(size <= 0 for size in x.shape):
        raise ValueError("x must have positive shape (M, N) or (E, M, N)")
    if x.shape[-2] % 128 or x.shape[-1] % 128:
        raise ValueError("x M and N dimensions must be divisible by 128")
    if type(return_normal) is not bool or type(return_transposed) is not bool:
        raise TypeError("return_normal and return_transposed must be booleans")
    if not return_normal and not return_transposed:
        raise ValueError("At least one quantized layout must be requested")
    matrices = x.unsqueeze(0) if x.ndim == 2 else x
    quantize = (
        _quantize_layout_triton
        if triton is not None and x.is_cuda
        else _quantize_layout_torch
    )
    outputs = []
    for transposed in (False, True):
        if not (return_transposed if transposed else return_normal):
            outputs += [None, None]
            continue
        data, scales = quantize(matrices, transposed)
        outputs += [data[0] if x.ndim == 2 else data, scales]
    return tuple(outputs)


def quantize_routed_weights(gate, up, down):
    """Prequantize BF16 routed expert weights in the MoK MXFP8 recipe.

    ``gate``/``up`` are ``[E, I, H]`` and ``down`` is ``[E, H, I]``. Returns
    ``(forward_weights, backward_weights)``: forward and context recompute use
    ``(data, scales)`` pairs for gate, up and down; backward uses
    ``(data, scales, data_t, scales_t)`` for gate and up and the transposed
    ``(data_t, scales_t)`` pair for down.
    """
    if (
        any(not isinstance(w, torch.Tensor) or w.ndim != 3 for w in (gate, up, down))
        or gate.shape != up.shape
        or down.shape != (gate.shape[0], gate.shape[2], gate.shape[1])
    ):
        raise ValueError("Expected gate/up [E, I, H] and down [E, H, I] weights")
    q_gate = mxfp8_quantize(gate)
    q_up = mxfp8_quantize(up)
    q_down = mxfp8_quantize(down)
    forward = ((q_gate[0], q_gate[1]), (q_up[0], q_up[1]), (q_down[0], q_down[1]))
    backward = (q_gate, q_up, (q_down[2], q_down[3]))
    return forward, backward
