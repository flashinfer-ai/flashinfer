# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""MoK's MXFP8 recipe in FP32 PyTorch arithmetic (reference for tests and examples).

E4M3 data with one E8M0 scale per 32-element block along the last axis:
``scale = max(amax / 448, 1e-12)`` rounded up to a power of two
(``cvt.rp.satfinite.ue8m0x2.f32``), values scaled by ``2 ** (127 - e)`` and rounded to
E4M3 with saturation (``cvt.rn.satfinite.e4m3x2.f32``). Scale bytes use MoK's
``[rows / 128, cols / 128, 32, 16]`` tiles, byte ``(row % 32) * 16 + (row // 32) * 4 + k_block``
within each 128 x 128 block. :func:`mxfp8_quantize_reference` matches the CUDA
``mxfp8_quantize`` bitwise; :func:`dequantize_mxfp8` inverts a pair in FP32.
"""

import torch

TILE = 128
K_BLOCK = 32
INV_E4M3_MAX = 0.002232142857  # MoK: amax * 0.002232142857f (1 / 448)
SCALE_FLOOR = 0.000000000001  # MoK: max(..., 0.000000000001f)


def _e8m0_round_up(scale):
    """``cvt.rp.satfinite.ue8m0x2.f32`` on positive normal FP32 values."""
    bits = scale.contiguous().view(torch.int32)
    exponent = (bits >> 23) & 0xFF
    mantissa = bits & 0x007FFFFF
    return torch.clamp(exponent + (mantissa != 0).to(torch.int32), max=254)


def _scale_layout(codes, blocks_m, blocks_n):
    """Per-(row, K block) codes ``[E, M, N // 32]`` -> MoK's ``[E * M // 128, N // 128, 32, 16]``."""
    experts = codes.shape[0]
    tiles = codes.reshape(
        experts, blocks_m, 4, 32, blocks_n, 4
    )  # (e, rb, g, s, cb, kb)
    tiles = tiles.permute(0, 1, 4, 3, 2, 5)  # (e, rb, cb, s, g, kb)
    return (
        tiles.reshape(experts * blocks_m, blocks_n, 32, 16).to(torch.uint8).contiguous()
    )


def _unscale_layout(sc, experts, blocks_m, blocks_n):
    """Inverse of :func:`_scale_layout`: ``[E * bm, bn, 32, 16]`` -> ``[E, bm * 128, bn * 4]``."""
    return (
        sc.reshape(experts, blocks_m, blocks_n, 32, 4, 4)
        .permute(0, 1, 4, 3, 2, 5)
        .reshape(experts, blocks_m * 128, blocks_n * 4)
    )


def _quantize_reference_3d(x):
    experts, rows, cols = x.shape
    blocks = x.float().reshape(experts, rows, cols // K_BLOCK, K_BLOCK)
    amax = blocks.abs().amax(-1)
    scale = torch.clamp_min(amax * INV_E4M3_MAX, SCALE_FLOOR)
    codes = _e8m0_round_up(scale)
    inverse = ((254 - codes) << 23).view(torch.float32)  # 2^(127 - e), MoK's reciprocal
    values = (
        (blocks * inverse[..., None])
        .to(torch.float8_e4m3fn)
        .reshape(experts, rows, cols)
    )
    return values, _scale_layout(codes, rows // TILE, cols // TILE)


def mxfp8_quantize_reference(x_bf16, return_normal=True, return_transposed=True):
    """MoK's ``mxfp8_quantize`` recipe in FP32 PyTorch arithmetic (finite inputs)."""
    x3 = x_bf16 if x_bf16.ndim == 3 else x_bf16.unsqueeze(0)
    x_fp8 = x_sc = x_fp8_t = x_sc_t = None
    if return_normal:
        x_fp8, x_sc = _quantize_reference_3d(x3)
        x_fp8 = x_fp8 if x_bf16.ndim == 3 else x_fp8.squeeze(0)
    if return_transposed:
        x_fp8_t, x_sc_t = _quantize_reference_3d(x3.transpose(1, 2).contiguous())
        x_fp8_t = x_fp8_t if x_bf16.ndim == 3 else x_fp8_t.squeeze(0)
    return x_fp8, x_sc, x_fp8_t, x_sc_t


def dequantize_mxfp8(x_fp8, x_sc):
    """FP32 values of an MoK (E4M3, E8M0 32-block) pair; 2-D inputs stay 2-D."""
    x3 = x_fp8 if x_fp8.ndim == 3 else x_fp8.unsqueeze(0)
    experts, rows, cols = x3.shape
    codes = _unscale_layout(x_sc, experts, rows // 128, cols // 128).to(torch.int32)
    scale = torch.ldexp(torch.ones_like(codes, dtype=torch.float32), codes - 127)
    values = x3.float() * scale.repeat_interleave(32, dim=2)
    return values.squeeze(0) if x_fp8.ndim == 2 else values
