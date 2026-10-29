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

import torch

_BLOCK = 32


def _e8m0_round_up(scale):
    """FP32 -> UE8M0 byte, rounding toward +inf and saturating to 254."""
    bits = scale.contiguous().view(torch.int32)
    exponent = (bits >> 23) & 0xFF
    mantissa = bits & 0x7FFFFF
    byte = exponent + (mantissa != 0).to(torch.int32)
    return byte.clamp(max=254).to(torch.uint8)


def _quantize_rows(x):
    rows, cols = x.shape
    blocks = x.float().reshape(rows, cols // _BLOCK, _BLOCK)
    amax = blocks.abs().amax(-1)
    scale = torch.clamp(
        amax * torch.tensor(0.002232142857, dtype=torch.float32, device=x.device),
        min=1e-12,
    )
    byte = _e8m0_round_up(scale)
    inverse = ((254 - byte.to(torch.int32)) << 23).view(torch.float32)
    data = (blocks * inverse[..., None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return data.reshape(rows, cols), byte


def _scale_tiles(byte):
    """[R, K/32] bytes -> [R/128, K/128, 32, 16] tensor-core scale tiles."""
    rows, kb = byte.shape
    tiles = byte.reshape(
        rows // 128, 4, 32, kb // 4, 4
    )  # [rb, row//32, row%32, kt, kb%4]
    return tiles.permute(0, 3, 2, 1, 4).reshape(rows // 128, kb // 4, 32, 16)


def mxfp8_quantize(x, return_normal=True, return_transposed=True):
    """MoK ``mxfp8_quantize`` for BF16 ``[M, N]`` or ``[E, M, N]`` matrices.

    Returns ``(fp8, scales, fp8_t, scales_t)``; entries not requested are
    ``None``. ``fp8`` is E4M3 with the input shape and ``scales`` is uint8
    ``[E * M // 128, N // 128, 32, 16]``; the transposed pair quantizes
    ``x.transpose(-1, -2)`` the same way. M and N must be multiples of 128.
    Experts are processed one at a time to bound temporary memory.
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
    outputs = []
    for transposed in (False, True):
        if not (return_transposed if transposed else return_normal):
            outputs += [None, None]
            continue
        data, scales = [], []
        for matrix in matrices:
            source = matrix.transpose(0, 1).contiguous() if transposed else matrix
            values, byte = _quantize_rows(source)
            data.append(values)
            scales.append(_scale_tiles(byte))
        data = torch.stack(data)
        outputs += [data[0] if x.ndim == 2 else data, torch.cat(scales)]
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
