# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Prepare signed INT4/K128-BF16 weights for Hopper MegaMoE.

Preparation recodes the packed payload and creates one eight-byte FP8 lookup
table per scale. It does not materialize an expanded FP8 weight tensor or
require the native fused-MoE module. The transforms work on CPU and CUDA.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


INT4_GROUP_SIZE = 128
INT4_FOLD_M = 64
# Bit 2 is the sign; bits 0, 1 and 3 encode abs(q)-1. Zero uses code 11.
INT4_RECODE = (11, 0, 1, 2, 3, 8, 9, 10, 15, 14, 13, 12, 7, 6, 5, 4)
TransformedWeightLeg = tuple[
    "torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"
]
TransformedMegaWeights = tuple[TransformedWeightLeg, TransformedWeightLeg]


@dataclass(frozen=True)
class Int4WeightPack:
    """Prequantized two's-complement INT4 weights, low nibble first.

    ``w13`` has shape ``[E,2*I,H/2]`` with gate rows followed by up rows;
    ``w2`` has shape ``[E,H,I/2]``. Both are packed int8/uint8 tensors.
    Positive finite BF16 scales have shapes ``[E,2*I,H/128]`` and
    ``[E,H,I/128]``. There is no zero point.
    """

    w13: "torch.Tensor"
    w13_scale: "torch.Tensor"
    w2: "torch.Tensor"
    w2_scale: "torch.Tensor"


def interleave_int4_weights(payload: "torch.Tensor") -> "torch.Tensor":
    """Apply the native SM90 INT4 payload permutation, preserving nibbles.

    Inside each M16/K64 block, the uint16 word at row ``8*h+u`` and K-word
    ``8*d+4*s+l`` moves to row ``8*s+u``, K-word ``8*d+2*l+h``.
    """
    import torch

    if payload.ndim != 3 or payload.dtype not in (torch.int8, torch.uint8):
        raise ValueError("INT4 payload must be int8/uint8 [E,M,K/2]")
    e, rows, packed_k = payload.shape
    if e < 1 or rows < 16 or rows % 16 or packed_k < 32 or packed_k % 32:
        raise ValueError("INT4 interleave requires positive E and M16/K64 alignment")
    return (
        payload.contiguous()
        .view(torch.uint8)
        .reshape(e, rows // 16, 2, 8, packed_k // 32, 2, 2, 4, 2)
        .permute(0, 1, 6, 3, 4, 5, 7, 2, 8)
        .contiguous()
        .reshape(e, rows, packed_k)
    )


def fold_int4_scales(scales: "torch.Tensor") -> "torch.Tensor":
    """Build UINT64 tables in ``[E,M/64,K/128,8,8]`` storage.

    Each table stores positive E4M3 bytes 1..4 and negative bytes 5..8.
    Multiplication rounds to BF16 before the saturating FP8 conversion.
    Each aligned 16-byte pair holds logical rows u and u+8 within an
    M16 block. A warp reads eight distinct pairs with lane-quad broadcasts.
    """
    import torch

    if scales.ndim != 3 or scales.dtype != torch.bfloat16:
        raise ValueError("INT4 scales must be BF16 [E,M,K/128]")
    e, rows, groups = scales.shape
    if e < 1 or rows < 64 or rows % 64 or groups < 1:
        raise ValueError("INT4 scale folding requires positive E/K and M64 alignment")
    magnitudes = torch.arange(1, 9, dtype=torch.bfloat16, device=scales.device)
    # Clamp after BF16 multiplication: this also saturates BF16 overflow and
    # avoids the NaN result of an out-of-range float8_e4m3fn cast.
    values = (scales.unsqueeze(-1) * magnitudes).clamp(max=448)
    fp8_bytes = values.to(torch.float8_e4m3fn).view(torch.uint8)
    fp8_bytes[..., 4:] |= 0x80
    tables = fp8_bytes.view(torch.uint64).squeeze(-1)
    return (
        tables.reshape(e, rows // 64, 4, 2, 8, groups)
        .permute(0, 1, 5, 2, 4, 3)
        .contiguous()
        .reshape(e, rows // 64, groups, 8, 8)
    )


def _interleave_gate_up_8(tensor: "torch.Tensor") -> "torch.Tensor":
    e, rows, cols = tensor.shape
    return (
        tensor.reshape(e, 2, rows // 16, 8, cols)
        .permute(0, 2, 1, 3, 4)
        .contiguous()
        .reshape(e, rows, cols)
    )


def _validate_dimensions(hidden_size: int, intermediate_size: int) -> None:
    for name, size in (
        ("hidden_size", hidden_size),
        ("intermediate_size", intermediate_size),
    ):
        if size <= 0 or size % INT4_GROUP_SIZE:
            raise ValueError(f"{name} must be a positive multiple of 128")


def preprocess_int4_mega_weights(
    weights: Int4WeightPack,
    *,
    hidden_size: int,
    intermediate_size: int,
    expert_chunk_size: int = 4,
) -> TransformedMegaWeights:
    """Prepare canonical INT4 weights for the fused SM90 launch.

    Each returned leg is ``(packed_K_major, register_tables,
    unit_activation_scale, unit_expert_scale)``. The UINT64 tables encode
    ``RN_E4M3_satfinite(RN_BF16(q * scale))``. FC1 receives the gate8/up8
    interleave. Floating-point weight quantization is not performed here.
    """
    import torch

    _validate_dimensions(hidden_size, intermediate_size)
    if expert_chunk_size <= 0:
        raise ValueError("expert_chunk_size must be positive")
    if weights.w13.ndim != 3:
        raise ValueError("w13 must be a packed [E,2*I,H/2] tensor")
    experts = weights.w13.shape[0]
    if experts < 1:
        raise ValueError("INT4 weights require at least one local expert")
    shapes = ((2 * intermediate_size, hidden_size), (hidden_size, intermediate_size))
    for name, (rows, k) in zip(("w13", "w2"), shapes, strict=True):
        payload, scales = getattr(weights, name), getattr(weights, name + "_scale")
        if payload.dtype not in (torch.int8, torch.uint8) or payload.shape != (
            experts,
            rows,
            k // 2,
        ):
            raise ValueError(
                f"{name} payload must be int8/uint8 {(experts, rows, k // 2)}"
            )
        if scales.dtype != torch.bfloat16 or scales.shape != (experts, rows, k // 128):
            raise ValueError(f"{name} scales must be BF16 {(experts, rows, k // 128)}")
        if not payload.is_contiguous() or not scales.is_contiguous():
            raise ValueError(f"{name} payload and scales must be contiguous")
        if payload.device != weights.w13.device or scales.device != payload.device:
            raise ValueError("INT4 weights and scales must share one device")
        if not bool((torch.isfinite(scales) & (scales > 0)).all()):
            raise ValueError(f"{name} scales must be finite and positive")

    byte_codes = torch.tensor(
        [INT4_RECODE[i & 15] | (INT4_RECODE[i >> 4] << 4) for i in range(256)],
        dtype=torch.uint8,
        device=weights.w13.device,
    )
    legs = []
    for leg, (name, (rows, k)) in enumerate(zip(("w13", "w2"), shapes, strict=True)):
        payload, scales = getattr(weights, name), getattr(weights, name + "_scale")
        output = torch.empty(
            (experts, rows, k // 2), dtype=torch.uint8, device=payload.device
        )
        output_scales = torch.empty(
            (experts, rows // 64, k // 128, 8, 8),
            dtype=torch.uint64,
            device=payload.device,
        )
        for begin in range(0, experts, expert_chunk_size):
            end = min(begin + expert_chunk_size, experts)
            payload_part, scales_part = payload[begin:end], scales[begin:end]
            if leg == 0:
                payload_part = _interleave_gate_up_8(payload_part)
                scales_part = _interleave_gate_up_8(scales_part)
            recoded = byte_codes[payload_part.to(torch.int64) & 255]
            output[begin:end].copy_(interleave_int4_weights(recoded))
            output_scales[begin:end].copy_(fold_int4_scales(scales_part))
        legs.append(
            (
                output.transpose(1, 2),
                output_scales,
                torch.ones(1, dtype=torch.float32, device=payload.device),
                torch.ones(experts, dtype=torch.float32, device=payload.device),
            )
        )
    return legs[0], legs[1]


__all__ = [
    "Int4WeightPack",
    "TransformedWeightLeg",
    "TransformedMegaWeights",
    "preprocess_int4_mega_weights",
    "interleave_int4_weights",
    "fold_int4_scales",
]
