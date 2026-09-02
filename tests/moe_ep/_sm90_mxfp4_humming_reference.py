"""Independent Pure-Torch reference for SM90 Humming MXFP4 preprocessing.

This module is test-owned: production code must not import it.  In particular,
it deliberately does not import either the FlashInfer implementation or the
external kernel donor, so permanent tests have an independent oracle.
"""

from __future__ import annotations

import functools
import struct

import torch


HUMMING_GROUP_SIZE = 32
HUMMING_MAX_RANGE = 11
HUMMING_FOLD_M = 64
HUMMING_FOLD_K = 128


def _require_uint8_3d(value: torch.Tensor, name: str) -> None:
    if value.dtype != torch.uint8 or value.dim() != 3:
        raise ValueError(
            f"{name} must be a 3D uint8 tensor; "
            f"got dtype={value.dtype}, shape={tuple(value.shape)}"
        )


def reference_scale_factorization(
    raw_scale: torch.Tensor,
    max_range: int = HUMMING_MAX_RANGE,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return logical ``(offset, residual, delta)`` using signed arithmetic."""

    _require_uint8_3d(raw_scale, "raw_scale")
    if not 0 <= max_range <= HUMMING_MAX_RANGE:
        raise ValueError(
            f"max_range must be in [0, {HUMMING_MAX_RANGE}]; got {max_range}"
        )

    experts = raw_scale.shape[0]
    exponent = raw_scale.contiguous().view(experts, -1).to(torch.int32)
    exponent_max = exponent.amax(dim=1, keepdim=True)
    exponent_min = exponent.amin(dim=1, keepdim=True)
    retained_span = torch.clamp(exponent_max - exponent_min, max=max_range)
    base = exponent_max - retained_span
    clamped = torch.maximum(exponent, base)

    offset = (clamped - base + 1).to(torch.uint8)
    delta = (clamped - exponent).to(torch.uint8)
    # Preserve the production operation order (including the E8M0=255
    # overflow behavior) instead of algebraically folding the final 0.5.
    residual = torch.exp2(base.squeeze(1).to(torch.float32) - 127.0) * 0.5
    return (
        offset.view_as(raw_scale).contiguous(),
        residual.contiguous(),
        delta.view_as(raw_scale).contiguous(),
    )


def _float_from_bits(bits: int) -> float:
    return struct.unpack("f", struct.pack("I", bits & 0xFFFFFFFF))[0]


def _bits_from_float(value: float) -> int:
    return struct.unpack("I", struct.pack("f", value))[0]


@functools.cache
def reference_payload_rewrite_lut() -> torch.Tensor:
    """Build the exact Humming ``delta x E2M1-code`` rewrite table on CPU."""

    def decode_e2m1(code: int) -> float:
        sign = (code & 0x8) << 28
        exponent_mantissa = (code & 0x7) << 22
        return _float_from_bits(sign | exponent_mantissa)

    def encode_e2m1(value: float) -> int:
        value_bits = _bits_from_float(value)
        mask = 0x81C00000
        round_zero_bits = value_bits & mask
        round_up_bits = (value_bits + 0x00200000) & mask
        round_zero = _float_from_bits(round_zero_bits)
        round_up = _float_from_bits(round_up_bits)
        # This intentionally matches Humming's round-up-on-tie convention.
        rounded = (
            round_up_bits
            if abs(value - round_zero) >= abs(value - round_up)
            else round_zero_bits
        )
        return ((rounded & 0x80000000) >> 28) | ((rounded & 0x01C00000) >> 22)

    result = torch.empty((256, 16), dtype=torch.uint8)
    for delta in range(256):
        # Humming constructs 2^-delta by subtracting from the FP32 exponent
        # field.  Preserve the 32-bit bit-pattern behavior for extreme deltas.
        scale = _float_from_bits(0x3F800000 - (delta << 23))
        for code in range(16):
            rewritten = 0 if code == 8 else code  # canonicalize negative zero
            if delta:
                rewritten = encode_e2m1(decode_e2m1(rewritten) * scale)
            result[delta, code] = rewritten
    return result


def reference_payload_rewrite(
    weight: torch.Tensor,
    delta: torch.Tensor,
) -> torch.Tensor:
    """Rewrite canonical packed E2M1 payload according to each K32 delta."""

    _require_uint8_3d(weight, "weight")
    _require_uint8_3d(delta, "delta")
    experts, rows, packed_k = weight.shape
    logical_k = packed_k * 2
    expected_delta_shape = (experts, rows, logical_k // HUMMING_GROUP_SIZE)
    if logical_k % HUMMING_GROUP_SIZE or tuple(delta.shape) != expected_delta_shape:
        raise ValueError(
            f"delta must have shape {expected_delta_shape}; got {tuple(delta.shape)}"
        )
    if weight.device != delta.device:
        raise ValueError("weight and delta must share a device")

    low = weight & 0x0F
    high = (weight >> 4) & 0x0F
    codes = torch.stack((low, high), dim=-1).reshape(experts, rows, logical_k)
    expanded_delta = delta.repeat_interleave(HUMMING_GROUP_SIZE, dim=-1).long()
    lut = reference_payload_rewrite_lut().to(weight.device)
    rewritten = lut[expanded_delta, codes.long()]
    return (rewritten[..., 0::2] | (rewritten[..., 1::2] << 4)).contiguous()


def reference_fold_offsets(offset: torch.Tensor) -> torch.Tensor:
    """Fold logical K32 offsets into ``[E, N/64, K/128, 16, 16]``."""

    _require_uint8_3d(offset, "offset")
    experts, rows, k32_groups = offset.shape
    if rows % HUMMING_FOLD_M:
        raise ValueError(f"offset rows must be divisible by 64; got {rows}")
    if k32_groups % (HUMMING_FOLD_K // HUMMING_GROUP_SIZE):
        raise ValueError(f"offset K32 groups must be divisible by 4; got {k32_groups}")

    output = torch.empty(
        (experts, rows // 64, k32_groups // 4, 16, 16),
        dtype=torch.uint8,
        device=offset.device,
    )
    for m64 in range(rows // 64):
        for k128 in range(k32_groups // 4):
            for folded_m in range(16):
                for m_slice in range(4):
                    row = m64 * 64 + m_slice * 16 + folded_m
                    for k32 in range(4):
                        output[:, m64, k128, folded_m, m_slice * 4 + k32] = offset[
                            :, row, k128 * 4 + k32
                        ]
    return output.contiguous()


def reference_unfold_offsets(folded: torch.Tensor) -> torch.Tensor:
    """Invert :func:`reference_fold_offsets`."""

    if folded.dtype != torch.uint8 or folded.dim() != 5:
        raise ValueError("folded offset must be a 5D uint8 tensor")
    experts, m64_blocks, k128_blocks, folded_m_size, physical_cols = folded.shape
    if (folded_m_size, physical_cols) != (16, 16):
        raise ValueError(
            "folded offset trailing shape must be (16, 16); "
            f"got {(folded_m_size, physical_cols)}"
        )

    output = torch.empty(
        (experts, m64_blocks * 64, k128_blocks * 4),
        dtype=torch.uint8,
        device=folded.device,
    )
    for m64 in range(m64_blocks):
        for k128 in range(k128_blocks):
            for folded_m in range(16):
                for m_slice in range(4):
                    row = m64 * 64 + m_slice * 16 + folded_m
                    for k32 in range(4):
                        output[:, row, k128 * 4 + k32] = folded[
                            :, m64, k128, folded_m, m_slice * 4 + k32
                        ]
    return output.contiguous()


def _preprocess_signs(word: int) -> int:
    exponent_mantissa = word & 0x77777777
    signs = (
        ((word & 0x00000008) << 4)
        | ((word & 0x00000080) << 8)
        | ((word & 0x00000800) << 12)
        | ((word & 0x00008000) << 16)
        | ((word & 0x00080000) >> 16)
        | ((word & 0x00800000) >> 12)
        | ((word & 0x08000000) >> 8)
        | ((word & 0x80000000) >> 4)
    )
    return (exponent_mantissa | signs) & 0xFFFFFFFF


def reference_interleave_weight(weight: torch.Tensor) -> torch.Tensor:
    """Scalar transcription of the SM90 FP4-for-FP8 physical interleave."""

    _require_uint8_3d(weight, "weight")
    experts, rows, packed_k = weight.shape
    logical_k = packed_k * 2
    if rows % 16:
        raise ValueError(f"weight rows must be divisible by 16; got {rows}")
    if logical_k % 64:
        raise ValueError(f"logical K must be divisible by 64; got {logical_k}")
    if packed_k % 2:
        raise ValueError("packed K byte count must be even")

    original_device = weight.device
    source = weight.detach().cpu().contiguous()
    source_u16 = source[..., 0::2].to(torch.int64) | (
        source[..., 1::2].to(torch.int64) << 8
    )
    output_u16 = torch.empty_like(source_u16)

    for expert in range(experts):
        for block_id in range(rows // 2):
            row = (block_id // 8) * 16 + block_id % 8
            for partition in range(logical_k // 64):
                for lane in range(16):
                    destination_row = row + ((lane % 8) // 4) * 8
                    source_column = partition * 16 + lane
                    destination_column = (
                        partition * 16 + (lane // 8) * 8 + (lane % 4) * 2
                    )
                    word = int(source_u16[expert, row, source_column]) | (
                        int(source_u16[expert, row + 8, source_column]) << 16
                    )
                    word = _preprocess_signs(word)
                    output_u16[expert, destination_row, destination_column] = (
                        word & 0xFFFF
                    )
                    output_u16[expert, destination_row, destination_column + 1] = (
                        word >> 16
                    ) & 0xFFFF

    output = torch.empty_like(source)
    output[..., 0::2] = (output_u16 & 0xFF).to(torch.uint8)
    output[..., 1::2] = ((output_u16 >> 8) & 0xFF).to(torch.uint8)
    return output.to(original_device).contiguous()


def reference_preprocess(
    weight: torch.Tensor,
    raw_scale: torch.Tensor,
    *,
    interleave: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference the complete Humming preprocessing contract."""

    _require_uint8_3d(weight, "weight")
    _require_uint8_3d(raw_scale, "raw_scale")
    expected_scale_shape = (
        weight.shape[0],
        weight.shape[1],
        weight.shape[2] * 2 // HUMMING_GROUP_SIZE,
    )
    if tuple(raw_scale.shape) != expected_scale_shape:
        raise ValueError(
            f"raw_scale must have shape {expected_scale_shape}; "
            f"got {tuple(raw_scale.shape)}"
        )
    if weight.device != raw_scale.device:
        raise ValueError("weight and raw_scale must share a device")

    offset, residual, delta = reference_scale_factorization(raw_scale)
    processed = reference_payload_rewrite(weight, delta)
    if not interleave:
        return processed, offset, residual
    return (
        reference_interleave_weight(processed),
        reference_fold_offsets(offset),
        residual,
    )


__all__ = [
    "HUMMING_GROUP_SIZE",
    "HUMMING_MAX_RANGE",
    "reference_fold_offsets",
    "reference_interleave_weight",
    "reference_payload_rewrite",
    "reference_payload_rewrite_lut",
    "reference_preprocess",
    "reference_scale_factorization",
    "reference_unfold_offsets",
]
