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


def reference_preprocess(
    weight: torch.Tensor,
    raw_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference the Humming payload and scales before physical interleaving."""

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
    return processed, offset, residual


__all__ = [
    "HUMMING_GROUP_SIZE",
    "HUMMING_MAX_RANGE",
    "reference_payload_rewrite",
    "reference_payload_rewrite_lut",
    "reference_preprocess",
    "reference_scale_factorization",
]
