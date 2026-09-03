"""Reference NVFP4 (E2M1) quantization helpers for tests.

Written against the OCP FP4 ``E2M1`` definition rather than against any kernel,
so kernel tests that quantize with these helpers exercise the kernel's own
understanding of the packed layout.

``E2M1`` encodes a 4-bit value as ``sign(1) | exponent(2) | mantissa(1)``, which
gives the eight magnitudes ``{0, 0.5, 1, 1.5, 2, 3, 4, 6}`` in code order, with
bit 3 as the sign bit. Two elements share one byte, the even element in the low
nibble.

Block scaling is applied on top: element ``i`` decodes to
``E2M1_VALUES[code] * scale[i // block_size]``. Callers supply the block scales,
because the on-the-wire scale encoding is format specific (``UE8M0`` exponent
bytes, ``E4M3`` bytes, or plain FP32).
"""

from __future__ import annotations

import torch

FLOAT4_E2M1_MAX = 6.0

# Magnitudes for codes 0..7; bit 3 of the nibble carries the sign.
E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)

# Midpoints between adjacent magnitudes, and the code that a value landing
# exactly on a midpoint rounds to under round-half-to-even.
_E2M1_MIDPOINTS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
_E2M1_TIE_CODES = (0, 2, 2, 4, 4, 6, 6)


def e2m1_magnitude_codes(x: torch.Tensor) -> torch.Tensor:
    """Round ``|x|`` to the nearest E2M1 magnitude code (0..7), ties to even."""
    a = x.abs().float()
    code = torch.zeros_like(a, dtype=torch.uint8)
    for midpoint in _E2M1_MIDPOINTS:
        code = code + (a > midpoint).to(torch.uint8)
    for i, midpoint in enumerate(_E2M1_MIDPOINTS):
        code = torch.where(
            a == midpoint, torch.full_like(code, _E2M1_TIE_CODES[i]), code
        )
    return code


def e2m1_encode(x: torch.Tensor) -> torch.Tensor:
    """Round ``x`` to E2M1 and return the 4-bit codes (0..15) as uint8."""
    sign = (torch.signbit(x.float())).to(torch.uint8) << 3
    return sign | e2m1_magnitude_codes(x)


def e2m1_decode(codes: torch.Tensor) -> torch.Tensor:
    """Decode E2M1 codes (0..15) to FP32 values."""
    lut = torch.tensor(E2M1_VALUES, dtype=torch.float32, device=codes.device)
    magnitude = lut[(codes & 0x7).long()]
    return torch.where(codes & 0x8 != 0, -magnitude, magnitude)


def e2m1_pack(codes: torch.Tensor) -> torch.Tensor:
    """Pack an even-length trailing dim of E2M1 codes 2-per-byte, low nibble first."""
    assert codes.shape[-1] % 2 == 0, "packed dimension must be even"
    low = codes[..., 0::2]
    high = codes[..., 1::2]
    return (low & 0xF) | ((high & 0xF) << 4)


def e2m1_unpack(packed: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`e2m1_pack`."""
    low = packed & 0xF
    high = (packed >> 4) & 0xF
    return torch.stack((low, high), dim=-1).flatten(start_dim=-2)


def ue8m0_block_scales(
    x: torch.Tensor, block_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-block power-of-2 scales sized so ``x / scale`` fits the E2M1 range.

    Returns ``(scale_fp32, ue8m0_bytes)`` with a trailing block dimension. The
    byte is the IEEE-754 exponent field of the FP32 power-of-2 scale, i.e.
    ``scale == 2 ** (byte - 127)``.
    """
    assert x.shape[-1] % block_size == 0
    blocks = x.float().unflatten(-1, (-1, block_size))
    amax = blocks.abs().amax(dim=-1).clamp(min=1e-30)
    scale = torch.pow(2.0, torch.ceil(torch.log2(amax / FLOAT4_E2M1_MAX)))
    ue8m0 = ((scale.view(torch.int32) >> 23) & 0xFF).to(torch.uint8)
    return scale, ue8m0


def ue8m0_to_scale(ue8m0: torch.Tensor) -> torch.Tensor:
    """Decode UE8M0 exponent bytes to FP32 scales."""
    return torch.pow(2.0, ue8m0.float() - 127.0)


def nvfp4_quantize_blocks(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Quantize ``x`` to packed E2M1 bytes given per-block ``scale``.

    ``scale`` has the same shape as ``x`` with the last dim divided by the block
    size. The returned tensor has the last dim of ``x`` halved.
    """
    block_size = x.shape[-1] // scale.shape[-1]
    blocks = x.float().unflatten(-1, (-1, block_size))
    codes = e2m1_encode(blocks / scale.unsqueeze(-1)).flatten(start_dim=-2)
    return e2m1_pack(codes)


def nvfp4_dequantize_blocks(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`nvfp4_quantize_blocks`; returns FP32 values."""
    codes = e2m1_unpack(packed)
    block_size = codes.shape[-1] // scale.shape[-1]
    values = e2m1_decode(codes).unflatten(-1, (-1, block_size))
    return (values * scale.unsqueeze(-1)).flatten(start_dim=-2)
