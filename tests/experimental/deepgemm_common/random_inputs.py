"""Random inputs in the DeepGEMM-family storage conventions, plus references.

The family tests under tests/experimental/*_generated.py build constant-filled
operands (``0x22`` E2M1 pairs, ``0x7F7F7F7F`` UE8M0 words, ``torch.ones`` FP8)
whose analytic result is ``K * alpha``; a layout or permutation bug that keeps
the sum intact passes them. These builders produce random operands in the same
storage conventions and return the dequantized float32 view of every operand,
so a test can compare the kernel against ``reference_gemm`` on data where every
element matters.

Conventions (as consumed by the generated programs):

- E2M1 (FP4): two codes per ``uint8``; the lower nibble is the earlier element
  along K. Codes follow ``E2M1_VALUES`` (sign bit 3, exponent bits 2..1,
  mantissa bit 0).
- UE8M0 scale words: one byte per 32-element K block, four consecutive K
  blocks per ``int32`` word (little-endian, the earliest block in the lowest
  byte), words indexed ``[K // 128, rows]`` i.e. K-major with the row in the
  trailing dimension. Byte ``0x7F`` is the scale ``2**0``.
- FP8 E4M3 with float32 per-block scales: one scale per 128 K elements
  (``[rows, K // 128]``, the activation side) or one per 128x128 tile
  (``[rows // 128, K // 128]``, the weight side), as the batched projection
  family uses.
- FP8 E4M3 with UE8M0 scale words at granularity ``gran_k`` (32 or 128 K
  elements per byte), as the mixed FP8xFP4 family uses.
"""

from __future__ import annotations

import torch

# E2M1 code -> value (code = sign<<3 | exponent<<1 | mantissa).
E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)

# Output-dtype tolerances for comparing a kernel against ``reference_gemm`` on
# operands that are exactly representable (the reference consumes the same
# quantized values the kernel reads, so only accumulation order and the output
# rounding differ). ``fp8`` applies to dequantized dynamic-FP8 outputs.
TOLERANCES = {
    torch.bfloat16: dict(atol=1e-2, rtol=1e-2),
    torch.float32: dict(atol=1e-3, rtol=1e-4),
    torch.float8_e4m3fn: dict(atol=0.1, rtol=0.1),
}


def seeded_generator(seed: int, device) -> torch.Generator:
    """A generator on ``device`` seeded with ``seed`` (reproducible rows per test)."""
    generator = torch.Generator(device=torch.device(device))
    generator.manual_seed(seed)
    return generator


def _e2m1_round(x: torch.Tensor) -> torch.Tensor:
    """Round ``x`` to the nearest E2M1 value (ties follow the FlashInfer FP4 test helper)."""
    sign = torch.sign(x)
    a = x.abs().clone()
    out = torch.zeros_like(a)
    out[(a > 0.25) & (a < 0.75)] = 0.5
    out[(a >= 0.75) & (a <= 1.25)] = 1.0
    out[(a > 1.25) & (a < 1.75)] = 1.5
    out[(a >= 1.75) & (a <= 2.5)] = 2.0
    out[(a > 2.5) & (a < 3.5)] = 3.0
    out[(a >= 3.5) & (a <= 5.0)] = 4.0
    out[a > 5.0] = 6.0
    return out * sign


def e2m1_pack(values: torch.Tensor) -> torch.Tensor:
    """Pack float values (already E2M1-representable) into ``uint8`` pairs along the last dim."""
    if values.shape[-1] % 2:
        raise ValueError("E2M1 packing needs an even trailing dimension")
    v = values.to(torch.float32)
    magnitudes = torch.tensor(E2M1_VALUES[:8], device=v.device, dtype=torch.float32)
    match = v.abs().unsqueeze(-1) == magnitudes
    if not torch.all(match.any(-1)):
        raise ValueError("values are not E2M1-representable; round them first")
    codes = match.to(torch.int8).argmax(-1).to(torch.uint8)
    codes = codes | (torch.signbit(v).to(torch.uint8) << 3)
    pairs = codes.reshape(*values.shape[:-1], values.shape[-1] // 2, 2)
    return (pairs[..., 0] | (pairs[..., 1] << 4)).to(torch.uint8)


def e2m1_unpack(packed: torch.Tensor) -> torch.Tensor:
    """Unpack ``uint8`` E2M1 pairs to float32 (lower nibble first)."""
    table = torch.tensor(E2M1_VALUES, device=packed.device, dtype=torch.float32)
    low = packed & 0xF
    high = (packed >> 4) & 0xF
    codes = torch.stack((low, high), dim=-1)
    return table[codes.long()].reshape(*packed.shape[:-1], packed.shape[-1] * 2)


def ue8m0_pack_words(exponents: torch.Tensor) -> torch.Tensor:
    """Pack UE8M0 exponent bytes ``[blocks, rows]`` into ``int32`` words ``[blocks // 4, rows]``."""
    if exponents.shape[0] % 4:
        raise ValueError("UE8M0 word packing needs a multiple of four K blocks")
    e = exponents.to(torch.int64)
    if torch.any((e < 0) | (e > 255)):
        raise ValueError("UE8M0 exponents must be in [0, 255]")
    groups = e.reshape(e.shape[0] // 4, 4, *e.shape[1:])
    words = groups[:, 0] | (groups[:, 1] << 8) | (groups[:, 2] << 16) | (groups[:, 3] << 24)
    # Reinterpret the unsigned 32-bit pattern as int32.
    words = torch.where(words >= 2**31, words - 2**32, words)
    return words.to(torch.int32)


def ue8m0_unpack_words(words: torch.Tensor) -> torch.Tensor:
    """Unpack ``int32`` UE8M0 words ``[words, rows]`` to exponent bytes ``[words * 4, rows]``."""
    w = words.to(torch.int64) & 0xFFFFFFFF
    bytes_ = torch.stack([(w >> (8 * i)) & 0xFF for i in range(4)], dim=1)
    return bytes_.reshape(words.shape[0] * 4, *words.shape[1:])


def _ue8m0_scale(exponents: torch.Tensor) -> torch.Tensor:
    """UE8M0 byte -> scale value ``2 ** (byte - 127)``."""
    return torch.exp2(exponents.to(torch.float32) - 127.0)


def random_fp4_operand(
    rows: int,
    k: int,
    *,
    generator: torch.Generator,
    device,
    exponent_range: tuple[int, int] = (125, 129),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Random E2M1 operand ``[rows, k]`` with UE8M0 block scales.

    Returns ``(packed uint8 [rows, k // 2], scale words int32 [k // 128, rows],
    dequantized float32 [rows, k])``.
    """
    if k % 128:
        raise ValueError("FP4 operands need K to be a multiple of 128 (four 32-element scale blocks per word)")
    raw = torch.randn((rows, k), generator=generator, device=device) * 2.0
    values = _e2m1_round(raw)
    packed = e2m1_pack(values)
    lo, hi = exponent_range
    exponents = torch.randint(lo, hi, (k // 32, rows), generator=generator, device=device)
    words = ue8m0_pack_words(exponents)
    scale = _ue8m0_scale(exponents).T.repeat_interleave(32, dim=1)  # [rows, k]
    return packed, words, values * scale


def random_fp8_blockwise(
    rows: int,
    k: int,
    *,
    generator: torch.Generator,
    device,
    block: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Random FP8 E4M3 operand ``[rows, k]`` with float32 scales per ``block`` K elements.

    Returns ``(fp8 [rows, k], scales float32 [rows, k // block], dequantized float32 [rows, k])``.
    """
    if k % block:
        raise ValueError(f"K={k} is not a multiple of the scale block {block}")
    raw = torch.randn((rows, k), generator=generator, device=device)
    finfo = torch.finfo(torch.float8_e4m3fn)
    blocks = raw.reshape(rows, k // block, block)
    amax = blocks.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    scales = (amax / finfo.max).to(torch.float32)
    quantized = (blocks / scales).clamp(finfo.min, finfo.max).to(torch.float8_e4m3fn)
    fp8 = quantized.reshape(rows, k)
    dequantized = (quantized.to(torch.float32) * scales).reshape(rows, k)
    return fp8, scales.reshape(rows, k // block), dequantized


def random_fp8_block2d(
    rows: int,
    k: int,
    *,
    generator: torch.Generator,
    device,
    block_rows: int = 128,
    block_k: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Random FP8 E4M3 operand ``[rows, k]`` with one float32 scale per ``block_rows x block_k`` tile.

    Returns ``(fp8 [rows, k], scales float32 [rows // block_rows, k // block_k],
    dequantized float32 [rows, k])``.
    """
    if rows % block_rows or k % block_k:
        raise ValueError(f"({rows}, {k}) is not tiled by ({block_rows}, {block_k})")
    raw = torch.randn((rows, k), generator=generator, device=device)
    finfo = torch.finfo(torch.float8_e4m3fn)
    tiles = raw.reshape(rows // block_rows, block_rows, k // block_k, block_k)
    amax = tiles.abs().amax(dim=(1, 3), keepdim=True).clamp(min=1e-12)
    scales = (amax / finfo.max).to(torch.float32)
    quantized = (tiles / scales).clamp(finfo.min, finfo.max).to(torch.float8_e4m3fn)
    fp8 = quantized.reshape(rows, k)
    dequantized = (quantized.to(torch.float32) * scales).reshape(rows, k)
    return fp8, scales.reshape(rows // block_rows, k // block_k), dequantized


def random_fp8_ue8m0(
    rows: int,
    k: int,
    *,
    generator: torch.Generator,
    device,
    gran_k: int = 32,
    exponent_range: tuple[int, int] = (125, 129),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Random FP8 E4M3 operand ``[rows, k]`` with UE8M0 scale words at ``gran_k`` granularity.

    Returns ``(fp8 [rows, k], scale words int32 [k // (4 * gran_k), rows],
    dequantized float32 [rows, k])``.
    """
    if k % (4 * gran_k):
        raise ValueError(f"K={k} is not a multiple of four {gran_k}-element scale blocks")
    raw = torch.randn((rows, k), generator=generator, device=device) * 64.0
    fp8 = raw.to(torch.float8_e4m3fn)
    lo, hi = exponent_range
    exponents = torch.randint(lo, hi, (k // gran_k, rows), generator=generator, device=device)
    words = ue8m0_pack_words(exponents)
    scale = _ue8m0_scale(exponents).T.repeat_interleave(gran_k, dim=1)
    return fp8, words, fp8.to(torch.float32) * scale


def reference_gemm(
    a: torch.Tensor, b: torch.Tensor, *, alpha: float = 1.0, out_dtype=torch.float32
) -> torch.Tensor:
    """``alpha * a @ b.T`` of dequantized operands, accumulated in float64."""
    result = torch.matmul(a.to(torch.float64), b.to(torch.float64).T) * alpha
    return result.to(out_dtype)


def assert_close(actual: torch.Tensor, expected: torch.Tensor, *, out_dtype=None) -> None:
    """``torch.testing.assert_close`` with the tolerance of the kernel's output dtype."""
    dtype = actual.dtype if out_dtype is None else out_dtype
    if dtype not in TOLERANCES:
        raise KeyError(f"no DeepGEMM-family tolerance for output dtype {dtype}")
    torch.testing.assert_close(
        actual.to(torch.float32), expected.to(torch.float32), **TOLERANCES[dtype]
    )
