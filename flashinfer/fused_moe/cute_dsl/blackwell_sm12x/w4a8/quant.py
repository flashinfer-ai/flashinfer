"""MXFP8 activation quantization for the W4A8 throughput-tier grouped GEMM.

Quantizes BF16 activations to E4M3 payload bytes plus per-32-element UE8M0
block scales, both in plain row-major layout:

    x  [m, K] bf16  ->  q [m, K] e4m3 (float8_e4m3fn), sf [m, K//32] uint8

Numerics match ``quant_dequant_mxfp8_torch`` / the in-kernel
``quantize_block_fp8_mx``: the block scale is ``pow2_ceil(amax / 448)``
(self-ranging, no global scale) and the payload is RN-saturating E4M3 of
``x * 2**-exp``. An all-zero block stores scale byte 0 (decoded as a zero
output scale) and a zero payload.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_MX_BLOCK = 32


@triton.jit
def mxfp8_quant_block(src):
    """Bit-exact MXFP8 quant of fp32 32-blocks: -> (e4m3 payload, u8 bytes).

    ``src`` is ``(BPP, 32)`` -- one row per 32-element block (use
    ``tl.reshape`` for a single block). The shared spec
    (quant_dequant_mxfp8_torch / in-kernel quantize_block_fp8_mx):
    pow2_ceil_ue8m0 of amax/448 via the fp32 exponent-bump (bump iff the
    mantissa is nonzero -- NOT ceil(log2), which diverges within ~1ulp of
    power-of-two boundaries); payload is RN-saturating E4M3 of
    src * 2^(127-byte). A zero block falls through to scale byte 0 + zero
    payload.
    """
    max_abs = tl.max(tl.abs(src), axis=1)
    scaled = max_abs * 0.002232142857142857  # fp32(1/448), as the oracle does
    bits = scaled.to(tl.int32, bitcast=True)
    mant = bits & 0x007FFFFF
    bumped = tl.where(mant != 0, (bits + 0x00800000) & 0x7F800000, bits)
    byte = (bumped >> 23) & 0xFF
    # inv = 2^(127 - byte); byte 0 decodes to a zero output scale.
    inv_bits = tl.maximum(254 - byte, 0) << 23
    inv = tl.where(byte == 0, 0.0, inv_bits.to(tl.float32, bitcast=True))
    payload = tl.minimum(tl.maximum(src * inv[:, None], -448.0), 448.0)
    return payload.to(tl.float8e4nv), byte.to(tl.uint8)


def pick_blocks_per_program(blocks_per_row: int) -> int:
    """Largest supported per-program block count dividing the row evenly."""
    for bpp in (8, 4, 2, 1):
        if blocks_per_row % bpp == 0:
            return bpp
    return 1


@triton.jit
def _mxfp8_row_quant_kernel(
    source,
    values,
    scales,
    num_programs,
    supers_per_row,
    source_stride_t,
    values_stride_t,
    scales_stride_t,
    BLOCK: tl.constexpr,
    BPP: tl.constexpr,
):
    prog = tl.program_id(0)
    if prog >= num_programs:
        return
    token = prog // supers_per_row
    sup = prog % supers_per_row
    offs = sup * (BPP * BLOCK) + tl.arange(0, BPP * BLOCK)

    src = tl.load(source + token * source_stride_t + offs).to(tl.float32)
    payload, byte = mxfp8_quant_block(tl.reshape(src, (BPP, BLOCK)))
    tl.store(
        values + token * values_stride_t + offs,
        tl.reshape(payload, (BPP * BLOCK,)),
    )
    tl.store(scales + token * scales_stride_t + sup * BPP + tl.arange(0, BPP), byte)


def mxfp8_quantize_rows(
    x: torch.Tensor,
    *,
    out_values: torch.Tensor | None = None,
    out_scales: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``x [m, K]`` to row-major MXFP8 payload + UE8M0 K/32 scales."""
    if x.dim() != 2:
        raise ValueError(f"x must be [m, K], got {tuple(x.shape)}")
    m, k = int(x.shape[0]), int(x.shape[1])
    if k % _MX_BLOCK != 0:
        raise ValueError(f"K must be divisible by {_MX_BLOCK}, got {k}")
    if x.stride(1) != 1:
        raise ValueError("x rows must be contiguous")
    blocks_per_row = k // _MX_BLOCK
    if out_values is None:
        out_values = torch.empty(m, k, dtype=torch.float8_e4m3fn, device=x.device)
    if out_scales is None:
        out_scales = torch.empty(m, blocks_per_row, dtype=torch.uint8, device=x.device)
    if out_values.stride(1) != 1 or out_scales.stride(1) != 1:
        raise ValueError("output rows must be contiguous")
    bpp = pick_blocks_per_program(blocks_per_row)
    supers_per_row = blocks_per_row // bpp
    num_programs = m * supers_per_row
    if num_programs > 0:
        _mxfp8_row_quant_kernel[(num_programs,)](
            x,
            out_values,
            out_scales,
            num_programs,
            supers_per_row,
            x.stride(0),
            out_values.stride(0),
            out_scales.stride(0),
            BLOCK=_MX_BLOCK,
            BPP=bpp,
            num_warps=1,
        )
    return out_values, out_scales
