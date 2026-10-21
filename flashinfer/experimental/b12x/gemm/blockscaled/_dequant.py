"""MXFP8 weight expansion for the large-M A16 lowering.

``mxfp8_a16_dequant`` plans expand the packed E4M3 weight with its UE8M0 block
scales into caller-provided BF16 scratch and run a BF16 GEMM. Every product of
an E4M3 value and a power-of-two scale is exact in BF16, so the result matches
A16 over the same weight values.
"""

from __future__ import annotations

import triton
import triton.language as tl

from ._quantize import _scale_offset

DEQUANT_BLOCK_K = 512


@triton.jit
def _dequantize_mxfp8(W, S, OUT, K: tl.constexpr, BLOCK_K: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    k = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)
    mask = k < K
    w = tl.load(W + row * K + k, mask, 0.0).to(tl.float32)
    e = tl.load(S + _scale_offset(row, k // 32, K // 32), mask, 127).to(tl.float32)
    tl.store(OUT + row * K + k, (w * tl.exp2(e - 127.0)).to(tl.bfloat16), mask)
