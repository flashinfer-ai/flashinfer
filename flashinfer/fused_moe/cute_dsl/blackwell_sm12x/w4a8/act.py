"""Fused SiLU-gate activation + MXFP8 quant for the W4A8 throughput tier.

Consumes the packed FC1 output and produces the quantized FC2 activation in
one pass:

    fc1_out [rows, 2n] bf16 -> q [rows, n] e4m3 + sf [rows, n/32] u8

Column convention (matches ``moe_reference_w4a8_mx`` and the kernel-order
[up; gate] weight layout): up = cols [0, n), gate = cols [n, 2n);
``y = silu(gate) * up = sigmoid(gate) * gate * up`` computed in fp32, then
the EXACT bit-math MXFP8 quantization shared with
:func:`b12x.moe.fused.w4a8.quant.mxfp8_quant_block`.

``valid_rows`` (optional, device i32 scalar) lets a capacity-sized launch
early-exit rows at/beyond the live packed-row count without any host sync --
required for the CUDA-graph-friendly pipeline where the grid is sized by
worst-case route capacity.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

from .quant import mxfp8_quant_block, pick_blocks_per_program

_MX_BLOCK = 32


@triton.jit
def _silu_mul_mxfp8_quant_kernel(
    fc1,
    values,
    scales,
    valid_rows_ptr,
    num_programs,
    supers_per_row,
    n_cols,
    fc1_stride_t,
    values_stride_t,
    scales_stride_t,
    HAS_VALID_ROWS: tl.constexpr,
    HAS_SWIGLU_LIMIT: tl.constexpr,
    SWIGLU_LIMIT: tl.constexpr,
    BLOCK: tl.constexpr,
    BPP: tl.constexpr,
):
    prog = tl.program_id(0)
    if prog >= num_programs:
        return
    token = prog // supers_per_row
    if HAS_VALID_ROWS:
        if token >= tl.load(valid_rows_ptr):
            return
    sup = prog % supers_per_row
    offs = sup * (BPP * BLOCK) + tl.arange(0, BPP * BLOCK)

    up = tl.load(fc1 + token * fc1_stride_t + offs).to(tl.float32)
    gate = tl.load(fc1 + token * fc1_stride_t + n_cols + offs).to(tl.float32)
    if HAS_SWIGLU_LIMIT:
        gate = tl.minimum(gate, SWIGLU_LIMIT)
        up = tl.minimum(tl.maximum(up, -SWIGLU_LIMIT), SWIGLU_LIMIT)
    y = (gate / (1.0 + tl.exp(-gate))) * up  # silu(gate) * up, fp32
    payload, byte = mxfp8_quant_block(tl.reshape(y, (BPP, BLOCK)))
    tl.store(
        values + token * values_stride_t + offs,
        tl.reshape(payload, (BPP * BLOCK,)),
    )
    tl.store(scales + token * scales_stride_t + sup * BPP + tl.arange(0, BPP), byte)


def silu_mul_mxfp8_quantize_rows(
    fc1_out: torch.Tensor,
    *,
    out_values: torch.Tensor | None = None,
    out_scales: torch.Tensor | None = None,
    valid_rows: torch.Tensor | None = None,
    swiglu_limit: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``fc1_out [rows, 2n]`` -> ``(q [rows, n] e4m3, sf [rows, n/32] u8)``.

    ``valid_rows``: optional device i32 scalar (e.g. packed_route_count);
    rows at/beyond it are skipped (their outputs keep stale bytes -- the
    pipeline never consumes them because their m-blocks carry expert -1).
    """
    if fc1_out.dim() != 2:
        raise ValueError(f"fc1_out must be [rows, 2n], got {tuple(fc1_out.shape)}")
    rows, two_n = int(fc1_out.shape[0]), int(fc1_out.shape[1])
    if two_n % (2 * _MX_BLOCK) != 0:
        raise ValueError(
            f"fc1_out columns must be divisible by {2 * _MX_BLOCK}, got {two_n}"
        )
    if fc1_out.stride(1) != 1:
        raise ValueError("fc1_out rows must be contiguous")
    n = two_n // 2
    blocks_per_row = n // _MX_BLOCK
    if out_values is None:
        out_values = torch.empty(
            rows, n, dtype=torch.float8_e4m3fn, device=fc1_out.device
        )
    if out_scales is None:
        out_scales = torch.empty(
            rows, blocks_per_row, dtype=torch.uint8, device=fc1_out.device
        )
    if out_values.stride(1) != 1 or out_scales.stride(1) != 1:
        raise ValueError("output rows must be contiguous")
    if valid_rows is not None and valid_rows.dtype != torch.int32:
        raise TypeError("valid_rows must be an int32 device scalar")
    has_swiglu_limit = swiglu_limit is not None
    swiglu_limit_value = 0.0
    if has_swiglu_limit:
        swiglu_limit_value = float(swiglu_limit)
        if not math.isfinite(swiglu_limit_value) or swiglu_limit_value <= 0.0:
            raise ValueError(
                f"swiglu_limit must be positive and finite, got {swiglu_limit_value}"
            )
    bpp = pick_blocks_per_program(blocks_per_row)
    supers_per_row = blocks_per_row // bpp
    num_programs = rows * supers_per_row
    if num_programs > 0:
        _silu_mul_mxfp8_quant_kernel[(num_programs,)](
            fc1_out,
            out_values,
            out_scales,
            valid_rows if valid_rows is not None else fc1_out,
            num_programs,
            supers_per_row,
            n,
            fc1_out.stride(0),
            out_values.stride(0),
            out_scales.stride(0),
            HAS_VALID_ROWS=valid_rows is not None,
            HAS_SWIGLU_LIMIT=has_swiglu_limit,
            SWIGLU_LIMIT=swiglu_limit_value,
            BLOCK=_MX_BLOCK,
            BPP=bpp,
            num_warps=1,
        )
    return out_values, out_scales
