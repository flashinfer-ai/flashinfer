# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Block-scaled (quantized) attention outputs shared by the PrimTS FMHA epilogues.

The output is quantized as the 2-D matrix ``[tokens, Hq * Dv]``: the format's
data dtype with one scale per ``sf_vec_size`` consecutive channels of a token
row. The formats are the ``QuantOutputFormat`` entries of
``prims_ts/quant_output.py``; kernel configs name them with cutlass dtypes, which
``quant_output_format`` maps to that table.

UE8M0 scales with E4M3 data (MXFP8) follow the following recipe: for a
block whose largest magnitude is ``amax``,

    scale = 2 ** (floor(log2(amax)) - 8)
    data  = cvt.rn.satfinite.e4m3(value / scale)

so the block's largest value lands in ``[256, 512)`` and saturates to 448
above the E4M3 maximum. An all-zero block stores exponent byte 0.

Every format stores its scales in the **128x4 swizzled** layout of the
block-scaled GEMMs (``mxfp8_quantize(..., is_sf_swizzled_layout=True)``): a
flat buffer of ``round_up(tokens, 128) * round_up(Hq * Dv / sf_vec_size, 4)``
one-byte scales, where token ``r`` and scale column ``c`` live at

    ((r // 128) * (cols // 4) + c // 4) * 512
        + (r % 32) * 16 + ((r % 128) // 32) * 4 + c % 4.

Padding rows and columns are never written.
"""

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T
from cutlass.experimental import primitives as prims

from ..quant_output import QuantOutputFormat
from ..quant_output import quant_output_format as _quant_output_format_for_torch

# Bytes of one 128-row by 4-column swizzled scale block.
_SF_BLOCK_BYTES = 512


# Cutlass dtypes that name quantized-output data and scales, with their torch
# counterparts in the format table.
_TORCH_DTYPES = {
    cutlass.Float8E4M3FN: torch.float8_e4m3fn,
    cutlass.Float8E8M0FNU: torch.float8_e8m0fnu,
}


def quant_output_format(data_dtype, sf_dtype) -> QuantOutputFormat | None:
    """The format of cutlass ``(data_dtype, sf_dtype)``; None without scales."""
    if sf_dtype is None:
        return None
    if data_dtype not in _TORCH_DTYPES or sf_dtype not in _TORCH_DTYPES:
        raise NotImplementedError(
            f"unsupported quantized output: data {data_dtype}, scales {sf_dtype}"
        )
    return _quant_output_format_for_torch(
        _TORCH_DTYPES[data_dtype], _TORCH_DTYPES[sf_dtype]
    )


@cutlass.dsl_user_op
def _max_abs3(x, y, z, *, flush_denormals=True, loc=None, ip=None):
    """Largest magnitude of three FP32 values in one SM100 instruction.

    ``flush_denormals`` selects ``.ftz``, which reads denormal inputs as zero.
    """
    ftz = ".ftz" if flush_denormals else ""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(v).ir_value(loc=loc, ip=ip) for v in (x, y, z)],
            f"max.abs{ftz}.f32 $0, $1, $2, $3;",
            "=f,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def block_amax(
    values,
    start: cutlass.Constexpr[int],
    count: cutlass.Constexpr[int],
    flush_denormals: cutlass.Constexpr[bool] = True,
):
    """Largest magnitude of ``values[start:start + count]``.

    ``count`` is a multiple of four; two independent chains shorten the
    dependency on the scalar maximum. ``flush_denormals=False`` keeps
    denormal magnitudes (see ``fragment_block_amax``).
    """
    m0 = Float32(0.0)
    m1 = Float32(0.0)
    for i in cutlass.range_constexpr(start, start + count, 4):
        m0 = _max_abs3(m0, values[i], values[i + 1], flush_denormals=flush_denormals)
        m1 = _max_abs3(
            m1, values[i + 2], values[i + 3], flush_denormals=flush_denormals
        )
    if cutlass.const_expr(flush_denormals):
        return cute.math.max(m0, m1)
    return _max_abs3(m0, m1, Float32(0.0), flush_denormals=False)


@cute.jit
def _ue8m0_e4m3_block_scale(amax: Float32):
    """UE8M0 byte ``max(e - 8, 0)`` for the biased FP32 exponent ``e`` of
    ``amax``, and the exact power of two ``2 ** (127 - byte)`` that maps the
    block onto the E4M3 grid."""
    biased = (amax.bitcast(Uint32) >> Uint32(23)) & Uint32(0xFF)
    exponent = Uint32(0)
    if biased > Uint32(8):
        exponent = biased - Uint32(8)
    inverse = ((Uint32(254) - exponent) << Uint32(23)).bitcast(Float32)
    return exponent, inverse


@cute.jit
def block_scale(amax: Float32, fmt: cutlass.Constexpr[QuantOutputFormat]):
    """Return ``(scale_bits, inverse)`` for a block with largest magnitude ``amax``.

    ``scale_bits`` is the stored scale byte; multiplying the block by
    ``inverse`` maps it onto the data dtype's grid.
    """
    if cutlass.const_expr(
        fmt.sf_dtype == torch.float8_e8m0fnu and fmt.data_dtype == torch.float8_e4m3fn
    ):
        return _ue8m0_e4m3_block_scale(amax)
    raise NotImplementedError(f"no block-scale recipe for {fmt}")


@cute.jit
def sf_offset(token: Int32, sf_col: Int32, sf_col_blocks: Int32) -> Int64:
    """Byte offset of scale column ``sf_col`` of ``token`` in the 128x4 layout.

    ``sf_col_blocks`` is the padded scale-column count divided by four.
    """
    block = Int64(token >> Int32(7)) * Int64(sf_col_blocks) + Int64(sf_col >> Int32(2))
    inner = (
        (token & Int32(31)) * Int32(16)
        + ((token >> Int32(5)) & Int32(3)) * Int32(4)
        + (sf_col & Int32(3))
    )
    return block * Int64(_SF_BLOCK_BYTES) + Int64(inner)


@cute.jit
def store_sf(o_sf: cute.Tensor, offset: Int64, scale_bits: Uint32) -> None:
    """Store one scale byte at ``offset`` of the flat scale buffer."""
    store_sf_at(o_sf.iterator.toint(), offset, scale_bits)


@cute.jit
def store_sf_at(o_sf_base: Int64, offset: Int64, scale_bits: Uint32) -> None:
    """Store one scale byte at ``o_sf_base + offset``."""
    byte_ptr = cutlass.inttoptr(
        o_sf_base + offset,
        mem_space=1,
        dtype=cutlass.Uint8,
    )
    byte_ptr.store(scale_bits.to(cutlass.Uint8))


@cute.jit
def _pack4(
    v0: Float32,
    v1: Float32,
    v2: Float32,
    v3: Float32,
    data_dtype: cutlass.Constexpr[torch.dtype],
) -> Int32:
    """Pack four FP32 values into one word of the data dtype, low byte first."""
    if cutlass.const_expr(data_dtype == torch.float8_e4m3fn):
        return _pack4_e4m3(v0, v1, v2, v3)
    raise NotImplementedError(f"no packer for {data_dtype}")


@cute.jit
def _pack4_e4m3(v0: Float32, v1: Float32, v2: Float32, v3: Float32) -> Int32:
    """Pack four FP32 values into one saturated e4m3x4 word, low byte first."""
    return cute.arch.inline_ptx(
        """{
            .reg .b16 lo, hi;
            cvt.rn.satfinite.e4m3x2.f32 lo, {$r1}, {$r0};
            cvt.rn.satfinite.e4m3x2.f32 hi, {$r3}, {$r2};
            mov.b32 {$w0}, {lo, hi};
        }""",
        write_only_types=[Int32],
        read_only_args=[v0, v1, v2, v3],
    )


@cute.jit
def group_amax(
    amax: Float32,
    lanes: cutlass.Constexpr[int],
    lane_stride: cutlass.Constexpr[int] = 1,
    flush_denormals: cutlass.Constexpr[bool] = True,
) -> Float32:
    """Combine ``amax`` across the ``lanes`` lanes sharing one scale block.

    Group members are ``lane_stride`` apart inside an aligned span of
    ``lanes * lane_stride`` lanes; the butterfly never leaves that span.

    Every lane of the warp must execute this call: no per-lane branch, such
    as a row predicate, may surround it. The shuffles use the full-warp
    mask. A runtime member mask would make ptxas assume partial-warp
    execution throughout the kernel. It then drops its uniform-register
    code elsewhere, for example by wrapping each decode MMA issue in an
    election loop, which slowed long-KV decode by about 10%. Lanes of
    invalid rows exchange harmless values within their own groups.
    """
    if cutlass.const_expr(lanes == 1):
        return amax
    for step in cutlass.range_constexpr(lanes.bit_length() - 1):
        peer = Float32(
            prims.shfl_sync(
                thread_mask=0xFFFFFFFF,
                val=amax,
                offset=(1 << step) * lane_stride,
                mask_and_clamp=0x1F,
                kind=prims.Shfl.BFLY,
            )
        )
        if cutlass.const_expr(flush_denormals):
            amax = cute.math.max(amax, peer)
        else:
            amax = _max_abs3(amax, peer, Float32(0.0), flush_denormals=False)
    return amax


@cute.jit
def fragment_block_amax(
    values,
    count: cutlass.Constexpr[int],
    fmt: cutlass.Constexpr[QuantOutputFormat],
    lane_stride: cutlass.Constexpr[int] = 1,
) -> Float32:
    """Unscaled maximum of the scale block that a ``count``-column fragment shares.

    For callers that store each row behind a per-lane branch: every lane of
    the warp computes this before the branch (see ``group_amax``) and passes
    it to ``quant_store_row_fragment(raw_amax=...)``, which multiplies it by
    the positive ``norm_scale``. Rounding is monotonic for a positive scale
    and denormals are kept, so that product has the exponent, and so the
    scale, of the maximum of the scaled block.
    """
    vec = fmt.sf_vec_size
    assert count < vec and vec % count == 0, "a fragment of one scale block"
    return group_amax(
        block_amax(values, 0, count, flush_denormals=False),
        vec // count,
        lane_stride,
        flush_denormals=False,
    )


@cute.jit
def quant_store_row_fragment(
    values,
    norm_scale,
    o_base: Int64,
    o_offset: Int64,
    o_sf_base: Int64,
    num_qo_heads: Int32,
    store_ok,
    head_dim: cutlass.Constexpr[int],
    count: cutlass.Constexpr[int],
    fmt: cutlass.Constexpr[QuantOutputFormat],
    lane_stride: cutlass.Constexpr[int] = 1,
    raw_amax: Float32 | None = None,
) -> None:
    """Quantize ``count`` contiguous output channels of one row and store them.

    O is the row-major ``[tokens * Hq, head_dim]`` matrix and ``o_offset`` is
    the element offset of ``values[0]``, so its row is ``token * Hq + head``,
    as every decode store addresses its output. Counts of whole blocks are
    owned by this lane. A ``count`` below the format's ``sf_vec_size`` is a
    fragment of one scale block whose ``sf_vec_size // count`` lanes,
    ``lane_stride`` apart, combine their maxima with full-warp shuffles
    (``group_amax``). Either every lane of the warp calls this function,
    with ``store_ok`` predicating the stores of invalid rows, or the caller
    passes ``raw_amax`` from ``fragment_block_amax``, computed on every lane
    before its row branch.
    ``norm_scale`` (``None`` for one) multiplies the values first.
    """
    vec = fmt.sf_vec_size
    assert fmt.data_dtype.itemsize == 1, "packed stores assume one-byte data"
    assert vec % count == 0 if count < vec else count % vec == 0
    lanes = vec // count if count < vec else 1
    assert raw_amax is None or lanes > 1, "raw_amax serves fragments of one block"
    block_count = min(count, vec)
    assert block_count % 4 == 0
    sfs_per_head = head_dim // vec
    row = Int32(o_offset // Int64(head_dim))
    col = Int32(o_offset - Int64(row) * Int64(head_dim))
    token = row // num_qo_heads
    head = row - token * num_qo_heads
    sf_col_blocks = (num_qo_heads * Int32(sfs_per_head) + Int32(3)) >> Int32(2)
    is_sf_writer = cutlass.Boolean(True)
    if cutlass.const_expr(lanes > 1):
        lane = cute.arch.lane_idx()
        is_sf_writer = lane // Int32(lane_stride) % Int32(lanes) == Int32(0)
    for block in cutlass.range_constexpr(count // block_count):
        first = block * block_count
        scaled = cutlass.Array(Float32, block_count, space=cutlass.AddressSpace.rmem)
        for i in cutlass.range_constexpr(block_count):
            value = Float32(values[first + i])
            if cutlass.const_expr(norm_scale is not None):
                value = value * norm_scale
            scaled[i] = value
        if cutlass.const_expr(raw_amax is not None):
            amax = raw_amax
            if cutlass.const_expr(norm_scale is not None):
                amax = raw_amax * norm_scale
        else:
            amax = group_amax(block_amax(scaled, 0, block_count), lanes, lane_stride)
        scale_bits, inverse = block_scale(amax, fmt)
        words = cutlass.Array(Int32, block_count // 4, space=cutlass.AddressSpace.rmem)
        for w in cutlass.range_constexpr(block_count // 4):
            words[w] = _pack4(
                scaled[w * 4] * inverse,
                scaled[w * 4 + 1] * inverse,
                scaled[w * 4 + 2] * inverse,
                scaled[w * 4 + 3] * inverse,
                fmt.data_dtype,
            )
        if store_ok:
            dst = o_base + o_offset + Int64(first)
            for chunk in cutlass.range_constexpr((block_count // 4 + 3) // 4):
                chunk_words = min(4, block_count // 4 - chunk * 4)
                cutlass.inttoptr(
                    dst + Int64(chunk * 16), mem_space=1, dtype=Int32
                ).store(
                    (words.data_ptr() + chunk * 4).load(count=chunk_words, alignment=4),
                    alignment=chunk_words * 4,
                )
            if is_sf_writer:
                sf_col = head * Int32(sfs_per_head) + (col + Int32(first)) // Int32(vec)
                store_sf_at(
                    o_sf_base, sf_offset(token, sf_col, sf_col_blocks), scale_bits
                )
