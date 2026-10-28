# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""SM90 NVFP4 x FP8 layouts and conversion primitives.

Weights remain packed E2M1, with one raw nonnegative E4M3 scale per K16.
The caller appends a replicated shared conversion table (1 to 32 KiB, see
``Nvfp4ConversionLayout``) to the scale ring and initializes it once before
warp specialization. Promotion computes RN_E4M3(q*s/8) and supplies register
operand A to K32 RS WGMMA; the existing epilogue applies 8G. MXFP4-prefixed
layout names preserve the private ABI.
"""

import functools

import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import Int32, T
from cutlass._mlir.dialects import llvm

# Packed payload layout is identical to MXFP4. Use the current vendor's
# views so new LDSM/layout fixes also apply to NVFP4.
from moe_hopper_fp8.mxfp4_cutedsl import (
    make_offset_smem_layout as _make_offset_smem_layout,
    make_offset_smem_layout_k256 as _make_offset_smem_layout_k256,
    make_packed_a_ldsm_views,
    make_packed_a_ldsm_views_k256,
    make_packed_a_ldsm_views_k256_half,
)

from .nvfp4_policy import NVFP4_DEFAULT_CONVERSION, Nvfp4ConversionLayout


MXFP4_GROUP_SIZE = 16
MXFP4_K_TILE = 128
MXFP4_FOLD_M = 64
MXFP4_FOLDED_M = 32
MXFP4_M_SLICES = MXFP4_FOLD_M // MXFP4_FOLDED_M
MXFP4_SCALE_GROUPS_PER_TILE = MXFP4_K_TILE // MXFP4_GROUP_SIZE
MXFP4_PHYSICAL_COLS = MXFP4_M_SLICES * MXFP4_SCALE_GROUPS_PER_TILE
MXFP4_FOLD_BLOCK_BYTES = MXFP4_FOLDED_M * MXFP4_PHYSICAL_COLS


@cute.jit
def nvfp4_lut(scale: Int32):
    """Build eight RN-E4M3(q*s/8) candidates, including subnormal scales."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [scale.ir_value()],
        HIGH_ASM,
        "=r,=r,r",
        has_side_effects=False,
    )
    return (
        Int32(llvm.extractvalue(T.i32(), result, [0])),
        Int32(llvm.extractvalue(T.i32(), result, [1])),
    )


@cute.jit
def nvfp4_word_original(
    src: Int32, scale: Int32, upper: cutlass.Constexpr, residual: cutlass.Constexpr
):
    assert not residual, "NVFP4 promotion has no residual operand"
    lo, hi = nvfp4_lut(scale)
    selector = src & Int32(0x77777777)
    signs = src
    if cutlass.const_expr(upper):
        selector = selector >> Int32(16)
        signs = src << Int32(4)
    result = llvm.inline_asm(
        T.i32(),
        [lo.ir_value(), hi.ir_value(), selector.ir_value(), signs.ir_value()],
        "{ .reg .b32 value, sign; prmt.b32 value, $1, $2, $3; "
        "and.b32 sign, $4, 0x80808080; xor.b32 $0, value, sign; }",
        "=r,r,r,r,r",
        has_side_effects=False,
    )
    return Int32(result)


HIGH_ASM = """{
.reg .b32 m, shift, e, b, c, d, u, v, bits;
.reg .b64 tab, shifted;
.reg .pred fast, normal;
.reg .f32 sf, f0, f1, f2, f3, f4, f5, f6, f7;
.reg .b16 h01, h23, h45, h67, h0, h1;
setp.ge.u32 fast, $2, 40;
@!fast bra SLOW;
and.b32 m, $2, 7;
shl.b32 shift, m, 3;
and.b32 e, $2, 120;
mov.b64 tab, 0x43424241403f3e3c;
shr.u64 shifted, tab, shift;
cvt.u32.u64 c, shifted;
and.b32 c, c, 255;
add.u32 c, c, e;
sub.u32 c, c, 80;
sub.u32 b, $2, 24;
sub.u32 u, b, 8;
shl.b32 $0, u, 8;
shl.b32 v, b, 16;
or.b32 $0, $0, v;
shl.b32 v, c, 24;
or.b32 $0, $0, v;
add.u32 $1, b, 8;
add.u32 u, c, 8;
shl.b32 u, u, 8;
or.b32 $1, $1, u;
add.u32 v, b, 16;
shl.b32 v, v, 16;
or.b32 $1, $1, v;
add.u32 u, c, 16;
shl.b32 u, u, 24;
or.b32 $1, $1, u;
bra DONE;
SLOW:
setp.ge.u32 normal, $2, 8;
shl.b32 bits, $2, 20;
add.u32 bits, bits, 0x3c000000;
mov.b32 sf, bits;
@normal bra SCALE_DONE;
cvt.rn.f32.u32 sf, $2;
mul.f32 sf, sf, 0f3b000000;
SCALE_DONE:
mul.f32 f0, sf, 0f00000000;
mul.f32 f1, sf, 0f3d800000;
mul.f32 f2, sf, 0f3e000000;
mul.f32 f3, sf, 0f3e400000;
mul.f32 f4, sf, 0f3e800000;
mul.f32 f5, sf, 0f3ec00000;
mul.f32 f6, sf, 0f3f000000;
mul.f32 f7, sf, 0f3f400000;
cvt.rn.satfinite.e4m3x2.f32 h01, f1, f0;
cvt.rn.satfinite.e4m3x2.f32 h23, f3, f2;
cvt.rn.satfinite.e4m3x2.f32 h45, f5, f4;
cvt.rn.satfinite.e4m3x2.f32 h67, f7, f6;
mov.b32 $0, {h01, h23};
mov.b32 $1, {h45, h67};
DONE:
}"""


def make_offset_smem_layout(tile_m: int, num_stages: int) -> cute.Layout:
    """Use the shared folded layout with K16 scale rows."""
    return _make_offset_smem_layout(tile_m, num_stages, folded_rows=MXFP4_FOLDED_M)


@cute.jit
def make_expanded_offset_view(
    raw_offsets: cute.Tensor,
    tile_m: cutlass.Constexpr,
) -> cute.Tensor:
    """Patch3's transposed K16 pairs; preserve the contiguous copy ABI."""
    m_blocks = tile_m // 64
    stages = raw_offsets.shape[4]
    expanded = cute.make_layout(
        ((16, 4, m_blocks), (16, ((2, 4), 1)), stages),
        stride=((2, 128, 512), (0, ((1, 32), 512)), m_blocks * 512),
    )
    return cute.make_tensor(raw_offsets.iterator, expanded)


def make_offset_smem_layout_k256(tile_m: int, num_stages: int) -> cute.Layout:
    """Use the shared folded layout with K16 scale rows."""
    return _make_offset_smem_layout_k256(tile_m, num_stages, folded_rows=MXFP4_FOLDED_M)


@cute.jit
def make_expanded_offset_view_k256(
    raw_offsets: cute.Tensor,
    tile_m: cutlass.Constexpr,
) -> cute.Tensor:
    """Patch3's transposed K16 pairs; preserve the contiguous copy ABI."""
    m_blocks = tile_m // 64
    stages = raw_offsets.shape[4]
    expanded = cute.make_layout(
        ((16, 4, m_blocks), (16, ((2, 4), 2)), stages),
        stride=((2, 128, 1024), (0, ((1, 32), 512)), m_blocks * 1024),
    )
    return cute.make_tensor(raw_offsets.iterator, expanded)


@cute.jit
def nvfp4_get_lut(
    raw_scales: cute.Tensor,
    layout: cutlass.Constexpr = NVFP4_DEFAULT_CONVERSION,
):
    ptr = cute.recast_ptr(
        raw_scales.iterator + cute.cosize(raw_scales.layout), dtype=cutlass.Uint64
    )
    return cute.make_tensor(ptr, cute.make_layout(128 * layout.lut_replicas))


@cute.jit
def nvfp4_initialize_lut(lookup: cute.Tensor, tidx: Int32):
    replicas = cute.size(lookup) // 128
    if tidx < Int32(128):
        bits = cutlass.Uint64(0)
        if tidx < Int32(127):
            lo, hi = nvfp4_lut(tidx)
            bits = cutlass.Uint64(cutlass.Uint32(lo)) | (
                cutlass.Uint64(cutlass.Uint32(hi)) << cutlass.Uint64(32)
            )
        for replica in cutlass.range_constexpr(replicas):
            lookup[tidx * Int32(replicas) + Int32(replica)] = bits


@cute.jit
def nvfp4_word_smem(
    src: Int32, scale: Int32, upper: cutlass.Constexpr, lookup: cute.Tensor
):
    # Lane-quad replica (all replicas are equal); tables with fewer than eight
    # replicas fold the eight quads onto the ones they have.
    replicas = cute.size(lookup) // 128
    replica = cute.arch.lane_idx() // Int32(4)
    if cutlass.const_expr(replicas < 8):
        replica = replica & Int32(replicas - 1)
    bits = lookup[scale * Int32(replicas) + replica]
    lo = Int32(bits & cutlass.Uint64(0xFFFFFFFF))
    hi = Int32(bits >> cutlass.Uint64(32))
    selector = src & Int32(0x77777777)
    signs = src
    if cutlass.const_expr(upper):
        selector = selector >> Int32(16)
        signs = src << Int32(4)
    result = llvm.inline_asm(
        T.i32(),
        [lo.ir_value(), hi.ir_value(), selector.ir_value(), signs.ir_value()],
        "{ .reg .b32 value, sign; prmt.b32 value, $1, $2, $3; "
        "and.b32 sign, $4, 0x80808080; or.b32 $0, value, sign; }",
        "=r,r,r,r,r",
        has_side_effects=False,
    )
    return Int32(result)


NVFP4_SCALE_TRANSPOSE = True


# Conversion PTX ported from patch3 final commit; original MMA/wait order retained.
@functools.lru_cache(maxsize=None)
def _nvfp4_super_asm(groups: int, layout: Nvfp4ConversionLayout) -> str:
    # Expose several K32 fragments to one inline PTX block. Retain the
    # production RS fragment coordinates; no scale equality is assumed.
    # Operands: $0.. outputs, then 4*groups packed words, 4*groups u16 scale
    # addresses (per fragment and pair: row-0 and row-1 pairs), LUT base.
    n = 8 * groups
    replicas = layout.lut_replicas
    prmt_rows = replicas == 32
    lines = [
        "{",
        f".reg .b32 sf<{n}>, addr<{n}>, lut<{2 * n}>, sel, signs, lane, base, tmp;",
    ]
    if prmt_rows:
        # 256-byte rows: a row address is base + ((scale << 8) | replica * 8),
        # assembled by one PRMT from the raw u16 scale pair and `rep`.
        lines += [".reg .b32 rep;", "mov.u32 lane, %laneid;"]
        if layout.lut_map == "quad":
            lines += ["shl.b32 rep, lane, 1;", "and.b32 rep, rep, 56;"]
        elif layout.lut_map == "pair":
            lines += ["shl.b32 rep, lane, 2;", "and.b32 rep, rep, 120;"]
        else:
            lines.append("shl.b32 rep, lane, 3;")
        lines.append(f"mov.u32 base, ${16 * groups};")
    else:
        lines.append("mov.u32 lane, %laneid;")
        if layout.lut_map == "quad":
            lines += [
                f"and.b32 tmp, lane, {4 * (replicas - 1)};",
                "shl.b32 tmp, tmp, 1;",
            ]
        elif layout.lut_map == "pair":
            lines += [
                "shl.b32 tmp, lane, 2;",
                f"and.b32 tmp, tmp, {8 * (replicas - 1)};",
            ]
        else:
            lines += [f"and.b32 tmp, lane, {replicas - 1};", "shl.b32 tmp, tmp, 3;"]
        lines.append(f"add.u32 base, ${16 * groups}, tmp;")

    # Output word j decodes with the LUT row loaded for word lut_of[j]; with a
    # K-permuted layout several words share one scale and one row load.
    lut_of = list(range(n))
    byte_sel = {}

    def scale_pair(lo, hi, address):
        # u16 at `address`: byte 0 is the scale of word lo, byte 1 of word hi.
        lines.append(f"ld.shared.u16 sf{lo}, [{address}];")
        if prmt_rows:
            # Keep the raw pair; the PRMT that builds each row address picks the byte.
            lines.append(f"mov.b32 sf{hi}, sf{lo};")
            byte_sel[lo], byte_sel[hi] = "0x3204", "0x3214"
        else:
            lines.extend(
                [f"shr.u32 sf{hi}, sf{lo}, 8;", f"and.b32 sf{lo}, sf{lo}, 255;"]
            )

    if layout.kperm == 64:
        assert groups % 2 == 0, "kperm64 converts K32 fragment pairs (one K64 chunk)"
        for u in range(groups // 2):
            g0 = 2 * u
            for pair in range(2):
                j0 = g0 * 8 + pair * 4
                # The caller pre-selects this lane's address in fragment g0's
                # row-0 operand: lane bit 0 picks the row pair, bit 1 the K32
                # fragment, and the u16 holds (row g, row g+8) of one K16 block.
                scale_pair(j0, j0 + 1, f"${12 * groups + g0 * 4 + pair * 2}")
                for frag in (g0, g0 + 1):
                    jj = frag * 8 + pair * 4
                    lut_of[jj : jj + 4] = [j0, j0 + 1, j0, j0 + 1]
    else:
        for g in range(groups):
            for pair in range(2):
                for row in range(2):
                    j = g * 8 + pair * 4 + row
                    arg = f"${12 * groups + g * 4 + pair * 2 + row}"
                    if layout.kperm == 32:
                        # The caller pre-selects this lane's row-0 operand (lane
                        # bit 1 picks the K16 block); the swapped scale layout
                        # puts (row g, row g+8) of that block in one u16.
                        if row == 0:
                            scale_pair(j, j + 1, arg)
                        lut_of[j + 2] = j
                    else:
                        scale_pair(j, j + 2, arg)
    for j in range(n):
        if lut_of[j] != j:
            continue
        if prmt_rows:
            lines += [
                f"prmt.b32 addr{j}, sf{j}, rep, {byte_sel[j]};",
                f"add.u32 addr{j}, addr{j}, base;",
            ]
        else:
            lines.append(f"mad.lo.u32 addr{j}, sf{j}, {8 * replicas}, base;")

    def decode(start):
        for j in range(start, start + 8):
            src = n + j // 2
            lut = lut_of[j]
            lines.append(f"and.b32 sel, ${src}, 0x77777777;")
            if j % 2:
                lines.append("shr.u32 sel, sel, 16;")
            lines.append(f"prmt.b32 ${j}, lut{2 * lut}, lut{2 * lut + 1}, sel;")
            if j % 2:
                lines.extend(
                    [f"shl.b32 signs, ${src}, 4;", "and.b32 signs, signs, 0x80808080;"]
                )
            else:
                lines.append(f"and.b32 signs, ${src}, 0x80808080;")
            lines.append(f"or.b32 ${j}, ${j}, signs;")

    for start in range(0, n, 8):
        for j in range(start, start + 8):
            if lut_of[j] == j:
                lines.append(
                    f"ld.shared.v2.b32 {{lut{2 * j}, lut{2 * j + 1}}}, [addr{j}];"
                )
        if start:
            decode(start - 8)
    decode(n - 8)
    return "\n".join(lines + ["}"])


@cute.jit
def convert_nvfp4_super(
    words: cute.Tensor,
    addresses: cute.Tensor,
    lut: Int32,
    output: cute.Tensor,
    groups: cutlass.Constexpr,
    layout: cutlass.Constexpr = NVFP4_DEFAULT_CONVERSION,
):
    values = []
    for j in cutlass.range_constexpr(4 * groups):
        values.append(words[j])
    for j in cutlass.range_constexpr(4 * groups):
        values.append(addresses[j])
    values.append(lut)
    asm = _nvfp4_super_asm(groups, layout)
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * (8 * groups)),
        [v.ir_value() for v in values],
        asm,
        ",".join(["=r"] * (8 * groups) + ["r"] * (8 * groups + 1)),
        has_side_effects=True,
    )
    for j in cutlass.range_constexpr(8 * groups):
        output[j] = Int32(llvm.extractvalue(T.i32(), result, [j]))


@cute.jit
def convert_nvfp4_group(
    packed: cute.Tensor,
    fp8: cute.Tensor,
    offsets: cute.Tensor,
    first_fragment: cutlass.Constexpr,
    first_offset: cutlass.Constexpr,
    count: cutlass.Constexpr,
    stage: Int32,
    raw_scales: cute.Tensor,
    layout: cutlass.Constexpr = NVFP4_DEFAULT_CONVERSION,
):
    assert count in (2, 4)
    assert layout.kperm != 64 or first_offset % 2 == 0, "kperm64 needs whole K64"
    words = cute.make_rmem_tensor(cute.make_layout(4 * count), Int32)
    addresses = cute.make_rmem_tensor(cute.make_layout(4 * count), Int32)
    output = cute.make_rmem_tensor(cute.make_layout(8 * count), Int32)
    for j in cutlass.range_constexpr(count):
        w = cute.recast_tensor(packed[(None, None, first_fragment + j)], Int32)
        assert cute.size(w) == 4
        off = offsets[(None, None, first_offset + j, stage)]
        div = cute.zipped_divide(off, cute.make_layout(8))
        vm = cute.group_modes(div, 1, cute.rank(div))
        for i in cutlass.range_constexpr(4):
            words[j * 4 + i] = w[i]
        for pair in cutlass.range_constexpr(2):
            row = cute.filter(vm[(None, pair * 2)])
            assert cute.size(row) in (2, 8)
            hi = 4 if cute.size(row) == 8 else 1
            addresses[j * 4 + pair * 2] = Int32(row.iterator.toint())
            addresses[j * 4 + pair * 2 + 1] = Int32(
                (row.iterator + cute.crd2idx(hi, row.layout)).toint()
            )
    if cutlass.const_expr(layout.kperm != 0):
        # Pre-select this lane's u16 in each fragment's row-0 operand (see
        # _nvfp4_super_asm). After the weight-side scale swap, the row-0/row-1
        # operands of a K32 fragment hold K16 block 0/1 for rows g and g+8.
        # kperm32: lane bit 1 picks the block. kperm64: lane bits 1:0 pick
        # block 2*fragment+row of the K64 chunk. The deltas are layout
        # constants, so the selected address remains a loop-invariant lane
        # offset plus an immediate in SASS (no SELP).
        lane = cute.arch.lane_idx()
        for j in cutlass.range_constexpr(count):
            for pair in cutlass.range_constexpr(2):
                a0 = addresses[j * 4 + pair * 2]
                d_row = addresses[j * 4 + pair * 2 + 1] - a0
                if cutlass.const_expr(layout.kperm == 32):
                    addresses[j * 4 + pair * 2] = (
                        a0 + ((lane >> Int32(1)) & Int32(1)) * d_row
                    )
                elif cutlass.const_expr(j % 2 == 0):
                    d_frag = addresses[(j + 1) * 4 + pair * 2] - a0
                    addresses[j * 4 + pair * 2] = (
                        a0
                        + (lane & Int32(1)) * d_row
                        + ((lane >> Int32(1)) & Int32(1)) * d_frag
                    )
    lut = Int32(nvfp4_get_lut(raw_scales, layout).iterator.toint())
    convert_nvfp4_super(words, addresses, lut, output, count, layout)
    for j in cutlass.range_constexpr(count):
        dst = cute.recast_tensor(fp8[(None, None, first_fragment + j)], Int32)
        for i in cutlass.range_constexpr(8):
            dst[i] = output[j * 8 + i]


@cute.jit
def convert_packed_a_kblock(
    packed_registers: cute.Tensor,
    fp8_fragment_a: cute.Tensor,
    partitioned_offsets: cute.Tensor,
    k_block: cutlass.Constexpr,
    stage_idx: Int32,
    raw_scales: cute.Tensor,
):
    convert_packed_a_kblock_from_offset(
        packed_registers,
        fp8_fragment_a,
        partitioned_offsets,
        k_block,
        k_block,
        stage_idx,
        raw_scales,
    )


@cute.jit
def convert_packed_a_kblock_from_offset(
    packed_registers: cute.Tensor,
    fp8_fragment_a: cute.Tensor,
    partitioned_offsets: cute.Tensor,
    fragment_k_block: cutlass.Constexpr,
    offset_k_block: cutlass.Constexpr,
    stage_idx: Int32,
    raw_scales: cute.Tensor,
):
    lookup = nvfp4_get_lut(raw_scales)
    packed_words = cute.recast_tensor(
        packed_registers[None, None, fragment_k_block], Int32
    )
    fp8_words = cute.recast_tensor(fp8_fragment_a[None, None, fragment_k_block], Int32)
    offsets = partitioned_offsets[None, None, offset_k_block, stage_idx]
    offsets_div = cute.zipped_divide(offsets, cute.make_layout(8))
    offsets_vm = cute.group_modes(offsets_div, 1, cute.rank(offsets_div))
    assert cute.size(packed_words) % 2 == 0
    for pair in cutlass.range_constexpr(cute.size(packed_words) // 2):
        row_scales = cute.filter(offsets_vm[None, pair * 2])
        scale_count = cute.size(row_scales)
        assert scale_count in (2, 8)
        hi_index = 4 if scale_count == 8 else 1
        low_ptr = cute.recast_ptr(row_scales.iterator, dtype=cutlass.Uint16)
        high_ptr = cute.recast_ptr(
            row_scales.iterator + cute.crd2idx(hi_index, row_scales.layout),
            dtype=cutlass.Uint16,
        )
        low_pair = Int32(cute.make_tensor(low_ptr, cute.make_layout(1))[0])
        high_pair = Int32(cute.make_tensor(high_ptr, cute.make_layout(1))[0])
        for half in cutlass.range_constexpr(2):
            word = pair * 2 + half
            s0 = low_pair >> Int32(half * 8) & Int32(255)
            s1 = high_pair >> Int32(half * 8) & Int32(255)
            fp8_words[word * 2] = nvfp4_word_smem(packed_words[word], s0, False, lookup)
            fp8_words[word * 2 + 1] = nvfp4_word_smem(
                packed_words[word], s1, True, lookup
            )


__all__ = [
    "make_packed_a_ldsm_views",
    "make_packed_a_ldsm_views_k256",
    "make_packed_a_ldsm_views_k256_half",
]
