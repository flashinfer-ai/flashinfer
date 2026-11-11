# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Register lookup for signed INT4 weights with prepared FP8 byte tables.

Prepared nibbles put the sign in bit 2 and the magnitude index in bits 0, 1
and 3, with code 11 reserved for zero. Each K128 row stores eight positive
FP8 magnitudes in a uint64. Two PRMT lookups and XOR recover signed values.
"""

import functools

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import Int32, T


INT4_FOLD_BLOCK_BYTES = 64 * 8


def _table_registers(row):
    return f".reg .b32 a{row}0, a{row}1, b{row}0, b{row}1;"


def _table_lines(row):
    return [
        f"xor.b32 a{row}1, a{row}0, 0x7f7f7f7f;",
        f"and.b32 b{row}0, b{row}1, 0x00ffffff;",
        f"or.b32 b{row}1, b{row}1, 0x80808080;",
    ]


def _decode_lines(src, lo, hi, out0, out1):
    # PRMT's selector bit 3 replicates the selected byte's sign bit.
    # A's bytes have sign 0; B's negative entries supply 0xff to complement
    # A, while its positive entries supply 0. Zero-filled tables remain zero.
    return [
        f"xor.b32 flip, {src}, 0x88888888;",
        f"prmt.b32 a, a{lo}0, a{lo}1, {src};",
        f"prmt.b32 b, b{lo}0, b{lo}1, flip;",
        f"xor.b32 {out0}, a, b;",
        f"shr.u32 idx, {src}, 16;",
        "shr.u32 flip, flip, 16;",
        f"prmt.b32 a, a{hi}0, a{hi}1, idx;",
        f"prmt.b32 b, b{hi}0, b{hi}1, flip;",
        f"xor.b32 {out1}, a, b;",
    ]


def _word_asm():
    lines = ["{", ".reg .b32 idx, flip, a, b, r0, r1;"]
    for row in range(2):
        lines += [
            _table_registers(row),
            f"mov.b64 {{a{row}0,b{row}1}}, ${3 + row};",
            *_table_lines(row),
        ]
    lines += _decode_lines("$2", 0, 1, "r0", "r1")
    return "\n".join(lines + ["mov.b32 $0, r0;", "mov.b32 $1, r1;", "}"])


@cute.jit
def convert_int4x8_to_fp8x8(
    src: Int32, lo_table: cutlass.Uint64, hi_table: cutlass.Uint64
):
    """Convert two rows of four recoded nibbles to two FP8x4 registers."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [src.ir_value(), lo_table.ir_value(), hi_table.ir_value()],
        _word_asm(),
        "=r,=r,r,l,l",
        has_side_effects=True,
    )
    return (
        Int32(llvm.extractvalue(T.i32(), result, [0])),
        Int32(llvm.extractvalue(T.i32(), result, [1])),
    )


@functools.lru_cache(maxsize=None)
def _group_asm(count):
    outputs, words = 8 * count, 4 * count
    address_base = outputs + words
    lines = ["{", ".reg .b32 idx, flip, a, b;"]
    for row in range(4):
        lines += [
            _table_registers(row),
            f"ld.shared.v2.b32 {{a{row}0,b{row}1}}, [${address_base + row}];",
            *_table_lines(row),
        ]
    for j in range(count):
        for word in range(4):
            row = (word // 2) * 2
            dest = j * 8 + word * 2
            lines += _decode_lines(
                f"${outputs + j * 4 + word}",
                row,
                row + 1,
                f"${dest}",
                f"${dest + 1}",
            )
    return "\n".join(lines + ["}"])


@cute.jit
def convert_int4_group(
    packed: cute.Tensor,
    fp8: cute.Tensor,
    offsets: cute.Tensor,
    first_fragment: cutlass.Constexpr,
    first_offset: cutlass.Constexpr,
    count: cutlass.Constexpr,
    stage: Int32,
):
    """Reuse four row tables across K32 fragments within one K128 group."""
    assert count in (1, 2, 4)
    assert first_offset // 4 == (first_offset + count - 1) // 4
    values = []
    for j in cutlass.range_constexpr(count):
        words = cute.recast_tensor(packed[(None, None, first_fragment + j)], Int32)
        assert cute.size(words) == 4
        for word in cutlass.range_constexpr(4):
            values.append(words[word])
    off = offsets[(None, None, first_offset, stage)]
    div = cute.zipped_divide(off, cute.make_layout(8))
    vm = cute.group_modes(div, 1, cute.rank(div))
    for pair in cutlass.range_constexpr(2):
        row = cute.filter(vm[(None, pair * 2)])
        assert cute.size(row) in (2, 8)
        hi = 4 if cute.size(row) == 8 else 1
        values.append(Int32(row.iterator.toint()))
        values.append(Int32((row.iterator + cute.crd2idx(hi, row.layout)).toint()))
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * (8 * count)),
        [v.ir_value() for v in values],
        _group_asm(count),
        ",".join(["=r"] * (8 * count) + ["r"] * (4 * count + 4)),
        has_side_effects=True,
    )
    for j in cutlass.range_constexpr(count):
        dst = cute.recast_tensor(fp8[(None, None, first_fragment + j)], Int32)
        for word in cutlass.range_constexpr(8):
            dst[word] = Int32(llvm.extractvalue(T.i32(), result, [j * 8 + word]))


def _make_scale_smem_layout(tile_m: int, num_stages: int, k128_blocks: int):
    elements_per_m = 64 * k128_blocks
    return cute.make_layout(
        (8, 8, tile_m // 64, k128_blocks, num_stages),
        stride=(1, 8, elements_per_m, 64, (tile_m // 64) * elements_per_m),
    )


def make_offset_smem_layout(tile_m: int, num_stages: int):
    return _make_scale_smem_layout(tile_m, num_stages, 1)


def make_offset_smem_layout_k256(tile_m: int, num_stages: int):
    return _make_scale_smem_layout(tile_m, num_stages, 2)


@cute.jit
def _make_expanded_scale_view(
    raw_scales: cute.Tensor,
    tile_m: cutlass.Constexpr,
    k128_blocks: cutlass.Constexpr,
):
    # Consecutive logical M rows occupy consecutive uint64 entries, avoiding
    # the bank conflicts introduced by the transposed 16-bit scale layout.
    elements_per_m = 64 * k128_blocks
    layout = cute.make_layout(
        ((8, 8, tile_m // 64), (128, k128_blocks), raw_scales.shape[4]),
        stride=(
            (1, 8, elements_per_m),
            (0, 64),
            (tile_m // 64) * elements_per_m,
        ),
    )
    return cute.make_tensor(raw_scales.iterator, layout)


@cute.jit
def make_expanded_offset_view(raw_scales: cute.Tensor, tile_m: cutlass.Constexpr):
    return _make_expanded_scale_view(raw_scales, tile_m, 1)


@cute.jit
def make_expanded_offset_view_k256(raw_scales: cute.Tensor, tile_m: cutlass.Constexpr):
    return _make_expanded_scale_view(raw_scales, tile_m, 2)
