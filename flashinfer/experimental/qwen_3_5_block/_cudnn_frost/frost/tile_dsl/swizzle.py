# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT


import cutlass
import cutlass.cute as cute


@cute.jit
def swizzle_xor_128b(row, col_elem, *, elem_bytes: cutlass.Constexpr[int] = 2):
    return col_elem ^ ((row & 7) * cutlass.const_expr(16 // elem_bytes))


@cute.jit
def swizzle_xor_64b(row, col_elem, *, elem_bytes: cutlass.Constexpr[int] = 2):
    chunk_elems = 16 // elem_bytes
    chunk_idx = col_elem // chunk_elems
    in_chunk = col_elem % chunk_elems
    swz_chunk = chunk_idx ^ ((row >> 1) & 3)
    return swz_chunk * chunk_elems + in_chunk


@cute.jit
def swizzle_xor_32b(row, col_elem, *, elem_bytes: cutlass.Constexpr[int] = 2):
    chunk_elems = 16 // elem_bytes
    chunk_idx = col_elem // chunk_elems
    in_chunk = col_elem % chunk_elems
    swz_chunk = chunk_idx ^ ((row >> 2) & 1)
    return swz_chunk * chunk_elems + in_chunk


@cute.jit
def swizzle_xor(
    row: cutlass.Int32,
    col: cutlass.Int32,
    row_stride: cutlass.Constexpr[int],
    elem_bytes: cutlass.Constexpr[int],
) -> cutlass.Int32:
    """Return the physical SMEM column for an XOR-swizzled row-major tile.

    The XOR is applied at the 16-byte boundary for all element widths.
    ``elem_bytes`` selects the element-domain shift and swizzle chunk size.
    """
    row_stride_bytes = cutlass.const_expr(row_stride * elem_bytes)
    if cutlass.const_expr(row_stride_bytes % 128 == 0):
        chunk_elems = 128 // elem_bytes
        swizzled = swizzle_xor_128b(row, col % chunk_elems, elem_bytes=elem_bytes)
    elif cutlass.const_expr(row_stride_bytes % 64 == 0):
        chunk_elems = 64 // elem_bytes
        swizzled = swizzle_xor_64b(row, col % chunk_elems, elem_bytes=elem_bytes)
    else:
        chunk_elems = 32 // elem_bytes
        swizzled = swizzle_xor_32b(row, col % chunk_elems, elem_bytes=elem_bytes)
    return (col // chunk_elems) * chunk_elems + swizzled


@cute.jit
def swizzle_lin_128b(lin, *, row_stride_log2: cutlass.Constexpr[int], elem_bytes: cutlass.Constexpr[int] = 2):
    chunk_log2 = cutlass.const_expr((16 // elem_bytes).bit_length() - 1)
    shift = cutlass.const_expr(row_stride_log2 - chunk_log2)
    mask = cutlass.const_expr(0x7 << chunk_log2)
    return lin ^ ((lin >> shift) & mask)
