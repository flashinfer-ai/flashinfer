# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""CuTeDSL building blocks for Hopper Humming MXFP4 x FP8.

This module contains only device-side layout and conversion primitives.  The
host preprocessing lives in :mod:`moe_hopper_fp8.mxfp4_humming`; keeping the
two layers separate makes it explicit that the runtime kernel consumes an
already interleaved packed payload and folded exponent offsets.

The SM90 mainloop uses true RS WGMMA: packed E2M1 weights are TMA-staged in
shared memory, loaded with ``ldmatrix.x4`` into registers, expanded to E4M3 by
the paired PRMT converter below, and supplied as WGMMA operand A.  Operand B
remains an FP8 shared-memory descriptor.
"""

from typing import Tuple

import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import Int32, T, dsl_user_op
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import warp


MXFP4_GROUP_SIZE = 32
MXFP4_K_TILE = 128
MXFP4_FOLD_M = 64
MXFP4_FOLDED_M = 16
MXFP4_M_SLICES = MXFP4_FOLD_M // MXFP4_FOLDED_M
MXFP4_SCALE_GROUPS_PER_TILE = MXFP4_K_TILE // MXFP4_GROUP_SIZE
MXFP4_PHYSICAL_COLS = MXFP4_M_SLICES * MXFP4_SCALE_GROUPS_PER_TILE
MXFP4_FOLD_BLOCK_BYTES = MXFP4_FOLDED_M * MXFP4_PHYSICAL_COLS


@cute.jit
def convert_mxfp4_pair_preprocessed_signs(
    src0: Int32,
    src1: Int32,
    lo_exp_offset: Int32,
    hi_exp_offset: Int32,
) -> Tuple[Int32, Int32, Int32, Int32]:
    """Convert two preprocessed ``fp4x8`` words into four ``fp8x4`` words.

    ``src0`` and ``src1`` are the two 32-bit packed operands contributed by
    one WGMMA A lane.  Their low FP8 halves share ``lo_exp_offset`` and their
    high halves share ``hi_exp_offset``.  The offline sign permutation lets
    this runtime path use only two PRMT lookups per half; no sign-gather PRMT
    is needed.

    The return order is ``(src0_lo, src0_hi, src1_lo, src1_hi)`` and can be
    written directly into four consecutive 32-bit words of the FP8 A fragment.
    """

    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()]),
        [
            src0.ir_value(),
            src1.ir_value(),
            lo_exp_offset.ir_value(),
            hi_exp_offset.ir_value(),
        ],
        "{\n"
        "  .reg .b32 em0, em1;\n"
        "  .reg .b32 lo_lut0, lo_lut1, hi_lut0, hi_lut1;\n"
        "  .reg .b32 lo_em0, lo_em1, hi_em0, hi_em1;\n"
        "  .reg .b32 hi_sel0, hi_sel1, hi_sign0, hi_sign1;\n"
        "  and.b32 em0, $4, 0x77777777;\n"
        "  and.b32 em1, $5, 0x77777777;\n"
        "  mad.lo.u32 lo_lut0, $6, 0x08080800, 0x0c080000;\n"
        "  mad.lo.u32 lo_lut1, $6, 0x08080808, 0x1c181410;\n"
        "  mad.lo.u32 hi_lut0, $7, 0x08080800, 0x0c080000;\n"
        "  mad.lo.u32 hi_lut1, $7, 0x08080808, 0x1c181410;\n"
        "  prmt.b32 lo_em0, lo_lut0, lo_lut1, em0;\n"
        "  prmt.b32 lo_em1, lo_lut0, lo_lut1, em1;\n"
        "  shr.u32 hi_sel0, em0, 16;\n"
        "  shr.u32 hi_sel1, em1, 16;\n"
        "  prmt.b32 hi_em0, hi_lut0, hi_lut1, hi_sel0;\n"
        "  prmt.b32 hi_em1, hi_lut0, hi_lut1, hi_sel1;\n"
        "  and.b32 $0, $4, 0x80808080;\n"
        "  and.b32 $2, $5, 0x80808080;\n"
        "  shl.b32 hi_sign0, $4, 4;\n"
        "  shl.b32 hi_sign1, $5, 4;\n"
        "  and.b32 hi_sign0, hi_sign0, 0x80808080;\n"
        "  and.b32 hi_sign1, hi_sign1, 0x80808080;\n"
        "  or.b32 $0, $0, lo_em0;\n"
        "  or.b32 $1, hi_sign0, hi_em0;\n"
        "  or.b32 $2, $2, lo_em1;\n"
        "  or.b32 $3, hi_sign1, hi_em1;\n"
        "}",
        "=r,=r,=r,=r,r,r,r,r",
        has_side_effects=False,
    )
    return (
        Int32(llvm.extractvalue(T.i32(), result, [0])),
        Int32(llvm.extractvalue(T.i32(), result, [1])),
        Int32(llvm.extractvalue(T.i32(), result, [2])),
        Int32(llvm.extractvalue(T.i32(), result, [3])),
    )


def make_offset_smem_layout(
    tile_m: int,
    num_stages: int,
) -> cute.Layout:
    """Return the raw folded-offset layout copied by the TMA-A warp.

    One M64/K128 fold block is 16x16 bytes.  K-tile is fixed at 128, so each
    stage contains ``tile_m / 64`` independent 256-byte blocks.
    """

    if tile_m % MXFP4_FOLD_M != 0:
        raise ValueError(f"tile_m ({tile_m}) must be divisible by {MXFP4_FOLD_M}")
    m_blocks = tile_m // MXFP4_FOLD_M
    stage_bytes = m_blocks * MXFP4_FOLD_BLOCK_BYTES
    return cute.make_layout(
        (
            MXFP4_PHYSICAL_COLS,
            MXFP4_FOLDED_M,
            m_blocks,
            1,
            num_stages,
        ),
        stride=(
            1,
            MXFP4_PHYSICAL_COLS,
            MXFP4_FOLD_BLOCK_BYTES,
            stage_bytes,
            stage_bytes,
        ),
    )


@cute.jit
def make_expanded_offset_view(
    raw_offsets: cute.Tensor,
    tile_m: cutlass.Constexpr,
) -> cute.Tensor:
    """Broadcast a raw folded-offset stage as logical ``(M, K, PIPE)``.

    The layout is the direct CuTeDSL transcription of FlashInfer/CUTLASS's
    ``SmemLayoutWeightScaleExpanded``.  Its K32 scale byte has stride zero
    across the 32 weights it controls, so ``partition_A`` produces offsets in
    exactly the same lane/value order as the packed A fragment.
    """

    m_blocks = tile_m // MXFP4_FOLD_M
    num_stages = raw_offsets.shape[4]
    stage_bytes = m_blocks * MXFP4_FOLD_BLOCK_BYTES
    expanded_layout = cute.make_layout(
        (
            (MXFP4_FOLDED_M, MXFP4_M_SLICES, m_blocks),
            (
                MXFP4_GROUP_SIZE,
                (MXFP4_SCALE_GROUPS_PER_TILE, 1),
            ),
            num_stages,
        ),
        stride=(
            (
                MXFP4_PHYSICAL_COLS,
                MXFP4_SCALE_GROUPS_PER_TILE,
                MXFP4_FOLD_BLOCK_BYTES,
            ),
            (0, (1, MXFP4_FOLD_BLOCK_BYTES)),
            stage_bytes,
        ),
    )
    return cute.make_tensor(raw_offsets.iterator, expanded_layout)


def make_offset_smem_layout_k256(
    tile_m: int,
    num_stages: int,
) -> cute.Layout:
    """Return a K256-only folded-offset layout without changing K128 ABI."""

    if tile_m % MXFP4_FOLD_M != 0:
        raise ValueError(f"tile_m ({tile_m}) must be divisible by {MXFP4_FOLD_M}")
    m_blocks = tile_m // MXFP4_FOLD_M
    k128_blocks_per_tile = 2
    bytes_per_m_block = k128_blocks_per_tile * MXFP4_FOLD_BLOCK_BYTES
    stage_bytes = m_blocks * bytes_per_m_block
    return cute.make_layout(
        (
            MXFP4_PHYSICAL_COLS,
            MXFP4_FOLDED_M,
            m_blocks,
            k128_blocks_per_tile,
            num_stages,
        ),
        stride=(
            1,
            MXFP4_PHYSICAL_COLS,
            bytes_per_m_block,
            MXFP4_FOLD_BLOCK_BYTES,
            stage_bytes,
        ),
    )


@cute.jit
def make_expanded_offset_view_k256(
    raw_offsets: cute.Tensor,
    tile_m: cutlass.Constexpr,
) -> cute.Tensor:
    """Expand two adjacent folded K128 blocks as one logical K256 tile."""

    m_blocks = tile_m // MXFP4_FOLD_M
    k128_blocks_per_tile = 2
    num_stages = raw_offsets.shape[4]
    bytes_per_m_block = k128_blocks_per_tile * MXFP4_FOLD_BLOCK_BYTES
    stage_bytes = m_blocks * bytes_per_m_block
    expanded_layout = cute.make_layout(
        (
            (MXFP4_FOLDED_M, MXFP4_M_SLICES, m_blocks),
            (
                MXFP4_GROUP_SIZE,
                (MXFP4_SCALE_GROUPS_PER_TILE, k128_blocks_per_tile),
            ),
            num_stages,
        ),
        stride=(
            (
                MXFP4_PHYSICAL_COLS,
                MXFP4_SCALE_GROUPS_PER_TILE,
                bytes_per_m_block,
            ),
            (0, (1, MXFP4_FOLD_BLOCK_BYTES)),
            stage_bytes,
        ),
    )
    return cute.make_tensor(raw_offsets.iterator, expanded_layout)


@cute.jit
def make_packed_a_ldsm_views(
    tiled_mma,
    packed_smem_a: cute.Tensor,
    fp8_fragment_a: cute.Tensor,
    thread_idx_in_warpgroup: Int32,
):
    """Build LDSM source/copy views and the nested-K E2M1 RF tensor.

    LDSM is typed as FP8/16-bit matrix data even though the bytes hold packed
    FP4.  Its K extent is therefore half the FP8 fragment's K extent.  Recasting
    the loaded registers back to E2M1 and restoring a nested ``(2, K/2)`` K
    mode prevents the two M64 fragments of an M128 warpgroup from aliasing.
    """

    ldsm_atom = cute.make_copy_atom(
        warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
        cutlass.Float8E4M3FN,
    )
    tiled_copy = cute.make_tiled_copy_A(ldsm_atom, tiled_mma)
    thread_copy = tiled_copy.get_slice(thread_idx_in_warpgroup)

    # `recast_tensor` currently drops a PDSL swizzle from the pointer. TMA
    # has already written the packed bytes with that swizzle, so LDSM must
    # retain it when viewing each pair of E2M1 values as one FP8-sized byte.
    packed_as_fp8 = cute.make_tensor(
        cute.recast_ptr(
            packed_smem_a.iterator,
            cute.make_swizzle(2, 4, 3),
            dtype=cutlass.Float8E4M3FN,
        ),
        cute.recast_layout(
            8,
            4,
            packed_smem_a.layout,
        ),
    )
    smem_partition = thread_copy.partition_S(packed_as_fp8)

    k_blocks = cute.size(fp8_fragment_a, mode=[2])
    ldsm_shape = (
        fp8_fragment_a.shape[0],
        fp8_fragment_a.shape[1],
        k_blocks // 2,
    )
    ldsm_registers = cute.make_rmem_tensor(ldsm_shape, cutlass.Float8E4M3FN)
    copy_view = thread_copy.retile(ldsm_registers)

    first_mode = cute.size(ldsm_registers, mode=[0])
    mma_m = ldsm_registers.shape[1]
    half_k = cute.size(ldsm_registers, mode=[2])
    packed_layout = cute.make_layout(
        (first_mode, mma_m, (2, half_k)),
        stride=(
            1,
            first_mode * 2,
            (first_mode, first_mode * 2 * cute.size(mma_m)),
        ),
    )
    packed_registers = cute.make_tensor(
        cute.recast_ptr(ldsm_registers.iterator, dtype=cutlass.Float4E2M1FN),
        packed_layout,
    )
    return tiled_copy, smem_partition, copy_view, packed_registers


@cute.jit
def make_packed_a_ldsm_views_k256(
    tiled_mma,
    packed_smem_a: cute.Tensor,
    fp8_fragment_a: cute.Tensor,
    thread_idx_in_warpgroup: Int32,
):
    """K256 LDSM views using the SM90 256-column S<3,4,3> swizzle."""

    ldsm_atom = cute.make_copy_atom(
        warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
        cutlass.Float8E4M3FN,
    )
    tiled_copy = cute.make_tiled_copy_A(ldsm_atom, tiled_mma)
    thread_copy = tiled_copy.get_slice(thread_idx_in_warpgroup)
    packed_as_fp8 = cute.make_tensor(
        cute.recast_ptr(
            packed_smem_a.iterator,
            cute.make_swizzle(3, 4, 3),
            dtype=cutlass.Float8E4M3FN,
        ),
        cute.recast_layout(
            8,
            4,
            packed_smem_a.layout,
        ),
    )
    smem_partition = thread_copy.partition_S(packed_as_fp8)

    k_blocks = cute.size(fp8_fragment_a, mode=[2])
    ldsm_shape = (
        fp8_fragment_a.shape[0],
        fp8_fragment_a.shape[1],
        k_blocks // 2,
    )
    ldsm_registers = cute.make_rmem_tensor(ldsm_shape, cutlass.Float8E4M3FN)
    copy_view = thread_copy.retile(ldsm_registers)

    first_mode = cute.size(ldsm_registers, mode=[0])
    mma_m = ldsm_registers.shape[1]
    half_k = cute.size(ldsm_registers, mode=[2])
    packed_layout = cute.make_layout(
        (first_mode, mma_m, (2, half_k)),
        stride=(
            1,
            first_mode * 2,
            (first_mode, first_mode * 2 * cute.size(mma_m)),
        ),
    )
    packed_registers = cute.make_tensor(
        cute.recast_ptr(ldsm_registers.iterator, dtype=cutlass.Float4E2M1FN),
        packed_layout,
    )
    return tiled_copy, smem_partition, copy_view, packed_registers


@cute.jit
def make_packed_a_ldsm_views_k256_half(
    tiled_mma,
    packed_smem_a: cute.Tensor,
    fp8_fragment_a: cute.Tensor,
    thread_idx_in_warpgroup: Int32,
):
    """Build a K128 register fragment over a K256 S<3,4,3> SMEM stage.

    The returned source partition retains all four packed K64 chunks from the
    K256 stage. Callers copy two chunks at a time into the two-chunk K128
    register view, keeping only half of the converted A tile live.
    """

    ldsm_atom = cute.make_copy_atom(
        warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
        cutlass.Float8E4M3FN,
    )
    tiled_copy = cute.make_tiled_copy_A(ldsm_atom, tiled_mma)
    thread_copy = tiled_copy.get_slice(thread_idx_in_warpgroup)
    packed_as_fp8 = cute.make_tensor(
        cute.recast_ptr(
            packed_smem_a.iterator,
            cute.make_swizzle(3, 4, 3),
            dtype=cutlass.Float8E4M3FN,
        ),
        cute.recast_layout(8, 4, packed_smem_a.layout),
    )
    smem_partition = thread_copy.partition_S(packed_as_fp8)

    k_blocks = cute.size(fp8_fragment_a, mode=[2])
    ldsm_registers = cute.make_rmem_tensor(
        (
            fp8_fragment_a.shape[0],
            fp8_fragment_a.shape[1],
            k_blocks // 2,
        ),
        cutlass.Float8E4M3FN,
    )
    copy_view = thread_copy.retile(ldsm_registers)

    first_mode = cute.size(ldsm_registers, mode=[0])
    mma_m = ldsm_registers.shape[1]
    half_k = cute.size(ldsm_registers, mode=[2])
    packed_layout = cute.make_layout(
        (first_mode, mma_m, (2, half_k)),
        stride=(
            1,
            first_mode * 2,
            (first_mode, first_mode * 2 * cute.size(mma_m)),
        ),
    )
    packed_registers = cute.make_tensor(
        cute.recast_ptr(ldsm_registers.iterator, dtype=cutlass.Float4E2M1FN),
        packed_layout,
    )
    return tiled_copy, smem_partition, copy_view, packed_registers


@cute.jit
def convert_packed_a_kblock(
    packed_registers: cute.Tensor,
    fp8_fragment_a: cute.Tensor,
    partitioned_offsets: cute.Tensor,
    k_block: cutlass.Constexpr,
    stage_idx: Int32,
) -> None:
    """Convert one K32 register slot using its partitioned folded offsets."""

    packed_slot = packed_registers[(None, None, k_block)]
    fp8_slot = fp8_fragment_a[(None, None, k_block)]
    packed_words = cute.recast_tensor(packed_slot, Int32)
    fp8_words = cute.recast_tensor(fp8_slot, Int32)

    offsets = partitioned_offsets[(None, None, k_block, stage_idx)]
    offsets_div = cute.zipped_divide(offsets, cute.make_layout(8))
    offsets_vm = cute.group_modes(offsets_div, 1, cute.rank(offsets_div))
    pair_count = cute.size(packed_words) // 2
    for pair in cutlass.range_constexpr(pair_count):
        row_offsets = cute.filter(offsets_vm[(None, pair * 2)])
        lo_offset = Int32(row_offsets[0])
        scale_value_count = cute.size(row_offsets)
        assert scale_value_count in (2, 8), (
            "Humming pair conversion expects two compact row offsets or "
            "one offset per E2M1 lane"
        )
        hi_index = 4 if scale_value_count == 8 else 1
        hi_offset = Int32(row_offsets[hi_index])
        out0, out1, out2, out3 = convert_mxfp4_pair_preprocessed_signs(
            packed_words[pair * 2],
            packed_words[pair * 2 + 1],
            lo_offset,
            hi_offset,
        )
        fp8_words[pair * 4] = out0
        fp8_words[pair * 4 + 1] = out1
        fp8_words[pair * 4 + 2] = out2
        fp8_words[pair * 4 + 3] = out3


@cute.jit
def convert_packed_a_kblock_from_offset(
    packed_registers: cute.Tensor,
    fp8_fragment_a: cute.Tensor,
    partitioned_offsets: cute.Tensor,
    fragment_k_block: cutlass.Constexpr,
    offset_k_block: cutlass.Constexpr,
    stage_idx: Int32,
) -> None:
    """Convert a local fragment slot with an independently selected SF slot."""

    packed_slot = packed_registers[(None, None, fragment_k_block)]
    fp8_slot = fp8_fragment_a[(None, None, fragment_k_block)]
    packed_words = cute.recast_tensor(packed_slot, Int32)
    fp8_words = cute.recast_tensor(fp8_slot, Int32)

    offsets = partitioned_offsets[(None, None, offset_k_block, stage_idx)]
    offsets_div = cute.zipped_divide(offsets, cute.make_layout(8))
    offsets_vm = cute.group_modes(offsets_div, 1, cute.rank(offsets_div))
    pair_count = cute.size(packed_words) // 2
    for pair in cutlass.range_constexpr(pair_count):
        row_offsets = cute.filter(offsets_vm[(None, pair * 2)])
        lo_offset = Int32(row_offsets[0])
        scale_value_count = cute.size(row_offsets)
        assert scale_value_count in (2, 8), (
            "Humming pair conversion expects two compact row offsets or "
            "one offset per E2M1 lane"
        )
        hi_index = 4 if scale_value_count == 8 else 1
        hi_offset = Int32(row_offsets[hi_index])
        out0, out1, out2, out3 = convert_mxfp4_pair_preprocessed_signs(
            packed_words[pair * 2],
            packed_words[pair * 2 + 1],
            lo_offset,
            hi_offset,
        )
        fp8_words[pair * 4] = out0
        fp8_words[pair * 4 + 1] = out1
        fp8_words[pair * 4 + 2] = out2
        fp8_words[pair * 4 + 3] = out3


__all__ = [
    "MXFP4_FOLD_BLOCK_BYTES",
    "MXFP4_FOLD_M",
    "MXFP4_GROUP_SIZE",
    "MXFP4_K_TILE",
    "convert_mxfp4_pair_preprocessed_signs",
    "convert_packed_a_kblock",
    "convert_packed_a_kblock_from_offset",
    "make_expanded_offset_view",
    "make_expanded_offset_view_k256",
    "make_offset_smem_layout",
    "make_offset_smem_layout_k256",
    "make_packed_a_ldsm_views",
    "make_packed_a_ldsm_views_k256",
    "make_packed_a_ldsm_views_k256_half",
]
