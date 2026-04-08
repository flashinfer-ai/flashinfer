# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Mask types and masking helper functions for attention kernels.

All helpers are standalone @cute.jit functions that take mask_type and
window_left as compile-time parameters, so they can be reused across
different kernel variants (prefill, decode).
"""

import enum

import cutlass
import cutlass.cute as cute
from cutlass.cute.typing import Int32, Float32


class MaskType(enum.Enum):
    NO_MASK = enum.auto()
    RESIDUAL_MASK = enum.auto()
    CAUSAL_MASK = enum.auto()
    SLIDING_WINDOW_MASK = enum.auto()


@cute.jit
def get_trip_count(
    mask_type: MaskType,
    window_left: int,
    blk_coord: cute.Coord,
    tile_shape: cute.Shape,
    seqlen_k: Int32,
    seqlen_q: Int32 = 0,
) -> Int32:
    """Number of KV tile blocks to process for this Q tile."""
    result = 0
    if mask_type == MaskType.NO_MASK or mask_type == MaskType.RESIDUAL_MASK:
        result = cute.ceil_div(seqlen_k, tile_shape[1])
    elif mask_type == MaskType.CAUSAL_MASK:
        max_blocks_k = cute.ceil_div(seqlen_k, tile_shape[1])
        causal_offset = seqlen_k - seqlen_q
        max_blocks_q = cute.ceil_div(
            (blk_coord[0] + 1) * tile_shape[0] + causal_offset, tile_shape[1]
        )
        result = cutlass.min(max_blocks_k, max_blocks_q)
    elif mask_type == MaskType.SLIDING_WINDOW_MASK:
        qk_offset = seqlen_k - seqlen_q
        first_q = blk_coord[0] * tile_shape[0] + qk_offset
        last_q = (blk_coord[0] + 1) * tile_shape[0] - 1 + qk_offset
        min_kv = cutlass.max(0, first_q - window_left)
        max_kv = cutlass.min(seqlen_k - 1, last_q + window_left)
        start_block = min_kv // tile_shape[1]
        end_block = cute.ceil_div(max_kv + 1, tile_shape[1])
        result = end_block - start_block
    return result


@cute.jit
def get_masked_trip_count(
    mask_type: MaskType,
    window_left: int,
    blk_coord: cute.Coord,
    tile_shape: cute.Shape,
    seqlen_k: Int32,
    seqlen_q: Int32 = 0,
) -> Int32:
    """Number of masked (boundary) KV tile blocks."""
    result = 0
    if mask_type == MaskType.NO_MASK:
        result = 0
    elif mask_type == MaskType.RESIDUAL_MASK:
        if seqlen_k % tile_shape[1] != 0:
            result = 1
        else:
            result = 0
    elif mask_type == MaskType.CAUSAL_MASK:
        trip_count = get_trip_count(
            mask_type, window_left, blk_coord, tile_shape, seqlen_k, seqlen_q
        )
        causal_offset = seqlen_k - seqlen_q
        first_boundary = (blk_coord[0] * tile_shape[0] + causal_offset) // tile_shape[1]
        last_boundary = (
            (blk_coord[0] + 1) * tile_shape[0] - 1 + causal_offset
        ) // tile_shape[1]
        result = cutlass.min(
            trip_count,
            last_boundary - first_boundary + 1,
        )
    elif mask_type == MaskType.SLIDING_WINDOW_MASK:
        trip_count = get_trip_count(
            mask_type, window_left, blk_coord, tile_shape, seqlen_k, seqlen_q
        )
        result = trip_count
    return result


@cute.jit
def get_unmasked_trip_count(
    mask_type: MaskType,
    window_left: int,
    blk_coord: cute.Coord,
    tile_shape: cute.Shape,
    seqlen_k: Int32,
    seqlen_q: Int32 = 0,
) -> Int32:
    """Number of fully unmasked KV tile blocks."""
    result = 0
    if mask_type == MaskType.NO_MASK:
        result = get_trip_count(mask_type, window_left, blk_coord, tile_shape, seqlen_k)
    elif mask_type == MaskType.RESIDUAL_MASK:
        if seqlen_k % tile_shape[1] != 0:
            result = (
                get_trip_count(mask_type, window_left, blk_coord, tile_shape, seqlen_k)
                - 1
            )
        else:
            result = get_trip_count(
                mask_type, window_left, blk_coord, tile_shape, seqlen_k
            )
    elif mask_type == MaskType.CAUSAL_MASK:
        result = get_trip_count(
            mask_type, window_left, blk_coord, tile_shape, seqlen_k, seqlen_q
        ) - get_masked_trip_count(
            mask_type, window_left, blk_coord, tile_shape, seqlen_k, seqlen_q
        )
    elif mask_type == MaskType.SLIDING_WINDOW_MASK:
        result = 0
    return result


@cute.jit
def get_kv_start_block_idx(
    mask_type: MaskType,
    window_left: int,
    blk_coord: cute.Coord,
    tile_shape: cute.Shape,
    seqlen_k: Int32,
    seqlen_q: Int32 = 0,
) -> Int32:
    """Starting KV block index (nonzero only for sliding window)."""
    if cutlass.const_expr(mask_type == MaskType.SLIDING_WINDOW_MASK):
        qk_offset = seqlen_k - seqlen_q
        first_q = blk_coord[0] * tile_shape[0] + qk_offset
        min_kv = cutlass.max(0, first_q - window_left)
        return min_kv // tile_shape[1]
    else:
        return 0


@cute.jit
def apply_mask(
    mask_type: MaskType,
    window_left: int,
    acc_qk: cute.Tensor,
    index_qk: cute.Tensor,
    seqlen_k: Int32,
    causal_offset: Int32 = 0,
):
    """Apply attention mask (causal, residual, or sliding window) to scores."""
    if mask_type == MaskType.RESIDUAL_MASK:
        for i in range(cute.size(acc_qk)):
            pos = index_qk[i]
            if pos[1] >= seqlen_k:
                acc_qk[i] = -Float32.inf
    elif mask_type == MaskType.CAUSAL_MASK:
        for i in range(cute.size(acc_qk)):
            pos = index_qk[i]
            if pos[0] + causal_offset < pos[1] or pos[1] >= seqlen_k:
                acc_qk[i] = -Float32.inf
    elif mask_type == MaskType.SLIDING_WINDOW_MASK:
        for i in range(cute.size(acc_qk)):
            pos = index_qk[i]
            if (
                pos[1] - pos[0] - causal_offset > window_left
                or pos[0] + causal_offset - pos[1] > window_left
                or pos[1] >= seqlen_k
            ):
                acc_qk[i] = -Float32.inf
