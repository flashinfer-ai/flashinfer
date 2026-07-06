# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Mask types and masking helper functions for attention kernels.

All helpers are standalone @cute.jit functions that take mask_type and
window_left as compile-time parameters, so they can be reused across
different kernel variants (prefill, decode).
"""

import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass, is_dataclass

import cutlass
import cutlass.cute as cute
from cutlass.cute.typing import Int32, Float32, Optional


# JIT-extendable masking configurations
# Currently only supports GQA/MHA decode kernel
# TODO: support skipping fully masked tiles for prefill integration
class AttentionMask(ABC):
    """ABC interface for attention masking configurations."""

    def __post_init__(self):
        assert is_dataclass(self), f"{type(self)} must be dataclass"

    # Optionally implemented for subclasses
    def can_implement(self, seqlen_q, seqlen_kv, tile_q, tile_kv):
        return

    @cute.jit
    @abstractmethod
    def is_oob_kv(self, idx_q, idx_kv, seqlen_q, seqlen_kv) -> bool:
        """Whether the given KV token at idx_kv should be masked as OOB."""
        ...

    @cute.jit
    @abstractmethod
    def get_range_args(
        self,
        seqlen_q,
        seqlen_kv,
        tile_q,
        tile_kv,
        num_tiles_kv,
        num_iters_kv,
        kv_splits,
        kv_split_idx,
        warpgroups_kv,
        warpgroup_kv_idx,
    ) -> tuple[tuple[Int32, Int32, Int32, bool], ...]:
        """Return args to cutlass.range and a mask codegen flag for each masking phase.

        For phases that generate masking code, range args must emit
        tile_kv coordinates (index into total number of tiles KV across entire KV sequence)
        so that kv token index can be correctly calculated.

        Parameters
        ----------
        seqlen_q
            Query sequence length
        seqlen_kv
            Key/Value sequence length
        tile_q
            Query sequence per threadblock
        tile_kv
            Key/Value sequence per threadblock per outer mainloop iteration
        num_tiles_kv
            KV tiles across entire KV sequence for this Q tile
        num_iters_kv
            KV tiles for this KV split, number of outer mainloop iterations
        kv_splits
            Threadblocks per KV sequence
        kv_split_idx
            Threadblock index into total KV splits
        warpgroups_kv
            Warpgroups per threadblock KV sequence
        warpgroup_kv_idx
            Warpgroup index into total KV warpgroups
        """
        ...


@dataclass(frozen=True)
class NoMask(AttentionMask):
    """Every token attends to every other token.

    OOB logits are TMA masked to 0 and will contribute to softmax denominator.
    Valid for when KV sequence is a multiple of KV sequence tile.
    """

    def can_implement(self, seqlen_q, seqlen_kv, tile_q, tile_kv):
        if seqlen_kv % tile_kv != 0:
            raise ValueError(
                f"NoMask requires seqlen_kv({seqlen_kv}) multiple of tile_kv({tile_kv})"
            )

    @cute.jit
    def is_oob_kv(self, idx_q, idx_kv, seqlen_q, seqlen_kv) -> bool:
        return False

    @cute.jit
    def get_range_args(
        self,
        seqlen_q,
        seqlen_kv,
        tile_q,
        tile_kv,
        num_tiles_kv,
        num_iters_kv,
        kv_splits,
        kv_split_idx,
        warpgroups_kv,
        warpgroup_kv_idx,
    ) -> tuple[tuple[Int32, Int32, Int32, bool], ...]:
        return ((warpgroup_kv_idx, num_iters_kv, warpgroups_kv, False),)


@dataclass(frozen=True)
class DenseMask(AttentionMask):  # aka ResidualMask
    """Every token attends to every other token."""

    @cute.jit
    def is_oob_kv(self, idx_q, idx_kv, seqlen_q, seqlen_kv) -> bool:
        return idx_kv >= seqlen_kv

    @cute.jit
    def get_range_args(
        self,
        seqlen_q,
        seqlen_kv,
        tile_q,
        tile_kv,
        num_tiles_kv,
        num_iters_kv,
        kv_splits,
        kv_split_idx,
        warpgroups_kv,
        warpgroup_kv_idx,
    ) -> tuple[tuple[Int32, Int32, Int32, bool], ...]:
        is_last_split = kv_split_idx == (num_tiles_kv - 1) % kv_splits
        is_last_phase = warpgroup_kv_idx == (num_iters_kv - 1) % warpgroups_kv

        unmasked_start = warpgroup_kv_idx
        unmasked_end = num_iters_kv
        unmasked_step = warpgroups_kv
        masked_start = masked_end = masked_step = 0

        if seqlen_kv % tile_kv != 0 and is_last_split and is_last_phase:
            unmasked_end -= 1
            masked_start = num_tiles_kv - 1
            masked_end = num_tiles_kv
            masked_step = 1

        return (
            (unmasked_start, unmasked_end, unmasked_step, False),
            (masked_start, masked_end, masked_step, True),
        )


@dataclass(frozen=True)
class CausalMask(AttentionMask):
    """Current tokens only attend to past tokens and itself."""

    def can_implement(self, seqlen_q, seqlen_kv, tile_q, tile_kv):
        # Only causal mask up to two tiles
        # Revisit this for prefill integration
        if seqlen_q > tile_kv:
            raise ValueError(
                f"seqlen_q({seqlen_q}) with causal mask can be at most tile_kv({tile_kv})"
            )

    @cute.jit
    def is_oob_kv(self, idx_q, idx_kv, seqlen_q, seqlen_kv) -> bool:
        idx_current = seqlen_kv - seqlen_q + idx_q
        return idx_kv > idx_current

    @cute.jit
    def get_range_args(
        self,
        seqlen_q,
        seqlen_kv,
        tile_q,
        tile_kv,
        num_tiles_kv,
        num_iters_kv,
        kv_splits,
        kv_split_idx,
        warpgroups_kv,
        warpgroup_kv_idx,
    ) -> tuple[tuple[Int32, Int32, Int32, bool], ...]:
        is_last_split = kv_split_idx == (num_tiles_kv - 1) % kv_splits
        is_prev_split = kv_split_idx == (num_tiles_kv - 2) % kv_splits
        is_last_phase = warpgroup_kv_idx == (num_iters_kv - 1) % warpgroups_kv
        is_prev_phase = warpgroup_kv_idx == (num_iters_kv - 2) % warpgroups_kv

        unmasked_start = warpgroup_kv_idx
        unmasked_end = num_iters_kv
        unmasked_step = warpgroups_kv
        masked_start = masked_end = masked_step = 0

        # 2 masked tiles
        if 0 < seqlen_kv % tile_kv < seqlen_q:
            # 1 split masks 2 tiles
            if kv_splits == 1:
                unmasked_end -= 2
                masked_start = num_tiles_kv - 2 + (0 if is_prev_phase else 1)
                masked_end = num_tiles_kv
                masked_step = warpgroups_kv
            # 2 splits mask 1 tile each
            elif (is_last_split or is_prev_split) and is_last_phase:
                unmasked_end -= 1
                masked_start = (
                    (num_tiles_kv - 1) if is_last_split else (num_tiles_kv - 2)
                )
                masked_end = num_tiles_kv
                masked_step = 2
        # 1 masked tile
        elif seqlen_q > 1 or seqlen_kv % tile_kv != 0:
            # 1 split masks 1 tile
            if is_last_split and is_last_phase:
                unmasked_end -= 1
                masked_start = num_tiles_kv - 1
                masked_end = num_tiles_kv
                masked_step = 1

        return (
            (unmasked_start, unmasked_end, unmasked_step, False),
            (masked_start, masked_end, masked_step, True),
        )


@dataclass(frozen=True)
class SlidingWindowMask(AttentionMask):
    """Current tokens attend to a specified window of past and future tokens.

    Parameters
    ----------
    window_left
        Number of past tokens to attend to.
        None will include all tokens left of causal diagonal.
    window_right
        Number of future tokens to attend to.
        None will include all tokens right of causal diagonal.
    """

    # use int for compile-time config and Int32 for runtime config
    window_left: Optional[int | Int32]
    window_right: Optional[int | Int32] = 0

    @cute.jit
    def is_oob_kv(self, idx_q, idx_kv, seqlen_q, seqlen_kv) -> bool:
        is_oob = idx_kv >= seqlen_kv
        idx_current = seqlen_kv - seqlen_q + idx_q
        if cutlass.const_expr(self.window_left is not None):
            is_oob |= idx_kv < idx_current - self.window_left
        if cutlass.const_expr(self.window_right is not None):
            is_oob |= idx_kv > idx_current + self.window_right
        return is_oob

    @cute.jit
    def get_range_args(
        self,
        seqlen_q,
        seqlen_kv,
        tile_q,
        tile_kv,
        num_tiles_kv,
        num_iters_kv,
        kv_splits,
        kv_split_idx,
        warpgroups_kv,
        warpgroup_kv_idx,
    ) -> tuple[tuple[Int32, Int32, Int32, bool], ...]:
        # Assume KV seqlen roughly matches window size and that
        # window is small enough to not need an unmasked phase.
        # Can revisit this for perf for very large windows.
        masked_start = kv_split_idx + warpgroup_kv_idx * kv_splits
        masked_stop = num_tiles_kv
        masked_step = warpgroups_kv * kv_splits
        return ((masked_start, masked_stop, masked_step, True),)


#
# Legacy enum-based masking configuration
# only supports GQA/MHA prefill kernel
#
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
