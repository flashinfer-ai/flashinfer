# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Composable band-mask helpers for attention kernels.

The set of KV positions a query row attends to is modeled as a contiguous
band.  With ``offset = seqlen_k - seqlen_q`` (queries right-aligned to the
end of KV, the append/prefill-with-cache convention), row ``q`` sees::

    k in [ lo(q), hi(q) )
    lo(q) = max(0, q + offset - window_left)                if window_left >= 0 else 0
    hi(q) = min(seqlen_k, q + offset + 1)                   if causal
          = min(seqlen_k, q + offset + window_right + 1)    elif window_right >= 0
          = seqlen_k                                        otherwise

``MaskSpec`` describes which bounds exist.  The spec is a compile-time
parameter: absent bounds compile away via ``cutlass.const_expr``, so e.g. a
no-mask kernel contains no masking code.  ``causal`` and ``window_left`` are
independent, matching the FlashInfer convention elsewhere
(``include/flashinfer/attention/variants.cuh``): ``window_left`` bounds
lookback and composes with ``causal``; ``window_right`` bounds lookahead for
non-causal masks (e.g. symmetric windows) and is mutually exclusive with
``causal``.

All helpers are standalone ``@cute.jit`` functions so they can be reused
across kernel variants (prefill, decode).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, is_dataclass

import cutlass
import cutlass.cute as cute
from cutlass.cute.typing import Int32, Float32, Optional


@dataclass(frozen=True)
class MaskSpec:
    """Compile-time description of the attention mask band.

    Attributes:
        causal: right bound follows the diagonal (``k <= q + offset``).
        window_left: max lookback distance; ``-1`` = unbounded.
        window_right: max lookahead distance for non-causal masks;
            ``-1`` = unbounded.  Mutually exclusive with ``causal``.
        check_kv_bounds: some ``seqlen_k`` is not a multiple of the KV tile,
            so the trailing partial tile needs ``k < seqlen_k`` masking even
            when no other bound is active.
    """

    causal: bool = False
    window_left: int = -1
    window_right: int = -1
    check_kv_bounds: bool = False

    def __post_init__(self):
        if self.causal and self.window_right >= 0:
            raise ValueError(
                "window_right is mutually exclusive with causal "
                "(causal already bounds lookahead at 0)"
            )

    @property
    def has_left_bound(self) -> bool:
        return self.window_left >= 0

    @property
    def has_right_bound(self) -> bool:
        return self.causal or self.window_right >= 0

    @property
    def needs_masking(self) -> bool:
        return self.has_left_bound or self.has_right_bound or self.check_kv_bounds


@cute.jit
def get_kv_block_range(
    spec: MaskSpec,
    blk_coord: cute.Coord,
    tile_shape: cute.Shape,
    seqlen_k: Int32,
    seqlen_q: Int32,
) -> tuple[Int32, Int32]:
    """[start, end) KV block range visited for this Q tile (union over rows).

    ``start`` is nonzero only with window_left; ``end`` shrinks with causal
    or window_right.
    """
    qk_offset = seqlen_k - seqlen_q
    start_block = 0
    if cutlass.const_expr(spec.window_left >= 0):
        first_q = blk_coord[0] * tile_shape[0] + qk_offset
        min_kv = cutlass.max(0, first_q - spec.window_left)
        start_block = min_kv // tile_shape[1]
    last_q = (blk_coord[0] + 1) * tile_shape[0] - 1 + qk_offset
    end_elem = seqlen_k
    if cutlass.const_expr(spec.causal):
        end_elem = cutlass.min(seqlen_k, last_q + 1)
    elif cutlass.const_expr(spec.window_right >= 0):
        end_elem = cutlass.min(seqlen_k, last_q + spec.window_right + 1)
    end_block = cute.ceil_div(end_elem, tile_shape[1])
    return start_block, end_block


@cute.jit
def get_trip_count(
    spec: MaskSpec,
    blk_coord: cute.Coord,
    tile_shape: cute.Shape,
    seqlen_k: Int32,
    seqlen_q: Int32,
) -> Int32:
    """Number of KV tile blocks to process for this Q tile."""
    start_block, end_block = get_kv_block_range(
        spec, blk_coord, tile_shape, seqlen_k, seqlen_q
    )
    return end_block - start_block


@cute.jit
def get_trip_segments(
    spec: MaskSpec,
    blk_coord: cute.Coord,
    tile_shape: cute.Shape,
    seqlen_k: Int32,
    seqlen_q: Int32,
) -> tuple[Int32, Int32, Int32, Int32]:
    """KV trip structure: (start_block, masked_left, unmasked, masked_right).

    ``start_block`` is the first KV block visited; the three counts partition
    the visited range and always sum to get_trip_count().  A block is
    unmasked when every row of the Q tile sees every column of the block;
    boundary blocks (window left edge, causal diagonal / right edge,
    seqlen_k tail) need apply_mask.
    """
    start_block, end_block = get_kv_block_range(
        spec, blk_coord, tile_shape, seqlen_k, seqlen_q
    )
    if cutlass.const_expr(not spec.needs_masking):
        return start_block, 0, end_block - start_block, 0
    else:
        qk_offset = seqlen_k - seqlen_q
        first_q = blk_coord[0] * tile_shape[0] + qk_offset
        last_q = first_q + tile_shape[0] - 1
        # Intersection over all rows of the tile: blocks fully inside
        # [lo_max, hi_min) are visible to every row.
        lo_max = 0
        if cutlass.const_expr(spec.window_left >= 0):
            lo_max = cutlass.max(0, last_q - spec.window_left)
        hi_min = seqlen_k
        if cutlass.const_expr(spec.causal):
            hi_min = cutlass.min(seqlen_k, first_q + 1)
        elif cutlass.const_expr(spec.window_right >= 0):
            hi_min = cutlass.min(seqlen_k, first_q + spec.window_right + 1)
        unmasked_start = cute.ceil_div(lo_max, tile_shape[1])
        unmasked_end = hi_min // tile_shape[1]
        unmasked_start = cutlass.min(
            cutlass.max(unmasked_start, start_block), end_block
        )
        unmasked_end = cutlass.min(cutlass.max(unmasked_end, unmasked_start), end_block)
        return (
            start_block,
            unmasked_start - start_block,
            unmasked_end - unmasked_start,
            end_block - unmasked_end,
        )


@cute.jit
def apply_mask(
    spec: MaskSpec,
    acc_qk: cute.Tensor,
    index_qk: cute.Tensor,
    seqlen_k: Int32,
    causal_offset: Int32,
) -> None:
    """Apply the band mask to scores.  ``pos = (q, k)`` are logical coords.

    For specs with a left bound (sliding windows — the regime where every
    visited block is a boundary block), the row's visible band ``[lo, hi)``
    is hoisted out of the element loop: a thread's accumulator fragment
    covers exactly one Q row under the tcgen05 TMEM-load partitioning (the
    softmax's scalar row_max reduction relies on the same fact), so only
    the K coordinate needs comparing per element.  ``hi`` folds the
    ``seqlen_k`` tail bound.  Measured: masked-block cost drops ~2x and the
    sliding-window kernel gains ~25% end-to-end.

    The no-left-bound specs (plain causal, right-window, residual) keep the
    per-element two-coordinate predicate.  Do NOT rewrite them as a compare
    against a hoisted scalar bound: when the DSL can fold the mask predicate
    into a single element-index-vs-invariant-threshold compare (``k >= hi``
    directly, or ``hi <= k or k >= seqlen_k`` after or-folding), MLIR lowers
    each element's conditional -inf write as a select over the ENTIRE packed
    f32x2 accumulator chunk instead of a per-element select (observed: 16
    selp.b64 per element, 8320 vs 512 kernel-wide), which ptxas cannot
    allocate — ~1850 local-memory spills and a kernel-wide ~2.5x slowdown,
    unmasked steps included.  The two-sided band compare above is immune
    because opposite-direction compares don't fold to one threshold
    (1024 selp.b64, no spills).  Seen with cutlass-dsl 4.x / CUDA 13.5.
    """
    if cutlass.const_expr(spec.needs_masking):
        if cutlass.const_expr(spec.window_left >= 0):
            row_q = index_qk[0][0] + causal_offset
            lo = row_q - spec.window_left
            hi = seqlen_k
            if cutlass.const_expr(spec.causal):
                hi = cutlass.min(row_q + 1, seqlen_k)
            elif cutlass.const_expr(spec.window_right >= 0):
                hi = cutlass.min(row_q + spec.window_right + 1, seqlen_k)
            for i in range(cute.size(acc_qk)):
                k = index_qk[i][1]
                if k < lo or k >= hi:
                    acc_qk[i] = -Float32.inf
        elif cutlass.const_expr(spec.causal):
            for i in range(cute.size(acc_qk)):
                pos = index_qk[i]
                if pos[0] + causal_offset < pos[1] or pos[1] >= seqlen_k:
                    acc_qk[i] = -Float32.inf
        elif cutlass.const_expr(spec.window_right >= 0):
            for i in range(cute.size(acc_qk)):
                pos = index_qk[i]
                if (
                    pos[1] - pos[0] - causal_offset > spec.window_right
                    or pos[1] >= seqlen_k
                ):
                    acc_qk[i] = -Float32.inf
        else:
            for i in range(cute.size(acc_qk)):
                if index_qk[i][1] >= seqlen_k:
                    acc_qk[i] = -Float32.inf


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
