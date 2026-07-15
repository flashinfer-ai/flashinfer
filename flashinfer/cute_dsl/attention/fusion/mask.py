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
    """Compile-time description of WHICH band bounds exist.

    Only bound *presence* is compile-time (it drives dead-code
    elimination: a no-window kernel contains no window instructions).
    The bound *values* (``window_left``/``window_right``) are runtime
    kernel arguments, so one compiled kernel serves every window size.
    Causal masking is a right bound with a runtime value of 0, not a
    separate spec kind.  The ``k < seqlen_k`` tail bound is always on:
    the trip-segment math yields zero masked blocks at runtime when
    every ``seqlen_k`` is tile-aligned, so alignment does not change
    which kernel is compiled (and cannot trigger recompiles).

    Attributes:
        has_window_left: a max-lookback bound exists.
        has_window_right: a max-lookahead bound exists (0 = causal).
    """

    has_window_left: bool = False
    has_window_right: bool = False

    @property
    def has_left_bound(self) -> bool:
        return self.has_window_left

    @property
    def has_right_bound(self) -> bool:
        return self.has_window_right

    @property
    def needs_masking(self) -> bool:
        # The seqlen_k tail bound is unconditional; see class docstring.
        return True


@cute.jit
def get_kv_block_range(
    spec: MaskSpec,
    blk_coord: cute.Coord,
    tile_shape: cute.Shape,
    seqlen_k: Int32,
    seqlen_q: Int32,
    window_left: Int32,
    window_right: Int32,
) -> tuple[Int32, Int32]:
    """[start, end) KV block range visited for this Q tile (union over rows).

    ``start`` is nonzero only with window_left; ``end`` shrinks with causal
    or window_right.
    """
    qk_offset = seqlen_k - seqlen_q
    start_block = 0
    if cutlass.const_expr(spec.has_window_left):
        first_q = blk_coord[0] * tile_shape[0] + qk_offset
        min_kv = cutlass.max(0, first_q - window_left)
        start_block = min_kv // tile_shape[1]
    last_q = (blk_coord[0] + 1) * tile_shape[0] - 1 + qk_offset
    end_elem = seqlen_k
    if cutlass.const_expr(spec.has_window_right):
        end_elem = cutlass.min(seqlen_k, last_q + window_right + 1)
    end_block = cute.ceil_div(end_elem, tile_shape[1])
    return start_block, end_block


@cute.jit
def get_trip_count(
    spec: MaskSpec,
    blk_coord: cute.Coord,
    tile_shape: cute.Shape,
    seqlen_k: Int32,
    seqlen_q: Int32,
    window_left: Int32,
    window_right: Int32,
) -> Int32:
    """Number of KV tile blocks to process for this Q tile."""
    start_block, end_block = get_kv_block_range(
        spec, blk_coord, tile_shape, seqlen_k, seqlen_q, window_left, window_right
    )
    return end_block - start_block


@cute.jit
def get_peel_sections(
    spec: MaskSpec,
    blk_coord: cute.Coord,
    tile_shape: cute.Shape,
    seqlen_k: Int32,
    seqlen_q: Int32,
    window_left: Int32,
    window_right: Int32,
) -> tuple[Int32, Int32, Int32, Int32, Int32]:
    """Head/main/tail split of the union KV range across the two 128-row halves.

    Returns ``(kv_start, head, main, tail, stage1_lo_extra)``:

    - ``kv_start`` — first union block (== stage 0's band start).
    - ``head``     — leading blocks visited only by stage 0 (its band starts
      earlier: ``lo`` is nondecreasing in the row).
    - ``main``     — blocks visited by both halves.
    - ``tail``     — trailing blocks visited only by stage 1.
    - ``stage1_lo_extra`` — blocks by which stage 1's band start was clamped
      down to keep the three sections contiguous (nonzero only when stage 1's
      rows are entirely out of the valid Q range and its raw band start lands
      beyond stage 0's band end).  Stage 1 treats these as extra masked-left
      blocks; their rows see an empty band and mask to -inf.

    ``head + main + tail`` equals the full-tile ``get_trip_count`` by
    construction — every role deriving its loop counts from this one helper
    agrees with the loader's union fetch.
    """
    stage_tiler = (tile_shape[0] // 2, tile_shape[1])
    lo0, hi0 = get_kv_block_range(
        spec,
        (blk_coord[0] * 2, blk_coord[1], blk_coord[2]),
        stage_tiler,
        seqlen_k,
        seqlen_q,
        window_left,
        window_right,
    )
    lo1, hi1 = get_kv_block_range(
        spec,
        (blk_coord[0] * 2 + 1, blk_coord[1], blk_coord[2]),
        stage_tiler,
        seqlen_k,
        seqlen_q,
        window_left,
        window_right,
    )
    # lo is nondecreasing and hi is nondecreasing in the row, so lo0 <= lo1
    # and hi0 <= hi1; only lo1 can escape past hi0 (OOB stage-1 rows).
    lo1_clamped = cutlass.min(lo1, hi0)
    head = lo1_clamped - lo0
    main = hi0 - lo1_clamped
    # Borrow one head block into main when the bands don't overlap
    # (window_left == 0, or fully-OOB stage-1 rows): main >= 1 lets every
    # consumer keep a straight-line main prologue/epilogue, and stage 1
    # masks the borrowed block to -inf (it is outside its band).  head >= 1
    # whenever main == 0, since stage 0's band is never empty.
    borrow = cutlass.min(head, cutlass.max(0, 1 - main))
    return (
        lo0,
        head - borrow,
        main + borrow,
        hi1 - hi0,
        (lo1 - lo1_clamped) + borrow,
    )


@cute.jit
def get_stage_peel_segments(
    spec: MaskSpec,
    blk_coord: cute.Coord,
    stage: int,
    tile_shape: cute.Shape,
    seqlen_k: Int32,
    seqlen_q: Int32,
    window_left: Int32,
    window_right: Int32,
) -> tuple[Int32, Int32, Int32, Int32, Int32, Int32]:
    """One 128-row stage's iteration plan under the head/tail peel.

    Returns ``(start_block, masked_left, unmasked, masked_right,
    token_pre_consume, token_post_produce)``: the stage's own band segments
    (stage 0 iterates the union's head+main blocks, stage 1 the main+tail
    blocks, starting from the clamped band start so blocks folded in by
    get_peel_sections mask to -inf), plus the sequence-token dummy counts
    that keep the s0->s1 token ring balanced (stage 1 pre-consumes stage 0's
    ``head`` tokens; stage 0 post-produces ``tail`` tokens).
    """
    union_start, head, _, tail, stage1_extra = get_peel_sections(
        spec, blk_coord, tile_shape, seqlen_k, seqlen_q, window_left, window_right
    )
    _, masked_left, unmasked, masked_right = get_trip_segments(
        spec,
        (blk_coord[0] * 2 + stage, blk_coord[1], blk_coord[2]),
        (tile_shape[0] // 2, tile_shape[1]),
        seqlen_k,
        seqlen_q,
        window_left,
        window_right,
    )
    if cutlass.const_expr(stage == 0):
        return union_start, masked_left, unmasked, masked_right, Int32(0), tail
    else:
        return (
            union_start + head,
            masked_left + stage1_extra,
            unmasked,
            masked_right,
            head,
            Int32(0),
        )


@cute.jit
def get_trip_segments(
    spec: MaskSpec,
    blk_coord: cute.Coord,
    tile_shape: cute.Shape,
    seqlen_k: Int32,
    seqlen_q: Int32,
    window_left: Int32,
    window_right: Int32,
) -> tuple[Int32, Int32, Int32, Int32]:
    """KV trip structure: (start_block, masked_left, unmasked, masked_right).

    ``start_block`` is the first KV block visited; the three counts partition
    the visited range and always sum to get_trip_count().  A block is
    unmasked when every row of the Q tile sees every column of the block;
    boundary blocks (window left edge, causal diagonal / right edge,
    seqlen_k tail) need apply_mask.
    """
    start_block, end_block = get_kv_block_range(
        spec, blk_coord, tile_shape, seqlen_k, seqlen_q, window_left, window_right
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
        if cutlass.const_expr(spec.has_window_left):
            lo_max = cutlass.max(0, last_q - window_left)
        hi_min = seqlen_k
        if cutlass.const_expr(spec.has_window_right):
            hi_min = cutlass.min(seqlen_k, first_q + window_right + 1)
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
    index_qk_static: cute.Tensor,
    window_left: Int32,
    window_right: Int32,
) -> None:
    """Set out-of-band scores to -inf.

    The row's visible band ``[lo, hi)`` (see module docstring) is hoisted
    out of the element loop, so each element only compares its K
    coordinate against row-invariant bounds.  ``index_qk`` is the
    identity partition of the score tile carrying global ``(q, k)``
    coordinates; ``index_qk_static`` is the same partition of the
    unshifted identity tensor, so its coordinates are trace-time
    constants.

    Layout assumptions:
    - One Q row per thread fragment, so the band bounds are uniform
      across a thread's elements (the softmax's scalar row_max reduction
      relies on the same fact).
    - ``index_qk[i] == index_qk_static[i] + domain_offset`` for all i,
      so the tile's dynamic base folds into the hoisted bounds.
    Fragment layouts that break these (e.g. head-packed decode rows, or
    a gathered/sparse KV axis) cannot use this function as-is; they need
    per-element two-coordinate predicates instead of hoisted bounds.

    Codegen rules — both are required to avoid register spills:
    - Masked writes use ``cutlass.select_``, never a traced ``if``.
      Register data is immutable SSA values in the IR, so a traced
      conditional element write is recorded as an ``scf.if`` that
      yields the whole fragment, and the backend does not reliably
      reduce that to a single-element select.  ``cutlass.select_``
      emits one scalar ``arith.select`` per element and is safe for
      any predicate shape.
    - Per-element K reads come from ``index_qk_static``: its
      coordinates are compile-time constants that lower to immediate
      operands, whereas reading the shifted partition materializes
      each global coordinate through per-element address arithmetic
      that inflates register pressure.
    """
    if cutlass.const_expr(spec.needs_masking):
        base_k = index_qk[0][1] - index_qk_static[0][1]
        row_q = index_qk[0][0] + causal_offset
        hi = seqlen_k
        if cutlass.const_expr(spec.has_window_right):
            hi = cutlass.min(row_q + window_right + 1, seqlen_k)
        hi_rel = hi - base_k
        if cutlass.const_expr(spec.has_window_left):
            lo_rel = row_q - window_left - base_k
            for i in range(cute.size(acc_qk)):
                k = index_qk_static[i][1]
                acc_qk[i] = cutlass.select_(
                    (k < lo_rel) | (k >= hi_rel),
                    -Float32.inf,
                    acc_qk[i],
                )
        else:
            # Right-bound-only specs (causal, right window, seqlen_k
            # tail): single-direction compare — safe here because
            # select_ has no scf.if region for MLIR to fold.
            for i in range(cute.size(acc_qk)):
                acc_qk[i] = cutlass.select_(
                    index_qk_static[i][1] >= hi_rel,
                    -Float32.inf,
                    acc_qk[i],
                )


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
