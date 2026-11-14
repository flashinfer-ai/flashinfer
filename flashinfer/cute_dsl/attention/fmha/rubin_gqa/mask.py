# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABC, abstractmethod
from dataclasses import dataclass, is_dataclass

import cutlass
import cutlass.cute as cute
from cutlass.cute.typing import Int32, Optional


# JIT-extendable masking configurations
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
    def get_tiles_kv(
        self,
        seqlen_qo,
        seqlen_kv,
        tile_kv,
        tile_qo,
        coord_qo,
    ) -> tuple[Int32, Int32]:  # coord_kv, tiles_kv
        """Returns KV tile offset and number of KV tiles to process across all splits"""
        ...

    @cute.jit
    @abstractmethod
    def get_range_args(
        self,
        seqlen_q,
        seqlen_kv,
        tile_q,
        tile_kv,
        coord_qo,
        coord_kv,
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
        coord_qo
            Query tile coordinate
        coord_kv
            Global KV tile offset returned by get_tiles_kv; zero when no tiles are skipped
        num_tiles_kv
            Number of KV tiles selected for this Q tile, starting at coord_kv
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
            raise ValueError(f"NoMask requires seqlen_kv({seqlen_kv}) multiple of tile_kv({tile_kv})")

    @cute.jit
    def is_oob_kv(self, idx_q, idx_kv, seqlen_q, seqlen_kv) -> bool:
        return False

    @cute.jit
    def get_tiles_kv(
        self,
        seqlen_qo,
        seqlen_kv,
        tile_kv,
        tile_qo,
        coord_qo,
    ) -> tuple[Int32, Int32]:  # coord_kv, tiles_kv
        tiles_kv = cute.ceil_div(seqlen_kv, tile_kv)
        coord_kv = 0

        return coord_kv, tiles_kv

    @cute.jit
    def get_range_args(
        self,
        seqlen_q,
        seqlen_kv,
        tile_q,
        tile_kv,
        coord_qo,
        coord_kv,
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
    def get_tiles_kv(
        self,
        seqlen_qo,
        seqlen_kv,
        tile_kv,
        tile_qo,
        coord_qo,
    ) -> tuple[Int32, Int32]:  # coord_kv, tiles_kv
        tiles_kv = cute.ceil_div(seqlen_kv, tile_kv)
        coord_kv = 0

        return coord_kv, tiles_kv

    @cute.jit
    def get_range_args(
        self,
        seqlen_q,
        seqlen_kv,
        tile_q,
        tile_kv,
        coord_qo,
        coord_kv,
        num_tiles_kv,
        num_iters_kv,
        kv_splits,
        kv_split_idx,
        warpgroups_kv,
        warpgroup_kv_idx,
    ) -> tuple[tuple[Int32, Int32, Int32, bool], ...]:
        # Producers execute at least two iterations even for a single KV tile.
        # Only complete, in-bounds tiles may skip masking; residual and padded
        # iterations must use their actual global KV coordinates.
        unmasked_tiles = min(seqlen_kv // tile_kv, num_tiles_kv)
        unmasked_start = warpgroup_kv_idx
        unmasked_end = min(
            num_iters_kv,
            max(cute.ceil_div(unmasked_tiles - kv_split_idx, kv_splits), 0),
        )
        unmasked_step = warpgroups_kv
        masked_iter = warpgroup_kv_idx + cute.ceil_div(
            max(unmasked_end - warpgroup_kv_idx, 0), warpgroups_kv
        ) * warpgroups_kv
        masked_start = kv_split_idx + masked_iter * kv_splits
        masked_end = kv_split_idx + num_iters_kv * kv_splits
        masked_step = warpgroups_kv * kv_splits

        return (
            (unmasked_start, unmasked_end, unmasked_step, False),
            (masked_start, masked_end, masked_step, True),
        )


@dataclass(frozen=True)
class CausalMask(AttentionMask):
    """Current tokens only attend to past tokens and itself."""
    def can_implement(self, seqlen_q, seqlen_kv, tile_q, tile_kv) -> bool:
        if seqlen_q > seqlen_kv:
            raise ValueError(
                f"seqlen_q({seqlen_q}) with causal mask can be at most seqlen_kv({seqlen_kv})"
            )

    @cute.jit
    def is_oob_kv(self, idx_q, idx_kv, seqlen_q, seqlen_kv) -> bool:
        idx_current = seqlen_kv - seqlen_q + idx_q
        return idx_kv > idx_current

    @cute.jit
    def get_tiles_kv(
        self,
        seqlen_qo,
        seqlen_kv,
        tile_kv,
        tile_qo,
        coord_qo,
    ) -> tuple[Int32, Int32]:  # coord_kv, tiles_kv
        # Padded Q rows must not extend the selected interval past KV.
        last_idx_q = min((coord_qo + 1) * tile_qo, seqlen_qo) - 1
        unmasked_kv = max(seqlen_kv - seqlen_qo + last_idx_q + 1, 0)
        tiles_kv = cute.ceil_div(unmasked_kv, tile_kv)
        coord_kv = 0

        return coord_kv, tiles_kv

    @cute.jit
    def get_range_args(
        self,
        seqlen_q,
        seqlen_kv,
        tile_q,
        tile_kv,
        coord_qo,
        coord_kv,
        num_tiles_kv,
        num_iters_kv,
        kv_splits,
        kv_split_idx,
        warpgroups_kv,
        warpgroup_kv_idx,
    ) -> tuple[tuple[Int32, Int32, Int32, bool], ...]:
        # A KV tile is fully visible to every row only if its last token
        # is at or before the causal diagonal of the first Q row.
        first_idx_q = coord_qo * tile_q
        unmasked_tokens = seqlen_kv - seqlen_q + first_idx_q + 1
        unmasked_tiles = min(unmasked_tokens // tile_kv, num_tiles_kv)

        # Split-local iteration i processes global KV tile kv_split_idx +
        # i * kv_splits. Partition that stream at the first masked tile.
        unmasked_start = warpgroup_kv_idx
        unmasked_end = min(
            num_iters_kv,
            max(cute.ceil_div(unmasked_tiles - kv_split_idx, kv_splits), 0),
        )
        unmasked_step = warpgroups_kv

        # Round up to this warpgroup's next iteration and convert back to
        # global KV coordinates for is_oob_kv. Include any padded iterations
        # in num_iters_kv so both phases consume the full producer stream.
        masked_iter = warpgroup_kv_idx + cute.ceil_div(
            max(unmasked_end - warpgroup_kv_idx, 0), warpgroups_kv
        ) * warpgroups_kv
        masked_start = kv_split_idx + masked_iter * kv_splits
        masked_end = kv_split_idx + num_iters_kv * kv_splits
        masked_step = warpgroups_kv * kv_splits

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
    def get_tiles_kv(
        self,
        seqlen_qo,
        seqlen_kv,
        tile_kv,
        tile_qo,
        coord_qo,
    ) -> tuple[Int32, Int32]:  # coord_kv, tiles_kv
        first_idx_q = coord_qo * tile_qo
        last_idx_q = min(first_idx_q + tile_qo, seqlen_qo) - 1
        diagonal_offset = seqlen_kv - seqlen_qo

        # Select the union of the windows for all valid rows in this Q tile.
        first_idx_kv = 0
        end_idx_kv = seqlen_kv
        if cutlass.const_expr(self.window_left is not None):
            first_idx_kv = min(
                max(diagonal_offset + first_idx_q - self.window_left, 0),
                seqlen_kv,
            )
        if cutlass.const_expr(self.window_right is not None):
            end_idx_kv = max(
                min(diagonal_offset + last_idx_q + self.window_right + 1, seqlen_kv),
                0,
            )

        # Keep the boundary tiles: individual rows still need element masking.
        coord_kv = first_idx_kv // tile_kv
        tiles_kv = max(cute.ceil_div(end_idx_kv, tile_kv) - coord_kv, 0)
        return coord_kv, tiles_kv

    @cute.jit
    def get_range_args(
        self,
        seqlen_q,
        seqlen_kv,
        tile_q,
        tile_kv,
        coord_qo,
        coord_kv,
        num_tiles_kv,
        num_iters_kv,
        kv_splits,
        kv_split_idx,
        warpgroups_kv,
        warpgroup_kv_idx,
    ) -> tuple[tuple[Int32, Int32, Int32, bool], ...]:
        # Mask all selected tiles, including padded prefetch iterations.
        # Split/warpgroup assignment is relative to the selected KV interval;
        # emit global coordinates so is_oob_kv sees the original token indices.
        masked_start = coord_kv + kv_split_idx + warpgroup_kv_idx * kv_splits
        masked_stop = coord_kv + kv_split_idx + num_iters_kv * kv_splits
        masked_step = warpgroups_kv * kv_splits
        return ((masked_start, masked_stop, masked_step, True),)
