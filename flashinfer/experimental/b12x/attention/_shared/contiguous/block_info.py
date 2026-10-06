from dataclasses import dataclass
from typing import Optional, Tuple

import cutlass
import cutlass.cute as cute
from cutlass import Int32, const_expr

from b12x.attention._shared.contiguous.seqlen_info import (
    SeqlenInfoQK,
)


@dataclass(frozen=True)
class BlockInfo:
    tile_m: cutlass.Constexpr[int]
    tile_n: cutlass.Constexpr[int]
    is_causal: cutlass.Constexpr[bool]
    is_local: cutlass.Constexpr[bool] = False
    window_size_left: Optional[Int32] = None
    window_size_right: Optional[Int32] = None
    qhead_per_kvhead_packgqa: cutlass.Constexpr[int] = 1
    # ---- block-list walk (video block-sparse attention) -------------------
    # When is_block_sparse is True the K blocks a q tile attends are given by an
    # authoritative CSR list instead of the [n_block_min, n_block_max) range:
    #   mBlockIndices: int32 [total_listed]     logical K block ids
    #   mBlockOffsets: int32 [num_q_tiles + 1]  CSR offsets, one entry per q tile
    # The list is authoritative: causal/local constraints are not re-applied.
    # Dense regions (text, audio) are expressed as contiguous runs in the list.
    is_block_sparse: cutlass.Constexpr[bool] = False
    mBlockIndices: Optional[cute.Tensor] = None
    mBlockOffsets: Optional[cute.Tensor] = None
    # One CSR per packed segment instead of one for the whole packed batch. The list is
    # indexed by the GLOBAL q tile of the segment, offset_q // tile_m + m_block, so
    # segment c reads mBlockOffsets[c_tiles + m_block]. Requires every q offset in
    # cu_seqlens_q to be a multiple of tile_m (tile-aligned segments), which is what a
    # per-head packed layout produces. False keeps the shared-list behaviour.
    per_segment_tiles: cutlass.Constexpr[bool] = False

    @cute.jit
    def n_block_list(self, seqlen_info: SeqlenInfoQK, m_block: Int32) -> Tuple[Int32, Int32]:
        """(offset, count) of this q tile's K blocks in the CSR list.

        With per_segment_tiles the CSR holds one list per packed segment and is indexed by
        the global q tile of the segment: offset_q // tile_m + m_block."""
        tile = m_block
        if const_expr(self.per_segment_tiles):
            tile = seqlen_info.offset_q // self.tile_m + m_block
        begin = self.mBlockOffsets[tile]
        end = self.mBlockOffsets[tile + 1]
        return begin, end - begin

    @cute.jit
    def n_block_from_list(self, offset: Int32, count: Int32, i: Int32) -> Int32:
        """i-th K block id for this tile, in the same reverse order the dense walk uses."""
        return self.mBlockIndices[offset + count - Int32(1) - i]

    @cute.jit
    def get_n_block_min_max(
        self,
        seqlen_info: SeqlenInfoQK,
        m_block: Int32,
    ) -> Tuple[Int32, Int32]:
        n_block_max = cute.ceil_div(seqlen_info.seqlen_k, self.tile_n)
        if const_expr(
            self.is_causal or (self.is_local and self.window_size_right is not None)
        ):
            m_idx_max = (m_block + 1) * self.tile_m
            if const_expr(self.qhead_per_kvhead_packgqa > 1):
                m_idx_max = cute.ceil_div(m_idx_max, self.qhead_per_kvhead_packgqa)
            n_idx = m_idx_max + seqlen_info.seqlen_k - seqlen_info.seqlen_q
            n_idx_right = (
                n_idx if const_expr(self.is_causal) else n_idx + self.window_size_right
            )
            n_block_max = min(n_block_max, cute.ceil_div(n_idx_right, self.tile_n))
        n_block_min = Int32(0)
        if const_expr(self.is_local and self.window_size_left is not None):
            m_idx_min = m_block * self.tile_m
            if const_expr(self.qhead_per_kvhead_packgqa > 1):
                m_idx_min = m_idx_min // self.qhead_per_kvhead_packgqa
            n_idx = m_idx_min + seqlen_info.seqlen_k - seqlen_info.seqlen_q
            n_idx_left = n_idx - self.window_size_left
            n_block_min = cutlass.max(n_idx_left // self.tile_n, 0)
        return n_block_min, n_block_max

    @cute.jit
    def get_n_block_min_causal_local_mask(
        self,
        seqlen_info: SeqlenInfoQK,
        m_block: Int32,
        n_block_min: Int32,
    ) -> Int32:
        m_idx_min = m_block * self.tile_m
        if const_expr(self.qhead_per_kvhead_packgqa > 1):
            m_idx_min = m_idx_min // self.qhead_per_kvhead_packgqa
        n_idx = m_idx_min + seqlen_info.seqlen_k - seqlen_info.seqlen_q
        n_idx_right = (
            n_idx
            if const_expr(not self.is_local or self.window_size_right is None)
            else n_idx + self.window_size_right
        )
        return cutlass.max(n_block_min, n_idx_right // self.tile_n)

    @cute.jit
    def get_n_block_min_before_local_mask(
        self,
        seqlen_info: SeqlenInfoQK,
        m_block: Int32,
        n_block_min: Int32,
    ) -> Int32:
        if const_expr(not self.is_local or self.window_size_left is None):
            return n_block_min
        m_idx_max = (m_block + 1) * self.tile_m
        if const_expr(self.qhead_per_kvhead_packgqa > 1):
            m_idx_max = cute.ceil_div(m_idx_max, self.qhead_per_kvhead_packgqa)
        n_idx = m_idx_max + seqlen_info.seqlen_k - seqlen_info.seqlen_q
        n_idx_left = n_idx - self.window_size_left
        return cutlass.max(n_block_min, cute.ceil_div(n_idx_left, self.tile_n))
