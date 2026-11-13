"""Two-CTA TensorMap transfers for mixed FP8/FP4 MMA stages."""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass import Uint32, Uint64

from ..ptx.ld_st import (
    EVICT_FIRST,
    EVICT_LAST,
    EVICT_NORMAL,
)
from ..ptx.tma import (
    tile_g2s_2cta_2d,
    tile_g2s_2cta_4d,
)


def use_small_tile_path(shared_experts: int, block_m: int) -> bool:
    """Whether the fused-SE geometry enables valid_m-shrunk routed tiles.

    Gates the prefix activation maps and the K128 descriptor reuse.
    """
    return shared_experts > 0 and block_m // 2 > 16


@cute.jit
def load_mixed_k128_tma(activation_map: cute.Pointer,
                        activation_sf_map: cute.Pointer,
                        weight_map: cute.Pointer,
                        weight_sf_map: cute.Pointer,
                        activation_smem: cute.Pointer,
                        activation_sf_smem: cute.Pointer,
                        weight_smem: cute.Pointer,
                        weight_sf_smem: cute.Pointer,
                        full_bar: cute.Pointer, empty_bar: cute.Pointer,
                        k_block: Uint32, stage_phase: Uint32,
                        block_idx: Uint32, n_block: Uint32,
                        local_expert: Uint32, valid_m: Uint32,
                        activation_rows: Uint32,
                        shape_k: Uint32, is_shared: Uint32,
                        cta_rank: Uint32,
                        loader_warp: cutlass.Constexpr,
                        block_m: cutlass.Constexpr,
                        sf_block_m: cutlass.Constexpr,
                        shared_weight_gran_k: cutlass.Constexpr,
                        has_shared: cutlass.Constexpr,
                        shared_weight_cache_hint: cutlass.Constexpr,
                        routed_weight_linger: cutlass.Constexpr) -> None:
    """Warp 0/1 stage the same K128 task through 2CTA TensorMaps.

    Both CTAs issue their own TMA transfers into local SMEM. The leader's
    full barrier counts two arrivals per CTA and source-byte transaction
    counts for both CTAs. The caller waits it before issuing mixed MMA.
    """
    cute.arch.mbarrier_wait(empty_bar, stage_phase ^ Uint32(1))
    k_idx = k_block * Uint32(128)
    # Every activation tile is read again by the other N-tiles of its
    # expert, so keep the stream resident while the weight stream cycles.
    cache_hint = Uint64(EVICT_LAST)
    if cutlass.const_expr(loader_warp == 0):
        aligned_m = (valid_m + Uint32(15)) & Uint32(~15)
        m_idx = block_idx * Uint32(block_m)
        if cta_rank != Uint32(0):
            m_idx += aligned_m // Uint32(2)
        with cute.arch.elect_one():
            tile_g2s_2cta_2d(activation_smem, activation_map, full_bar,
                             k_idx, m_idx, cache_hint)
            tile_g2s_2cta_2d(activation_sf_smem, activation_sf_map, full_bar,
                             block_idx * Uint32(sf_block_m), k_block,
                             cache_hint)
            if cta_rank == Uint32(0):
                cute.arch.mbarrier_arrive_and_expect_tx(
                    full_bar, 2 * (activation_rows * Uint32(128) + Uint32(sf_block_m * 4)))
            else:
                cute.arch.mbarrier_arrive(full_bar, peer_cta_rank_in_cluster=0)
    else:
        n_idx = n_block * Uint32(128)
        # Every M-block of an expert walks the same weight tiles, so the
        # host lets routed FP4 linger in L2 when an expert spans two or
        # more M-blocks. Shared FP8 keeps the policy chosen by the host.
        routed_cache_hint = (Uint64(EVICT_NORMAL) if routed_weight_linger
                             else Uint64(EVICT_FIRST))
        weight_cache_hint = routed_cache_hint
        gran = Uint32(32)
        if cutlass.const_expr(has_shared):
            weight_cache_hint = (routed_cache_hint if is_shared == Uint32(0)
                                 else Uint64(shared_weight_cache_hint))
            if cutlass.const_expr(shared_weight_gran_k == 128):
                if is_shared != Uint32(0):
                    gran = Uint32(128)
        sf_stages = gran // Uint32(32)
        sf_index = k_block // sf_stages
        sf_packs_per_expert = (shape_k + gran * Uint32(4) - Uint32(1)) // \
            (gran * Uint32(4))
        load_sf = k_block % sf_stages == Uint32(0)
        weight_bytes = Uint32(128 * 128 // 2)
        if cutlass.const_expr(has_shared):
            if is_shared != Uint32(0):
                weight_bytes = Uint32(128 * 128)
        sf_bytes = Uint32(128 * 4) if load_sf else Uint32(0)
        with cute.arch.elect_one():
            # Tile-repacked weight maps: (K-tile, rows, kb, expert), so the
            # op reads one contiguous 8KB (FP4) / 16KB (FP8) global range.
            tile_g2s_2cta_4d(weight_smem, weight_map, full_bar,
                             Uint32(0), n_idx, k_block, local_expert,
                             weight_cache_hint)
            if load_sf:
                tile_g2s_2cta_2d(weight_sf_smem, weight_sf_map, full_bar,
                                 n_idx, local_expert * sf_packs_per_expert + sf_index,
                                 weight_cache_hint)
            if cta_rank == Uint32(0):
                cute.arch.mbarrier_arrive_and_expect_tx(
                    full_bar, (weight_bytes + sf_bytes) * Uint32(2))
            else:
                cute.arch.mbarrier_arrive(full_bar, peer_cta_rank_in_cluster=0)
    cute.arch.sync_warp()
