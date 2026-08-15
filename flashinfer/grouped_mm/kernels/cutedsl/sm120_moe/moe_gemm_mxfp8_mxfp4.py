# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Self-written CuteDSL act-MXFP8 x weight-MXFP4 token-packed grouped MoE GEMM for SM120a."""
import torch

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils
import cutlass.cute.nvgpu.warp.mma as warp_mma
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import from_dlpack

from .core.sm120_gemm_builder import (
    Sm120GemmBuilder, MmaConfig, LoadABConfig, StoreMethod, STORE_CONFIGS)
from .core.sm120_blockscaled_layout import Sm120SfConfigMxfp8Mxfp4
from .core._common import SF_M_ALIGN
from .core._common import ceil_div
from .core import scheduler, store_output


# The arm's SF granularity contract; the op layer reads it here rather than keeping a copy.
GRANK_A, GRANK_B = 128, 32
ATOM_MNK = (16, 8, 32)


def is_swapab(tile):
    return ATOM_MNK[0] > tile[0]  # bm == 8


def make_cfg(tile, ab_stage, store=StoreMethod.R2G_WG, grank_a=GRANK_A):
    """``tile`` in is problem space; everything the builder sees below is kernel space, never mixed."""
    e4m3, fp4, f32, bf16, i8, ue8m0 = (cutlass.Float8E4M3FN, cutlass.Float4E2M1FN, cutlass.Float32,
                                       cutlass.BFloat16, cutlass.Int8, cutlass.Float8E8M0FNU)
    num_math_warps = 8
    bm, bn, bk = tile
    swap = is_swapab(tile)
    if swap:
        bm, bn = bn, bm
        a_dtype, b_dtype = fp4, e4m3
        unpack = dict(a_smem_dtype=i8, a_tma_internal=i8, a_unpack_bits=4)
        grank_a, grank_b = GRANK_B, grank_a
        store = StoreMethod.DIRECT_STG
    else:
        # DIRECT_STG is the swapped epilogue: store_swap writes D^T and indexes with n_block
        assert store is not StoreMethod.DIRECT_STG, (
            f"{store} needs the swapped geometry; this tile is not swap-AB")
        a_dtype, b_dtype = e4m3, fp4
        unpack = dict(b_smem_dtype=i8, b_tma_internal=i8, b_unpack_bits=4)
        grank_b = GRANK_B
    tile = (bm, bn, bk)
    # the whole warpgroup drains sC itself, which is what lets sC alias sA instead of costing smem
    union = store is StoreMethod.R2G_WG
    return Sm120GemmBuilder(
        MmaConfig(warp_mma.MmaMXF8F6F4Op(a_dtype, b_dtype, f32, ue8m0), tile[:2], num_math_warps,
                  swap_ab=swap),
        LoadABConfig(tile, ab_stage, a_dtype, b_dtype, **unpack),
        Sm120SfConfigMxfp8Mxfp4(grank_a, grank_b),
        STORE_CONFIGS[store](bf16, num_math_warps * 32),
        ab_stage, tile, epi_bar_id=3, union_smem=union)


@cute.jit
def load_ab(tma_atom_a, tma_atom_b, tma_atom_sfb, tma_tensor_a, tma_tensor_b, tma_tensor_sfb,
            sA, sB, sSFB, tile_mnk, tile, ab_full, ab_empty, ab_bytes, k_tile_count,
            ab_stages, ab_phase):
    """One tile's A/B/SFB G2S: partition A at the expert's row offset, then run the k-loop."""
    i32 = cutlass.Int32
    cluster = cute.make_layout((1, 1, 1))
    multicast = cute.make_layout(cute.slice_(cluster, (0, None, 0)).shape)
    mA = cute.domain_offset((tile.m_offset, 0), tma_tensor_a)
    gA_mkl = cute.local_tile(mA, cute.slice_(tile_mnk, (None, 0, None)), (None, None))
    tAsA, tAgA = cpasync.tma_partition(tma_atom_a, i32(0), multicast, cute.group_modes(sA, 0, 2),
                                       cute.group_modes(gA_mkl, 0, 2))
    gB_nkl = cute.local_tile(tma_tensor_b, cute.slice_(tile_mnk, (0, None, None)), (None, None, None))
    tBsB, tBgB = cpasync.tma_partition(tma_atom_b, i32(0), multicast, cute.group_modes(sB, 0, 2),
                                       cute.group_modes(gB_nkl, 0, 2))
    gSFB_nl = cute.local_tile(tma_tensor_sfb, (tile_mnk[1], 1), (None, None, None))
    tBsSFB, tBgSFB = cpasync.tma_partition(tma_atom_sfb, i32(0), multicast,
                                           cute.group_modes(sSFB, 0, 2), cute.group_modes(gSFB_nl, 0, 2))
    for kt_base in cutlass.range(0, k_tile_count, ab_stages):
        for s in cutlass.range_constexpr(ab_stages):
            kt = kt_base + s
            cute.arch.mbarrier_wait(ab_empty + s, ab_phase)
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(ab_full + s, ab_bytes)
            cute.copy(tma_atom_a, tAgA[(None, tile.m_block, kt)], tAsA[(None, s)], tma_bar_ptr=ab_full + s)
            cute.copy(tma_atom_b, tBgB[(None, tile.n_block, kt, tile.group)], tBsB[(None, s)], tma_bar_ptr=ab_full + s)
            cute.copy(tma_atom_sfb, tBgSFB[(None, tile.n_block, kt, tile.group)], tBsSFB[(None, s)], tma_bar_ptr=ab_full + s)
        ab_phase ^= 1
    return ab_phase


@cute.jit
def load_ab_swap(tma_atom_a, tma_atom_b, tma_atom_sfa, tma_tensor_a, tma_tensor_b, tma_tensor_sfa,
                 sA, sB, sSFA, tile_mnk, tile, ab_full, ab_empty, ab_bytes, k_tile_count,
                 ab_stages, ab_phase):
    """A/B/SFA G2S with the operands exchanged; the swap makes SFA the side that rides this ring."""
    i32 = cutlass.Int32
    cluster = cute.make_layout((1, 1, 1))
    multicast = cute.make_layout(cute.slice_(cluster, (0, None, 0)).shape)
    gA_mkl = cute.local_tile(tma_tensor_a, cute.slice_(tile_mnk, (None, 0, None)), (None, None, None))
    tAsA, tAgA = cpasync.tma_partition(tma_atom_a, i32(0), multicast, cute.group_modes(sA, 0, 2),
                                       cute.group_modes(gA_mkl, 0, 2))
    mB = cute.domain_offset((tile.m_offset, 0), tma_tensor_b)
    gB_nkl = cute.local_tile(mB, cute.slice_(tile_mnk, (0, None, None)), (None, None))
    tBsB, tBgB = cpasync.tma_partition(tma_atom_b, i32(0), multicast, cute.group_modes(sB, 0, 2),
                                       cute.group_modes(gB_nkl, 0, 2))
    gSFA_ml = cute.local_tile(tma_tensor_sfa, (tile_mnk[0], 1), (None, None, None))
    tAsSFA, tAgSFA = cpasync.tma_partition(tma_atom_sfa, i32(0), multicast,
                                           cute.group_modes(sSFA, 0, 2), cute.group_modes(gSFA_ml, 0, 2))
    for kt_base in cutlass.range(0, k_tile_count, ab_stages):
        for s in cutlass.range_constexpr(ab_stages):
            kt = kt_base + s
            cute.arch.mbarrier_wait(ab_empty + s, ab_phase)
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(ab_full + s, ab_bytes)
            cute.copy(tma_atom_a, tAgA[(None, tile.n_block, kt, tile.group)], tAsA[(None, s)], tma_bar_ptr=ab_full + s)
            cute.copy(tma_atom_b, tBgB[(None, tile.m_block, kt)], tBsB[(None, s)], tma_bar_ptr=ab_full + s)
            cute.copy(tma_atom_sfa, tAgSFA[(None, tile.n_block, kt, tile.group)], tAsSFA[(None, s)], tma_bar_ptr=ab_full + s)
        ab_phase ^= 1
    return ab_phase


@cute.jit
def load_sf(tma_atom_sfa, tma_tensor_sfa, sSFA, tile_m, tile, sf_full, sf_empty, sfa_bytes,
            m_align, num_sf_cycles, sf_stages, sf_phase):
    """One tile's SFA G2S, issued in whole SF cycles."""
    i32 = cutlass.Int32
    cluster = cute.make_layout((1, 1, 1))
    multicast = cute.make_layout(cute.slice_(cluster, (0, None, 0)).shape)
    sf_m_off = (tile.m_offset + tile.group * i32(m_align - 1)) & i32(-m_align)
    mSFA = cute.domain_offset((sf_m_off, 0, 0), tma_tensor_sfa)
    gSFA_ml = cute.local_tile(mSFA, (tile_m, 1), (None, None, None))
    tAsSFA, tAgSFA = cpasync.tma_partition(tma_atom_sfa, i32(0), multicast,
                                           cute.group_modes(sSFA, 0, 2), cute.group_modes(gSFA_ml, 0, 2))
    for sf_cycle in cutlass.range(0, num_sf_cycles):
        for s in cutlass.range_constexpr(sf_stages):
            cute.arch.mbarrier_wait(sf_empty + s, sf_phase)
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(sf_full + s, sfa_bytes)
            cute.copy(tma_atom_sfa, tAgSFA[(None, tile.m_block, sf_cycle * sf_stages + s, 0)],
                      tAsSFA[(None, s)], tma_bar_ptr=sf_full + s)
        sf_phase ^= 1
    return sf_phase


@cute.jit
def load_sf_swap(tma_atom_sfb, tma_tensor_sfb, sSFB, tile_n, tile, sf_full, sf_empty, sfb_bytes,
                 m_align, num_sf_cycles, sf_stages, sf_phase):
    """The token-side SF, which the exchange makes the coarse one, so it is SFB that gets this warp."""
    i32 = cutlass.Int32
    cluster = cute.make_layout((1, 1, 1))
    multicast = cute.make_layout(cute.slice_(cluster, (0, None, 0)).shape)
    sf_m_off = (tile.m_offset + tile.group * i32(m_align - 1)) & i32(-m_align)
    mSFB = cute.domain_offset((sf_m_off, 0, 0), tma_tensor_sfb)
    gSFB_nl = cute.local_tile(mSFB, (tile_n, 1), (None, None, None))
    tBsSFB, tBgSFB = cpasync.tma_partition(tma_atom_sfb, i32(0), multicast,
                                           cute.group_modes(sSFB, 0, 2), cute.group_modes(gSFB_nl, 0, 2))
    for sf_cycle in cutlass.range(0, num_sf_cycles):
        for s in cutlass.range_constexpr(sf_stages):
            cute.arch.mbarrier_wait(sf_empty + s, sf_phase)
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(sf_full + s, sfb_bytes)
            cute.copy(tma_atom_sfb, tBgSFB[(None, tile.m_block, sf_cycle * sf_stages + s, 0)],
                      tBsSFB[(None, s)], tma_bar_ptr=sf_full + s)
        sf_phase ^= 1
    return sf_phase


@cute.jit
def mma(tiledmma, mma_cfg, sf_cfg, sA, sB, sSFA, sSFB, tile_mn, bk, acc_dtype,
        a_dtype, b_smem_dtype, a_is_m_major, unpack_bits, tidx,
        ab_full, ab_empty, sf_full, sf_empty,
        num_sf_cycles, sf_stages, kt_per_pack_a, ab_stages, ab_phase, sf_phase,
        union_smem, epi_bar_id, mma_threads):
    """One tile's math: the SF cycle outside, the A/B ring masked off the position inside it."""
    bm, bn = tile_mn
    thr = tiledmma.get_slice(tidx)
    tCrA = tiledmma.make_fragment_A(thr.partition_A(sA)[None, None, None, 0])
    tCrB = tiledmma.make_fragment_B(thr.partition_B(sB)[None, None, None, 0])
    acc = cute.make_rmem_tensor(tiledmma.partition_shape_C((bm, bn)), acc_dtype)
    s2r_a = mma_cfg.make_s2r_a(tiledmma, a_dtype, a_is_m_major)
    s2r_b = mma_cfg.make_s2r_b(tiledmma, b_smem_dtype, False, unpack_bits)
    thr_a, thr_b = s2r_a.get_slice(tidx), s2r_b.get_slice(tidx)
    tCrA_v, tCrB_v = thr_a.retile(tCrA), thr_b.retile(tCrB)
    s2r_sfa = sf_cfg.make_s2r_sf(sf_cfg.get_layoutSFA_TV(tiledmma), (cute.size(tiledmma.permutation_mnk[0]), 1))
    s2r_sfb = sf_cfg.make_s2r_sf(sf_cfg.get_layoutSFB_TV(tiledmma), (cute.size(tiledmma.permutation_mnk[1]), 1))
    thr_sfa, thr_sfb = s2r_sfa.get_slice(tidx), s2r_sfb.get_slice(tidx)
    tCrSFA = sf_cfg.partition_fragment_SFA(cute.slice_(sSFA, (None, None, 0)), thr, tidx)
    tCrSFA_frg = sf_cfg.make_sfa_ue8m0_view(tCrSFA, bk)
    tCrSFB = sf_cfg.partition_fragment_SFB(cute.slice_(sSFB, (None, None, 0)), thr, tidx)
    tCrSFB_frg = sf_cfg.make_sfb_ue8m0_view(tCrSFB, bk)
    k_blocks = cute.size(tCrA_v, mode=[2])

    acc.fill(0.0)
    for sf_cycle in cutlass.range(0, num_sf_cycles - 1):
        for sf_stage in cutlass.range_constexpr(sf_stages):
            cute.arch.mbarrier_wait(sf_full + sf_stage, sf_phase)
            cute.copy(s2r_sfa, thr_sfa.partition_S(cute.slice_(sSFA, (None, None, sf_stage))),
                      thr_sfa.retile(tCrSFA))
            for k_in_sf in cutlass.range_constexpr(kt_per_pack_a):
                s = (sf_stage * kt_per_pack_a + k_in_sf) & (ab_stages - 1)
                cute.arch.mbarrier_wait(ab_full + s, ab_phase)
                cute.copy(s2r_sfb, thr_sfb.partition_S(cute.slice_(sSFB, (None, None, s))),
                          thr_sfb.retile(tCrSFB))
                tAsA_s = thr_a.partition_S(sA)[None, None, None, s]
                tBsB_s = thr_b.partition_S(sB)[None, None, None, s]
                for k in cutlass.range_constexpr(0, k_blocks):
                    cute.copy(s2r_a, tAsA_s[None, None, k], tCrA_v[None, None, k])
                    cute.copy(s2r_b, tBsB_s[None, None, k], tCrB_v[None, None, k])
                # e2m1 lands in the low nibble; MXF8F6F4 wants it mid-byte (mma_traits_sm120.hpp:210-216)
                tCrB_i8 = cute.recast_tensor(tCrB, cutlass.Int8)
                tCrB_i8.store(tCrB_i8.load() * cutlass.Int8(4))
                for k_block in cutlass.range_constexpr(0, k_blocks):
                    cute.gemm(tiledmma, acc,
                              [tCrA[None, None, k_block], tCrSFA_frg[None, None, k_block, k_in_sf]],
                              [tCrB[None, None, k_block], tCrSFB_frg[None, None, k_block, 0]], acc)
                cute.arch.mbarrier_arrive(ab_empty + s)
                if cutlass.const_expr(s == ab_stages - 1):
                    ab_phase ^= 1
            cute.arch.mbarrier_arrive(sf_empty + sf_stage)
        sf_phase ^= 1
    # unrolled so is_last_k_tile is trace-time constant (upstream kernel_impl.cuh:313-341)
    for sf_stage in cutlass.range_constexpr(sf_stages):
        cute.arch.mbarrier_wait(sf_full + sf_stage, sf_phase)
        cute.copy(s2r_sfa, thr_sfa.partition_S(cute.slice_(sSFA, (None, None, sf_stage))),
                  thr_sfa.retile(tCrSFA))
        for k_in_sf in cutlass.range_constexpr(kt_per_pack_a):
            s = (sf_stage * kt_per_pack_a + k_in_sf) & (ab_stages - 1)
            cute.arch.mbarrier_wait(ab_full + s, ab_phase)
            cute.copy(s2r_sfb, thr_sfb.partition_S(cute.slice_(sSFB, (None, None, s))),
                      thr_sfb.retile(tCrSFB))
            tAsA_s = thr_a.partition_S(sA)[None, None, None, s]
            tBsB_s = thr_b.partition_S(sB)[None, None, None, s]
            for k in cutlass.range_constexpr(0, k_blocks):
                cute.copy(s2r_a, tAsA_s[None, None, k], tCrA_v[None, None, k])
                cute.copy(s2r_b, tBsB_s[None, None, k], tCrB_v[None, None, k])
            # e2m1 lands in the low nibble; MXF8F6F4 wants it mid-byte (mma_traits_sm120.hpp:210-216)
            tCrB_i8 = cute.recast_tensor(tCrB, cutlass.Int8)
            tCrB_i8.store(tCrB_i8.load() * cutlass.Int8(4))
            if cutlass.const_expr(union_smem and sf_stage == sf_stages - 1
                                  and k_in_sf == kt_per_pack_a - 1):
                cute.arch.barrier(barrier_id=epi_bar_id, number_of_threads=mma_threads)
            for k_block in cutlass.range_constexpr(0, k_blocks):
                cute.gemm(tiledmma, acc,
                          [tCrA[None, None, k_block], tCrSFA_frg[None, None, k_block, k_in_sf]],
                          [tCrB[None, None, k_block], tCrSFB_frg[None, None, k_block, 0]], acc)
            cute.arch.mbarrier_arrive(ab_empty + s)
            if cutlass.const_expr(s == ab_stages - 1):
                ab_phase ^= 1
        cute.arch.mbarrier_arrive(sf_empty + sf_stage)
    sf_phase ^= 1
    return acc, ab_phase, sf_phase


@cute.jit
def mma_swap(tiledmma, mma_cfg, sf_cfg, sA, sB, sSFA, sSFB, tile_mn, bk, acc_dtype,
             a_smem_dtype, b_dtype, unpack_bits, tidx,
             ab_full, ab_empty, sf_full, sf_empty,
             num_sf_cycles, sf_stages, kt_per_pack_b, ab_stages, ab_phase, sf_phase,
        union_smem, epi_bar_id, mma_threads):
    """Mirror of mma: the exchange makes SFB the coarse side, so SFB owns the outer cycle."""
    bm, bn = tile_mn
    thr = tiledmma.get_slice(tidx)
    tCrA = tiledmma.make_fragment_A(thr.partition_A(sA)[None, None, None, 0])
    tCrB = tiledmma.make_fragment_B(thr.partition_B(sB)[None, None, None, 0])
    acc = cute.make_rmem_tensor(tiledmma.partition_shape_C((bm, bn)), acc_dtype)
    s2r_a = mma_cfg.make_s2r_a(tiledmma, a_smem_dtype, False, unpack_bits)
    s2r_b = mma_cfg.make_s2r_b(tiledmma, b_dtype, False)
    thr_a, thr_b = s2r_a.get_slice(tidx), s2r_b.get_slice(tidx)
    tCrA_v, tCrB_v = thr_a.retile(tCrA), thr_b.retile(tCrB)
    s2r_sfa = sf_cfg.make_s2r_sf(sf_cfg.get_layoutSFA_TV(tiledmma), (cute.size(tiledmma.permutation_mnk[0]), 1))
    s2r_sfb = sf_cfg.make_s2r_sf(sf_cfg.get_layoutSFB_TV(tiledmma), (cute.size(tiledmma.permutation_mnk[1]), 1))
    thr_sfa, thr_sfb = s2r_sfa.get_slice(tidx), s2r_sfb.get_slice(tidx)
    tCrSFA = sf_cfg.partition_fragment_SFA(cute.slice_(sSFA, (None, None, 0)), thr, tidx)
    tCrSFA_frg = sf_cfg.make_sfa_ue8m0_view(tCrSFA, bk)
    tCrSFB = sf_cfg.partition_fragment_SFB(cute.slice_(sSFB, (None, None, 0)), thr, tidx)
    tCrSFB_frg = sf_cfg.make_sfb_ue8m0_view(tCrSFB, bk)
    k_blocks = cute.size(tCrA_v, mode=[2])

    acc.fill(0.0)
    for sf_cycle in cutlass.range(0, num_sf_cycles - 1):
        for sf_stage in cutlass.range_constexpr(sf_stages):
            cute.arch.mbarrier_wait(sf_full + sf_stage, sf_phase)
            cute.copy(s2r_sfb, thr_sfb.partition_S(cute.slice_(sSFB, (None, None, sf_stage))),
                      thr_sfb.retile(tCrSFB))
            for k_in_sf in cutlass.range_constexpr(kt_per_pack_b):
                s = (sf_stage * kt_per_pack_b + k_in_sf) & (ab_stages - 1)
                cute.arch.mbarrier_wait(ab_full + s, ab_phase)
                cute.copy(s2r_sfa, thr_sfa.partition_S(cute.slice_(sSFA, (None, None, s))),
                          thr_sfa.retile(tCrSFA))
                tAsA_s = thr_a.partition_S(sA)[None, None, None, s]
                tBsB_s = thr_b.partition_S(sB)[None, None, None, s]
                for k in cutlass.range_constexpr(0, k_blocks):
                    cute.copy(s2r_a, tAsA_s[None, None, k], tCrA_v[None, None, k])
                    cute.copy(s2r_b, tBsB_s[None, None, k], tCrB_v[None, None, k])
                # the fp4 ABI shift follows the tensor, which the swap puts on A
                tCrA_i8 = cute.recast_tensor(tCrA, cutlass.Int8)
                tCrA_i8.store(tCrA_i8.load() * cutlass.Int8(4))
                for k_block in cutlass.range_constexpr(0, k_blocks):
                    cute.gemm(tiledmma, acc,
                              [tCrA[None, None, k_block], tCrSFA_frg[None, None, k_block, 0]],
                              [tCrB[None, None, k_block], tCrSFB_frg[None, None, k_block, k_in_sf]], acc)
                cute.arch.mbarrier_arrive(ab_empty + s)
                if cutlass.const_expr(s == ab_stages - 1):
                    ab_phase ^= 1
            cute.arch.mbarrier_arrive(sf_empty + sf_stage)
        sf_phase ^= 1
    # unrolled so is_last_k_tile is trace-time constant (upstream kernel_impl.cuh:313-341)
    for sf_stage in cutlass.range_constexpr(sf_stages):
        cute.arch.mbarrier_wait(sf_full + sf_stage, sf_phase)
        cute.copy(s2r_sfb, thr_sfb.partition_S(cute.slice_(sSFB, (None, None, sf_stage))),
                  thr_sfb.retile(tCrSFB))
        for k_in_sf in cutlass.range_constexpr(kt_per_pack_b):
            s = (sf_stage * kt_per_pack_b + k_in_sf) & (ab_stages - 1)
            cute.arch.mbarrier_wait(ab_full + s, ab_phase)
            cute.copy(s2r_sfa, thr_sfa.partition_S(cute.slice_(sSFA, (None, None, s))),
                      thr_sfa.retile(tCrSFA))
            tAsA_s = thr_a.partition_S(sA)[None, None, None, s]
            tBsB_s = thr_b.partition_S(sB)[None, None, None, s]
            for k in cutlass.range_constexpr(0, k_blocks):
                cute.copy(s2r_a, tAsA_s[None, None, k], tCrA_v[None, None, k])
                cute.copy(s2r_b, tBsB_s[None, None, k], tCrB_v[None, None, k])
            # the fp4 ABI shift follows the tensor, which the swap puts on A
            tCrA_i8 = cute.recast_tensor(tCrA, cutlass.Int8)
            tCrA_i8.store(tCrA_i8.load() * cutlass.Int8(4))
            if cutlass.const_expr(union_smem and sf_stage == sf_stages - 1
                                  and k_in_sf == kt_per_pack_b - 1):
                cute.arch.barrier(barrier_id=epi_bar_id, number_of_threads=mma_threads)
            for k_block in cutlass.range_constexpr(0, k_blocks):
                cute.gemm(tiledmma, acc,
                          [tCrA[None, None, k_block], tCrSFA_frg[None, None, k_block, 0]],
                          [tCrB[None, None, k_block], tCrSFB_frg[None, None, k_block, k_in_sf]], acc)
            cute.arch.mbarrier_arrive(ab_empty + s)
            if cutlass.const_expr(s == ab_stages - 1):
                ab_phase ^= 1
        cute.arch.mbarrier_arrive(sf_empty + sf_stage)
    sf_phase ^= 1
    return acc, ab_phase, sf_phase


@cute.jit
def store_staged(acc, thr, sC, out_dtype, store_full, store_empty, store_phase):
    """Hand the tile to the store warp through its own sC and go back to the math."""
    cute.arch.mbarrier_wait(store_empty, store_phase)
    store_output.acc_to_smem(store_output.convert_acc(acc, out_dtype), thr, sC)
    cute.arch.mbarrier_arrive(store_full)
    return store_phase ^ 1


@cute.jit
def store_swap(acc, thr, gD_t, tile, tile_mn, n_total, out_dtype):
    """Straight from the MMA's C partition to gmem through D^T: axis 0 is the output channel."""
    bm, bn = tile_mn
    gD_tile = cute.local_tile(cute.domain_offset((0, tile.m_offset), gD_t), (bm, bn),
                              (tile.n_block, tile.m_block))
    store_output.store_direct(thr, store_output.convert_acc(acc, out_dtype), gD_tile, (bm, bn),
                              tile.n_block * bm, tile.m_offset + tile.m_block * bn,
                              n_total, tile.m_boundary)


class CutedslSm120MoeMxfp8Mxfp4Grouped:
    def __init__(self, cfg, grid_x):
        self.cfg = cfg
        self.grid_x = grid_x
        self.mma = cfg.mma
        self.sf = cfg.load_sf
        bk = cfg.TILE[2]
        swap = cfg.mma.swap_ab
        kt_fine = self.sf.k_tiles_per_pack_a(bk) if swap else self.sf.k_tiles_per_pack_b(bk)
        assert kt_fine == 1, (
            f"tile-K {bk} puts {kt_fine} k-tiles in one fine-side SF pack; that side shares the A/B "
            f"ring and needs exactly one")
        kt_per_coarse_cycle = ((self.sf.sfb_stages(cfg.ab_stage, bk) * self.sf.k_tiles_per_pack_b(bk))
                               if swap else
                               (self.sf.sfa_stages(cfg.ab_stage, bk) * self.sf.k_tiles_per_pack_a(bk)))
        assert kt_per_coarse_cycle % cfg.ab_stage == 0, (
            f"the A/B ring of {cfg.ab_stage} does not divide the {kt_per_coarse_cycle} k-tiles of one "
            f"coarse-SF cycle; every stage index in the k-loop is a trace-time constant that has to "
            f"repeat with that cycle")
        self.sched_warp = cfg.sched_warp
        self.ab_warp = cfg.ab_warp
        self.sf_warp = cfg.sf_warp

    @cute.jit
    def __call__(self, gA: cute.Tensor, gB_u8: cute.Tensor, gSFA_u8: cute.Tensor, gSFB_u8: cute.Tensor,
                 gD: cute.Tensor, offsets: cute.Tensor, stream):
        cfg = self.cfg
        tiledmma = cfg.mma.make_tiled_mma(cfg.TILE)
        gB_e = cute.recast_tensor(gB_u8, cutlass.Float4E2M1FN)
        E, N, K = cute.size(gB_e, mode=[0]), cute.size(gB_e, mode=[1]), cute.size(gB_e, mode=[2])
        gW = cute.make_tensor(gB_e.iterator, cute.make_layout((N, K, E), stride=(K, 1, N * K)))
        M = cute.size(gA, mode=[0])
        m_padded_sf = cute.size(gSFA_u8, mode=[1]) // 4
        sf_dtype = cfg.load_sf.sf_dtype
        # every operand and every scale changes places here and nowhere else
        if cutlass.const_expr(cfg.mma.swap_ab):
            t_a, t_b = gW, gA
            gSFA = cute.make_tensor(cute.recast_ptr(gSFB_u8.iterator, dtype=sf_dtype),
                                    self.sf.deduce_sfa_layout(N, K, E))
            gSFB = cute.make_tensor(cute.recast_ptr(gSFA_u8.iterator, dtype=sf_dtype),
                                    self.sf.deduce_sfb_layout(m_padded_sf, K, 1))
        else:
            t_a, t_b = gA, gW
            gSFA = cute.make_tensor(cute.recast_ptr(gSFA_u8.iterator, dtype=sf_dtype),
                                    self.sf.deduce_sfa_layout(m_padded_sf, K, 1))
            gSFB = cute.make_tensor(cute.recast_ptr(gSFB_u8.iterator, dtype=sf_dtype),
                                    self.sf.deduce_sfb_layout(N, K, E))
        a_layout = cutlass.utils.LayoutEnum.from_tensor(t_a)
        b_layout = cutlass.utils.LayoutEnum.from_tensor(t_b)
        assert a_layout.is_k_major_a() and b_layout.is_k_major_b(), "LoadABConfig is k-major only"
        a_smem = cfg.load_ab.make_smem_layout_a()
        b_smem = cfg.load_ab.make_smem_layout_b()
        epi_smem = cfg.store.make_smem_layout(cfg.TILE)
        self.a_is_m_major = a_layout.is_m_major_a()

        bm, bn, bk = cfg.TILE[0], cfg.TILE[1], cfg.TILE[2]
        sfa_stages = cfg.load_sf.sfa_stages(cfg.ab_stage, bk)
        sfb_stages = cfg.load_sf.sfb_stages(cfg.ab_stage, bk)
        sfa_smem = cfg.load_sf.make_smem_layout_sfa(bm, sfa_stages)
        sfb_smem = cfg.load_sf.make_smem_layout_sfb(bn, sfb_stages)

        tma_atom_a, tma_tensor_a = cfg.load_ab.make_tma_atom_a(t_a, a_smem)
        tma_atom_b, tma_tensor_b = cfg.load_ab.make_tma_atom_b(t_b, b_smem)
        tma_atom_sfa, tma_tensor_sfa = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), gSFA, cute.slice_(sfa_smem, (None, None, 0)), (bm, 1), num_multicast=1)
        tma_atom_sfb, tma_tensor_sfb = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), gSFB, cute.slice_(sfb_smem, (None, None, 0)), (bn, 1), num_multicast=1)

        ring_smem = sfa_smem if cutlass.const_expr(cfg.mma.swap_ab) else sfb_smem
        warp_smem = sfb_smem if cutlass.const_expr(cfg.mma.swap_ab) else sfa_smem
        coarse_stages = sfb_stages if cutlass.const_expr(cfg.mma.swap_ab) else sfa_stages
        self.ab_sfb_bytes = (cfg.load_ab.tma_bytes_ab
                         + cute.size_in_bytes(sf_dtype, cute.slice_(ring_smem, (None, None, 0))))
        self.sf_bytes = cute.size_in_bytes(sf_dtype, cute.slice_(warp_smem, (None, None, 0)))

        # what the store method does not use is not allocated
        store_full_bars = cfg.store_stages if cfg.store.HAS_STORE_WARP else 0
        store_empty_bars = 0 if cfg.store.METHOD is StoreMethod.DIRECT_STG else cfg.store_stages
        epi_elems = cute.cosize(epi_smem) if cfg.store.METHOD is StoreMethod.STAGED_R2G else 0

        @cute.struct
        class SharedStorage:
            ab_full: cute.struct.MemRange[cfg.I64, cfg.ab_stage]
            ab_empty: cute.struct.MemRange[cfg.I64, cfg.ab_stage]
            sf_full: cute.struct.MemRange[cfg.I64, coarse_stages]
            sf_empty: cute.struct.MemRange[cfg.I64, coarse_stages]
            store_full: cute.struct.MemRange[cfg.I64, store_full_bars]
            store_empty: cute.struct.MemRange[cfg.I64, store_empty_bars]
            sfull: cute.struct.MemRange[cfg.I64, cfg.sched_stages]
            sempty: cute.struct.MemRange[cfg.I64, cfg.sched_stages]
            work: cute.struct.MemRange[cfg.I32, cfg.sched_stages * cfg.fields]
            sA: cute.struct.Align[cute.struct.MemRange[cfg.load_ab.a_smem_dtype, cute.cosize(a_smem)], 128]
            sB: cute.struct.Align[cute.struct.MemRange[cfg.load_ab.b_smem_dtype, cute.cosize(b_smem)], 128]
            sSFB: cute.struct.Align[cute.struct.MemRange[cfg.load_sf.sf_dtype, cute.cosize(sfb_smem)], 128]
            sSFA: cute.struct.Align[cute.struct.MemRange[cfg.load_sf.sf_dtype, cute.cosize(sfa_smem)], 128]
            sC: cute.struct.Align[cute.struct.MemRange[cfg.store.out_dtype, epi_elems], 128]

        # keep the analytic cost model honest against the struct the kernel really allocates
        assert cfg.smem_bytes <= SharedStorage.__sizeof__() <= cfg.smem_bytes + cfg.MBAR_RESERVE, (
            f"smem model {cfg.smem_bytes} B vs allocated {SharedStorage.__sizeof__()} B")

        self.storage = SharedStorage
        self.kernel(tiledmma, tma_atom_a, tma_atom_b, tma_atom_sfa, tma_atom_sfb,
                    tma_tensor_a, tma_tensor_b, tma_tensor_sfa, tma_tensor_sfb, gD, offsets,
                    a_smem, b_smem, sfa_smem, sfb_smem, epi_smem).launch(
            grid=[self.grid_x, 1, 1], block=[cfg.threads, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, tiledmma, tma_atom_a, tma_atom_b, tma_atom_sfa, tma_atom_sfb,
               tma_tensor_a: cute.Tensor, tma_tensor_b: cute.Tensor, tma_tensor_sfa: cute.Tensor,
               tma_tensor_sfb: cute.Tensor, gD: cute.Tensor, offsets: cute.Tensor,
               a_smem, b_smem, sfa_smem, sfb_smem, epi_smem):
        cfg = self.cfg
        i32 = cutlass.Int32
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        bidx, _, _ = cute.arch.block_idx()
        bm, bn, bk = cfg.TILE[0], cfg.TILE[1], cfg.TILE[2]
        swap = cfg.mma.swap_ab
        # the scheduler and gD stay in problem space; only the buffers and the MMA are swapped
        pm, pn = (bn, bm) if cutlass.const_expr(swap) else (bm, bn)
        M, N = cute.size(gD, mode=[0]), cute.size(gD, mode=[1])
        K = cute.size(tma_tensor_b, mode=[1])
        num_groups = cute.size(offsets, mode=[0]) - 1
        num_n_blocks = ceil_div(N, pn)
        # the mirror itself: no data moves, D is just read as its own transpose
        gD_t = cute.make_tensor(gD.iterator, cute.make_layout((N, M), stride=(1, N)))
        # the coarse SF side owns the outer cycle, and the swap is what decides which side that is
        grank_c = self.sf.grank_b if cutlass.const_expr(swap) else self.sf.grank_a
        kt_per_pack_c = (self.sf.k_tiles_per_pack_b(bk) if cutlass.const_expr(swap)
                         else self.sf.k_tiles_per_pack_a(bk))
        sf_stages_c = (self.sf.sfb_stages(cfg.ab_stage, bk) if cutlass.const_expr(swap)
                       else self.sf.sfa_stages(cfg.ab_stage, bk))
        # whole cycles only, so a short final one never leaves the consumer waiting
        num_sf_cycles = ceil_div(K, grank_c * self.sf.PACK_NSF * sf_stages_c)
        k_tile_count = num_sf_cycles * sf_stages_c * kt_per_pack_c

        smem = cutlass.utils.SmemAllocator()
        stg = smem.allocate(self.storage)
        sA = stg.sA.get_tensor(a_smem.outer, swizzle=a_smem.inner)
        sB = stg.sB.get_tensor(b_smem.outer, swizzle=b_smem.inner)
        sSFB = stg.sSFB.get_tensor(sfb_smem)
        sSFA = stg.sSFA.get_tensor(sfa_smem)
        if cutlass.const_expr(cfg.store.METHOD is not StoreMethod.DIRECT_STG):
            if cutlass.const_expr(cfg.union_smem):
                sC_stages = cute.make_tensor(
                    cute.recast_ptr(stg.sA.data_ptr(), dtype=cfg.store.out_dtype), epi_smem)
            else:
                sC_stages = stg.sC.get_tensor(epi_smem.outer, swizzle=epi_smem.inner)
            # one epilogue stage today (StoreConfig.epi_stage); the index rotates once there are more
            sC = cute.slice_(sC_stages, (None, None, 0))
        ab_full, ab_empty = stg.ab_full.data_ptr(), stg.ab_empty.data_ptr()
        sf_full, sf_empty = stg.sf_full.data_ptr(), stg.sf_empty.data_ptr()
        # a zero-length MemRange yields the next field's address, so an unused handle stays None
        store_full = stg.store_full.data_ptr() if cutlass.const_expr(cfg.store.HAS_STORE_WARP) else None
        store_empty = (stg.store_empty.data_ptr()
                       if cutlass.const_expr(cfg.store.METHOD is not StoreMethod.DIRECT_STG) else None)
        sfull, sempty = stg.sfull.data_ptr(), stg.sempty.data_ptr()
        sWork = stg.work.get_tensor(cute.make_layout((cfg.sched_stages, cfg.fields)))

        if warp_idx == 0:
            with cute.arch.elect_one():
                for s in cutlass.range_constexpr(cfg.ab_stage):
                    cute.arch.mbarrier_init(ab_full + s, 1)
                    cute.arch.mbarrier_init(ab_empty + s, cfg.mma_threads)
                for s in cutlass.range_constexpr(sf_stages_c):
                    cute.arch.mbarrier_init(sf_full + s, 1)
                    cute.arch.mbarrier_init(sf_empty + s, cfg.mma_threads)
                if cutlass.const_expr(cfg.store.HAS_STORE_WARP):
                    cute.arch.mbarrier_init(store_full, cfg.mma_threads)
                if cutlass.const_expr(cfg.store.METHOD is not StoreMethod.DIRECT_STG):
                    cute.arch.mbarrier_init(store_empty, cfg.store_threads)
                for s in cutlass.range_constexpr(cfg.sched_stages):
                    cute.arch.mbarrier_init(sfull + s, 1)
                    cute.arch.mbarrier_init(sempty + s, cfg.num_sched_consumers)
        cute.arch.mbarrier_init_fence()
        cute.arch.barrier()

        # built before any warp-role branch: a config method call inside one carries cfg across it
        if cutlass.const_expr(cfg.store.METHOD is not StoreMethod.DIRECT_STG):
            st_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cfg.store.out_dtype)
            tiled_st = cfg.store.make_tiled_s2g(st_atom, cfg.TILE)

        if cfg.is_prod_wg(warp_idx):
            cute.arch.setmaxregister_decrease(cfg.reg_prod)

            if warp_idx == self.sched_warp:
                sched = scheduler.MoeTileScheduler.create(pm, num_groups, num_n_blocks, self.grid_x, bidx, offsets)
                prod = scheduler.MoeSchedProducer.create(cfg.sched_stages)
                has, e, m_tile, n_tile, m_off, m_bnd = sched.get_next_block(offsets)
                while has:
                    prod.publish(sWork, sfull, sempty,
                                 scheduler.MoeWorkTile(m_tile, n_tile, e, m_off, m_bnd, i32(1)))
                    has, e, m_tile, n_tile, m_off, m_bnd = sched.get_next_block(offsets)
                prod.publish_sentinel(sWork, sfull, sempty)

            if warp_idx == self.ab_warp:
                cons = scheduler.MoeSchedConsumer.create(cfg.sched_stages)
                tile = cons.get_next_tile(sWork, sfull, sempty)
                abphase, stphase = i32(1), i32(1)
                while tile.valid != i32(0):
                    # sC overlays sA, and ab_empty alone would let this warp refill it mid-epilogue
                    if cutlass.const_expr(cfg.union_smem):
                        cute.arch.mbarrier_wait(store_empty, stphase)
                        stphase ^= 1
                    if cutlass.const_expr(swap):
                        abphase = load_ab_swap(
                            tma_atom_a, tma_atom_b, tma_atom_sfa, tma_tensor_a, tma_tensor_b,
                            tma_tensor_sfa, sA, sB, sSFA, cfg.TILE, tile, ab_full, ab_empty,
                            self.ab_sfb_bytes, k_tile_count, cfg.ab_stage, abphase)
                    else:
                        abphase = load_ab(
                            tma_atom_a, tma_atom_b, tma_atom_sfb, tma_tensor_a, tma_tensor_b,
                            tma_tensor_sfb, sA, sB, sSFB, cfg.TILE, tile, ab_full, ab_empty,
                            self.ab_sfb_bytes, k_tile_count, cfg.ab_stage, abphase)
                    tile = cons.get_next_tile(sWork, sfull, sempty)

            if warp_idx == self.sf_warp:
                cons = scheduler.MoeSchedConsumer.create(cfg.sched_stages)
                tile = cons.get_next_tile(sWork, sfull, sempty)
                sfaphase = i32(1)
                while tile.valid != i32(0):
                    if cutlass.const_expr(swap):
                        sfaphase = load_sf_swap(
                            tma_atom_sfb, tma_tensor_sfb, sSFB, bn, tile, sf_full, sf_empty,
                            self.sf_bytes, SF_M_ALIGN, num_sf_cycles, sf_stages_c, sfaphase)
                    else:
                        sfaphase = load_sf(
                            tma_atom_sfa, tma_tensor_sfa, sSFA, bm, tile, sf_full, sf_empty,
                            self.sf_bytes, SF_M_ALIGN, num_sf_cycles, sf_stages_c, sfaphase)
                    tile = cons.get_next_tile(sWork, sfull, sempty)

            if cutlass.const_expr(cfg.store.HAS_STORE_WARP):
                if warp_idx == cfg.store_warp:
                    thr_st = tiled_st.get_slice(tidx - cfg.store_warp * 32)
                    cons = scheduler.MoeSchedConsumer.create(cfg.sched_stages)
                    tile = cons.get_next_tile(sWork, sfull, sempty)
                    stphase = i32(0)
                    while tile.valid != i32(0):
                        cute.arch.mbarrier_wait(store_full, stphase)
                        stphase ^= 1
                        gD_tile = cute.local_tile(cute.domain_offset((tile.m_offset, 0), gD), (bm, bn), (tile.m_block, tile.n_block))
                        store_output.store_predicated(
                            st_atom, thr_st, store_output.smem_to_reg(thr_st, sC), gD_tile, (bm, bn),
                            tile.m_offset + tile.m_block * bm, tile.n_block * bn, tile.m_boundary, N)
                        cute.arch.mbarrier_arrive(store_empty)
                        tile = cons.get_next_tile(sWork, sfull, sempty)

        else:
            cute.arch.setmaxregister_increase(cfg.reg_math)
            thr = tiledmma.get_slice(tidx)
            if cutlass.const_expr(cfg.store.METHOD is StoreMethod.R2G_WG):
                thr_st = tiled_st.get_slice(tidx)
            abphase, sfaphase, stphase = i32(0), i32(0), i32(1)
            cons = scheduler.MoeSchedConsumer.create(cfg.sched_stages)
            tile = cons.get_next_tile(sWork, sfull, sempty)
            while tile.valid != i32(0):
                if cutlass.const_expr(swap):
                    acc, abphase, sfaphase = mma_swap(
                        tiledmma, self.mma, self.sf, sA, sB, sSFA, sSFB, (bm, bn), bk, cfg.ACC,
                        cfg.load_ab.a_smem_dtype, cfg.load_ab.b_dtype,
                        cfg.load_ab.a_unpack_bits, tidx, ab_full, ab_empty, sf_full, sf_empty,
                        num_sf_cycles, sf_stages_c, kt_per_pack_c, cfg.ab_stage, abphase, sfaphase,
                        cfg.union_smem, cfg.epi_bar_id, cfg.mma_threads)
                else:
                    acc, abphase, sfaphase = mma(
                        tiledmma, self.mma, self.sf, sA, sB, sSFA, sSFB, (bm, bn), bk, cfg.ACC,
                        cfg.load_ab.a_dtype, cfg.load_ab.b_smem_dtype, self.a_is_m_major,
                        cfg.load_ab.b_unpack_bits, tidx, ab_full, ab_empty, sf_full, sf_empty,
                        num_sf_cycles, sf_stages_c, kt_per_pack_c, cfg.ab_stage, abphase, sfaphase,
                        cfg.union_smem, cfg.epi_bar_id, cfg.mma_threads)

                if cutlass.const_expr(cfg.store.METHOD is StoreMethod.DIRECT_STG):
                    store_swap(acc, thr, gD_t, tile, (bm, bn), N, cfg.store.out_dtype)
                elif cutlass.const_expr(cfg.store.HAS_STORE_WARP):
                    stphase = store_staged(acc, thr, sC, cfg.store.out_dtype,
                                           store_full, store_empty, stphase)
                else:
                    # the last-k-tile barrier that guards sC-over-sA already ran inside mma
                    tD = store_output.convert_acc(acc, cfg.store.out_dtype)
                    store_output.acc_to_smem(tD, thr, sC)
                    cute.arch.barrier(barrier_id=cfg.epi_bar_id, number_of_threads=cfg.mma_threads)
                    tDrD = store_output.smem_to_reg(thr_st, sC)
                    cute.arch.barrier(barrier_id=cfg.epi_bar_id, number_of_threads=cfg.mma_threads)
                    # sC (aliasing sA) returns to the producers before the gmem store drains
                    cute.arch.mbarrier_arrive(store_empty)
                    gD_tile = cute.local_tile(cute.domain_offset((tile.m_offset, 0), gD), (bm, bn), (tile.m_block, tile.n_block))
                    store_output.store_predicated(
                        st_atom, thr_st, tDrD, gD_tile, (bm, bn),
                        tile.m_offset + tile.m_block * bm, tile.n_block * bn, tile.m_boundary, N)
                tile = cons.get_next_tile(sWork, sfull, sempty)


def _stream():
    return cuda.CUstream(torch.cuda.current_stream().cuda_stream)


def make_args(a_q, b_q, a_scale, b_scale, out, m_indptr):
    sf = lambda t: from_dlpack(t.contiguous(), assumed_align=16).mark_layout_dynamic()
    return (from_dlpack(a_q).mark_layout_dynamic(), from_dlpack(b_q).mark_layout_dynamic(),
            sf(a_scale), sf(b_scale), from_dlpack(out).mark_layout_dynamic(),
            from_dlpack(m_indptr).mark_layout_dynamic(), _stream())
