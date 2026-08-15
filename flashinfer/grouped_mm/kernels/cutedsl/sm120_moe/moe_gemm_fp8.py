# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Self-written CuteDSL fp8 block-scaled (1x128x128) token-packed grouped MoE GEMM for SM120a."""
import torch

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.warp.mma as warp_mma
import cutlass.utils as utils
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import from_dlpack

from .core.sm120_gemm_builder import (
    Sm120GemmBuilder, MmaConfig, LoadABConfig, StoreMethod, STORE_CONFIGS)
from .core.sm120_blockscaled_layout import Sm120SfConfigFp8
from .core._common import TMA_ALIGN_BYTES
from .core._common import ceil_div, compute_padded_offset
from .core import scheduler, store_output


# The arm's scale-granularity contract (DSv3 1x128x128); the op layer reads it here, not a copy.
GRAN_M, GRAN_N, GRAN_K = 1, 128, 128

ATOM_MNK = (16, 8, 32)

def is_swapab(tile):
    return ATOM_MNK[0] > tile[0] # bm == 8


def make_cfg(tile, ab_stage, store=StoreMethod.STAGED_R2G):
    """``tile`` in is problem space; everything the builder sees below is kernel space, never mixed."""
    e4m3, f32, bf16 = cutlass.Float8E4M3FN, cutlass.Float32, cutlass.BFloat16
    num_math_warps = 8
    bm, bn, bk = tile
    swap = is_swapab(tile)
    if swap:
        bm, bn = bn, bm
        gran_m, gran_n = GRAN_N, GRAN_M
        store = StoreMethod.DIRECT_STG
    else:
        # DIRECT_STG is the swapped epilogue: store_swap writes D^T and indexes with n_block
        assert store is not StoreMethod.DIRECT_STG, (
            f"{store} needs the swapped geometry; this tile is not swap-AB")
        gran_m, gran_n = GRAN_M, GRAN_N
    tile = (bm, bn, bk)
    # the whole warpgroup drains sC itself, which is what lets sC alias sA instead of costing smem
    union = store is StoreMethod.R2G_WG
    return Sm120GemmBuilder(
        MmaConfig(warp_mma.MmaFP8Op(e4m3, f32, ATOM_MNK), tile[:2], num_math_warps, swap_ab=swap),
        LoadABConfig(tile, ab_stage, e4m3, e4m3),
        Sm120SfConfigFp8(gran_m, gran_n, GRAN_K),
        STORE_CONFIGS[store](bf16, num_math_warps * 32),
        ab_stage, tile, epi_bar_id=3, union_smem=union)


@cute.jit
def load_ab(tma_atom_a, tma_atom_b, tma_tensor_a, tma_tensor_b, sA, sB, tile_mnk, tile,
            ab_full, ab_empty, ab_bytes, k_tile_count, ab_stages, ab_stage, ab_phase):
    """One tile's A/B G2S: partition A at the expert's row offset, then run the k-loop."""
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
    for k_tile_idx in cutlass.range(0, k_tile_count):
        cute.arch.mbarrier_wait(ab_empty + ab_stage, ab_phase)
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(ab_full + ab_stage, ab_bytes)
        cute.copy(tma_atom_a, tAgA[(None, tile.m_block, k_tile_idx)], tAsA[(None, ab_stage)], tma_bar_ptr=ab_full + ab_stage)
        cute.copy(tma_atom_b, tBgB[(None, tile.n_block, k_tile_idx, tile.group)], tBsB[(None, ab_stage)], tma_bar_ptr=ab_full + ab_stage)
        ab_stage += 1
        if ab_stage == ab_stages:
            ab_stage = i32(0)
            ab_phase ^= 1
    return ab_stage, ab_phase


@cute.jit
def load_ab_swap(tma_atom_a, tma_atom_b, tma_tensor_a, tma_tensor_b, sA, sB, tile_mnk, tile,
                 ab_full, ab_empty, ab_bytes, k_tile_count, ab_stages, ab_stage, ab_phase):
    """One tile's A/B G2S with the operands exchanged: A is the expert's weights, B the tokens."""
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
    for k_tile_idx in cutlass.range(0, k_tile_count):
        cute.arch.mbarrier_wait(ab_empty + ab_stage, ab_phase)
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(ab_full + ab_stage, ab_bytes)
        cute.copy(tma_atom_a, tAgA[(None, tile.m_block, k_tile_idx, tile.group)], tAsA[(None, ab_stage)], tma_bar_ptr=ab_full + ab_stage)
        cute.copy(tma_atom_b, tBgB[(None, tile.n_block, k_tile_idx)], tBsB[(None, ab_stage)], tma_bar_ptr=ab_full + ab_stage)
        ab_stage += 1
        if ab_stage == ab_stages:
            ab_stage = i32(0)
            ab_phase ^= 1
    return ab_stage, ab_phase


@cute.jit
def load_sf_tma(tma_atom_sfa, tma_tensor_sfa, sSFA_tma, sf_atom, gSFB, sSFB, tile,
                sf_full, sf_empty, sfa_bytes, num_copy_threads,
                k_tile_count, sf_stages, sf_stage, sf_phase):
    """One tile's SF G2S with SFA on TMA and SFB on cp.async, the non-swap pairing."""
    i32 = cutlass.Int32
    cluster = cute.make_layout((1, 1, 1))
    multicast = cute.make_layout(cute.slice_(cluster, (0, None, 0)).shape)
    tscale_m, tscale_n = cute.size(sSFA_tma, mode=[0]), cute.size(sSFB, mode=[0])
    scale_n = cute.size(gSFB, mode=[0])
    m_align = TMA_ALIGN_BYTES // (sSFB.element_type.width // 8)
    lane_idx = cute.arch.lane_idx()

    sf_m_off = compute_padded_offset(tile.m_offset, tile.group, i32(m_align))
    mSFA = cute.domain_offset((sf_m_off, 0, 0), tma_tensor_sfa)
    gSFA_ml_t = cute.local_tile(mSFA, (tscale_m, 1), (None, None, None))
    tSAs, tSAg = cpasync.tma_partition(tma_atom_sfa, i32(0), multicast, cute.group_modes(sSFA_tma, 0, 2),
                                       cute.group_modes(gSFA_ml_t, 0, 2))

    cSFB = cute.make_identity_tensor(gSFB.shape)
    gSFB_nk = cute.local_tile(gSFB, (tscale_n,), (tile.n_block, None, tile.group))
    coordSFB = cute.local_tile(cSFB, (tscale_n,), (tile.n_block, None, tile.group))
    scale_copy_b = cute.make_tiled_copy_tv(sf_atom, cute.make_layout(num_copy_threads),
                                           cute.make_layout(1))
    thr_scale_copy_b = scale_copy_b.get_slice(lane_idx)
    tBgSFB = thr_scale_copy_b.partition_S(gSFB_nk)
    tBcSFB = thr_scale_copy_b.partition_S(coordSFB)
    tBsSFB = thr_scale_copy_b.partition_D(sSFB)
    tBpSFB = cute.make_fragment_like(tBcSFB[None, None, 0], cutlass.Boolean)
    for i in cutlass.range_constexpr(cute.size(tBpSFB)):
        tBpSFB[i] = lane_idx < tscale_n and tBcSFB[i][0] < scale_n

    for k_tile_idx in cutlass.range(0, k_tile_count):
        cute.arch.mbarrier_wait(sf_empty + sf_stage, sf_phase)
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(sf_full + sf_stage, sfa_bytes)
        cute.copy(tma_atom_sfa, tSAg[(None, tile.m_block, k_tile_idx, 0)], tSAs[(None, sf_stage)], tma_bar_ptr=sf_full + sf_stage)
        cute.arch.sync_warp()
        # a predicated-off cp.async zero-fills rather than skipping, so out-of-tile lanes must not issue
        if lane_idx < tscale_n:
            cute.copy(scale_copy_b, tBgSFB[None, None, k_tile_idx], tBsSFB[None, None, sf_stage], pred=tBpSFB)
        cute.arch.cp_async_mbarrier_arrive_noinc(sf_full + sf_stage)
        sf_stage += 1
        if sf_stage == sf_stages:
            sf_stage = i32(0)
            sf_phase ^= 1
    return sf_stage, sf_phase


@cute.jit
def load_sf(gSFA, sSFA, gSFB, sSFB, sf_atom, tile, sf_full, sf_empty, num_copy_threads,
            k_tile_count, sf_stages, sf_stage, sf_phase):
    """One tile's SF G2S with both sides on cp.async, the swap-AB pairing; tile blocks are kernel-space."""
    i32 = cutlass.Int32
    tscale_m, tscale_n = cute.size(sSFA, mode=[0]), cute.size(sSFB, mode=[0])
    scale_m, scale_n = cute.size(gSFA, mode=[0]), cute.size(gSFB, mode=[0])
    m_align = TMA_ALIGN_BYTES // (sSFB.element_type.width // 8)
    lane_idx = cute.arch.lane_idx()

    sf_m_off = compute_padded_offset(tile.m_offset, tile.group, i32(m_align))
    mSFB = cute.domain_offset((sf_m_off, 0, 0), gSFB)
    cSFA = cute.make_identity_tensor(gSFA.shape)
    cSFB = cute.domain_offset((sf_m_off, 0, 0), cute.make_identity_tensor(gSFB.shape))

    gSFA_mk = cute.local_tile(gSFA, (tscale_m,), (tile.m_block, None, tile.group))
    coordSFA = cute.local_tile(cSFA, (tscale_m,), (tile.m_block, None, tile.group))
    gSFB_nk = cute.local_tile(mSFB, (tscale_n,), (tile.n_block, None, 0))
    coordSFB = cute.local_tile(cSFB, (tscale_n,), (tile.n_block, None, 0))

    scale_copy_a = cute.make_tiled_copy_tv(sf_atom, cute.make_layout(num_copy_threads),
                                           cute.make_layout(1))
    scale_copy_b = cute.make_tiled_copy_tv(sf_atom, cute.make_layout(num_copy_threads),
                                           cute.make_layout(1))
    thr_scale_copy_a = scale_copy_a.get_slice(lane_idx)
    thr_scale_copy_b = scale_copy_b.get_slice(lane_idx)
    tAgSFA = thr_scale_copy_a.partition_S(gSFA_mk)
    tAcSFA = thr_scale_copy_a.partition_S(coordSFA)
    tAsSFA = thr_scale_copy_a.partition_D(sSFA)
    tBgSFB = thr_scale_copy_b.partition_S(gSFB_nk)
    tBcSFB = thr_scale_copy_b.partition_S(coordSFB)
    tBsSFB = thr_scale_copy_b.partition_D(sSFB)
    tApSFA = cute.make_fragment_like(tAcSFA[None, None, 0], cutlass.Boolean)
    for i in cutlass.range_constexpr(cute.size(tApSFA)):
        tApSFA[i] = lane_idx < tscale_m and tAcSFA[i][0] < scale_m
    tBpSFB = cute.make_fragment_like(tBcSFB[None, None, 0], cutlass.Boolean)
    for i in cutlass.range_constexpr(cute.size(tBpSFB)):
        tBpSFB[i] = lane_idx < tscale_n and tBcSFB[i][0] < scale_n

    for k_tile_idx in cutlass.range(0, k_tile_count):
        cute.arch.mbarrier_wait(sf_empty + sf_stage, sf_phase)
        # a predicated-off cp.async zero-fills rather than skipping, so out-of-tile lanes must not issue
        if lane_idx < tscale_m:
            cute.copy(scale_copy_a, tAgSFA[None, None, k_tile_idx], tAsSFA[None, None, sf_stage], pred=tApSFA)
        if lane_idx < tscale_n:
            cute.copy(scale_copy_b, tBgSFB[None, None, k_tile_idx], tBsSFB[None, None, sf_stage], pred=tBpSFB)
        cute.arch.cp_async_mbarrier_arrive_noinc(sf_full + sf_stage)
        sf_stage += 1
        if sf_stage == sf_stages:
            sf_stage = i32(0)
            sf_phase ^= 1
    return sf_stage, sf_phase


@cute.jit
def copy_scale_s2r(stage, tCsSFA, tCsSFB, tCrSFA, tCrSFB, tscale_mn):
    """Both scales into registers, folding the broadcast side into the varying one."""
    tscale_m, tscale_n = tscale_mn
    cute.autovec_copy(tCsSFA[None, None, None, stage], tCrSFA)
    cute.autovec_copy(tCsSFB[None, None, None, stage], tCrSFB)
    if tscale_m == 1 and tscale_n == 1:
        tCrSFA[0] = tCrSFA[0] * tCrSFB[0]
    elif tscale_m > 1 and tscale_n == 1:
        tCrSFA.store(tCrSFA.load() * tCrSFB.load()[0])
    elif tscale_m == 1 and tscale_n > 1:
        tCrSFB.store(tCrSFB.load() * tCrSFA.load()[0])


@cute.jit
def rescale(acc, tmp, tCrSFA, tCrSFB, tscale_mn):
    """Fold one k-tile's plain accumulator into the scaled one and clear it."""
    tscale_m, tscale_n = tscale_mn
    if tscale_m == 1 and tscale_n == 1:
        acc.store(acc.load() + tmp.load() * tCrSFA.load()[0])
    elif tscale_m > 1 and tscale_n == 1:
        acc.store(acc.load() + tmp.load() * tCrSFA.load())
    elif tscale_m == 1 and tscale_n > 1:
        acc.store(acc.load() + tmp.load() * tCrSFB.load())
    else:
        acc.store(acc.load() + tmp.load() * tCrSFA.load() * tCrSFB.load())
    tmp.fill(0.0)


@cute.jit
def store_staged(acc, thr, sC, out_dtype, store_full, store_empty, store_phase):
    """Hand the tile to the store warp through its own sC and go back to the math."""
    cute.arch.mbarrier_wait(store_empty, store_phase)
    store_output.acc_to_smem(store_output.convert_acc(acc, out_dtype), thr, sC)
    cute.arch.mbarrier_arrive(store_full)
    return store_phase ^ 1


@cute.jit
def store_wg(acc, thr, thr_st, sC, st_atom, gD, tile, tile_mn, n_boundary, out_dtype,
             store_empty, epi_bar_id, mma_threads):
    """The math warpgroup drains sC itself; sC aliases sA, so it is released before the gmem store."""
    bm, bn = tile_mn
    tD = store_output.convert_acc(acc, out_dtype)
    cute.arch.barrier(barrier_id=epi_bar_id, number_of_threads=mma_threads)
    store_output.acc_to_smem(tD, thr, sC)
    cute.arch.barrier(barrier_id=epi_bar_id, number_of_threads=mma_threads)
    tDrD = store_output.smem_to_reg(thr_st, sC)
    cute.arch.barrier(barrier_id=epi_bar_id, number_of_threads=mma_threads)
    cute.arch.mbarrier_arrive(store_empty)
    gD_tile = cute.local_tile(cute.domain_offset((tile.m_offset, 0), gD), (bm, bn),
                              (tile.m_block, tile.n_block))
    store_output.store_predicated(st_atom, thr_st, tDrD, gD_tile, (bm, bn),
                                  tile.m_offset + tile.m_block * bm, tile.n_block * bn,
                                  tile.m_boundary, n_boundary)


@cute.jit
def store_swap(acc, thr, gD_t, tile, tile_mn, n_total, out_dtype):
    """Straight from the MMA's C partition to gmem through D^T: axis 0 is the output channel."""
    bm, bn = tile_mn
    gD_tile = cute.local_tile(cute.domain_offset((0, tile.m_offset), gD_t), (bm, bn),
                              (tile.m_block, tile.n_block))
    store_output.store_direct(thr, store_output.convert_acc(acc, out_dtype), gD_tile, (bm, bn),
                              tile.m_block * bm, tile.m_offset + tile.n_block * bn,
                              n_total, tile.m_boundary)


@cute.jit
def mma(tiledmma, mma_cfg, sf_cfg, sA, sB, sSFA, sSFB, tile_mn, acc_dtype,
        a_dtype, b_smem_dtype, a_is_m_major, b_is_n_major, tidx,
        ab_full, ab_empty, sf_full, sf_empty,
        k_tile_count, ab_stages, read_stage, ab_phase):
    """One tile's math: a k_block-deep s2r prefetch over the scaled k-loop, accumulator out."""
    i32 = cutlass.Int32
    bm, bn = tile_mn
    thr = tiledmma.get_slice(tidx)
    tCrA = tiledmma.make_fragment_A(thr.partition_A(sA)[None, None, None, 0])
    tCrB = tiledmma.make_fragment_B(thr.partition_B(sB)[None, None, None, 0])
    acc = cute.make_rmem_tensor(tiledmma.partition_shape_C((bm, bn)), acc_dtype)
    tmp = cute.make_rmem_tensor(tiledmma.partition_shape_C((bm, bn)), acc_dtype)
    s2r_a = mma_cfg.make_s2r_a(tiledmma, a_dtype, a_is_m_major)
    s2r_b = mma_cfg.make_s2r_b(tiledmma, b_smem_dtype, b_is_n_major)
    thr_a, thr_b = s2r_a.get_slice(tidx), s2r_b.get_slice(tidx)
    tCrA_v, tCrB_v = thr_a.retile(tCrA), thr_b.retile(tCrB)
    tCsSFA, tCsSFB, tCrSFA, tCrSFB = sf_cfg.partition_scale_as_c(sSFA, sSFB, thr)
    tscale_mn = (cute.size(sSFA, mode=[0]), cute.size(sSFB, mode=[0]))
    tXsA, tXsB = thr_a.partition_S(sA), thr_b.partition_S(sB)
    k_blocks = cute.size(tCrA_v, mode=[2])

    acc.fill(0.0)
    tmp.fill(0.0)

    cute.arch.mbarrier_wait(sf_full + read_stage, ab_phase)
    copy_scale_s2r(read_stage, tCsSFA, tCsSFB, tCrSFA, tCrSFB, tscale_mn)
    cute.arch.mbarrier_arrive(sf_empty + read_stage)

    cute.arch.mbarrier_wait(ab_full + read_stage, ab_phase)
    tAsA_s = tXsA[None, None, None, read_stage]
    tBsB_s = tXsB[None, None, None, read_stage]
    cute.copy(s2r_a, tAsA_s[None, None, 0], tCrA_v[None, None, 0])
    cute.copy(s2r_b, tBsB_s[None, None, 0], tCrB_v[None, None, 0])

    for k_tile_idx in cutlass.range(0, k_tile_count - 1):
        for k_block in cutlass.range_constexpr(0, k_blocks):
            k_next = 0 if k_block + 1 == k_blocks else k_block + 1
            if k_block != k_blocks - 1:
                cute.copy(s2r_a, tAsA_s[None, None, k_next], tCrA_v[None, None, k_next])
                cute.copy(s2r_b, tBsB_s[None, None, k_next], tCrB_v[None, None, k_next])
            cute.gemm(tiledmma, tmp, tCrA[None, None, k_block], tCrB[None, None, k_block], tmp)
            if k_block == k_blocks - 1:
                cute.arch.mbarrier_arrive(ab_empty + read_stage)
                read_stage += 1
                if read_stage == ab_stages:
                    read_stage = i32(0)
                    ab_phase ^= 1
                tAsA_s = tXsA[None, None, None, read_stage]
                tBsB_s = tXsB[None, None, None, read_stage]
                cute.arch.mbarrier_wait(ab_full + read_stage, ab_phase)
                cute.copy(s2r_a, tAsA_s[None, None, 0], tCrA_v[None, None, 0])
                cute.copy(s2r_b, tBsB_s[None, None, 0], tCrB_v[None, None, 0])
                rescale(acc, tmp, tCrSFA, tCrSFB, tscale_mn)
                cute.arch.mbarrier_wait(sf_full + read_stage, ab_phase)
                copy_scale_s2r(read_stage, tCsSFA, tCsSFB, tCrSFA, tCrSFB, tscale_mn)
                cute.arch.mbarrier_arrive(sf_empty + read_stage)

    for k_block in cutlass.range_constexpr(0, k_blocks):
        k_next = 0 if k_block + 1 == k_blocks else k_block + 1
        if k_next > 0:
            cute.copy(s2r_a, tAsA_s[None, None, k_next], tCrA_v[None, None, k_next])
            cute.copy(s2r_b, tBsB_s[None, None, k_next], tCrB_v[None, None, k_next])
        if k_block == k_blocks - 1:
            cute.arch.mbarrier_arrive(ab_empty + read_stage)
            read_stage += 1
            if read_stage == ab_stages:
                read_stage = i32(0)
                ab_phase ^= 1
        cute.gemm(tiledmma, tmp, tCrA[None, None, k_block], tCrB[None, None, k_block], tmp)
    rescale(acc, tmp, tCrSFA, tCrSFB, tscale_mn)
    return acc, read_stage, ab_phase


class CutedslSm120MoeFp8Grouped:
    def __init__(self, cfg, grid_x):
        self.cfg = cfg
        self.mma = cfg.mma
        self.scale = cfg.load_sf
        self.scale.assert_k_invariant(cfg.TILE[2])
        self.scale.assert_mn_invariant(cfg.TILE[0], cfg.TILE[1])
        self.sf_stages = self.scale.sf_stages(cfg.ab_stage, cfg.TILE[2])
        assert self.sf_stages == cfg.ab_stage, (
            f"SF ring is {self.sf_stages} deep against {cfg.ab_stage} for A/B, but the math warp "
            f"indexes both with one stage counter")
        self.grid_x = grid_x

    @cute.jit
    def __call__(self, gA: cute.Tensor, gB_e: cute.Tensor, gSFA_f32: cute.Tensor, gSFB_f32: cute.Tensor,
                 gD: cute.Tensor, offsets: cute.Tensor, stream):
        cfg = self.cfg
        tiledmma = cfg.mma.make_tiled_mma(cfg.TILE)
        E, N, K = cute.size(gB_e, mode=[0]), cute.size(gB_e, mode=[1]), cute.size(gB_e, mode=[2])
        gB = cute.make_tensor(gB_e.iterator, cute.make_layout((N, K, E), stride=(K, 1, N * K)))
        m_padded_sf = cute.size(gSFA_f32, mode=[1])
        # every operand and every scale changes places here and nowhere else
        if cutlass.const_expr(cfg.mma.swap_ab):
            t_a, t_b = gB, gA
            gSFA = cute.make_tensor(gSFB_f32.iterator, cfg.load_sf.deduce_sfa_layout(N, K, E))
            gSFB = cute.make_tensor(gSFA_f32.iterator, cfg.load_sf.deduce_sfb_layout(m_padded_sf, K, 1))
        else:
            t_a, t_b = gA, gB
            gSFA = cute.make_tensor(gSFA_f32.iterator, cfg.load_sf.deduce_sfa_layout(m_padded_sf, K, 1))
            gSFB = cute.make_tensor(gSFB_f32.iterator, cfg.load_sf.deduce_sfb_layout(N, K, E))
        a_layout = utils.LayoutEnum.from_tensor(t_a)
        b_layout = utils.LayoutEnum.from_tensor(t_b)
        assert a_layout.is_k_major_a() and b_layout.is_k_major_b(), "LoadABConfig is k-major only"
        a_smem = cfg.load_ab.make_smem_layout_a()
        b_smem = cfg.load_ab.make_smem_layout_b()
        sfa_smem = cfg.load_sf.make_smem_layout_sfa(cfg.TILE[0], cfg.ab_stage)
        sfa_tma_smem = cfg.load_sf.make_tma_smem_layout_sfa(cfg.TILE[0], cfg.ab_stage)
        sfb_smem = cfg.load_sf.make_smem_layout_sfb(cfg.TILE[1], cfg.ab_stage)
        epi_smem = cfg.store.make_smem_layout(cfg.TILE)
        self.a_is_m_major = a_layout.is_m_major_a()
        self.b_is_n_major = b_layout.is_n_major_b()

        tma_atom_a, tma_tensor_a = cfg.load_ab.make_tma_atom_a(t_a, a_smem)
        tma_atom_b, tma_tensor_b = cfg.load_ab.make_tma_atom_b(t_b, b_smem)
        tscale_m = cfg.TILE[0] // cfg.load_sf.gran_m
        # swapped, one M granule is one scale and a 4-byte TMA box is unrepresentable, so SFA drops to cp.async
        if cutlass.const_expr(cfg.mma.swap_ab):
            tma_atom_sfa, tma_tensor_sfa = None, None
        else:
            tma_atom_sfa, tma_tensor_sfa = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileG2SOp(), gSFA, cute.slice_(sfa_tma_smem, (None, None, 0)), (tscale_m, 1), num_multicast=1)
        self.ab_bytes = cfg.load_ab.tma_bytes_ab
        self.sfa_bytes = cute.size_in_bytes(cfg.load_sf.sf_dtype, cute.slice_(sfa_tma_smem, (None, None, 0)))

        # what the store method does not use is not allocated
        store_full_bars = cfg.store_stages if cfg.store.HAS_STORE_WARP else 0
        store_empty_bars = 0 if cfg.store.METHOD is StoreMethod.DIRECT_STG else cfg.store_stages
        epi_elems = cute.cosize(epi_smem) if cfg.store.METHOD is StoreMethod.STAGED_R2G else 0

        @cute.struct
        class SharedStorage:
            ab_full: cute.struct.MemRange[cfg.I64, cfg.ab_stage]
            ab_empty: cute.struct.MemRange[cfg.I64, cfg.ab_stage]
            sf_full: cute.struct.MemRange[cfg.I64, cfg.ab_stage]
            sf_empty: cute.struct.MemRange[cfg.I64, cfg.ab_stage]
            store_full: cute.struct.MemRange[cfg.I64, store_full_bars]
            store_empty: cute.struct.MemRange[cfg.I64, store_empty_bars]
            sfull: cute.struct.MemRange[cfg.I64, cfg.sched_stages]
            sempty: cute.struct.MemRange[cfg.I64, cfg.sched_stages]
            work: cute.struct.MemRange[cfg.I32, cfg.sched_stages * cfg.fields]
            sA: cute.struct.Align[cute.struct.MemRange[cfg.load_ab.a_dtype, cute.cosize(a_smem)], 128]
            sB: cute.struct.Align[cute.struct.MemRange[cfg.load_ab.b_smem_dtype, cute.cosize(b_smem)], 128]
            sSFA: cute.struct.Align[cute.struct.MemRange[cfg.load_sf.sf_dtype, cute.cosize(sfa_tma_smem)], 128]
            sSFB: cute.struct.Align[cute.struct.MemRange[cfg.load_sf.sf_dtype, cute.cosize(sfb_smem)], 128]
            sC: cute.struct.Align[cute.struct.MemRange[cfg.store.out_dtype, epi_elems], 128]

        # keep the analytic cost model honest against the struct the kernel really allocates
        assert cfg.smem_bytes <= SharedStorage.__sizeof__() <= cfg.smem_bytes + cfg.MBAR_RESERVE, (
            f"smem model {cfg.smem_bytes} B vs allocated {SharedStorage.__sizeof__()} B")

        self.storage = SharedStorage
        self.kernel(tiledmma, tma_atom_a, tma_atom_b, tma_atom_sfa, tma_tensor_a, tma_tensor_b, tma_tensor_sfa,
                    gSFA, gSFB, gD, offsets, a_smem, b_smem, sfa_smem, sfa_tma_smem, sfb_smem, epi_smem).launch(
            grid=[self.grid_x, 1, 1], block=[cfg.threads, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, tiledmma, tma_atom_a, tma_atom_b, tma_atom_sfa, tma_tensor_a: cute.Tensor,
               tma_tensor_b: cute.Tensor, tma_tensor_sfa, gSFA: cute.Tensor,
               gSFB: cute.Tensor, gD: cute.Tensor,
               offsets: cute.Tensor, a_smem, b_smem, sfa_smem, sfa_tma_smem, sfb_smem, epi_smem):
        cfg = self.cfg
        i32 = cutlass.Int32
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        bidx, _, _ = cute.arch.block_idx()
        swap = cfg.mma.swap_ab
        bm, bn, bk = cfg.TILE[0], cfg.TILE[1], cfg.TILE[2]
        # the scheduler and gD stay in problem space; only the buffers and the MMA are swapped
        pm, pn = (bn, bm) if swap else (bm, bn)
        M, N = cute.size(gD, mode=[0]), cute.size(gD, mode=[1])
        K = cute.size(tma_tensor_a, mode=[1])
        num_groups = cute.size(offsets, mode=[0]) - 1
        num_n_blocks = ceil_div(N, pn)
        k_tile_count = ceil_div(K, bk)

        smem = cutlass.utils.SmemAllocator()
        stg = smem.allocate(self.storage)
        sA = stg.sA.get_tensor(a_smem.outer, swizzle=a_smem.inner)
        sB = stg.sB.get_tensor(b_smem.outer, swizzle=b_smem.inner)
        sSFA = stg.sSFA.get_tensor(sfa_smem)
        if cutlass.const_expr(not cfg.mma.swap_ab):
            sSFA_tma = cute.make_tensor(sSFA.iterator, sfa_tma_smem)
        sSFB = stg.sSFB.get_tensor(sfb_smem)
        if cutlass.const_expr(cfg.store.METHOD is not StoreMethod.DIRECT_STG):
            if cutlass.const_expr(cfg.union_smem):
                sC_stages = cute.make_tensor(
                    cute.recast_ptr(stg.sA.data_ptr(), dtype=cfg.store.out_dtype), epi_smem)
            else:
                sC_stages = stg.sC.get_tensor(epi_smem.outer, swizzle=epi_smem.inner)
            # one epilogue stage today (StoreConfig.epi_stage); the index rotates once there are more
            sC = cute.slice_(sC_stages, (None, None, 0))
        else:
            gD_t = cute.make_tensor(gD.iterator, cute.make_layout((N, M), stride=(1, N)))
        ab_full, ab_empty = stg.ab_full.data_ptr(), stg.ab_empty.data_ptr()
        sf_full, sf_empty = stg.sf_full.data_ptr(), stg.sf_empty.data_ptr()
        # a zero-length MemRange yields the next field's address, so an unused handle stays None
        store_full = stg.store_full.data_ptr() if cutlass.const_expr(cfg.store.HAS_STORE_WARP) else None
        store_empty = (stg.store_empty.data_ptr()
                       if cutlass.const_expr(cfg.store.METHOD is not StoreMethod.DIRECT_STG) else None)
        sfull, sempty = stg.sfull.data_ptr(), stg.sempty.data_ptr()
        sWork = stg.work.get_tensor(cute.make_layout((cfg.sched_stages, cfg.fields)))

        # the copy lanes plus SFA's own TMA arrival, which swap-AB drops by putting SFA on cp.async too
        sf_arrivals = cfg.load_sf.NUM_SCALE_COPY_THREADS + (0 if cfg.mma.swap_ab else 1)
        if warp_idx == 0:
            with cute.arch.elect_one():
                for s in cutlass.range_constexpr(cfg.ab_stage):
                    cute.arch.mbarrier_init(ab_full + s, 1)
                    cute.arch.mbarrier_init(ab_empty + s, cfg.mma_threads)
                    cute.arch.mbarrier_init(sf_full + s, sf_arrivals)
                    cute.arch.mbarrier_init(sf_empty + s, cfg.mma_threads)
                for s in cutlass.range_constexpr(cfg.store_stages):
                    if cutlass.const_expr(cfg.store.HAS_STORE_WARP):
                        cute.arch.mbarrier_init(store_full + s, cfg.mma_threads)
                    if cutlass.const_expr(cfg.store.METHOD is not StoreMethod.DIRECT_STG):
                        cute.arch.mbarrier_init(store_empty + s, cfg.store_threads)
                for s in cutlass.range_constexpr(cfg.sched_stages):
                    cute.arch.mbarrier_init(sfull + s, 1)
                    cute.arch.mbarrier_init(sempty + s, cfg.num_sched_consumers)
        cute.arch.mbarrier_init_fence()
        cute.arch.barrier()

        sf_atom = cute.make_copy_atom(cpasync.CopyG2SOp(), cfg.load_sf.sf_dtype, num_bits_per_copy=32)
        # built before any warp-role branch: a config method call inside one carries cfg across it
        if cutlass.const_expr(cfg.store.METHOD is not StoreMethod.DIRECT_STG):
            st_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cfg.store.out_dtype)
            tiled_st = cfg.store.make_tiled_s2g(st_atom, cfg.TILE)

        if cfg.is_prod_wg(warp_idx):
            cute.arch.setmaxregister_decrease(cfg.reg_prod)

            if warp_idx == cfg.sched_warp:
                sched = scheduler.MoeTileScheduler.create(pm, num_groups, num_n_blocks, self.grid_x, bidx, offsets)
                prod = scheduler.MoeSchedProducer.create(cfg.sched_stages)
                has, e, m_tile, n_tile, m_off, m_bnd = sched.get_next_block(offsets)
                while has:
                    # published in the kernel's own axes, so no consumer carries a swap of its own
                    if cutlass.const_expr(swap):
                        m_tile, n_tile = n_tile, m_tile
                    prod.publish(sWork, sfull, sempty,
                                 scheduler.MoeWorkTile(m_tile, n_tile, e, m_off, m_bnd, i32(1)))
                    has, e, m_tile, n_tile, m_off, m_bnd = sched.get_next_block(offsets)
                prod.publish_sentinel(sWork, sfull, sempty)

            if warp_idx == cfg.ab_warp:
                cons = scheduler.MoeSchedConsumer.create(cfg.sched_stages)
                tile = cons.get_next_tile(sWork, sfull, sempty)
                ab_stage, ab_phase, store_phase = i32(0), i32(1), i32(1)
                while tile.valid != i32(0):
                    # sC overlays sA, and ab_empty alone would let this warp refill it mid-epilogue
                    if cutlass.const_expr(cfg.union_smem):
                        cute.arch.mbarrier_wait(store_empty, store_phase)
                        store_phase ^= 1
                    loader = load_ab_swap if cutlass.const_expr(swap) else load_ab
                    ab_stage, ab_phase = loader(
                        tma_atom_a, tma_atom_b, tma_tensor_a, tma_tensor_b, sA, sB, cfg.TILE, tile,
                        ab_full, ab_empty, self.ab_bytes, k_tile_count, cfg.ab_stage,
                        ab_stage, ab_phase)
                    tile = cons.get_next_tile(sWork, sfull, sempty)

            if warp_idx == cfg.sf_warp:
                cons = scheduler.MoeSchedConsumer.create(cfg.sched_stages)
                tile = cons.get_next_tile(sWork, sfull, sempty)
                sf_stage, sf_phase = i32(0), i32(1)
                while tile.valid != i32(0):
                    if cutlass.const_expr(cfg.mma.swap_ab):
                        sf_stage, sf_phase = load_sf(
                            gSFA, sSFA, gSFB, sSFB, sf_atom, tile, sf_full, sf_empty,
                            cfg.load_sf.NUM_SCALE_COPY_THREADS, k_tile_count, self.sf_stages,
                            sf_stage, sf_phase)
                    else:
                        sf_stage, sf_phase = load_sf_tma(
                            tma_atom_sfa, tma_tensor_sfa, sSFA_tma, sf_atom, gSFB, sSFB, tile,
                            sf_full, sf_empty, self.sfa_bytes, cfg.load_sf.NUM_SCALE_COPY_THREADS,
                            k_tile_count, self.sf_stages, sf_stage, sf_phase)
                    tile = cons.get_next_tile(sWork, sfull, sempty)

            if cutlass.const_expr(cfg.store.HAS_STORE_WARP):
                if warp_idx == cfg.store_warp:
                    store_tid = tidx - cfg.store_warp * 32
                    thr_st = tiled_st.get_slice(store_tid)
                    cons = scheduler.MoeSchedConsumer.create(cfg.sched_stages)
                    tile = cons.get_next_tile(sWork, sfull, sempty)
                    store_cons_phase = i32(0)
                    while tile.valid != i32(0):
                        cute.arch.mbarrier_wait(store_full, store_cons_phase)
                        gD_tile = cute.local_tile(cute.domain_offset((tile.m_offset, 0), gD), (bm, bn), (tile.m_block, tile.n_block))
                        store_output.store_predicated(
                            st_atom, thr_st, store_output.smem_to_reg(thr_st, sC), gD_tile, (bm, bn),
                            tile.m_offset + tile.m_block * bm, tile.n_block * bn, tile.m_boundary, N)
                        cute.arch.mbarrier_arrive(store_empty)
                        store_cons_phase ^= 1
                        tile = cons.get_next_tile(sWork, sfull, sempty)

        else:
            cute.arch.setmaxregister_increase(cfg.reg_math)
            thr = tiledmma.get_slice(tidx)
            if cutlass.const_expr(cfg.store.METHOD is StoreMethod.R2G_WG):
                thr_st = tiled_st.get_slice(tidx)
            cons = scheduler.MoeSchedConsumer.create(cfg.sched_stages)
            tile = cons.get_next_tile(sWork, sfull, sempty)
            read_stage, ab_phase, store_phase = i32(0), i32(0), i32(1)
            while tile.valid != i32(0):
                acc, read_stage, ab_phase = mma(
                    tiledmma, cfg.mma, cfg.load_sf, sA, sB, sSFA, sSFB, (bm, bn),
                    cfg.ACC, cfg.load_ab.a_dtype, cfg.load_ab.b_smem_dtype,
                    self.a_is_m_major, self.b_is_n_major, tidx,
                    ab_full, ab_empty, sf_full, sf_empty,
                    k_tile_count, cfg.ab_stage, read_stage, ab_phase)
                if cutlass.const_expr(cfg.store.METHOD is StoreMethod.DIRECT_STG):
                    store_swap(acc, thr, gD_t, tile, (bm, bn), N, cfg.store.out_dtype)
                elif cutlass.const_expr(cfg.store.HAS_STORE_WARP):
                    store_phase = store_staged(acc, thr, sC, cfg.store.out_dtype,
                                               store_full, store_empty, store_phase)
                else:
                    store_wg(acc, thr, thr_st, sC, st_atom, gD, tile, (bm, bn), N,
                             cfg.store.out_dtype, store_empty, cfg.epi_bar_id, cfg.mma_threads)
                tile = cons.get_next_tile(sWork, sfull, sempty)


def _stream():
    return cuda.CUstream(torch.cuda.current_stream().cuda_stream)


def make_args(a_q, a_scale, b_q, b_scale, out, m_indptr):
    sf = lambda t: from_dlpack(t.contiguous(), assumed_align=16).mark_layout_dynamic()
    return (from_dlpack(a_q).mark_layout_dynamic(), from_dlpack(b_q).mark_layout_dynamic(),
            sf(a_scale), sf(b_scale), from_dlpack(out).mark_layout_dynamic(),
            from_dlpack(m_indptr).mark_layout_dynamic(), _stream())
