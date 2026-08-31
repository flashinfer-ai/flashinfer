# Copyright (c) 2025 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Self-written CuteDSL fp8 fused fc1_gate_up + SiLU activation for SM120a."""

import torch

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.warp.mma as warp_mma
import cutlass.utils as utils
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import from_dlpack

from ._moe_utils.moe_epilogue import EPI_CONFIGS, EpiMethod
from ._moe_utils.moe_kernel_builder import FC1ActBuilder, MmaConfig, LoadABConfig
from ._moe_utils.sm12x_blockscaled_layout import (
    Sm120SfConfigFp8,
    copy_scale_s2r,
    rescale,
)
from ....tllm_enums import (
    DEFAULT_SITU_BETA as SITU_BETA,
    DEFAULT_SITU_LINEAR_BETA as SITU_LINEAR_BETA,
)
from ._moe_utils.sm12x_blockscaled_layout import TMA_ALIGN_BYTES
from ....utils import ceil_div
from ._moe_utils.sm12x_blockscaled_layout import compute_padded_offset
from ._moe_utils import moe_activation, moe_scheduler, moe_epilogue


GRAN_M, GRAN_N, GRAN_K = 1, 128, 128

ATOM_MNK = (16, 8, 32)


def is_swapab(tile):
    return ATOM_MNK[0] > tile[0]


REG_PROD_BY_TACTIC = {
    (128, 64, EpiMethod.R2G_WG): 40,
    (64, 128, EpiMethod.R2G_WG): 40,
    (32, 128, EpiMethod.R2G_WG): 40,
    (8, 128, EpiMethod.DIRECT_STG): 40,
}


def make_cfg(
    tile,
    ab_stage,
    epi=EpiMethod.R2G_WG,
    *,
    activation,
    fastmath=False,
    enable_pdl=False,
    situ_beta=SITU_BETA,
    situ_linear_beta=SITU_LINEAR_BETA,
):
    e4m3, f32, bf16 = cutlass.Float8E4M3FN, cutlass.Float32, cutlass.BFloat16
    num_math_warps = 8
    bm, bn, bk = tile
    ptile = (bm, bn)
    swap = is_swapab(tile)
    if swap:
        bm, bn = bn, bm
        gran_m, gran_n = GRAN_N, GRAN_M
        epi = EpiMethod.DIRECT_STG
    else:
        assert epi is EpiMethod.R2G_WG, f"{epi} is not implemented here; sC aliases A/B"
        gran_m, gran_n = GRAN_M, GRAN_N
    tile = (bm, bn, bk)
    return FC1ActBuilder(
        MmaConfig(
            warp_mma.MmaFP8Op(e4m3, f32, ATOM_MNK),
            tile[:2],
            num_math_warps,
            swap_ab=swap,
        ),
        LoadABConfig(tile, ab_stage, e4m3, e4m3),
        Sm120SfConfigFp8(gran_m, gran_n, GRAN_K, tile_n=tile[1]),
        EPI_CONFIGS[epi](bf16, num_math_warps * 32),
        ab_stage,
        tile,
        epi_bar_id=3,
        union_smem=not swap,
        reg_prod=REG_PROD_BY_TACTIC[(*ptile, epi)],
        activation=moe_activation.resolve_activation_fn(
            activation, situ_beta, situ_linear_beta
        ),
        fastmath=fastmath,
        enable_pdl=enable_pdl,
    )


@cute.jit
def load_ab(
    tma_atom_a,
    tma_atom_b,
    tma_tensor_a,
    tma_tensor_b,
    sA,
    sB,
    sB_g,
    tile_mnk,
    tile,
    ab_full,
    ab_empty,
    ab_bytes,
    k_tile_count,
    ab_stages,
    ab_stage,
    ab_phase,
    n_gate_off,
):
    i32 = cutlass.Int32
    cluster = cute.make_layout((1, 1, 1))
    multicast = cute.make_layout(cute.slice_(cluster, (0, None, 0)).shape)
    mA = cute.domain_offset((tile.m_offset, 0), tma_tensor_a)
    gA_mkl = cute.local_tile(mA, cute.slice_(tile_mnk, (None, 0, None)), (None, None))
    tAsA, tAgA = cpasync.tma_partition(
        tma_atom_a,
        i32(0),
        multicast,
        cute.group_modes(sA, 0, 2),
        cute.group_modes(gA_mkl, 0, 2),
    )
    gB_nkl = cute.local_tile(
        tma_tensor_b, cute.slice_(tile_mnk, (0, None, None)), (None, None, None)
    )
    tBsB, tBgB = cpasync.tma_partition(
        tma_atom_b,
        i32(0),
        multicast,
        cute.group_modes(sB, 0, 2),
        cute.group_modes(gB_nkl, 0, 2),
    )
    mB_g = cute.domain_offset((n_gate_off, 0, 0), tma_tensor_b)
    gG_nkl = cute.local_tile(
        mB_g, cute.slice_(tile_mnk, (0, None, None)), (None, None, None)
    )
    tGsG, tGgG = cpasync.tma_partition(
        tma_atom_b,
        i32(0),
        multicast,
        cute.group_modes(sB_g, 0, 2),
        cute.group_modes(gG_nkl, 0, 2),
    )
    for k_tile_idx in cutlass.range(0, k_tile_count):
        cute.arch.mbarrier_wait(ab_empty + ab_stage, ab_phase)
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(ab_full + ab_stage, ab_bytes)
        cute.copy(
            tma_atom_a,
            tAgA[(None, tile.m_block, k_tile_idx)],
            tAsA[(None, ab_stage)],
            tma_bar_ptr=ab_full + ab_stage,
        )
        cute.copy(
            tma_atom_b,
            tBgB[(None, tile.n_block, k_tile_idx, tile.group)],
            tBsB[(None, ab_stage)],
            tma_bar_ptr=ab_full + ab_stage,
        )
        cute.copy(
            tma_atom_b,
            tGgG[(None, tile.n_block, k_tile_idx, tile.group)],
            tGsG[(None, ab_stage)],
            tma_bar_ptr=ab_full + ab_stage,
        )
        ab_stage += 1
        if ab_stage == ab_stages:
            ab_stage = i32(0)
            ab_phase ^= 1
    return ab_stage, ab_phase


@cute.jit
def load_sf_tma(
    tma_atom_sfa,
    tma_tensor_sfa,
    sSFA_tma,
    sf_atom,
    gSFB,
    sSFB,
    sSFB_g,
    tile,
    sf_full,
    sf_empty,
    sfa_bytes,
    num_copy_threads,
    k_tile_count,
    sf_stages,
    sf_stage,
    sf_phase,
    pn,
    out_n,
    gran_n,
):
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
    tSAs, tSAg = cpasync.tma_partition(
        tma_atom_sfa,
        i32(0),
        multicast,
        cute.group_modes(sSFA_tma, 0, 2),
        cute.group_modes(gSFA_ml_t, 0, 2),
    )

    cSFB = cute.make_identity_tensor(gSFB.shape)
    n_offset = tile.n_block * i32(pn)
    up_coord = n_offset // i32(gran_n)
    gSFB_nk = cute.local_tile(gSFB, (tscale_n,), (up_coord, None, tile.group))
    coordSFB = cute.local_tile(cSFB, (tscale_n,), (up_coord, None, tile.group))
    scale_copy_b = cute.make_tiled_copy_tv(
        sf_atom, cute.make_layout(num_copy_threads), cute.make_layout(1)
    )
    thr_scale_copy_b = scale_copy_b.get_slice(lane_idx)
    tBgSFB = thr_scale_copy_b.partition_S(gSFB_nk)
    tBcSFB = thr_scale_copy_b.partition_S(coordSFB)
    tBsSFB = thr_scale_copy_b.partition_D(sSFB)
    tBpSFB = cute.make_fragment_like(tBcSFB[None, None, 0], cutlass.Boolean)
    for i in cutlass.range_constexpr(cute.size(tBpSFB)):
        tBpSFB[i] = tBcSFB[i][0] < scale_n
    gate_coord = (i32(out_n) + n_offset) // i32(gran_n)
    gSFG_nk = cute.local_tile(gSFB, (tscale_n,), (gate_coord, None, tile.group))
    tGgSFG = thr_scale_copy_b.partition_S(gSFG_nk)
    tGsSFG = thr_scale_copy_b.partition_D(sSFB_g)

    for k_tile_idx in cutlass.range(0, k_tile_count):
        cute.arch.mbarrier_wait(sf_empty + sf_stage, sf_phase)
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(sf_full + sf_stage, sfa_bytes)
        cute.copy(
            tma_atom_sfa,
            tSAg[(None, tile.m_block, k_tile_idx, 0)],
            tSAs[(None, sf_stage)],
            tma_bar_ptr=sf_full + sf_stage,
        )
        cute.arch.sync_warp()
        if lane_idx < tscale_n:
            cute.copy(
                scale_copy_b,
                tBgSFB[None, None, k_tile_idx],
                tBsSFB[None, None, sf_stage],
                pred=tBpSFB,
            )
            cute.copy(
                scale_copy_b,
                tGgSFG[None, None, k_tile_idx],
                tGsSFG[None, None, sf_stage],
                pred=tBpSFB,
            )
        cute.arch.cp_async_mbarrier_arrive_noinc(sf_full + sf_stage)
        sf_stage += 1
        if sf_stage == sf_stages:
            sf_stage = i32(0)
            sf_phase ^= 1
    return sf_stage, sf_phase


@cute.jit
def mma(
    tiledmma,
    mma_cfg,
    sf_cfg,
    activation,
    fastmath,
    sA,
    sB,
    sB_g,
    sSFA,
    sSFB,
    sSFB_g,
    tile_mn,
    acc_dtype,
    a_dtype,
    b_smem_dtype,
    tidx,
    ab_full,
    ab_empty,
    sf_full,
    sf_empty,
    k_tile_count,
    ab_stages,
    read_stage,
    ab_phase,
):
    i32 = cutlass.Int32
    bm, bn = tile_mn
    thr = tiledmma.get_slice(tidx)
    tCrA = tiledmma.make_fragment_A(thr.partition_A(sA)[None, None, None, 0])
    tCrU = tiledmma.make_fragment_B(thr.partition_B(sB)[None, None, None, 0])
    tCrG = tiledmma.make_fragment_B(thr.partition_B(sB_g)[None, None, None, 0])
    shape_c = tiledmma.partition_shape_C((bm, bn))
    acc_u = cute.make_rmem_tensor(shape_c, acc_dtype)
    tmp_u = cute.make_rmem_tensor(shape_c, acc_dtype)
    acc_g = cute.make_rmem_tensor(shape_c, acc_dtype)
    tmp_g = cute.make_rmem_tensor(shape_c, acc_dtype)
    s2r_a = mma_cfg.make_s2r_a(tiledmma, a_dtype, False)
    s2r_b = mma_cfg.make_s2r_b(tiledmma, b_smem_dtype, False)
    thr_a, thr_b = s2r_a.get_slice(tidx), s2r_b.get_slice(tidx)
    tCrA_v, tCrU_v, tCrG_v = thr_a.retile(tCrA), thr_b.retile(tCrU), thr_b.retile(tCrG)
    tCsSFA, tCsSFU, tCrSFA_u, tCrSFU = sf_cfg.partition_scale_as_c(sSFA, sSFB, thr)
    _, tCsSFG, tCrSFA_g, tCrSFG = sf_cfg.partition_scale_as_c(sSFA, sSFB_g, thr)
    tscale_mn = (cute.size(sSFA, mode=[0]), cute.size(sSFB, mode=[0]))
    tXsA, tXsU, tXsG = (
        thr_a.partition_S(sA),
        thr_b.partition_S(sB),
        thr_b.partition_S(sB_g),
    )
    k_blocks = cute.size(tCrA_v, mode=[2])

    acc_u.fill(0.0)
    tmp_u.fill(0.0)
    acc_g.fill(0.0)
    tmp_g.fill(0.0)

    for _k_tile_idx in cutlass.range(0, k_tile_count):
        cute.arch.mbarrier_wait(sf_full + read_stage, ab_phase)
        copy_scale_s2r(read_stage, tCsSFA, tCsSFU, tCrSFA_u, tCrSFU, tscale_mn)
        copy_scale_s2r(read_stage, tCsSFA, tCsSFG, tCrSFA_g, tCrSFG, tscale_mn)
        cute.arch.mbarrier_arrive(sf_empty + read_stage)
        cute.arch.mbarrier_wait(ab_full + read_stage, ab_phase)
        tAsA_s = tXsA[None, None, None, read_stage]
        tUsU_s = tXsU[None, None, None, read_stage]
        tGsG_s = tXsG[None, None, None, read_stage]
        for k_block in cutlass.range_constexpr(0, k_blocks):
            cute.copy(s2r_a, tAsA_s[None, None, k_block], tCrA_v[None, None, k_block])
            cute.copy(s2r_b, tUsU_s[None, None, k_block], tCrU_v[None, None, k_block])
            cute.copy(s2r_b, tGsG_s[None, None, k_block], tCrG_v[None, None, k_block])
            cute.gemm(
                tiledmma,
                tmp_u,
                tCrA[None, None, k_block],
                tCrU[None, None, k_block],
                tmp_u,
            )
            cute.gemm(
                tiledmma,
                tmp_g,
                tCrA[None, None, k_block],
                tCrG[None, None, k_block],
                tmp_g,
            )
        cute.arch.mbarrier_arrive(ab_empty + read_stage)
        rescale(acc_u, tmp_u, tCrSFA_u, tCrSFU, tscale_mn)
        rescale(acc_g, tmp_g, tCrSFA_g, tCrSFG, tscale_mn)
        read_stage += 1
        if read_stage == ab_stages:
            read_stage = i32(0)
            ab_phase ^= 1

    activation(acc_u, acc_g, fastmath)
    return acc_u, read_stage, ab_phase


@cute.jit
def load_ab_swap(
    tma_atom_a,
    tma_atom_b,
    tma_tensor_a,
    tma_tensor_b,
    sA,
    sA_g,
    sB,
    tile_mnk,
    tile,
    ab_full,
    ab_empty,
    ab_bytes,
    k_tile_count,
    ab_stages,
    ab_stage,
    ab_phase,
    n_gate_off,
):
    i32 = cutlass.Int32
    cluster = cute.make_layout((1, 1, 1))
    multicast = cute.make_layout(cute.slice_(cluster, (0, None, 0)).shape)
    gA_mkl = cute.local_tile(
        tma_tensor_a, cute.slice_(tile_mnk, (None, 0, None)), (None, None, None)
    )
    tAsA, tAgA = cpasync.tma_partition(
        tma_atom_a,
        i32(0),
        multicast,
        cute.group_modes(sA, 0, 2),
        cute.group_modes(gA_mkl, 0, 2),
    )
    mA_g = cute.domain_offset((n_gate_off, 0, 0), tma_tensor_a)
    gG_mkl = cute.local_tile(
        mA_g, cute.slice_(tile_mnk, (None, 0, None)), (None, None, None)
    )
    tGsG, tGgG = cpasync.tma_partition(
        tma_atom_a,
        i32(0),
        multicast,
        cute.group_modes(sA_g, 0, 2),
        cute.group_modes(gG_mkl, 0, 2),
    )
    mB = cute.domain_offset((tile.m_offset, 0), tma_tensor_b)
    gB_nkl = cute.local_tile(mB, cute.slice_(tile_mnk, (0, None, None)), (None, None))
    tBsB, tBgB = cpasync.tma_partition(
        tma_atom_b,
        i32(0),
        multicast,
        cute.group_modes(sB, 0, 2),
        cute.group_modes(gB_nkl, 0, 2),
    )
    for k_tile_idx in cutlass.range(0, k_tile_count):
        cute.arch.mbarrier_wait(ab_empty + ab_stage, ab_phase)
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(ab_full + ab_stage, ab_bytes)
        cute.copy(
            tma_atom_a,
            tAgA[(None, tile.m_block, k_tile_idx, tile.group)],
            tAsA[(None, ab_stage)],
            tma_bar_ptr=ab_full + ab_stage,
        )
        cute.copy(
            tma_atom_a,
            tGgG[(None, tile.m_block, k_tile_idx, tile.group)],
            tGsG[(None, ab_stage)],
            tma_bar_ptr=ab_full + ab_stage,
        )
        cute.copy(
            tma_atom_b,
            tBgB[(None, tile.n_block, k_tile_idx)],
            tBsB[(None, ab_stage)],
            tma_bar_ptr=ab_full + ab_stage,
        )
        ab_stage += 1
        if ab_stage == ab_stages:
            ab_stage = i32(0)
            ab_phase ^= 1
    return ab_stage, ab_phase


@cute.jit
def load_sf_swap(
    gSFA,
    sSFA,
    sSFA_g,
    gSFB,
    sSFB,
    sf_atom,
    tile,
    gran_n,
    sf_full,
    sf_empty,
    num_copy_threads,
    k_tile_count,
    sf_stages,
    sf_stage,
    sf_phase,
    n_gate_blocks,
):
    i32 = cutlass.Int32
    tscale_m, tscale_n = cute.size(sSFA, mode=[0]), cute.size(sSFB, mode=[0])
    scale_m = cute.size(gSFA, mode=[0])
    m_align = TMA_ALIGN_BYTES // (sSFB.element_type.width // 8)
    lane_idx = cute.arch.lane_idx()

    sf_m_off = compute_padded_offset(tile.m_offset, tile.group, i32(m_align))
    scale_n = sf_m_off + ceil_div(tile.m_boundary - tile.m_offset, gran_n)
    mSFB = cute.domain_offset((sf_m_off, 0, 0), gSFB)
    cSFA = cute.make_identity_tensor(gSFA.shape)
    cSFB = cute.domain_offset((sf_m_off, 0, 0), cute.make_identity_tensor(gSFB.shape))

    gSFA_mk = cute.local_tile(gSFA, (tscale_m,), (tile.m_block, None, tile.group))
    coordSFA = cute.local_tile(cSFA, (tscale_m,), (tile.m_block, None, tile.group))
    gSFG_mk = cute.local_tile(
        gSFA, (tscale_m,), (tile.m_block + n_gate_blocks, None, tile.group)
    )
    coordSFG = cute.local_tile(
        cSFA, (tscale_m,), (tile.m_block + n_gate_blocks, None, tile.group)
    )
    gSFB_nk = cute.local_tile(mSFB, (tscale_n,), (tile.n_block, None, 0))
    coordSFB = cute.local_tile(cSFB, (tscale_n,), (tile.n_block, None, 0))

    scale_copy = cute.make_tiled_copy_tv(
        sf_atom, cute.make_layout(num_copy_threads), cute.make_layout(1)
    )
    thr = scale_copy.get_slice(lane_idx)
    tAgSFA, tAcSFA, tAsSFA = (
        thr.partition_S(gSFA_mk),
        thr.partition_S(coordSFA),
        thr.partition_D(sSFA),
    )
    tGgSFG, tGcSFG, tGsSFG = (
        thr.partition_S(gSFG_mk),
        thr.partition_S(coordSFG),
        thr.partition_D(sSFA_g),
    )
    tBgSFB, tBcSFB, tBsSFB = (
        thr.partition_S(gSFB_nk),
        thr.partition_S(coordSFB),
        thr.partition_D(sSFB),
    )
    tApSFA = cute.make_fragment_like(tAcSFA[None, None, 0], cutlass.Boolean)
    tGpSFG = cute.make_fragment_like(tGcSFG[None, None, 0], cutlass.Boolean)
    tBpSFB = cute.make_fragment_like(tBcSFB[None, None, 0], cutlass.Boolean)
    for i in cutlass.range_constexpr(cute.size(tApSFA)):
        tApSFA[i] = tAcSFA[i][0] < scale_m
        tGpSFG[i] = tGcSFG[i][0] < scale_m
    for i in cutlass.range_constexpr(cute.size(tBpSFB)):
        tBpSFB[i] = tBcSFB[i][0] < scale_n

    for k_tile_idx in cutlass.range(0, k_tile_count):
        cute.arch.mbarrier_wait(sf_empty + sf_stage, sf_phase)
        if lane_idx < tscale_m:
            cute.copy(
                scale_copy,
                tAgSFA[None, None, k_tile_idx],
                tAsSFA[None, None, sf_stage],
                pred=tApSFA,
            )
            cute.copy(
                scale_copy,
                tGgSFG[None, None, k_tile_idx],
                tGsSFG[None, None, sf_stage],
                pred=tGpSFG,
            )
        if lane_idx < tscale_n:
            cute.copy(
                scale_copy,
                tBgSFB[None, None, k_tile_idx],
                tBsSFB[None, None, sf_stage],
                pred=tBpSFB,
            )
        cute.arch.cp_async_mbarrier_arrive_noinc(sf_full + sf_stage)
        sf_stage += 1
        if sf_stage == sf_stages:
            sf_stage = i32(0)
            sf_phase ^= 1
    return sf_stage, sf_phase


@cute.jit
def mma_swap(
    tiledmma,
    mma_cfg,
    sf_cfg,
    activation,
    fastmath,
    sA,
    sA_g,
    sB,
    sSFA,
    sSFA_g,
    sSFB,
    tile_mn,
    acc_dtype,
    a_dtype,
    b_smem_dtype,
    tidx,
    ab_full,
    ab_empty,
    sf_full,
    sf_empty,
    k_tile_count,
    ab_stages,
    read_stage,
    ab_phase,
):
    i32 = cutlass.Int32
    bm, bn = tile_mn
    thr = tiledmma.get_slice(tidx)
    tCrU = tiledmma.make_fragment_A(thr.partition_A(sA)[None, None, None, 0])
    tCrG = tiledmma.make_fragment_A(thr.partition_A(sA_g)[None, None, None, 0])
    tCrB = tiledmma.make_fragment_B(thr.partition_B(sB)[None, None, None, 0])
    shape_c = tiledmma.partition_shape_C((bm, bn))
    acc_u = cute.make_rmem_tensor(shape_c, acc_dtype)
    tmp_u = cute.make_rmem_tensor(shape_c, acc_dtype)
    acc_g = cute.make_rmem_tensor(shape_c, acc_dtype)
    tmp_g = cute.make_rmem_tensor(shape_c, acc_dtype)
    s2r_a = mma_cfg.make_s2r_a(tiledmma, a_dtype, False)
    s2r_b = mma_cfg.make_s2r_b(tiledmma, b_smem_dtype, False)
    thr_a, thr_b = s2r_a.get_slice(tidx), s2r_b.get_slice(tidx)
    tCrU_v, tCrG_v, tCrB_v = thr_a.retile(tCrU), thr_a.retile(tCrG), thr_b.retile(tCrB)
    tCsSFA_u, tCsSFB, tCrSFA_u, tCrSFB_u = sf_cfg.partition_scale_as_c(sSFA, sSFB, thr)
    tCsSFA_g, _, tCrSFA_g, tCrSFB_g = sf_cfg.partition_scale_as_c(sSFA_g, sSFB, thr)
    tscale_mn = (cute.size(sSFA, mode=[0]), cute.size(sSFB, mode=[0]))
    tXsU, tXsG, tXsB = (
        thr_a.partition_S(sA),
        thr_a.partition_S(sA_g),
        thr_b.partition_S(sB),
    )
    k_blocks = cute.size(tCrU_v, mode=[2])

    acc_u.fill(0.0)
    tmp_u.fill(0.0)
    acc_g.fill(0.0)
    tmp_g.fill(0.0)

    for _k_tile_idx in cutlass.range(0, k_tile_count):
        cute.arch.mbarrier_wait(sf_full + read_stage, ab_phase)
        copy_scale_s2r(read_stage, tCsSFA_u, tCsSFB, tCrSFA_u, tCrSFB_u, tscale_mn)
        copy_scale_s2r(read_stage, tCsSFA_g, tCsSFB, tCrSFA_g, tCrSFB_g, tscale_mn)
        cute.arch.mbarrier_arrive(sf_empty + read_stage)
        cute.arch.mbarrier_wait(ab_full + read_stage, ab_phase)
        tUsU_s = tXsU[None, None, None, read_stage]
        tGsG_s = tXsG[None, None, None, read_stage]
        tBsB_s = tXsB[None, None, None, read_stage]
        for k_block in cutlass.range_constexpr(0, k_blocks):
            cute.copy(s2r_a, tUsU_s[None, None, k_block], tCrU_v[None, None, k_block])
            cute.copy(s2r_a, tGsG_s[None, None, k_block], tCrG_v[None, None, k_block])
            cute.copy(s2r_b, tBsB_s[None, None, k_block], tCrB_v[None, None, k_block])
            cute.gemm(
                tiledmma,
                tmp_u,
                tCrU[None, None, k_block],
                tCrB[None, None, k_block],
                tmp_u,
            )
            cute.gemm(
                tiledmma,
                tmp_g,
                tCrG[None, None, k_block],
                tCrB[None, None, k_block],
                tmp_g,
            )
        cute.arch.mbarrier_arrive(ab_empty + read_stage)
        rescale(acc_u, tmp_u, tCrSFA_u, tCrSFB_u, tscale_mn)
        rescale(acc_g, tmp_g, tCrSFA_g, tCrSFB_g, tscale_mn)
        read_stage += 1
        if read_stage == ab_stages:
            read_stage = i32(0)
            ab_phase ^= 1

    activation(acc_u, acc_g, fastmath)
    return acc_u, read_stage, ab_phase


class CuteDslSm120MoeFp8Fc1Act:
    def __init__(self, cfg, grid_x):
        self.cfg = cfg
        self.mma = cfg.mma
        self.scale = cfg.load_sf
        self.scale.assert_k_invariant(cfg.TILE[2])
        self.scale.assert_mn_invariant(cfg.TILE[0], cfg.TILE[1])
        self.sf_stages = self.scale.sf_stages(cfg.ab_stage, cfg.TILE[2])
        assert self.sf_stages == cfg.ab_stage, (
            f"SF ring is {self.sf_stages} deep against {cfg.ab_stage} for A/B, but the math warp "
            f"indexes both with one stage counter"
        )
        self.grid_x = grid_x

    @cute.jit
    def __call__(
        self,
        gA: cute.Tensor,
        gB_e: cute.Tensor,
        gSFA_f32: cute.Tensor,
        gSFB_f32: cute.Tensor,
        gD: cute.Tensor,
        offsets: cute.Tensor,
        stream,
    ):
        cfg = self.cfg
        tiledmma = cfg.mma.make_tiled_mma(cfg.TILE)
        E, N, K = (
            cute.size(gB_e, mode=[0]),
            cute.size(gB_e, mode=[1]),
            cute.size(gB_e, mode=[2]),
        )
        gB = cute.make_tensor(
            gB_e.iterator, cute.make_layout((N, K, E), stride=(K, 1, N * K))
        )
        m_padded_sf = cute.size(gSFA_f32, mode=[1])
        if cutlass.const_expr(cfg.mma.swap_ab):
            t_a, t_b = gB, gA
            gSFA = cute.make_tensor(
                gSFB_f32.iterator, cfg.load_sf.deduce_sfa_layout(N, K, E)
            )
            gSFB = cute.make_tensor(
                gSFA_f32.iterator, cfg.load_sf.deduce_sfb_layout(m_padded_sf, K, 1)
            )
        else:
            t_a, t_b = gA, gB
            gSFA = cute.make_tensor(
                gSFA_f32.iterator, cfg.load_sf.deduce_sfa_layout(m_padded_sf, K, 1)
            )
            gSFB = cute.make_tensor(
                gSFB_f32.iterator, cfg.load_sf.deduce_sfb_layout(N, K, E)
            )
        a_layout = utils.LayoutEnum.from_tensor(t_a)
        b_layout = utils.LayoutEnum.from_tensor(t_b)
        assert a_layout.is_k_major_a() and b_layout.is_k_major_b(), (
            "LoadABConfig is k-major only"
        )
        a_smem = cfg.load_ab.make_smem_layout_a()
        b_smem = cfg.load_ab.make_smem_layout_b()
        sfa_smem = cfg.load_sf.make_smem_layout_sfa(cfg.TILE[0], cfg.ab_stage)
        sfa_tma_smem = cfg.load_sf.make_tma_smem_layout_sfa(cfg.TILE[0], cfg.ab_stage)
        sfb_smem = cfg.load_sf.make_smem_layout_sfb(cfg.TILE[1], cfg.ab_stage)
        epi_smem = cfg.epi.make_smem_layout(cfg.TILE)
        self.a_is_m_major = a_layout.is_m_major_a()
        self.b_is_n_major = b_layout.is_n_major_b()

        tma_atom_a, tma_tensor_a = cfg.load_ab.make_tma_atom_a(t_a, a_smem)
        tma_atom_b, tma_tensor_b = cfg.load_ab.make_tma_atom_b(t_b, b_smem)
        tscale_m = cfg.TILE[0] // cfg.load_sf.gran_m
        if cutlass.const_expr(cfg.mma.swap_ab):
            tma_atom_sfa, tma_tensor_sfa = None, None
        else:
            tma_atom_sfa, tma_tensor_sfa = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileG2SOp(),
                gSFA,
                cute.slice_(sfa_tma_smem, (None, None, 0)),
                (tscale_m, 1),
                num_multicast=1,
            )
        self.ab_bytes = cfg.load_ab.tma_bytes_ab + (
            cfg.load_ab.tma_bytes_a if cfg.mma.swap_ab else cfg.load_ab.tma_bytes_b
        )
        self.sfa_bytes = cute.size_in_bytes(
            cfg.load_sf.sf_dtype, cute.slice_(sfa_tma_smem, (None, None, 0))
        )

        store_full_bars = cfg.store_stages if cfg.epi.HAS_STORE_WARP else 0
        store_empty_bars = (
            0 if cfg.epi.METHOD is EpiMethod.DIRECT_STG else cfg.store_stages
        )
        epi_elems = (
            cute.cosize(epi_smem) if cfg.epi.METHOD is EpiMethod.STAGED_R2G else 0
        )
        gate_b_elems = cute.cosize(a_smem if cfg.mma.swap_ab else b_smem)
        gate_sf_elems = cute.cosize(sfa_smem if cfg.mma.swap_ab else sfb_smem)

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
            sA: cute.struct.Align[
                cute.struct.MemRange[cfg.load_ab.a_dtype, cute.cosize(a_smem)], 128
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[cfg.load_ab.b_smem_dtype, cute.cosize(b_smem)], 128
            ]
            sBg: cute.struct.Align[
                cute.struct.MemRange[cfg.load_ab.b_smem_dtype, gate_b_elems], 128
            ]
            sSFA: cute.struct.Align[
                cute.struct.MemRange[cfg.load_sf.sf_dtype, cute.cosize(sfa_tma_smem)],
                128,
            ]
            sSFB: cute.struct.Align[
                cute.struct.MemRange[cfg.load_sf.sf_dtype, cute.cosize(sfb_smem)], 128
            ]
            sSFBg: cute.struct.Align[
                cute.struct.MemRange[cfg.load_sf.sf_dtype, gate_sf_elems], 128
            ]
            sC: cute.struct.Align[
                cute.struct.MemRange[cfg.epi.out_dtype, epi_elems], 128
            ]

        assert (
            cfg.smem_bytes
            <= SharedStorage.__sizeof__()
            <= cfg.smem_bytes + cfg.MBAR_RESERVE
        ), f"smem model {cfg.smem_bytes} B vs allocated {SharedStorage.__sizeof__()} B"

        self.storage = SharedStorage
        self.kernel(
            tiledmma,
            tma_atom_a,
            tma_atom_b,
            tma_atom_sfa,
            tma_tensor_a,
            tma_tensor_b,
            tma_tensor_sfa,
            gSFA,
            gSFB,
            gD,
            offsets,
            a_smem,
            b_smem,
            sfa_smem,
            sfa_tma_smem,
            sfb_smem,
            epi_smem,
        ).launch(
            grid=[self.grid_x, 1, 1],
            block=[cfg.threads, 1, 1],
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=cfg.enable_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        tiledmma,
        tma_atom_a,
        tma_atom_b,
        tma_atom_sfa,
        tma_tensor_a: cute.Tensor,
        tma_tensor_b: cute.Tensor,
        tma_tensor_sfa,
        gSFA: cute.Tensor,
        gSFB: cute.Tensor,
        gD: cute.Tensor,
        offsets: cute.Tensor,
        a_smem,
        b_smem,
        sfa_smem,
        sfa_tma_smem,
        sfb_smem,
        epi_smem,
    ):
        cfg = self.cfg
        i32 = cutlass.Int32
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        bidx, _, _ = cute.arch.block_idx()
        swap = cfg.mma.swap_ab
        bm, bn, bk = cfg.TILE[0], cfg.TILE[1], cfg.TILE[2]
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
        gate_smem = a_smem if cutlass.const_expr(swap) else b_smem
        sBg = stg.sBg.get_tensor(gate_smem.outer, swizzle=gate_smem.inner)
        sSFBg = stg.sSFBg.get_tensor(sfa_smem if cutlass.const_expr(swap) else sfb_smem)
        if cutlass.const_expr(cfg.epi.METHOD is not EpiMethod.DIRECT_STG):
            if cutlass.const_expr(cfg.union_smem):
                sC_stages = cute.make_tensor(
                    cute.recast_ptr(stg.sA.data_ptr(), dtype=cfg.epi.out_dtype),
                    epi_smem,
                )
            else:
                sC_stages = stg.sC.get_tensor(epi_smem.outer, swizzle=epi_smem.inner)
            sC = cute.slice_(sC_stages, (None, None, 0))
        else:
            gD_t = cute.make_tensor(
                gD.iterator, cute.make_layout((N, M), stride=(1, N))
            )
        ab_full, ab_empty = stg.ab_full.data_ptr(), stg.ab_empty.data_ptr()
        sf_full, sf_empty = stg.sf_full.data_ptr(), stg.sf_empty.data_ptr()
        store_full = (
            stg.store_full.data_ptr()
            if cutlass.const_expr(cfg.epi.HAS_STORE_WARP)
            else None
        )
        store_empty = (
            stg.store_empty.data_ptr()
            if cutlass.const_expr(cfg.epi.METHOD is not EpiMethod.DIRECT_STG)
            else None
        )
        sfull, sempty = stg.sfull.data_ptr(), stg.sempty.data_ptr()
        sWork = stg.work.get_tensor(cute.make_layout((cfg.sched_stages, cfg.fields)))

        sf_arrivals = cfg.load_sf.NUM_SCALE_COPY_THREADS + (0 if cfg.mma.swap_ab else 1)
        if warp_idx == 0:
            with cute.arch.elect_one():
                for s in cutlass.range_constexpr(cfg.ab_stage):
                    cute.arch.mbarrier_init(ab_full + s, 1)
                    cute.arch.mbarrier_init(ab_empty + s, cfg.mma_threads)
                    cute.arch.mbarrier_init(sf_full + s, sf_arrivals)
                    cute.arch.mbarrier_init(sf_empty + s, cfg.mma_threads)
                for s in cutlass.range_constexpr(cfg.store_stages):
                    if cutlass.const_expr(cfg.epi.HAS_STORE_WARP):
                        cute.arch.mbarrier_init(store_full + s, cfg.mma_threads)
                    if cutlass.const_expr(cfg.epi.METHOD is not EpiMethod.DIRECT_STG):
                        cute.arch.mbarrier_init(store_empty + s, cfg.store_threads)
                for s in cutlass.range_constexpr(cfg.sched_stages):
                    cute.arch.mbarrier_init(sfull + s, 1)
                    cute.arch.mbarrier_init(sempty + s, cfg.num_sched_consumers)
        cute.arch.mbarrier_init_fence()
        cute.arch.barrier()

        sf_atom = cute.make_copy_atom(
            cpasync.CopyG2SOp(), cfg.load_sf.sf_dtype, num_bits_per_copy=32
        )
        if cutlass.const_expr(cfg.epi.METHOD is not EpiMethod.DIRECT_STG):
            st_atom = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), cfg.epi.out_dtype
            )
            tiled_st = cfg.epi.make_tiled_s2g(st_atom, cfg.TILE)

        if cfg.is_prod_wg(warp_idx):
            cute.arch.setmaxregister_decrease(cfg.reg_prod)

            if warp_idx == cfg.sched_warp:
                sched = moe_scheduler.MoeTileScheduler.create(
                    pm, num_groups, num_n_blocks, self.grid_x, bidx, offsets
                )
                prod = moe_scheduler.MoeSchedProducer.create(cfg.sched_stages)
                has, e, m_tile, n_tile, m_off, m_bnd = sched.get_next_block(offsets)
                while has:
                    if cutlass.const_expr(swap):
                        m_tile, n_tile = n_tile, m_tile
                    prod.publish(
                        sWork,
                        sfull,
                        sempty,
                        moe_scheduler.MoeWorkTile(
                            m_tile, n_tile, e, m_off, m_bnd, i32(1)
                        ),
                    )
                    has, e, m_tile, n_tile, m_off, m_bnd = sched.get_next_block(offsets)
                prod.publish_sentinel(sWork, sfull, sempty)

            if warp_idx == cfg.ab_warp:
                cons = moe_scheduler.MoeSchedConsumer.create(cfg.sched_stages)
                tile = cons.get_next_tile(sWork, sfull, sempty)
                ab_stage, ab_phase, store_phase = i32(0), i32(1), i32(1)
                while tile.valid != i32(0):
                    if cutlass.const_expr(cfg.union_smem):
                        cute.arch.mbarrier_wait(store_empty, store_phase)
                        store_phase ^= 1
                    if cutlass.const_expr(swap):
                        ab_stage, ab_phase = load_ab_swap(
                            tma_atom_a,
                            tma_atom_b,
                            tma_tensor_a,
                            tma_tensor_b,
                            sA,
                            sBg,
                            sB,
                            cfg.TILE,
                            tile,
                            ab_full,
                            ab_empty,
                            self.ab_bytes,
                            k_tile_count,
                            cfg.ab_stage,
                            ab_stage,
                            ab_phase,
                            N,
                        )
                    else:
                        ab_stage, ab_phase = load_ab(
                            tma_atom_a,
                            tma_atom_b,
                            tma_tensor_a,
                            tma_tensor_b,
                            sA,
                            sB,
                            sBg,
                            cfg.TILE,
                            tile,
                            ab_full,
                            ab_empty,
                            self.ab_bytes,
                            k_tile_count,
                            cfg.ab_stage,
                            ab_stage,
                            ab_phase,
                            N,
                        )
                    tile = cons.get_next_tile(sWork, sfull, sempty)

            if warp_idx == cfg.sf_warp:
                cons = moe_scheduler.MoeSchedConsumer.create(cfg.sched_stages)
                tile = cons.get_next_tile(sWork, sfull, sempty)
                sf_stage, sf_phase = i32(0), i32(1)
                while tile.valid != i32(0):
                    if cutlass.const_expr(swap):
                        sf_stage, sf_phase = load_sf_swap(
                            gSFA,
                            sSFA,
                            sSFBg,
                            gSFB,
                            sSFB,
                            sf_atom,
                            tile,
                            cfg.load_sf.gran_n,
                            sf_full,
                            sf_empty,
                            cfg.load_sf.NUM_SCALE_COPY_THREADS,
                            k_tile_count,
                            self.sf_stages,
                            sf_stage,
                            sf_phase,
                            num_n_blocks,
                        )
                    else:
                        sf_stage, sf_phase = load_sf_tma(
                            tma_atom_sfa,
                            tma_tensor_sfa,
                            sSFA_tma,
                            sf_atom,
                            gSFB,
                            sSFB,
                            sSFBg,
                            tile,
                            sf_full,
                            sf_empty,
                            self.sfa_bytes,
                            cfg.load_sf.NUM_SCALE_COPY_THREADS,
                            k_tile_count,
                            self.sf_stages,
                            sf_stage,
                            sf_phase,
                            bn,
                            N,
                            cfg.load_sf.gran_n,
                        )
                    tile = cons.get_next_tile(sWork, sfull, sempty)

        else:
            cute.arch.setmaxregister_increase(cfg.reg_math)
            thr = tiledmma.get_slice(tidx)
            if cutlass.const_expr(cfg.epi.METHOD is EpiMethod.R2G_WG):
                thr_st = tiled_st.get_slice(tidx)
            cons = moe_scheduler.MoeSchedConsumer.create(cfg.sched_stages)
            tile = cons.get_next_tile(sWork, sfull, sempty)
            read_stage, ab_phase, store_phase = i32(0), i32(0), i32(1)
            while tile.valid != i32(0):
                if cutlass.const_expr(swap):
                    acc, read_stage, ab_phase = mma_swap(
                        tiledmma,
                        cfg.mma,
                        cfg.load_sf,
                        cfg.activation,
                        cfg.fastmath,
                        sA,
                        sBg,
                        sB,
                        sSFA,
                        sSFBg,
                        sSFB,
                        (bm, bn),
                        cfg.ACC,
                        cfg.load_ab.a_dtype,
                        cfg.load_ab.b_smem_dtype,
                        tidx,
                        ab_full,
                        ab_empty,
                        sf_full,
                        sf_empty,
                        k_tile_count,
                        cfg.ab_stage,
                        read_stage,
                        ab_phase,
                    )
                    moe_epilogue.store_swap(
                        acc, thr, gD_t, tile, (bm, bn), N, cfg.epi.out_dtype
                    )
                else:
                    acc, read_stage, ab_phase = mma(
                        tiledmma,
                        cfg.mma,
                        cfg.load_sf,
                        cfg.activation,
                        cfg.fastmath,
                        sA,
                        sB,
                        sBg,
                        sSFA,
                        sSFB,
                        sSFBg,
                        (bm, bn),
                        cfg.ACC,
                        cfg.load_ab.a_dtype,
                        cfg.load_ab.b_smem_dtype,
                        tidx,
                        ab_full,
                        ab_empty,
                        sf_full,
                        sf_empty,
                        k_tile_count,
                        cfg.ab_stage,
                        read_stage,
                        ab_phase,
                    )
                    moe_epilogue.store_wg(
                        acc,
                        thr,
                        thr_st,
                        sC,
                        st_atom,
                        gD,
                        tile,
                        (bm, bn),
                        N,
                        cfg.epi.out_dtype,
                        store_empty,
                        cfg.epi_bar_id,
                        cfg.mma_threads,
                    )
                tile = cons.get_next_tile(sWork, sfull, sempty)


def _stream():
    return cuda.CUstream(torch.cuda.current_stream().cuda_stream)


def make_args(a_q, a_scale, b_q, b_scale, out, m_indptr):
    sf = lambda t: from_dlpack(t.contiguous(), assumed_align=16).mark_layout_dynamic()
    return (
        from_dlpack(a_q).mark_layout_dynamic(),
        from_dlpack(b_q).mark_layout_dynamic(),
        sf(a_scale),
        sf(b_scale),
        from_dlpack(out).mark_layout_dynamic(),
        from_dlpack(m_indptr).mark_layout_dynamic(),
        _stream(),
    )
