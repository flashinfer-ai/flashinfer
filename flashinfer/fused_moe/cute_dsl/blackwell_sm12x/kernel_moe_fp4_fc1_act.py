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
"""CuteDSL NVFP4 grouped FC1 activation kernel for SM120a."""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.warp.mma as warp_mma
import cutlass.utils as utils
import torch
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import from_dlpack

from ....tllm_enums import DEFAULT_SITU_BETA as SITU_BETA
from ....tllm_enums import DEFAULT_SITU_LINEAR_BETA as SITU_LINEAR_BETA
from ....tllm_enums import ActivationType
from ....utils import ceil_div
from ._moe_utils import moe_activation, moe_epilogue, moe_scheduler
from ._moe_utils.moe_epilogue import EPI_CONFIGS, EpiMethod
from ._moe_utils.moe_kernel_builder import LoadABConfig, MmaConfig, Sm12xGatedGemmConfig
from ._moe_utils.sm12x_blockscaled_layout import (
    Sm120SfConfigNvfp4,
    compute_padded_offset,
)

ATOM_MNK = (16, 8, 64)


def is_swapab(tile):
    return ATOM_MNK[0] > tile[0]


REG_PROD_BY_TACTIC = {
    (128, 128, EpiMethod.R2G_WG): 40,
    (64, 128, EpiMethod.R2G_WG): 88,
    (32, 128, EpiMethod.R2G_WG): 88,
    (8, 128, EpiMethod.DIRECT_STG): 88,
}


def make_cfg(
    tile,
    ab_stage,
    epi=EpiMethod.R2G_WG,
    *,
    activation=ActivationType.Swiglu,
    fastmath=False,
    situ_beta=SITU_BETA,
    situ_linear_beta=SITU_LINEAR_BETA,
):
    fp4, f32, bf16, sf = (
        cutlass.Float4E2M1FN,
        cutlass.Float32,
        cutlass.BFloat16,
        cutlass.Float8E4M3FN,
    )
    assert tuple(tile) in (
        (128, 128, 128),
        (64, 128, 128),
        (32, 128, 128),
        (8, 128, 128),
    )
    bm, bn, bk = tile
    tactic = (bm, bn, EpiMethod.DIRECT_STG if is_swapab(tile) else epi)
    swap = is_swapab(tile)
    if swap:
        bm, bn = bn, bm
        epi = EpiMethod.DIRECT_STG
    else:
        assert epi is EpiMethod.R2G_WG
    tile = (bm, bn, bk)
    return Sm12xGatedGemmConfig(
        MmaConfig(warp_mma.MmaMXF4NVF4Op(fp4, f32, sf), tile[:2], 8, swap_ab=swap),
        LoadABConfig(tile, ab_stage, fp4, fp4),
        Sm120SfConfigNvfp4(sf),
        EPI_CONFIGS[epi](bf16, 8 * 32),
        ab_stage,
        tile,
        epi_bar_id=3,
        union_smem=not swap,
        reg_prod=REG_PROD_BY_TACTIC[tactic],
        activation=moe_activation.make_gated_activation(
            activation, fastmath, situ_beta=situ_beta, situ_linear_beta=situ_linear_beta
        ),
    )


@cute.jit
def make_a_sfa_partitions(
    tma_atom_a,
    tma_atom_sfa,
    tma_tensor_a,
    tma_tensor_sfa,
    sA,
    sSFA,
    tile_mnk,
    tile,
    sfa_tile_m,
    sfa_tiles_per_block,
    swap,
):
    cta_layout = cute.make_layout(1)
    if cutlass.const_expr(swap):
        gA_mkl = cute.local_tile(
            tma_tensor_a, cute.slice_(tile_mnk, (None, 0, None)), (None, None, None)
        )
    else:
        mA = cute.domain_offset((tile.m_offset, 0), tma_tensor_a)
        gA_mkl = cute.local_tile(
            mA, cute.slice_(tile_mnk, (None, 0, None)), (None, None)
        )
    tAsA, tAgA = cpasync.tma_partition(
        tma_atom_a,
        cutlass.Int32(0),
        cta_layout,
        cute.group_modes(sA, 0, 2),
        cute.group_modes(gA_mkl, 0, 2),
    )
    if cutlass.const_expr(swap):
        gSFA_mkl = cute.local_tile(
            tma_tensor_sfa,
            (sfa_tile_m, cute.size(sSFA, mode=[1])),
            (None, None, None),
        )
    else:
        sf_m_off = compute_padded_offset(tile.m_offset, tile.group, cutlass.Int32(128))
        mSFA = cute.domain_offset((sf_m_off, 0, 0), tma_tensor_sfa)
        gSFA_mkl = cute.local_tile(
            mSFA, (sfa_tile_m, cute.size(sSFA, mode=[1])), (None, None, None)
        )
    tAsSFA, tAgSFA = cpasync.tma_partition(
        tma_atom_sfa,
        cutlass.Int32(0),
        cta_layout,
        cute.group_modes(sSFA, 0, 2),
        cute.group_modes(gSFA_mkl, 0, 2),
    )
    if cutlass.const_expr(sfa_tiles_per_block == 1):
        sfa_m_block = tile.m_block
    elif cutlass.const_expr(sfa_tiles_per_block == 2):
        sfa_m_block = tile.m_block >> 1
    else:
        sfa_m_block = tile.m_block >> 2
    return (
        tAsA,
        tAgA,
        cute.filter_zeros(tAsSFA),
        cute.filter_zeros(tAgSFA),
        sfa_m_block,
    )


@cute.jit
def make_b_sfb_partitions(
    tma_atom_b,
    tma_atom_sfb,
    tma_tensor_b,
    tma_tensor_sfb,
    sB,
    sSFB,
    tile_mnk,
    tile,
    sfb_tile_n,
    sfb_tiles_per_block,
    swap,
):
    cta_layout = cute.make_layout(1)
    if cutlass.const_expr(swap):
        mB = cute.domain_offset((tile.m_offset, 0), tma_tensor_b)
        gB_nkl = cute.local_tile(
            mB, cute.slice_(tile_mnk, (0, None, None)), (None, None)
        )
        sf_m_off = compute_padded_offset(tile.m_offset, tile.group, cutlass.Int32(128))
        mSFB = cute.domain_offset((sf_m_off, 0, 0), tma_tensor_sfb)
        gSFB_nkl = cute.local_tile(
            mSFB, (sfb_tile_n, cute.size(sSFB, mode=[1])), (None, None, None)
        )
    else:
        gB_nkl = cute.local_tile(
            tma_tensor_b, cute.slice_(tile_mnk, (0, None, None)), (None, None, None)
        )
        gSFB_nkl = cute.local_tile(
            tma_tensor_sfb,
            (sfb_tile_n, cute.size(sSFB, mode=[1])),
            (None, None, None),
        )
    tBsB, tBgB = cpasync.tma_partition(
        tma_atom_b,
        cutlass.Int32(0),
        cta_layout,
        cute.group_modes(sB, 0, 2),
        cute.group_modes(gB_nkl, 0, 2),
    )
    tBsSFB, tBgSFB = cpasync.tma_partition(
        tma_atom_sfb,
        cutlass.Int32(0),
        cta_layout,
        cute.group_modes(sSFB, 0, 2),
        cute.group_modes(gSFB_nkl, 0, 2),
    )
    if cutlass.const_expr(swap):
        if cutlass.const_expr(sfb_tiles_per_block == 1):
            sfb_n_block = tile.n_block
        elif cutlass.const_expr(sfb_tiles_per_block == 2):
            sfb_n_block = tile.n_block >> 1
        elif cutlass.const_expr(sfb_tiles_per_block == 4):
            sfb_n_block = tile.n_block >> 2
        elif cutlass.const_expr(sfb_tiles_per_block == 8):
            sfb_n_block = tile.n_block >> 3
        else:
            sfb_n_block = tile.n_block >> 4
    else:
        sfb_n_block = tile.n_block
    return (
        tBsB,
        tBgB,
        cute.filter_zeros(tBsSFB),
        cute.filter_zeros(tBgSFB),
        sfb_n_block,
    )


@cute.jit
def copy_a_sfa(
    tma_atom_a,
    tma_atom_sfa,
    tAgA,
    tAsA,
    tAgSFA,
    tAsSFA,
    tile,
    sfa_m_block,
    a_full,
    a_empty,
    tx_bytes,
    k_tile_count,
    ab_stages,
    stage,
    phase,
    swap,
):
    for kt in cutlass.range(0, k_tile_count):
        cute.arch.mbarrier_wait(a_empty + stage, phase)
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(a_full + stage, tx_bytes)
        if cutlass.const_expr(swap):
            cute.copy(
                tma_atom_a,
                tAgA[(None, tile.m_block, kt, tile.group)],
                tAsA[(None, stage)],
                tma_bar_ptr=a_full + stage,
            )
            cute.copy(
                tma_atom_sfa,
                tAgSFA[(None, sfa_m_block, kt, tile.group)],
                tAsSFA[(None, stage)],
                tma_bar_ptr=a_full + stage,
            )
        else:
            cute.copy(
                tma_atom_a,
                tAgA[(None, tile.m_block, kt)],
                tAsA[(None, stage)],
                tma_bar_ptr=a_full + stage,
            )
            cute.copy(
                tma_atom_sfa,
                tAgSFA[(None, sfa_m_block, kt, 0)],
                tAsSFA[(None, stage)],
                tma_bar_ptr=a_full + stage,
            )
        stage += 1
        if stage == ab_stages:
            stage = cutlass.Int32(0)
            phase ^= 1
    return stage, phase


@cute.jit
def copy_b_sfb(
    tma_atom_b,
    tma_atom_sfb,
    tBgB,
    tBsB,
    tBgSFB,
    tBsSFB,
    tile,
    sfb_n_block,
    b_full,
    b_empty,
    tx_bytes,
    k_tile_count,
    ab_stages,
    stage,
    phase,
    swap,
):
    for kt in cutlass.range(0, k_tile_count):
        cute.arch.mbarrier_wait(b_empty + stage, phase)
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(b_full + stage, tx_bytes)
        if cutlass.const_expr(swap):
            cute.copy(
                tma_atom_b,
                tBgB[(None, tile.n_block, kt)],
                tBsB[(None, stage)],
                tma_bar_ptr=b_full + stage,
            )
            cute.copy(
                tma_atom_sfb,
                tBgSFB[(None, sfb_n_block, kt, 0)],
                tBsSFB[(None, stage)],
                tma_bar_ptr=b_full + stage,
            )
        else:
            cute.copy(
                tma_atom_b,
                tBgB[(None, tile.n_block, kt, tile.group)],
                tBsB[(None, stage)],
                tma_bar_ptr=b_full + stage,
            )
            cute.copy(
                tma_atom_sfb,
                tBgSFB[(None, tile.n_block, kt, tile.group)],
                tBsSFB[(None, stage)],
                tma_bar_ptr=b_full + stage,
            )
        stage += 1
        if stage == ab_stages:
            stage = cutlass.Int32(0)
            phase ^= 1
    return stage, phase


@cute.jit
def make_b_sfb_gate_partitions(
    tma_atom_b,
    tma_atom_sfb,
    tma_tensor_b,
    tma_tensor_sfb,
    sB,
    sG,
    sSFB,
    sSFG,
    tile_mnk,
    tile,
    n_gate_off,
):
    cta_layout = cute.make_layout(1)
    gB_nkl = cute.local_tile(
        tma_tensor_b, cute.slice_(tile_mnk, (0, None, None)), (None, None, None)
    )
    gG_nkl = cute.local_tile(
        cute.domain_offset((n_gate_off, 0, 0), tma_tensor_b),
        cute.slice_(tile_mnk, (0, None, None)),
        (None, None, None),
    )
    gSFB_nkl = cute.local_tile(
        tma_tensor_sfb,
        (tile_mnk[1], cute.size(sSFB, mode=[1])),
        (None, None, None),
    )
    gSFG_nkl = cute.local_tile(
        cute.domain_offset((n_gate_off, 0, 0), tma_tensor_sfb),
        (tile_mnk[1], cute.size(sSFG, mode=[1])),
        (None, None, None),
    )
    tBsB, tBgB = cpasync.tma_partition(
        tma_atom_b,
        cutlass.Int32(0),
        cta_layout,
        cute.group_modes(sB, 0, 2),
        cute.group_modes(gB_nkl, 0, 2),
    )
    tGsG, tGgG = cpasync.tma_partition(
        tma_atom_b,
        cutlass.Int32(0),
        cta_layout,
        cute.group_modes(sG, 0, 2),
        cute.group_modes(gG_nkl, 0, 2),
    )
    tBsSFB, tBgSFB = cpasync.tma_partition(
        tma_atom_sfb,
        cutlass.Int32(0),
        cta_layout,
        cute.group_modes(sSFB, 0, 2),
        cute.group_modes(gSFB_nkl, 0, 2),
    )
    tGsSFG, tGgSFG = cpasync.tma_partition(
        tma_atom_sfb,
        cutlass.Int32(0),
        cta_layout,
        cute.group_modes(sSFG, 0, 2),
        cute.group_modes(gSFG_nkl, 0, 2),
    )
    return (
        tBsB,
        tBgB,
        tGsG,
        tGgG,
        cute.filter_zeros(tBsSFB),
        cute.filter_zeros(tBgSFB),
        cute.filter_zeros(tGsSFG),
        cute.filter_zeros(tGgSFG),
    )


@cute.jit
def make_a_sfa_gate_partitions_swap(
    tma_atom_a,
    tma_atom_sfa,
    tma_tensor_a,
    tma_tensor_sfa,
    sA,
    sG,
    sSFA,
    sSFG,
    tile_mnk,
    tile,
    n_gate_off,
):
    cta_layout = cute.make_layout(1)
    gA_mkl = cute.local_tile(
        tma_tensor_a, cute.slice_(tile_mnk, (None, 0, None)), (None, None, None)
    )
    gG_mkl = cute.local_tile(
        cute.domain_offset((n_gate_off, 0, 0), tma_tensor_a),
        cute.slice_(tile_mnk, (None, 0, None)),
        (None, None, None),
    )
    gSFA_mkl = cute.local_tile(
        tma_tensor_sfa,
        (tile_mnk[0], cute.size(sSFA, mode=[1])),
        (None, None, None),
    )
    gSFG_mkl = cute.local_tile(
        cute.domain_offset((n_gate_off, 0, 0), tma_tensor_sfa),
        (tile_mnk[0], cute.size(sSFG, mode=[1])),
        (None, None, None),
    )
    tAsA, tAgA = cpasync.tma_partition(
        tma_atom_a,
        cutlass.Int32(0),
        cta_layout,
        cute.group_modes(sA, 0, 2),
        cute.group_modes(gA_mkl, 0, 2),
    )
    tGsG, tGgG = cpasync.tma_partition(
        tma_atom_a,
        cutlass.Int32(0),
        cta_layout,
        cute.group_modes(sG, 0, 2),
        cute.group_modes(gG_mkl, 0, 2),
    )
    tAsSFA, tAgSFA = cpasync.tma_partition(
        tma_atom_sfa,
        cutlass.Int32(0),
        cta_layout,
        cute.group_modes(sSFA, 0, 2),
        cute.group_modes(gSFA_mkl, 0, 2),
    )
    tGsSFG, tGgSFG = cpasync.tma_partition(
        tma_atom_sfa,
        cutlass.Int32(0),
        cta_layout,
        cute.group_modes(sSFG, 0, 2),
        cute.group_modes(gSFG_mkl, 0, 2),
    )
    return (
        tAsA,
        tAgA,
        tGsG,
        tGgG,
        cute.filter_zeros(tAsSFA),
        cute.filter_zeros(tAgSFA),
        cute.filter_zeros(tGsSFG),
        cute.filter_zeros(tGgSFG),
    )


@cute.jit
def copy_a_sfa_gate_swap(
    tma_atom_a,
    tma_atom_sfa,
    tAgA,
    tAsA,
    tGgG,
    tGsG,
    tAgSFA,
    tAsSFA,
    tGgSFG,
    tGsSFG,
    tile,
    a_full,
    a_empty,
    tx_bytes,
    k_tile_count,
    ab_stages,
    stage,
    phase,
):
    for kt in cutlass.range(0, k_tile_count):
        cute.arch.mbarrier_wait(a_empty + stage, phase)
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(a_full + stage, tx_bytes)
        cute.copy(
            tma_atom_a,
            tAgA[(None, tile.m_block, kt, tile.group)],
            tAsA[(None, stage)],
            tma_bar_ptr=a_full + stage,
        )
        cute.copy(
            tma_atom_a,
            tGgG[(None, tile.m_block, kt, tile.group)],
            tGsG[(None, stage)],
            tma_bar_ptr=a_full + stage,
        )
        cute.copy(
            tma_atom_sfa,
            tAgSFA[(None, tile.m_block, kt, tile.group)],
            tAsSFA[(None, stage)],
            tma_bar_ptr=a_full + stage,
        )
        cute.copy(
            tma_atom_sfa,
            tGgSFG[(None, tile.m_block, kt, tile.group)],
            tGsSFG[(None, stage)],
            tma_bar_ptr=a_full + stage,
        )
        stage += 1
        if stage == ab_stages:
            stage = cutlass.Int32(0)
            phase ^= 1
    return stage, phase


@cute.jit
def copy_b_sfb_gate(
    tma_atom_b,
    tma_atom_sfb,
    tBgB,
    tBsB,
    tGgG,
    tGsG,
    tBgSFB,
    tBsSFB,
    tGgSFG,
    tGsSFG,
    tile,
    b_full,
    b_empty,
    tx_bytes,
    k_tile_count,
    ab_stages,
    stage,
    phase,
):
    for kt in cutlass.range(0, k_tile_count):
        cute.arch.mbarrier_wait(b_empty + stage, phase)
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(b_full + stage, tx_bytes)
        cute.copy(
            tma_atom_b,
            tBgB[(None, tile.n_block, kt, tile.group)],
            tBsB[(None, stage)],
            tma_bar_ptr=b_full + stage,
        )
        cute.copy(
            tma_atom_b,
            tGgG[(None, tile.n_block, kt, tile.group)],
            tGsG[(None, stage)],
            tma_bar_ptr=b_full + stage,
        )
        cute.copy(
            tma_atom_sfb,
            tBgSFB[(None, tile.n_block, kt, tile.group)],
            tBsSFB[(None, stage)],
            tma_bar_ptr=b_full + stage,
        )
        cute.copy(
            tma_atom_sfb,
            tGgSFG[(None, tile.n_block, kt, tile.group)],
            tGsSFG[(None, stage)],
            tma_bar_ptr=b_full + stage,
        )
        stage += 1
        if stage == ab_stages:
            stage = cutlass.Int32(0)
            phase ^= 1
    return stage, phase


@cute.jit
def mma(
    tiledmma,
    mma_cfg,
    sf_cfg,
    activation,
    sA,
    sB,
    sG,
    sSFA,
    sSFB,
    sSFG,
    tile_mn,
    acc_dtype,
    a_dtype,
    b_dtype,
    a_is_m_major,
    b_is_n_major,
    unpack_bits,
    tidx,
    a_full,
    a_empty,
    b_full,
    b_empty,
    k_tile_count,
    ab_stages,
    stage,
    phase,
    up_scale,
    gate_scale,
):
    bm, bn = tile_mn
    thr = tiledmma.get_slice(tidx)
    tCrA = tiledmma.make_fragment_A(thr.partition_A(sA)[None, None, None, 0])
    tCrB = tiledmma.make_fragment_B(thr.partition_B(sB)[None, None, None, 0])
    tCrG = tiledmma.make_fragment_B(thr.partition_B(sG)[None, None, None, 0])
    shape_c = tiledmma.partition_shape_C((bm, bn))
    acc = cute.make_rmem_tensor(shape_c, acc_dtype)
    acc_g = cute.make_rmem_tensor(shape_c, acc_dtype)
    s2r_a = mma_cfg.make_s2r_a(tiledmma, a_dtype, a_is_m_major)
    s2r_b = mma_cfg.make_s2r_b(tiledmma, b_dtype, b_is_n_major, unpack_bits)
    thr_a, thr_b = s2r_a.get_slice(tidx), s2r_b.get_slice(tidx)
    tCrA_v = thr_a.retile(tCrA)
    tCrB_v, tCrG_v = thr_b.retile(tCrB), thr_b.retile(tCrG)
    tCrSFA = sf_cfg.partition_fragment_SFA(sSFA[None, None, 0], thr, tidx)
    tCrSFB = sf_cfg.partition_fragment_SFB(sSFB[None, None, 0], thr, tidx)
    tCrSFG = sf_cfg.partition_fragment_SFB(sSFG[None, None, 0], thr, tidx)
    tCrSFA_frg = sf_cfg.make_sfa_e4m3_view(tCrSFA)
    tCrSFB_frg = sf_cfg.make_sfb_e4m3_view(tCrSFB)
    tCrSFG_frg = sf_cfg.make_sfb_e4m3_view(tCrSFG)
    s2r_sfa = sf_cfg.make_s2r_sf(
        sf_cfg.get_layoutSFA_TV(tiledmma),
        (cute.size(tiledmma.permutation_mnk[0]), sf_cfg.KTILE_SF),
    )
    s2r_sfb = sf_cfg.make_s2r_sf(
        sf_cfg.get_layoutSFB_TV(tiledmma),
        (cute.size(tiledmma.permutation_mnk[1]), sf_cfg.KTILE_SF),
    )
    thr_sfa, thr_sfb = s2r_sfa.get_slice(tidx), s2r_sfb.get_slice(tidx)
    tCrSFA_v = thr_sfa.retile(tCrSFA)
    tCrSFB_v, tCrSFG_v = thr_sfb.retile(tCrSFB), thr_sfb.retile(tCrSFG)
    k_blocks = cute.size(tCrA_v, mode=[2])
    sf_blocks = cute.size(tCrSFA_v, mode=[2])
    acc.fill(0.0)
    acc_g.fill(0.0)
    for _ in cutlass.range(0, k_tile_count):
        cute.arch.mbarrier_wait(a_full + stage, phase)
        cute.arch.mbarrier_wait(b_full + stage, phase)
        tAsA = thr_a.partition_S(sA)[None, None, None, stage]
        tBsB = thr_b.partition_S(sB)[None, None, None, stage]
        tGsG = thr_b.partition_S(sG)[None, None, None, stage]
        tAsSFA = thr_sfa.partition_S(sSFA)[None, None, None, stage]
        tBsSFB = thr_sfb.partition_S(sSFB)[None, None, None, stage]
        tGsSFG = thr_sfb.partition_S(sSFG)[None, None, None, stage]
        for k in cutlass.range_constexpr(0, k_blocks):
            cute.copy(s2r_a, tAsA[None, None, k], tCrA_v[None, None, k])
            cute.copy(s2r_b, tBsB[None, None, k], tCrB_v[None, None, k])
            cute.copy(s2r_b, tGsG[None, None, k], tCrG_v[None, None, k])
        for k in cutlass.range_constexpr(0, sf_blocks):
            cute.copy(s2r_sfa, tAsSFA[None, None, k], tCrSFA_v[None, None, k])
            cute.copy(s2r_sfb, tBsSFB[None, None, k], tCrSFB_v[None, None, k])
            cute.copy(s2r_sfb, tGsSFG[None, None, k], tCrSFG_v[None, None, k])
        cute.gemm(tiledmma, acc, [tCrA, tCrSFA_frg], [tCrB, tCrSFB_frg], acc)
        cute.gemm(tiledmma, acc_g, [tCrA, tCrSFA_frg], [tCrG, tCrSFG_frg], acc_g)
        cute.arch.mbarrier_arrive(a_empty + stage)
        cute.arch.mbarrier_arrive(b_empty + stage)
        stage += 1
        if stage == ab_stages:
            stage = cutlass.Int32(0)
            phase ^= 1
    acc.store(acc.load() * up_scale)
    acc_g.store(acc_g.load() * gate_scale)
    activation(acc, acc_g)
    return acc, stage, phase


@cute.jit
def mma_swap(
    tiledmma,
    mma_cfg,
    sf_cfg,
    activation,
    sA,
    sG,
    sB,
    sSFA,
    sSFG,
    sSFB,
    tile_mn,
    acc_dtype,
    a_dtype,
    b_dtype,
    a_is_m_major,
    b_is_n_major,
    unpack_bits,
    tidx,
    a_full,
    a_empty,
    b_full,
    b_empty,
    k_tile_count,
    ab_stages,
    stage,
    phase,
    up_scale,
    gate_scale,
):
    bm, bn = tile_mn
    thr = tiledmma.get_slice(tidx)
    tCrA = tiledmma.make_fragment_A(thr.partition_A(sA)[None, None, None, 0])
    tCrG = tiledmma.make_fragment_A(thr.partition_A(sG)[None, None, None, 0])
    tCrB = tiledmma.make_fragment_B(thr.partition_B(sB)[None, None, None, 0])
    shape_c = tiledmma.partition_shape_C((bm, bn))
    acc = cute.make_rmem_tensor(shape_c, acc_dtype)
    acc_g = cute.make_rmem_tensor(shape_c, acc_dtype)
    s2r_a = mma_cfg.make_s2r_a(tiledmma, a_dtype, a_is_m_major)
    s2r_b = mma_cfg.make_s2r_b(tiledmma, b_dtype, b_is_n_major, unpack_bits)
    thr_a, thr_b = s2r_a.get_slice(tidx), s2r_b.get_slice(tidx)
    tCrA_v, tCrG_v = thr_a.retile(tCrA), thr_a.retile(tCrG)
    tCrB_v = thr_b.retile(tCrB)
    tCrSFA = sf_cfg.partition_fragment_SFA(sSFA[None, None, 0], thr, tidx)
    tCrSFG = sf_cfg.partition_fragment_SFA(sSFG[None, None, 0], thr, tidx)
    tCrSFB = sf_cfg.partition_fragment_SFB(sSFB[None, None, 0], thr, tidx)
    tCrSFA_frg = sf_cfg.make_sfa_e4m3_view(tCrSFA)
    tCrSFG_frg = sf_cfg.make_sfa_e4m3_view(tCrSFG)
    tCrSFB_frg = sf_cfg.make_sfb_e4m3_view(tCrSFB)
    s2r_sfa = sf_cfg.make_s2r_sf(
        sf_cfg.get_layoutSFA_TV(tiledmma),
        (cute.size(tiledmma.permutation_mnk[0]), sf_cfg.KTILE_SF),
    )
    s2r_sfb = sf_cfg.make_s2r_sf(
        sf_cfg.get_layoutSFB_TV(tiledmma),
        (cute.size(tiledmma.permutation_mnk[1]), sf_cfg.KTILE_SF),
    )
    thr_sfa, thr_sfb = s2r_sfa.get_slice(tidx), s2r_sfb.get_slice(tidx)
    tCrSFA_v, tCrSFG_v = thr_sfa.retile(tCrSFA), thr_sfa.retile(tCrSFG)
    tCrSFB_v = thr_sfb.retile(tCrSFB)
    k_blocks = cute.size(tCrA_v, mode=[2])
    sf_blocks = cute.size(tCrSFA_v, mode=[2])
    acc.fill(0.0)
    acc_g.fill(0.0)
    for _ in cutlass.range(0, k_tile_count):
        cute.arch.mbarrier_wait(a_full + stage, phase)
        cute.arch.mbarrier_wait(b_full + stage, phase)
        tAsA = thr_a.partition_S(sA)[None, None, None, stage]
        tGsG = thr_a.partition_S(sG)[None, None, None, stage]
        tBsB = thr_b.partition_S(sB)[None, None, None, stage]
        tAsSFA = thr_sfa.partition_S(sSFA)[None, None, None, stage]
        tGsSFG = thr_sfa.partition_S(sSFG)[None, None, None, stage]
        tBsSFB = thr_sfb.partition_S(sSFB)[None, None, None, stage]
        for k in cutlass.range_constexpr(0, k_blocks):
            cute.copy(s2r_a, tAsA[None, None, k], tCrA_v[None, None, k])
            cute.copy(s2r_a, tGsG[None, None, k], tCrG_v[None, None, k])
            cute.copy(s2r_b, tBsB[None, None, k], tCrB_v[None, None, k])
        for k in cutlass.range_constexpr(0, sf_blocks):
            cute.copy(s2r_sfa, tAsSFA[None, None, k], tCrSFA_v[None, None, k])
            cute.copy(s2r_sfa, tGsSFG[None, None, k], tCrSFG_v[None, None, k])
            cute.copy(s2r_sfb, tBsSFB[None, None, k], tCrSFB_v[None, None, k])
        cute.gemm(tiledmma, acc, [tCrA, tCrSFA_frg], [tCrB, tCrSFB_frg], acc)
        cute.gemm(tiledmma, acc_g, [tCrG, tCrSFG_frg], [tCrB, tCrSFB_frg], acc_g)
        cute.arch.mbarrier_arrive(a_empty + stage)
        cute.arch.mbarrier_arrive(b_empty + stage)
        stage += 1
        if stage == ab_stages:
            stage = cutlass.Int32(0)
            phase ^= 1
    acc.store(acc.load() * up_scale)
    acc_g.store(acc_g.load() * gate_scale)
    activation(acc, acc_g)
    return acc, stage, phase


class CuteDslSm120MoeNvfp4Fc1Act:
    def __init__(self, cfg, grid_x):
        self.cfg = cfg
        self.grid_x = grid_x
        self.mma = cfg.mma
        self.sf = cfg.load_sf
        self.num_math_warps = cfg.num_math_warps
        self.mma_threads = cfg.mma_threads
        self.sched_warp = cfg.sched_warp
        self.load_warp_0, self.load_warp_1 = cfg.load_warp_0, cfg.load_warp_1
        self.threads = cfg.threads
        self.num_sched_consumers = cfg.num_sched_consumers
        self.reg_math, self.reg_prod = cfg.reg_math, cfg.reg_prod

    @cute.jit
    def __call__(
        self,
        gA_u8: cute.Tensor,
        gB_u8: cute.Tensor,
        gSFA_u8: cute.Tensor,
        gSFB_u8: cute.Tensor,
        gD: cute.Tensor,
        offsets: cute.Tensor,
        up_scale: cute.Tensor,
        gate_scale: cute.Tensor,
        stream,
    ):
        cfg = self.cfg
        tiledmma = cfg.mma.make_tiled_mma(cfg.TILE)
        tiled_r2s = (
            cfg.epi.make_tiled_r2s(tiledmma)
            if cutlass.const_expr(cfg.epi.HAS_R2S)
            else None
        )
        gA = cute.recast_tensor(gA_u8, cfg.load_ab.a_dtype)
        gB_e = cute.recast_tensor(gB_u8, cfg.load_ab.b_dtype)
        E, N2, K = (
            cute.size(gB_e, mode=[0]),
            cute.size(gB_e, mode=[1]),
            cute.size(gB_e, mode=[2]),
        )
        gW = cute.make_tensor(
            gB_e.iterator, cute.make_layout((N2, K, E), stride=(K, 1, N2 * K))
        )
        bm, bn, bk = cfg.TILE[0], cfg.TILE[1], cfg.TILE[2]
        m_padded_sf = cute.size(gSFA_u8, mode=[0])
        if cutlass.const_expr(cfg.mma.swap_ab):
            t_a, t_b = gW, gA
            gSFA = cute.make_tensor(
                cute.recast_ptr(gSFB_u8.iterator, dtype=cfg.load_sf.sf_dtype),
                cfg.load_sf.deduce_sfa_layout(N2, K, E),
            )
            gSFB = cute.make_tensor(
                cute.recast_ptr(gSFA_u8.iterator, dtype=cfg.load_sf.sf_dtype),
                cfg.load_sf.deduce_sfb_layout(m_padded_sf, K, 1),
            )
        else:
            t_a, t_b = gA, gW
            gSFA = cute.make_tensor(
                cute.recast_ptr(gSFA_u8.iterator, dtype=cfg.load_sf.sf_dtype),
                cfg.load_sf.deduce_sfa_layout(m_padded_sf, K, 1),
            )
            gSFB = cute.make_tensor(
                cute.recast_ptr(gSFB_u8.iterator, dtype=cfg.load_sf.sf_dtype),
                cfg.load_sf.deduce_sfb_layout(N2, K, E),
            )
        a_layout = utils.LayoutEnum.from_tensor(t_a)
        b_layout = utils.LayoutEnum.from_tensor(t_b)
        a_smem = cfg.load_ab.make_smem_layout_a()
        b_smem = cfg.load_ab.make_smem_layout_b()
        sfa_smem = cfg.load_sf.make_smem_layout_sfa(tiledmma, cfg.TILE, cfg.ab_stage)
        sfb_smem = cfg.load_sf.make_smem_layout_sfb(tiledmma, cfg.TILE, cfg.ab_stage)
        gate_smem = a_smem if cutlass.const_expr(cfg.mma.swap_ab) else b_smem
        gate_sf_smem = sfa_smem if cutlass.const_expr(cfg.mma.swap_ab) else sfb_smem
        epi_smem = cfg.epi.make_smem_layout(cfg.TILE)
        self.a_is_m_major = a_layout.is_m_major_a()
        self.b_is_n_major = b_layout.is_n_major_b()

        a_stage = cute.slice_(a_smem, (None, None, 0))
        b_stage = cute.slice_(b_smem, (None, None, 0))
        sfa_stage = cute.slice_(sfa_smem, (None, None, 0))
        sfb_stage = cute.slice_(sfb_smem, (None, None, 0))
        tma_atom_a, tma_tensor_a = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), t_a, a_stage, (bm, bk), num_multicast=1
        )
        tma_atom_sfa, tma_tensor_sfa = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            gSFA,
            sfa_stage,
            cfg.load_sf.sfa_tiler(cfg.TILE),
            num_multicast=1,
            internal_type=cfg.I16,
        )
        tma_atom_b, tma_tensor_b = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), t_b, b_stage, (bn, bk), num_multicast=1
        )
        tma_atom_sfb, tma_tensor_sfb = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            gSFB,
            sfb_stage,
            cfg.load_sf.sfb_tiler(cfg.TILE),
            num_multicast=1,
            internal_type=cfg.I16,
        )
        self.a_sfa_bytes = (
            cute.size_in_bytes(cfg.load_ab.a_dtype, a_stage)
            + cute.size_in_bytes(cfg.load_sf.sf_dtype, sfa_stage)
        ) * (2 if cutlass.const_expr(cfg.mma.swap_ab) else 1)
        self.b_sfb_bytes = (
            cute.size_in_bytes(cfg.load_ab.b_dtype, b_stage)
            + cute.size_in_bytes(cfg.load_sf.sf_dtype, sfb_stage)
        ) * (1 if cutlass.const_expr(cfg.mma.swap_ab) else 2)

        epi_full_bars = cfg.epi.num_full_barriers
        epi_empty_bars = cfg.epi.num_empty_barriers
        epi_elems = cute.cosize(epi_smem) if cfg.owns_epi_smem else 0

        @cute.struct
        class SharedStorage:
            a_full: cute.struct.MemRange[cfg.I64, cfg.ab_stage]
            a_empty: cute.struct.MemRange[cfg.I64, cfg.ab_stage]
            b_full: cute.struct.MemRange[cfg.I64, cfg.ab_stage]
            b_empty: cute.struct.MemRange[cfg.I64, cfg.ab_stage]
            epi_full: cute.struct.MemRange[cfg.I64, epi_full_bars]
            epi_empty: cute.struct.MemRange[cfg.I64, epi_empty_bars]
            sfull: cute.struct.MemRange[cfg.I64, cfg.sched_stages]
            sempty: cute.struct.MemRange[cfg.I64, cfg.sched_stages]
            work: cute.struct.MemRange[cfg.I32, cfg.sched_stages * cfg.fields]
            sA: cute.struct.Align[
                cute.struct.MemRange[cfg.load_ab.a_dtype, cute.cosize(a_smem)], 128
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[cfg.load_ab.b_dtype, cute.cosize(b_smem)], 128
            ]
            sG: cute.struct.Align[
                cute.struct.MemRange[cfg.load_ab.a_dtype, cute.cosize(gate_smem)], 128
            ]
            sSFA: cute.struct.Align[
                cute.struct.MemRange[cfg.load_sf.sf_dtype, cute.cosize(sfa_smem)], 128
            ]
            sSFB: cute.struct.Align[
                cute.struct.MemRange[cfg.load_sf.sf_dtype, cute.cosize(sfb_smem)], 128
            ]
            sSFG: cute.struct.Align[
                cute.struct.MemRange[cfg.load_sf.sf_dtype, cute.cosize(gate_sf_smem)],
                128,
            ]
            sC: cute.struct.Align[
                cute.struct.MemRange[cfg.epi.out_dtype, epi_elems], 128
            ]

        smem_size = SharedStorage.size_in_bytes()
        assert cfg.smem_bytes <= smem_size <= cfg.smem_bytes + cfg.MBAR_RESERVE, (
            f"smem model {cfg.smem_bytes} B vs allocated {smem_size} B"
        )

        self.storage = SharedStorage
        self.kernel(
            tiledmma,
            tiled_r2s,
            tma_atom_a,
            tma_atom_b,
            tma_atom_sfa,
            tma_atom_sfb,
            tma_tensor_a,
            tma_tensor_b,
            tma_tensor_sfa,
            tma_tensor_sfb,
            gD,
            offsets,
            up_scale,
            gate_scale,
            a_smem,
            b_smem,
            sfa_smem,
            sfb_smem,
            epi_smem,
        ).launch(
            grid=[self.grid_x, 1, 1],
            block=[self.threads, 1, 1],
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        tiledmma,
        tiled_r2s,
        tma_atom_a,
        tma_atom_b,
        tma_atom_sfa,
        tma_atom_sfb,
        tma_tensor_a: cute.Tensor,
        tma_tensor_b: cute.Tensor,
        tma_tensor_sfa: cute.Tensor,
        tma_tensor_sfb: cute.Tensor,
        gD: cute.Tensor,
        offsets: cute.Tensor,
        up_scale: cute.Tensor,
        gate_scale: cute.Tensor,
        a_smem,
        b_smem,
        sfa_smem,
        sfb_smem,
        epi_smem,
    ):
        cfg = self.cfg
        epi_full_bars = cfg.epi.num_full_barriers
        epi_empty_bars = cfg.epi.num_empty_barriers
        i32 = cutlass.Int32
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        bidx, _, _ = cute.arch.block_idx()
        bm, bn, bk = cfg.TILE[0], cfg.TILE[1], cfg.TILE[2]
        swap = cfg.mma.swap_ab
        gate_smem = a_smem if cutlass.const_expr(swap) else b_smem
        gate_sf_smem = sfa_smem if cutlass.const_expr(swap) else sfb_smem
        pm, pn = (bn, bm) if cutlass.const_expr(swap) else (bm, bn)
        M, N = cute.size(gD, mode=[0]), cute.size(gD, mode=[1])
        K = cute.size(tma_tensor_a, mode=[1])
        num_groups = cute.size(offsets, mode=[0]) - 1
        num_n_blocks = ceil_div(N, pn)
        k_tile_count = ceil_div(K, bk)
        sfa_tile_m = cfg.load_sf.sfa_tile_m(bm)
        sfa_tiles_per_block = cfg.load_sf.sfa_tiles_per_block(bm)
        sfb_tile_n = cfg.load_sf.sfb_tile_n(bn)
        sfb_tiles_per_block = cfg.load_sf.sfb_tiles_per_block(bn)

        smem = cutlass.utils.SmemAllocator()
        stg = smem.allocate(self.storage)
        sA = stg.sA.get_tensor(a_smem.outer, swizzle=a_smem.inner)
        sB = stg.sB.get_tensor(b_smem.outer, swizzle=b_smem.inner)
        sG = stg.sG.get_tensor(gate_smem.outer, swizzle=gate_smem.inner)
        sSFA = stg.sSFA.get_tensor(sfa_smem)
        sSFB = stg.sSFB.get_tensor(sfb_smem)
        sSFG = stg.sSFG.get_tensor(gate_sf_smem)
        if cutlass.const_expr(cfg.epi.HAS_R2S):
            if cutlass.const_expr(cfg.union_smem):
                sC_stages = cute.make_tensor(
                    cute.recast_ptr(stg.sA.data_ptr(), dtype=cfg.epi.out_dtype),
                    epi_smem,
                )
            else:
                sC_stages = stg.sC.get_tensor(epi_smem.outer, swizzle=epi_smem.inner)
            if cutlass.const_expr(not cfg.epi.HAS_STORE_WARP):
                sC = cute.slice_(sC_stages, (None, None, 0))
        else:
            gD_t = cute.make_tensor(
                gD.iterator, cute.make_layout((N, M), stride=(1, N))
            )
        a_full, a_empty = stg.a_full.data_ptr(), stg.a_empty.data_ptr()
        b_full, b_empty = stg.b_full.data_ptr(), stg.b_empty.data_ptr()
        epi_full = (
            stg.epi_full.data_ptr() if cutlass.const_expr(epi_full_bars > 0) else None
        )
        epi_empty = (
            stg.epi_empty.data_ptr() if cutlass.const_expr(epi_empty_bars > 0) else None
        )
        sfull, sempty = stg.sfull.data_ptr(), stg.sempty.data_ptr()
        sWork = stg.work.get_tensor(cute.make_layout((cfg.sched_stages, cfg.fields)))

        if warp_idx == 0:
            with cute.arch.elect_one():
                for s in cutlass.range_constexpr(cfg.ab_stage):
                    cute.arch.mbarrier_init(a_full + s, 1)
                    cute.arch.mbarrier_init(a_empty + s, self.mma_threads)
                    cute.arch.mbarrier_init(b_full + s, 1)
                    cute.arch.mbarrier_init(b_empty + s, self.mma_threads)
                for s in cutlass.range_constexpr(epi_full_bars):
                    cute.arch.mbarrier_init(epi_full + s, cfg.epi.full_barrier_arrivals)
                for s in cutlass.range_constexpr(epi_empty_bars):
                    cute.arch.mbarrier_init(
                        epi_empty + s, cfg.epi.empty_barrier_arrivals
                    )
                for s in cutlass.range_constexpr(cfg.sched_stages):
                    cute.arch.mbarrier_init(sfull + s, 1)
                    cute.arch.mbarrier_init(sempty + s, self.num_sched_consumers)
        cute.arch.mbarrier_init_fence()
        cute.arch.barrier()

        if cutlass.const_expr(cfg.epi.HAS_R2S):
            tiled_s2r = cfg.epi.make_tiled_s2r(cfg.TILE)

        if cfg.is_prod_wg(warp_idx):
            cute.arch.setmaxregister_decrease(self.reg_prod)

            if warp_idx == self.sched_warp:
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

            if warp_idx == self.load_warp_0:
                wstage, wphase = i32(0), i32(1)
                epi_empty_phase = i32(1)
                cons = moe_scheduler.MoeSchedConsumer.create(cfg.sched_stages)
                tile = cons.get_next_tile(sWork, sfull, sempty)
                while tile.valid != i32(0):
                    if cutlass.const_expr(cfg.union_smem):
                        cute.arch.mbarrier_wait(epi_empty, epi_empty_phase)
                        epi_empty_phase ^= 1
                    if cutlass.const_expr(swap):
                        (tAsA, tAgA, tGsG, tGgG, tAsSFA, tAgSFA, tGsSFG, tGgSFG) = (
                            make_a_sfa_gate_partitions_swap(
                                tma_atom_a,
                                tma_atom_sfa,
                                tma_tensor_a,
                                tma_tensor_sfa,
                                sA,
                                sG,
                                sSFA,
                                sSFG,
                                cfg.TILE,
                                tile,
                                N,
                            )
                        )
                        wstage, wphase = copy_a_sfa_gate_swap(
                            tma_atom_a,
                            tma_atom_sfa,
                            tAgA,
                            tAsA,
                            tGgG,
                            tGsG,
                            tAgSFA,
                            tAsSFA,
                            tGgSFG,
                            tGsSFG,
                            tile,
                            a_full,
                            a_empty,
                            self.a_sfa_bytes,
                            k_tile_count,
                            cfg.ab_stage,
                            wstage,
                            wphase,
                        )
                    else:
                        tAsA, tAgA, tAsSFA, tAgSFA, sfa_m_block = make_a_sfa_partitions(
                            tma_atom_a,
                            tma_atom_sfa,
                            tma_tensor_a,
                            tma_tensor_sfa,
                            sA,
                            sSFA,
                            cfg.TILE,
                            tile,
                            sfa_tile_m,
                            sfa_tiles_per_block,
                            swap,
                        )
                        wstage, wphase = copy_a_sfa(
                            tma_atom_a,
                            tma_atom_sfa,
                            tAgA,
                            tAsA,
                            tAgSFA,
                            tAsSFA,
                            tile,
                            sfa_m_block,
                            a_full,
                            a_empty,
                            self.a_sfa_bytes,
                            k_tile_count,
                            cfg.ab_stage,
                            wstage,
                            wphase,
                            swap,
                        )
                    tile = cons.get_next_tile(sWork, sfull, sempty)

            if warp_idx == self.load_warp_1:
                wstage, wphase = i32(0), i32(1)
                epi_empty_phase = i32(1)
                cons = moe_scheduler.MoeSchedConsumer.create(cfg.sched_stages)
                tile = cons.get_next_tile(sWork, sfull, sempty)
                while tile.valid != i32(0):
                    if cutlass.const_expr(cfg.union_smem):
                        cute.arch.mbarrier_wait(epi_empty, epi_empty_phase)
                        epi_empty_phase ^= 1
                    if cutlass.const_expr(swap):
                        tBsB, tBgB, tBsSFB, tBgSFB, sfb_n_block = make_b_sfb_partitions(
                            tma_atom_b,
                            tma_atom_sfb,
                            tma_tensor_b,
                            tma_tensor_sfb,
                            sB,
                            sSFB,
                            cfg.TILE,
                            tile,
                            sfb_tile_n,
                            sfb_tiles_per_block,
                            swap,
                        )
                        wstage, wphase = copy_b_sfb(
                            tma_atom_b,
                            tma_atom_sfb,
                            tBgB,
                            tBsB,
                            tBgSFB,
                            tBsSFB,
                            tile,
                            sfb_n_block,
                            b_full,
                            b_empty,
                            self.b_sfb_bytes,
                            k_tile_count,
                            cfg.ab_stage,
                            wstage,
                            wphase,
                            swap,
                        )
                    else:
                        (tBsB, tBgB, tGsG, tGgG, tBsSFB, tBgSFB, tGsSFG, tGgSFG) = (
                            make_b_sfb_gate_partitions(
                                tma_atom_b,
                                tma_atom_sfb,
                                tma_tensor_b,
                                tma_tensor_sfb,
                                sB,
                                sG,
                                sSFB,
                                sSFG,
                                cfg.TILE,
                                tile,
                                N,
                            )
                        )
                        wstage, wphase = copy_b_sfb_gate(
                            tma_atom_b,
                            tma_atom_sfb,
                            tBgB,
                            tBsB,
                            tGgG,
                            tGsG,
                            tBgSFB,
                            tBsSFB,
                            tGgSFG,
                            tGsSFG,
                            tile,
                            b_full,
                            b_empty,
                            self.b_sfb_bytes,
                            k_tile_count,
                            cfg.ab_stage,
                            wstage,
                            wphase,
                        )
                    tile = cons.get_next_tile(sWork, sfull, sempty)

        else:
            cute.arch.setmaxregister_increase(self.reg_math)
            thr = tiledmma.get_slice(tidx)
            if cutlass.const_expr(cfg.epi.HAS_R2S):
                thr_r2s = tiled_r2s.get_slice(tidx)
                thr_s2r = tiled_s2r.get_slice(tidx)
            rstage, rphase = i32(0), i32(0)
            cons = moe_scheduler.MoeSchedConsumer.create(cfg.sched_stages)
            tile = cons.get_next_tile(sWork, sfull, sempty)
            while tile.valid != i32(0):
                if cutlass.const_expr(sfa_tiles_per_block > 1):
                    sfa_tile_offset = tile.m_block & i32(sfa_tiles_per_block - 1)
                    sSFA_tile = cute.local_tile(
                        sSFA,
                        (bm, cute.size(sSFA, mode=[1])),
                        (sfa_tile_offset, 0, None),
                    )
                else:
                    sSFA_tile = sSFA
                if cutlass.const_expr(sfb_tiles_per_block > 1):
                    sfb_tile_offset = tile.n_block & i32(sfb_tiles_per_block - 1)
                    sSFB_tile = cute.local_tile(
                        sSFB,
                        (bn, cute.size(sSFB, mode=[1])),
                        (sfb_tile_offset, 0, None),
                    )
                else:
                    sSFB_tile = sSFB
                if cutlass.const_expr(swap):
                    acc, rstage, rphase = mma_swap(
                        tiledmma,
                        self.mma,
                        self.sf,
                        cfg.activation,
                        sA,
                        sG,
                        sB,
                        sSFA_tile,
                        sSFG,
                        sSFB_tile,
                        (bm, bn),
                        cfg.ACC,
                        cfg.load_ab.a_dtype,
                        cfg.load_ab.b_smem_dtype,
                        self.a_is_m_major,
                        self.b_is_n_major,
                        cfg.load_ab.b_unpack_bits,
                        tidx,
                        a_full,
                        a_empty,
                        b_full,
                        b_empty,
                        k_tile_count,
                        cfg.ab_stage,
                        rstage,
                        rphase,
                        up_scale[tile.group].to(cutlass.Float32),
                        gate_scale[tile.group].to(cutlass.Float32),
                    )
                else:
                    acc, rstage, rphase = mma(
                        tiledmma,
                        self.mma,
                        self.sf,
                        cfg.activation,
                        sA,
                        sB,
                        sG,
                        sSFA_tile,
                        sSFB_tile,
                        sSFG,
                        (bm, bn),
                        cfg.ACC,
                        cfg.load_ab.a_dtype,
                        cfg.load_ab.b_smem_dtype,
                        self.a_is_m_major,
                        self.b_is_n_major,
                        cfg.load_ab.b_unpack_bits,
                        tidx,
                        a_full,
                        a_empty,
                        b_full,
                        b_empty,
                        k_tile_count,
                        cfg.ab_stage,
                        rstage,
                        rphase,
                        up_scale[tile.group].to(cutlass.Float32),
                        gate_scale[tile.group].to(cutlass.Float32),
                    )
                if cutlass.const_expr(cfg.epi.METHOD is EpiMethod.DIRECT_STG):
                    moe_epilogue.store_swap(
                        acc, thr, gD_t, tile, (bm, bn), N, cfg.epi.out_dtype
                    )
                else:
                    moe_epilogue.store_wg(
                        cfg.epi,
                        acc,
                        thr_r2s,
                        thr_s2r,
                        sC,
                        gD,
                        tile,
                        (bm, bn),
                        epi_empty,
                        cfg.epi_bar_id,
                        cfg.mma_threads,
                    )
                tile = cons.get_next_tile(sWork, sfull, sempty)


def _stream():
    return cuda.CUstream(torch.cuda.current_stream().cuda_stream)


def make_args(
    a_q, a_scale, b_q, b_scale, out, m_indptr, up_scale, gate_scale, epi_cfg, tile
):
    sf = lambda tensor: from_dlpack(
        tensor.contiguous(), assumed_align=16
    ).mark_layout_dynamic()
    epi_cfg.check_output(tile, out)
    if epi_cfg.STORE_BITS == 128:
        out_arg = (
            from_dlpack(out, assumed_align=epi_cfg.store_bytes)
            .mark_layout_dynamic(leading_dim=1)
            .mark_compact_shape_dynamic(
                mode=1, stride_order=(0, 1), divisibility=epi_cfg.store_elements
            )
        )
    else:
        out_arg = from_dlpack(out).mark_layout_dynamic()
    return (
        from_dlpack(a_q).mark_layout_dynamic(),
        from_dlpack(b_q).mark_layout_dynamic(),
        sf(a_scale),
        sf(b_scale),
        out_arg,
        from_dlpack(m_indptr).mark_layout_dynamic(),
        from_dlpack(up_scale).mark_layout_dynamic(),
        from_dlpack(gate_scale).mark_layout_dynamic(),
        _stream(),
    )
