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
"""CuteDSL NVFP4 grouped FC1 activation and Q1 kernel for SM120a."""

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
from ....utils import round_up as align
from ._moe_utils import moe_activation, moe_epilogue, moe_scheduler
from ._moe_utils.moe_epilogue import (
    EPI_CONFIGS,
    EpiMethod,
    WgQuantSwapEpiConfig,
)
from ._moe_utils.moe_kernel_builder import (
    LoadABConfig,
    MmaConfig,
    Sm12xDecodeFc1GemmConfig,
    Sm12xGatedGemmConfig,
    Sm12xUnaryGemmConfig,
)
from ._moe_utils.sm12x_blockscaled_layout import (
    Sm120SfConfigNvfp4,
    compute_padded_offset,
)

ATOM_MNK = (16, 8, 64)


def is_swapab(tile):
    return ATOM_MNK[0] > tile[0]


REG_PROD_BY_TACTIC = {
    (128, 64, EpiMethod.WG_S2R_QUANT): 40,
    (64, 64, EpiMethod.WG_S2R_QUANT): 88,
    (32, 64, EpiMethod.WG_S2R_QUANT): 88,
    (128, 128, EpiMethod.WG_S2R_QUANT): 40,
    (64, 128, EpiMethod.WG_S2R_QUANT): 88,
    (32, 128, EpiMethod.WG_S2R_QUANT): 88,
}


GATHER_REG_PROD_BY_TACTIC = {
    (128, 64, EpiMethod.WG_S2R_QUANT): 32,
    (64, 64, EpiMethod.WG_S2R_QUANT): 88,
    (32, 64, EpiMethod.WG_S2R_QUANT): 88,
    (128, 128, EpiMethod.WG_S2R_QUANT): 64,
    (64, 128, EpiMethod.WG_S2R_QUANT): 88,
    (32, 128, EpiMethod.WG_S2R_QUANT): 88,
}


def make_cfg(
    tile,
    ab_stage,
    epi=EpiMethod.WG_S2R_QUANT,
    *,
    activation=ActivationType.Swiglu,
    fastmath=False,
    enable_pdl=False,
    swiglu_limit=None,
    situ_beta=SITU_BETA,
    situ_linear_beta=SITU_LINEAR_BETA,
    gather=False,
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
        (128, 64, 128),
        (64, 64, 128),
        (32, 64, 128),
    )
    assert epi is EpiMethod.WG_S2R_QUANT
    tactic = (tile[0], tile[1], epi)
    gated = activation is not ActivationType.Relu2
    config_cls = Sm12xGatedGemmConfig if gated else Sm12xUnaryGemmConfig
    activation_op = (
        moe_activation.make_gated_activation(
            activation,
            fastmath,
            swiglu_limit=swiglu_limit,
            situ_beta=situ_beta,
            situ_linear_beta=situ_linear_beta,
        )
        if gated
        else moe_activation.ReLU2Activation()
    )
    return config_cls(
        MmaConfig(warp_mma.MmaMXF4NVF4Op(fp4, f32, sf), tile[:2], 8),
        LoadABConfig(tile, ab_stage, fp4, fp4),
        Sm120SfConfigNvfp4(sf),
        EPI_CONFIGS[epi](bf16, 8 * 32, store_threads=8 * 32),
        ab_stage,
        tile,
        epi_bar_id=3,
        union_smem=True,
        reg_prod=(
            GATHER_REG_PROD_BY_TACTIC[tactic] if gather else REG_PROD_BY_TACTIC[tactic]
        ),
        activation=activation_op,
        enable_pdl=enable_pdl,
    )


@cute.jit
def make_a_sfa_gather_partitions(tiled_copy_a, gSFA_u8, sA, tidx):
    tAsA = tiled_copy_a.get_slice(tidx & 31).partition_D(sA)
    gSFA_i32 = cute.recast_ptr(gSFA_u8.iterator, dtype=cutlass.Int32)
    return tAsA, gSFA_i32


@cute.jit
def gather_a_sfa(
    g2s_a,
    g2s_sfa,
    gA_u8,
    gSFA_i32,
    token_map,
    tAsA,
    sSFA,
    tile,
    a_full,
    a_empty,
    k_tile_count,
    ab_stages,
    stage,
    phase,
    K,
    tile_mnk: cutlass.Constexpr,
):
    bm, _, bk = tile_mnk
    lane = cute.arch.lane_idx()
    begin = tile.m_offset + tile.m_block * bm
    rows = max(cutlass.Int32(0), min(tile.m_boundary - begin, cutlass.Int32(bm)))
    source_layout = cute.make_layout(((16, 1),), stride=((1, 0),))
    for kt in cutlass.range(0, k_tile_count):
        cute.arch.mbarrier_wait(a_empty + stage, phase)
        cute.arch.sync_warp()
        for part in cutlass.range_constexpr(cute.size(tAsA, mode=[1])):
            row = (lane >> 3) + part * 4
            if row < rows:
                token = token_map[begin + row]
                address = (
                    gA_u8.iterator + token * (K >> 1) + kt * (bk >> 1) + (lane & 7) * 8
                )
                pointer = cute.make_ptr(
                    cutlass.Float4E2M1FN,
                    address.toint(),
                    cute.AddressSpace.gmem,
                    assumed_align=8,
                )
                source = cute.make_tensor(pointer, source_layout)
                cute.copy(g2s_a, source, tAsA[None, part, 0, stage])
        for part in cutlass.range_constexpr(ceil_div(bm, 32)):
            row = lane + part * 32
            if row < rows:
                token = token_map[begin + row]
                for word in cutlass.range_constexpr(cute.size(sSFA, mode=[1])):
                    source = cute.make_tensor(
                        gSFA_i32 + token * (K >> 6) + kt * (bk >> 6) + word,
                        cute.make_layout(1),
                    )
                    target = cute.make_tensor(
                        sSFA.iterator + cute.crd2idx((row, word, stage), sSFA.layout),
                        cute.make_layout(1),
                    )
                    cute.copy(g2s_sfa, source, target)
        cute.arch.cp_async_mbarrier_arrive_noinc(a_full + stage)
        stage += 1
        if stage == ab_stages:
            stage = cutlass.Int32(0)
            phase ^= 1
    return stage, phase


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
):
    cta_layout = cute.make_layout(1)
    mA = cute.domain_offset((tile.m_offset, 0), tma_tensor_a)
    gA_mkl = cute.local_tile(mA, cute.slice_(tile_mnk, (None, 0, None)), (None, None))
    tAsA, tAgA = cpasync.tma_partition(
        tma_atom_a,
        cutlass.Int32(0),
        cta_layout,
        cute.group_modes(sA, 0, 2),
        cute.group_modes(gA_mkl, 0, 2),
    )
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
):
    cta_layout = cute.make_layout(1)
    gB_nkl = cute.local_tile(
        tma_tensor_b, cute.slice_(tile_mnk, (0, None, None)), (None, None, None)
    )
    gSFB_nkl = cute.local_tile(
        tma_tensor_sfb, (sfb_tile_n, cute.size(sSFB, mode=[1])), (None, None, None)
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
    sfb_n_block = tile.n_block
    if cutlass.const_expr(sfb_tiles_per_block == 2):
        sfb_n_block = tile.n_block >> 1
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
):
    for kt in cutlass.range(0, k_tile_count):
        cute.arch.mbarrier_wait(a_empty + stage, phase)
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(a_full + stage, tx_bytes)
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
            tma_atom_sfb,
            tBgSFB[(None, sfb_n_block, kt, tile.group)],
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
        tma_tensor_sfb, (tile_mnk[1], cute.size(sSFB, mode=[1])), (None, None, None)
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
    scale,
    gated: cutlass.Constexpr = True,
):
    bm, bn = tile_mn
    thr = tiledmma.get_slice(tidx)
    tCrA = tiledmma.make_fragment_A(thr.partition_A(sA)[None, None, None, 0])
    tCrB = tiledmma.make_fragment_B(thr.partition_B(sB)[None, None, None, 0])
    if cutlass.const_expr(gated):
        tCrG = tiledmma.make_fragment_B(thr.partition_B(sG)[None, None, None, 0])
    shape_c = tiledmma.partition_shape_C((bm, bn))
    acc = cute.make_rmem_tensor(shape_c, acc_dtype)
    if cutlass.const_expr(gated):
        acc_up = cute.make_rmem_tensor(shape_c, acc_dtype)
    s2r_a = mma_cfg.make_s2r_a(tiledmma, a_dtype, a_is_m_major)
    s2r_b = mma_cfg.make_s2r_b(tiledmma, b_dtype, b_is_n_major, unpack_bits)
    thr_a, thr_b = s2r_a.get_slice(tidx), s2r_b.get_slice(tidx)
    tCrA_v = thr_a.retile(tCrA)
    tCrB_v = thr_b.retile(tCrB)
    if cutlass.const_expr(gated):
        tCrG_v = thr_b.retile(tCrG)
    tCrSFA = sf_cfg.partition_fragment_SFA(sSFA[None, None, 0], thr, tidx)
    tCrSFB = sf_cfg.partition_fragment_SFB(sSFB[None, None, 0], thr, tidx)
    if cutlass.const_expr(gated):
        tCrSFG = sf_cfg.partition_fragment_SFB(sSFG[None, None, 0], thr, tidx)
    tCrSFA_frg = sf_cfg.make_sfa_e4m3_view(tCrSFA)
    tCrSFB_frg = sf_cfg.make_sfb_e4m3_view(tCrSFB)
    if cutlass.const_expr(gated):
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
    tCrSFB_v = thr_sfb.retile(tCrSFB)
    if cutlass.const_expr(gated):
        tCrSFG_v = thr_sfb.retile(tCrSFG)
    k_blocks = cute.size(tCrA_v, mode=[2])
    sf_blocks = cute.size(tCrSFA_v, mode=[2])
    acc.fill(0.0)
    if cutlass.const_expr(gated):
        acc_up.fill(0.0)
    for _ in cutlass.range(0, k_tile_count):
        cute.arch.mbarrier_wait(a_full + stage, phase)
        cute.arch.mbarrier_wait(b_full + stage, phase)
        tAsA = thr_a.partition_S(sA)[None, None, None, stage]
        tBsB = thr_b.partition_S(sB)[None, None, None, stage]
        if cutlass.const_expr(gated):
            tGsG = thr_b.partition_S(sG)[None, None, None, stage]
        tAsSFA = thr_sfa.partition_S(sSFA)[None, None, None, stage]
        tBsSFB = thr_sfb.partition_S(sSFB)[None, None, None, stage]
        if cutlass.const_expr(gated):
            tGsSFG = thr_sfb.partition_S(sSFG)[None, None, None, stage]
        for k in cutlass.range_constexpr(0, k_blocks):
            cute.copy(s2r_a, tAsA[None, None, k], tCrA_v[None, None, k])
            cute.copy(s2r_b, tBsB[None, None, k], tCrB_v[None, None, k])
            if cutlass.const_expr(gated):
                cute.copy(s2r_b, tGsG[None, None, k], tCrG_v[None, None, k])
        for k in cutlass.range_constexpr(0, sf_blocks):
            cute.copy(s2r_sfa, tAsSFA[None, None, k], tCrSFA_v[None, None, k])
            cute.copy(s2r_sfb, tBsSFB[None, None, k], tCrSFB_v[None, None, k])
            if cutlass.const_expr(gated):
                cute.copy(s2r_sfb, tGsSFG[None, None, k], tCrSFG_v[None, None, k])
        cute.gemm(tiledmma, acc, [tCrA, tCrSFA_frg], [tCrB, tCrSFB_frg], acc)
        if cutlass.const_expr(gated):
            cute.gemm(tiledmma, acc_up, [tCrA, tCrSFA_frg], [tCrG, tCrSFG_frg], acc_up)
        cute.arch.mbarrier_arrive(a_empty + stage)
        cute.arch.mbarrier_arrive(b_empty + stage)
        stage += 1
        if stage == ab_stages:
            stage = cutlass.Int32(0)
            phase ^= 1
    acc.store(acc.load() * scale)
    if cutlass.const_expr(gated):
        acc_up.store(acc_up.load() * scale)
        activation(acc_up, acc)
    else:
        activation(acc)
    if cutlass.const_expr(gated):
        return acc_up, stage, phase
    else:
        return acc, stage, phase


class CuteDslSm120MoeNvfp4Fc1ActQ1:
    def __init__(self, cfg, grid_x, gather=False):
        self.cfg = cfg
        self.grid_x = grid_x
        self.gather = gather
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
        gQ_u8: cute.Tensor,
        gSFQ_u8: cute.Tensor,
        offsets: cute.Tensor,
        weight_global_scale: cute.Tensor,
        input_global_scale: cute.Tensor,
        global_scale: cute.Tensor,
        stream,
        token_map=None,
    ):
        cfg = self.cfg
        assert self.gather == (token_map is not None)
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
        self.single_expert = False
        if cutlass.const_expr(cute.is_static(E)):
            self.single_expert = E == 1 and not self.gather
        gQ = cute.recast_tensor(gQ_u8, cutlass.Float4E2M1FN)
        gSFQ = cute.make_tensor(gSFQ_u8.iterator, cute.make_layout(cute.size(gSFQ_u8)))
        sfq_cols = cute.size(gSFQ_u8, mode=[1])
        gW = cute.make_tensor(
            gB_e.iterator, cute.make_layout((N2, K, E), stride=(K, 1, N2 * K))
        )
        bm, bn, bk = cfg.TILE[0], cfg.TILE[1], cfg.TILE[2]
        m_padded_sf = cute.size(gSFA_u8, mode=[0])
        t_a, t_b = gA, gW
        gSFA = None
        if cutlass.const_expr(not self.gather):
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
        gate_smem = b_smem
        gate_sf_smem = sfb_smem
        epi_smem = cfg.epi.make_smem_layout(cfg.TILE)
        self.a_is_m_major = a_layout.is_m_major_a()
        self.b_is_n_major = b_layout.is_n_major_b()

        a_stage = cute.slice_(a_smem, (None, None, 0))
        b_stage = cute.slice_(b_smem, (None, None, 0))
        sfa_stage = cute.slice_(sfa_smem, (None, None, 0))
        sfb_stage = cute.slice_(sfb_smem, (None, None, 0))
        tma_atom_a, tma_tensor_a, tma_atom_sfa, tma_tensor_sfa = None, None, None, None
        g2s_a, tiled_copy_a, g2s_sfa = None, None, None
        if cutlass.const_expr(self.gather):
            g2s_a = cute.make_copy_atom(
                cpasync.CopyG2SOp(), cutlass.Float4E2M1FN, num_bits_per_copy=64
            )
            tiled_copy_a = cute.make_tiled_copy_tv(
                g2s_a,
                cute.make_ordered_layout((4, 8), order=(1, 0)),
                cute.make_layout((1, 16)),
            )
            g2s_sfa = cute.make_copy_atom(
                cpasync.CopyG2SOp(), cutlass.Int32, num_bits_per_copy=32
            )
        else:
            tma_atom_a, tma_tensor_a = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileG2SOp(),
                t_a,
                a_stage,
                (bm, bk),
                num_multicast=1,
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
        self.a_sfa_bytes = cute.size_in_bytes(
            cfg.load_ab.a_dtype, a_stage
        ) + cute.size_in_bytes(cfg.load_sf.sf_dtype, sfa_stage)
        self.b_sfb_bytes = (
            cute.size_in_bytes(cfg.load_ab.b_dtype, b_stage)
            + cute.size_in_bytes(cfg.load_sf.sf_dtype, sfb_stage)
        ) * (2 if cutlass.const_expr(cfg.gated) else 1)

        epi_full_bars = cfg.epi.num_full_barriers
        epi_empty_bars = cfg.epi.num_empty_barriers
        epi_elems = cute.cosize(epi_smem) if cfg.owns_epi_smem else 0
        gate_elems = cute.cosize(gate_smem) if cfg.gated else 0
        gate_sf_elems = cute.cosize(gate_sf_smem) if cfg.gated else 0

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
                cute.struct.MemRange[cfg.load_ab.a_dtype, gate_elems], 128
            ]
            sSFA: cute.struct.Align[
                cute.struct.MemRange[cfg.load_sf.sf_dtype, cute.cosize(sfa_smem)], 128
            ]
            sSFB: cute.struct.Align[
                cute.struct.MemRange[cfg.load_sf.sf_dtype, cute.cosize(sfb_smem)], 128
            ]
            sSFG: cute.struct.Align[
                cute.struct.MemRange[cfg.load_sf.sf_dtype, gate_sf_elems], 128
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
            gQ,
            gSFQ,
            offsets,
            weight_global_scale,
            input_global_scale,
            global_scale,
            sfq_cols,
            a_smem,
            b_smem,
            sfa_smem,
            sfb_smem,
            epi_smem,
            gA_u8,
            gSFA_u8,
            token_map,
            g2s_a,
            tiled_copy_a,
            g2s_sfa,
        ).launch(
            grid=[self.grid_x, 1, 1],
            block=[self.threads, 1, 1],
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=cfg.enable_pdl,
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
        tma_tensor_a,
        tma_tensor_b,
        tma_tensor_sfa,
        tma_tensor_sfb,
        gQ: cute.Tensor,
        gSFQ: cute.Tensor,
        offsets: cute.Tensor,
        weight_global_scale: cute.Tensor,
        input_global_scale: cute.Tensor,
        global_scale: cute.Tensor,
        sfq_cols,
        a_smem,
        b_smem,
        sfa_smem,
        sfb_smem,
        epi_smem,
        gA_u8: cute.Tensor,
        gSFA_u8: cute.Tensor,
        token_map,
        g2s_a,
        tiled_copy_a,
        g2s_sfa,
    ):
        cfg = self.cfg
        epi_full_bars = cfg.epi.num_full_barriers
        epi_empty_bars = cfg.epi.num_empty_barriers
        i32 = cutlass.Int32
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        bidx, _, _ = cute.arch.block_idx()
        bm, bn, bk = cfg.TILE[0], cfg.TILE[1], cfg.TILE[2]
        gate_smem = b_smem
        gate_sf_smem = sfb_smem
        M, N = cute.size(gQ, mode=[0]), cute.size(gQ, mode=[1])
        if cutlass.const_expr(self.gather):
            K = cute.size(gA_u8, mode=[1]) * 2
        else:
            K = cute.size(tma_tensor_a, mode=[1])
        if cutlass.const_expr(not self.single_expert):
            num_groups = cute.size(offsets, mode=[0]) - 1
            num_n_blocks = ceil_div(N, bn)
        k_tile_count = ceil_div(K, bk)
        sfa_tile_m = cfg.load_sf.sfa_tile_m(bm)
        sfa_tiles_per_block = cfg.load_sf.sfa_tiles_per_block(bm)
        sfb_tile_n = cfg.load_sf.sfb_tile_n(bn)
        sfb_tiles_per_block = cfg.load_sf.sfb_tiles_per_block(bn)

        smem = cutlass.utils.SmemAllocator()
        stg = smem.allocate(self.storage)
        sA = stg.sA.get_tensor(a_smem.outer, swizzle=a_smem.inner)
        sB = stg.sB.get_tensor(b_smem.outer, swizzle=b_smem.inner)
        sG, sSFG = None, None
        if cutlass.const_expr(cfg.gated):
            sG = stg.sG.get_tensor(gate_smem.outer, swizzle=gate_smem.inner)
            sSFG = stg.sSFG.get_tensor(gate_sf_smem)
        sSFA = stg.sSFA.get_tensor(sfa_smem)
        sSFB = stg.sSFB.get_tensor(sfb_smem)
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
                    cute.arch.mbarrier_init(
                        a_full + s, 32 if cutlass.const_expr(self.gather) else 1
                    )
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
                prod = moe_scheduler.MoeSchedProducer.create(cfg.sched_stages)
                if cutlass.const_expr(self.single_expert):
                    sched = moe_scheduler.StaticTileScheduler.create(
                        moe_scheduler.GemmType.NORMAL,
                        bm,
                        bn,
                        M,
                        N,
                        1,
                        self.grid_x,
                        bidx,
                    )
                    has, dense_m, dense_n = sched.get_next_block()
                    while has:
                        m_tile, n_tile = dense_m, dense_n
                        prod.publish(
                            sWork,
                            sfull,
                            sempty,
                            moe_scheduler.MoeWorkTile(
                                m_tile, n_tile, i32(0), i32(0), M, i32(1)
                            ),
                        )
                        has, dense_m, dense_n = sched.get_next_block()
                else:
                    sched = moe_scheduler.MoeTileScheduler.create(
                        bm, num_groups, num_n_blocks, self.grid_x, bidx, offsets
                    )
                    has, e, m_tile, n_tile, m_off, m_bnd = sched.get_next_block(offsets)
                    while has:
                        prod.publish(
                            sWork,
                            sfull,
                            sempty,
                            moe_scheduler.MoeWorkTile(
                                m_tile, n_tile, e, m_off, m_bnd, i32(1)
                            ),
                        )
                        has, e, m_tile, n_tile, m_off, m_bnd = sched.get_next_block(
                            offsets
                        )
                prod.publish_sentinel(sWork, sfull, sempty)

            if warp_idx == self.load_warp_0:
                if cutlass.const_expr(cfg.enable_pdl):
                    cute.arch.griddepcontrol_wait()
                wstage, wphase = i32(0), i32(1)
                epi_empty_phase = i32(1)
                cons = moe_scheduler.MoeSchedConsumer.create(cfg.sched_stages)
                tile = cons.get_next_tile(sWork, sfull, sempty)
                while tile.valid != i32(0):
                    if cutlass.const_expr(cfg.union_smem):
                        cute.arch.mbarrier_wait(epi_empty, epi_empty_phase)
                        epi_empty_phase ^= 1
                    if cutlass.const_expr(self.gather):
                        tAsA, gSFA_i32 = make_a_sfa_gather_partitions(
                            tiled_copy_a, gSFA_u8, sA, tidx
                        )
                        wstage, wphase = gather_a_sfa(
                            g2s_a,
                            g2s_sfa,
                            gA_u8,
                            gSFA_i32,
                            token_map,
                            tAsA,
                            sSFA,
                            tile,
                            a_full,
                            a_empty,
                            k_tile_count,
                            cfg.ab_stage,
                            wstage,
                            wphase,
                            K,
                            cfg.TILE,
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
                    if cutlass.const_expr(not cfg.gated):
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
            if cutlass.const_expr(cfg.epi.HAS_R2S):
                thr_r2s = tiled_r2s.get_slice(tidx)
                thr_s2r = tiled_s2r.get_slice(tidx)
            rstage, rphase = i32(0), i32(0)
            if cutlass.const_expr(cfg.enable_pdl):
                cute.arch.griddepcontrol_wait()
            cons = moe_scheduler.MoeSchedConsumer.create(cfg.sched_stages)
            tile = cons.get_next_tile(sWork, sfull, sempty)
            while tile.valid != i32(0):
                if cutlass.const_expr(self.gather):
                    sSFA_tile = cute.local_tile(
                        sSFA, (bm, cute.size(sSFA, mode=[1])), (0, 0, None)
                    )
                elif cutlass.const_expr(sfa_tiles_per_block > 1):
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
                    (weight_global_scale[tile.group] / input_global_scale[0]).to(
                        cutlass.Float32
                    ),
                    gated=cfg.gated,
                )
                if cutlass.const_expr(cfg.enable_pdl):
                    next_tile = cons.get_next_tile(sWork, sfull, sempty)
                    if next_tile.valid == i32(0):
                        cute.arch.griddepcontrol_launch_dependents()
                moe_epilogue.store_wg_nvfp4_q1(
                    acc,
                    thr_r2s,
                    thr_s2r,
                    sC,
                    gQ,
                    gSFQ,
                    global_scale,
                    tile,
                    (bm, bn),
                    sfq_cols,
                    epi_empty,
                    cfg.epi_bar_id,
                    cfg.mma_threads,
                )
                if cutlass.const_expr(cfg.enable_pdl):
                    tile = next_tile
                else:
                    tile = cons.get_next_tile(sWork, sfull, sempty)


def make_decode_cfg(
    tile,
    ab_stage,
    epi=EpiMethod.WG_QUANT_SWAP,
    *,
    activation=ActivationType.Swiglu,
    fastmath=False,
    enable_pdl=False,
    swiglu_limit=None,
    situ_beta=SITU_BETA,
    situ_linear_beta=SITU_LINEAR_BETA,
    gather=False,
    k=None,
):
    assert epi == EpiMethod.WG_QUANT_SWAP
    del gather
    fused_gated = tile[1] == 64 and activation is not ActivationType.Relu2
    tile = (tile[1] * (2 if fused_gated else 1), tile[0], tile[2])
    store_tile = (tile[0] // 2, tile[1], tile[2]) if fused_gated else tile
    fp4, sf, bf16 = cutlass.Float4E2M1FN, cutlass.Float8E4M3FN, cutlass.BFloat16
    mma = MmaConfig(
        warp_mma.MmaMXF4NVF4Op(fp4, cutlass.Float32, sf), tile[:2], 8, swap_ab=True
    )
    threads = mma.num_math_warps * Sm12xDecodeFc1GemmConfig.WARP_SIZE
    gated = activation is not ActivationType.Relu2
    activation_op = (
        moe_activation.make_gated_activation(
            activation,
            fastmath,
            swiglu_limit=swiglu_limit,
            situ_beta=situ_beta,
            situ_linear_beta=situ_linear_beta,
        )
        if gated
        else moe_activation.ReLU2Activation()
    )
    return Sm12xDecodeFc1GemmConfig(
        mma,
        LoadABConfig(tile, ab_stage, fp4, fp4),
        Sm120SfConfigNvfp4(sf),
        WgQuantSwapEpiConfig(
            bf16,
            threads,
            store_threads=WgQuantSwapEpiConfig.store_threads_for(bf16, store_tile),
        ),
        ab_stage,
        tile,
        activation=activation_op,
        gated=gated,
        fused_gated=fused_gated,
        k=k,
        enable_pdl=enable_pdl,
    )


@cute.jit
def copy_stage(
    gW_u8,
    gA_u8,
    gSFA,
    gSFB,
    gSFB_i32,
    token_map,
    tAsA,
    tGsG,
    tBsB,
    sSFA,
    sSFG,
    sSFB,
    atom,
    sf_atom,
    tile,
    kt,
    stage,
    tidx,
    cfg,
    gather: cutlass.Constexpr,
):
    bm, bn, bk = cfg.TILE
    row_base = cutlass.Int64(tile.group) * cute.size(gW_u8, mode=[1])
    k_bytes = cute.size(gW_u8, mode=[2])
    weight_values = cfg.G2S_BITS // cfg.load_ab.a_dtype.width
    weight_copy_bytes = cfg.G2S_BITS // cutlass.Uint8.width
    weight_lanes_k = ceil_div(bk, weight_values)
    weight_shift_k = weight_lanes_k.bit_length() - 1
    weight_rows_per_warp = cfg.WARP_SIZE >> weight_shift_k
    lane, warp = tidx & (cfg.WARP_SIZE - 1), tidx >> (cfg.WARP_SIZE.bit_length() - 1)
    weight_layout = cute.make_layout(((weight_values, 1),), stride=((1, 0),))
    for step in cutlass.range_constexpr(
        ceil_div(cute.size(tAsA, mode=[1]), cfg.mma.num_math_warps)
    ):
        part = warp + step * cfg.mma.num_math_warps
        row = tile.m_block * bm + (lane >> weight_shift_k) + part * weight_rows_per_warp
        address = (
            gW_u8.iterator
            + (row_base + row) * k_bytes
            + kt * (bk >> 1)
            + (lane & (weight_lanes_k - 1)) * weight_copy_bytes
        )
        tAgA = cute.make_tensor(
            cute.make_ptr(
                cfg.load_ab.a_dtype,
                address.toint(),
                cute.AddressSpace.gmem,
                assumed_align=weight_copy_bytes,
            ),
            weight_layout,
        )
        cute.copy(atom, tAgA, tAsA[None, part, 0, stage])
    shift_m = bm.bit_length() - 1
    weight_words = cute.size(sSFA, mode=[1])
    for step in cutlass.range_constexpr(ceil_div(bm * weight_words, cfg.threads)):
        index = tidx + step * cfg.threads
        row, word = index & (bm - 1), index >> shift_m
        tAgSFA = cute.make_tensor(
            gSFA.iterator
            + cute.crd2idx(
                (tile.m_block * bm + row, kt * weight_words + word, tile.group),
                gSFA.layout,
            ),
            cute.make_layout(1),
        )
        tAsSFA = cute.make_tensor(
            sSFA.iterator + cute.crd2idx((row, word, stage), sSFA.layout),
            cute.make_layout(1),
        )
        cute.copy(sf_atom, tAgSFA, tAsSFA)
    if cutlass.const_expr(cfg.gated):
        row_offset = cute.size(gW_u8, mode=[1]) >> 1
        for step in cutlass.range_constexpr(
            ceil_div(cute.size(tGsG, mode=[1]), cfg.mma.num_math_warps)
        ):
            part = warp + step * cfg.mma.num_math_warps
            row = (
                tile.m_block * bm
                + (lane >> weight_shift_k)
                + part * weight_rows_per_warp
                + row_offset
            )
            address = (
                gW_u8.iterator
                + (row_base + row) * k_bytes
                + kt * (bk >> 1)
                + (lane & (weight_lanes_k - 1)) * weight_copy_bytes
            )
            tGgG = cute.make_tensor(
                cute.make_ptr(
                    cfg.load_ab.a_dtype,
                    address.toint(),
                    cute.AddressSpace.gmem,
                    assumed_align=weight_copy_bytes,
                ),
                weight_layout,
            )
            cute.copy(atom, tGgG, tGsG[None, part, 0, stage])
        gate_words = cute.size(sSFG, mode=[1])
        for step in cutlass.range_constexpr(ceil_div(bm * gate_words, cfg.threads)):
            index = tidx + step * cfg.threads
            row, word = index & (bm - 1), index >> shift_m
            tGgSFG = cute.make_tensor(
                gSFA.iterator
                + cute.crd2idx(
                    (
                        tile.m_block * bm + row + row_offset,
                        kt * gate_words + word,
                        tile.group,
                    ),
                    gSFA.layout,
                ),
                cute.make_layout(1),
            )
            tGsSFG = cute.make_tensor(
                sSFG.iterator + cute.crd2idx((row, word, stage), sSFG.layout),
                cute.make_layout(1),
            )
            cute.copy(sf_atom, tGgSFG, tGsSFG)
    values = cfg.G2S_BITS // cfg.load_ab.b_dtype.width
    copy_bytes = cfg.G2S_BITS // cutlass.Uint8.width
    lanes_k = ceil_div(bk, values)
    shift_k, shift_n = lanes_k.bit_length() - 1, bn.bit_length() - 1
    rows_per_warp = cfg.WARP_SIZE >> shift_k
    lane, warp = tidx & (cfg.WARP_SIZE - 1), tidx >> (cfg.WARP_SIZE.bit_length() - 1)
    begin = tile.m_offset + tile.n_block * bn
    rows = max(cutlass.Int32(0), min(tile.m_boundary - begin, cutlass.Int32(bn)))
    K_bytes = cute.size(gA_u8, mode=[1])
    source_layout = cute.make_layout(((values, 1),), stride=((1, 0),))
    if warp < cute.size(tBsB, mode=[1]):
        row = (lane >> shift_k) + warp * rows_per_warp
        if row < rows:
            source_row = begin + row
            if cutlass.const_expr(gather):
                source_row = token_map[source_row]
            address = (
                gA_u8.iterator
                + cutlass.Int64(source_row) * K_bytes
                + kt * (bk >> 1)
                + (lane & (lanes_k - 1)) * copy_bytes
            )
            tBgB = cute.make_tensor(
                cute.make_ptr(
                    cfg.load_ab.b_dtype,
                    address.toint(),
                    cute.AddressSpace.gmem,
                    assumed_align=copy_bytes,
                ),
                source_layout,
            )
            cute.copy(atom, tBgB, tBsB[None, warp, 0, stage])
    words = cute.size(sSFB, mode=[1])
    sf_begin = (
        compute_padded_offset(tile.m_offset, tile.group, cutlass.Int32(128))
        + tile.n_block * bn
    )
    if tidx < bn * words:
        row, word = tidx & (bn - 1), tidx >> shift_n
        if row < rows:
            if cutlass.const_expr(gather):
                token = token_map[begin + row]
                tBgSFB = cute.make_tensor(
                    gSFB_i32
                    + cutlass.Int64(token) * (K_bytes >> 5)
                    + kt * words
                    + word,
                    cute.make_layout(1),
                )
            else:
                tBgSFB = cute.make_tensor(
                    gSFB.iterator
                    + cute.crd2idx((sf_begin + row, kt * words + word, 0), gSFB.layout),
                    cute.make_layout(1),
                )
            target = cute.make_tensor(
                sSFB.iterator + cute.crd2idx((row, word, stage), sSFB.layout),
                cute.make_layout(1),
            )
            cute.copy(sf_atom, tBgSFB, target)


@cute.jit
def mma_multistage(
    tiledmma,
    cfg,
    activation,
    sA,
    sG,
    sB,
    sSFA,
    sSFG,
    sSFB,
    gW_u8,
    gA_u8,
    gSFA,
    gSFB,
    gSFB_i32,
    token_map,
    tAsA,
    tGsG,
    tBsB,
    atom,
    sf_atom,
    tile,
    tidx,
    k_tile_count,
    scale,
    gather: cutlass.Constexpr,
):
    thr = tiledmma.get_slice(tidx)
    tCrA = tiledmma.make_fragment_A(thr.partition_A(sA)[None, None, None, 0])
    tCrB = tiledmma.make_fragment_B(thr.partition_B(sB)[None, None, None, 0])
    tCrG = None
    if cutlass.const_expr(cfg.gated):
        tCrG = tiledmma.make_fragment_A(thr.partition_A(sG)[None, None, None, 0])
    acc = cute.make_rmem_tensor(
        tiledmma.partition_shape_C(cfg.TILE[:2]), cutlass.Float32
    )
    acc_up = None
    if cutlass.const_expr(cfg.gated):
        acc_up = cute.make_rmem_tensor(
            tiledmma.partition_shape_C(cfg.TILE[:2]), cutlass.Float32
        )
        acc_up.fill(0.0)
    s2r_a = cfg.mma.make_s2r_a(tiledmma, cfg.load_ab.a_smem_dtype, False)
    s2r_b = cfg.mma.make_s2r_b(tiledmma, cfg.load_ab.b_smem_dtype, False)
    thr_a, thr_b = s2r_a.get_slice(tidx), s2r_b.get_slice(tidx)
    tCrA_v, tCrB_v = thr_a.retile(tCrA), thr_b.retile(tCrB)
    tCrG_v = None
    if cutlass.const_expr(cfg.gated):
        tCrG_v = thr_a.retile(tCrG)
    sf_cfg = cfg.load_sf
    tCrSFA = sf_cfg.partition_fragment_SFA(sSFA[None, None, 0], thr, tidx)
    tCrSFB = sf_cfg.partition_fragment_SFB(sSFB[None, None, 0], thr, tidx)
    tCrSFA_frg, tCrSFB_frg = (
        sf_cfg.make_sfa_e4m3_view(tCrSFA),
        sf_cfg.make_sfb_e4m3_view(tCrSFB),
    )
    tCrSFG, tCrSFG_frg = None, None
    if cutlass.const_expr(cfg.gated):
        tCrSFG = sf_cfg.partition_fragment_SFA(sSFG[None, None, 0], thr, tidx)
        tCrSFG_frg = sf_cfg.make_sfa_e4m3_view(tCrSFG)
    s2r_sfa = sf_cfg.make_s2r_sf(
        sf_cfg.get_layoutSFA_TV(tiledmma),
        (cute.size(tiledmma.permutation_mnk[0]), sf_cfg.KTILE_SF),
    )
    s2r_sfb = sf_cfg.make_s2r_sf(
        sf_cfg.get_layoutSFB_TV(tiledmma),
        (cute.size(tiledmma.permutation_mnk[1]), sf_cfg.KTILE_SF),
    )
    thr_sfa, thr_sfb = s2r_sfa.get_slice(tidx), s2r_sfb.get_slice(tidx)
    tCrSFA_v, tCrSFB_v = thr_sfa.retile(tCrSFA), thr_sfb.retile(tCrSFB)
    tCrSFG_v = None
    if cutlass.const_expr(cfg.gated):
        tCrSFG_v = thr_sfa.retile(tCrSFG)
    for initial in cutlass.range_constexpr(cfg.ab_stage - 1):
        if initial < k_tile_count:
            copy_stage(
                gW_u8,
                gA_u8,
                gSFA,
                gSFB,
                gSFB_i32,
                token_map,
                tAsA,
                tGsG,
                tBsB,
                sSFA,
                sSFG,
                sSFB,
                atom,
                sf_atom,
                tile,
                initial,
                initial,
                tidx,
                cfg,
                gather,
            )
        cute.arch.cp_async_commit_group()
    acc.fill(0.0)
    for kt in cutlass.range(k_tile_count):
        stage = kt & (cfg.ab_stage - 1)
        cute.arch.cp_async_wait_group(cfg.ab_stage - 2)
        cute.arch.sync_threads()
        tAsA_r = thr_a.partition_S(sA)[None, None, None, stage]
        tBsB_r = thr_b.partition_S(sB)[None, None, None, stage]
        tGsG_r = None
        if cutlass.const_expr(cfg.gated):
            tGsG_r = thr_a.partition_S(sG)[None, None, None, stage]
        tAsSFA = thr_sfa.partition_S(sSFA)[None, None, None, stage]
        tBsSFB = thr_sfb.partition_S(sSFB)[None, None, None, stage]
        tGsSFG = None
        if cutlass.const_expr(cfg.gated):
            tGsSFG = thr_sfa.partition_S(sSFG)[None, None, None, stage]
        for k in cutlass.range_constexpr(cute.size(tCrA_v, mode=[2])):
            cute.copy(s2r_a, tAsA_r[None, None, k], tCrA_v[None, None, k])
            if cutlass.const_expr(cfg.gated):
                cute.copy(s2r_a, tGsG_r[None, None, k], tCrG_v[None, None, k])
            cute.copy(s2r_b, tBsB_r[None, None, k], tCrB_v[None, None, k])
        for k in cutlass.range_constexpr(cute.size(tCrSFA_v, mode=[2])):
            cute.copy(s2r_sfa, tAsSFA[None, None, k], tCrSFA_v[None, None, k])
            if cutlass.const_expr(cfg.gated):
                cute.copy(s2r_sfa, tGsSFG[None, None, k], tCrSFG_v[None, None, k])
            cute.copy(s2r_sfb, tBsSFB[None, None, k], tCrSFB_v[None, None, k])
        future = kt + cfg.ab_stage - 1
        if future < k_tile_count:
            copy_stage(
                gW_u8,
                gA_u8,
                gSFA,
                gSFB,
                gSFB_i32,
                token_map,
                tAsA,
                tGsG,
                tBsB,
                sSFA,
                sSFG,
                sSFB,
                atom,
                sf_atom,
                tile,
                future,
                future & (cfg.ab_stage - 1),
                tidx,
                cfg,
                gather,
            )
        cute.arch.cp_async_commit_group()
        cute.gemm(tiledmma, acc, [tCrA, tCrSFA_frg], [tCrB, tCrSFB_frg], acc)
        if cutlass.const_expr(cfg.gated):
            cute.gemm(tiledmma, acc_up, [tCrG, tCrSFG_frg], [tCrB, tCrSFB_frg], acc_up)
    cute.arch.cp_async_wait_group(0)
    cute.arch.sync_threads()
    acc.store(acc.load() * scale)
    if cutlass.const_expr(cfg.gated):
        acc_up.store(acc_up.load() * scale)
        activation(acc_up, acc)
    else:
        activation(acc)
    if cutlass.const_expr(cfg.gated):
        return acc_up
    else:
        return acc


def _gmem_ptr(tensor, dtype, byte_offset, alignment=16):
    return cute.make_ptr(
        dtype,
        tensor.iterator.toint() + byte_offset,
        cute.AddressSpace.gmem,
        assumed_align=alignment,
    )


def _smem_ptr(pointer, word_offset, alignment=16):
    return cute.make_ptr(
        cutlass.Uint32,
        pointer.toint() + word_offset * 4,
        cute.AddressSpace.smem,
        assumed_align=alignment,
    )


@cute.jit
def copy_stage_gated(
    gSFA,
    sToken,
    tAgA,
    tAsA,
    tBsB,
    sSFA_words,
    atom_a,
    atom_b,
    cp_sf,
    source_layout_b,
    tid,
    rows,
    n0,
    sf_expert,
    token_base,
    token_row,
    kt,
    stage,
    output_n,
    cfg,
    gAS_u8,
    gTokenSF,
    sSFB_words,
    sf_atom_b,
    begin,
    mtile,
    gather: cutlass.Constexpr,
):
    bn, bm, bk, threads, K = cfg.TILE[1], cfg.TILE[0], cfg.TILE[2], cfg.threads, cfg.k
    i32, i64, u8, u32 = cutlass.Int32, cutlass.Int64, cutlass.Uint8, cutlass.Uint32
    fp4 = cfg.load_ab.a_dtype
    nk, NKT = cfg.load_sf.sf_in_tile_k(bk), cfg.load_sf.sf_in_tile_k(K)
    tpr = max(4, bk // 32)
    rows_pass = threads // tpr
    rows_pass_x = min(bn, rows_pass)
    tpr_x = min(threads // rows_pass_x, bk // 8)
    cp_per_x = bk // tpr_x
    npass_x = ceil_div(bn, rows_pass_x)
    align_x = min(16, max(4, cp_per_x // 2))
    nfuse = (bm // 2) * nk
    weight_sf_copies = max(1, ceil_div(nfuse, threads))
    shift_half, shift_sf = (32 * nk).bit_length() - 1, nk.bit_length() - 1
    for ip in cutlass.range_constexpr(npass_x):
        row = token_row + i32(ip * rows_pass_x)
        if row < rows:
            source = cute.make_tensor(
                cute.recast_ptr(
                    cute.make_ptr(
                        u8,
                        token_base + i64(sToken[row]) * (K // 2) + i64(kt) * (bk // 2),
                        cute.AddressSpace.gmem,
                        assumed_align=align_x,
                    ),
                    dtype=fp4,
                ),
                source_layout_b,
            )
            cute.copy(atom_b, source, tBsB[None, ip, 0, stage])
    cute.copy(atom_a, tAgA[None, None, None, kt], tAsA[None, None, None, stage])
    for it in cutlass.range_constexpr(weight_sf_copies):
        index = tid + it * threads
        if index < i32(nfuse):
            half = index >> shift_half
            group, column = (index >> 5) & (nk - 1), index & 31
            row = i32(n0) + half * output_n + column
            source_word = cute.crd2idx((row, kt * nk + group, sf_expert), gSFA.layout)
            target_word = group * 128 + column * 4 + half * 2 + stage * (nk * 128)
            cute.copy(
                cp_sf,
                cute.make_tensor(
                    _gmem_ptr(gSFA, u32, source_word * 4, 8), cute.make_layout(2)
                ),
                cute.make_tensor(
                    _smem_ptr(sSFA_words.iterator, target_word, 8), cute.make_layout(2)
                ),
            )
    if cutlass.const_expr(cfg.sf_staged):
        for it in cutlass.range_constexpr(ceil_div(bn * nk, threads)):
            index = tid + it * threads
            row, word = index >> shift_sf, index & (nk - 1)
            if i32(row) < rows:
                token = i64(sToken[row & (bn - 1)])
                sf_offset = token * NKT + i64(kt * nk + word)
                if cutlass.const_expr(not gather):
                    sf_row = (
                        compute_padded_offset(begin, sf_expert, i32(128))
                        + mtile * bn
                        + row
                    )
                    sf_offset = cute.crd2idx(
                        (sf_row, kt * nk + word, 0), gTokenSF.layout
                    )
                target_word = row + word * bn + stage * bn * nk
                cute.copy(
                    sf_atom_b,
                    cute.make_tensor(
                        _gmem_ptr(gAS_u8, u32, sf_offset * 4, 4), cute.make_layout(1)
                    ),
                    cute.make_tensor(
                        _smem_ptr(sSFB_words.iterator, target_word, 4),
                        cute.make_layout(1),
                    ),
                )


class CuteDslSm120MoeNvfp4Fc1ActQ1Decode:
    def __init__(self, cfg, grid_x, gather=False):
        self.cfg, self.grid_x, self.gather = cfg, grid_x, gather

    @cute.jit
    def __call__(
        self,
        gA_u8: cute.Tensor,
        gW_u8: cute.Tensor,
        gAS_u8: cute.Tensor,
        gWS_u8: cute.Tensor,
        gQ_u8: cute.Tensor,
        gSF_u8: cute.Tensor,
        offsets: cute.Tensor,
        weight_global_scale: cute.Tensor,
        input_global_scale: cute.Tensor,
        global_scale: cute.Tensor,
        stream,
        token_map=None,
        worklist=None,
        workcount=None,
    ):
        if cutlass.const_expr(self.cfg.fused_gated):
            assert self.gather == (token_map is not None)
            self.single_expert = False
            if cutlass.const_expr(cute.is_static(cute.size(gW_u8, mode=[0]))):
                self.single_expert = cute.size(gW_u8, mode=[0]) == 1 and not self.gather
            self.kernel_gated(
                gA_u8,
                gW_u8,
                gAS_u8,
                gWS_u8,
                gQ_u8,
                gSF_u8,
                offsets,
                weight_global_scale,
                input_global_scale,
                global_scale,
                token_map,
                worklist,
                workcount,
            ).launch(
                grid=[self.grid_x, 1, 1],
                block=[self.cfg.threads, 1, 1],
                stream=stream,
                min_blocks_per_mp=self.cfg.min_blocks_per_sm,
                use_pdl=self.cfg.enable_pdl,
            )
        else:
            self._call_default(
                gA_u8,
                gW_u8,
                gAS_u8,
                gWS_u8,
                gQ_u8,
                gSF_u8,
                offsets,
                weight_global_scale,
                input_global_scale,
                global_scale,
                stream,
                token_map,
                worklist,
                workcount,
            )

    @cute.jit
    def _call_default(
        self,
        gA_u8: cute.Tensor,
        gW_u8: cute.Tensor,
        gAS_u8: cute.Tensor,
        gWS_u8: cute.Tensor,
        gQ_u8: cute.Tensor,
        gSF_u8: cute.Tensor,
        offsets: cute.Tensor,
        weight_global_scale: cute.Tensor,
        input_global_scale: cute.Tensor,
        global_scale: cute.Tensor,
        stream,
        token_map=None,
        worklist=None,
        workcount=None,
    ):
        cfg = self.cfg
        assert self.gather == (token_map is not None)
        tiledmma = cfg.mma.make_tiled_mma(cfg.TILE)
        tiled_r2s, tiled_s2r = (
            cfg.epi.make_tiled_r2s(tiledmma),
            cfg.epi.make_tiled_s2r(cfg.TILE),
        )
        a_smem, b_smem = (
            cfg.load_ab.make_smem_layout_a(),
            cfg.load_ab.make_smem_layout_b(),
        )
        sfa_smem = cfg.load_sf.make_smem_layout_sfa(tiledmma, cfg.TILE, cfg.ab_stage)
        sfb_smem = cfg.load_sf.make_smem_layout_sfb(tiledmma, cfg.TILE, cfg.ab_stage)
        epi_smem = cfg.epi.make_smem_layout(cfg.TILE)
        N2, K, E = (
            cute.size(gW_u8, mode=[1]),
            cute.size(gA_u8, mode=[1]) << 1,
            cute.size(gW_u8, mode=[0]),
        )
        self.single_expert = False
        if cutlass.const_expr(cute.is_static(E)):
            self.single_expert = E == 1 and not self.gather
        gSFA = cute.make_tensor(
            cute.recast_ptr(gWS_u8.iterator, dtype=cfg.load_sf.sf_dtype),
            cfg.load_sf.deduce_sfa_layout(N2, K, E),
        )
        gSFB = None
        if cutlass.const_expr(not self.gather):
            gSFB = cute.make_tensor(
                cute.recast_ptr(gAS_u8.iterator, dtype=cfg.load_sf.sf_dtype),
                cfg.load_sf.deduce_sfb_layout(cute.size(gAS_u8, mode=[0]), K, 1),
            )
        gSFB_i32 = cute.recast_ptr(gAS_u8.iterator, dtype=cfg.load_sf.sf_dtype)
        gQ = cute.recast_tensor(gQ_u8, cutlass.Float4E2M1FN)
        gSF = cute.make_tensor(gSF_u8.iterator, cute.make_layout(cute.size(gSF_u8)))
        values = cfg.G2S_BITS // cfg.load_ab.a_dtype.width
        lanes_k = ceil_div(cfg.TILE[2], values)
        lanes_m = cfg.WARP_SIZE >> (lanes_k.bit_length() - 1)
        atom = cute.make_copy_atom(
            cpasync.CopyG2SOp(), cfg.load_ab.a_dtype, num_bits_per_copy=cfg.G2S_BITS
        )
        tiled = cute.make_tiled_copy_tv(
            atom,
            cute.make_ordered_layout((lanes_m, lanes_k), order=(1, 0)),
            cute.make_layout((1, values)),
        )
        sf_atom = cute.make_copy_atom(
            cpasync.CopyG2SOp(),
            cfg.load_sf.sf_dtype,
            num_bits_per_copy=cfg.load_sf.sf_dtype.width,
        )
        gate_elems = cute.cosize(a_smem) if cfg.gated else 0
        gate_sf_elems = cute.cosize(sfa_smem) if cfg.gated else 0

        @cute.struct
        class SharedStorage:
            sA: cute.struct.Align[
                cute.struct.MemRange[cfg.load_ab.a_dtype, cute.cosize(a_smem)],
                cfg.SMEM_ALIGN_BYTES,
            ]
            sG: cute.struct.Align[
                cute.struct.MemRange[cfg.load_ab.a_dtype, gate_elems],
                cfg.SMEM_ALIGN_BYTES,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[cfg.load_ab.b_dtype, cute.cosize(b_smem)],
                cfg.SMEM_ALIGN_BYTES,
            ]
            sSFA: cute.struct.Align[
                cute.struct.MemRange[cfg.load_sf.sf_dtype, cute.cosize(sfa_smem)],
                cfg.SMEM_ALIGN_BYTES,
            ]
            sSFG: cute.struct.Align[
                cute.struct.MemRange[cfg.load_sf.sf_dtype, gate_sf_elems],
                cfg.SMEM_ALIGN_BYTES,
            ]
            sSFB: cute.struct.Align[
                cute.struct.MemRange[cfg.load_sf.sf_dtype, cute.cosize(sfb_smem)],
                cfg.SMEM_ALIGN_BYTES,
            ]
            sC: cute.struct.Align[
                cute.struct.MemRange[cfg.epi.out_dtype, cute.cosize(epi_smem)],
                cfg.SMEM_ALIGN_BYTES,
            ]

        self.storage = SharedStorage
        assert SharedStorage.size_in_bytes() == cfg.smem_bytes
        self.kernel(
            gA_u8,
            gW_u8,
            gSFA,
            gSFB,
            gSFB_i32,
            gQ,
            gSF,
            offsets,
            weight_global_scale,
            input_global_scale,
            global_scale,
            cute.size(gSF_u8, mode=[1]),
            token_map,
            worklist,
            workcount,
            tiledmma,
            tiled_r2s,
            tiled_s2r,
            a_smem,
            b_smem,
            sfa_smem,
            sfb_smem,
            epi_smem,
            atom,
            tiled,
            sf_atom,
        ).launch(
            grid=[self.grid_x, 1, 1],
            block=[cfg.threads, 1, 1],
            stream=stream,
            min_blocks_per_mp=cfg.min_blocks_per_sm,
            use_pdl=cfg.enable_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        gA_u8: cute.Tensor,
        gW_u8: cute.Tensor,
        gSFA: cute.Tensor,
        gSFB,
        gSFB_i32,
        gQ: cute.Tensor,
        gSF: cute.Tensor,
        offsets: cute.Tensor,
        weight_global_scale: cute.Tensor,
        input_global_scale: cute.Tensor,
        global_scale: cute.Tensor,
        sf_cols,
        token_map,
        worklist,
        workcount,
        tiledmma,
        tiled_r2s,
        tiled_s2r,
        a_smem,
        b_smem,
        sfa_smem,
        sfb_smem,
        epi_smem,
        atom,
        tiled,
        sf_atom,
    ):
        cfg = self.cfg
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        bm, bn, bk = cfg.TILE
        M, N = cute.size(gQ, mode=[0]), cute.size(gQ, mode=[1])
        sfb_subtiles = cfg.load_sf.sfb_tiles_per_block(bn)
        if cutlass.const_expr(cfg.enable_pdl):
            cute.arch.griddepcontrol_wait()
        i32 = cutlass.Int32
        n_blocks = i32(ceil_div(N, bm))
        n_tile, first_work = bidx % n_blocks, bidx // n_blocks
        work_stride = i32(self.grid_x) // n_blocks
        shift_m = bn.bit_length() - 1
        total_work = i32(0)
        if cutlass.const_expr(self.single_expert):
            total_work = i32(ceil_div(M, bn))
        else:
            total_work = workcount[0]
        storage = cutlass.utils.SmemAllocator().allocate(self.storage)
        for widx in cutlass.range(first_work, total_work, work_stride, unroll=1):
            if widx != first_work:
                cute.arch.sync_threads()
            group, begin, r0, end, m_tile = i32(0), i32(0), i32(0), i32(0), i32(0)
            if cutlass.const_expr(self.single_expert):
                r0 = widx * bn
                end = min(r0 + i32(bn), i32(M))
                m_tile = widx
            else:
                group, r0, rows = (
                    worklist[0, widx],
                    worklist[1, widx],
                    worklist[2, widx],
                )
                begin = offsets[group]
                end = r0 + rows
                m_tile = (r0 - begin) >> shift_m
            tile = moe_scheduler.MoeWorkTile(
                n_tile, m_tile, group, begin, end, widx < total_work
            )
            sA = storage.sA.get_tensor(a_smem.outer, swizzle=a_smem.inner)
            sB = storage.sB.get_tensor(b_smem.outer, swizzle=b_smem.inner)
            sSFA, sSFB = (
                storage.sSFA.get_tensor(sfa_smem),
                storage.sSFB.get_tensor(sfb_smem),
            )
            sG, sSFG, tGsG = None, None, None
            if cutlass.const_expr(cfg.gated):
                sG = storage.sG.get_tensor(a_smem.outer, swizzle=a_smem.inner)
                sSFG = storage.sSFG.get_tensor(sfa_smem)
                tGsG = tiled.get_slice(tidx & (cfg.WARP_SIZE - 1)).partition_D(sG)
            subtile = cutlass.Int32(0)
            if cutlass.const_expr(not self.gather):
                subtile = tile.n_block & (sfb_subtiles - 1)
            sSFB_tile = cute.local_tile(
                sSFB, (bn, cute.size(sSFB, mode=[1])), (subtile, 0, None)
            )
            tAsA = tiled.get_slice(tidx & (cfg.WARP_SIZE - 1)).partition_D(sA)
            tBsB = tiled.get_slice(tidx & (cfg.WARP_SIZE - 1)).partition_D(sB)
            acc = mma_multistage(
                tiledmma,
                cfg,
                cfg.activation,
                sA,
                sG,
                sB,
                sSFA,
                sSFG,
                sSFB_tile,
                gW_u8,
                gA_u8,
                gSFA,
                gSFB,
                gSFB_i32,
                token_map,
                tAsA,
                tGsG,
                tBsB,
                atom,
                sf_atom,
                tile,
                tidx,
                ceil_div(cute.size(gA_u8, mode=[1]), bk >> 1),
                (weight_global_scale[group] / input_global_scale[0]).to(
                    cutlass.Float32
                ),
                self.gather,
            )
            sC = storage.sC.get_tensor(epi_smem.outer, swizzle=epi_smem.inner)[
                None, None, 0
            ]
            gQ_t = cute.make_tensor(
                gQ.iterator, cute.make_layout((N, M), stride=(1, N))
            )
            moe_epilogue.store_wg_nvfp4_q1_swap(
                acc,
                tiled_r2s.get_slice(tidx),
                tiled_s2r.get_slice(tidx),
                sC,
                gQ_t,
                gSF,
                global_scale,
                tile,
                (bm, bn),
                sf_cols,
                cfg.epi_bar_id,
                cfg.mma_threads,
                tidx,
                cfg.epi.num_store_threads,
            )
        if cutlass.const_expr(cfg.enable_pdl):
            cute.arch.griddepcontrol_launch_dependents()

    @cute.kernel
    def kernel_gated(
        self,
        gA_u8: cute.Tensor,
        gW_u8: cute.Tensor,
        gAS_u8: cute.Tensor,
        gWS_u8: cute.Tensor,
        gQ_u8: cute.Tensor,
        gSF_u8: cute.Tensor,
        offsets: cute.Tensor,
        weight_global_scale: cute.Tensor,
        input_global_scale: cute.Tensor,
        global_scale: cute.Tensor,
        token_map,
        worklist,
        workcount,
    ):
        cfg = self.cfg
        bn, bm, bk, ab_stage, threads, K = (
            cfg.TILE[1],
            cfg.TILE[0],
            cfg.TILE[2],
            cfg.ab_stage,
            cfg.threads,
            cfg.k,
        )
        fp4, f32, bf16 = cutlass.Float4E2M1FN, cutlass.Float32, cutlass.BFloat16
        i32, i64, u8, u32 = cutlass.Int32, cutlass.Int64, cutlass.Uint8, cutlass.Uint32
        sf_cfg, mma_cfg, load_ab, epi_cfg = cfg.load_sf, cfg.mma, cfg.load_ab, cfg.epi
        epi_tile = (bm // 2, bn, bk)
        nk, NKT, k_tiles = (
            sf_cfg.sf_in_tile_k(bk),
            sf_cfg.sf_in_tile_k(K),
            ceil_div(K, bk),
        )
        tpr = max(4, bk // 32)
        rows_pass, cp_per = threads // tpr, bk // tpr
        rows_pass_x = min(bn, rows_pass)
        tpr_x = min(threads // rows_pass_x, bk // 8)
        cp_per_x = bk // tpr_x
        copy_threads_x = rows_pass_x * tpr_x
        sf_token_words, sf_weight_words = bn * nk * cfg.sf_stages, bm * nk * ab_stage
        sf_copies = ceil_div(sf_token_words, threads)
        epi_padding = max(0, bn * bm * 4 - (sf_token_words + sf_weight_words) * 4)
        tid, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        if cutlass.const_expr(cfg.enable_pdl):
            cute.arch.griddepcontrol_wait()
        gQ = cute.recast_tensor(gQ_u8, fp4)
        M, N = cute.size(gQ, mode=[0]), cute.size(gQ, mode=[1])
        E = cute.size(gW_u8, mode=[0])
        n_blocks = ceil_div(N, bm >> 1)
        ntile, first_work = bidx % n_blocks, bidx // n_blocks
        work_stride = self.grid_x // n_blocks
        tiled_mma = mma_cfg.make_tiled_mma((bm, bn, bk))
        weight_layout, token_layout = (
            load_ab.make_smem_layout_a(),
            load_ab.make_smem_layout_b(),
        )
        gSFA = cute.make_tensor(
            cute.recast_ptr(gWS_u8.iterator, dtype=sf_cfg.sf_dtype),
            sf_cfg.deduce_sfa_layout(cute.size(gW_u8, mode=[1]), K, E),
        )
        gTokenSF = None
        if cutlass.const_expr(not self.gather):
            gTokenSF = cute.make_tensor(
                cute.recast_ptr(gAS_u8.iterator, dtype=sf_cfg.sf_dtype),
                sf_cfg.deduce_sfb_layout(cute.size(gAS_u8, mode=[0]), K, 1),
            )
        q_stride = cute.assume(N, 128)
        gQ_t = cute.make_tensor(
            gQ.iterator, cute.make_layout((N, M), stride=(1, q_stride))
        )
        gSF = cute.make_tensor(gSF_u8.iterator, cute.make_layout(cute.size(gSF_u8)))
        epi_layout = epi_cfg.make_smem_layout(epi_tile)
        tiled_s2r = epi_cfg.make_tiled_s2r(epi_tile)
        weight_op = (
            cpasync.CopyG2SOp(cache_mode=cute.nvgpu.LoadCacheMode.GLOBAL)
            if cutlass.const_expr(cp_per * 4 >= 128)
            else cpasync.CopyG2SOp()
        )
        atom_a = cute.make_copy_atom(weight_op, fp4, num_bits_per_copy=cp_per * 4)
        atom_b = cute.make_copy_atom(
            cpasync.CopyG2SOp(), fp4, num_bits_per_copy=cp_per_x * 4
        )
        sf_atom_b = cute.make_copy_atom(cpasync.CopyG2SOp(), u32, num_bits_per_copy=32)
        sf_atom_a = cute.make_copy_atom(cpasync.CopyG2SOp(), u32, num_bits_per_copy=64)
        tiled_copy_a = cute.make_tiled_copy_tv(
            atom_a,
            cute.make_ordered_layout((rows_pass, tpr), order=(1, 0)),
            cute.make_layout((1, cp_per)),
        )
        tiled_copy_b = cute.make_tiled_copy_tv(
            atom_b,
            cute.make_ordered_layout((rows_pass_x, tpr_x), order=(1, 0)),
            cute.make_layout((1, cp_per_x)),
        )
        sm_cp_a = mma_cfg.make_s2r_a(tiled_mma, fp4, False)
        sm_cp_b = mma_cfg.make_s2r_b(tiled_mma, fp4, False)
        sm_cp_sfa = sf_cfg.make_s2r_sf(
            sf_cfg.get_layoutSFA_TV(tiled_mma),
            (cute.size(tiled_mma.permutation_mnk[0]), sf_cfg.KTILE_SF),
        )
        sm_cp_sfb = sf_cfg.make_s2r_sf(
            sf_cfg.get_layoutSFB_TV(tiled_mma),
            (cute.size(tiled_mma.permutation_mnk[1]), sf_cfg.KTILE_SF),
        )
        smem = utils.SmemAllocator()
        sB = smem.allocate_tensor(
            fp4, token_layout.outer, byte_alignment=1024, swizzle=token_layout.inner
        )
        sA = smem.allocate_tensor(
            fp4, weight_layout.outer, byte_alignment=1024, swizzle=weight_layout.inner
        )
        sToken = smem.allocate_tensor(i32, cute.make_layout(bn), byte_alignment=16)
        sSFB_words = smem.allocate_tensor(
            i32, cute.make_layout(sf_token_words), byte_alignment=1024
        )
        sSFA_words = smem.allocate_tensor(
            i32, cute.make_layout(sf_weight_words), byte_alignment=16
        )
        if cutlass.const_expr(epi_padding > 0):
            smem.allocate_tensor(
                u32, cute.make_layout(ceil_div(epi_padding, 4)), byte_alignment=16
            )
        sSFA = cute.make_tensor(
            sSFA_words.iterator,
            sf_cfg.make_smem_layout_sfa(tiled_mma, (bm, bn, bk), ab_stage),
        )
        sSFB = cute.make_tensor(
            sSFB_words.iterator,
            cute.make_layout((bn, nk, cfg.sf_stages), stride=(1, bn, bn * nk)),
        )
        sC = cute.make_tensor(
            cute.recast_ptr(sSFB_words.iterator, dtype=f32),
            cute.make_layout((bn, bm), stride=(bm, 1)),
        )
        sEp = cute.make_tensor(
            cute.recast_ptr(sSFB_words.iterator, dtype=bf16), epi_layout
        )[None, None, 0]
        thr_mma = tiled_mma.get_slice(tid)
        tCrA = tiled_mma.make_fragment_A(thr_mma.partition_A(sA)[None, None, None, 0])
        tCrB = tiled_mma.make_fragment_B(thr_mma.partition_B(sB)[None, None, None, 0])
        tCrSFA = sf_cfg.partition_fragment_SFA(sSFA[None, None, 0], thr_mma, tid)
        tCrSFB = sf_cfg.partition_fragment_SFB(sSFB[None, None, 0], thr_mma, tid)
        tCrSFA_mma = sf_cfg.make_sfa_e4m3_view(tCrSFA)
        tCrSFB_mma = sf_cfg.make_sfb_e4m3_view(tCrSFB)
        ta, tb = sm_cp_a.get_slice(tid), sm_cp_b.get_slice(tid)
        tsa, tsb = sm_cp_sfa.get_slice(tid), sm_cp_sfb.get_slice(tid)
        tAsA_v, tCrA_v = ta.partition_S(sA), ta.retile(tCrA)
        tBsB_v, tCrB_v = tb.partition_S(sB), tb.retile(tCrB)
        tAsSFA, tCrSFA_v = tsa.partition_S(sSFA), tsa.retile(tCrSFA)
        tBsSFB, tCrSFB_v = tsb.partition_S(sSFB), tsb.retile(tCrSFB)
        thr_copy_a = tiled_copy_a.get_slice(tid)
        thr_copy_b = tiled_copy_b.get_slice(tid & (copy_threads_x - 1))
        tAsA = thr_copy_a.partition_D(sA)
        tBsB = thr_copy_b.partition_D(sB)
        source_layout_b = cute.make_layout(((cp_per_x, 1),), stride=((1, 0),))
        token_base = gA_u8.iterator.toint() + i64((tid & (tpr_x - 1)) * (cp_per_x // 2))
        token_row = tid >> (tpr_x.bit_length() - 1)
        shift_m, shift_sf = bn.bit_length() - 1, NKT.bit_length() - 1
        idt = thr_mma.partition_C(cute.make_identity_tensor((bm, bn)))

        total_work = i32(0)
        if cutlass.const_expr(self.single_expert):
            total_work = ceil_div(M, bn)
        else:
            total_work = workcount[0]
        for widx in cutlass.range(first_work, total_work, work_stride, unroll=1):
            if widx != first_work:
                cute.arch.sync_threads()
            expert, r0, rows, begin, mtile, end = (
                i32(0),
                i32(0),
                i32(0),
                i32(0),
                i32(0),
                i32(0),
            )
            if cutlass.const_expr(self.single_expert):
                r0 = widx * bn
                rows = min(M - r0, i32(bn))
                mtile, end = widx, r0 + rows
            else:
                expert, r0, rows = (
                    worklist[0, widx],
                    worklist[1, widx],
                    worklist[2, widx],
                )
                begin = offsets[expert]
                mtile = (r0 - begin) >> shift_m
                end = r0 + rows
            n0 = i32(ntile) * (bm // 2)
            if tid < i32(bn):
                token = i32(0)
                if tid < rows:
                    token = r0 + tid
                    if cutlass.const_expr(self.gather):
                        token = token_map[token]
                sToken[tid] = token
            cute.arch.sync_threads()
            gW_tile_u8 = cute.make_tensor(
                _gmem_ptr(gW_u8, u8, (i64(expert) * (2 * N) + i64(n0)) * (K // 2)),
                cute.make_layout(
                    ((bm >> 1, 2), K // 2), stride=((K // 2, N * (K // 2)), 1)
                ),
            )
            gW = cute.local_tile(
                cute.recast_tensor(gW_tile_u8, fp4), (bm, bk), (0, None)
            )
            tAgA = thr_copy_a.partition_S(gW)
            sf_expert = expert
            if cutlass.const_expr(not cfg.sf_staged):
                for it in cutlass.range_constexpr(sf_copies):
                    index = tid + it * threads
                    if cutlass.const_expr(NKT & (NKT - 1) == 0):
                        row, group = index >> shift_sf, index & (NKT - 1)
                    else:
                        row, group = index // NKT, index % NKT
                    if i32(row) < rows:
                        token = i64(sToken[row & (bn - 1)])
                        sf_offset = token * NKT + i64(group)
                        if cutlass.const_expr(not self.gather):
                            sf_row = (
                                compute_padded_offset(begin, expert, i32(128))
                                + mtile * bn
                                + row
                            )
                            sf_offset = cute.crd2idx(
                                (sf_row, group, 0), gTokenSF.layout
                            )
                        cute.copy(
                            sf_atom_b,
                            cute.make_tensor(
                                _gmem_ptr(gAS_u8, u32, sf_offset * 4, 4),
                                cute.make_layout(1),
                            ),
                            cute.make_tensor(
                                _smem_ptr(sSFB_words.iterator, row + group * bn, 4),
                                cute.make_layout(1),
                            ),
                        )
                cute.arch.cp_async_commit_group()
            for initial_stage in cutlass.range_constexpr(ab_stage - 1):
                if cutlass.const_expr(initial_stage < k_tiles):
                    copy_stage_gated(
                        gSFA,
                        sToken,
                        tAgA,
                        tAsA,
                        tBsB,
                        sSFA_words,
                        atom_a,
                        atom_b,
                        sf_atom_a,
                        source_layout_b,
                        tid,
                        rows,
                        n0,
                        sf_expert,
                        token_base,
                        token_row,
                        initial_stage,
                        initial_stage,
                        N,
                        self.cfg,
                        gAS_u8,
                        gTokenSF,
                        sSFB_words,
                        sf_atom_b,
                        begin,
                        mtile,
                        self.gather,
                    )
                cute.arch.cp_async_commit_group()
            acc = cute.make_rmem_tensor(idt.shape, f32)
            acc.fill(0.0)
            if cutlass.const_expr(k_tiles <= ab_stage - 1):
                cute.arch.cp_async_wait_group(0)
                cute.arch.sync_threads()
            for kt in cutlass.range(0, k_tiles, 1, unroll=2):
                if cutlass.const_expr(k_tiles > ab_stage - 1):
                    cute.arch.cp_async_wait_group(ab_stage - 2)
                    cute.arch.sync_threads()
                future = kt + ab_stage - 1
                if future < i32(k_tiles):
                    if cutlass.const_expr(ab_stage & (ab_stage - 1) == 0):
                        next_stage = future & (ab_stage - 1)
                    else:
                        next_stage = future % ab_stage
                    copy_stage_gated(
                        gSFA,
                        sToken,
                        tAgA,
                        tAsA,
                        tBsB,
                        sSFA_words,
                        atom_a,
                        atom_b,
                        sf_atom_a,
                        source_layout_b,
                        tid,
                        rows,
                        n0,
                        sf_expert,
                        token_base,
                        token_row,
                        future,
                        next_stage,
                        N,
                        self.cfg,
                        gAS_u8,
                        gTokenSF,
                        sSFB_words,
                        sf_atom_b,
                        begin,
                        mtile,
                        self.gather,
                    )
                cute.arch.cp_async_commit_group()
                if cutlass.const_expr(ab_stage & (ab_stage - 1) == 0):
                    stage = kt & (ab_stage - 1)
                else:
                    stage = kt % ab_stage
                for kb in cutlass.range_constexpr(nk):
                    cute.copy(
                        sm_cp_a, tAsA_v[None, None, kb, stage], tCrA_v[None, None, kb]
                    )
                    cute.copy(
                        sm_cp_b, tBsB_v[None, None, kb, stage], tCrB_v[None, None, kb]
                    )
                    cute.copy(
                        sm_cp_sfa,
                        tAsSFA[None, None, kb, stage],
                        tCrSFA_v[None, None, kb],
                    )
                    sf_stage = stage if cutlass.const_expr(cfg.sf_staged) else kt
                    cute.copy(
                        sm_cp_sfb,
                        tBsSFB[None, None, kb, sf_stage],
                        tCrSFB_v[None, None, kb],
                    )
                    cute.gemm(
                        tiled_mma,
                        acc,
                        [tCrA[None, None, kb], tCrSFA_mma[None, None, kb]],
                        [tCrB[None, None, kb], tCrSFB_mma[None, None, kb]],
                        acc,
                    )
            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()
            for i in cutlass.range_constexpr(cute.size(acc)):
                if idt[i][1] < rows:
                    sC[idt[i][1], idt[i][0]] = acc[i]
            cute.arch.sync_threads()
            up = acc
            gate = cute.make_rmem_tensor(acc.layout, f32)
            for i in cutlass.range_constexpr(cute.size(acc)):
                if (idt[i][0] < i32(bm // 2)) & (idt[i][1] < rows):
                    up[i] = sC[idt[i][1], idt[i][0] + bm // 2] * (
                        weight_global_scale[expert] / input_global_scale[0]
                    )
                    gate[i] = sC[idt[i][1], idt[i][0]] * (
                        weight_global_scale[expert] / input_global_scale[0]
                    )
            cute.arch.sync_threads()
            act_up = cute.make_rmem_tensor(cute.make_layout(1), f32)
            act_gate = cute.make_rmem_tensor(cute.make_layout(1), f32)
            for i in cutlass.range_constexpr(cute.size(acc)):
                if (idt[i][0] < i32(bm // 2)) & (idt[i][1] < rows):
                    act_up[0], act_gate[0] = up[i], gate[i]
                    self.cfg.activation(act_up, act_gate)
                    sEp[idt[i][0], idt[i][1]] = act_up[0].to(bf16)
            cute.arch.sync_threads()
            if tid < self.cfg.epi.num_store_threads:
                gQ_tile = cute.local_tile(
                    cute.domain_offset((0, begin), gQ_t), epi_tile[:2], (ntile, mtile)
                )
                sf_row_base = (
                    compute_padded_offset(begin, expert, i32(128)) + mtile * bn
                )
                moe_epilogue.quantize_nvfp4_store(
                    tiled_s2r.get_slice(tid),
                    sEp,
                    gQ_tile,
                    gSF,
                    global_scale,
                    sf_row_base,
                    ntile * (bm // 2 // self.cfg.load_sf.SF_VEC),
                    r0,
                    end,
                    cute.size(gSF_u8, mode=[1]),
                    epi_tile[:2],
                    True,
                )
        if cutlass.const_expr(cfg.enable_pdl):
            cute.arch.griddepcontrol_launch_dependents()


def _stream(device):
    return cuda.CUstream(torch.cuda.current_stream(device).cuda_stream)


def make_args(
    a_q,
    a_scale,
    b_q,
    b_scale,
    out_q,
    out_sf,
    m_indptr,
    weight_global_scale,
    input_global_scale,
    global_scale,
    epi_cfg,
    tile,
    *,
    token_map=None,
    worklist=None,
    workcount=None,
):
    assert (worklist is None) == (workcount is None)
    b_scale = b_scale.view(torch.uint8).view(
        int(b_q.shape[0]) * align(int(b_q.shape[1]), 128),
        ceil_div(int(b_q.shape[2]) * 2, 16),
    )
    sf = lambda tensor: from_dlpack(
        tensor.contiguous(), assumed_align=16
    ).mark_layout_dynamic()
    if b_q.shape[0] == 1:
        b_arg = (
            from_dlpack(b_q)
            .mark_compact_shape_dynamic(mode=1, stride_order=(0, 1, 2))
            .mark_compact_shape_dynamic(mode=2, stride_order=(0, 1, 2))
        )
    else:
        b_arg = from_dlpack(b_q).mark_layout_dynamic()
    del epi_cfg, tile
    out_arg = from_dlpack(out_q, assumed_align=16).mark_layout_dynamic()
    args = (
        from_dlpack(a_q).mark_layout_dynamic(),
        b_arg,
        sf(a_scale),
        sf(b_scale),
        out_arg,
        sf(out_sf),
        from_dlpack(m_indptr).mark_layout_dynamic(),
        from_dlpack(weight_global_scale).mark_layout_dynamic(),
        from_dlpack(input_global_scale).mark_layout_dynamic(),
        from_dlpack(global_scale).mark_layout_dynamic(),
        _stream(a_q.device),
        (
            from_dlpack(token_map, assumed_align=4).mark_layout_dynamic()
            if token_map is not None
            else None
        ),
    )
    if worklist is not None:
        assert workcount is not None
        args += (
            from_dlpack(worklist, assumed_align=4).mark_layout_dynamic(),
            from_dlpack(workcount, assumed_align=4).mark_layout_dynamic(),
        )
    return args
