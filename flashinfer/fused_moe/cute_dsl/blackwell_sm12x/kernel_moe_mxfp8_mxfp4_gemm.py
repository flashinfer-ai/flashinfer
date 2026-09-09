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
"""Self-written CuteDSL act-MXFP8 x weight-MXFP4 token-packed grouped MoE GEMM for SM120a."""

import torch

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils
import cutlass.cute.nvgpu.warp.mma as warp_mma
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import from_dlpack

from ._moe_utils.moe_epilogue import EPI_CONFIGS, EpiMethod
from ._moe_utils.moe_kernel_builder import Sm12xGemmConfig, MmaConfig, LoadABConfig
from ._moe_utils.sm12x_blockscaled_layout import Sm120SfConfigMxfp8Mxfp4
from ._moe_utils.sm12x_blockscaled_layout import SF_M_ALIGN
from ....utils import ceil_div
from ._moe_utils import moe_scheduler, moe_epilogue


GRANK_A, GRANK_B = 128, 32
ATOM_MNK = (16, 8, 32)


def is_swapab(tile):
    return ATOM_MNK[0] > tile[0]


REG_PROD_BY_TACTIC = {
    (128, 128, EpiMethod.R2G_WG): 56,
    (128, 128, EpiMethod.STAGED_R2G): 104,
    (64, 128, EpiMethod.R2G_WG): 56,
    (64, 128, EpiMethod.STAGED_R2G): 88,
    (32, 128, EpiMethod.R2G_WG): 56,
    (8, 128, EpiMethod.DIRECT_STG): 88,
}


def make_cfg(tile, ab_stage, epi=EpiMethod.R2G_WG, enable_pdl=False):
    e4m3, fp4, f32, bf16, i8, ue8m0 = (
        cutlass.Float8E4M3FN,
        cutlass.Float4E2M1FN,
        cutlass.Float32,
        cutlass.BFloat16,
        cutlass.Int8,
        cutlass.Float8E8M0FNU,
    )
    num_math_warps = 8
    bm, bn, bk = tile
    swap = is_swapab(tile)
    grank_a = GRANK_A
    tactic = (bm, bn, EpiMethod.DIRECT_STG if swap else epi)
    if swap:
        bm, bn = bn, bm
        a_dtype, b_dtype = fp4, e4m3
        unpack = dict(a_smem_dtype=i8, a_tma_internal=i8, a_unpack_bits=4)
        grank_a, grank_b = GRANK_B, grank_a
        epi = EpiMethod.DIRECT_STG
    else:
        assert epi is not EpiMethod.DIRECT_STG, (
            f"{epi} needs the swapped geometry; this tile is not swap-AB"
        )
        a_dtype, b_dtype = e4m3, fp4
        unpack = dict(b_smem_dtype=i8, b_tma_internal=i8, b_unpack_bits=4)
        grank_b = GRANK_B
    tile = (bm, bn, bk)
    union = epi is EpiMethod.R2G_WG
    return Sm12xGemmConfig(
        MmaConfig(
            warp_mma.MmaMXF8F6F4Op(a_dtype, b_dtype, f32, ue8m0),
            tile[:2],
            num_math_warps,
            swap_ab=swap,
        ),
        LoadABConfig(tile, ab_stage, a_dtype, b_dtype, **unpack),
        Sm120SfConfigMxfp8Mxfp4(grank_a, grank_b),
        EPI_CONFIGS[epi](bf16, num_math_warps * 32),
        ab_stage,
        tile,
        epi_bar_id=3,
        union_smem=union,
        reg_prod=REG_PROD_BY_TACTIC[tactic],
        enable_pdl=enable_pdl,
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
    m_align,
    swap,
):
    i32 = cutlass.Int32
    cluster = cute.make_layout((1, 1, 1))
    multicast = cute.make_layout(cute.slice_(cluster, (0, None, 0)).shape)
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
        i32(0),
        multicast,
        cute.group_modes(sA, 0, 2),
        cute.group_modes(gA_mkl, 0, 2),
    )
    if cutlass.const_expr(swap):
        gSFA_ml = cute.local_tile(tma_tensor_sfa, (tile_mnk[0], 1), (None, None, None))
    else:
        sf_m_off = (tile.m_offset + tile.group * i32(m_align - 1)) & i32(-m_align)
        mSFA = cute.domain_offset((sf_m_off, 0, 0), tma_tensor_sfa)
        gSFA_ml = cute.local_tile(mSFA, (tile_mnk[0], 1), (None, None, None))
    tAsSFA, tAgSFA = cpasync.tma_partition(
        tma_atom_sfa,
        i32(0),
        multicast,
        cute.group_modes(sSFA, 0, 2),
        cute.group_modes(gSFA_ml, 0, 2),
    )
    return tAsA, tAgA, tAsSFA, tAgSFA


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
    m_align,
    swap,
):
    i32 = cutlass.Int32
    cluster = cute.make_layout((1, 1, 1))
    multicast = cute.make_layout(cute.slice_(cluster, (0, None, 0)).shape)
    if cutlass.const_expr(swap):
        mB = cute.domain_offset((tile.m_offset, 0), tma_tensor_b)
        gB_nkl = cute.local_tile(
            mB, cute.slice_(tile_mnk, (0, None, None)), (None, None)
        )
    else:
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
    if cutlass.const_expr(swap):
        sf_m_off = (tile.m_offset + tile.group * i32(m_align - 1)) & i32(-m_align)
        mSFB = cute.domain_offset((sf_m_off, 0, 0), tma_tensor_sfb)
        gSFB_nl = cute.local_tile(mSFB, (tile_mnk[1], 1), (None, None, None))
    else:
        gSFB_nl = cute.local_tile(tma_tensor_sfb, (tile_mnk[1], 1), (None, None, None))
    tBsSFB, tBgSFB = cpasync.tma_partition(
        tma_atom_sfb,
        i32(0),
        multicast,
        cute.group_modes(sSFB, 0, 2),
        cute.group_modes(gSFB_nl, 0, 2),
    )
    return tBsB, tBgB, tBsSFB, tBgSFB


@cute.jit
def copy_a_sfa(
    tma_atom_a,
    tma_atom_sfa,
    tAgA,
    tAsA,
    tAgSFA,
    tAsSFA,
    tile,
    a_full,
    a_empty,
    a_bytes,
    sfa_bytes,
    sf_cycle_idx,
    sf_stages,
    kt_per_pack,
    ab_stages,
    a_phase,
    swap,
):
    for sf_stage in cutlass.range_constexpr(sf_stages):
        global_sf_stage = sf_cycle_idx * sf_stages + sf_stage
        for k_in_sf in cutlass.range_constexpr(kt_per_pack):
            local_k_tile_idx = sf_stage * kt_per_pack + k_in_sf
            global_k_tile_idx = (
                sf_cycle_idx * sf_stages * kt_per_pack + local_k_tile_idx
            )
            stage_idx = local_k_tile_idx & (ab_stages - 1)
            cute.arch.mbarrier_wait(a_empty + stage_idx, a_phase)
            tx_bytes = a_bytes
            if cutlass.const_expr(swap or k_in_sf == 0):
                tx_bytes += sfa_bytes
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(a_full + stage_idx, tx_bytes)
            if cutlass.const_expr(swap):
                cute.copy(
                    tma_atom_sfa,
                    tAgSFA[(None, tile.m_block, global_k_tile_idx, tile.group)],
                    tAsSFA[(None, stage_idx)],
                    tma_bar_ptr=a_full + stage_idx,
                )
                cute.copy(
                    tma_atom_a,
                    tAgA[(None, tile.m_block, global_k_tile_idx, tile.group)],
                    tAsA[(None, stage_idx)],
                    tma_bar_ptr=a_full + stage_idx,
                )
            else:
                if cutlass.const_expr(k_in_sf == 0):
                    cute.copy(
                        tma_atom_sfa,
                        tAgSFA[(None, tile.m_block, global_sf_stage, 0)],
                        tAsSFA[(None, sf_stage)],
                        tma_bar_ptr=a_full + stage_idx,
                    )
                cute.copy(
                    tma_atom_a,
                    tAgA[(None, tile.m_block, global_k_tile_idx)],
                    tAsA[(None, stage_idx)],
                    tma_bar_ptr=a_full + stage_idx,
                )
            if cutlass.const_expr(stage_idx == ab_stages - 1):
                a_phase ^= 1
    return a_phase


@cute.jit
def copy_b_sfb(
    tma_atom_b,
    tma_atom_sfb,
    tBgB,
    tBsB,
    tBgSFB,
    tBsSFB,
    tile,
    b_full,
    b_empty,
    b_bytes,
    sfb_bytes,
    sf_cycle_idx,
    sf_stages,
    kt_per_pack,
    ab_stages,
    b_phase,
    swap,
):
    for sf_stage in cutlass.range_constexpr(sf_stages):
        global_sf_stage = sf_cycle_idx * sf_stages + sf_stage
        for k_in_sf in cutlass.range_constexpr(kt_per_pack):
            local_k_tile_idx = sf_stage * kt_per_pack + k_in_sf
            global_k_tile_idx = (
                sf_cycle_idx * sf_stages * kt_per_pack + local_k_tile_idx
            )
            stage_idx = local_k_tile_idx & (ab_stages - 1)
            cute.arch.mbarrier_wait(b_empty + stage_idx, b_phase)
            tx_bytes = b_bytes
            if cutlass.const_expr(not swap or k_in_sf == 0):
                tx_bytes += sfb_bytes
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(b_full + stage_idx, tx_bytes)
            if cutlass.const_expr(swap):
                cute.copy(
                    tma_atom_b,
                    tBgB[(None, tile.n_block, global_k_tile_idx)],
                    tBsB[(None, stage_idx)],
                    tma_bar_ptr=b_full + stage_idx,
                )
                if cutlass.const_expr(k_in_sf == 0):
                    cute.copy(
                        tma_atom_sfb,
                        tBgSFB[(None, tile.n_block, global_sf_stage, 0)],
                        tBsSFB[(None, sf_stage)],
                        tma_bar_ptr=b_full + stage_idx,
                    )
            else:
                cute.copy(
                    tma_atom_b,
                    tBgB[(None, tile.n_block, global_k_tile_idx, tile.group)],
                    tBsB[(None, stage_idx)],
                    tma_bar_ptr=b_full + stage_idx,
                )
                cute.copy(
                    tma_atom_sfb,
                    tBgSFB[(None, tile.n_block, global_k_tile_idx, tile.group)],
                    tBsSFB[(None, stage_idx)],
                    tma_bar_ptr=b_full + stage_idx,
                )
            if cutlass.const_expr(stage_idx == ab_stages - 1):
                b_phase ^= 1
    return b_phase


@cute.jit
def unpack(frg):
    words = cute.recast_tensor(frg, cutlass.Uint32)
    words.store(words.load() * cutlass.Uint32(4))


@cute.jit
def mma(
    tiledmma,
    mma_cfg,
    sf_cfg,
    sA,
    sB,
    sSFA,
    sSFB,
    tile_mn,
    bk,
    acc,
    a_dtype,
    b_dtype,
    a_is_m_major,
    unpack_bits,
    tidx,
    a_full,
    a_empty,
    b_full,
    b_empty,
    sf_stages,
    kt_per_pack,
    ab_stages,
    a_phase,
    b_phase,
    swap,
):
    bm, bn = tile_mn
    thr = tiledmma.get_slice(tidx)
    tCrA = tiledmma.make_fragment_A(thr.partition_A(sA)[None, None, None, 0])
    tCrB = tiledmma.make_fragment_B(thr.partition_B(sB)[None, None, None, 0])
    if cutlass.const_expr(swap):
        s2r_a = mma_cfg.make_s2r_a(tiledmma, a_dtype, False, unpack_bits)
        s2r_b = mma_cfg.make_s2r_b(tiledmma, b_dtype, False)
    else:
        s2r_a = mma_cfg.make_s2r_a(tiledmma, a_dtype, a_is_m_major)
        s2r_b = mma_cfg.make_s2r_b(tiledmma, b_dtype, False, unpack_bits)
    thr_a, thr_b = s2r_a.get_slice(tidx), s2r_b.get_slice(tidx)
    tCrA_v, tCrB_v = thr_a.retile(tCrA), thr_b.retile(tCrB)
    s2r_sfa = sf_cfg.make_s2r_sf(
        sf_cfg.get_layoutSFA_TV(tiledmma), (cute.size(tiledmma.permutation_mnk[0]), 1)
    )
    s2r_sfb = sf_cfg.make_s2r_sf(
        sf_cfg.get_layoutSFB_TV(tiledmma), (cute.size(tiledmma.permutation_mnk[1]), 1)
    )
    thr_sfa, thr_sfb = s2r_sfa.get_slice(tidx), s2r_sfb.get_slice(tidx)
    tCrSFA = sf_cfg.partition_fragment_SFA(
        cute.slice_(sSFA, (None, None, 0)), thr, tidx
    )
    tCrSFA_frg = sf_cfg.make_sfa_ue8m0_view(tCrSFA, bk)
    tCrSFB = sf_cfg.partition_fragment_SFB(
        cute.slice_(sSFB, (None, None, 0)), thr, tidx
    )
    tCrSFB_frg = sf_cfg.make_sfb_ue8m0_view(tCrSFB, bk)
    k_blocks = cute.size(tCrA_v, mode=[2])

    for sf_stage in cutlass.range_constexpr(sf_stages):
        for k_in_sf in cutlass.range_constexpr(kt_per_pack):
            stage_idx = (sf_stage * kt_per_pack + k_in_sf) & (ab_stages - 1)
            cute.arch.mbarrier_wait(a_full + stage_idx, a_phase)
            if cutlass.const_expr(swap):
                cute.copy(
                    s2r_sfa,
                    thr_sfa.partition_S(cute.slice_(sSFA, (None, None, stage_idx))),
                    thr_sfa.retile(tCrSFA),
                )
            elif cutlass.const_expr(k_in_sf == 0):
                cute.copy(
                    s2r_sfa,
                    thr_sfa.partition_S(cute.slice_(sSFA, (None, None, sf_stage))),
                    thr_sfa.retile(tCrSFA),
                )
            cute.arch.mbarrier_wait(b_full + stage_idx, b_phase)
            if cutlass.const_expr(swap):
                if cutlass.const_expr(k_in_sf == 0):
                    cute.copy(
                        s2r_sfb,
                        thr_sfb.partition_S(cute.slice_(sSFB, (None, None, sf_stage))),
                        thr_sfb.retile(tCrSFB),
                    )
            else:
                cute.copy(
                    s2r_sfb,
                    thr_sfb.partition_S(cute.slice_(sSFB, (None, None, stage_idx))),
                    thr_sfb.retile(tCrSFB),
                )
            tAsA_s = thr_a.partition_S(sA)[None, None, None, stage_idx]
            tBsB_s = thr_b.partition_S(sB)[None, None, None, stage_idx]
            for k in cutlass.range_constexpr(0, k_blocks):
                cute.copy(s2r_a, tAsA_s[None, None, k], tCrA_v[None, None, k])
                cute.copy(s2r_b, tBsB_s[None, None, k], tCrB_v[None, None, k])
            unpack(tCrA if swap else tCrB)
            ka, kb = (0, k_in_sf) if swap else (k_in_sf, 0)
            for k_block in cutlass.range_constexpr(0, k_blocks):
                cute.gemm(
                    tiledmma,
                    acc,
                    [tCrA[None, None, k_block], tCrSFA_frg[None, None, k_block, ka]],
                    [tCrB[None, None, k_block], tCrSFB_frg[None, None, k_block, kb]],
                    acc,
                )
            cute.arch.mbarrier_arrive(a_empty + stage_idx)
            cute.arch.mbarrier_arrive(b_empty + stage_idx)
            if cutlass.const_expr(stage_idx == ab_stages - 1):
                a_phase ^= 1
                b_phase ^= 1
    return acc, a_phase, b_phase


class CuteDslSm120MoeMxfp8Mxfp4Grouped:
    def __init__(self, cfg, grid_x):
        self.cfg = cfg
        self.grid_x = grid_x
        self.mma = cfg.mma
        self.sf = cfg.load_sf
        bk = cfg.TILE[2]
        swap = cfg.mma.swap_ab
        kt_fine = (
            self.sf.k_tiles_per_pack_a(bk) if swap else self.sf.k_tiles_per_pack_b(bk)
        )
        assert kt_fine == 1, (
            f"tile-K {bk} puts {kt_fine} k-tiles in one fine-side SF pack; that side shares the A/B "
            f"ring and needs exactly one"
        )
        kt_per_coarse_cycle = (
            (self.sf.sfb_stages(cfg.ab_stage, bk) * self.sf.k_tiles_per_pack_b(bk))
            if swap
            else (self.sf.sfa_stages(cfg.ab_stage, bk) * self.sf.k_tiles_per_pack_a(bk))
        )
        assert kt_per_coarse_cycle % cfg.ab_stage == 0, (
            f"the A/B ring of {cfg.ab_stage} does not divide the {kt_per_coarse_cycle} k-tiles of one "
            f"coarse-SF cycle; every stage index in the k-loop is a trace-time constant that has to "
            f"repeat with that cycle"
        )

    @cute.jit
    def __call__(
        self,
        gA: cute.Tensor,
        gB_u8: cute.Tensor,
        gSFA_u8: cute.Tensor,
        gSFB_u8: cute.Tensor,
        gD: cute.Tensor,
        offsets: cute.Tensor,
        stream,
    ):
        cfg = self.cfg
        tiledmma = cfg.mma.make_tiled_mma(cfg.TILE)
        tiled_r2s = (
            cfg.epi.make_tiled_r2s(tiledmma)
            if cutlass.const_expr(cfg.epi.HAS_R2S)
            else None
        )
        gB_e = cute.recast_tensor(gB_u8, cutlass.Float4E2M1FN)
        E, N, K = (
            cute.size(gB_e, mode=[0]),
            cute.size(gB_e, mode=[1]),
            cute.size(gB_e, mode=[2]),
        )
        gW = cute.make_tensor(
            gB_e.iterator, cute.make_layout((N, K, E), stride=(K, 1, N * K))
        )
        m_padded_sf = cute.size(gSFA_u8, mode=[1])
        sf_dtype = cfg.load_sf.sf_dtype
        if cutlass.const_expr(cfg.mma.swap_ab):
            t_a, t_b = gW, gA
            gSFA = cute.make_tensor(
                cute.recast_ptr(gSFB_u8.iterator, dtype=sf_dtype),
                self.sf.deduce_sfa_layout(N, K, E),
            )
            gSFB = cute.make_tensor(
                cute.recast_ptr(gSFA_u8.iterator, dtype=sf_dtype),
                self.sf.deduce_sfb_layout(m_padded_sf, K, 1),
            )
        else:
            t_a, t_b = gA, gW
            gSFA = cute.make_tensor(
                cute.recast_ptr(gSFA_u8.iterator, dtype=sf_dtype),
                self.sf.deduce_sfa_layout(m_padded_sf, K, 1),
            )
            gSFB = cute.make_tensor(
                cute.recast_ptr(gSFB_u8.iterator, dtype=sf_dtype),
                self.sf.deduce_sfb_layout(N, K, E),
            )
        a_layout = cutlass.utils.LayoutEnum.from_tensor(t_a)
        b_layout = cutlass.utils.LayoutEnum.from_tensor(t_b)
        assert a_layout.is_k_major_a() and b_layout.is_k_major_b(), (
            "LoadABConfig is k-major only"
        )
        a_smem = cfg.load_ab.make_smem_layout_a()
        b_smem = cfg.load_ab.make_smem_layout_b()
        epi_smem = cfg.epi.make_smem_layout(cfg.TILE)
        self.a_is_m_major = a_layout.is_m_major_a()

        bm, bn, bk = cfg.TILE[0], cfg.TILE[1], cfg.TILE[2]
        sfa_stages = cfg.load_sf.sfa_stages(cfg.ab_stage, bk)
        sfb_stages = cfg.load_sf.sfb_stages(cfg.ab_stage, bk)
        sfa_smem = cfg.load_sf.make_smem_layout_sfa(bm, sfa_stages)
        sfb_smem = cfg.load_sf.make_smem_layout_sfb(bn, sfb_stages)

        tma_atom_a, tma_tensor_a = cfg.load_ab.make_tma_atom_a(t_a, a_smem)
        tma_atom_b, tma_tensor_b = cfg.load_ab.make_tma_atom_b(t_b, b_smem)
        tma_atom_sfa, tma_tensor_sfa = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            gSFA,
            cute.slice_(sfa_smem, (None, None, 0)),
            (bm, 1),
            num_multicast=1,
        )
        tma_atom_sfb, tma_tensor_sfb = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            gSFB,
            cute.slice_(sfb_smem, (None, None, 0)),
            (bn, 1),
            num_multicast=1,
        )

        self.a_bytes = cfg.load_ab.tma_bytes_a
        self.b_bytes = cfg.load_ab.tma_bytes_b
        self.sfa_bytes = cute.size_in_bytes(
            sf_dtype, cute.slice_(sfa_smem, (None, None, 0))
        )
        self.sfb_bytes = cute.size_in_bytes(
            sf_dtype, cute.slice_(sfb_smem, (None, None, 0))
        )

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
                cute.struct.MemRange[cfg.load_ab.a_smem_dtype, cute.cosize(a_smem)], 128
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[cfg.load_ab.b_smem_dtype, cute.cosize(b_smem)], 128
            ]
            sSFB: cute.struct.Align[
                cute.struct.MemRange[cfg.load_sf.sf_dtype, cute.cosize(sfb_smem)], 128
            ]
            sSFA: cute.struct.Align[
                cute.struct.MemRange[cfg.load_sf.sf_dtype, cute.cosize(sfa_smem)], 128
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
            a_smem,
            b_smem,
            sfa_smem,
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
        pm, pn = (bn, bm) if cutlass.const_expr(swap) else (bm, bn)
        M, N = cute.size(gD, mode=[0]), cute.size(gD, mode=[1])
        K = cute.size(tma_tensor_b, mode=[1])
        num_groups = cute.size(offsets, mode=[0]) - 1
        num_n_blocks = ceil_div(N, pn)
        gD_t = cute.make_tensor(gD.iterator, cute.make_layout((N, M), stride=(1, N)))
        grank_c = self.sf.grank_b if cutlass.const_expr(swap) else self.sf.grank_a
        kt_per_pack_c = (
            self.sf.k_tiles_per_pack_b(bk)
            if cutlass.const_expr(swap)
            else self.sf.k_tiles_per_pack_a(bk)
        )
        sf_stages = (
            self.sf.sfb_stages(cfg.ab_stage, bk)
            if cutlass.const_expr(swap)
            else self.sf.sfa_stages(cfg.ab_stage, bk)
        )
        num_sf_cycles = ceil_div(K, grank_c * self.sf.PACK_NSF * sf_stages)

        smem = cutlass.utils.SmemAllocator()
        stg = smem.allocate(self.storage)
        sA = stg.sA.get_tensor(a_smem.outer, swizzle=a_smem.inner)
        sB = stg.sB.get_tensor(b_smem.outer, swizzle=b_smem.inner)
        sSFB = stg.sSFB.get_tensor(sfb_smem)
        sSFA = stg.sSFA.get_tensor(sfa_smem)
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
                    cute.arch.mbarrier_init(a_full + s, 1)
                    cute.arch.mbarrier_init(a_empty + s, cfg.mma_threads)
                    cute.arch.mbarrier_init(b_full + s, 1)
                    cute.arch.mbarrier_init(b_empty + s, cfg.mma_threads)
                for s in cutlass.range_constexpr(epi_full_bars):
                    cute.arch.mbarrier_init(epi_full + s, cfg.epi.full_barrier_arrivals)
                for s in cutlass.range_constexpr(epi_empty_bars):
                    cute.arch.mbarrier_init(
                        epi_empty + s, cfg.epi.empty_barrier_arrivals
                    )
                for s in cutlass.range_constexpr(cfg.sched_stages):
                    cute.arch.mbarrier_init(sfull + s, 1)
                    cute.arch.mbarrier_init(sempty + s, cfg.num_sched_consumers)
        cute.arch.mbarrier_init_fence()
        cute.arch.barrier()

        if cutlass.const_expr(cfg.epi.HAS_R2S):
            tiled_s2r = cfg.epi.make_tiled_s2r(cfg.TILE)
        epi_m, epi_n = cfg.epi.epi_tile(cfg.TILE)
        num_epi_m, num_epi_n = cfg.epi.num_epi(cfg.TILE)

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

            if warp_idx == cfg.load_warp_0:
                if cutlass.const_expr(cfg.enable_pdl and not swap):
                    cute.arch.griddepcontrol_wait()
                cons = moe_scheduler.MoeSchedConsumer.create(cfg.sched_stages)
                tile = cons.get_next_tile(sWork, sfull, sempty)
                a_phase, epi_empty_phase = i32(1), i32(1)
                while tile.valid != i32(0):
                    if cutlass.const_expr(cfg.union_smem):
                        cute.arch.mbarrier_wait(epi_empty, epi_empty_phase)
                        epi_empty_phase ^= 1
                    tAsA, tAgA, tAsSFA, tAgSFA = make_a_sfa_partitions(
                        tma_atom_a,
                        tma_atom_sfa,
                        tma_tensor_a,
                        tma_tensor_sfa,
                        sA,
                        sSFA,
                        cfg.TILE,
                        tile,
                        SF_M_ALIGN,
                        swap,
                    )
                    for sf_cycle in cutlass.range(num_sf_cycles):
                        a_phase = copy_a_sfa(
                            tma_atom_a,
                            tma_atom_sfa,
                            tAgA,
                            tAsA,
                            tAgSFA,
                            tAsSFA,
                            tile,
                            a_full,
                            a_empty,
                            self.a_bytes,
                            self.sfa_bytes,
                            sf_cycle,
                            sf_stages,
                            kt_per_pack_c,
                            cfg.ab_stage,
                            a_phase,
                            swap,
                        )
                    tile = cons.get_next_tile(sWork, sfull, sempty)

            if warp_idx == cfg.load_warp_1:
                if cutlass.const_expr(cfg.enable_pdl and swap):
                    cute.arch.griddepcontrol_wait()
                cons = moe_scheduler.MoeSchedConsumer.create(cfg.sched_stages)
                tile = cons.get_next_tile(sWork, sfull, sempty)
                b_phase, epi_empty_phase = i32(1), i32(1)
                while tile.valid != i32(0):
                    if cutlass.const_expr(cfg.union_smem):
                        cute.arch.mbarrier_wait(epi_empty, epi_empty_phase)
                        epi_empty_phase ^= 1
                    tBsB, tBgB, tBsSFB, tBgSFB = make_b_sfb_partitions(
                        tma_atom_b,
                        tma_atom_sfb,
                        tma_tensor_b,
                        tma_tensor_sfb,
                        sB,
                        sSFB,
                        cfg.TILE,
                        tile,
                        SF_M_ALIGN,
                        swap,
                    )
                    for sf_cycle in cutlass.range(num_sf_cycles):
                        b_phase = copy_b_sfb(
                            tma_atom_b,
                            tma_atom_sfb,
                            tBgB,
                            tBsB,
                            tBgSFB,
                            tBsSFB,
                            tile,
                            b_full,
                            b_empty,
                            self.b_bytes,
                            self.sfb_bytes,
                            sf_cycle,
                            sf_stages,
                            kt_per_pack_c,
                            cfg.ab_stage,
                            b_phase,
                            swap,
                        )
                    tile = cons.get_next_tile(sWork, sfull, sempty)

            if cutlass.const_expr(cfg.epi.HAS_STORE_WARP):
                if warp_idx == cfg.store_warp:
                    thr_s2r = tiled_s2r.get_slice(tidx - cfg.store_warp * 32)
                    cons = moe_scheduler.MoeSchedConsumer.create(cfg.sched_stages)
                    tile = cons.get_next_tile(sWork, sfull, sempty)
                    epi_stage, epi_full_phase = i32(0), i32(0)
                    while tile.valid != i32(0):
                        for epi_n_idx in cutlass.range_constexpr(num_epi_n):
                            for epi_m_idx in cutlass.range_constexpr(num_epi_m):
                                sC_stage = cute.slice_(
                                    sC_stages, (None, None, epi_stage)
                                )
                                moe_epilogue.staged_s2g_step(
                                    cfg.epi,
                                    thr_s2r,
                                    sC_stage,
                                    gD,
                                    tile,
                                    (bm, bn),
                                    (epi_m, epi_n),
                                    (epi_m_idx, epi_n_idx),
                                    epi_full + epi_stage,
                                    epi_empty + epi_stage,
                                    epi_full_phase,
                                )
                                epi_stage += 1
                                if epi_stage == cfg.epi.epi_stages:
                                    epi_stage = i32(0)
                                    epi_full_phase ^= 1
                        tile = cons.get_next_tile(sWork, sfull, sempty)

        else:
            cute.arch.setmaxregister_increase(cfg.reg_math)
            thr = tiledmma.get_slice(tidx)
            if cutlass.const_expr(cfg.epi.HAS_R2S):
                thr_r2s = tiled_r2s.get_slice(tidx)
            if cutlass.const_expr(cfg.epi.METHOD is EpiMethod.R2G_WG):
                thr_s2r = tiled_s2r.get_slice(tidx)
            if cutlass.const_expr(swap):
                a_dtype, b_dtype = cfg.load_ab.a_smem_dtype, cfg.load_ab.b_dtype
                unpack_bits = cfg.load_ab.a_unpack_bits
            else:
                a_dtype, b_dtype = cfg.load_ab.a_dtype, cfg.load_ab.b_smem_dtype
                unpack_bits = cfg.load_ab.b_unpack_bits
            if cutlass.const_expr(cfg.enable_pdl):
                cute.arch.griddepcontrol_wait()
            a_phase, b_phase = i32(0), i32(0)
            epi_stage, epi_empty_phase = i32(0), i32(1)
            cons = moe_scheduler.MoeSchedConsumer.create(cfg.sched_stages)
            tile = cons.get_next_tile(sWork, sfull, sempty)
            while tile.valid != i32(0):
                acc = cute.make_rmem_tensor(
                    tiledmma.partition_shape_C((bm, bn)), cfg.ACC
                )
                acc.fill(0.0)
                for sf_cycle in cutlass.range(num_sf_cycles):
                    acc, a_phase, b_phase = mma(
                        tiledmma,
                        self.mma,
                        self.sf,
                        sA,
                        sB,
                        sSFA,
                        sSFB,
                        (bm, bn),
                        bk,
                        acc,
                        a_dtype,
                        b_dtype,
                        self.a_is_m_major,
                        unpack_bits,
                        tidx,
                        a_full,
                        a_empty,
                        b_full,
                        b_empty,
                        sf_stages,
                        kt_per_pack_c,
                        cfg.ab_stage,
                        a_phase,
                        b_phase,
                        swap,
                    )

                if cutlass.const_expr(cfg.epi.METHOD is EpiMethod.DIRECT_STG):
                    moe_epilogue.store_swap(
                        acc, thr, gD_t, tile, (bm, bn), N, cfg.epi.out_dtype
                    )
                elif cutlass.const_expr(cfg.epi.HAS_STORE_WARP):
                    tD = moe_epilogue.convert_acc(acc, cfg.epi.out_dtype)
                    for epi_n_idx in cutlass.range_constexpr(num_epi_n):
                        for epi_m_idx in cutlass.range_constexpr(num_epi_m):
                            sC_stage = cute.slice_(sC_stages, (None, None, epi_stage))
                            moe_epilogue.staged_r2s_step(
                                tD,
                                thr_r2s,
                                sC_stage,
                                (bm, bn),
                                (epi_m, epi_n),
                                (epi_m_idx, epi_n_idx),
                                epi_full + epi_stage,
                                epi_empty + epi_stage,
                                epi_empty_phase,
                                cfg.epi_bar_id,
                                cfg.mma_threads,
                            )
                            epi_stage += 1
                            if epi_stage == cfg.epi.epi_stages:
                                epi_stage = i32(0)
                                epi_empty_phase ^= 1
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


def make_args(a_q, a_scale, b_q, b_scale, out, m_indptr, epi_cfg, tile):
    sf = lambda t: from_dlpack(t.contiguous(), assumed_align=16).mark_layout_dynamic()
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
        _stream(),
    )
