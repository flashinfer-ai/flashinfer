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
"""CuteDSL act-MXFP8 x weight-MXFP4 fc2 with finalize fused into its epilogue, for SM120a."""

import torch

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils
import cutlass.cute.nvgpu.warp.mma as warp_mma
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import from_dlpack

from ._moe_utils.moe_epilogue import EPI_CONFIGS, EpiMethod
from ._moe_utils.moe_kernel_builder import Sm120GemmBuilder, MmaConfig, LoadABConfig
from ._moe_utils.sm12x_blockscaled_layout import Sm120SfConfigMxfp8Mxfp4
from ._moe_utils.sm12x_blockscaled_layout import SF_M_ALIGN
from ....utils import ceil_div
from ._moe_utils import moe_scheduler, moe_epilogue


GRANK_A, GRANK_B = 128, 32
ATOM_MNK = (16, 8, 32)


REG_PROD_BY_TACTIC = {
    (128, 128, EpiMethod.WG_SCATTER): 56,
    (64, 128, EpiMethod.WG_SCATTER): 56,
    (32, 128, EpiMethod.WG_SCATTER): 56,
}

SCATTER_EPIS = (EpiMethod.WG_SCATTER,)


def make_cfg(tile, ab_stage, epi=EpiMethod.WG_SCATTER, enable_pdl=False):
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
    assert epi in SCATTER_EPIS, (
        f"{epi} is not implemented here; this epilogue scatters into rows"
    )
    assert bm >= ATOM_MNK[0], "the fused scatter reads sC, and a swapped tile has none"
    union = EPI_CONFIGS[epi].DRAINS_SC_IN_WG
    return Sm120GemmBuilder(
        MmaConfig(
            warp_mma.MmaMXF8F6F4Op(e4m3, fp4, f32, ue8m0), tile[:2], num_math_warps
        ),
        LoadABConfig(
            tile,
            ab_stage,
            e4m3,
            fp4,
            b_smem_dtype=i8,
            b_tma_internal=i8,
            b_unpack_bits=4,
        ),
        Sm120SfConfigMxfp8Mxfp4(GRANK_A, GRANK_B),
        EPI_CONFIGS[epi](bf16, num_math_warps * 32),
        ab_stage,
        tile,
        epi_bar_id=3,
        union_smem=union,
        reg_prod=REG_PROD_BY_TACTIC[(bm, bn, epi)],
        enable_pdl=enable_pdl,
    )


@cute.jit
def load_ab(
    tma_atom_a,
    tma_atom_b,
    tma_atom_sfb,
    tma_tensor_a,
    tma_tensor_b,
    tma_tensor_sfb,
    sA,
    sB,
    sSFB,
    tile_mnk,
    tile,
    ab_full,
    ab_empty,
    ab_bytes,
    k_tile_count,
    ab_stages,
    ab_phase,
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
    gSFB_nl = cute.local_tile(tma_tensor_sfb, (tile_mnk[1], 1), (None, None, None))
    tBsSFB, tBgSFB = cpasync.tma_partition(
        tma_atom_sfb,
        i32(0),
        multicast,
        cute.group_modes(sSFB, 0, 2),
        cute.group_modes(gSFB_nl, 0, 2),
    )
    for kt_base in cutlass.range(0, k_tile_count, ab_stages):
        for s in cutlass.range_constexpr(ab_stages):
            kt = kt_base + s
            cute.arch.mbarrier_wait(ab_empty + s, ab_phase)
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(ab_full + s, ab_bytes)
            cute.copy(
                tma_atom_a,
                tAgA[(None, tile.m_block, kt)],
                tAsA[(None, s)],
                tma_bar_ptr=ab_full + s,
            )
            cute.copy(
                tma_atom_b,
                tBgB[(None, tile.n_block, kt, tile.group)],
                tBsB[(None, s)],
                tma_bar_ptr=ab_full + s,
            )
            cute.copy(
                tma_atom_sfb,
                tBgSFB[(None, tile.n_block, kt, tile.group)],
                tBsSFB[(None, s)],
                tma_bar_ptr=ab_full + s,
            )
        ab_phase ^= 1
    return ab_phase


@cute.jit
def load_sf(
    tma_atom_sfa,
    tma_tensor_sfa,
    sSFA,
    tile_m,
    tile,
    sf_full,
    sf_empty,
    sfa_bytes,
    m_align,
    num_sf_cycles,
    sf_stages,
    sf_phase,
):
    i32 = cutlass.Int32
    cluster = cute.make_layout((1, 1, 1))
    multicast = cute.make_layout(cute.slice_(cluster, (0, None, 0)).shape)
    sf_m_off = (tile.m_offset + tile.group * i32(m_align - 1)) & i32(-m_align)
    mSFA = cute.domain_offset((sf_m_off, 0, 0), tma_tensor_sfa)
    gSFA_ml = cute.local_tile(mSFA, (tile_m, 1), (None, None, None))
    tAsSFA, tAgSFA = cpasync.tma_partition(
        tma_atom_sfa,
        i32(0),
        multicast,
        cute.group_modes(sSFA, 0, 2),
        cute.group_modes(gSFA_ml, 0, 2),
    )
    for sf_cycle in cutlass.range(0, num_sf_cycles):
        for s in cutlass.range_constexpr(sf_stages):
            cute.arch.mbarrier_wait(sf_empty + s, sf_phase)
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(sf_full + s, sfa_bytes)
            cute.copy(
                tma_atom_sfa,
                tAgSFA[(None, tile.m_block, sf_cycle * sf_stages + s, 0)],
                tAsSFA[(None, s)],
                tma_bar_ptr=sf_full + s,
            )
        sf_phase ^= 1
    return sf_phase


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
    acc_dtype,
    a_dtype,
    b_dtype,
    a_is_m_major,
    unpack_bits,
    tidx,
    ab_full,
    ab_empty,
    sf_full,
    sf_empty,
    num_sf_cycles,
    sf_stages,
    kt_per_pack,
    ab_stages,
    ab_phase,
    sf_phase,
):
    bm, bn = tile_mn
    thr = tiledmma.get_slice(tidx)
    tCrA = tiledmma.make_fragment_A(thr.partition_A(sA)[None, None, None, 0])
    tCrB = tiledmma.make_fragment_B(thr.partition_B(sB)[None, None, None, 0])
    acc = cute.make_rmem_tensor(tiledmma.partition_shape_C((bm, bn)), acc_dtype)
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

    acc.fill(0.0)
    for _sf_cycle in cutlass.range(0, num_sf_cycles):
        for sf_stage in cutlass.range_constexpr(sf_stages):
            cute.arch.mbarrier_wait(sf_full + sf_stage, sf_phase)
            cute.copy(
                s2r_sfa,
                thr_sfa.partition_S(cute.slice_(sSFA, (None, None, sf_stage))),
                thr_sfa.retile(tCrSFA),
            )
            for k_in_sf in cutlass.range_constexpr(kt_per_pack):
                s = (sf_stage * kt_per_pack + k_in_sf) & (ab_stages - 1)
                cute.arch.mbarrier_wait(ab_full + s, ab_phase)
                cute.copy(
                    s2r_sfb,
                    thr_sfb.partition_S(cute.slice_(sSFB, (None, None, s))),
                    thr_sfb.retile(tCrSFB),
                )
                tAsA_s = thr_a.partition_S(sA)[None, None, None, s]
                tBsB_s = thr_b.partition_S(sB)[None, None, None, s]
                for k in cutlass.range_constexpr(0, k_blocks):
                    cute.copy(s2r_a, tAsA_s[None, None, k], tCrA_v[None, None, k])
                    cute.copy(s2r_b, tBsB_s[None, None, k], tCrB_v[None, None, k])
                unpack(tCrB)
                ka, kb = k_in_sf, 0
                for k_block in cutlass.range_constexpr(0, k_blocks):
                    cute.gemm(
                        tiledmma,
                        acc,
                        [
                            tCrA[None, None, k_block],
                            tCrSFA_frg[None, None, k_block, ka],
                        ],
                        [
                            tCrB[None, None, k_block],
                            tCrSFB_frg[None, None, k_block, kb],
                        ],
                        acc,
                    )
                cute.arch.mbarrier_arrive(ab_empty + s)
                if cutlass.const_expr(s == ab_stages - 1):
                    ab_phase ^= 1
            cute.arch.mbarrier_arrive(sf_empty + sf_stage)
        sf_phase ^= 1
    return acc, ab_phase, sf_phase


class CuteDslSm120MoeMxfp8Mxfp4Fc2Finalize:
    def __init__(self, cfg, grid_x):
        self.cfg = cfg
        self.grid_x = grid_x
        self.mma = cfg.mma
        self.sf = cfg.load_sf
        bk = cfg.TILE[2]
        kt_fine = self.sf.k_tiles_per_pack_b(bk)
        assert kt_fine == 1, (
            f"tile-K {bk} puts {kt_fine} k-tiles in one fine-side SF pack; that side shares the A/B "
            f"ring and needs exactly one"
        )
        kt_per_coarse_cycle = self.sf.sfa_stages(
            cfg.ab_stage, bk
        ) * self.sf.k_tiles_per_pack_a(bk)
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
        gTok: cute.Tensor,
        gWt: cute.Tensor,
        offsets: cute.Tensor,
        stream,
    ):
        cfg = self.cfg
        tiledmma = cfg.mma.make_tiled_mma(cfg.TILE)
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

        ring_smem, warp_smem, coarse_stages = sfb_smem, sfa_smem, sfa_stages
        self.ab_sfb_bytes = cfg.load_ab.tma_bytes_ab + cute.size_in_bytes(
            sf_dtype, cute.slice_(ring_smem, (None, None, 0))
        )
        self.sf_bytes = cute.size_in_bytes(
            sf_dtype, cute.slice_(warp_smem, (None, None, 0))
        )

        store_full_bars, store_empty_bars = 0, cfg.store_stages
        epi_elems = 0

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
            tma_atom_a,
            tma_atom_b,
            tma_atom_sfa,
            tma_atom_sfb,
            tma_tensor_a,
            tma_tensor_b,
            tma_tensor_sfa,
            tma_tensor_sfb,
            gD,
            gTok,
            gWt,
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
        tma_atom_a,
        tma_atom_b,
        tma_atom_sfa,
        tma_atom_sfb,
        tma_tensor_a: cute.Tensor,
        tma_tensor_b: cute.Tensor,
        tma_tensor_sfa: cute.Tensor,
        tma_tensor_sfb: cute.Tensor,
        gD: cute.Tensor,
        gTok: cute.Tensor,
        gWt: cute.Tensor,
        offsets: cute.Tensor,
        a_smem,
        b_smem,
        sfa_smem,
        sfb_smem,
        epi_smem,
    ):
        cfg = self.cfg
        i32 = cutlass.Int32
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        bidx, _, _ = cute.arch.block_idx()
        bm, bn, bk = cfg.TILE[0], cfg.TILE[1], cfg.TILE[2]
        pm, pn = bm, bn
        N = cute.size(gD, mode=[1])
        K = cute.size(tma_tensor_b, mode=[1])
        num_groups = cute.size(offsets, mode=[0]) - 1
        num_n_blocks = ceil_div(N, pn)
        grank_c = self.sf.grank_a
        kt_per_pack_c = self.sf.k_tiles_per_pack_a(bk)
        sf_stages_c = self.sf.sfa_stages(cfg.ab_stage, bk)
        num_sf_cycles = ceil_div(K, grank_c * self.sf.PACK_NSF * sf_stages_c)
        k_tile_count = num_sf_cycles * sf_stages_c * kt_per_pack_c

        smem = cutlass.utils.SmemAllocator()
        stg = smem.allocate(self.storage)
        sA = stg.sA.get_tensor(a_smem.outer, swizzle=a_smem.inner)
        sB = stg.sB.get_tensor(b_smem.outer, swizzle=b_smem.inner)
        sSFB = stg.sSFB.get_tensor(sfb_smem)
        sSFA = stg.sSFA.get_tensor(sfa_smem)
        sC_stages = cute.make_tensor(
            cute.recast_ptr(stg.sA.data_ptr(), dtype=cfg.epi.out_dtype), epi_smem
        )
        sC = cute.slice_(sC_stages, (None, None, 0))
        sTok_ptr = cute.recast_ptr(
            sC_stages.iterator + cute.cosize(epi_smem), dtype=cfg.I32
        )
        sTok = cute.make_tensor(sTok_ptr, cute.make_layout(bm))
        sWt = cute.make_tensor(
            cute.recast_ptr(sTok_ptr + bm, dtype=cutlass.Float32), cute.make_layout(bm)
        )
        ab_full, ab_empty = stg.ab_full.data_ptr(), stg.ab_empty.data_ptr()
        sf_full, sf_empty = stg.sf_full.data_ptr(), stg.sf_empty.data_ptr()
        store_empty = stg.store_empty.data_ptr()
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
                for s in cutlass.range_constexpr(cfg.store_stages):
                    cute.arch.mbarrier_init(store_empty + s, cfg.store_threads)
                for s in cutlass.range_constexpr(cfg.sched_stages):
                    cute.arch.mbarrier_init(sfull + s, 1)
                    cute.arch.mbarrier_init(sempty + s, cfg.num_sched_consumers)
        cute.arch.mbarrier_init_fence()
        cute.arch.barrier()

        st_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cfg.epi.out_dtype)
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
                abphase, stphase = i32(1), i32(1)
                while tile.valid != i32(0):
                    if cutlass.const_expr(cfg.union_smem):
                        cute.arch.mbarrier_wait(store_empty, stphase)
                        stphase ^= 1
                    abphase = load_ab(
                        tma_atom_a,
                        tma_atom_b,
                        tma_atom_sfb,
                        tma_tensor_a,
                        tma_tensor_b,
                        tma_tensor_sfb,
                        sA,
                        sB,
                        sSFB,
                        cfg.TILE,
                        tile,
                        ab_full,
                        ab_empty,
                        self.ab_sfb_bytes,
                        k_tile_count,
                        cfg.ab_stage,
                        abphase,
                    )
                    tile = cons.get_next_tile(sWork, sfull, sempty)

            if warp_idx == cfg.sf_warp:
                cons = moe_scheduler.MoeSchedConsumer.create(cfg.sched_stages)
                tile = cons.get_next_tile(sWork, sfull, sempty)
                sfaphase = i32(1)
                while tile.valid != i32(0):
                    sfaphase = load_sf(
                        tma_atom_sfa,
                        tma_tensor_sfa,
                        sSFA,
                        bm,
                        tile,
                        sf_full,
                        sf_empty,
                        self.sf_bytes,
                        SF_M_ALIGN,
                        num_sf_cycles,
                        sf_stages_c,
                        sfaphase,
                    )
                    tile = cons.get_next_tile(sWork, sfull, sempty)

        else:
            cute.arch.setmaxregister_increase(cfg.reg_math)
            thr = tiledmma.get_slice(tidx)
            thr_st = tiled_st.get_slice(tidx)
            a_dtype, b_dtype = cfg.load_ab.a_dtype, cfg.load_ab.b_smem_dtype
            unpack_bits = cfg.load_ab.b_unpack_bits
            abphase, sfaphase, stphase = i32(0), i32(0), i32(1)
            cons = moe_scheduler.MoeSchedConsumer.create(cfg.sched_stages)
            tile = cons.get_next_tile(sWork, sfull, sempty)
            while tile.valid != i32(0):
                acc, abphase, sfaphase = mma(
                    tiledmma,
                    self.mma,
                    self.sf,
                    sA,
                    sB,
                    sSFA,
                    sSFB,
                    (bm, bn),
                    bk,
                    cfg.ACC,
                    a_dtype,
                    b_dtype,
                    self.a_is_m_major,
                    unpack_bits,
                    tidx,
                    ab_full,
                    ab_empty,
                    sf_full,
                    sf_empty,
                    num_sf_cycles,
                    sf_stages_c,
                    kt_per_pack_c,
                    cfg.ab_stage,
                    abphase,
                    sfaphase,
                )

                moe_epilogue.store_wg_scatter(
                    acc,
                    thr,
                    thr_st,
                    sC,
                    sTok,
                    sWt,
                    gTok,
                    gWt,
                    gD,
                    tile,
                    (bm, bn),
                    N,
                    cfg.epi.out_dtype,
                    store_empty,
                    cfg.epi_bar_id,
                    cfg.mma_threads,
                    tidx,
                )
                tile = cons.get_next_tile(sWork, sfull, sempty)


def _stream():
    return cuda.CUstream(torch.cuda.current_stream().cuda_stream)


def make_args(a_q, a_scale, b_q, b_scale, out, src_token, pair_scales, m_indptr):
    sf = lambda t: from_dlpack(t.contiguous(), assumed_align=16).mark_layout_dynamic()
    return (
        from_dlpack(a_q).mark_layout_dynamic(),
        from_dlpack(b_q).mark_layout_dynamic(),
        sf(a_scale),
        sf(b_scale),
        from_dlpack(out).mark_layout_dynamic(),
        sf(src_token),
        sf(pair_scales),
        from_dlpack(m_indptr).mark_layout_dynamic(),
        _stream(),
    )
