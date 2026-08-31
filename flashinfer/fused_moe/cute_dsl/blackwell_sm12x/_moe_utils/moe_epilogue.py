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
"""SM120 epilogue: the sC contract and the three phases that touch it (R2S / S2R / R2G)."""

import abc
import dataclasses
import enum
from typing import Optional

import cutlass
import cutlass.utils
import cutlass.cute as cute
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

MXFP8_MAX = 448.0


class EpiMethod(enum.Enum):
    R2G_WG = enum.auto()
    STAGED_R2G = enum.auto()
    DIRECT_STG = enum.auto()
    WG_S2R_QUANT = enum.auto()
    WG_QUANT_R2S = enum.auto()
    WG_QUANT_SWAP = enum.auto()
    WG_SCATTER = enum.auto()


class QuantPoint(enum.Enum):
    AFTER_S2R = enum.auto()
    BEFORE_R2S = enum.auto()


class EpiConfig(abc.ABC):
    STG_BYTES = 16
    METHOD: Optional[EpiMethod] = None
    HAS_STORE_WARP: Optional[bool] = None
    QUANT_AT: Optional[QuantPoint] = None
    DRAINS_SC_IN_WG = False
    _PERIOD_BYTES = {
        cute.nvgpu.warpgroup.SmemLayoutAtomKind.K_SW128: 128,
        cute.nvgpu.warpgroup.SmemLayoutAtomKind.K_SW64: 64,
        cute.nvgpu.warpgroup.SmemLayoutAtomKind.K_SW32: 32,
        cute.nvgpu.warpgroup.SmemLayoutAtomKind.K_INTER: STG_BYTES,
        cute.nvgpu.warpgroup.SmemLayoutAtomKind.MN_SW128: 128,
        cute.nvgpu.warpgroup.SmemLayoutAtomKind.MN_SW64: 64,
        cute.nvgpu.warpgroup.SmemLayoutAtomKind.MN_SW32: 32,
        cute.nvgpu.warpgroup.SmemLayoutAtomKind.MN_INTER: STG_BYTES,
    }

    def __init__(self, out_dtype, mma_threads, epi_stage=1, store_threads=None):
        self.out_dtype = out_dtype
        self.mma_threads = mma_threads
        self.epi_stage = epi_stage
        self.quant_at = self.QUANT_AT
        self._store_threads = mma_threads if store_threads is None else store_threads

    @property
    def s2g_vec(self):
        vec = self.STG_BYTES * 8 // self.out_dtype.width
        assert vec >= 1 and vec & (vec - 1) == 0, (
            f"{self.out_dtype} gives {vec} elements per {self.STG_BYTES} B store; the swizzle's M "
            f"mode counts elements and needs a power of two"
        )
        return vec

    def smem_layout_atom_kind(self, tile):
        return sm90_utils.get_smem_layout_atom(
            cutlass.utils.LayoutEnum.ROW_MAJOR, self.out_dtype, self.epi_tile(tile)[1]
        )

    def epi_tile(self, tile):
        return tile[0], tile[1]

    def num_epi(self, tile):
        epi_m, epi_n = self.epi_tile(tile)
        return tile[0] // epi_m, tile[1] // epi_n

    @property
    @abc.abstractmethod
    def num_store_threads(self):
        pass

    def smem_bytes(self, tile):
        epi_m, epi_n = self.epi_tile(tile)
        return epi_m * epi_n * self.epi_stage * self.out_dtype.width // 8

    def aux_smem_bytes(self, tile):
        return 0

    def make_smem_layout(self, tile):
        epi_m, epi_n = self.epi_tile(tile)
        atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            self.smem_layout_atom_kind(tile), self.out_dtype
        )
        return cute.tile_to_shape(atom, (epi_m, epi_n, self.epi_stage), order=(0, 1, 2))

    def s2g_threads_n(self, tile):
        return self._PERIOD_BYTES[self.smem_layout_atom_kind(tile)] // self.STG_BYTES

    def s2g_thr_layout(self, tile):
        threads_n = self.s2g_threads_n(tile)
        threads_m, rem = divmod(self.num_store_threads, threads_n)
        assert rem == 0, (
            f"{type(self).__name__}: {self.num_store_threads} draining threads do not split into "
            f"rows of {threads_n}; the S2G thread layout must cover the participating threads "
            f"exactly, or the threads left out write past their slice of gD"
        )
        return cute.make_layout((threads_m, threads_n), stride=(threads_n, 1))

    def make_tiled_s2g(self, atom, tile):
        return cute.make_tiled_copy_tv(
            atom, self.s2g_thr_layout(tile), cute.make_layout((1, self.s2g_vec))
        )


class R2GWgEpiConfig(EpiConfig):
    METHOD = EpiMethod.R2G_WG
    HAS_STORE_WARP = False
    DRAINS_SC_IN_WG = True

    @property
    def num_store_threads(self):
        return self._store_threads


class StagedR2GEpiConfig(EpiConfig):
    METHOD = EpiMethod.STAGED_R2G
    HAS_STORE_WARP = True
    STORE_THREADS = 32

    @property
    def num_store_threads(self):
        return self.STORE_THREADS


class DirectStgEpiConfig(EpiConfig):
    METHOD = EpiMethod.DIRECT_STG
    HAS_STORE_WARP = False

    @property
    def num_store_threads(self):
        return self.mma_threads

    def smem_bytes(self, tile):
        return 0


class WgS2rQuantEpiConfig(R2GWgEpiConfig):
    METHOD = EpiMethod.WG_S2R_QUANT
    QUANT_AT = QuantPoint.AFTER_S2R


class WgQuantR2sEpiConfig(R2GWgEpiConfig):
    METHOD = EpiMethod.WG_QUANT_R2S
    QUANT_AT = QuantPoint.BEFORE_R2S


class WgQuantSwapEpiConfig(R2GWgEpiConfig):
    METHOD = EpiMethod.WG_QUANT_SWAP
    QUANT_AT = QuantPoint.AFTER_S2R
    DRAINS_SC_IN_WG = False

    @staticmethod
    def store_threads_for(out_dtype, tile):
        kind = sm90_utils.get_smem_layout_atom(
            cutlass.utils.LayoutEnum.COL_MAJOR, out_dtype, tile[0]
        )
        return EpiConfig._PERIOD_BYTES[kind] // EpiConfig.STG_BYTES * tile[1]

    def smem_layout_atom_kind(self, tile):
        return sm90_utils.get_smem_layout_atom(
            cutlass.utils.LayoutEnum.COL_MAJOR, self.out_dtype, self.epi_tile(tile)[0]
        )

    def s2g_thr_layout(self, tile):
        threads_m = self.s2g_threads_n(tile)
        epi_n = self.epi_tile(tile)[1]
        assert self.num_store_threads % cute.arch.WARP_SIZE == 0, (
            f"{self.num_store_threads} draining threads is not a whole number of warps, so the "
            f"guard around the reduction diverges inside a warp and the shuffle wedges"
        )
        threads_n, rem = divmod(self.num_store_threads, threads_m)
        assert rem == 0 and threads_n == epi_n, (
            f"{self.num_store_threads} draining threads do not split into {threads_m} along M by "
            f"{epi_n} along N; a column would be split across lane groups the reduce cannot cross"
        )
        return cute.make_layout((threads_m, threads_n), stride=(1, threads_m))

    def make_tiled_s2g(self, atom, tile):
        return cute.make_tiled_copy_tv(
            atom, self.s2g_thr_layout(tile), cute.make_layout((self.s2g_vec, 1))
        )


class WgScatterEpiConfig(R2GWgEpiConfig):
    METHOD = EpiMethod.WG_SCATTER

    def aux_smem_bytes(self, tile):
        return tile[0] * (cutlass.Int32.width + cutlass.Float32.width) // 8


EPI_CONFIGS = {
    EpiMethod.STAGED_R2G: StagedR2GEpiConfig,
    EpiMethod.R2G_WG: R2GWgEpiConfig,
    EpiMethod.DIRECT_STG: DirectStgEpiConfig,
    EpiMethod.WG_S2R_QUANT: WgS2rQuantEpiConfig,
    EpiMethod.WG_QUANT_R2S: WgQuantR2sEpiConfig,
    EpiMethod.WG_QUANT_SWAP: WgQuantSwapEpiConfig,
    EpiMethod.WG_SCATTER: WgScatterEpiConfig,
}


def scatter_supports(hidden: int) -> bool:
    return hidden % 8 == 0


def convert_acc(acc, out_dtype):
    tD = cute.make_fragment_like(acc, out_dtype)
    tD.store(acc.load().to(out_dtype))
    return tD


def rmem_to_smem(tD, thr_mma, sC):
    cute.autovec_copy(tD, thr_mma.partition_C(sC))


@cute.jit
def smem_to_gmem(
    atom, thr_st, sC, gD_tile, tile_mn, m_base, n_base, m_boundary, n_boundary
):
    bm, bn = tile_mn
    cD = cute.make_identity_tensor((bm, bn))
    tDsC = thr_st.partition_S(sC)
    tDgD = thr_st.partition_D(gD_tile)
    tDcD = thr_st.partition_S(cD)
    n_vec, m_iter, n_iter = (
        cute.size(tDsC, mode=[0]),
        cute.size(tDsC, mode=[1]),
        cute.size(tDsC, mode=[2]),
    )
    tDpD = cute.make_rmem_tensor(
        cute.make_layout((n_vec, m_iter, n_iter), stride=(1, 0, n_vec)), cutlass.Boolean
    )
    for v in cutlass.range_constexpr(n_vec):
        for j in cutlass.range_constexpr(n_iter):
            tDpD[v, 0, j] = (n_base + tDcD[v, 0, j][1]) < n_boundary
    for i in cutlass.range_constexpr(m_iter):
        if m_base + tDcD[0, i, 0][0] < m_boundary:
            for j in cutlass.range_constexpr(n_iter):
                src = tDsC[None, i, j]
                reg = cute.make_fragment_like(src)
                cute.autovec_copy(src, reg)
                cute.copy(atom, reg, tDgD[None, i, j], pred=tDpD[None, i, j])


@cute.jit
def rmem_to_gmem(
    thr_mma, tDrD, gD_tile, tile_mn, m_base, n_base, m_boundary, n_boundary
):
    bm, bn = tile_mn
    cD = cute.make_identity_tensor((bm, bn))
    tDgD = thr_mma.partition_C(gD_tile)
    tDcD = thr_mma.partition_C(cD)
    for i in cutlass.range_constexpr(cute.size(tDrD)):
        mi, ni = tDcD[i]
        if m_base + mi < m_boundary and n_base + ni < n_boundary:
            tDgD[i] = tDrD[i]


@cute.jit
def store_staged(acc, thr, sC, out_dtype, store_full, store_empty, store_phase):
    cute.arch.mbarrier_wait(store_empty, store_phase)
    rmem_to_smem(convert_acc(acc, out_dtype), thr, sC)
    cute.arch.mbarrier_arrive(store_full)
    return store_phase ^ 1


@cute.jit
def store_wg(
    acc,
    thr,
    thr_st,
    sC,
    st_atom,
    gD,
    tile,
    tile_mn,
    n_boundary,
    out_dtype,
    store_empty,
    epi_bar_id,
    mma_threads,
):
    bm, bn = tile_mn
    cute.arch.barrier(barrier_id=epi_bar_id, number_of_threads=mma_threads)
    rmem_to_smem(convert_acc(acc, out_dtype), thr, sC)
    cute.arch.barrier(barrier_id=epi_bar_id, number_of_threads=mma_threads)
    gD_tile = cute.local_tile(
        cute.domain_offset((tile.m_offset, 0), gD),
        (bm, bn),
        (tile.m_block, tile.n_block),
    )
    smem_to_gmem(
        st_atom,
        thr_st,
        sC,
        gD_tile,
        (bm, bn),
        tile.m_offset + tile.m_block * bm,
        tile.n_block * bn,
        tile.m_boundary,
        n_boundary,
    )
    cute.arch.barrier(barrier_id=epi_bar_id, number_of_threads=mma_threads)
    cute.arch.mbarrier_arrive(store_empty)


@dsl_user_op
def ptr_as_int64(tensor: cute.Tensor, offset, *, loc=None, ip=None) -> cutlass.Int64:
    return cutlass.Int64(
        llvm.ptrtoint(T.i64(), (tensor.iterator + offset).llvm_ptr, loc=loc, ip=ip)
    )


@dsl_user_op
def scatter_add_v4_bf16x2(addr, v0, v1, v2, v3, v4, v5, v6, v7, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [cutlass.Int64(addr).ir_value(loc=loc, ip=ip)]
        + [v.ir_value(loc=loc, ip=ip) for v in (v0, v1, v2, v3, v4, v5, v6, v7)],
        "{ .reg .b32 p0,p1,p2,p3;"
        " cvt.rn.satfinite.bf16x2.f32 p0, $2, $1;"
        " cvt.rn.satfinite.bf16x2.f32 p1, $4, $3;"
        " cvt.rn.satfinite.bf16x2.f32 p2, $6, $5;"
        " cvt.rn.satfinite.bf16x2.f32 p3, $8, $7;"
        " red.global.add.noftz.v4.bf16x2 [$0], {p0, p1, p2, p3}; }",
        "l,f,f,f,f,f,f,f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@cute.jit
def emit_scatter(gOut, base, rem, v0, v1, v2, v3, v4, v5, v6, v7):
    if rem >= cutlass.Int32(8):
        scatter_add_v4_bf16x2(ptr_as_int64(gOut, base), v0, v1, v2, v3, v4, v5, v6, v7)


@cute.jit
def stage_scatter_meta(sTok, sWt, gTok, gWt, m_base, m_valid, tidx, bm):
    i32 = cutlass.Int32
    assert bm & (bm - 1) == 0, (
        f"bm={bm} is not a power of two, so the mask below is not a modulo"
    )
    slot = tidx & (bm - 1)
    safe = min(slot, max(m_valid - i32(1), i32(0)))
    keep = min(max(m_valid - slot, i32(0)), i32(1))
    sTok[slot] = gTok[m_base + safe]
    sWt[slot] = gWt[m_base + safe] * keep.to(cutlass.Float32)


@cute.jit
def smem_to_gmem_scatter(thr_st, sC, sTok, sWt, gOut, tile_mn, n_base, hidden):
    bm, bn = tile_mn
    f32 = cutlass.Float32
    tDsC = thr_st.partition_S(sC)
    tDcD = thr_st.partition_S(cute.make_identity_tensor((bm, bn)))
    m_iter, n_iter = cute.size(tDsC, mode=[1]), cute.size(tDsC, mode=[2])
    for i in cutlass.range_constexpr(m_iter):
        local = tDcD[0, i, 0][0]
        tok = sTok[local].to(cutlass.Int64)
        w = sWt[local].to(f32)
        for j in cutlass.range_constexpr(n_iter):
            src = tDsC[None, i, j]
            reg = cute.make_fragment_like(src)
            cute.autovec_copy(src, reg)
            col = n_base + tDcD[0, i, j][1]
            base = tok * hidden.to(cutlass.Int64) + col.to(cutlass.Int64)
            v0, v1 = w * reg[0].to(f32), w * reg[1].to(f32)
            v2, v3 = w * reg[2].to(f32), w * reg[3].to(f32)
            v4, v5 = w * reg[4].to(f32), w * reg[5].to(f32)
            v6, v7 = w * reg[6].to(f32), w * reg[7].to(f32)
            emit_scatter(gOut, base, hidden - col, v0, v1, v2, v3, v4, v5, v6, v7)


@cute.jit
def store_wg_scatter(
    acc,
    thr,
    thr_st,
    sC,
    sTok,
    sWt,
    gTok,
    gWt,
    gOut,
    tile,
    tile_mn,
    hidden,
    out_dtype,
    store_empty,
    epi_bar_id,
    mma_threads,
    tidx,
):
    bm, bn = tile_mn
    i32 = cutlass.Int32
    cute.arch.barrier(barrier_id=epi_bar_id, number_of_threads=mma_threads)
    rmem_to_smem(convert_acc(acc, out_dtype), thr, sC)
    m_base = tile.m_offset + tile.m_block * bm
    stage_scatter_meta(
        sTok,
        sWt,
        gTok,
        gWt,
        m_base,
        min(max(tile.m_boundary - m_base, i32(0)), i32(bm)),
        tidx,
        bm,
    )
    cute.arch.barrier(barrier_id=epi_bar_id, number_of_threads=mma_threads)
    smem_to_gmem_scatter(
        thr_st, sC, sTok, sWt, gOut, (bm, bn), tile.n_block * bn, hidden
    )
    cute.arch.barrier(barrier_id=epi_bar_id, number_of_threads=mma_threads)
    cute.arch.mbarrier_arrive(store_empty)


@cute.jit
def store_swap(acc, thr, gD_t, tile, tile_mn, n_total, out_dtype):
    bm, bn = tile_mn
    gD_tile = cute.local_tile(
        cute.domain_offset((0, tile.m_offset), gD_t),
        (bm, bn),
        (tile.m_block, tile.n_block),
    )
    rmem_to_gmem(
        thr,
        convert_acc(acc, out_dtype),
        gD_tile,
        (bm, bn),
        tile.m_block * bm,
        tile.m_offset + tile.n_block * bn,
        n_total,
        tile.m_boundary,
    )


@cute.jit
def drain_staged(
    st_atom, thr_st, sC, gD, tile, tile_mn, n_boundary, store_full, store_empty, phase
):
    bm, bn = tile_mn
    cute.arch.mbarrier_wait(store_full, phase)
    gD_tile = cute.local_tile(
        cute.domain_offset((tile.m_offset, 0), gD),
        (bm, bn),
        (tile.m_block, tile.n_block),
    )
    smem_to_gmem(
        st_atom,
        thr_st,
        sC,
        gD_tile,
        (bm, bn),
        tile.m_offset + tile.m_block * bm,
        tile.n_block * bn,
        tile.m_boundary,
        n_boundary,
    )
    cute.arch.mbarrier_arrive(store_empty)
    return phase ^ 1


@dataclasses.dataclass(frozen=True)
class QuantConst:
    amax_floor: float = 1e-4
    sf_max: float = MXFP8_MAX
    is_ue8m0: bool = True


@cute.jit
def ue8m0_parts(sf):
    buf = cute.make_rmem_tensor(cute.make_layout(1), cutlass.Float32)
    buf[0] = sf
    words = cute.recast_tensor(buf, cutlass.Int32)
    e = (words[0] + cutlass.Int32(0x7FFFFF)) >> cutlass.Int32(23)
    e = max(cutlass.Int32(1), min(cutlass.Int32(254), e))
    words[0] = e << cutlass.Int32(23)
    return buf[0], e


@cute.jit
def quant_row_parts(amax, cfg):
    sf = amax * cutlass.Float32(1.0 / cfg.sf_max)
    if cutlass.const_expr(cfg.is_ue8m0):
        return ue8m0_parts(sf)
    return sf, sf


@cute.jit
def quant_apply(vals, sf, cfg):
    if cutlass.const_expr(cfg.is_ue8m0):
        return vals * (cutlass.Float32(1.0) / sf)
    return vals / sf


@cute.jit
def sf_ctx_of(sf_col, block, pack_nsf, cfg):
    if cutlass.const_expr(cfg.is_ue8m0):
        return sf_col, block >> (pack_nsf.bit_length() - 1), block & (pack_nsf - 1)
    return sf_col, block


@cute.jit
def sf_store(gSF, ctx, row, sf_val, cfg):
    if cutlass.const_expr(cfg.is_ue8m0):
        col, pack, byte = ctx
        gSF[pack, col + row, byte] = sf_val.to(cutlass.Uint8)
    else:
        col, k_block = ctx
        gSF[k_block, col + row] = sf_val


@cute.jit
def epi_s2r_quant_stg(
    q_atom,
    thr_st,
    sC,
    gQ_tile,
    gSF,
    sf_ctx,
    cfg,
    tile_mn,
    m_base,
    n_base,
    m_boundary,
    n_boundary,
    threads_n,
):
    bm, bn = tile_mn
    cD = cute.make_identity_tensor((bm, bn))
    tS = thr_st.partition_S(sC)
    tD = thr_st.partition_D(gQ_tile)
    tCc = thr_st.partition_S(cD)
    n_vec, m_iter, n_iter = (
        cute.size(tS, mode=[0]),
        cute.size(tS, mode=[1]),
        cute.size(tS, mode=[2]),
    )
    tP = cute.make_rmem_tensor(
        cute.make_layout((n_vec, m_iter, n_iter), stride=(1, 0, n_vec)), cutlass.Boolean
    )
    for v in cutlass.range_constexpr(n_vec):
        for j in cutlass.range_constexpr(n_iter):
            tP[v, 0, j] = (n_base + tCc[v, 0, j][1]) < n_boundary
    for mi in cutlass.range_constexpr(m_iter):
        row = tCc[0, mi, 0][0]
        amax = cutlass.Float32(0.0)
        for j in cutlass.range_constexpr(n_iter):
            vals = tS[None, mi, j].load().to(cutlass.Float32)
            amax = cute.math.max(
                amax,
                cute.math.abs(vals).reduce(
                    cute.ReductionOp.MAX,
                    init_val=cutlass.Float32(0.0),
                    reduction_profile=0,
                ),
            )
        amax = cute.arch.warp_reduction_max(amax, threads_in_group=threads_n)
        amax = cute.math.max(amax, cutlass.Float32(cfg.amax_floor))
        sf, sf_val = quant_row_parts(amax, cfg)
        if m_base + row < m_boundary:
            sf_store(gSF, sf_ctx, row, sf_val, cfg)
            for j in cutlass.range_constexpr(n_iter):
                src = tS[None, mi, j]
                reg = cute.make_fragment_like(src)
                cute.autovec_copy(src, reg)
                frg = cute.make_fragment_like(src, gQ_tile.element_type)
                frg.store(
                    quant_apply(reg.load().to(cutlass.Float32), sf, cfg).to(
                        gQ_tile.element_type
                    )
                )
                cute.copy(q_atom, frg, tD[None, mi, j], pred=tP[None, mi, j])


@cute.jit
def store_wg_q1_after_s2r(
    acc,
    thr,
    thr_st,
    sC,
    q_atom,
    gQ,
    gSFQ,
    tile,
    tile_mn,
    n_boundary,
    sC_dtype,
    qcfg,
    store_empty,
    epi_bar_id,
    mma_threads,
    sf_m_align,
    pack_nsf,
    threads_n,
):
    bm, bn = tile_mn
    cute.arch.barrier(barrier_id=epi_bar_id, number_of_threads=mma_threads)
    rmem_to_smem(convert_acc(acc, sC_dtype), thr, sC)
    cute.arch.barrier(barrier_id=epi_bar_id, number_of_threads=mma_threads)
    gQ_tile = cute.local_tile(
        cute.domain_offset((tile.m_offset, 0), gQ),
        (bm, bn),
        (tile.m_block, tile.n_block),
    )
    sf_col = (
        (tile.m_offset + tile.group * (sf_m_align - 1)) & -sf_m_align
    ) + tile.m_block * bm
    ctx = sf_ctx_of(sf_col, tile.n_block, pack_nsf, qcfg)
    epi_s2r_quant_stg(
        q_atom,
        thr_st,
        sC,
        gQ_tile,
        gSFQ,
        ctx,
        qcfg,
        (bm, bn),
        tile.m_offset + tile.m_block * bm,
        tile.n_block * bn,
        tile.m_boundary,
        n_boundary,
        threads_n,
    )
    cute.arch.barrier(barrier_id=epi_bar_id, number_of_threads=mma_threads)
    cute.arch.mbarrier_arrive(store_empty)


@cute.jit
def epi_quant_s2r_stg(
    st_atom,
    thr,
    thr_st,
    tX_acc,
    sD,
    xchg,
    warp_n,
    num_warp_n,
    gQ_tile,
    gSF,
    sf_ctx,
    cfg,
    tile_mn,
    m_base,
    n_base,
    m_boundary,
    n_boundary,
    bar_id,
    bar_threads,
    sC_dtype,
):
    bm, bn = tile_mn
    tX = convert_acc(tX_acc, sC_dtype)
    tC = thr.partition_C(cute.make_identity_tensor((bm, bn)))
    mrows, n_ext = 2, 2
    scl = cute.make_rmem_tensor(cute.make_layout(mrows), cutlass.Float32)
    q_dtype = gQ_tile.element_type
    for r in cutlass.range_constexpr(mrows):
        vals = cute.slice_(tX, ((None, r), None, None)).load()
        a = cute.math.abs(vals).reduce(
            cute.ReductionOp.MAX, init_val=cutlass.Float32(0.0), reduction_profile=0
        )
        xchg[warp_n * bm + tC[r * n_ext][0]] = cute.arch.warp_reduction_max(
            a, threads_in_group=4
        )
    cute.arch.barrier(barrier_id=bar_id, number_of_threads=bar_threads)
    for r in cutlass.range_constexpr(mrows):
        row = tC[r * n_ext][0]
        amax = cutlass.Float32(cfg.amax_floor)
        for w in cutlass.range_constexpr(num_warp_n):
            amax = cute.math.max(amax, xchg[w * bm + row])
        sf, sf_val = quant_row_parts(amax, cfg)
        if m_base + row < m_boundary:
            sf_store(gSF, sf_ctx, row, sf_val, cfg)
        scl[r] = sf
    frg = cute.make_fragment_like(tX, q_dtype)
    for r in cutlass.range_constexpr(mrows):
        acc_src = cute.slice_(tX, ((None, r), None, None))
        acc_dst = cute.slice_(frg, ((None, r), None, None))
        acc_dst.store(
            quant_apply(acc_src.load().to(cutlass.Float32), scl[r], cfg).to(q_dtype)
        )
    cute.autovec_copy(frg, thr.partition_C(sD))
    cute.arch.barrier(barrier_id=bar_id, number_of_threads=bar_threads)
    cD = cute.make_identity_tensor((bm, bn))
    tS = thr_st.partition_S(sD)
    tD = thr_st.partition_D(gQ_tile)
    tCc = thr_st.partition_S(cD)
    n_vec, m_iter, n_iter = (
        cute.size(tS, mode=[0]),
        cute.size(tS, mode=[1]),
        cute.size(tS, mode=[2]),
    )
    tP = cute.make_rmem_tensor(
        cute.make_layout((n_vec, m_iter, n_iter), stride=(1, 0, n_vec)), cutlass.Boolean
    )
    for v in cutlass.range_constexpr(n_vec):
        for j in cutlass.range_constexpr(n_iter):
            tP[v, 0, j] = (n_base + tCc[v, 0, j][1]) < n_boundary
    for mi in cutlass.range_constexpr(m_iter):
        if m_base + tCc[0, mi, 0][0] < m_boundary:
            for j in cutlass.range_constexpr(n_iter):
                src = tS[None, mi, j]
                reg = cute.make_fragment_like(src)
                cute.autovec_copy(src, reg)
                cute.copy(st_atom, reg, tD[None, mi, j], pred=tP[None, mi, j])


@cute.jit
def epi_swap_quant_stg(
    thr_st,
    sC,
    gQ_tile,
    gSF,
    sf_ctx,
    cfg,
    tile_mn,
    tok_base,
    tok_boundary,
    tidx,
    threads_m,
    n_store,
):
    bm, bn = tile_mn
    tS = thr_st.partition_S(sC)
    tD = thr_st.partition_D(gQ_tile)
    tCc = thr_st.partition_S(cute.make_identity_tensor((bm, bn)))
    reg = cute.make_fragment_like(tS)
    frg = cute.make_fragment_like(tS, gQ_tile.element_type)
    if tidx < n_store:
        cute.autovec_copy(tS, reg)
        a = cute.math.abs(reg.load().to(cutlass.Float32)).reduce(
            cute.ReductionOp.MAX, init_val=cutlass.Float32(0.0), reduction_profile=0
        )
        a = cute.arch.warp_reduction_max(a, threads_in_group=threads_m)
        amax = cute.math.max(a, cutlass.Float32(cfg.amax_floor))
        sf, sf_val = quant_row_parts(amax, cfg)
        frg.store(
            quant_apply(reg.load().to(cutlass.Float32), sf, cfg).to(
                gQ_tile.element_type
            )
        )
        if tok_base + tCc[0][1] < tok_boundary:
            cute.autovec_copy(frg, tD)
            if tCc[0][0] == 0:
                sf_store(gSF, sf_ctx, tCc[0][1], sf_val, cfg)


@cute.jit
def store_wg_q1_swap(
    acc,
    thr,
    thr_st,
    sC,
    gQ_t,
    gSFQ,
    tile,
    tile_mn,
    sC_dtype,
    qcfg,
    epi_bar_id,
    mma_threads,
    sf_m_align,
    pack_nsf,
    tidx,
    threads_m,
    n_store,
):
    bm, bn = tile_mn
    rmem_to_smem(convert_acc(acc, sC_dtype), thr, sC)
    cute.arch.barrier(barrier_id=epi_bar_id, number_of_threads=mma_threads)
    gQ_tile = cute.local_tile(
        cute.domain_offset((0, tile.m_offset), gQ_t),
        (bm, bn),
        (tile.m_block, tile.n_block),
    )
    sf_col = (
        (tile.m_offset + tile.group * (sf_m_align - 1)) & -sf_m_align
    ) + tile.n_block * bn
    ctx = sf_ctx_of(sf_col, tile.m_block, pack_nsf, qcfg)
    epi_swap_quant_stg(
        thr_st,
        sC,
        gQ_tile,
        gSFQ,
        ctx,
        qcfg,
        (bm, bn),
        tile.m_offset + tile.n_block * bn,
        tile.m_boundary,
        tidx,
        threads_m,
        n_store,
    )
    cute.arch.barrier(barrier_id=epi_bar_id, number_of_threads=mma_threads)


@cute.jit
def store_wg_q1_before_r2s(
    acc,
    thr,
    thr_st,
    sD,
    xchg,
    warp_n,
    num_warp_n,
    st_atom,
    gQ,
    gSFQ,
    tile,
    tile_mn,
    n_boundary,
    sC_dtype,
    qcfg,
    store_empty,
    epi_bar_id,
    mma_threads,
    sf_m_align,
    pack_nsf,
):
    bm, bn = tile_mn
    cute.arch.barrier(barrier_id=epi_bar_id, number_of_threads=mma_threads)
    gQ_tile = cute.local_tile(
        cute.domain_offset((tile.m_offset, 0), gQ),
        (bm, bn),
        (tile.m_block, tile.n_block),
    )
    sf_col = (
        (tile.m_offset + tile.group * (sf_m_align - 1)) & -sf_m_align
    ) + tile.m_block * bm
    ctx = sf_ctx_of(sf_col, tile.n_block, pack_nsf, qcfg)
    epi_quant_s2r_stg(
        st_atom,
        thr,
        thr_st,
        acc,
        sD,
        xchg,
        warp_n,
        num_warp_n,
        gQ_tile,
        gSFQ,
        ctx,
        qcfg,
        (bm, bn),
        tile.m_offset + tile.m_block * bm,
        tile.n_block * bn,
        tile.m_boundary,
        n_boundary,
        epi_bar_id,
        mma_threads,
        sC_dtype,
    )
    cute.arch.barrier(barrier_id=epi_bar_id, number_of_threads=mma_threads)
    cute.arch.mbarrier_arrive(store_empty)
