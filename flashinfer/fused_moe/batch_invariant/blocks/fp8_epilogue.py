# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The group-masked epilogue of the fp8 per-tensor MoE classes: TMEM -> registers, the W8A8 chains, the vector stores.

The bf16 epilogue (:mod:`.epilogue`) with the quantized classes' chains (:class:`Fp8Chain`):

* ``glu`` / ``act`` with the e4m3 output: GEMM1's fused quantization (``spec_fp8.glu_quant_e4m3_elem`` on the (gate, up)
  pair in one thread's fragment, ``act_quant_e4m3_elem`` on the single accumulator): the activation on the alpha-scaled
  fp32 accumulators, the ``mul.rn`` by the intermediate multiplier (``mSi[0]``, read as given), then two outputs per
  ``cvt.rn.satfinite.e4m3x2.f32`` (each value's conversion is independent of its pairing) into byte registers;
* ``plain`` scaled with the bf16 output: GEMM2 (``spec_fp8.scaled_out_elem``, ``alpha2 = mul(s_inter, s_w2[e])``);
* ``plain`` unscaled with the fp32 output: the raw accumulators (the oracle of the device suite).

The per-tile alpha ``mul(mSa[0], mAlpha[e])`` is formed once per tile (``spec_fp8.w8a8_alpha``: the dense fp8 class's
alpha). The stores are the bf16 epilogue's staged, row-coalesced path for the 8- and 16-bit outputs (8-byte chunks for
e4m3; the register vectors of an 8-bit output are 8 bytes, the staging rows being skewed by ``STAGE_PAD`` bytes) and its
direct path for the fp32 oracle output. Either path moves the same converted bytes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass import Boolean, Int32, Uint8
from cutlass.cute.nvgpu import tcgen05
from cutlass.utils.blackwell_helpers import get_tmem_load_op
from cutlass.utils.gemm.sm100 import transform_partitioned_tensor_layout

from ....batch_invariant.fp_interface.exact import f32_bits
from ....batch_invariant.quantization.cvt_fp8 import cvt_f32x2_to_fp8x2
from ....gemm.batch_invariant.blocks.operands import ACC_DTYPE
from ..spec import ACT_PARAMS, ACTS, UNGATED_ACTS
from ..spec_fp8 import act_quant_pre, glu_quant_pre, scaled_out_elem, w8a8_alpha
from .epilogue import (
    BAR_EPI_STAGE,
    GROUP,
    MoeF32,
    fragment_offsets,
    staging_bytes,
    staging_layout,
    store_plan,
)

__all__ = [
    "Fp8Chain",
    "GEMM2_CHAIN",
    "OUT_FMTS_FP8",
    "grouped_epilogue_loop_fp8",
    "staged_store_fp8",
    "staging_bytes",
    "staging_layout",
]

KINDS = ("plain", "glu", "act")
OUT_FMTS_FP8 = ("e4m3", "bf16", "f32")
STAGED_OUT_FMTS_FP8 = ("e4m3", "bf16")


def staged_store_fp8(out_fmt: str) -> bool:
    """Whether a grouped fp8 kernel with this output format stages its stores."""
    return out_fmt in STAGED_OUT_FMTS_FP8


@dataclass(frozen=True)
class Fp8Chain:
    """The epilogue chain of a grouped fp8 GEMM: ``kind`` (``glu`` on the (gate, up) pair, ``act`` on one accumulator, or
    ``plain``), the activation and its binary32 parameters, whether the class's alpha scales the accumulators (``scaled``:
    the W8A8 chains; unscaled ``plain`` / ``f32`` is the raw-accumulator oracle) and the output format. ``describe()``
    names it, parameters included, so every parameter set is its own compiled kernel."""

    kind: str = "plain"
    act: str = "silu"
    out_fmt: str = "bf16"
    params: Tuple[Optional[float], ...] = ()
    scaled: bool = True

    def __post_init__(self):
        assert self.kind in KINDS and self.out_fmt in OUT_FMTS_FP8, self
        assert self.act in (UNGATED_ACTS if self.kind == "act" else ACTS), self
        expected = len(ACT_PARAMS.get(self.act, ())) if self.kind == "glu" else 0
        assert len(self.params) == expected, (self, expected)
        if self.kind in ("glu", "act"):
            assert self.scaled and self.out_fmt == "e4m3", (
                "the activation chains are GEMM1's fused quantization: scaled, e4m3 out"
            )
        else:
            assert self.out_fmt in ("bf16", "f32") and (
                self.scaled or self.out_fmt == "f32"
            ), (
                "GEMM2 converts the scaled accumulator to bf16; the unscaled chain is the fp32 oracle"
            )

    @property
    def pair(self) -> bool:
        return self.kind == "glu"

    def describe(self) -> str:
        suffix = "".join(
            "_pn" if v is None else f"_p{f32_bits(v):08x}" for v in self.params
        )
        scale = "w8a8" if self.scaled else "raw"
        return f"{self.kind}_{self.act}_{scale}_{self.out_fmt}{suffix}"


#: GEMM2 of the W8A8 classes: the scaled accumulator converted to bf16.
GEMM2_CHAIN = Fp8Chain("plain", "silu", "bf16", (), True)


def _chain_elem(chain: Fp8Chain, acc, acc_gate, alpha_e, inv):
    """One output element of the chain in fp32 (before the paired e4m3 conversion) for the e4m3 outputs, converted for the
    others; every op through ``MoeF32``."""
    fp = MoeF32
    if chain.pair:
        return fp.mul(
            glu_quant_pre(fp, acc_gate, acc, alpha_e, chain.act, chain.params), inv
        )
    if chain.kind == "act":
        return fp.mul(act_quant_pre(fp, acc, alpha_e, chain.act, chain.params), inv)
    if chain.scaled:
        return scaled_out_elem(fp, acc, alpha_e, chain.out_fmt)
    return fp.cvt_out(acc, chain.out_fmt)


@cute.jit
def grouped_epilogue_loop_fp8(
    st,
    chain: cutlass.Constexpr,
    tidx: Int32,
    tile_sched,
    tCtAcc_base: cute.Tensor,
    thr_mma,
    mOut: cute.Tensor,
    Tk: Int32,
    N: Int32,
    epi_tile: cute.Tile,
    acc_pipeline,
    mma_m_in_cluster,
    mAlpha: cute.Tensor = None,
    mSa: cute.Tensor = None,
    mSi: cute.Tensor = None,
    epi_warpgroups: cutlass.Constexpr = 1,
    sC: cute.Tensor = None,
    cta_v: Int32 = None,
):
    """``mAlpha``: the per-expert weight dequantization scale ``s_w[e]``; ``mSa``: the A operand's dequantization scale
    (``s_x`` for GEMM1, ``s_inter`` for GEMM2; 1 element); ``mSi``: the intermediate multiplier ``m_inter`` (GEMM1's ``inv``,
    1 element; read as given). ``sC`` / ``cta_v`` as in the bf16 epilogue."""
    pair = chain.pair
    e4m3_out = chain.out_fmt == "e4m3"
    out_dt = mOut.element_type
    wg = tidx // 128
    tidx = tidx % 128
    warp_row0 = Int32(32) * ((tidx // Int32(32)) % Int32(4))
    vec_elems = 16 // (out_dt.width // 8)
    ld = cute.assume(mOut.layout.stride[0], divby=vec_elems)
    tiler_mn = cute.slice_(st.mma_tiler, (None, None, 0))
    if cutlass.const_expr(pair):
        col_shape = (GROUP, 2, N // (2 * GROUP))
        mC = cute.make_tensor(
            mOut.iterator,
            cute.make_layout((Tk, col_shape, 1), stride=(ld, (1, 0, GROUP), 0)),
        )
    else:
        col_shape = N
        mC = cute.make_tensor(
            mOut.iterator, cute.make_layout((Tk, N, 1), stride=(ld, 1, 0))
        )
    idC = cute.make_identity_tensor((Tk, col_shape, 1))
    cC = cute.local_tile(idC, tiler_mn, (None, None, None))
    tCcC = transform_partitioned_tensor_layout(thr_mma.partition_C(cC))
    idF = cute.make_identity_tensor((Tk, N, 1))
    cF = cute.local_tile(idF, tiler_mn, (None, None, None))
    tCcF = transform_partitioned_tensor_layout(thr_mma.partition_C(cF))
    tCtAcc = transform_partitioned_tensor_layout(tCtAcc_base)
    copy_atom_t2r = get_tmem_load_op(
        st.cta_tile_shape_mnk,
        st.c_layout,
        st.c_dtype,
        ACC_DTYPE,
        epi_tile,
        st.use_2cta_instrs,
    )
    tAcc_epi = cute.flat_divide(tCtAcc, epi_tile)
    tiled_copy_t2r = tcgen05.make_tmem_copy(
        copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)]
    )
    thr_t2r = tiled_copy_t2r.get_slice(tidx)
    tTR_tAcc_base = thr_t2r.partition_S(tAcc_epi)
    cC_epi = cute.flat_divide(tCcC, epi_tile)
    cF_epi = cute.flat_divide(tCcF, epi_tile)
    frag = thr_t2r.partition_D(cF_epi)[(None, None, None, 0, 0, 0, 0, 0)]
    offs = fragment_offsets(frag.layout)
    # the register vectors: 16 bytes, or 8 for the e4m3 output (its staging rows are skewed by STAGE_PAD bytes)
    ve, partner, is_out = store_plan(
        offs, pair, 8 if out_dt.width == 8 else 16 // (out_dt.width // 8)
    )
    nv = len(offs)
    rows_in_frag = sorted({m for m, _ in offs})
    tTR_rAcc = cute.make_rmem_tensor(frag.shape, ACC_DTYPE)
    tTR_rC = cute.make_rmem_tensor(frag.shape, out_dt)
    tTR_rC.fill(0)
    atom_r2g = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), out_dt, num_bits_per_copy=ve * out_dt.width
    )
    tiled_copy_r2g = cute.make_tiled_copy_D(atom_r2g, tiled_copy_t2r)
    thr_r2g = tiled_copy_r2g.get_slice(tidx)
    tRG_rC = tiled_copy_r2g.retile(tTR_rC)
    tRG_cC_all = thr_r2g.partition_D(cC_epi)
    nvec = cute.size(tRG_rC.shape[0][1])
    n1, n2 = cute.size(tRG_rC.shape[1]), cute.size(tRG_rC.shape[2])
    staged = sC is not None and cta_v is not None and out_dt.width in (8, 16)
    tn = st.mma_tiler[1]
    cta_m = st.cta_tile_shape_mnk[0]
    epi_m = epi_n = CH = RS = SM = SN = 1
    N_out = N
    if cutlass.const_expr(staged):
        epi_m, epi_n = cute.size(epi_tile[0]), cute.size(epi_tile[1])
        n_lay = epi_tile[1]
        nested_n = cute.rank(n_lay) == 2
        P = cute.size(n_lay.shape[0]) if nested_n else epi_n
        NP = cute.size(n_lay.shape[1]) if nested_n else 1
        S = int(n_lay.stride[1]) if nested_n else 0
        step = 2 * GROUP if pair else GROUP
        CH = P // step
        RS = 128 // CH
        assert (
            P % step == 0
            and 128 % CH == 0
            and epi_m % RS == 0
            and epi_m == cta_m
            and epi_n == P * NP
        ), (
            epi_tile,
            step,
            cta_m,
        )
        N_out = N // 2 if pair else N
        tiled_copy_r2s = cute.make_tiled_copy_D(atom_r2g, tiled_copy_t2r)
        thr_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sC = thr_r2s.partition_D(sC)
        tRS_rC = tiled_copy_r2s.retile(tTR_rC)
        sC_wg = sC[(None, None, wg)]
        atom_s2g = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), out_dt, num_bits_per_copy=GROUP * out_dt.width
        )  # 16 B (bf16), 8 B (e4m3)
        tiled_s2g = cute.make_tiled_copy_tv(
            atom_s2g,
            cute.make_layout((RS, CH), stride=(CH, 1)),
            cute.make_layout((1, GROUP)),
        )
        thr_s2g = tiled_s2g.get_slice(tidx)
        sel = cute.make_layout((epi_m, (GROUP, CH)), stride=(1, (epi_m, step * epi_m)))
        tS_sC = [
            thr_s2g.partition_S(
                cute.composition(cute.local_tile(sC_wg, (epi_m, P), (0, pc)), sel)
            )
            for pc in range(NP)
        ]
        cO = cute.make_identity_tensor((epi_m, GROUP * CH))
        tS_cO = thr_s2g.partition_D(cO)
        SM, SN = cute.size(tS_cO.shape[1]), cute.size(tS_cO.shape[2])
        mOutFlat = cute.make_tensor(
            mOut.iterator, cute.make_layout((Tk, N_out), stride=(ld, 1))
        )
    inv = None
    if cutlass.const_expr(e4m3_out):
        inv = mSi[0]  # the intermediate multiplier, as given
    acc_state = pipeline.make_pipeline_state(
        pipeline.PipelineUserType.Consumer, st.num_acc_stage
    )
    work_tile = tile_sched.initial_work_tile_info()
    while work_tile.is_valid_tile:
        row0, row_end, n_tile, e = (
            work_tile.row0,
            work_tile.row_end,
            work_tile.n_tile,
            work_tile.e,
        )
        tile_sched.advance_to_next_work()
        work_tile = tile_sched.get_current_work()
        rows_left = row_end - row0
        alpha_e = None
        if cutlass.const_expr(chain.scaled):
            alpha_e = w8a8_alpha(
                MoeF32, mSa[0], mAlpha[e]
            )  # the tile's alpha: one mul.rn per tile
        gC = cute.local_tile(
            cute.domain_offset((row0, 0, 0), mC), tiler_mn, (None, None, None)
        )
        tCgC = transform_partitioned_tensor_layout(thr_mma.partition_C(gC))
        gC_epi = cute.flat_divide(tCgC, epi_tile)
        tRG_gC = thr_r2g.partition_D(gC_epi)[
            (None, None, None, None, None, mma_m_in_cluster, n_tile, 0)
        ]
        tRG_cC = tRG_cC_all[(None, None, None, None, None, mma_m_in_cluster, n_tile, 0)]
        tRG_gC = cute.group_modes(tRG_gC, 3, cute.rank(tRG_gC))
        tRG_cC = cute.group_modes(tRG_cC, 3, cute.rank(tRG_cC))
        cta_row_rel = Int32(0)
        if cutlass.const_expr(staged):
            cta_row_rel = mma_m_in_cluster * Int32(st.mma_tiler[0]) + cta_v * Int32(
                cta_m
            )
        tTR_tAcc = tTR_tAcc_base[(None, None, None, None, None, acc_state.index)]
        acc_pipeline.consumer_wait(acc_state)
        tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
        subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
        warp_active = Boolean(True)
        if cutlass.const_expr(
            len(rows_in_frag) == 1 and st.cta_tile_shape_mnk[0] == 128
        ):
            warp_active = rows_left > warp_row0
        for s in range(wg, subtile_cnt, epi_warpgroups):
            if warp_active:
                cute.copy(tiled_copy_t2r, tTR_tAcc[(None, None, None, s)], tTR_rAcc)
                cC_sub = tRG_cC[(None, None, None, s)]
                if cutlass.const_expr(e4m3_out):
                    # two outputs per cvt.rn.satfinite.e4m3x2.f32 (any two output positions: each byte goes to its own register)
                    outs = [v for v in range(nv) if is_out[v]]
                    assert len(outs) % 2 == 0, offs
                    for q in cutlass.range_constexpr(0, len(outs), 2):
                        v0, v1 = outs[q], outs[q + 1]
                        g0 = tTR_rAcc[partner[v0]] if cutlass.const_expr(pair) else None
                        g1 = tTR_rAcc[partner[v1]] if cutlass.const_expr(pair) else None
                        y0 = _chain_elem(chain, tTR_rAcc[v0], g0, alpha_e, inv)
                        y1 = _chain_elem(chain, tTR_rAcc[v1], g1, alpha_e, inv)
                        pr = cvt_f32x2_to_fp8x2(y0, y1, "e4m3")
                        tTR_rC[v0] = (pr & 0xFF).to(Uint8)
                        tTR_rC[v1] = (pr >> 8).to(Uint8)
                else:
                    for v in cutlass.range_constexpr(nv):
                        if cutlass.const_expr(is_out[v]):
                            tTR_rC[v] = _chain_elem(
                                chain, tTR_rAcc[v], None, alpha_e, inv
                            )
                if cutlass.const_expr(staged):
                    cute.copy(tiled_copy_r2s, tRS_rC, tRS_sC[(None, None, None, wg)])
                else:
                    gC_sub = tRG_gC[(None, None, None, s)]
                    for j in cutlass.range_constexpr(nvec):
                        for mi in cutlass.range_constexpr(n1):
                            for ni in cutlass.range_constexpr(n2):
                                if cutlass.const_expr(is_out[j * ve]):
                                    crd = cC_sub[((0, j), mi, ni)]
                                    if cute.elem_less(crd, (rows_left, col_shape, 1)):
                                        cute.copy(
                                            atom_r2g,
                                            tRG_rC[((None, j), mi, ni)],
                                            gC_sub[((None, j), mi, ni)],
                                        )
            if cutlass.const_expr(staged):
                cute.arch.barrier(
                    barrier_id=Int32(BAR_EPI_STAGE) + wg, number_of_threads=128
                )
                for pc in cutlass.range_constexpr(NP):
                    acc_col0 = n_tile * Int32(tn) + s * Int32(P) + Int32(pc * S)
                    out_col0 = (
                        acc_col0 // Int32(2) if cutlass.const_expr(pair) else acc_col0
                    )
                    out_col0 = cute.assume(out_col0, divby=GROUP)
                    gO = cute.local_tile(
                        cute.domain_offset((row0 + cta_row_rel, out_col0), mOutFlat),
                        (epi_m, GROUP * CH),
                        (0, 0),
                    )
                    tS_gO = thr_s2g.partition_D(gO)
                    for i in cutlass.range_constexpr(SM):
                        for j in cutlass.range_constexpr(SN):
                            crd = tS_cO[(0, i, j)]
                            if (cta_row_rel + crd[0] < rows_left) & (
                                out_col0 + crd[1] < N_out
                            ):
                                cute.copy(
                                    atom_s2g,
                                    tS_sC[pc][(None, i, j)],
                                    tS_gO[(None, i, j)],
                                )
                cute.arch.barrier(
                    barrier_id=Int32(BAR_EPI_STAGE) + wg, number_of_threads=128
                )
        with cute.arch.elect_one():
            acc_pipeline.consumer_release(acc_state)
        acc_state.advance()
