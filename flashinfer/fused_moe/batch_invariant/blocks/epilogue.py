# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Group-masked epilogue block: TMEM -> registers, the chain, predicated vector stores.

Chains (:class:`GroupedChain`; the class's ``plain`` / ``swiglu`` / ``relu2`` names map onto it):

* ``plain``: every accumulator element ``cvt.rn`` to the output type (GEMM2, or the fp32 oracle output of either GEMM);
* ``glu``: GEMM1's B rows are gate/up-interleaved in groups of ``GROUP = 8`` (accumulator columns ``16q + r`` hold
  ``u_{8q+r}``, ``16q + 8 + r`` hold ``g_{8q+r}``), so a thread's register fragment holds both halves of every pair it
  owns, for every TMEM load layout the space uses (checked statically below). Each pair becomes the gated-activation
  term (``spec.glu_elem`` through ``CuteF32``) at output column ``8q + r``: the output is addressed through a view of the
  ``(Tk, I)`` buffer in the accumulator's column space, ``(Tk, (8, 2, N / 16)) : (ld, (1, 0, 8))``, in which the gate
  positions alias the up positions and are never stored;
* ``act``: the ungated chain (``spec.act_elem``: ReLU^2 of one accumulator, GEMM1 of the ``relu2`` class) on every
  accumulator element, stored like ``plain``.

Stores: the converted fragment goes registers -> shared memory in the TMEM load's thread layout (its widest vectors,
into a row-padded buffer, ``STAGE_PAD``; one subtile buffer per epilogue warpgroup), then shared -> global by the
warpgroup with consecutive lanes on consecutive 8-element chunks of one row (``make_tiled_copy_tv``: 16-byte chunks,
whole lines per warp instruction), each chunk copied under a plain branch on ``row < row_end`` (the group's real end:
rows of the tile beyond it are padding or belong to other experts and are owned by their tiles) and the column bound;
for glu the chunks are the up positions (accumulator columns ``16q .. 16q + 7`` -> output columns ``8q ..``). The direct
register -> global path (thread = row, one copy per fragment vector under the same branches) remains for the f32 oracle
outputs. Two facts of the lowering shape both paths: a predicated ``cute.copy`` lowers to element-wise stores, and a
dynamic row stride needs ``cute.assume(divby)`` or the vector copies scalarize. Either path moves the same converted
bytes: the chain is computed by the same thread from the same accumulator value in the same fp32 sequence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass import Boolean, Float32, Int32
from cutlass._mlir.dialects import arith
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.cute.nvgpu import tcgen05
from cutlass.utils.blackwell_helpers import get_tmem_load_op
from cutlass.utils.gemm.sm100 import transform_partitioned_tensor_layout

from ....batch_invariant.fp_interface.exact import f32_bits
from ....batch_invariant.quantization.cute_ext import CuteF32, bitcast_i32_to_f32
from ....gemm.batch_invariant.blocks.operands import ACC_DTYPE
from ..spec import ACT_PARAMS, ACTS, UNGATED_ACTS, act_elem, glu_elem

GROUP = 8  # gate/up interleave granularity (columns): u x8, g x8, u x8, ...
KINDS = ("plain", "glu", "act")
OUT_FMTS = ("bf16", "f16", "f32")
STAGED_STORE = True  # the shared-memory staged, row-coalesced store of the 16-bit outputs (False: the direct path everywhere)
BAR_EPI_STAGE = (
    4  # named barrier ids BAR_EPI_STAGE + warpgroup (the dense blocks use 1 and 2)
)
BAR_EPI_JOIN = (
    3  # every epilogue warpgroup once, after its last TMEM load, before the TMEM free
)
STAGED_OUT_FMTS = ("bf16", "f16")
# elements of row padding in the staging buffer: a 16-byte skew per row, so the TMEM lanes' (thread = row) 16-byte stores
# spread over the banks
STAGE_PAD = 8


def staged_store(out_fmt: str) -> bool:
    """Whether a grouped kernel with this output format stages its stores (its kernel then allocates the staging buffers)."""
    return STAGED_STORE and out_fmt in STAGED_OUT_FMTS


def staging_layout(out_dt, epi_tile, warpgroups: int):
    """The shared-memory layout of the staging buffers: one epilogue subtile of the output type per epilogue warpgroup,
    rows padded by ``STAGE_PAD`` elements (plain strides: the registers -> smem stores are 16-byte vectors in the TMEM
    load's thread layout, the smem -> global reads 16-byte chunks along a row)."""
    epi_m, epi_n = cute.size(epi_tile[0]), cute.size(epi_tile[1])
    ld = epi_n + STAGE_PAD
    return cute.make_layout((epi_m, epi_n, warpgroups), stride=(ld, 1, epi_m * ld))


def staging_bytes(out_dt, epi_tile, warpgroups: int) -> int:
    return cute.size_in_bytes(out_dt, staging_layout(out_dt, epi_tile, warpgroups))


@dataclass(frozen=True)
class GroupedChain:
    """The epilogue chain of a grouped GEMM: ``kind`` (``plain``, the gated ``glu`` on an accumulator pair, or the
    ungated ``act`` on one accumulator), the activation, the output format and the activation's parameters (binary32
    values, ``None`` for an absent clamp / scale: ``spec.ACT_PARAMS``). ``describe()`` names it, parameters included
    (their bit patterns), so every parameter set is its own compiled kernel."""

    kind: str = "plain"
    act: str = "silu"
    out_fmt: str = "bf16"
    params: Tuple[Optional[float], ...] = ()

    def __post_init__(self):
        assert self.kind in KINDS and self.out_fmt in OUT_FMTS, self
        assert self.act in (UNGATED_ACTS if self.kind == "act" else ACTS), self
        if self.kind == "glu" and self.out_fmt != "bf16":
            raise AssertionError("the gated chain converts to bf16 once")
        expected = len(ACT_PARAMS.get(self.act, ())) if self.kind == "glu" else 0
        assert len(self.params) == expected, (self, expected)

    @property
    def pair(self) -> bool:
        return self.kind == "glu"

    def describe(self) -> str:
        suffix = "".join(
            "_pn" if v is None else f"_p{f32_bits(v):08x}" for v in self.params
        )
        if self.kind == "act":
            return f"{self.act}_{self.out_fmt}"  # the ungated chains: `relu2_bf16`
        if self.act == "silu":  # the bf16 class's names
            return ("swiglu" if self.pair else "plain") + f"_{self.out_fmt}"
        return f"{self.kind}_{self.act}_{self.out_fmt}{suffix}"

    @staticmethod
    def legacy(chain: str, out_fmt: str = "bf16") -> "GroupedChain":
        """The bf16 class's ``plain`` / ``swiglu`` / ``relu2`` chain names."""
        assert chain in ("plain", "swiglu", "relu2"), chain
        if chain == "relu2":
            return GroupedChain("act", "relu2", out_fmt)
        return GroupedChain(
            "glu" if chain == "swiglu" else "plain",
            "silu",
            "bf16" if chain == "swiglu" else out_fmt,
        )


def fragment_offsets(frag_layout) -> List[Tuple[int, int]]:
    """Static ``(row, column)`` offsets of every register of one thread's fragment, from the partition of a flat
    ``(Tk, N, 1)`` identity tensor (the accumulator's own column space)."""
    flat = cute.flatten(frag_layout)
    out = []
    for v in range(cute.size(frag_layout)):
        crd = cute.crd2idx(v, flat)
        crd = tuple(crd) if isinstance(crd, (tuple, list)) else (crd, 0)
        assert not any(isinstance(x, (tuple, list)) for x in crd), (
            f"flat coordinates expected, got {crd}"
        )
        out.append((int(crd[0]), int(crd[1]) if len(crd) > 1 else 0))
    return out


def store_plan(
    offs: List[Tuple[int, int]], pair: bool, max_vec: int
) -> Tuple[int, List[Optional[int]], List[bool]]:
    """``(ve, partner, is_out)``: the store vector width (registers per vector, all in one row at consecutive columns,
    every vector starting at a multiple of ``ve``), the partner register of every up position (glu) and which registers
    are stored (plain: all; glu: up positions)."""
    nv = len(offs)
    index = {o: v for v, o in enumerate(offs)}
    partner: List[Optional[int]] = [None] * nv
    is_out = [True] * nv
    if pair:
        for v, (m, n) in enumerate(offs):
            if n % (2 * GROUP) < GROUP:
                p = index.get((m, n + GROUP))
                if p is None:
                    raise ValueError(
                        f"gate partner of register {v} at {(m, n)} is not in this thread's fragment: {offs}"
                    )
                partner[v] = p
            else:
                is_out[v] = False
    ve = 1
    for cand in (16, 8, 4, 2):
        if cand > max_vec or nv % cand:
            continue
        ok = True
        for j in range(0, nv, cand):
            m0, n0 = offs[j]
            ok = (
                ok
                and all(offs[j + i] == (m0, n0 + i) for i in range(cand))
                and n0 % cand == 0
            )
            ok = ok and len({is_out[j + i] for i in range(cand)}) == 1
        if ok:
            ve = cand
            break
    return ve, partner, is_out


def bitcast_f32_to_i32(a):
    """``mov.b32 r, f``: reinterpret a Float32 as an Int32 (no conversion)."""
    from cutlass._mlir.dialects import llvm
    from cutlass.cutlass_dsl import T

    res = llvm.inline_asm(
        T.i32(),
        [Float32(a).ir_value()],
        "mov.b32 $0, $1;",
        "=r,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Int32(res)


class MoeF32(CuteF32):
    """``CuteF32`` with the MoE chains' exact mantissa / exponent split of the declared log (``spec.log_term``): bit
    operations on the binary32 pattern, no floating-point rounding."""

    @staticmethod
    def frexpm(a):
        """The significand field of ``|a|`` under the exponent of 1/2: a binary32 in [1/2, 1) for a normal ``a``."""
        bits = bitcast_f32_to_i32(a) & Int32(0x7FFFFFFF)
        return bitcast_i32_to_f32((bits & Int32(0x007FFFFF)) | Int32(0x3F000000))

    @staticmethod
    def frexpe(a):
        """The biased exponent field of ``|a|`` minus 126, as a Float32 (exact)."""
        bits = bitcast_f32_to_i32(a) & Int32(0x7FFFFFFF)
        return (((bits >> 23) & Int32(0xFF)) - Int32(126)).to(Float32)


@dsl_user_op
def select_f32(cond: Boolean, a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    """``a`` if ``cond`` else ``b`` (``arith.select`` on a Boolean: a choice between two values, no arithmetic)."""
    return Float32(
        arith.select(
            Boolean(cond).ir_value(loc=loc, ip=ip),
            Float32(a).ir_value(loc=loc, ip=ip),
            Float32(b).ir_value(loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )
    )


def _chain_elem(chain: GroupedChain, acc, acc_gate):
    """One output element of the chain, converted: the gated term of the (gate, up) accumulator pair, the ungated
    activation of one accumulator, or the plain conversion; every op in fp32 through ``MoeF32`` (the chain's
    parameters are compile-time constants of the kernel)."""
    fp = MoeF32
    if chain.pair:
        return glu_elem(fp, acc_gate, acc, chain.act, chain.out_fmt, chain.params)
    if chain.kind == "act":
        return act_elem(fp, acc, chain.act, chain.out_fmt, chain.params)
    return fp.cvt_out(acc, chain.out_fmt)


@cute.jit
def grouped_epilogue_loop(
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
    epi_warpgroups: cutlass.Constexpr = 1,
    sC: cute.Tensor = None,
    cta_v: Int32 = None,
):
    """``sC``: the staging buffers (``staging_layout(out, epi_tile, epi_warpgroups)``) of the staged store, or None;
    ``cta_v``: this CTA's index inside its MMA (0, or 0 / 1 for the 2-CTA instructions), which with ``mma_m_in_cluster``
    locates its rows in the cluster tile."""
    pair = chain.pair
    out_dt = mOut.element_type
    wg = tidx // 128  # the epilogue warpgroup (warp-uniform)
    tidx = tidx % 128  # the thread's lane slice inside it
    warp_row0 = Int32(32) * (
        (tidx // Int32(32)) % Int32(4)
    )  # the first tile row of this warp's TMEM lanes (128-row layouts)
    vec_elems = 16 // (out_dt.width // 8)  # elements per 16-byte vector
    ld = cute.assume(
        mOut.layout.stride[0], divby=vec_elems
    )  # 16-byte rows (the API asserts it): the copies vectorize
    tiler_mn = cute.slice_(st.mma_tiler, (None, None, 0))
    # the output and its coordinates in the accumulator's column space
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
    idC = cute.make_identity_tensor(
        (Tk, col_shape, 1)
    )  # store predicate (nested for glu)
    cC = cute.local_tile(idC, tiler_mn, (None, None, None))
    tCcC = transform_partitioned_tensor_layout(thr_mma.partition_C(cC))
    idF = cute.make_identity_tensor((Tk, N, 1))  # flat column space: register offsets
    cF = cute.local_tile(idF, tiler_mn, (None, None, None))
    tCcF = transform_partitioned_tensor_layout(thr_mma.partition_C(cF))
    tCtAcc = transform_partitioned_tensor_layout(tCtAcc_base)
    # TMEM -> registers
    copy_atom_t2r = get_tmem_load_op(
        st.cta_tile_shape_mnk,
        st.c_layout,
        st.c_dtype,
        ACC_DTYPE,
        epi_tile,
        st.use_2cta_instrs,
    )
    tAcc_epi = cute.flat_divide(tCtAcc, epi_tile)  # (EPI_M, EPI_N, SUB_M, SUB_N, STAGE)
    tiled_copy_t2r = tcgen05.make_tmem_copy(
        copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)]
    )
    thr_t2r = tiled_copy_t2r.get_slice(tidx)
    tTR_tAcc_base = thr_t2r.partition_S(
        tAcc_epi
    )  # (T2R, T2R_M, T2R_N, SUB_M, SUB_N, STAGE)
    cC_epi = cute.flat_divide(
        tCcC, epi_tile
    )  # (EPI_M, EPI_N, SUB_M, SUB_N, RestM, RestN, RestL)
    cF_epi = cute.flat_divide(tCcF, epi_tile)
    frag = thr_t2r.partition_D(cF_epi)[(None, None, None, 0, 0, 0, 0, 0)]
    offs = fragment_offsets(frag.layout)
    ve, partner, is_out = store_plan(offs, pair, 16 // (out_dt.width // 8))
    nv = len(offs)
    rows_in_frag = sorted(
        {m for m, _ in offs}
    )  # distinct row offsets of the fragment (1 for the 128-row layouts)
    tTR_rAcc = cute.make_rmem_tensor(frag.shape, ACC_DTYPE)
    tTR_rC = cute.make_rmem_tensor(frag.shape, out_dt)
    tTR_rC.fill(0)
    # registers -> global, in the TMEM load's thread layout
    atom_r2g = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), out_dt, num_bits_per_copy=ve * out_dt.width
    )
    tiled_copy_r2g = cute.make_tiled_copy_D(atom_r2g, tiled_copy_t2r)
    thr_r2g = tiled_copy_r2g.get_slice(tidx)
    tRG_rC = tiled_copy_r2g.retile(tTR_rC)  # ((ve, NVEC), T2R_M, T2R_N)
    tRG_cC_all = thr_r2g.partition_D(
        cC_epi
    )  # ((ve, NVEC), T2R_M, T2R_N, SUB_M, SUB_N, RestM, RestN, RestL)
    nvec = cute.size(tRG_rC.shape[0][1])
    n1, n2 = cute.size(tRG_rC.shape[1]), cute.size(tRG_rC.shape[2])
    # the staged, row-coalesced store (16-bit outputs, 128-row layouts: thread = TMEM lane = tile row)
    staged = sC is not None and cta_v is not None and out_dt.width == 16
    tn = st.mma_tiler[1]
    cta_m = st.cta_tile_shape_mnk[0]
    epi_m = epi_n = CH = RS = SM = SN = (
        1  # (stable Python types on both paths of the static branch)
    )
    N_out = N
    if cutlass.const_expr(staged):
        epi_m, epi_n = cute.size(epi_tile[0]), cute.size(epi_tile[1])
        # the epilogue tile's column mode: NP contiguous pieces of P accumulator columns at a stride of S (the library's tile
        # for the 64-row 2-CTA layouts is ((32, 2) : (1, 64)): two 32-column pieces 64 apart; the subtiles then step by P
        # columns)
        n_lay = epi_tile[1]
        nested_n = (
            cute.rank(n_lay) == 2
        )  # (Python-level selections: the DSL turns `if` into regions)
        P = cute.size(n_lay.shape[0]) if nested_n else epi_n
        NP = cute.size(n_lay.shape[1]) if nested_n else 1
        S = int(n_lay.stride[1]) if nested_n else 0
        step = (
            2 * GROUP if pair else GROUP
        )  # accumulator columns per output chunk (GROUP elements)
        CH = P // step  # output chunks per row of a piece
        RS = 128 // CH  # rows per copy step: CH consecutive lanes share a row
        assert (
            P % step == 0
            and 128 % CH == 0
            and epi_m % RS == 0
            and epi_m == cta_m
            and epi_n == P * NP
        ), (epi_tile, step, cta_m)
        N_out = N // 2 if pair else N
        tiled_copy_r2s = cute.make_tiled_copy_D(
            atom_r2g, tiled_copy_t2r
        )  # the fragment's widest vectors, the TMEM load's thread layout
        thr_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sC = thr_r2s.partition_D(
            sC
        )  # (R2S, R2S_M, R2S_N, WG): the subtile in its local columns
        tRS_rC = tiled_copy_r2s.retile(tTR_rC)
        sC_wg = sC[
            (None, None, wg)
        ]  # this warpgroup's buffer (EPI_M, EPI_N local columns)
        atom_s2g = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), out_dt, num_bits_per_copy=GROUP * out_dt.width
        )  # 16 B
        tiled_s2g = cute.make_tiled_copy_tv(
            atom_s2g,
            cute.make_layout((RS, CH), stride=(CH, 1)),
            cute.make_layout((1, GROUP)),
        )
        thr_s2g = tiled_s2g.get_slice(tidx)
        sel = cute.make_layout(
            (epi_m, (GROUP, CH)), stride=(1, (epi_m, step * epi_m))
        )  # (row, (element, chunk)) -> the output positions of a piece
        tS_sC = [
            thr_s2g.partition_S(
                cute.composition(cute.local_tile(sC_wg, (epi_m, P), (0, pc)), sel)
            )
            for pc in range(NP)
        ]  # (V, SM, SN) per piece
        cO = cute.make_identity_tensor((epi_m, GROUP * CH))
        tS_cO = thr_s2g.partition_D(cO)
        SM, SN = cute.size(tS_cO.shape[1]), cute.size(tS_cO.shape[2])
        mOutFlat = cute.make_tensor(
            mOut.iterator, cute.make_layout((Tk, N_out), stride=(ld, 1))
        )
    acc_state = pipeline.make_pipeline_state(
        pipeline.PipelineUserType.Consumer, st.num_acc_stage
    )
    work_tile = tile_sched.initial_work_tile_info()
    while work_tile.is_valid_tile:
        row0, row_end, n_tile = work_tile.row0, work_tile.row_end, work_tile.n_tile
        tile_sched.advance_to_next_work()
        work_tile = tile_sched.get_current_work()
        rows_left = row_end - row0
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
        if cutlass.const_expr(
            cta_v is not None
        ):  # this CTA's first row inside the cluster tile
            cta_row_rel = mma_m_in_cluster * Int32(st.mma_tiler[0]) + cta_v * Int32(
                cta_m
            )
        tTR_tAcc = tTR_tAcc_base[(None, None, None, None, None, acc_state.index)]
        acc_pipeline.consumer_wait(acc_state)  # the fold of this tile is complete
        tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
        subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
        # a warp whose 32 rows all lie past the group's real end (a partial tile: the padded / stale rows) has nothing to
        # store, so it skips its subtiles (warp-uniform: tcgen05.ld is a warp instruction); 128-row CTA tiles only; the
        # warp's rows are counted from the cluster tile's first row, as the store predicates are (the peer CTA of a pair and
        # the later CTAs of a cluster along M sit past the first 128 rows)
        warp_active = Boolean(True)
        if cutlass.const_expr(
            len(rows_in_frag) == 1 and st.cta_tile_shape_mnk[0] == 128
        ):  # not the 64-row CTA tiles
            warp_active = rows_left > cta_row_rel + warp_row0
        for s in range(wg, subtile_cnt, epi_warpgroups):  # this warpgroup's subtiles
            if warp_active:
                cute.copy(tiled_copy_t2r, tTR_tAcc[(None, None, None, s)], tTR_rAcc)
                cC_sub = tRG_cC[(None, None, None, s)]
                crd0 = cC_sub[((0, 0), 0, 0)]
                # a dead row's accumulators (past the group's end) are replaced by a benign constant right after the TMEM
                # load (a choice, no arithmetic): its lanes would otherwise run the division's slow path on the padded rows'
                # zero or garbage operands, and its values are never stored, so the chain runs unconditionally on every lane
                # (a per-lane region instead costs the compiler the interleaving of the chains)
                live = {
                    m: (crd0[0] - Int32(offs[0][0]) + Int32(m)) < rows_left
                    for m in rows_in_frag
                }
                for v in cutlass.range_constexpr(nv):
                    tTR_rAcc[v] = select_f32(
                        live[offs[v][0]], tTR_rAcc[v], Float32(1.0)
                    )
                # the chain, in fp32, per element (pair)
                for v in cutlass.range_constexpr(nv):
                    if cutlass.const_expr(is_out[v]):
                        tTR_rC[v] = _chain_elem(
                            chain,
                            tTR_rAcc[v],
                            tTR_rAcc[partner[v]] if cutlass.const_expr(pair) else None,
                        )
                if cutlass.const_expr(staged):
                    # registers -> this warpgroup's staging buffer (the TMEM load's thread layout); the rows of a skipped warp
                    # (all past the group's end) stay stale and are predicated off below
                    cute.copy(tiled_copy_r2s, tRS_rC, tRS_sC[(None, None, None, wg)])
                else:
                    # one unpredicated vector copy per (group end, tensor bounds, up position) check: a predicated ``cute.copy``
                    # lowers to element-wise stores
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
                # the buffer -> global by the whole warpgroup (the barriers need all 128 threads, whatever their warps' rows),
                # row by row: CH consecutive lanes write the CH 16-byte chunks of one output row (whole lines per warp
                # instruction)
                cute.arch.barrier(
                    barrier_id=Int32(BAR_EPI_STAGE) + wg, number_of_threads=128
                )
                for pc in cutlass.range_constexpr(
                    NP
                ):  # the pieces of the subtile's columns
                    acc_col0 = n_tile * Int32(tn) + s * Int32(P) + Int32(pc * S)
                    out_col0 = (
                        acc_col0 // Int32(2) if cutlass.const_expr(pair) else acc_col0
                    )
                    out_col0 = cute.assume(
                        out_col0, divby=GROUP
                    )  # a multiple of the chunk: the global chunk copies vectorize
                    gO = cute.local_tile(
                        cute.domain_offset((row0 + cta_row_rel, out_col0), mOutFlat),
                        (epi_m, GROUP * CH),
                        (0, 0),
                    )
                    tS_gO = thr_s2g.partition_D(gO)
                    # one unpredicated vector copy per chunk under a plain branch: a predicated ``cute.copy`` lowers to
                    # element-wise 16-bit stores
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
                )  # the buffer is free again
        with cute.arch.elect_one():
            acc_pipeline.consumer_release(
                acc_state
            )  # TMEM buffer free for the next fold
        acc_state.advance()
