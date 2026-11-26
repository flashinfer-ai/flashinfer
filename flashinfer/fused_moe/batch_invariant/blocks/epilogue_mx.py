# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The epilogue block of the MXFP8 classes (:class:`MxChain`) and its fused requantization: the grouped GEMM1 epilogue that
writes the ``mxfp8.quant`` class's codes and block scales of the bf16-rounded chain output instead of the bf16 output itself.

The block names the chain (the bf16 classes' gated / ungated activation or the plain conversion, with its parameters), the
output format (``bf16`` / ``f16`` / ``f32``, or ``mxfp8``: the fused requantization, whose codes and scales are two outputs)
and the instruction class the operands run on (``in_fmt``: ``mxfp8`` on SM100 / SM103, ``mxfp8_k64`` on SM107, the operand
block's key). The non-quantising chains run the shared epilogue loops through ``elem`` (the bf16 chain's term: the UE8M0
block scales are operands of the instruction, so the accumulators are dequantised and the chain has no scale); the
quantising chain runs the loops below (``grouped_loop`` / ``swapped_loop``) with its half-block amax exchange buffer
(``smem_layout``); both unified bodies (:mod:`.kernel`, :mod:`.swap`) call the hooks.

The term is the standalone quantizer's, applied to the same values: the chain's output rounded to bf16 (the class's
intermediate), widened exactly to fp32; the block amax of 32 consecutive output columns by the quantizer's canonical pairwise
``fmaxc`` tree over ``fabs`` (``mx_spec.row_reduction``'s ``elem`` digit: one thread holds the whole block, so the tree is
in-thread); the scale chain (``scale_norm`` -> ``cvt_f32_to_ue8m0`` -> ``inv_scale``) and the element chain (``quant_pre`` ->
``saturate`` -> the declared e4m3 conversion, ``pack_block_codes``) of ``batch_invariant.quantization.mx_spec`` / ``.blocks``. A placement
of the standalone pass: the same bytes on the live rows (asserted bit for bit by the suites), no bf16 intermediate in memory.

The placement: the 128-row TMEM load hands each epilogue thread one row of the tile (thread = TMEM lane = tile row) and the
subtile's accumulator columns; the gated chain's (gate, up) pairs are interleaved by ``GROUP`` so a block of 32 outputs is
the output of ``32 // outs_per_subtile`` consecutive subtiles of the same thread (two 32-column subtiles for the gated chain,
one for the ungated). The subtiles go round-robin to the epilogue warpgroups as in the bf16 epilogue, so every warpgroup
works on the 128-wide tiles too; the two subtiles of a gated block land in a pair of warpgroups, which exchange their
half-block amax through shared memory (one named barrier per block; the canonical tree over 32 leaves is the combine of the
two halves' trees, and the max of non-negative values is exact in every order) and each quantizes its 16 outputs under the
shared scale. Codes: one 16-byte vector per thread per subtile into ``out [Tk, N_out]`` (uint8 e4m3 bytes); the scale: one
byte into the 128x4 swizzled layout of ``(Tk, N_out)`` (``sf_offset_128x4_dev``), stored by the block's first half. Rows
past the group's real end and blocks past ``N_out`` are not stored.

The swapped body (:mod:`.swap`: the weights on the MMA's M side, the group's tokens a narrow N tile) runs
the same term in the transposed orientation (:func:`swapped_epilogue_quant_loop`): thread = weight row = one output
column of every token of the tile, so a block's 32 outputs of one token sit in 32 lanes -- the ungated chain's block is
one warp, the gated chain's (its up lanes are the 8-column halves of two 16-row groups) spans two warps. The block amax
is the same ``fmaxc`` tree over the same ``fabs`` leaves gathered across lanes by ``shfl.bfly`` and, for the gated chain, a
shared-memory exchange between the two warps: the max of non-negative values is exact in every order and ``+0`` is its
identity, so the gate lanes' leaves enter as ``+0`` and the bytes are the standalone quantizer's. Codes: one byte per
(token, column) per lane; the scale byte by the block's first lane.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Optional, Tuple

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass import Boolean, Float32, Int32, Int64, Uint8, Uint32
from cutlass.cute.nvgpu import tcgen05
from cutlass.utils.blackwell_helpers import get_tmem_load_op
from cutlass.utils.gemm.sm100 import transform_partitioned_tensor_layout

from ....batch_invariant.canon import _pairwise
from ....batch_invariant.quantization.cvt_fp8 import cvt_f32x2_to_fp8x2
from ....batch_invariant.quantization.blocks import (
    cvt_f32_to_ue8m0,
    pack_block_codes,
    sf_offset_128x4_dev,
)
from ....batch_invariant.quantization.cute_ext import CuteF32
from ....batch_invariant.quantization.layout import SF_COL_TILE
from ....batch_invariant.quantization.mx_spec import (
    BLOCK,
    DEFAULT_RULE,
    MXFP8,
    elem_digit,
    inv_scale,
    leaf_value,
    quant_pre,
    saturate,
    scale_norm,
)
from ....batch_invariant.fp_interface.exact import f32_bits
from ....gemm.batch_invariant.blocks.operands import ACC_DTYPE
from ..spec import ACT_PARAMS, ACTS, UNGATED_ACTS
from ..spec_mxfp4w8a import MX_CLASSES as MX_MIXED_CLASSES
from ..spec_mxfp8 import MX_CLASSES
from .epilogue import (
    GROUP,
    KINDS,
    GroupedChain,
    GroupedEpilogueBlock,
    MoeF32,
    fragment_offsets,
    select_f32,
    store_plan,
)

OUT_FMT = "mxfp8"  # the out_fmt of the fused requantization: e4m3 codes + UE8M0 block scales of the bf16-rounded chain
MX_OUT_FMTS = ("bf16", "f16", "f32", OUT_FMT)  # the MXFP8 classes' output formats
VEC_BYTES = 16
XCH_BUFFERS = 2  # the half-block amax exchange of the gated chain, double-buffered by block parity (one barrier per block)
BAR_EPI_XCH = 8  # named barrier ids BAR_EPI_XCH + warpgroup pair (the staged store uses BAR_EPI_STAGE + warpgroup, the
# TMEM join BAR_EPI_JOIN: `.epilogue`)


# The accumulator a dead lane or dead token column computes its (never stored) chain on: a normal operand through the chain's
# exact division and the block scale's reciprocal, where a dead row's zero or arbitrary accumulators take the slow sequences.
DEAD_ACC = 1.0
#: The chains whose per-column epilogue cost is the block-amax butterfly rather than the chain (no transcendental, no
#: division): the swapped fused requantization keeps the straight-line form for them, where the columns' butterflies
#: interleave; the other chains skip the dead token columns (the chain dominates).
CHEAP_ACTS = ("relu2", "relu", "identity")


def xch_layout(epi_warpgroups: int):
    """The shared-memory exchange of the gated chain's half-block amax: fp32 per (buffer, epilogue warpgroup, tile row)."""
    return cute.make_layout((XCH_BUFFERS, epi_warpgroups, 128))


def xch_bytes(chain, epi_warpgroups: int) -> int:
    return (
        4 * XCH_BUFFERS * epi_warpgroups * 128
        if chain.out_fmt == OUT_FMT and chain.pair
        else 0
    )


SWAP_XCH_WARPS = 4  # the swapped body's one epilogue warpgroup: the gated chain's block spans warps 2q / 2q + 1


def swap_xch_layout(token_tile: int):
    """The swapped body's half-block amax exchange of the gated chain: fp32 per (buffer, warp, token column of the tile)."""
    return cute.make_layout((XCH_BUFFERS, SWAP_XCH_WARPS, token_tile))


def swap_xch_bytes(chain, token_tile: int) -> int:
    return (
        4 * XCH_BUFFERS * SWAP_XCH_WARPS * token_tile
        if chain.out_fmt == OUT_FMT and chain.pair
        else 0
    )


def bf16_twin(chain) -> GroupedChain:
    """The bf16 chain whose rounded output this epilogue quantizes."""
    assert chain.out_fmt == OUT_FMT, chain
    return GroupedChain(chain.kind, chain.act, "bf16", chain.params)


@dataclass(frozen=True)
class MxChain(GroupedEpilogueBlock):
    """The epilogue block of a grouped GEMM of the MXFP8 classes: ``kind`` (``plain``, the gated ``glu`` on the (gate, up)
    accumulator pair, or the ungated ``act`` on one accumulator), the activation and its binary32 parameters, the output
    format (``bf16`` / ``f16`` / ``f32``: the bf16 chain's conversion through the shared loops; ``mxfp8``: the fused
    requantization of the bf16-rounded chain, the codes and the 128x4 block scales through the loops of this module) and
    the instruction class the operands run on (``in_fmt``: the operand block's key, not part of the chain's name -- the
    compiled object carries it separately; the MXFP8 classes' or the mixed MXFP4-weight x MXFP8-activation classes', whose
    GEMM1 requantises the same way). The accumulators are dequantised by the instruction's block scales, so the chain takes
    no scale operands. ``describe()`` names the chain as the bf16 classes name theirs."""

    kind: str = "plain"
    act: str = "silu"
    out_fmt: str = "bf16"
    params: Tuple[Optional[float], ...] = ()
    in_fmt: str = "mxfp8"

    def __post_init__(self):
        assert self.kind in KINDS and self.out_fmt in MX_OUT_FMTS, self
        assert self.in_fmt in MX_CLASSES or self.in_fmt in MX_MIXED_CLASSES, self
        assert self.act in (UNGATED_ACTS if self.kind == "act" else ACTS), self
        if self.kind == "glu" and self.out_fmt not in ("bf16", OUT_FMT):
            raise AssertionError(
                "the gated chain converts to bf16 once (or the fused requantization quantizes that value in place)"
            )
        if self.out_fmt == OUT_FMT and self.kind not in ("glu", "act"):
            raise AssertionError("the fused requantization quantizes a GEMM1 chain")
        expected = len(ACT_PARAMS.get(self.act, ())) if self.kind == "glu" else 0
        assert len(self.params) == expected, (self, expected)

    @property
    def quant(self) -> bool:
        return self.out_fmt == OUT_FMT

    @property
    def twin(self) -> GroupedChain:
        """The bf16 classes' chain with this chain's term: the one the shared loops convert, or the one the fused
        requantization rounds and quantizes."""
        return GroupedChain(
            self.kind, self.act, "bf16" if self.quant else self.out_fmt, self.params
        )

    def with_class(self, in_fmt: str) -> "MxChain":
        """The same chain on another instruction class."""
        return dataclasses.replace(self, in_fmt=in_fmt)

    def at(self, in_fmt: str) -> "MxChain":
        """The same chain on another instruction class (the block-scaled chains' shared spelling)."""
        return self.with_class(in_fmt)

    @staticmethod
    def of(chain: GroupedEpilogueBlock, in_fmt: str) -> "MxChain":
        """The MX chain at ``in_fmt`` of a bf16 chain object (its kind, activation, output format and parameters) or of an MX
        chain (its class replaced); any other epilogue block is refused."""
        if isinstance(chain, MxChain):
            return chain.with_class(in_fmt)
        assert isinstance(chain, GroupedChain), chain
        return MxChain(chain.kind, chain.act, chain.out_fmt, chain.params, in_fmt)

    def elem(self, fp, acc, acc_gate, alpha_e=None, inv=None):
        """One output element of a non-quantising chain: the bf16 chain's term, converted (the shared loops' path)."""
        assert not self.quant, "the fused requantization runs its own loops"
        return self.twin.elem(fp, acc, acc_gate)

    def describe(self) -> str:
        suffix = "".join(
            "_pn" if v is None else f"_p{f32_bits(v):08x}" for v in self.params
        )
        if self.kind == "act":
            return f"{self.act}_{self.out_fmt}"
        if self.act == "silu":
            return ("swiglu" if self.pair else "plain") + f"_{self.out_fmt}"
        return f"{self.kind}_{self.act}_{self.out_fmt}{suffix}"

    def smem_layout(self, epi_warpgroups: int, token_tile: Optional[int] = None):
        """The half-block amax exchange of the gated fused requantization: per epilogue warpgroup and tile row in the
        grouped body, per warp and token column in the swapped one (``token_tile`` given); None otherwise."""
        if not (self.quant and self.pair):
            return None
        if token_tile is None:
            return xch_layout(epi_warpgroups)
        return swap_xch_layout(token_tile)

    def grouped_loop(self):
        return grouped_epilogue_quant_loop if self.quant else None

    def swapped_loop(self):
        return swapped_epilogue_quant_loop if self.quant else None


def block_amax(fp, ys):
    """The quantizer's block reduction on one thread's values: ``fabs`` leaves, the ``elem`` digit's pairwise ``fmaxc`` tree
    (the canonical order of ``mx_spec.row_reduction``, which every placement of the standalone kernel realises); on a half
    block (16 values) the subtree whose combine with the other half's is the block's tree."""
    d = elem_digit()
    return _pairwise([leaf_value(fp, y) for y in ys], lambda a, b: d.combine(fp, a, b))


@cute.jit
def grouped_epilogue_quant_loop(
    st,
    chain: cutlass.Constexpr,
    tidx: Int32,
    tile_sched,
    tCtAcc_base: cute.Tensor,
    thr_mma,
    mOut: cute.Tensor,
    mSf: cute.Tensor,
    Tk: Int32,
    N: Int32,
    epi_tile: cute.Tile,
    acc_pipeline,
    mma_m_in_cluster,
    epi_warpgroups: cutlass.Constexpr = 1,
    sX: cute.Tensor = None,
    s_off=None,
    s_off_pad=None,
    cta_v: Int32 = None,
):
    """``mOut [Tk, N_out]`` uint8 (the e4m3 codes), ``mSf`` the flat uint8 scale bytes in the 128x4 layout of ``(Tk, N_out)``
    (of the padded rows: with the dense packing, ``s_off`` / ``s_off_pad`` give each group's shift from its dense to its padded
    position and the scale byte of row ``r`` lands at padded row ``r + shift``); ``sX`` the exchange buffers (``xch_layout``, fp32)
    of the gated chain (None for the ungated). 128-row CTA tiles only (thread = tile row); ``cta_v``: this CTA's index inside its
    MMA (0 / 1 for the 2-CTA instructions)."""
    pair = chain.pair
    twin = bf16_twin(chain)
    fp = CuteF32
    rule, fmt = DEFAULT_RULE, MXFP8
    wg = tidx // 128  # the epilogue warpgroup (warp-uniform)
    tidx = tidx % 128
    warp_row0 = Int32(32) * (
        (tidx // Int32(32)) % Int32(4)
    )  # the first tile row of this warp's TMEM lanes
    tiler_mn = cute.slice_(st.mma_tiler, (None, None, 0))
    N_out = N // Int32(2) if cutlass.const_expr(pair) else N
    nb = N_out // Int32(BLOCK)  # scale columns of a row
    pcols = ((nb + Int32(SF_COL_TILE - 1)) // Int32(SF_COL_TILE)) * Int32(SF_COL_TILE)
    # the flat accumulator column space: per-register (row, column) offsets of the fragment, the chain's pairing
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
    tAcc_epi = cute.flat_divide(tCtAcc, epi_tile)  # (EPI_M, EPI_N, SUB_M, SUB_N, STAGE)
    tiled_copy_t2r = tcgen05.make_tmem_copy(
        copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)]
    )
    thr_t2r = tiled_copy_t2r.get_slice(tidx)
    tTR_tAcc_base = thr_t2r.partition_S(
        tAcc_epi
    )  # (T2R, T2R_M, T2R_N, SUB_M, SUB_N, STAGE)
    cF_epi = cute.flat_divide(
        tCcF, epi_tile
    )  # (EPI_M, EPI_N, SUB_M, SUB_N, RestM, RestN, RestL)
    tTR_cF_all = thr_t2r.partition_D(cF_epi)  # the registers' coordinates
    frag = tTR_cF_all[(None, None, None, 0, 0, 0, 0, 0)]
    offs = fragment_offsets(frag.layout)
    _, partner_reg, is_out = store_plan(offs, pair, 16)
    nv = len(offs)
    rows_in_frag = sorted({m for m, _ in offs})
    assert len(rows_in_frag) == 1 and st.cta_tile_shape_mnk[0] == 128, (
        "the fused requantization takes the 128-row layouts",
        offs,
    )
    P = cute.size(epi_tile[1])  # accumulator columns per subtile
    V = P // 2 if pair else P  # this thread's outputs per subtile
    assert BLOCK % V == 0, (BLOCK, V)
    SB = BLOCK // V  # subtiles per output block (2 gated, 1 ungated)
    assert epi_warpgroups % SB == 0, (
        epi_warpgroups,
        SB,
    )  # a block's subtiles land in SB consecutive warpgroups
    assert SB == 1 or sX is not None
    # register -> output column within the subtile (a comprehension: the DSL's preprocessor turns a plain `if` into a region)
    out_col = {
        v: ((n // (2 * GROUP)) * GROUP + n % GROUP if pair else n)
        for v, (m, n) in enumerate(offs)
        if is_out[v]
    }
    assert sorted(out_col.values()) == list(range(V)), out_col
    tTR_rAcc = cute.make_rmem_tensor(frag.shape, ACC_DTYPE)
    ys_r = cute.make_rmem_tensor(
        cute.make_layout((V,)), Float32
    )  # the subtile's bf16-rounded chain values, widened
    fy_words = cute.make_rmem_tensor(
        cute.make_layout((V // 4,)), Uint32
    )  # the subtile's code bytes as words
    fy = cute.make_tensor(
        cute.recast_ptr(fy_words.iterator, dtype=Uint8), cute.make_layout((V,))
    )
    atom16 = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), Uint8, num_bits_per_copy=8 * VEC_BYTES
    )
    tn = st.mma_tiler[1]
    blocks_per_tile = tn // (BLOCK * (2 if pair else 1))
    half = wg % Int32(SB)  # this warpgroup's half of its blocks (0 ungated)
    pair_id = wg // Int32(SB)
    partner = wg ^ Int32(1) if cutlass.const_expr(SB == 2) else wg
    parity = Int32(0)
    d = elem_digit()
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
        sf_shift = Int32(0)
        if cutlass.const_expr(s_off_pad is not None):
            sf_shift = (
                s_off_pad[e] - s_off[e]
            )  # dense A: the group's padded (atom-aligned) position
        tile_sched.advance_to_next_work()
        work_tile = tile_sched.get_current_work()
        rows_left = row_end - row0
        tTR_cF = tTR_cF_all[(None, None, None, None, None, mma_m_in_cluster, n_tile, 0)]
        tTR_cF = cute.group_modes(tTR_cF, 3, cute.rank(tTR_cF))
        tTR_tAcc = tTR_tAcc_base[(None, None, None, None, None, acc_state.index)]
        acc_pipeline.consumer_wait(acc_state)  # the fold of this tile is complete
        tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
        subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
        # a warp whose 32 rows all lie past the group's real end skips its subtiles (warp-uniform); the warp's rows are
        # counted from the cluster tile's first row, as the store predicate is (the peer CTA of a 2-CTA pair and the later
        # CTAs of a cluster along M sit past the first 128 rows: the bf16 epilogue's rule)
        cta_row_rel = Int32(0)
        if cutlass.const_expr(cta_v is not None):
            cta_row_rel = mma_m_in_cluster * Int32(st.mma_tiler[0]) + cta_v * Int32(
                st.cta_tile_shape_mnk[0]
            )
        warp_active = rows_left > cta_row_rel + warp_row0
        for s in range(
            wg, subtile_cnt, epi_warpgroups
        ):  # this warpgroup's subtiles (half blocks when gated)
            blk = n_tile * Int32(blocks_per_tile) + s // Int32(SB)
            crd0 = tTR_cF[(None, None, None, s)][
                0
            ]  # register 0's (row, column) of the subtile
            row_rel = crd0[0] - Int32(
                offs[0][0]
            )  # the thread's row inside the cluster tile
            amax = Float32(0.0)
            live = row_rel < rows_left  # this lane's row is one of the group's
            if warp_active:
                cute.copy(tiled_copy_t2r, tTR_tAcc[(None, None, None, s)], tTR_rAcc)
                # a dead lane (a row past the group's real end, never stored) computes the chain on a benign constant instead
                # of its own accumulators: on a dead row's zero or arbitrary operands the chain's exact division and the block
                # scale's reciprocal take the hardware's slow sequence, and the warp pays it for every lane that does
                for v in cutlass.range_constexpr(nv):
                    tTR_rAcc[v] = select_f32(live, tTR_rAcc[v], Float32(DEAD_ACC))
                # the chain, rounded to bf16 (the class's intermediate), widened exactly: the subtile's outputs in column order
                for v in cutlass.range_constexpr(nv):
                    if cutlass.const_expr(is_out[v]):
                        g = (
                            tTR_rAcc[partner_reg[v]]
                            if cutlass.const_expr(pair)
                            else None
                        )
                        yb = twin.elem(MoeF32, tTR_rAcc[v], g)
                        ys_r[out_col[v]] = yb.to(Float32)
                amax = block_amax(fp, [ys_r[j] for j in range(V)])
                if cutlass.const_expr(SB == 2):
                    sX[(parity, wg, tidx)] = amax
            if cutlass.const_expr(SB == 2):
                cute.arch.barrier(
                    barrier_id=Int32(BAR_EPI_XCH) + pair_id, number_of_threads=128 * SB
                )  # the pair's halves are in
                if warp_active:
                    amax = d.combine(
                        fp, amax, sX[(parity, partner, tidx)]
                    )  # the block's tree: the combine of the two halves'
                parity = Int32(1) - parity
            if warp_active:
                # the quantizer class's term on the block (the scale from the block amax, the codes of this subtile's outputs)
                sf_byte, sf_d = cvt_f32_to_ue8m0(
                    scale_norm(fp, amax, fmt, rule), rule.round
                )
                inv = inv_scale(fp, sf_d, rule)
                yq = [
                    saturate(fp, quant_pre(fp, ys_r[j], inv, rule), fmt, rule)
                    for j in range(V)
                ]
                for w, word in enumerate(pack_block_codes(fp, yq, fmt.elem_fmt)):
                    fy_words[w] = word
                stored = live & (blk * Int32(BLOCK) < N_out)
                if stored:
                    row_g = row0 + row_rel
                    dst = mOut[(row_g, None)]
                    for h in cutlass.range_constexpr(V // VEC_BYTES):
                        cute.copy(
                            atom16,
                            cute.local_tile(fy, (VEC_BYTES,), (h,)),
                            cute.local_tile(
                                dst,
                                (VEC_BYTES,),
                                (
                                    blk * Int32(BLOCK // VEC_BYTES)
                                    + half * Int32(V // VEC_BYTES)
                                    + h,
                                ),
                            ),
                        )
                if stored & (half == Int32(0)):
                    mSf[
                        sf_offset_128x4_dev(
                            (row0 + row_rel + sf_shift).to(Int64), blk, pcols
                        )
                    ] = sf_byte.to(Uint8)
        with cute.arch.elect_one():
            acc_pipeline.consumer_release(
                acc_state
            )  # TMEM buffer free for the next fold
        acc_state.advance()


@cute.jit
def swapped_epilogue_quant_loop(
    st,
    chain: cutlass.Constexpr,
    tidx: Int32,
    tile_sched,
    tCtAcc_base: cute.Tensor,
    thr_mma,
    mOut: cute.Tensor,
    mSf: cute.Tensor,
    Tk: Int32,
    N: Int32,
    epi_tile: cute.Tile,
    acc_pipeline,
    sX: cute.Tensor,
    s_off: cute.Tensor,
    s_off_pad: cute.Tensor,
    weight_tile: cutlass.Constexpr = 128,
    subtile=None,
):
    """The fused requantization in the swapped body's transposed orientation: the accumulator tile is (weight rows,
    tokens); thread ``t`` of an epilogue warpgroup holds weight row ``t`` of the tile and the token columns of its subtile
    (``subtile``: this warpgroup's one subtile of the tile when several warpgroups split the token columns, the accumulator
    released once every warp has arrived; None: one warpgroup takes every subtile). A token column past the group's rows is
    never stored, so its chain and reduction are skipped (a warp-uniform region) in a subtile with dead columns; a subtile
    whose columns are all live, and every subtile of the ReLU family (``CHEAP_ACTS``: the per-column cost is the butterfly,
    which interleaves), runs every column straight-line. Gated
    chain: the gate accumulators come from lane ``+8`` (``shfl.sync.down``), the up lanes (row % 16 < 8) own output
    column ``(row // 16) * 8 + row % 8`` of the tile's ``weight_tile / 2`` outputs, so a block of 32 output columns is
    the 16 up lanes of warp ``2q`` and the 16 of warp ``2q + 1``: the block amax is the in-warp butterfly (xor 1, 2, 4
    over the 8 consecutive up lanes, xor 16 over the two groups; the gate lanes' leaves ``+0``) combined with the partner
    warp's through ``sX`` (:func:`swap_xch_layout`, one named barrier per subtile). Ungated chain: output column =
    weight row, a block = one warp, the butterfly xor 1 .. 16. Then the quantizer class's term per (token, block): the
    scale from the block amax, the code of this lane's output; token ``j`` of the tile is dense row ``row0 + j``, stored
    iff ``j < rows_left`` (and the column exists); the scale byte at the padded position ``row0 + j + (off_pad[e] -
    off[e])`` by the block's first lane. ``mOut [Tk', N_out]`` uint8 codes, ``mSf`` the flat 128x4 bytes of the padded
    rows."""
    pair = chain.pair
    twin = bf16_twin(chain)
    fp = CuteF32
    rule, fmt = DEFAULT_RULE, MXFP8
    d = elem_digit()
    tiler_mn = cute.slice_(st.mma_tiler, (None, None, 0))
    N_out = N // Int32(2) if cutlass.const_expr(pair) else N
    nb = N_out // Int32(BLOCK)
    pcols = ((nb + Int32(SF_COL_TILE - 1)) // Int32(SF_COL_TILE)) * Int32(SF_COL_TILE)
    tCtAcc = transform_partitioned_tensor_layout(tCtAcc_base)
    copy_atom_t2r = get_tmem_load_op(
        st.cta_tile_shape_mnk, st.c_layout, st.c_dtype, ACC_DTYPE, epi_tile, False
    )
    tAcc_epi = cute.flat_divide(tCtAcc, epi_tile)  # (EPI_M, EPI_N, SUB_M, SUB_N, STAGE)
    tiled_copy_t2r = tcgen05.make_tmem_copy(
        copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)]
    )
    thr_t2r = tiled_copy_t2r.get_slice(tidx)
    tTR_tAcc_base = thr_t2r.partition_S(
        tAcc_epi
    )  # (T2R, T2R_M, T2R_N, SUB_M, SUB_N, STAGE)
    idF = cute.make_identity_tensor(
        (st.cta_tile_shape_mnk[0], st.cta_tile_shape_mnk[1], 1)
    )  # the tile's own (weight row, token) coordinates
    cF = cute.local_tile(idF, tiler_mn, (None, None, None))
    tCcF = transform_partitioned_tensor_layout(thr_mma.partition_C(cF))
    cF_epi = cute.flat_divide(tCcF, epi_tile)
    tTR_cF = thr_t2r.partition_D(cF_epi)
    frag = tTR_cF[(None, None, None, 0, 0, 0, 0, 0)]
    offs = fragment_offsets(frag.layout)
    nv = len(offs)
    assert (
        len({m for m, _ in offs}) == 1
        and st.cta_tile_shape_mnk[0] == weight_tile == 128
    ), (
        "the 128-row TMEM layout: one weight row per thread",
        offs,
    )
    assert nv % 2 == 0, offs
    assert (weight_tile // (2 if pair else 1)) % BLOCK == 0, weight_tile
    cols = [
        n - offs[0][1] for _, n in offs
    ]  # the register's token column inside the subtile
    span = max(cols) + 1  # the subtile's token columns
    outs_per_tile = weight_tile // 2 if pair else weight_tile
    blocks_per_tile = outs_per_tile // BLOCK
    tTR_rAcc = cute.make_rmem_tensor(frag.shape, ACC_DTYPE)
    tTR_cF = cute.group_modes(tTR_cF, 3, cute.rank(tTR_cF))
    ys = cute.make_rmem_tensor(
        cute.make_layout((nv,)), Float32
    )  # the bf16-rounded chain values of this lane's column, widened
    am = cute.make_rmem_tensor(
        cute.make_layout((nv,)), Float32
    )  # the block amax per token column
    lane = cute.arch.lane_idx()
    warp = (tidx % Int32(128)) // Int32(32)
    partner = warp ^ Int32(1)
    parity = Int32(0)
    wg_id = Int32(0)
    if cutlass.const_expr(subtile is not None):
        wg_id = subtile  # the named barrier of this warpgroup's exchange
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
        rows_left = cutlass.min(row_end, Tk) - row0  # valid tokens of this tile
        sf_shift = s_off_pad[e] - s_off[e]  # the scale bytes at the padded positions
        tTR_tAcc = tTR_tAcc_base[(None, None, None, None, None, acc_state.index)]
        acc_pipeline.consumer_wait(acc_state)
        tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
        subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
        subtiles = (
            [subtile]
            if cutlass.const_expr(subtile is not None)
            else list(range(subtile_cnt))
        )
        for s in subtiles:
            cute.copy(tiled_copy_t2r, tTR_tAcc[(None, None, None, s)], tTR_rAcc)
            crd0 = tTR_cF[(None, None, None, s)][(0, 0, 0)]
            wrow = crd0[0]  # this thread's weight row in the tile
            col0 = crd0[1] - Int32(offs[0][1])  # the subtile's first token column
            n_glob = n_tile * Int32(weight_tile) + wrow
            if cutlass.const_expr(pair):
                is_up = (wrow % Int32(2 * GROUP)) < Int32(GROUP)
                out_loc = (wrow // Int32(2 * GROUP)) * Int32(GROUP) + wrow % Int32(
                    GROUP
                )
                store_ok = is_up & (n_glob < N)
            else:
                out_loc = wrow
                store_ok = n_glob < N
            out_col = n_tile * Int32(outs_per_tile) + out_loc
            blk = n_tile * Int32(blocks_per_tile) + out_loc // Int32(BLOCK)
            leader = store_ok & (
                (out_loc % Int32(BLOCK)) == Int32(0)
            )  # the block's first column stores its scale byte
            # the chain and the block amax (the in-warp butterfly) of the token columns. The dead-column skip pays where the
            # per-column cost is the chain itself: a subtile with a column past the group's end (never stored) computes the live
            # columns only -- a warp-uniform region: every lane of the warpgroup holds the same column, so the shuffles inside
            # see every lane; a dead column keeps a benign constant for the quantizer's division. A subtile whose columns are
            # all live, and every subtile of the ReLU family (CHEAP_ACTS: the butterflies interleave), runs every column
            # straight-line.
            if cutlass.const_expr(chain.act in CHEAP_ACTS):
                full = Boolean(True)
            else:
                full = (
                    col0 + Int32(span)
                ) <= rows_left  # every token column of the subtile is a live token
            if full:
                for v in cutlass.range_constexpr(nv):
                    g = None
                    if cutlass.const_expr(pair):
                        g = cute.arch.shuffle_sync_down(tTR_rAcc[v], offset=8)
                    yb = twin.elem(MoeF32, tTR_rAcc[v], g)
                    yf = yb.to(Float32)
                    leaf = leaf_value(fp, yf)
                    if cutlass.const_expr(pair):
                        lf = Float32(0.0)
                        if is_up:
                            lf = leaf
                        leaf = lf
                        for m in (1, 2, 4, 16):
                            leaf = d.combine(
                                fp, leaf, cute.arch.shuffle_sync_bfly(leaf, offset=m)
                            )
                    else:
                        for m in (1, 2, 4, 8, 16):
                            leaf = d.combine(
                                fp, leaf, cute.arch.shuffle_sync_bfly(leaf, offset=m)
                            )
                    ys[v] = yf
                    am[v] = leaf
            else:
                for v in cutlass.range_constexpr(nv):
                    yf = Float32(DEAD_ACC)
                    leaf = Float32(DEAD_ACC)
                    if col0 + Int32(cols[v]) < rows_left:
                        g = None
                        if cutlass.const_expr(pair):
                            g = cute.arch.shuffle_sync_down(tTR_rAcc[v], offset=8)
                        yb = twin.elem(MoeF32, tTR_rAcc[v], g)
                        yf = yb.to(Float32)
                        leaf = leaf_value(fp, yf)
                        if cutlass.const_expr(pair):
                            lf = Float32(0.0)
                            if is_up:
                                lf = leaf
                            leaf = lf
                            for m in (1, 2, 4, 16):
                                leaf = d.combine(
                                    fp,
                                    leaf,
                                    cute.arch.shuffle_sync_bfly(leaf, offset=m),
                                )
                        else:
                            for m in (1, 2, 4, 8, 16):
                                leaf = d.combine(
                                    fp,
                                    leaf,
                                    cute.arch.shuffle_sync_bfly(leaf, offset=m),
                                )
                    ys[v] = yf
                    am[v] = leaf
            if cutlass.const_expr(pair):
                # the partner warp's half of the block's amax (one named barrier per warpgroup and subtile)
                if lane == Int32(0):
                    for v in cutlass.range_constexpr(nv):
                        sX[(parity, warp, Int32(cols[v]) + col0)] = am[v]
                cute.arch.barrier(
                    barrier_id=Int32(BAR_EPI_XCH) + wg_id, number_of_threads=128
                )  # the four warps' halves are in
                for v in cutlass.range_constexpr(nv):
                    am[v] = d.combine(
                        fp, am[v], sX[(parity, partner, Int32(cols[v]) + col0)]
                    )
                parity = Int32(1) - parity
            # the quantizer class's term per (token, block): the scale from the block amax, this lane's code; two tokens
            # per conversion, a pair with a live column only
            for q in cutlass.range_constexpr(0, nv, 2):
                v0, v1 = q, q + 1
                j0 = col0 + Int32(cols[v0])
                j1 = col0 + Int32(cols[v1])
                if (j0 < rows_left) | (j1 < rows_left):
                    sf0, sfd0 = cvt_f32_to_ue8m0(
                        scale_norm(fp, am[v0], fmt, rule), rule.round
                    )
                    sf1, sfd1 = cvt_f32_to_ue8m0(
                        scale_norm(fp, am[v1], fmt, rule), rule.round
                    )
                    y0 = saturate(
                        fp,
                        quant_pre(fp, ys[v0], inv_scale(fp, sfd0, rule), rule),
                        fmt,
                        rule,
                    )
                    y1 = saturate(
                        fp,
                        quant_pre(fp, ys[v1], inv_scale(fp, sfd1, rule), rule),
                        fmt,
                        rule,
                    )
                    pr = cvt_f32x2_to_fp8x2(y0, y1, fmt.elem_fmt)
                    ok0 = store_ok & (j0 < rows_left)
                    ok1 = store_ok & (j1 < rows_left)
                    if ok0:
                        mOut[(row0 + j0, out_col)] = (pr & 0xFF).to(Uint8)
                    if ok1:
                        mOut[(row0 + j1, out_col)] = (pr >> 8).to(Uint8)
                    if leader & (j0 < rows_left):
                        mSf[
                            sf_offset_128x4_dev(
                                (row0 + j0 + sf_shift).to(Int64), blk, pcols
                            )
                        ] = sf0.to(Uint8)
                    if leader & (j1 < rows_left):
                        mSf[
                            sf_offset_128x4_dev(
                                (row0 + j1 + sf_shift).to(Int64), blk, pcols
                            )
                        ] = sf1.to(Uint8)
        with cute.arch.elect_one():
            acc_pipeline.consumer_release(
                acc_state
            )  # TMEM buffer free for the next fold
        acc_state.advance()
