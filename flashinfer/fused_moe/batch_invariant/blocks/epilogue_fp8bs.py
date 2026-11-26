# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The epilogue block of the block-scale fp8 MoE classes (:class:`Fp8BsChain`) and the loops of their promoted operand block
(:class:`.operands.PromotedOperands`): the unified grouped and swapped bodies (:mod:`.kernel`, :mod:`.swap`) run the classes
``moe.e4m3bs.<act>`` (128-wide blocks) and ``moe.e4m3bs32.<act>`` (32-wide) as the dense block-scale body
(:mod:`flashinfer.gemm.batch_invariant.blocks.blockscale_fp8`, class ``gemm.e4m3.kblock128`` / ``kblock32``) under the MoE
group scheduler. Every expanded row's output element is ``fold_kb fma(mul(s_a[row, kb], s_b[blk(n), kb]), P_kb, acc)`` -- the
dense fp8 ``mma32`` fold of each K block from +0, promoted in block order by the two promotion warpgroups -- whatever expert
group, tile, cluster or grid computes it.

Operands: A ``(Tk, K, 1)`` K-major e4m3 (the permuted activations), SFA ``(Tk, K / block)`` fp32 with any strides (the API hands
the MN-major layout: a block's row scales contiguous), B ``(N, K, E)`` K-major e4m3 (the stacked expert weights), SFB ``(E, N /
block, K / block)`` fp32 (the per-expert block scales), ``counts (E,)`` int32 (the dense packing: group ``e`` holds rows ``[off_e,
off_e + counts[e])``), out ``(Tk, Nout)`` row-major. Three outputs, one promotion: the bf16 / f32 chain output (``Nout = N / 2``
for the gated chain, ``N`` otherwise; the group-masked staged store of the bf16 class, the direct store for the f32 oracle
output), or the fused requantization (``out_fmt = "e4m3bs"``): the chain's bf16-rounded value, widened, quantized per
128-column block by the ``fp8bs.quant`` class (:mod:`..spec_fp8bs`: the block amax tree, ``s = fmaxc(div(amax, 448), 1e-12)``,
``inv = div(1, s)``, ``cvt.rn.satfinite.e4m3(mul(y, inv))``) with the e4m3 codes to ``out`` (uint8) and the fp32 scales to
``sfo (Nout / 128, Tk)`` (the MN layout GEMM2's ring reads). A block of 128 outputs lies inside one tile (the fused chain's
acceptance: the tile's outputs are whole blocks); its two 64-output halves land in the two promotion warpgroups, which exchange
their half-block amax through shared memory (the chain's ``smem_layout``; one named barrier per tile between the two warps holding
the same rows; the rounded outputs are kept as packed bf16 pairs and the amax is folded as they form: max is exact in every order, so
the fold is the canonical tree's value and the block's is the combine of the two halves'), or one warpgroup holds a whole block (the
ungated chain on the 256-wide tile).

The 32-wide class (``block = 32``, the dense core ``gemm.e4m3.kblock32``) runs on the same bodies with the block as the operand
block's parameter: a 128-element smem stage holds ``spb = 4`` blocks, the MMA warp commits every instruction step to its own
accumulator stage (:func:`grouped_blockscale_mma_loop`), the scale ring stage carries the stage's four block scales per row and
per weight slot (a tile spans ``tile_n / 32`` weight scale blocks: one scale product per 32 columns per thread and block), and
the promotion drains the four partials in block order -- four times the 128-wide class's TMEM reads and fmas per K, the recipe's
cost. The fused requantization is the 128-wide class's alone (the 32-wide class requantizes by the standalone pass).

The gated chain's weight is gate / up interleaved in groups of 8 rows (:func:`..api.interleave_fc1`) AFTER its quantization:
the scale of an interleaved column is its original row's block, so a tile holds one up block and one gate block (``tile_n <=
256``, ``I % 128 == 0``), the scale ring stage carries both, and the promotion multiplies each register by the block its column
belongs to (a static choice per register: the fold is the spec's per element). The weight interleave is the chain's
``interleaved`` (the f32 oracle output of GEMM1 reads the interleaved weight with the plain chain). The swapped body computes
``C^T`` per expert (the expert's weight rows on the MMA's M side in a 64- or 128-row tile, the group's tokens an 8- to 64-wide N
tile of the dense packing): the fp8 step is symmetric in its operands, so every output element's block partial is the same
``P_kb`` and the promotion runs in block order from +0 in one epilogue thread's registers (:func:`swap_bs_epilogue_loop`); the
fused requantization is not a spelling of that body. The scale ring, the accumulator ring, the per-block promotion and the
``fma.rn.f32x2`` packing are the dense body's; the group scheduler, the group tables and the row predicates are the grouped
bodies'. Every placement here moves no arithmetic.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Optional, Tuple

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass import Float32, Int32, Uint32
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.utils.blackwell_helpers import get_tmem_load_op
from cutlass.utils.gemm.sm100 import transform_partitioned_tensor_layout

from ....batch_invariant.fp_interface.exact import f32_bits
from ....batch_invariant.quantization.blocks import pack_block_codes
from ....batch_invariant.quantization.cute_ext import CuteF32, bitcast_i32_to_f32
from ....gemm.batch_invariant.blocks.blockscale_fp8 import (
    NUM_PROMO_GROUPS,
    NUM_PROMO_WARPS,
    NUM_SCALE_STAGES,
    PROMO_BATCH,
    _elem,
    _vec4,
)
from ....gemm.batch_invariant.blocks.mainloop import mma_fold
from ....gemm.batch_invariant.blocks.operands import ACC_DTYPE
from ....gemm.batch_invariant.generator import Knobs
from ....gemm.batch_invariant.spec import BLOCK_K
from .. import pdl
from ..spec import (
    ACT_PARAMS,
    ACTS,
    UNGATED_ACTS,
    act_elem,
    glu_elem,
    glu_pre,
    params_key,
    ungated_term,
)
from ..spec_fp8bs import BLOCK as QBLOCK
from ..spec_fp8bs import block_chain, check_block, elem_digit, leaf_value, quant_pre
from .epilogue import (
    BAR_EPI_STAGE,
    GROUP,
    GroupedEpilogueBlock,
    MoeF32,
    fragment_offsets,
    store_plan,
)
from .scheduler import MAX_E

#: The operand format of the block-scale fp8 classes (the dense block-scale GEMM class's key).
IN_FMT_BS = "e4m3_bs"
#: The chain out_fmt of the fused requantization: e4m3 codes + fp32 block scales of the bf16-rounded chain.
OUT_FMT_QUANT = "e4m3bs"
OUT_FMTS = ("bf16", "f32", OUT_FMT_QUANT)
KINDS = ("plain", "glu", "act")
#: The grouped space's N tiles: two promotion warpgroups each own a contiguous column half (the narrow decode tiles have one subtile).
TILES_N_GROUPED = (64, 128, 256)
XCH_BUFFERS = 2  # the half-block amax exchange, double-buffered by tile parity (one barrier per tile)
BAR_PROMO_JOIN = 9  # both promotion warpgroups once, after their last TMEM loads, before the TMEM free
BAR_XCH = 10  # named barrier ids BAR_XCH + warp: the amax exchange of the two promotion warps holding the same rows
CODE_STAGE_BYTES = 1024  # per promotion warp: one round of rows of the fused requantization's staged code stores
VEC_BYTES = 16
#: The scale ring's depth in smem stages per class: the 128-wide class's (one block per stage) and the 32-wide class's (four
#: blocks per stage: a stage holds four times the scales, so fewer stages hold as many blocks ahead).
NUM_SCALE_STAGES_OF = {128: NUM_SCALE_STAGES, 32: NUM_SCALE_STAGES // 2}


def blocks_per_stage(block: int) -> int:
    """Scale blocks per 128-element smem stage: 1 for the 128-wide class, 4 for the 32-wide one."""
    assert BLOCK_K % check_block(block) == 0, block
    return BLOCK_K // block


def n_scale_slots(cols: int, block: int, interleaved: bool) -> int:
    """The weight scale blocks a span of ``cols`` interleaved-or-plain weight columns reads: plain -- the span's ``block``-column
    blocks (at least one); interleaved -- its up rows' blocks and its gate rows' blocks (the span's ``cols / 2`` up rows: at least
    one block each half). The ring stage holds one slot per block (slot ``t``: :func:`slot_blocks`); the two halves of the
    interleaved weight occupy the slots' first and second half."""
    return 2 * max(1, (cols // 2) // block) if interleaved else max(1, cols // block)


def slot_blocks(n0, NB, NS: int, block: int, interleaved: bool):
    """The weight scale block (clamped to the tensor: a tile past N reads a finite scale and multiplies a zero-filled partial) of
    every slot of a tile whose first interleaved-or-plain column is ``n0``: plain -- ``n0 / block + t``; interleaved -- the up
    blocks of the tile's up rows ``(n0 / 2) / block + t`` for the first half of the slots, the gate blocks ``NB / 2 + (n0 / 2) /
    block + t`` for the second."""
    out = []
    for t in range(NS):
        if interleaved:
            half, tt = NS // 2, t % (NS // 2)
            base = (n0 // Int32(2)) // Int32(block) + Int32(tt)
            if t >= half:
                base = base + NB // Int32(2)
        else:
            base = n0 // Int32(block) + Int32(t)
        out.append(cutlass.min(base, NB - Int32(1)))
    return out


def group_slot_table(
    per_group: int, V: int, P: int, cols_g: int, NS: int, block: int, interleaved: bool
):
    """The weight scale slot of every register of a promotion group (static per register; the group's slot base is a runtime
    offset that enters addresses only): ``(n_slots_g, g_mult, slot_of[k][e])`` -- the group's slots per ring stage, the group
    index's multiplier and the local slot of register ``e`` of the group's subtile ``k`` (``P`` accumulator columns per subtile).
    Plain: the register's ``block``-column block of the group's ``cols_g`` columns (one slot where they lie in one block); gated:
    the block of the register's original up row (the up columns 16q .. 16q + 7 of the tile) in the slots' first half, its gate
    column's block in the second (:func:`n_scale_slots`, :func:`slot_blocks`)."""
    slot_of = []
    if interleaved:
        half_cols = cols_g // 2  # the group's up (= gate) rows
        n_slots_g = 2 * max(1, half_cols // block)
        g_mult = (n_slots_g // 2) if half_cols >= block else 0
        for k in range(per_group):
            row = []
            for e in range(V):
                u = (
                    k * (P // 2) + (e // (2 * GROUP)) * GROUP + e % GROUP
                )  # the register's original row within the group's half
                t = (u // block) if half_cols >= block else 0
                if (e % (2 * GROUP)) >= GROUP:
                    t += n_slots_g // 2  # a gate column: the gate half's slots
                row.append(t)
            slot_of.append(row)
        assert g_mult * (NUM_PROMO_GROUPS - 1) + n_slots_g // 2 <= NS // 2, (
            g_mult,
            n_slots_g,
            NS,
        )
    else:
        n_slots_g = max(1, cols_g // block)
        g_mult = n_slots_g if cols_g >= block else 0
        for k in range(per_group):
            slot_of.append(
                [((k * P + e) // block) if cols_g >= block else 0 for e in range(V)]
            )
        assert g_mult * (NUM_PROMO_GROUPS - 1) + n_slots_g <= NS, (
            g_mult,
            n_slots_g,
            NS,
        )
    assert all(
        slot_of[k][e] == slot_of[k][e + 1]
        for k in range(per_group)
        for e in range(0, V, 2)
    ), "a packed pair straddles two weight blocks"
    return n_slots_g, g_mult, slot_of


def group_slot_indices(gsel, g_mult: int, n_slots_g: int, NS: int, interleaved: bool):
    """The ring-stage slot of every local slot of promotion group ``gsel`` (a runtime index): the group's base ``gsel * g_mult``
    into the up (or plain) slots, then into the gate slots at ``NS / 2``."""
    out = []
    for t in range(n_slots_g):
        if interleaved and t >= n_slots_g // 2:
            out.append(
                Int32(NS // 2) + gsel * Int32(g_mult) + Int32(t - n_slots_g // 2)
            )
        else:
            out.append(gsel * Int32(g_mult) + Int32(t))
    return out


def grouped_scale_ring_bytes(knobs: Knobs, block: int, interleaved: bool) -> int:
    """Smem of the scale ring: per stage the CTA tile's row scales and the tile's weight slot scales, ``blocks_per_stage`` of
    each, fp32."""
    spb = blocks_per_stage(block)
    return (
        NUM_SCALE_STAGES_OF[block]
        * (knobs.cta_tile_mn[0] + n_scale_slots(knobs.tile_n, block, interleaved))
        * spb
        * (ACC_DTYPE.width // 8)
    )


@dataclass(frozen=True)
class Fp8BsChain(GroupedEpilogueBlock):
    """The epilogue block of a grouped block-scale GEMM: ``kind`` (``plain``, the gated ``glu`` on an accumulator pair, or the
    ungated ``act`` on one accumulator), the activation, the output format (``bf16``: the chain's bf16 output; ``f32``: the
    promoted fold as it stands, the oracle output; ``e4m3bs``: the fused requantization of the bf16-rounded chain, the codes
    and the block scales its two outputs), the activation's parameters (binary32 values, ``None`` for an absent clamp /
    scale: ``spec.ACT_PARAMS``), the scale block size (``block``: the class, 128 or 32) and the weight's row interleave
    (``interleaved``: the gated chain's weight is gate / up interleaved; ``None`` -- the chain's own pairing). The block
    scales are promoted into the fold by the body, so the chain takes no scale operands; the operand block
    (:class:`.operands.PromotedOperands`) reads the block and the interleave off the chain. ``describe()`` names it, parameters
    included (their bit patterns), so every parameter set is its own compiled kernel."""

    kind: str = "plain"
    act: str = "silu"
    out_fmt: str = "bf16"
    params: Tuple[Optional[float], ...] = ()
    block: int = QBLOCK
    interleaved: Optional[bool] = None

    def __post_init__(self):
        assert self.kind in KINDS and self.out_fmt in OUT_FMTS, self
        assert self.act in (UNGATED_ACTS if self.kind == "act" else ACTS), self
        if self.kind == "plain" and self.out_fmt == OUT_FMT_QUANT:
            raise AssertionError("the plain chain has no requantization")
        if self.kind in ("glu", "act") and self.out_fmt == "f32":
            raise AssertionError("the activation chains round to bf16 once")
        expected = len(ACT_PARAMS.get(self.act, ())) if self.kind == "glu" else 0
        assert len(self.params) == expected, (self, expected)
        check_block(self.block)
        if self.out_fmt == OUT_FMT_QUANT and self.block != QBLOCK:
            raise AssertionError("the fused requantization is the 128-wide class's")
        if self.pair and not self.il:
            raise AssertionError(
                "the gated chain pairs the interleaved weight's columns"
            )

    @property
    def in_fmt(self) -> str:
        return IN_FMT_BS

    @property
    def quant(self) -> bool:
        return self.out_fmt == OUT_FMT_QUANT

    @property
    def il(self) -> bool:
        """The weight's gate / up row interleave the chain reads (the chain's pairing unless given)."""
        return self.pair if self.interleaved is None else bool(self.interleaved)

    @property
    def twin(self) -> "Fp8BsChain":
        """The bf16 chain whose rounded output the fused requantization quantizes (the chain itself otherwise)."""
        return quant_twin(self) if self.quant else self

    def at(
        self, block: Optional[int] = None, interleaved: Optional[bool] = None
    ) -> "Fp8BsChain":
        """The same chain at another scale block size and / or weight interleave (``None`` keeps the field)."""
        return dataclasses.replace(
            self,
            block=self.block if block is None else block,
            interleaved=self.interleaved if interleaved is None else interleaved,
        )

    def smem_layout(self, epi_warpgroups: int, token_tile: Optional[int] = None):
        """The half-block amax exchange of the fused requantization (fp32 per buffer, promotion warpgroup and tile row); the
        swapped body has no requantization."""
        if not self.quant or token_tile is not None:
            return None
        return cute.make_layout((XCH_BUFFERS, epi_warpgroups, 128))

    def stage_layout(self):
        """The fused requantization's code staging (bytes per promotion warp): the codes of a round of rows go through the
        warp's smem rows so consecutive lanes store whole 16-byte chunks of one row."""
        return code_stage_layout() if self.quant else None

    def record_ok(self, knobs: Knobs) -> bool:
        """The grouped records the promoted body takes: a tile of two promotion warpgroups (N 64 / 128 / 256) and, for the
        fused requantization, the 128-wide class's tiles whose outputs are whole 128-column blocks."""
        if knobs.tile_n not in TILES_N_GROUPED:
            return False
        return not self.quant or fused_quant_ok(self, knobs.tile_n)

    def swapped_record_ok(self, knobs: Knobs) -> bool:
        """The swapped body runs the bf16 / f32 chains (the fused requantization is not a spelling of it)."""
        return not self.quant

    def shape_ok(self, N: int, K: int, E: int) -> bool:
        """The operand rule both bodies share: K a whole number of smem stages (128 elements: every block one stage, or four
        blocks one stage at the 32-wide class), the weight's rows whole scale blocks (the interleaved weight: whole blocks of
        the up and the gate half, so a tile's columns read whole up and gate blocks, and whole ``2 x GROUP`` row groups)."""
        if K <= 0 or N <= 0 or K % BLOCK_K or N % self.block:
            return False
        if self.il and (N % (2 * self.block) or N % (2 * GROUP)):
            return False
        return 1 <= E <= MAX_E

    def elem(self, fp, acc, acc_gate, alpha_e=None, inv=None):
        """One output element of a bf16 / f32 chain: the bf16 classes' term, converted (the fused requantization runs the
        body's own epilogue)."""
        assert not self.quant, (
            "the fused requantization runs the promotion warpgroups' epilogue"
        )
        if self.pair:
            return glu_elem(fp, acc_gate, acc, self.act, self.out_fmt, self.params)
        if self.kind == "act":
            return act_elem(fp, acc, self.act, self.out_fmt, self.params)
        return fp.cvt_out(acc, self.out_fmt)

    def describe(self) -> str:
        suffix = "".join(
            "_pn" if v is None else f"_p{f32_bits(v):08x}" for v in self.params
        )
        block = "" if self.block == QBLOCK else f"_b{self.block}"
        il = "_il" if self.il != self.pair else ""
        return f"{self.kind}_{self.act}_{self.out_fmt}{suffix}{block}{il}"


GEMM2_CHAIN = Fp8BsChain(
    "plain", "silu", "bf16"
)  # GEMM2: the plain bf16 conversion (the block scales are in the fold)
F32_CHAIN = Fp8BsChain(
    "plain", "silu", "f32"
)  # the oracle output: the promoted fp32 fold as it stands


def gemm1_chain(
    act: str = "silu", params=(), gated: bool = True, fused: bool = False
) -> Fp8BsChain:
    """GEMM1 of the block-scale class: the bf16 class's gated chain on the (gate, up) pair (``act`` in ``spec.ACTS`` with its
    scalars) or its ungated chain on the single accumulator (``spec.UNGATED_ACTS``), rounded to bf16 -- stored as bf16 (the
    standalone requantization pass follows) or requantized in the epilogue (``fused``: the e4m3 codes and the block scales of
    the ``fp8bs.quant`` class)."""
    return Fp8BsChain(
        "glu" if gated else "act",
        act,
        OUT_FMT_QUANT if fused else "bf16",
        params_key(params) if gated else (),
    )


def xch_floats() -> int:
    return XCH_BUFFERS * NUM_PROMO_GROUPS * 128


def code_stage_bytes() -> int:
    """Smem of the fused requantization's code staging: one round of rows per promotion warp."""
    return NUM_PROMO_WARPS * CODE_STAGE_BYTES


def code_stage_layout():
    return cute.make_layout(
        (NUM_PROMO_WARPS, CODE_STAGE_BYTES), stride=(CODE_STAGE_BYTES, 1)
    )  # a warp's rows contiguous


def quant_twin(chain: Fp8BsChain) -> Fp8BsChain:
    """The bf16 chain whose rounded output the fused requantization quantizes."""
    assert chain.out_fmt == OUT_FMT_QUANT, chain
    return Fp8BsChain(chain.kind, chain.act, "bf16", chain.params)


def fused_quant_ok(chain: Fp8BsChain, tile_n: int) -> bool:
    """Whether the fused requantization fits the record: the tile's outputs (half its accumulator columns for the gated chain)
    are whole 128-column blocks, so every block's amax is formed inside one tile (by one warpgroup or the pair's exchange)."""
    outs = tile_n // 2 if chain.pair else tile_n
    return chain.out_fmt == OUT_FMT_QUANT and outs % QBLOCK == 0 and outs >= QBLOCK


def _chain_elem_bs(chain: Fp8BsChain, acc, acc_gate):
    """One output element of a bf16 / f32 chain through ``MoeF32`` (the chain's own ``elem``)."""
    return chain.elem(MoeF32, acc, acc_gate)


def _chain_pre_bs(chain: Fp8BsChain, acc, acc_gate):
    """The chain's value before its bf16 rounding (``glu_pre`` / ``ungated_term`` through ``MoeF32``): the fused requantization
    rounds two of them with one packed conversion, RNE per element like ``cvt_out``."""
    if chain.pair:
        return glu_pre(MoeF32, acc_gate, acc, chain.act, chain.params)
    assert chain.kind == "act", chain
    return ungated_term(MoeF32, chain.act, acc)


def pack_bf16x2(lo, hi):
    """``cvt.rn.bf16x2.f32``: two values rounded to bf16 (RNE, no saturation) in one word, ``lo`` in the low half."""
    from cutlass._mlir.dialects import llvm
    from cutlass.cutlass_dsl import T

    res = llvm.inline_asm(
        T.i32(),
        [Float32(hi).ir_value(), Float32(lo).ir_value()],
        "cvt.rn.bf16x2.f32 $0, $1, $2;",
        "=r,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Int32(res)


def unpack_bf16x2(w):
    """The word's two bf16 halves widened to f32 (exact: a shift and a mask of the bit pattern), ``(lo, hi)``."""
    return bitcast_i32_to_f32(w << 16), bitcast_i32_to_f32(w & Int32(-65536))


# ---- the grouped body's loops


@cute.jit
def grouped_blockscale_mma_loop(
    lowering,
    spb: cutlass.Constexpr,
    num_acc_stage: cutlass.Constexpr,
    k_tile_cnt: Int32,
    is_leader_cta,
    tile_sched,
    tiled_mma: cute.TiledMma,
    ab_consumer,
    acc_pipeline,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    tCtAcc_base: cute.Tensor,
):
    """Per tile, per k-tile (stages ascending), per scale block of the stage (``spb`` of them, ascending): the block's steps from
    a reset accumulator into the next accumulator stage, committed to the epilogue -- the dense block-scale body's mainloop
    (``blockscale_mma_loop``: one block per stage, ``mma_fold`` once per k-tile) for the 128-wide class, and one instruction
    step per block for the 32-wide class (the stage released after its last block's steps are issued). Every block's partial is
    the spec's ``P_kb``."""
    assert cute.size(tCrA, mode=[2]) == lowering.inner and lowering.inner % spb == 0, (
        "one smem stage holds exactly the lowering's inner steps"
    )
    steps = lowering.inner // spb  # instruction steps per scale block
    acc_state = pipeline.make_pipeline_state(
        pipeline.PipelineUserType.Producer, num_acc_stage
    )
    work_tile = tile_sched.initial_work_tile_info()
    while work_tile.is_valid_tile:
        if cutlass.const_expr(spb == 1):
            for _kb in range(
                k_tile_cnt
            ):  # blocks ascending, each a fold from +0 into the next stage
                ab_consumer, acc_state = mma_fold(
                    lowering,
                    1,
                    is_leader_cta,
                    tiled_mma,
                    ab_consumer,
                    acc_pipeline,
                    acc_state,
                    tCrA,
                    tCrB,
                    tCtAcc_base,
                )
        else:
            ab_consumer.reset()
            peek_full = cutlass.Boolean(1)
            if is_leader_cta:
                peek_full = ab_consumer.try_wait()
            for _kt in range(k_tile_cnt):  # stages ascending
                if is_leader_cta:
                    handle = ab_consumer.wait_and_advance(peek_full)
                    for jb in cutlass.range_constexpr(
                        spb
                    ):  # the stage's blocks ascending, each from +0 into its own stage
                        acc_pipeline.producer_acquire(
                            acc_state
                        )  # the epilogue has drained this TMEM buffer
                        tCtAcc = tCtAcc_base[(None, None, None, acc_state.index)]
                        for j in cutlass.range_constexpr(steps):
                            tiled_mma.set(
                                tcgen05.Field.ACCUMULATE, j > 0
                            )  # off on the block's first step only
                            crd = (None, None, jb * steps + j, handle.index)
                            cute.gemm(tiled_mma, tCtAcc, tCrA[crd], tCrB[crd], tCtAcc)
                        acc_pipeline.producer_commit(
                            acc_state
                        )  # the block's partial is complete: hand to the promotion
                        acc_state.advance()
                    handle.release()
                    peek_full = cutlass.Boolean(1)
                    if handle.count + 1 < k_tile_cnt:
                        peek_full = ab_consumer.try_wait()
                else:
                    acc_state.advance(spb)
        tile_sched.advance_to_next_work()
        work_tile = tile_sched.get_current_work()
    acc_pipeline.producer_tail(acc_state)


@cute.jit
def grouped_blockscale_producer_loop(
    st,
    k_tile_cnt: Int32,
    tile_sched,
    ab_producer,
    tma_atom_a: cute.CopyAtom,
    tma_atom_b: cute.CopyAtom,
    mA_mkl: cute.Tensor,
    tBgB: cute.Tensor,
    sA: cute.Tensor,
    tBsB: cute.Tensor,
    thr_mma,
    a_cta_coord,
    a_cta_layout,
    mma_m_in_cluster,
    a_mcast_mask,
    b_mcast_mask,
    scale_pipeline,
    mSFA: cute.Tensor,
    mSFB: cute.Tensor,
    sSFA: cute.Tensor,
    sSFB: cute.Tensor,
    cta_row_rel: Int32,
    Tk: Int32,
    interleaved: cutlass.Constexpr,
    block: cutlass.Constexpr,
):
    """The grouped TMA producer (A re-partitioned from the group-offset view, B indexed by the expert) with the dense body's
    scale ring: per stage (its ``spb`` blocks) the warp's 32 lanes ``cp.async`` this CTA's 128 row scales of each block
    (``mSFA[(row, kb)]``, rows clamped to the last expanded row: a row past the end multiplies a partial that is never stored)
    and lanes ``t < NS`` the tile's weight slot scales (:func:`slot_blocks`: the gated chain's up and gate blocks of the
    interleaved weight, or the plain tile's blocks; the 32-wide class's four blocks of a slot in one 16-byte copy -- the weight
    scales are K-contiguous and ``K / 32`` a multiple of four) into the next ring stage. Order and staging only."""
    lane = cute.arch.lane_idx()
    NB = mSFB.shape[1]
    tile_n = st.mma_tiler[1]
    rows_per_lane = cute.size(sSFA, mode=[0]) // 32
    spb = cute.size(sSFA, mode=[1])
    NS = cute.size(sSFB, mode=[0])
    n_stages = cute.size(sSFA, mode=[2])
    atom_g2s = cute.make_copy_atom(
        cpasync.CopyG2SOp(), ACC_DTYPE, num_bits_per_copy=ACC_DTYPE.width
    )
    atom_g2s4 = cute.make_copy_atom(
        cpasync.CopyG2SOp(), ACC_DTYPE, num_bits_per_copy=4 * ACC_DTYPE.width
    )
    scale_state = pipeline.make_pipeline_state(
        pipeline.PipelineUserType.Producer, n_stages
    )
    work_tile = tile_sched.initial_work_tile_info()
    while work_tile.is_valid_tile:
        real_a = cute.domain_offset((work_tile.row0, 0, 0), mA_mkl)
        gA = cute.local_tile(
            real_a, cute.slice_(st.mma_tiler, (None, 0, None)), (None, None, None)
        )  # (bM, bK, RestM, RestK, RestL)
        tCgA = thr_mma.partition_A(gA)
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            a_cta_coord,
            a_cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        tAgA_slice = tAgA[(None, mma_m_in_cluster, None, 0)]
        tBgB_slice = tBgB[(None, work_tile.n_tile, None, work_tile.e)]
        e = work_tile.e
        row_base = work_tile.row0 + cta_row_rel
        n0 = work_tile.n_tile * Int32(tile_n)
        nbs = slot_blocks(n0, NB, NS, block, interleaved)
        ab_producer.reset()
        peek_empty = ab_producer.try_acquire()
        for _k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
            handle = ab_producer.acquire_and_advance(peek_empty)
            cute.copy(
                tma_atom_a,
                tAgA_slice[(None, handle.count)],
                tAsA[(None, handle.index)],
                tma_bar_ptr=handle.barrier,
                mcast_mask=a_mcast_mask,
            )
            cute.copy(
                tma_atom_b,
                tBgB_slice[(None, handle.count)],
                tBsB[(None, handle.index)],
                tma_bar_ptr=handle.barrier,
                mcast_mask=b_mcast_mask,
            )
            scale_pipeline.producer_acquire(
                scale_state
            )  # ring stage free (the promotion released it)
            kb0 = handle.count * Int32(spb)  # the stage's first scale block
            for r in cutlass.range_constexpr(rows_per_lane):
                i = r * 32 + lane
                row = cutlass.min(row_base + i, Tk - Int32(1))
                for j in cutlass.range_constexpr(spb):
                    cute.copy(
                        atom_g2s,
                        _elem(mSFA, (row, kb0 + Int32(j))),
                        _elem(sSFA, (i, j, scale_state.index)),
                    )
            for t in cutlass.range_constexpr(NS):
                if lane == t:
                    if cutlass.const_expr(spb == 4):
                        cute.copy(
                            atom_g2s4,
                            _vec4(mSFB, (e, nbs[t], kb0)),
                            _vec4(sSFB, (t, 0, scale_state.index)),
                        )
                    else:
                        cute.copy(
                            atom_g2s,
                            _elem(mSFB, (e, nbs[t], kb0)),
                            _elem(sSFB, (t, 0, scale_state.index)),
                        )
            scale_pipeline.producer_commit(
                scale_state
            )  # arrives when this lane's copies have landed
            scale_state.advance()
            peek_empty = cutlass.Boolean(1)
            if handle.count + 1 < k_tile_cnt:
                peek_empty = ab_producer.try_acquire()
        tile_sched.advance_to_next_work()
        work_tile = tile_sched.get_current_work()
    pdl.trigger_late()
    ab_producer.tail()
    scale_pipeline.producer_tail(scale_state)


def block_combine(a, b):
    """The ``elem`` digit's combine (``fmaxc``) of two sub-tree amaxes, through ``CuteF32``."""
    return elem_digit().combine(CuteF32, a, b)


def block_amax_fold(fp, amax, lo, hi):
    """The ``elem`` digit's combine of a thread's running amax with a rounded pair's ``fabs`` leaves (``None`` starts it): max is
    exact in every order, so the fold over the thread's pairs is the value of the canonical pairwise tree
    (``spec_fp8bs.block_reduction``); on a half block the subtree whose combine with the other half's is the block's."""
    d = elem_digit()
    a = d.combine(fp, leaf_value(fp, lo), leaf_value(fp, hi))
    return a if amax is None else d.combine(fp, amax, a)


@cute.jit
def grouped_blockscale_epilogue_loop(
    st,
    chain: cutlass.Constexpr,
    interleaved: cutlass.Constexpr,
    tidx: Int32,
    warp_idx: Int32,
    tile_sched,
    tCtAcc_base: cute.Tensor,
    thr_mma,
    mOut: cute.Tensor,
    mSfo: cute.Tensor,
    Tk: Int32,
    N: Int32,
    epi_tile: cute.Tile,
    acc_pipeline,
    scale_pipeline,
    sSFA: cute.Tensor,
    sSFB: cute.Tensor,
    k_tile_cnt: Int32,
    mma_m_in_cluster,
    cta_row_rel: Int32,
    sC: cute.Tensor = None,
    sX: cute.Tensor = None,
    block: cutlass.Constexpr = QBLOCK,
    sQ: cute.Tensor = None,
):
    """The promotion (the dense body's: two warpgroups, each a contiguous column half of the tile, the block partials folded
    with packed fmas into a register running sum, one accumulator stage released per block, one scale ring stage per smem
    stage) and the group-masked store of the chain's output: the staged row-coalesced store (bf16), the direct vector store
    (the f32 oracle output) or the fused requantization (e4m3 codes and fp32 block scales, module docstring). Every register
    reads the scale of its own weight block (a static slot per register: the gated chain's up or gate block, the 32-wide
    class's ``block``-column block of the group's columns)."""
    pair = chain.pair
    quant = chain.out_fmt == OUT_FMT_QUANT
    twin = chain.twin  # the fused requantization's rounding chain (the bf16 twin); the chain itself otherwise
    out_dt = mOut.element_type
    g = warp_idx // 4  # this warp's promotion group
    tidx_g = tidx % 128
    warp_q = (
        (tidx_g // Int32(32)) % Int32(4)
    )  # this warp's TMEM lane quadrant: its rows, shared with one warp of the other group
    warp_row0 = Int32(32) * warp_q  # the first tile row of this warp's TMEM lanes
    tiler_mn = cute.slice_(st.mma_tiler, (None, None, 0))
    tn = st.mma_tiler[1]
    cta_m = st.cta_tile_shape_mnk[0]
    N_out = N // Int32(2) if cutlass.const_expr(pair) else N
    # the accumulator's flat column space: per-register (row, column) offsets, the chain's pairing
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
    thr_t2r = tiled_copy_t2r.get_slice(tidx_g)
    tTR_tAcc_base = thr_t2r.partition_S(
        tAcc_epi
    )  # (T2R, T2R_M, T2R_N, SUB_M, SUB_N, STAGE)
    cF_epi = cute.flat_divide(tCcF, epi_tile)
    tTR_cF_all = thr_t2r.partition_D(cF_epi)  # the registers' coordinates
    frag = tTR_cF_all[(None, None, None, 0, 0, 0, 0, 0)]
    offs = fragment_offsets(frag.layout)
    V = len(offs)
    sub_m, sub_n = cute.size(tAcc_epi, mode=[2]), cute.size(tAcc_epi, mode=[3])
    assert sub_m == 1 and cta_m == 128, (
        "the epilogue subtiles run along N only: the 128-row CTA tile is one subtile row"
    )
    assert len({m for m, _ in offs}) == 1, (
        "the fragment spans more than one output row",
        offs,
    )
    P = cute.size(epi_tile[1])  # accumulator columns per subtile
    assert sorted(n for _, n in offs) == list(range(P)) and all(
        offs[e][1] == e for e in range(V)
    ), offs
    subtile_cnt = sub_n
    groups = NUM_PROMO_GROUPS if subtile_cnt >= NUM_PROMO_GROUPS else 1
    per_group = subtile_cnt // groups
    assert per_group * groups == subtile_cnt and groups == NUM_PROMO_GROUPS, (
        subtile_cnt,
        groups,
    )
    cols_g = per_group * P  # this group's columns
    assert cols_g % 2 == 0, cols_g
    assert per_group % PROMO_BATCH == 0 or per_group < PROMO_BATCH, (
        per_group,
        PROMO_BATCH,
    )
    batch = min(per_group, PROMO_BATCH)
    VW = per_group * V
    spb = cute.size(sSFA, mode=[1])  # scale blocks per smem stage (1 / 4)
    NS = cute.size(
        sSFB, mode=[0]
    )  # weight scale slots per ring stage (the tile's blocks)
    n_stages = cute.size(sSFA, mode=[2])
    # the weight scale slot of every register (static per register; the group's slot base is a runtime offset that enters
    # addresses only): plain -- the register's `block`-column block of the group's columns; gated -- the block of the register's
    # original up row in the slots' first half, its gate column's block in the second (`group_slot_table`)
    assert P % (2 * GROUP) == 0, P
    n_slots_g, g_mult, slot_of = group_slot_table(
        per_group, V, P, cols_g, NS, block, interleaved
    )
    bufs = [cute.make_rmem_tensor(frag.shape, ACC_DTYPE) for _ in range(batch)]
    tTR_rSum = cute.make_rmem_tensor((VW,), ACC_DTYPE)
    # the chain's output positions: the up positions of the pair (half the registers) or every register; the outputs of a
    # subtile in column order (`out_col`), the group's outputs in order over its subtiles
    _, partner, is_out = store_plan(offs, pair, 16)
    out_col = {
        v: ((n // (2 * GROUP)) * GROUP + n % GROUP if pair else n)
        for v, (m, n) in enumerate(offs)
        if is_out[v]
    }
    V_out = len(out_col)
    assert [out_col[v] for v in range(V) if is_out[v]] == list(range(V_out)) and (
        V_out % 2 == 0
    ), out_col  # a subtile's outputs form in column order: consecutive pairs
    OUT_G = per_group * V_out  # this group's output columns
    # the stores
    tTR_rC = cute.make_rmem_tensor(frag.shape, out_dt)
    tTR_rC.fill(0)
    staged = chain.out_fmt == "bf16"
    direct = chain.out_fmt == "f32"
    acc_state = pipeline.make_pipeline_state(
        pipeline.PipelineUserType.Consumer, st.num_acc_stage
    )
    scale_state = pipeline.make_pipeline_state(
        pipeline.PipelineUserType.Consumer, n_stages
    )
    gsel = g
    slot_idx = group_slot_indices(
        gsel, g_mult, n_slots_g, NS, interleaved
    )  # the group's slots in a ring stage
    # -- the staged bf16 store (the bf16 class's: this group's subtiles through its staging buffer, whole lines per warp instruction)
    epi_m = epi_n = CH = RS = SM = SN = NP = 1
    S_ = 0
    if cutlass.const_expr(staged):
        epi_m, epi_n = cute.size(epi_tile[0]), cute.size(epi_tile[1])
        n_lay = epi_tile[1]
        nested_n = cute.rank(n_lay) == 2
        Pp = cute.size(n_lay.shape[0]) if nested_n else epi_n
        NP = cute.size(n_lay.shape[1]) if nested_n else 1
        S_ = int(n_lay.stride[1]) if nested_n else 0
        step = 2 * GROUP if pair else GROUP
        CH = Pp // step
        RS = 128 // CH
        assert (
            Pp % step == 0
            and 128 % CH == 0
            and epi_m % RS == 0
            and epi_m == cta_m
            and Pp * NP == epi_n
        ), (
            epi_tile,
            step,
            cta_m,
        )
        atom_r2g = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            out_dt,
            num_bits_per_copy=store_plan(offs, pair, 16 // (out_dt.width // 8))[0]
            * out_dt.width,
        )  # 16-byte vectors: the staging rows' skew
        tiled_copy_r2s = cute.make_tiled_copy_D(atom_r2g, tiled_copy_t2r)
        thr_r2s = tiled_copy_r2s.get_slice(tidx_g)
        tRS_sC = thr_r2s.partition_D(sC)
        tRS_rC = tiled_copy_r2s.retile(tTR_rC)
        sC_g = sC[(None, None, gsel)]
        atom_s2g = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), out_dt, num_bits_per_copy=GROUP * out_dt.width
        )
        tiled_s2g = cute.make_tiled_copy_tv(
            atom_s2g,
            cute.make_layout((RS, CH), stride=(CH, 1)),
            cute.make_layout((1, GROUP)),
        )
        thr_s2g = tiled_s2g.get_slice(tidx_g)
        selL = cute.make_layout((epi_m, (GROUP, CH)), stride=(1, (epi_m, step * epi_m)))
        tS_sC = [
            thr_s2g.partition_S(
                cute.composition(cute.local_tile(sC_g, (epi_m, Pp), (0, pc)), selL)
            )
            for pc in range(NP)
        ]
        cO = cute.make_identity_tensor((epi_m, GROUP * CH))
        tS_cO = thr_s2g.partition_D(cO)
        SM, SN = cute.size(tS_cO.shape[1]), cute.size(tS_cO.shape[2])
        ld = cute.assume(mOut.layout.stride[0], divby=8)
        mOutFlat = cute.make_tensor(
            mOut.iterator, cute.make_layout((Tk, N_out), stride=(ld, 1))
        )
    if cutlass.const_expr(direct):
        ve_d = store_plan(offs, pair, 4)[0]
        atom_r2g = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), out_dt, num_bits_per_copy=ve_d * out_dt.width
        )
        tiled_copy_r2g = cute.make_tiled_copy_D(atom_r2g, tiled_copy_t2r)
        thr_r2g = tiled_copy_r2g.get_slice(tidx_g)
        tRG_rC = tiled_copy_r2g.retile(tTR_rC)
        nvec = cute.size(tRG_rC.shape[0][1])
        n1, n2 = cute.size(tRG_rC.shape[1]), cute.size(tRG_rC.shape[2])
        ldd = cute.assume(mOut.layout.stride[0], divby=4)
        if cutlass.const_expr(pair):
            col_shape = (GROUP, 2, N // (2 * GROUP))
            mC = cute.make_tensor(
                mOut.iterator,
                cute.make_layout((Tk, col_shape, 1), stride=(ldd, (1, 0, GROUP), 0)),
            )
        else:
            col_shape = N
            mC = cute.make_tensor(
                mOut.iterator, cute.make_layout((Tk, N, 1), stride=(ldd, 1, 0))
            )
        idC = cute.make_identity_tensor((Tk, col_shape, 1))
        cC = cute.local_tile(idC, tiler_mn, (None, None, None))
        tCcC = transform_partitioned_tensor_layout(thr_mma.partition_C(cC))
        cC_epi = cute.flat_divide(tCcC, epi_tile)
        tRG_cC_all = thr_r2g.partition_D(cC_epi)
    # -- the fused requantization: the group's outputs, bf16-rounded in packed pairs, the block amax (a half or a whole block)
    SB = 1
    blocks_per_tile = 1
    yw = fy_words = None
    if cutlass.const_expr(quant):
        tile_outs = tn // 2 if pair else tn
        assert (
            tile_outs % QBLOCK == 0 and OUT_G % VEC_BYTES == 0 and QBLOCK % OUT_G == 0
        ), (tile_outs, OUT_G)
        SB = QBLOCK // OUT_G  # groups per output block (2: the halves exchange)
        blocks_per_tile = tile_outs // QBLOCK
        assert (
            SB in (1, 2)
            and blocks_per_tile * QBLOCK == tile_outs
            and (SB == 1 or blocks_per_tile == 1)
        )
        yw = cute.make_rmem_tensor(
            cute.make_layout((OUT_G // 2,)), Int32
        )  # the group's bf16-rounded outputs, packed in pairs
        fy_words = cute.make_rmem_tensor(
            cute.make_layout((OUT_G // 4,)), Uint32
        )  # their e4m3 codes as words
        # the staged code stores: NC chunks per row, RPR rows per round through the warp's smem rows (CODE_STAGE_BYTES), the
        # read-back by NC consecutive lanes per row, RPI rows per store instruction
        NC = OUT_G // VEC_BYTES
        RPR = CODE_STAGE_BYTES // OUT_G
        RPI = 32 // NC
        assert (
            32 % NC == 0 and 32 % RPR == 0 and RPR % RPI == 0 and NC & (NC - 1) == 0
        ), (NC, RPR, RPI)
    atom_w4 = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), Uint32, num_bits_per_copy=8 * VEC_BYTES
    )  # a 16-byte chunk as four words
    mOut32 = None
    if cutlass.const_expr(quant):
        ldw = cute.assume(mOut.layout.stride[0], divby=16) // Int32(4)
        mOut32 = cute.make_tensor(
            cute.recast_ptr(mOut.iterator, dtype=Uint32),
            cute.make_layout((Tk, N_out // Int32(4)), stride=(ldw, 1)),
        )
    parity = Int32(0)
    work_tile = tile_sched.initial_work_tile_info()
    while work_tile.is_valid_tile:
        row0, row_end, n_tile = work_tile.row0, work_tile.row_end, work_tile.n_tile
        tile_sched.advance_to_next_work()
        work_tile = tile_sched.get_current_work()
        rows_left = row_end - row0
        tTR_cF = tTR_cF_all[(None, None, None, None, None, mma_m_in_cluster, n_tile, 0)]
        tTR_cF = cute.group_modes(tTR_cF, 3, cute.rank(tTR_cF))
        crd0 = tTR_cF[(None, None, None, 0)][
            0
        ]  # register 0's (row, column) of the tile
        row_rel = crd0[0] - Int32(
            offs[0][0]
        )  # this thread's row inside the cluster tile
        irow = (
            row_rel - cta_row_rel
        )  # its row inside the CTA tile: the ring slot of its A scale
        # a warp whose 32 rows all lie past the group's end skips its TMEM loads and fmas (warp-uniform; the stage releases and
        # the barriers are kept); the pair tile's rows are counted from the cluster tile's first row
        warp_active = rows_left > cta_row_rel + warp_row0
        for i in cutlass.range_constexpr(VW):
            tTR_rSum[i] = CuteF32.const(0.0)  # acc_0 = +0
        for _kt in range(
            k_tile_cnt
        ):  # stages ascending, their blocks ascending: the block fold digit
            scale_pipeline.consumer_wait(
                scale_state
            )  # the stage's blocks' scales are in the ring
            sidx = scale_state.index
            scs = []  # [block of the stage][group slot]: mul.rn(s_a[m, kb], s_b[blk, kb]), once per slot and block
            for jb in cutlass.range_constexpr(spb):
                sa = sSFA[(irow, jb, sidx)]
                scs.append(
                    [
                        CuteF32.mul(sa, sSFB[(slot_idx[t], jb, sidx)])
                        for t in range(n_slots_g)
                    ]
                )
            scale_pipeline.consumer_release(scale_state)
            scale_state.advance()
            for jb in cutlass.range_constexpr(spb):
                acc_pipeline.consumer_wait(acc_state)  # the block's fold is complete
                if warp_active:
                    tTR_tAcc = cute.group_modes(
                        tTR_tAcc_base[(None, None, None, None, None, acc_state.index)],
                        3,
                        5,
                    )
                    for b0 in cutlass.range_constexpr(
                        0, per_group, batch
                    ):  # the group's subtiles in batches: a batch's loads first
                        for j in cutlass.range_constexpr(batch):
                            cute.copy(
                                tiled_copy_t2r,
                                tTR_tAcc[(None, None, None, gsel * per_group + b0 + j)],
                                bufs[j],
                            )
                        for j in cutlass.range_constexpr(batch):
                            k = b0 + j
                            for e in cutlass.range_constexpr(0, V, 2):
                                # acc = fma(scale, P_kb, acc) for two elements per fma.rn.f32x2 (each lane the same correctly
                                # rounded fma) with the scale of the register's weight block (a static slot per register)
                                sc = scs[jb][slot_of[k][e]]
                                r0, r1 = CuteF32.fma2(
                                    (sc, sc),
                                    (bufs[j][e], bufs[j][e + 1]),
                                    (tTR_rSum[k * V + e], tTR_rSum[k * V + e + 1]),
                                )
                                tTR_rSum[k * V + e] = r0
                                tTR_rSum[k * V + e + 1] = r1
                cute.arch.fence_view_async_tmem_load()  # this warp's reads of the partial are complete
                with cute.arch.elect_one():
                    acc_pipeline.consumer_release(
                        acc_state
                    )  # one arrive per warp: the stage is free once all arrived
                acc_state.advance()
        # ===== the chain and the store of this group's half of the tile's fold =====
        if cutlass.const_expr(quant):
            # the fused requantization: the chain's values rounded to bf16 in packed pairs as they form (half the registers of the
            # widened values: the promotion's budget holds them), the block amax folded over the pairs, the half-block amax
            # exchanged with the warp holding the same rows in the other warpgroup
            amax = Float32(0.0)
            if warp_active:
                run = None
                pend = None
                for k in cutlass.range_constexpr(per_group):
                    for v in cutlass.range_constexpr(V):
                        if cutlass.const_expr(is_out[v]):
                            if cutlass.const_expr(pair):
                                y = _chain_pre_bs(
                                    twin,
                                    tTR_rSum[k * V + v],
                                    tTR_rSum[k * V + partner[v]],
                                )
                            else:
                                y = _chain_pre_bs(twin, tTR_rSum[k * V + v], None)
                            j = k * V_out + out_col[v]
                            if cutlass.const_expr(j % 2 == 0):
                                pend = y
                            else:
                                w = pack_bf16x2(pend, y)
                                yw[j // 2] = w
                                lo, hi = unpack_bf16x2(w)
                                run = block_amax_fold(CuteF32, run, lo, hi)
                amax = run
                if cutlass.const_expr(SB == 2):
                    sX[(parity, gsel, tidx_g)] = amax
            if cutlass.const_expr(SB == 2):
                cute.arch.barrier(
                    barrier_id=Int32(BAR_XCH) + warp_q, number_of_threads=64
                )  # both halves' amaxes of this warp's rows are in
                if warp_active:
                    other = sX[(parity, Int32(1) - gsel, tidx_g)]
                    lo = amax  # the block's tree: combine(first half, second half)
                    hi = other
                    if gsel == 1:
                        lo = other
                        hi = amax
                    amax = block_combine(lo, hi)
                parity = Int32(1) - parity
            if warp_active:
                s_blk, inv = block_chain(CuteF32, amax)
                for q in cutlass.range_constexpr(OUT_G // 4):
                    l0, h0 = unpack_bf16x2(yw[2 * q])
                    l1, h1 = unpack_bf16x2(yw[2 * q + 1])
                    yq = [quant_pre(CuteF32, y, inv) for y in (l0, h0, l1, h1)]
                    fy_words[q] = pack_block_codes(CuteF32, yq, "e4m3")[0]
                blk = n_tile * Int32(blocks_per_tile) + (
                    gsel if cutlass.const_expr(SB == 1) else Int32(0)
                )
                half = gsel if cutlass.const_expr(SB == 2) else Int32(0)
                # rows past the group's real end and blocks past the output's columns (a tile wider than N, or an N-cluster's
                # tiles beyond the last one: their B boxes were zero-filled) are never stored
                stored = (row_rel < rows_left) & (blk * Int32(QBLOCK) < N_out)
                row_g = row0 + row_rel
                if stored & (half == Int32(0)):
                    mSfo[(blk, row_g)] = s_blk
                # the code stores, a round of rows at a time through the warp's smem rows: a lane stages its row's chunks, then
                # NC consecutive lanes read and store one row's chunks (whole 16-byte chunks of a line per store instruction);
                # the chunk slot is swizzled by the row so both the staging and the read-back are bank-conflict-free
                lane = tidx_g % Int32(32)
                col0 = blk * Int32(QBLOCK // VEC_BYTES) + half * Int32(NC)
                blk_ok = blk * Int32(QBLOCK) < N_out
                sQw32 = cute.make_tensor(
                    cute.recast_ptr(sQ[(warp_idx, None)].iterator, dtype=Uint32),
                    cute.make_layout((CODE_STAGE_BYTES // 4,)),
                )
                for t in cutlass.range_constexpr(32 // RPR):
                    cute.arch.sync_warp()  # the previous round's chunks are read
                    rr = irow - warp_row0 - Int32(t * RPR)
                    if (rr >= Int32(0)) & (rr < Int32(RPR)):
                        swz = ((rr * Int32(NC)) >> 3) & Int32(NC - 1)
                        for c in cutlass.range_constexpr(NC):
                            cute.copy(
                                atom_w4,
                                cute.local_tile(fy_words, (4,), (c,)),
                                cute.local_tile(
                                    sQw32, (4,), (rr * Int32(NC) + (Int32(c) ^ swz),)
                                ),
                            )
                    cute.arch.sync_warp()
                    for i in cutlass.range_constexpr(RPR // RPI):
                        rr = Int32(i * RPI) + lane // Int32(NC)
                        c = lane % Int32(NC)
                        swz = ((rr * Int32(NC)) >> 3) & Int32(NC - 1)
                        row_w = (
                            warp_row0 + Int32(t * RPR) + rr
                        )  # the row inside the CTA tile
                        if (cta_row_rel + row_w < rows_left) & blk_ok:
                            cute.copy(
                                atom_w4,
                                cute.local_tile(
                                    sQw32, (4,), (rr * Int32(NC) + (c ^ swz),)
                                ),
                                cute.local_tile(
                                    mOut32[(row0 + cta_row_rel + row_w, None)],
                                    (4,),
                                    (col0 + c,),
                                ),
                            )
        else:
            for k in cutlass.range_constexpr(per_group):
                s = gsel * per_group + k  # the group's subtile index within the tile
                if warp_active:
                    for v in cutlass.range_constexpr(V):
                        if cutlass.const_expr(is_out[v]):
                            if cutlass.const_expr(pair):
                                tTR_rC[v] = _chain_elem_bs(
                                    chain,
                                    tTR_rSum[k * V + v],
                                    tTR_rSum[k * V + partner[v]],
                                )
                            else:
                                tTR_rC[v] = _chain_elem_bs(
                                    chain, tTR_rSum[k * V + v], None
                                )
                    if cutlass.const_expr(staged):
                        cute.copy(
                            tiled_copy_r2s, tRS_rC, tRS_sC[(None, None, None, gsel)]
                        )
                    if cutlass.const_expr(direct):
                        tRG_cC = tRG_cC_all[
                            (None, None, None, None, None, mma_m_in_cluster, n_tile, 0)
                        ]
                        tRG_cC = cute.group_modes(tRG_cC, 3, cute.rank(tRG_cC))
                        gC = cute.local_tile(
                            cute.domain_offset((row0, 0, 0), mC),
                            tiler_mn,
                            (None, None, None),
                        )
                        tCgC = transform_partitioned_tensor_layout(
                            thr_mma.partition_C(gC)
                        )
                        gC_epi = cute.flat_divide(tCgC, epi_tile)
                        tRG_gC = thr_r2g.partition_D(gC_epi)[
                            (None, None, None, None, None, mma_m_in_cluster, n_tile, 0)
                        ]
                        tRG_gC = cute.group_modes(tRG_gC, 3, cute.rank(tRG_gC))
                        cC_sub = tRG_cC[(None, None, None, s)]
                        gC_sub = tRG_gC[(None, None, None, s)]
                        for jv in cutlass.range_constexpr(nvec):
                            for mi in cutlass.range_constexpr(n1):
                                for ni in cutlass.range_constexpr(n2):
                                    if cutlass.const_expr(is_out[jv * ve_d]):
                                        crd = cC_sub[((0, jv), mi, ni)]
                                        if cute.elem_less(
                                            crd, (rows_left, col_shape, 1)
                                        ):
                                            cute.copy(
                                                atom_r2g,
                                                tRG_rC[((None, jv), mi, ni)],
                                                gC_sub[((None, jv), mi, ni)],
                                            )
                if cutlass.const_expr(staged):
                    # the buffer -> global by the whole warpgroup: CH consecutive lanes write the CH 16-byte chunks of one output row
                    cute.arch.barrier(
                        barrier_id=Int32(BAR_EPI_STAGE) + gsel, number_of_threads=128
                    )
                    for pc in cutlass.range_constexpr(NP):
                        acc_col0 = n_tile * Int32(tn) + s * Int32(Pp) + Int32(pc * S_)
                        out_col0 = (
                            acc_col0 // Int32(2)
                            if cutlass.const_expr(pair)
                            else acc_col0
                        )
                        out_col0 = cute.assume(out_col0, divby=GROUP)
                        gO = cute.local_tile(
                            cute.domain_offset(
                                (row0 + cta_row_rel, out_col0), mOutFlat
                            ),
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
                        barrier_id=Int32(BAR_EPI_STAGE) + gsel, number_of_threads=128
                    )  # the buffer is free again


# ---- the swapped body's loops


def swap_scale_ring_bytes(
    knobs: Knobs, block: int = QBLOCK, interleaved: bool = True
) -> int:
    """Smem of the scale ring: per stage the tile's token scales and the weight tile's slot scales, ``blocks_per_stage`` of each,
    fp32."""
    spb = blocks_per_stage(block)
    return (
        NUM_SCALE_STAGES_OF[block]
        * (knobs.tile_n + n_scale_slots(knobs.tile_m, block, interleaved))
        * spb
        * (ACC_DTYPE.width // 8)
    )


@cute.jit
def swap_bs_producer_loop(
    st,
    k_tile_cnt: Int32,
    tile_sched,
    ab_producer,
    tma_atom_w: cute.CopyAtom,
    tma_atom_tok: cute.CopyAtom,
    tWgW: cute.Tensor,
    tWsW: cute.Tensor,
    mTok_nkl: cute.Tensor,
    sTok: cute.Tensor,
    thr_mma,
    b_cta_coord,
    b_cta_layout,
    scale_pipeline,
    mSFA: cute.Tensor,
    mSFW: cute.Tensor,
    sSFA: cute.Tensor,
    sSFB: cute.Tensor,
    R: Int32,
    interleaved: cutlass.Constexpr,
    block: cutlass.Constexpr,
):
    """The TMA producer: per tile the expert's weight rows ``n_tile`` (the expert as the batch coordinate) and the token box re-addressed
    at the group's dense row ``row0`` (``token_tile`` rows: the rows past the group's end are computed and never stored, the rows past the
    buffer zero-filled); k-tiles (smem stages of ``spb`` scale blocks) ascending, one ring slot holds both. Next to each stage the warp's
    lanes ``cp.async`` the tile's token scales of the stage's blocks (lane ``j < token_tile``: ``mSFA[(row0 + j, kb)]``, the row clamped
    to the buffer) and lanes ``t < NS`` the weight tile's slot scales (:func:`slot_blocks`: the interleaved weight's up
    and gate blocks, the plain weight's blocks; the 32-wide class's four blocks of a slot in one 16-byte copy) into the scale ring. The
    weights are inputs of the whole chain, the token rows and their scales the predecessor's output: the first tile's weight stages are
    issued before the programmatic-dependent-launch wait and their token boxes right after it. Order and staging only."""
    lane = cute.arch.lane_idx()
    NB = mSFW.shape[1]
    tn = st.cta_tile_shape_mnk[1]
    wt = st.cta_tile_shape_mnk[0]
    rows_per_lane = (tn + 31) // 32
    spb = cute.size(sSFA, mode=[1])
    NS = cute.size(sSFB, mode=[0])
    n_stages = cute.size(sSFA, mode=[2])
    atom_g2s = cute.make_copy_atom(
        cpasync.CopyG2SOp(), ACC_DTYPE, num_bits_per_copy=ACC_DTYPE.width
    )
    atom_g2s4 = cute.make_copy_atom(
        cpasync.CopyG2SOp(), ACC_DTYPE, num_bits_per_copy=4 * ACC_DTYPE.width
    )
    scale_state = pipeline.make_pipeline_state(
        pipeline.PipelineUserType.Producer, n_stages
    )
    work_tile = tile_sched.initial_work_tile_info()
    n_pre = cutlass.min(Int32(st.num_ab_stage), k_tile_cnt)
    if not work_tile.is_valid_tile:
        n_pre = Int32(0)  # a CTA without tiles issues nothing
    tWgW_slice = tWgW[(None, work_tile.n_tile, None, work_tile.e)]
    ab_producer.reset()
    tok_producer = (
        ab_producer.clone()
    )  # the same stages, walked again for the token boxes
    for _k_tile in cutlass.range(0, n_pre, 1, unroll=1):
        handle = ab_producer.acquire_and_advance()
        cute.copy(
            tma_atom_w,
            tWgW_slice[(None, handle.count)],
            tWsW[(None, handle.index)],
            tma_bar_ptr=handle.barrier,
        )
    cute.arch.griddepcontrol_wait()  # the token rows and their scales are the predecessor's
    k_start = n_pre
    first = Int32(1)
    while work_tile.is_valid_tile:
        e = work_tile.e
        row0 = work_tile.row0
        n0 = work_tile.n_tile * Int32(wt)
        tWgW_slice = tWgW[(None, work_tile.n_tile, None, e)]
        real_tok = cute.domain_offset((row0, 0, 0), mTok_nkl)
        gTok = cute.local_tile(
            real_tok, cute.slice_(st.mma_tiler, (0, None, None)), (None, None, None)
        )
        tCgTok = thr_mma.partition_B(gTok)
        tTsTok, tTgTok = cpasync.tma_partition(
            tma_atom_tok,
            b_cta_coord,
            b_cta_layout,
            cute.group_modes(sTok, 0, 3),
            cute.group_modes(tCgTok, 0, 3),
        )
        tTgTok_slice = tTgTok[(None, 0, None, 0)]
        nbs = slot_blocks(n0, NB, NS, block, interleaved)
        if first == 1:
            # the first tile's prefetched weight stages: their token boxes and scales
            for _k_tile in cutlass.range(0, n_pre, 1, unroll=1):
                handle = tok_producer.current_handle()
                cute.copy(
                    tma_atom_tok,
                    tTgTok_slice[(None, handle.count)],
                    tTsTok[(None, handle.index)],
                    tma_bar_ptr=handle.barrier,
                )
                tok_producer.advance()
                scale_pipeline.producer_acquire(scale_state)
                swap_bs_stage_scales(
                    lane,
                    handle.count,
                    scale_state.index,
                    row0,
                    R,
                    e,
                    nbs,
                    mSFA,
                    mSFW,
                    sSFA,
                    sSFB,
                    atom_g2s,
                    atom_g2s4,
                    tn,
                    rows_per_lane,
                    spb,
                    NS,
                )
                scale_pipeline.producer_commit(scale_state)
                scale_state.advance()
        first = Int32(0)
        peek_empty = cutlass.Boolean(1)
        if k_start < k_tile_cnt:
            peek_empty = ab_producer.try_acquire()
        for _k_tile in cutlass.range(k_start, k_tile_cnt, 1, unroll=1):
            handle = ab_producer.acquire_and_advance(peek_empty)
            cute.copy(
                tma_atom_w,
                tWgW_slice[(None, handle.count)],
                tWsW[(None, handle.index)],
                tma_bar_ptr=handle.barrier,
            )
            cute.copy(
                tma_atom_tok,
                tTgTok_slice[(None, handle.count)],
                tTsTok[(None, handle.index)],
                tma_bar_ptr=handle.barrier,
            )
            scale_pipeline.producer_acquire(
                scale_state
            )  # ring stage free (the promotion released it)
            swap_bs_stage_scales(
                lane,
                handle.count,
                scale_state.index,
                row0,
                R,
                e,
                nbs,
                mSFA,
                mSFW,
                sSFA,
                sSFB,
                atom_g2s,
                atom_g2s4,
                tn,
                rows_per_lane,
                spb,
                NS,
            )
            scale_pipeline.producer_commit(
                scale_state
            )  # arrives when this lane's copies have landed
            scale_state.advance()
            peek_empty = cutlass.Boolean(1)
            if handle.count + 1 < k_tile_cnt:
                peek_empty = ab_producer.try_acquire()
        ab_producer.reset()  # the next tile's k-tiles count from 0
        k_start = Int32(0)
        tile_sched.advance_to_next_work()
        work_tile = tile_sched.get_current_work()
    pdl.trigger_late()
    ab_producer.tail()
    scale_pipeline.producer_tail(scale_state)


@cute.jit
def swap_bs_gather_producer_loop(
    st,
    k_tile_cnt: Int32,
    tile_sched,
    ab_producer,
    tma_atom_w: cute.CopyAtom,
    tma_atom_tok: cute.CopyAtom,
    tWgW: cute.Tensor,
    tWsW: cute.Tensor,
    mX: cute.Tensor,
    mSrc: cute.Tensor,
    k_div: Int32,
    sTok: cute.Tensor,
    Tk: Int32,
    tn: cutlass.Constexpr,
    tile_k: cutlass.Constexpr,
    scale_pipeline,
    mSFA: cute.Tensor,
    mSFW: cute.Tensor,
    sSFA: cute.Tensor,
    sSFB: cute.Tensor,
    interleaved: cutlass.Constexpr,
    block: cutlass.Constexpr,
):
    """The gather form of :func:`swap_bs_producer_loop`: the weight tile as there; the token box gathered by the TMA's gather4 mode
    from the unpermuted rows ``mX (T, K)`` through the plan's map (row ``p`` reads ``mX[mSrc[p] // k_div]``; the rows past the
    group's end repeat its last row and are never stored), one swizzle atom of K per stage; the token scales of a stage's blocks
    from the unpermuted scale columns at the same rows (``mSFA (T, KB)``), the weight slot scales as there. The row map is the
    predecessor's: the first tile's weight stages are issued before the programmatic-dependent-launch wait, the token gathers and
    the scales after it. Order and staging only: the stage holds the bytes the box form's stage holds."""
    lane = cute.arch.lane_idx()
    NB = mSFW.shape[1]
    wt = st.cta_tile_shape_mnk[0]
    rows_per_lane = (tn + 31) // 32
    spb = cute.size(sSFA, mode=[1])
    NS = cute.size(sSFB, mode=[0])
    n_stages = cute.size(sSFA, mode=[2])
    atom_g2s = cute.make_copy_atom(
        cpasync.CopyG2SOp(), ACC_DTYPE, num_bits_per_copy=ACC_DTYPE.width
    )
    atom_g2s4 = cute.make_copy_atom(
        cpasync.CopyG2SOp(), ACC_DTYPE, num_bits_per_copy=4 * ACC_DTYPE.width
    )
    stage_elems = tn * tile_k
    sTok_plain = cute.make_tensor(
        sTok.iterator,
        cute.make_layout(
            (tn, tile_k, st.num_ab_stage), stride=(tile_k, 1, stage_elems)
        ),
    )  # the atom applies the swizzle
    sTok_plain = cute.group_modes(sTok_plain, 0, 2)
    idx = cute.make_rmem_tensor(cute.make_layout((tn,)), Int32)
    tok_idx = cute.make_rmem_tensor(cute.make_layout((rows_per_lane,)), Int32)
    gX = cute.local_tile(mX, (tn, tile_k), (0, None))  # (tn, tile_k, n_kt)
    n_kt = cute.size(gX, mode=[2])
    scale_state = pipeline.make_pipeline_state(
        pipeline.PipelineUserType.Producer, n_stages
    )
    work_tile = tile_sched.initial_work_tile_info()
    n_pre = cutlass.min(Int32(st.num_ab_stage), k_tile_cnt)
    if not work_tile.is_valid_tile:
        n_pre = Int32(0)
    tWgW_slice = tWgW[(None, work_tile.n_tile, None, work_tile.e)]
    ab_producer.reset()
    tok_producer = ab_producer.clone()
    for _k_tile in cutlass.range(0, n_pre, 1, unroll=1):
        handle = ab_producer.acquire_and_advance()
        cute.copy(
            tma_atom_w,
            tWgW_slice[(None, handle.count)],
            tWsW[(None, handle.index)],
            tma_bar_ptr=handle.barrier,
        )
    cute.arch.griddepcontrol_wait()  # the row map, the rows and their scales are the predecessor's
    k_start = n_pre
    first = Int32(1)
    while work_tile.is_valid_tile:
        e = work_tile.e
        row0 = work_tile.row0
        n0 = work_tile.n_tile * Int32(wt)
        tWgW_slice = tWgW[(None, work_tile.n_tile, None, e)]
        last = cutlass.max(cutlass.min(work_tile.row_end, Tk) - Int32(1), Int32(0))
        for j in cutlass.range_constexpr(tn):
            idx[j] = mSrc[cutlass.min(row0 + Int32(j), last)] // k_div
        for r in cutlass.range_constexpr(rows_per_lane):
            tok_idx[r] = mSrc[cutlass.min(row0 + Int32(r * 32) + lane, last)] // k_div
        gI = cute.make_tensor(
            idx.iterator, cute.make_layout((tn, tile_k, n_kt), stride=(1, 0, 0))
        )
        tTsTok, tTgX, tTgI = cpasync.tma_partition(
            tma_atom_tok,
            0,
            cute.make_layout(1),
            sTok_plain,
            [cute.group_modes(gX, 0, 2), cute.group_modes(gI, 0, 2)],
        )
        nbs = slot_blocks(n0, NB, NS, block, interleaved)
        if first == 1:
            for _k_tile in cutlass.range(0, n_pre, 1, unroll=1):
                handle = tok_producer.current_handle()
                cute.copy(
                    tma_atom_tok,
                    [tTgX[(None, handle.count)], tTgI[(None, handle.count)]],
                    tTsTok[(None, handle.index)],
                    tma_bar_ptr=handle.barrier,
                )
                tok_producer.advance()
                scale_pipeline.producer_acquire(scale_state)
                swap_bs_stage_scales(
                    lane,
                    handle.count,
                    scale_state.index,
                    row0,
                    Tk,
                    e,
                    nbs,
                    mSFA,
                    mSFW,
                    sSFA,
                    sSFB,
                    atom_g2s,
                    atom_g2s4,
                    tn,
                    rows_per_lane,
                    spb,
                    NS,
                    tok_idx=tok_idx,
                )
                scale_pipeline.producer_commit(scale_state)
                scale_state.advance()
        first = Int32(0)
        peek_empty = cutlass.Boolean(1)
        if k_start < k_tile_cnt:
            peek_empty = ab_producer.try_acquire()
        for _k_tile in cutlass.range(k_start, k_tile_cnt, 1, unroll=1):
            handle = ab_producer.acquire_and_advance(peek_empty)
            cute.copy(
                tma_atom_w,
                tWgW_slice[(None, handle.count)],
                tWsW[(None, handle.index)],
                tma_bar_ptr=handle.barrier,
            )
            cute.copy(
                tma_atom_tok,
                [tTgX[(None, handle.count)], tTgI[(None, handle.count)]],
                tTsTok[(None, handle.index)],
                tma_bar_ptr=handle.barrier,
            )
            scale_pipeline.producer_acquire(scale_state)
            swap_bs_stage_scales(
                lane,
                handle.count,
                scale_state.index,
                row0,
                Tk,
                e,
                nbs,
                mSFA,
                mSFW,
                sSFA,
                sSFB,
                atom_g2s,
                atom_g2s4,
                tn,
                rows_per_lane,
                spb,
                NS,
                tok_idx=tok_idx,
            )
            scale_pipeline.producer_commit(scale_state)
            scale_state.advance()
            peek_empty = cutlass.Boolean(1)
            if handle.count + 1 < k_tile_cnt:
                peek_empty = ab_producer.try_acquire()
        ab_producer.reset()
        k_start = Int32(0)
        tile_sched.advance_to_next_work()
        work_tile = tile_sched.get_current_work()
    pdl.trigger_late()
    ab_producer.tail()
    scale_pipeline.producer_tail(scale_state)


@cute.jit
def swap_bs_stage_scales(
    lane,
    kt,
    sidx,
    row0: Int32,
    R: Int32,
    e,
    nbs,
    mSFA: cute.Tensor,
    mSFW: cute.Tensor,
    sSFA: cute.Tensor,
    sSFB: cute.Tensor,
    atom_g2s,
    atom_g2s4,
    tn: cutlass.Constexpr,
    rows_per_lane: cutlass.Constexpr,
    spb: cutlass.Constexpr,
    NS: cutlass.Constexpr,
    tok_idx=None,
):
    """One ring stage's copies for smem stage ``kt`` (scale blocks ``kt * spb ..``): the tile's token scales per block (lanes ``j <
    tn``, the row clamped to the buffer; in the gather form ``tok_idx[r]``, the lane's token row of the unpermuted scales) and the
    weight tile's slot scales (lanes ``t < NS``; the four blocks of a slot one 16-byte copy at ``spb = 4``: the weight scales are
    K-contiguous and ``K / 32`` a multiple of four)."""
    kb0 = kt * Int32(spb)
    for r in cutlass.range_constexpr(rows_per_lane):
        j = r * 32 + lane
        if j < tn:
            if cutlass.const_expr(tok_idx is not None):
                row = tok_idx[r]
            else:
                row = cutlass.min(row0 + j, R - Int32(1))
            for jb in cutlass.range_constexpr(spb):
                cute.copy(
                    atom_g2s,
                    _elem(mSFA, (row, kb0 + Int32(jb))),
                    _elem(sSFA, (j, jb, sidx)),
                )
    for t in cutlass.range_constexpr(NS):
        if lane == t:
            if cutlass.const_expr(spb == 4):
                cute.copy(
                    atom_g2s4, _vec4(mSFW, (e, nbs[t], kb0)), _vec4(sSFB, (t, 0, sidx))
                )
            else:
                cute.copy(
                    atom_g2s, _elem(mSFW, (e, nbs[t], kb0)), _elem(sSFB, (t, 0, sidx))
                )


@cute.jit
def swap_bs_mma_loop(
    lowering,
    spb: cutlass.Constexpr,
    num_acc_stage: cutlass.Constexpr,
    k_tile_cnt: Int32,
    tile_sched,
    tiled_mma: cute.TiledMma,
    ab_consumer,
    acc_pipeline,
    tCrW: cute.Tensor,
    tCrTok: cute.Tensor,
    tCtAcc_base: cute.Tensor,
):
    """Per tile, per k-tile (stages ascending), per scale block of the stage (``spb``, ascending): the block's steps from a reset
    accumulator into the next accumulator stage, committed to the epilogue -- the dense block-scale body's mainloop
    (``blockscale_mma_loop``) with the weights as the A operand; the 32-wide class commits every instruction step."""
    assert cute.size(tCrW, mode=[2]) == lowering.inner and lowering.inner % spb == 0, (
        "one smem stage must hold exactly the lowering's inner steps"
    )
    steps = lowering.inner // spb  # instruction steps per scale block
    acc_state = pipeline.make_pipeline_state(
        pipeline.PipelineUserType.Producer, num_acc_stage
    )
    work_tile = tile_sched.initial_work_tile_info()
    while work_tile.is_valid_tile:
        ab_consumer.reset()
        peek_full = ab_consumer.try_wait()
        for _kt in range(k_tile_cnt):
            handle = ab_consumer.wait_and_advance(peek_full)
            for jb in cutlass.range_constexpr(
                spb
            ):  # the stage's blocks ascending, each from +0 into its own stage
                acc_pipeline.producer_acquire(
                    acc_state
                )  # the epilogue has drained this TMEM stage
                tCtAcc = tCtAcc_base[(None, None, None, acc_state.index)]
                for j in cutlass.range_constexpr(
                    steps
                ):  # the block's steps ascending, the first from +0
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, j > 0)
                    crd = (None, None, jb * steps + j, handle.index)
                    cute.gemm(tiled_mma, tCtAcc, tCrW[crd], tCrTok[crd], tCtAcc)
                acc_pipeline.producer_commit(
                    acc_state
                )  # the block's partial is complete: hand to the promotion
                acc_state.advance()
            handle.release()
            peek_full = cutlass.Boolean(1)
            if handle.count + 1 < k_tile_cnt:
                peek_full = ab_consumer.try_wait()
        tile_sched.advance_to_next_work()
        work_tile = tile_sched.get_current_work()
    acc_pipeline.producer_tail(acc_state)


@cute.jit
def swap_bs_epilogue_loop(
    st,
    chain: cutlass.Constexpr,
    interleaved: cutlass.Constexpr,
    tidx: Int32,
    tile_sched,
    tCtAcc_base: cute.Tensor,
    thr_mma,
    mOut: cute.Tensor,
    Tk: Int32,
    N: Int32,
    epi_tile: cute.Tile,
    acc_pipeline,
    scale_pipeline,
    sSFA: cute.Tensor,
    sSFB: cute.Tensor,
    k_tile_cnt: Int32,
    block: cutlass.Constexpr = QBLOCK,
):
    """The promotion and the transposed group-masked store: the accumulator tile is (weight rows, tokens). The TMEM load's layout gives
    each thread a static set of (weight row, token column) offsets from a per-thread base (the 128-row tile: one row, consecutive
    columns; the 64-row tile: two rows 8 apart, every fourth column). Per block the thread forms ``mul.rn(s_a[tok j, kb], s_b[blk(row),
    kb])`` per register (packed in pairs: each lane the scalar op; the weight block of a register's row is a slot fixed per thread -- the
    row's ``block``-row block of the tile, the interleaved tile's gate rows reading the gate half's slots) and folds the block partial
    with ``fma.rn.f32x2`` into its register running sums (the 32-wide class: four blocks per smem stage, one ring stage). After the
    last block the chain -- gated: an up row's gate row (8 rows below) is in the thread's own fragment or at lane ``+ 8`` -- and the
    store: token ``j`` of the tile is expanded row ``row0 + j``, stored iff ``j < rows_left``; the up rows store output column ``(n //
    16) * 8 + n % 8`` (gated), every row column ``n`` (plain)."""
    pair = chain.pair
    wt = st.cta_tile_shape_mnk[0]
    spb = cute.size(sSFA, mode=[1])
    NS = cute.size(sSFB, mode=[0])
    n_stages = cute.size(sSFA, mode=[2])
    tiler_mn = cute.slice_(st.mma_tiler, (None, None, 0))
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
    cF_epi = cute.flat_divide(
        tCcF, epi_tile
    )  # (EPI_M, EPI_N, SUB_M, SUB_N, RestM, RestN, RestL)
    tTR_cF = thr_t2r.partition_D(cF_epi)  # (T2R, T2R_M, T2R_N, SUB_M, SUB_N, 1, 1, 1)
    frag = tTR_cF[(None, None, None, 0, 0, 0, 0, 0)]
    offs = fragment_offsets(
        frag.layout
    )  # the registers' static (row, column) offsets from the thread's base
    nv = len(offs)
    assert nv % 2 == 0, offs
    index = {o: v for v, o in enumerate(offs)}
    # the gate row of an up register (8 rows below it): the thread's own register when the layout holds that row, else lane + 8
    partner = [index.get((m + GROUP, n)) for m, n in offs]
    sub_m, sub_n = cute.size(tAcc_epi, mode=[2]), cute.size(tAcc_epi, mode=[3])
    assert sub_m == 1, "the epilogue subtiles run along the token columns only"
    NSUB = sub_n
    VW = NSUB * nv
    tTR_cF = cute.group_modes(tTR_cF, 3, cute.rank(tTR_cF))
    crd0 = tTR_cF[(None, None, None, 0)][(0, 0, 0)]
    base_row = crd0[0] - Int32(offs[0][0])  # this thread's base weight row in the tile
    col_base = [
        tTR_cF[(None, None, None, s)][(0, 0, 0)][1] - Int32(offs[0][1])
        for s in range(NSUB)
    ]  # its base token column per subtile
    rows = sorted({m for m, _ in offs})
    buf = cute.make_rmem_tensor(frag.shape, ACC_DTYPE)
    tTR_rSum = cute.make_rmem_tensor((VW,), ACC_DTYPE)
    # the weight rows of the thread's registers: gate rows of the interleaved tile read the gate half's slots; the slot of a row is
    # its `block`-row block of the tile (`slot_blocks`: the interleaved tile's up rows (n // 16) * 8 + n % 8 index the up half's slots)
    gate_row = {
        m: ((base_row + Int32(m)) % Int32(2 * GROUP)) >= Int32(GROUP) for m in rows
    }
    slot_row = {}
    for m in rows:
        wrow = base_row + Int32(m)
        if cutlass.const_expr(interleaved):
            u = (wrow // Int32(2 * GROUP)) * Int32(GROUP) + wrow % Int32(
                GROUP
            )  # the row's original row within the tile's half
            slot = (u // Int32(block)) if (wt // 2) >= block else Int32(0)
            if gate_row[m]:
                slot = slot + Int32(
                    NS // 2
                )  # the gate half's slots follow the up half's
            slot_row[m] = slot
        else:
            slot_row[m] = (wrow // Int32(block)) if wt >= block else Int32(0)
    acc_state = pipeline.make_pipeline_state(
        pipeline.PipelineUserType.Consumer, st.num_acc_stage
    )
    scale_state = pipeline.make_pipeline_state(
        pipeline.PipelineUserType.Consumer, n_stages
    )
    work_tile = tile_sched.initial_work_tile_info()
    while work_tile.is_valid_tile:
        row0, row_end, n_tile = work_tile.row0, work_tile.row_end, work_tile.n_tile
        tile_sched.advance_to_next_work()
        work_tile = tile_sched.get_current_work()
        rows_left = cutlass.min(row_end, Tk) - row0  # valid tokens of this tile
        for i in cutlass.range_constexpr(VW):
            tTR_rSum[i] = CuteF32.const(0.0)  # acc_0 = +0
        for _kt in range(
            k_tile_cnt
        ):  # stages ascending, their blocks ascending: the block fold digit
            scale_pipeline.consumer_wait(
                scale_state
            )  # the stage's blocks' scales are in the ring
            sidx = scale_state.index
            for jb in cutlass.range_constexpr(spb):
                sb = {}
                for mi in cutlass.range_constexpr(len(rows)):
                    m = rows[mi]
                    sb[m] = sSFB[(slot_row[m], jb, sidx)]
                scs = []
                for s in cutlass.range_constexpr(NSUB):
                    for v in cutlass.range_constexpr(0, nv, 2):
                        a0 = sSFA[(col_base[s] + Int32(offs[v][1]), jb, sidx)]
                        a1 = sSFA[(col_base[s] + Int32(offs[v + 1][1]), jb, sidx)]
                        p0, p1 = CuteF32.mul2(
                            (a0, a1), (sb[offs[v][0]], sb[offs[v + 1][0]])
                        )  # mul.rn(s_a[tok, kb], s_b[blk(row), kb]) per register
                        scs.append(p0)
                        scs.append(p1)
                if cutlass.const_expr(jb == spb - 1):
                    scale_pipeline.consumer_release(scale_state)
                    scale_state.advance()
                acc_pipeline.consumer_wait(acc_state)  # the block's fold is complete
                tTR_tAcc = cute.group_modes(
                    tTR_tAcc_base[(None, None, None, None, None, acc_state.index)], 3, 5
                )
                for s in cutlass.range_constexpr(NSUB):
                    cute.copy(tiled_copy_t2r, tTR_tAcc[(None, None, None, s)], buf)
                    for v in cutlass.range_constexpr(0, nv, 2):
                        # acc = fma(scale, P_kb, acc) for two registers per fma.rn.f32x2 (each lane the same correctly rounded fma)
                        r0, r1 = CuteF32.fma2(
                            (scs[s * nv + v], scs[s * nv + v + 1]),
                            (buf[v], buf[v + 1]),
                            (tTR_rSum[s * nv + v], tTR_rSum[s * nv + v + 1]),
                        )
                        tTR_rSum[s * nv + v] = r0
                        tTR_rSum[s * nv + v + 1] = r1
                cute.arch.fence_view_async_tmem_load()  # this warp's reads of the partial are complete
                with cute.arch.elect_one():
                    acc_pipeline.consumer_release(
                        acc_state
                    )  # one arrive per warp: the stage is free once all arrived
                acc_state.advance()
        # ===== the chain and the transposed store =====
        # the gate rows' sums for the up rows (gated): the thread's own register 8 rows below, or lane + 8 (rows 16q + 8 + r sit 8 lanes
        # below rows 16q + r in the 128-row layout); the shuffles are warp-uniform, before any branch
        gate = [None] * VW
        if cutlass.const_expr(pair):
            for s in cutlass.range_constexpr(NSUB):
                for v in cutlass.range_constexpr(nv):
                    if cutlass.const_expr(partner[v] is not None):
                        gate[s * nv + v] = tTR_rSum[s * nv + partner[v]]
                    else:
                        gate[s * nv + v] = cute.arch.shuffle_sync_down(
                            tTR_rSum[s * nv + v], offset=GROUP
                        )
        for mi in cutlass.range_constexpr(len(rows)):
            m = rows[mi]
            wrow = base_row + Int32(m)  # this register row's weight row in the tile
            n_glob = n_tile * Int32(wt) + wrow
            if cutlass.const_expr(pair):
                out_col = (
                    n_tile * Int32(wt // 2)
                    + (wrow // Int32(2 * GROUP)) * Int32(GROUP)
                    + wrow % Int32(GROUP)
                )
                store_ok = (~gate_row[m]) & (n_glob < N)
            else:
                out_col = n_glob
                store_ok = n_glob < N
            for s in cutlass.range_constexpr(NSUB):
                for v in cutlass.range_constexpr(nv):
                    if cutlass.const_expr(offs[v][0] == m):
                        i = s * nv + v
                        j = col_base[s] + Int32(offs[v][1])
                        if (j < rows_left) & store_ok:
                            val = _chain_elem_bs(
                                chain,
                                tTR_rSum[i],
                                gate[i] if cutlass.const_expr(pair) else None,
                            )
                            mOut[(row0 + j, out_col)] = val
