# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Group-masked epilogue block: TMEM -> registers, the chain, predicated vector stores.

The epilogue block of the grouped bodies (:mod:`.kernel`, :mod:`.swap`) is a chain object (:class:`GroupedEpilogueBlock`:
the bf16 classes' :class:`GroupedChain`, the fp8 per-tensor classes' :class:`Fp8Chain`); the loops below take it as a
compile-time constant, so every format-dependent choice (the operand format, the per-tile scale, the paired 8-bit
conversion, the store vector) is a ``const_expr`` branch and each format compiles to its own kernel.

Chains of the bf16 classes (:class:`GroupedChain`; the class's ``plain`` / ``swiglu`` / ``relu2`` names map onto it):

* ``plain``: every accumulator element ``cvt.rn`` to the output type (GEMM2, or the fp32 oracle output of either GEMM);
* ``glu``: GEMM1's B rows are gate/up-interleaved in groups of ``GROUP = 8`` (accumulator columns ``16q + r`` hold
  ``u_{8q+r}``, ``16q + 8 + r`` hold ``g_{8q+r}``), so a thread's register fragment holds both halves of every pair it
  owns, for every TMEM load layout the space uses (checked statically below). Each pair becomes the gated-activation
  term (``spec.glu_elem`` through ``CuteF32``) at output column ``8q + r``: the output is addressed through a view of the
  ``(Tk, I)`` buffer in the accumulator's column space, ``(Tk, (8, 2, N / 16)) : (ld, (1, 0, 8))``, in which the gate
  positions alias the up positions and are never stored;
* ``act``: the ungated chain (``spec.act_elem``: ReLU^2 of one accumulator, GEMM1 of the ``relu2`` class) on every
  accumulator element, stored like ``plain``.

Chains of the fp8 per-tensor classes (:class:`Fp8Chain`, the W8A8 chains of :mod:`..spec_fp8`):

* ``glu`` / ``act`` with the e4m3 output: GEMM1's fused quantization (``spec_fp8.glu_quant_e4m3_elem`` on the (gate, up)
  pair in one thread's fragment, ``act_quant_e4m3_elem`` on the single accumulator): the activation on the alpha-scaled
  fp32 accumulators, the ``mul.rn`` by the intermediate multiplier (``mSi[0]``, read as given), then two outputs per
  ``cvt.rn.satfinite.e4m3x2.f32`` (each value's conversion is independent of its pairing) into byte registers;
* ``plain`` scaled with the bf16 output: GEMM2 (``spec_fp8.scaled_out_elem``, ``alpha2 = mul(s_inter, s_w2[e])``);
* ``plain`` unscaled with the fp32 output: the raw accumulators (the oracle of the device suite).

The per-tile alpha ``mul(mSa[0], mAlpha[e])`` is formed once per tile (``spec_fp8.w8a8_alpha``: the dense fp8 class's
alpha); the three scale tensors are the fp8 chains' extra operands (:attr:`GroupedEpilogueBlock.scale_operands`), absent
from the bf16 kernels.

Chains of the NVFP4 classes (:class:`W4A4Chain`, the W4A4 chains of :mod:`..spec_nvfp4`): the bf16 chains on the row-scaled
accumulators, ``s = mul(alpha[e], pts[row])`` (the expert's alpha as given times the row's per-token scale, read at the padded
position of the dense packing), ``scale == "w4a4"``; GEMM2 of the static-intermediate class with ``s = mul(alpha[e],
inv2[e])`` once per tile (``w4a4s``); the static class's GEMM1 (``out_fmt == "nvfp4"``): the fp32 chain value quantized per
16-column block with the expert's static scale (the per-tensor quantizer's block chain: codes and scale bytes out); the unscaled
oracle chains (``none``). The chain declares which further operands the kernel takes (:attr:`GroupedEpilogueBlock.operands`:
``alpha`` / ``sa`` / ``si`` / ``pts`` / ``aux`` / ``out_sf``) and forms its per-tile constant and its per-row scale through
``tile_scale`` / ``row_scale``, so the loops below stay one body.

Stores: the converted fragment goes registers -> shared memory in the TMEM load's thread layout (its widest vectors,
into a row-padded buffer, ``STAGE_PAD``; one subtile buffer per epilogue warpgroup), then shared -> global by the
warpgroup with consecutive lanes on consecutive 8-element chunks of one row (``make_tiled_copy_tv``: 16-byte chunks,
whole lines per warp instruction), each chunk copied under a plain branch on ``row < row_end`` (the group's real end:
rows of the tile beyond it are padding or belong to other experts and are owned by their tiles) and the column bound;
for glu the chunks are the up positions (accumulator columns ``16q .. 16q + 7`` -> output columns ``8q ..``); an 8-bit
output stages 8-byte register vectors and 8-byte chunks (its staging rows are skewed by ``STAGE_PAD`` bytes). The direct
register -> global path (thread = row, one copy per fragment vector under the same branches) remains for the f32 oracle
outputs. Two facts of the lowering shape both paths: a predicated ``cute.copy`` lowers to element-wise stores, and a
dynamic row stride needs ``cute.assume(divby)`` or the vector copies scalarize. Either path moves the same converted
bytes: the chain is computed by the same thread from the same accumulator value in the same fp32 sequence.
"""

from __future__ import annotations

from dataclasses import dataclass
import dataclasses
from typing import List, Optional, Tuple

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass import Boolean, Float32, Int32, Int64, Uint8, Uint32
from cutlass._mlir.dialects import arith
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.cute.nvgpu import tcgen05
from cutlass.utils.blackwell_helpers import get_tmem_load_op
from cutlass.utils.gemm.sm100 import transform_partitioned_tensor_layout

from ....batch_invariant.fp_interface.cute import ex2_approx_ftz, rcp_approx_ftz
from ....batch_invariant.fp_interface.exact import f32_bits
from ....batch_invariant.quantization.cute_ext import CuteF32, bitcast_i32_to_f32
from ....batch_invariant.quantization.cvt_fp8 import (
    cvt_f32_to_e4m3,
    cvt_f32x2_to_fp8x2,
    cvt_f32x8_to_e2m1x8,
)
from ....gemm.batch_invariant.blocks.operands import ACC_DTYPE
from ....gemm.batch_invariant.spec import MXFP4_FMTS, NVFP4_FMTS
from .. import spec_nvfp4 as Q
from ..spec import (
    ACT_PARAMS,
    ACTS,
    UNGATED_ACTS,
    act_elem,
    glu_elem,
    hw_arith_applies,
)
from ..spec_fp8 import act_quant_pre, glu_quant_pre, scaled_out_elem, w8a8_alpha

GROUP = 8  # gate/up interleave granularity (columns): u x8, g x8, u x8, ...
KINDS = ("plain", "glu", "act")
OUT_FMTS = ("bf16", "f16", "f32")  # the bf16 classes' output formats
OUT_FMTS_FP8 = ("e4m3", "bf16", "f32")  # the fp8 per-tensor classes' output formats
#: The NVFP4 classes' output formats: the class's bf16, the fp32 oracle and ``nvfp4`` -- the static class's GEMM1, whose
#: epilogue quantizes the fp32 chain value per 16-column block (codes and scale bytes out).
OUT_FMTS_W4A4 = ("bf16", "f32", "nvfp4")
#: The scale sources of the W4A4 chains: ``w4a4`` (the row scale ``mul(alpha[e], pts[row])``), ``w4a4s`` (the static class's
#: GEMM2 scale ``mul(alpha[e], inv2[e])``, once per tile), ``none`` (the unscaled oracle).
W4A4_SCALES = ("none", "w4a4", "w4a4s")
#: The further kernel operands a chain may declare: the per-expert weight scale, the A operand's scale, the intermediate
#: multiplier, the per-row scales (at the padded positions), the per-expert auxiliary scales, the scale-byte output.
CHAIN_OPERANDS = ("alpha", "sa", "si", "pts", "aux", "out_sf")
STAGED_STORE = True  # the shared-memory staged, row-coalesced store of the 16-bit outputs (False: the direct path everywhere)
BAR_EPI_STAGE = (
    4  # named barrier ids BAR_EPI_STAGE + warpgroup (the dense blocks use 1 and 2)
)
BAR_EPI_JOIN = (
    3  # every epilogue warpgroup once, after its last TMEM load, before the TMEM free
)
STAGED_OUT_FMTS = (
    "bf16",
    "f16",
    "e4m3",
)  # the 16- and 8-bit outputs stage; the f32 oracle takes the direct path
# elements of row padding in the staging buffer: a 16-byte skew per row, so the TMEM lanes' (thread = row) 16-byte stores
# spread over the banks
STAGE_PAD = 8


def staged_store(out_fmt: str) -> bool:
    """Whether a grouped kernel with this output format stages its stores (its kernel then allocates the staging buffers)."""
    return STAGED_STORE and out_fmt in STAGED_OUT_FMTS


def store_vec_max(out_dt) -> int:
    """The widest register vector of the stores: 16 bytes, or 8 for an 8-bit output (its staging rows are skewed by
    ``STAGE_PAD`` bytes)."""
    return 8 if out_dt.width == 8 else 16 // (out_dt.width // 8)


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
class GroupedEpilogueBlock:
    """The epilogue block of the grouped bodies: what a chain object declares to the kernels, resolved at trace time.

    * ``in_fmt``: the operand format the chain belongs to (the kernels' operand block, :class:`.kernel.DenseOperands`, is
      keyed by it); ``out_fmt`` / ``pair`` / ``kind`` / ``act`` / ``params``: the chain itself; ``describe()`` names the
      compiled object;
    * ``operands``: the further operand tensors the kernel takes for the chain (a subset of :data:`CHAIN_OPERANDS`:
      ``alpha [E]`` the per-expert weight scale, ``sa [1]`` the A operand's scale, ``si [1]`` the intermediate multiplier --
      the fp8 classes' three --, ``pts`` the per-row scales at the padded positions, ``aux [E]`` the per-expert auxiliary
      scales, ``out_sf`` the scale-byte output); ``scale_operands``: the fp8 classes' three; ``scaled``: whether the fp8
      per-tile alpha ``mul(sa[0], alpha[e])`` is formed; ``paired_cvt``: whether two outputs convert per instruction into
      byte registers (the e4m3 output); ``per_row_scale``: whether the chain reads a per-row scale (``pts`` at the row's
      padded position); ``static_quant``: the fused block quantization (the codes to ``out``, the scale bytes to ``out_sf``);
    * ``tile_scale(fp, mAlpha, mSa, mAux, e)``: the chain's per-tile constant (``None`` for none); ``row_scale(fp, tile, mPts,
      p)``: the row's scale from the tile constant and the per-row scale at ``p``; ``inv(mSi)``: the intermediate multiplier
      read as given (the paired e4m3 conversion); ``static_consts(fp, mAux, e)``: the static block chain's per-tile constants;
    * ``elem(fp, acc, acc_gate, s, inv)``: one output element of the chain from the accumulator (and its gate partner for a
      paired chain) and the row's scale ``s``, every op through ``fp`` (the kernels' ``MoeF32``): converted for the 16- and
      32-bit outputs, the fp32 value before the paired conversion for the 8-bit one and before the block quantization for the
      ``nvfp4`` one.
    * ``quant``: whether the chain writes a second output (the block scales of a requantising chain) through its own
      epilogue loops (``grouped_loop`` / ``swapped_loop``: the loop functions the bodies call in place of the ones below)
      with its own shared storage (``smem_layout``: a layout of fp32 words, or None).

    A requantising epilogue block of another block-scaled class (its own operands, its chain and conversion) implements this
    surface; the loops below need nothing else of it."""

    kind: str
    act: str
    out_fmt: str
    params: Tuple[Optional[float], ...]

    @property
    def in_fmt(self) -> str:
        raise NotImplementedError

    @property
    def pair(self) -> bool:
        """Whether the chain pairs two accumulators (the gated chains: the (gate, up) pair)."""
        return self.kind == "glu"

    def describe(self) -> str:
        raise NotImplementedError

    @property
    def scale_operands(self) -> bool:
        return False

    @property
    def operands(self) -> Tuple[str, ...]:
        return ("alpha", "sa", "si") if self.scale_operands else ()

    @property
    def scaled(self) -> bool:
        return False

    @property
    def paired_cvt(self) -> bool:
        return False

    @property
    def dead_lane_select(self) -> bool:
        """Whether the grouped epilogue replaces a dead row's accumulators by a constant after the TMEM load (the bf16 and fp8
        chains: the division's slow path on garbage operands); the W4A4 chains leave the dead lanes as they are."""
        return True

    @property
    def per_row_scale(self) -> bool:
        return False

    @property
    def static_quant(self) -> bool:
        return False

    @property
    def needs_aux(self) -> bool:
        """Whether the chain reads the per-expert auxiliary scales (the static NVFP4 class's chains)."""
        return False

    def record_ok(self, knobs) -> bool:
        """The chain's own rule on a grouped record beyond the format's (the block-scale fp8 chains' promotion tiles)."""
        return True

    def swapped_record_ok(self, knobs) -> bool:
        """The chain's own rule on a swapped record beyond the format's."""
        return True

    def shape_ok(self, N: int, K: int, E: int) -> bool:
        """The chain's own operand rule beyond the format's alignment (whole scale blocks of a block-scale weight)."""
        return True

    def tile_scale(self, fp, mAlpha, mSa, mAux, e):
        return None

    def row_scale(self, fp, tile, mPts, p):
        return tile

    def inv(self, mSi):
        return None

    def static_consts(self, fp, mAux, e):
        raise NotImplementedError

    def elem(self, fp, acc, acc_gate, s=None, inv=None):
        raise NotImplementedError

    @property
    def quant(self) -> bool:
        """Whether the chain writes a second output (a requantising chain's block scales) through its own loops."""
        return False

    def smem_layout(self, epi_warpgroups: int, token_tile: Optional[int] = None):
        """The chain's own shared storage in the grouped body (``token_tile`` None) or the swapped one, as a layout of fp32
        words; None for a chain without one."""
        return None

    def stage_layout(self):
        """The chain's own staging rows for its stores in the grouped body, as a layout of bytes; None for a chain without."""
        return None

    def grouped_loop(self):
        """The chain's own grouped epilogue loop (a requantising chain), or None for :func:`grouped_epilogue_loop`."""
        return None

    def swapped_loop(self):
        """The chain's own swapped epilogue loop (a requantising chain), or None for the swapped body's shared loop."""
        return None


@dataclass(frozen=True)
class GroupedChain(GroupedEpilogueBlock):
    """The epilogue chain of a grouped GEMM of the bf16 classes: ``kind`` (``plain``, the gated ``glu`` on an accumulator
    pair, or the ungated ``act`` on one accumulator), the activation, the output format and the activation's parameters
    (binary32 values, ``None`` for an absent clamp / scale: ``spec.ACT_PARAMS``). ``describe()`` names it, parameters
    included (their bit patterns), so every parameter set is its own compiled kernel."""

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
    def in_fmt(self) -> str:
        return "bf16"

    @property
    def hw(self) -> bool:
        """Whether the chain's exponential and divisions run on the part's SFU: never for the bf16 classes (the declared
        sequences); the fp4 chains override it (:func:`fp4_chain_hw`)."""
        return False

    def elem(self, fp, acc, acc_gate, s=None, inv=None):
        """One output element, converted: the gated term of the (gate, up) accumulator pair, the ungated activation of
        one accumulator, or the plain conversion; every op in fp32 through ``fp`` (the chain's parameters are
        compile-time constants of the kernel)."""
        if self.pair:
            return glu_elem(
                fp, acc_gate, acc, self.act, self.out_fmt, self.params, self.hw
            )
        if self.kind == "act":
            return act_elem(fp, acc, self.act, self.out_fmt, self.params, self.hw)
        return fp.cvt_out(acc, self.out_fmt)

    def describe(self) -> str:
        suffix = "".join(
            "_pn" if v is None else f"_p{f32_bits(v):08x}" for v in self.params
        ) + ("_hw" if self.hw else "")
        if self.kind == "act":
            return (
                f"{self.act}_{self.out_fmt}{suffix}"  # the ungated chains: `relu2_bf16`
            )
        if self.act == "silu":  # the bf16 class's names
            return ("swiglu" if self.pair else "plain") + f"_{self.out_fmt}{suffix}"
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


@dataclass(frozen=True)
class Fp8Chain(GroupedEpilogueBlock):
    """The epilogue chain of a grouped GEMM of the fp8 per-tensor classes: ``kind`` (``glu`` on the (gate, up) pair, ``act``
    on one accumulator, or ``plain``), the activation and its binary32 parameters, whether the class's alpha scales the
    accumulators (``scaled``: the W8A8 chains; unscaled ``plain`` / ``f32`` is the raw-accumulator oracle) and the output
    format. The kernels of these chains take the three scale operands. ``describe()`` names it, parameters included, so
    every parameter set is its own compiled kernel."""

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
    def in_fmt(self) -> str:
        return "e4m3"

    @property
    def scale_operands(self) -> bool:
        return True

    @property
    def paired_cvt(self) -> bool:
        return self.out_fmt == "e4m3"

    def tile_scale(self, fp, mAlpha, mSa, mAux, e):
        """The tile's alpha ``mul(sa[0], alpha[e])``: one ``mul.rn`` per tile (the unscaled oracle chain forms none)."""
        return w8a8_alpha(fp, mSa[0], mAlpha[e]) if self.scaled else None

    def inv(self, mSi):
        return mSi[0]

    def elem(self, fp, acc, acc_gate, s=None, inv=None):
        """One output element in fp32 (before the paired e4m3 conversion) for the e4m3 outputs, converted for the others;
        every op through ``fp``; ``s`` the tile's alpha."""
        if self.pair:
            return fp.mul(
                glu_quant_pre(fp, acc_gate, acc, s, self.act, self.params), inv
            )
        if self.kind == "act":
            return fp.mul(act_quant_pre(fp, acc, s, self.act, self.params), inv)
        if self.scaled:
            return scaled_out_elem(fp, acc, s, self.out_fmt)
        return fp.cvt_out(acc, self.out_fmt)

    def describe(self) -> str:
        suffix = "".join(
            "_pn" if v is None else f"_p{f32_bits(v):08x}" for v in self.params
        )
        scale = "w8a8" if self.scaled else "raw"
        return f"{self.kind}_{self.act}_{scale}_{self.out_fmt}{suffix}"


#: GEMM2 of the W8A8 classes: the scaled accumulator converted to bf16.
GEMM2_CHAIN = Fp8Chain("plain", "silu", "bf16", (), True)


def fp4_chain_hw(chain: GroupedChain) -> bool:
    """The fp4 classes' arithmetic: a gated or ungated chain with an exponential or a division computes them on the part's
    SFU (``spec.hw_arith_applies``); a plain conversion never does."""
    return chain.kind != "plain" and hw_arith_applies(chain.act, chain.kind == "glu")


@dataclass(frozen=True)
class W4A4Chain(GroupedChain):
    """A grouped chain of the NVFP4 classes: the bf16 class's chain with the W4A4 scale source ``scale`` (:data:`W4A4_SCALES`)
    on the instruction class ``in_fmt`` (a key of the block-scaled plans: the sm_100 / sm_103 step or the sm_107 one; the
    kernel name carries it, ``describe()`` does not); the output format ``nvfp4`` is the static class's GEMM1 (the fused block
    quantization). ``describe()`` keys the kernel cache; the scaled chains carry their tag so they never share an object with
    the bf16 class's chains."""

    scale: str = "w4a4"
    in_fmt: str = "nvfp4"

    def __post_init__(self):
        assert self.scale in W4A4_SCALES and self.in_fmt in NVFP4_FMTS, self
        if self.out_fmt == "nvfp4":
            # the static class's GEMM1: validated as the bf16 chain of the same kind, then the output format replaced
            object.__setattr__(self, "out_fmt", "bf16")
            super().__post_init__()
            object.__setattr__(self, "out_fmt", "nvfp4")
            assert self.kind in ("glu", "act") and self.scale == "w4a4", self
        else:
            super().__post_init__()
        if self.scale == "w4a4s":
            assert self.kind == "plain" and self.out_fmt == "bf16", self

    def at(self, in_fmt: str) -> "W4A4Chain":
        """The same chain on another instruction class."""
        return dataclasses.replace(self, in_fmt=in_fmt)

    @property
    def hw(self) -> bool:
        return fp4_chain_hw(self)

    @property
    def static_quant(self) -> bool:
        """The static class's GEMM1: the fused block quantization of the fp32 chain value."""
        return self.out_fmt == "nvfp4"

    @property
    def needs_aux(self) -> bool:
        """The per-expert static scales the chain reads (``gs2[e]`` for the fused quantization, ``inv2[e]`` for GEMM2)."""
        return self.out_fmt == "nvfp4" or self.scale == "w4a4s"

    @property
    def dead_lane_select(self) -> bool:
        return False

    @property
    def per_row_scale(self) -> bool:
        return self.scale == "w4a4"

    @property
    def operands(self) -> Tuple[str, ...]:
        if self.scale == "none":
            return ()
        if self.scale == "w4a4s":
            return ("alpha", "aux")
        return (
            ("alpha", "pts", "aux", "out_sf") if self.static_quant else ("alpha", "pts")
        )

    def tile_scale(self, fp, mAlpha, mSa, mAux, e):
        """The tile's expert constant: the alpha as given (``w4a4``: the row scale multiplies it), or the static class's GEMM2
        scale ``mul(alpha2[e], inv2[e])``."""
        if self.scale == "w4a4s":
            return Q.w4a4_row_scale(fp, mAlpha[e], mAux[e])
        if self.scale == "w4a4":
            return mAlpha[e]
        return None

    def row_scale(self, fp, tile, mPts, p):
        """``s = mul(alpha_e, pts[p])``: the row's scale from the tile's alpha and the per-row scale at the padded position."""
        return Q.w4a4_row_scale(fp, tile, mPts[p])

    def static_consts(self, fp, mAux, e):
        """The expert's static intermediate scale as the block chain reads it: ``mul(gs, 1/6)`` and ``div(1, gs)`` once per tile
        (the block chain's own ops)."""
        gs_e = mAux[e]
        return fp.mul(gs_e, fp.const(Q.NVFP4_C_SIXTH)), fp.div(fp.const(1.0), gs_e)

    def elem(self, fp, acc, acc_gate, s=None, inv=None):
        """One output element of the W4A4 chain: the gated term of the scaled (gate, up) pair, the ungated activation of the
        scaled accumulator, or GEMM2's ``cvt(mul(acc, s))``; the unscaled chains are the bf16 class's; the static class's GEMM1
        returns the fp32 value before the block quantization (the loop forms the blocks)."""
        if self.scale == "none":
            return GroupedChain.elem(self, fp, acc, acc_gate)
        if self.out_fmt == "nvfp4":
            if self.pair:
                return Q.glu_scaled_pre(fp, acc_gate, acc, s, self.act, self.params)
            return Q.act_scaled_pre(fp, acc, s, self.act, self.params)
        if self.pair:
            return Q.glu_scaled_elem(
                fp, acc_gate, acc, s, self.act, self.out_fmt, self.params
            )
        if self.kind == "act":
            return Q.act_scaled_elem(fp, acc, s, self.act, self.out_fmt, self.params)
        return Q.scaled_out_elem(fp, acc, s, self.out_fmt)

    def describe(self) -> str:
        base = (
            super().describe()
        )  # the output format names the static class's GEMM1 (`..._nvfp4`)
        return base if self.scale == "none" else f"{self.scale}_{base}"


@dataclass(frozen=True)
class Mxfp4Chain(GroupedChain):
    """A grouped chain of the MXFP4 classes (``moe.mxfp4.<act>``: MXFP4 experts x MXFP4 activations, the e8m0 block scales of
    both operands applied inside the instruction, no global scale): the bf16 class's chain on the raw accumulators of the
    dense MX class -- GEMM1's gated / ungated activation to bf16 (the intermediate the ``mxfp4.quant`` class then quantizes
    per row), GEMM2's plain conversion -- on the instruction class ``in_fmt`` (a key of the block-scaled plans; the kernel
    name carries it, ``describe()`` does not). The chain declares no further operand; ``describe()`` keys the kernel cache
    apart from the bf16 class's chains."""

    in_fmt: str = "mxfp4"

    def __post_init__(self):
        super().__post_init__()
        assert self.in_fmt in MXFP4_FMTS, self

    def at(self, in_fmt: str) -> "Mxfp4Chain":
        """The same chain on another instruction class."""
        return dataclasses.replace(self, in_fmt=in_fmt)

    @property
    def hw(self) -> bool:
        return fp4_chain_hw(self)

    def describe(self) -> str:
        return "mxfp4_" + super().describe()


#: GEMM2 of the MXFP4 classes: the accumulator converted to bf16 (the dense MX class's chain at a unit alpha).
MXFP4_GEMM2_CHAIN = Mxfp4Chain("plain", "silu", "bf16", ())


def static_block_plan(offs: List[Tuple[int, int]], pair: bool) -> List[List[int]]:
    """The static class's blocks of one thread's fragment: for every (row, 16-output block) the 16 registers in output-column
    order (the gated chain's up positions ``16q + r`` -> output ``8q + r``; the ungated chain's every column). Raises when the
    fragment does not hold whole blocks (such a record is not in the static class's space)."""
    key = (lambda n: (n // (2 * GROUP)) * GROUP + n % GROUP) if pair else (lambda n: n)
    groups: dict = {}
    for v, (m, n) in enumerate(offs):
        if pair and n % (2 * GROUP) >= GROUP:
            continue
        groups.setdefault((m, key(n) // Q.NVFP4_BLOCK), []).append(
            (key(n) % Q.NVFP4_BLOCK, v)
        )
    blocks = []
    for (m, b), lst in sorted(groups.items()):
        if sorted(pos for pos, _ in lst) != list(range(Q.NVFP4_BLOCK)):
            raise ValueError(
                f"the fragment holds a partial 16-output block at row {m}, block {b}: {offs}"
            )
        blocks.append([v for _, v in sorted(lst)])
    return blocks


@cute.jit
def sf_byte_offset(row: Int32, col: Int32, pcols: Int32) -> Int64:
    """The byte of block column ``col`` of row ``row`` in the 128x4 swizzled scale-factor layout (``sf_offset_128x4`` on device
    values)."""
    r = row.to(Int64)
    r128 = (r % 128).to(Int32)
    inner = (
        (col % Int32(4))
        + (col // Int32(4)) * Int32(512)
        + (r128 % 32) * 16
        + (r128 // 32) * 4
    )
    return (r // 128) * (Int32(128) * pcols).to(Int64) + inner.to(Int64)


@cute.jit
def static_block_quant(vals, gs6, inv_gs):
    """One block of the static class's GEMM1 output from its 16 fp32 values: ``(sf_byte, w0, w1)`` -- the e4m3 scale byte
    (``cvt(mul(amax, mul(gs, 1/6)))``, ``gs6 = mul(gs, 1/6)`` formed once per tile) and the 16 e2m1 codes as two little-endian
    words (``cvt(mul(v, inv))``, ``inv = fmin(div(1, mul(sf_d, div(1, gs))), FLT_MAX)``, ``inv_gs = div(1, gs)`` formed once per
    tile): the per-tensor quantizer's block chain, every op in fp32 through ``MoeF32``."""
    fp = MoeF32
    amax = Q.nvfp4_block_amax(fp, vals)
    sf_byte, sf_d = cvt_f32_to_e4m3(fp.mul(amax, gs6))
    inv = fp.fmin(
        fp.div(fp.const(1.0), fp.mul(sf_d, inv_gs)), fp.const(3.4028234663852886e38)
    )
    ys = [fp.mul(v, inv) for v in vals]
    return sf_byte, cvt_f32x8_to_e2m1x8(*ys[:8]), cvt_f32x8_to_e2m1x8(*ys[8:])


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
    operations on the binary32 pattern, no floating-point rounding; and the two SFU primitives of the fp4 classes' chains
    (``ex2.approx.ftz`` / ``rcp.approx.ftz``, pinned as inline asm)."""

    @staticmethod
    def ex2(a):
        return ex2_approx_ftz(a)

    @staticmethod
    def rcp(a):
        return rcp_approx_ftz(a)

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
    mAlpha: cute.Tensor = None,
    mSa: cute.Tensor = None,
    mSi: cute.Tensor = None,
    mPts: cute.Tensor = None,
    mAux: cute.Tensor = None,
    mOutSf: cute.Tensor = None,
    s_off: cute.Tensor = None,
    s_off_pad: cute.Tensor = None,
):
    """``chain``: the epilogue block (:class:`GroupedEpilogueBlock`); ``sC``: the staging buffers (``staging_layout(out,
    epi_tile, epi_warpgroups)``) of the staged store, or None; ``cta_v``: this CTA's index inside its MMA (0, or 0 / 1 for
    the 2-CTA instructions), which with ``mma_m_in_cluster`` locates its rows in the cluster tile; ``mAlpha`` / ``mSa`` /
    ``mSi``: the scale operands of a chain with ``scale_operands`` (the per-expert weight dequantization scale ``s_w[e]``,
    the A operand's scale, the intermediate multiplier read as given), None otherwise; ``mPts`` / ``mAux`` / ``mOutSf``: the
    block-scaled chains' operands (the per-row scales at the padded positions, the per-expert auxiliary scales, the
    scale-byte output of the static class's GEMM1), with ``s_off`` / ``s_off_pad`` the dense and the padded group offsets of
    the dense packing (the row scale of output row ``p`` of expert ``e`` sits at ``p + off_pad[e] - off[e]``; the padded
    packing passes ``s_off_pad = None``)."""
    pair = chain.pair
    paired_cvt = (
        chain.paired_cvt
    )  # the 8-bit output: two values per conversion into byte registers
    static_q = chain.static_quant  # the fused block quantization: codes and scale bytes from the thread's own registers
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
    ve, partner, is_out = store_plan(offs, pair, store_vec_max(out_dt))
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
    # the staged, row-coalesced store (the 8- and 16-bit outputs, 128-row layouts: thread = TMEM lane = tile row)
    staged = sC is not None and cta_v is not None and out_dt.width in (8, 16)
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
        )  # one chunk of GROUP elements
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
    inv = None
    if cutlass.const_expr(paired_cvt):
        inv = chain.inv(mSi)  # the intermediate multiplier, as given
    blocks = []
    mOut32 = None
    n_blocks = pcols = Int32(0)
    if cutlass.const_expr(static_q):
        assert len(rows_in_frag) == 1 and not staged, (
            "the static class's fused quantization takes the 128-row layouts (thread = row), unstaged"
        )
        tTR_cF_all = thr_t2r.partition_D(
            cF_epi
        )  # the flat (row, accumulator column) coordinates
        blocks = static_block_plan(offs, pair)
        N_out = N // 2 if pair else N
        n_blocks = N_out // Int32(Q.NVFP4_BLOCK)  # scale blocks per output row
        pcols = ((n_blocks + Int32(3)) // Int32(4)) * Int32(
            4
        )  # scale columns of the 128x4 buffer
        ld32 = cute.assume(
            ld // Int32(4), divby=4
        )  # the code row in 32-bit words (16-byte rows)
        mOut32 = cute.make_tensor(
            cute.recast_ptr(mOut.iterator, dtype=Uint32),
            cute.make_layout((Tk, n_blocks * Int32(2)), stride=(ld32, 1)),
        )
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
        pts_shift = None
        if cutlass.const_expr(chain.per_row_scale):
            pts_shift = Int32(0)
            if cutlass.const_expr(s_off_pad is not None):
                pts_shift = (
                    s_off_pad[e] - s_off[e]
                )  # dense A: the row scales sit at the padded positions
        tile_sched.advance_to_next_work()
        work_tile = tile_sched.get_current_work()
        rows_left = row_end - row0
        alpha_e = chain.tile_scale(
            MoeF32, mAlpha, mSa, mAux, e
        )  # the tile's constant (the fp8 alpha: one mul.rn per tile; the W4A4 alpha as given)
        gs6 = inv_gs = None
        if cutlass.const_expr(static_q):
            gs6, inv_gs = chain.static_consts(MoeF32, mAux, e)
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
        if cutlass.const_expr(static_q):
            tTR_cF_tile = tTR_cF_all[
                (None, None, None, None, None, mma_m_in_cluster, n_tile, 0)
            ]
            tTR_cF_tile = cute.group_modes(tTR_cF_tile, 3, cute.rank(tTR_cF_tile))
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
                if cutlass.const_expr(chain.dead_lane_select):
                    for v in cutlass.range_constexpr(nv):
                        tTR_rAcc[v] = select_f32(
                            live[offs[v][0]], tTR_rAcc[v], Float32(1.0)
                        )
                # the row's scale once per distinct row of the fragment: the tile's constant, or for a per-row chain
                # s = mul(alpha_e, pts[row]) at the row's padded position (a row past the group's end reads a valid entry and
                # stores nothing)
                row_scale = {m: alpha_e for m in rows_in_frag}
                if cutlass.const_expr(chain.per_row_scale):
                    for m in rows_in_frag:
                        p = row0 + pts_shift + (crd0[0] - Int32(offs[0][0]) + Int32(m))
                        p = cutlass.min(
                            cutlass.max(p, Int32(0)), mPts.shape[0] - Int32(1)
                        )
                        row_scale[m] = chain.row_scale(MoeF32, alpha_e, mPts, p)
                # the chain, in fp32, per element (pair)
                if cutlass.const_expr(static_q):
                    # the static class: the fp32 chain values, then every 16-output block of the row quantized from this thread's
                    # registers; 8 code bytes to the packed-code row, the scale byte to the (padded) row's 128x4 position
                    crdF0 = tTR_cF_tile[(None, None, None, s)][(0, 0, 0)]
                    pre = [None] * nv
                    for v in cutlass.range_constexpr(nv):
                        if cutlass.const_expr(is_out[v]):
                            pre[v] = chain.elem(
                                MoeF32,
                                tTR_rAcc[v],
                                tTR_rAcc[partner[v]]
                                if cutlass.const_expr(pair)
                                else None,
                                row_scale[offs[v][0]],
                                inv,
                            )
                    for blk in blocks:
                        rrow = crdF0[0] - Int32(offs[0][0]) + Int32(offs[blk[0]][0])
                        acc_col = crdF0[1] - Int32(offs[0][1]) + Int32(offs[blk[0]][1])
                        out_col = (
                            (acc_col // Int32(2 * GROUP)) * Int32(GROUP)
                            + acc_col % Int32(GROUP)
                            if cutlass.const_expr(pair)
                            else acc_col
                        )
                        bcol = out_col // Int32(Q.NVFP4_BLOCK)
                        if (rrow < rows_left) & (bcol < n_blocks):
                            sf_byte, w0, w1 = static_block_quant(
                                [pre[v] for v in blk], gs6, inv_gs
                            )
                            prow = row0 + rrow
                            mOut32[(prow, Int32(2) * bcol)] = w0
                            mOut32[(prow, Int32(2) * bcol + Int32(1))] = w1
                            mOutSf[sf_byte_offset(prow + pts_shift, bcol, pcols)] = (
                                sf_byte.to(Uint8)
                            )
                elif cutlass.const_expr(paired_cvt):
                    # two outputs per cvt.rn.satfinite.e4m3x2.f32 (any two output positions: each byte goes to its own register)
                    outs = [v for v in range(nv) if is_out[v]]
                    assert len(outs) % 2 == 0, offs
                    for q in cutlass.range_constexpr(0, len(outs), 2):
                        v0, v1 = outs[q], outs[q + 1]
                        g0 = tTR_rAcc[partner[v0]] if cutlass.const_expr(pair) else None
                        g1 = tTR_rAcc[partner[v1]] if cutlass.const_expr(pair) else None
                        y0 = chain.elem(
                            MoeF32, tTR_rAcc[v0], g0, row_scale[offs[v0][0]], inv
                        )
                        y1 = chain.elem(
                            MoeF32, tTR_rAcc[v1], g1, row_scale[offs[v1][0]], inv
                        )
                        pr = cvt_f32x2_to_fp8x2(y0, y1, "e4m3")
                        tTR_rC[v0] = (pr & 0xFF).to(Uint8)
                        tTR_rC[v1] = (pr >> 8).to(Uint8)
                else:
                    for v in cutlass.range_constexpr(nv):
                        if cutlass.const_expr(is_out[v]):
                            tTR_rC[v] = chain.elem(
                                MoeF32,
                                tTR_rAcc[v],
                                tTR_rAcc[partner[v]]
                                if cutlass.const_expr(pair)
                                else None,
                                row_scale[offs[v][0]],
                                inv,
                            )
                if cutlass.const_expr(staged):
                    # registers -> this warpgroup's staging buffer (the TMEM load's thread layout); the rows of a skipped warp
                    # (all past the group's end) stay stale and are predicated off below
                    cute.copy(tiled_copy_r2s, tRS_rC, tRS_sC[(None, None, None, wg)])
                elif cutlass.const_expr(not static_q):
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
