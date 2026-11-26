# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The block-scale fp8 MoE classes ``moe.e4m3bs.<act>`` (DeepSeek-style fp8: e4m3 experts with one fp32 scale per 128x128
block, e4m3 activations with one fp32 scale per 1x128 block) and ``moe.e4m3bs32.<act>`` (the same recipe on 32x32 weight
blocks and 1x32 activation blocks: ``deepseek_fp8_block_size=32``, the re-blocked TP shards prims-TS serves), and the
quantizer class ``fp8bs.quant`` they are built on (at 128 or 32 elements per scale).

The 32-wide class is the 128-wide class with the block as a parameter: the same chains, bodies and record spaces; its GEMM
core is ``gemm.e4m3.kblock32`` (:func:`reduction_kblock`: ONE ``mma32`` step per block, promoted per step; a different term from
``gemm.e4m3.kblock128`` at the same operands, so a different class and fingerprint), its quantizer the radix-32 instance of the
block digit. The dense block-scale GEMM module keeps the 128-wide class alone; the 32-wide core's term and fingerprint are
declared here (the MoE package's own).

The routing, the activations (:func:`.spec.glu_elem` / :func:`.spec.act_elem`: every typed activation of the layer) and
the combine are the bf16 classes' (:mod:`.spec`). The GEMM folds are the dense block-scale class's
(:mod:`flashinfer.gemm.batch_invariant.spec`: ``gemm.e4m3.kblock128``, the per-tensor fp8 ``mma32`` step folded per
128-element K block and promoted in block order as ``fma.rn(mul.rn(s_a, s_b), P_kb, acc)``) at ``K = H`` and ``K = I``; the
grouped GEMM is a placement of that class. The block scales are the fold's own operands, so GEMM1 is the bf16 class's chain
on the promoted accumulators as they stand (no alpha, no per-token scale, no static scale) and GEMM2 the plain ``cvt_bf16``.

The quantizer class ``fp8bs.quant`` (the activations per token, the intermediate per expanded row, the expert weights per
128x128 block at preparation) is the canon engine's block digit and three chains, declared once here and evaluated by every
placement (the symbolic terms, the exact and the vectorised CPU references, the standalone kernel, the fused GEMM1 epilogue):

* ``amax = fmaxc-tree(fabs(x_e))``, ``e`` in 0..127: one ``pairwise`` digit of radix 128 (:func:`block_reduction`), placed
  ``[(V, reg), (128 / V, lane)]`` by whoever computes it (:func:`block_amax_placement`);
* ``s = fmaxc(div(amax, 448), fp32(1e-12))`` (:func:`scale_term`): the fp32 dequantization scale the GEMM class reads (the
  floor gives a zero block a finite scale: FlashInfer's recipe ``(amax / 448).clamp(1e-12)``);
* ``inv = div(1, s)`` (:func:`inv_term`): the correctly rounded reciprocal, formed once per block;
* ``q = cvt.rn.satfinite.e4m3(mul(x, inv))`` (:func:`quant_elem`): one multiply and the declared saturating conversion per
  element (FlashInfer's torch recipe divides by ``s``; the class multiplies by the reciprocal: one term in every placement).

Every S-expression this module builds is the reference model's (the Lean cases compare them textually:
``lean/BIKernels/FP8BlockQuant.lean`` for the quantizer, ``MoEFp8Bs.lean`` for the class).
"""

from __future__ import annotations

import hashlib
from typing import Any, Callable, Dict, List, Optional, Sequence

from ...batch_invariant.canon import Digit, Placement, Reduction
from ...batch_invariant.fp_interface import exact as fp32
from ...batch_invariant.quantization.qterm import Term as QTerm
from ...batch_invariant.quantization.qterm import TermAlgebra as QTermAlgebra
from ...batch_invariant.quantization.qterm import sexpr as qsexpr
from ...gemm.batch_invariant.spec import spec_fingerprint as gemm_fingerprint
from .mterm import Term, TermAlgebra, fingerprint
from .spec import ACTS, SOFTMAX_RENORM, UNGATED_ACTS, MoESpec, RoutingMethod

CLASS = "moe.e4m3bs"
IN_FMT = "e4m3_bs"
#: Elements per fp32 scale of the 128-wide class (FlashInfer's ``scale_granularity_mnk = (1, 128, 128)``).
BLOCK = 128
#: The block sizes of the two classes (``deepseek_fp8_block_size``).
BLOCKS = (128, 32)
#: The format key, class name and GEMM core per block size.
IN_FMT_OF = {128: "e4m3_bs", 32: "e4m3_bs32"}
CLASS_OF = {128: "moe.e4m3bs", 32: "moe.e4m3bs32"}
GEMM_CLASS_OF = {128: "gemm.e4m3.kblock128", 32: "gemm.e4m3.kblock32"}
#: One fp8 smem stage of the bodies: the 128-wide block, or four 32-wide blocks.
STAGE_K = 128
#: The largest finite e4m3 magnitude: the scale maps the block amax onto it.
E4M3_MAX = 448.0
#: The floor of a block's scale (a zero block quantizes to zeros under a finite scale).
SCALE_EPS = fp32.to_f32(1e-12)
AMAX_IDENTITY_NOTE = (
    "fmaxc(v, +0) == v for every v >= +0; every leaf is fabs(x) >= +0 and never -0"
)
PLACEMENT_VS = (128, 64, 32, 16, 8, 4)


# ----------------------------------------------------------------------------- the reduction
def leaf_value(fp: Any, x: Any) -> Any:
    """The reduction leaf: ``fabs(x)`` of the (exactly widened) input element."""
    return fp.fabs(x)


def check_block(block: int) -> int:
    if block not in BLOCKS:
        raise ValueError(
            f"fp8bs.quant: deepseek_fp8_block_size must be one of {BLOCKS}, got {block}"
        )
    return block


def elem_digit(block: int = BLOCK) -> Digit:
    return Digit(
        "elem",
        check_block(block),
        "pairwise",
        combine=lambda fp, a, b: fp.fmaxc(a, b),
        identity_note=AMAX_IDENTITY_NOTE,
    )


def block_reduction(leaf_x: Callable, block: int = BLOCK) -> Reduction:
    """``amax = fmaxc-tree over fabs(x_e)``, ``e`` in 0..block-1; ``leaf_x(fp, e)`` is element ``e`` of the block as an f32 term /
    value (the caller fixes the row and the block)."""
    return Reduction(
        "fp8bs.block_amax",
        (elem_digit(block),),
        lambda fp, idx: leaf_value(fp, leaf_x(fp, idx["elem"])),
    )


def block_amax_placement(V: int, block: int = BLOCK) -> Placement:
    """The placement of the ``elem`` digit a thread holding ``V`` consecutive elements of a block uses: ``[(V, reg),
    (block / V, lane)]`` (the ``block / V`` lanes of a block are adjacent lanes; the block amax is finished by ``log2(block / V)``
    xor shuffles). ``V`` in ``PLACEMENT_VS`` up to the block: a whole block per thread (the fused epilogue's half-block form is
    the register tree over 64 leaves, whose combine with the other half's is the block's tree) down to four elements per lane."""
    if V not in PLACEMENT_VS or check_block(block) < V:
        raise ValueError(
            f"fp8bs.quant: no placement with {V} elements per lane at block {block}"
        )
    return Placement.make({"elem": [(V, "reg"), (block // V, "lane")]})


# ----------------------------------------------------------------------------- the chains
def scale_term(fp: Any, amax: Any) -> Any:
    """``s = fmaxc(div(amax, 448), 1e-12)``: the block's fp32 dequantization scale (the ``max.f32`` the kernels issue is
    commutative; the canonical S-expression sorts its operands)."""
    return fp.fmaxc(fp.div(amax, fp.const(E4M3_MAX)), fp.const(SCALE_EPS))


def inv_term(fp: Any, s: Any) -> Any:
    """``inv = div(1, s)``: the correctly rounded reciprocal of the block scale, formed once per block."""
    return fp.div(fp.const(1.0), s)


def quant_pre(fp: Any, x: Any, inv: Any) -> Any:
    """``mul(x, inv)``: the value the e4m3 conversion rounds (the fp32 part of :func:`quant_elem`)."""
    return fp.mul(x, inv)


def quant_elem(fp: Any, x: Any, inv: Any) -> Any:
    """``q = cvt.rn.satfinite.e4m3(mul(x, inv))`` as the element value (the caller packs the code)."""
    return fp.cvt_out(quant_pre(fp, x, inv), "e4m3")


def block_chain(fp: Any, amax: Any):
    """``(s, inv)`` of a block from its amax."""
    s = scale_term(fp, amax)
    return s, inv_term(fp, s)


# ----------------------------------------------------------------------------- the quantizer class
class BSQuantSpec:
    """Symbolic (term) view of one row of the class ``fp8bs.quant``: ``x(row, i)`` is the input element widened to f32 (exact
    for bf16 / f16, the identity for fp32: one class per input dtype family, not per dtype). A 128x128 weight block is the
    same chain on a block whose 16384 leaves are read row-major (the max of non-negative values is exact in every order,
    so the 2-D block's tree is the row-major pairwise tree by declaration)."""

    def __init__(self, K: int, alg: Optional[QTermAlgebra] = None, block: int = BLOCK):
        check_block(block)
        if K % block:
            raise ValueError(
                f"fp8bs.quant: K = {K} is not a whole number of {block}-element blocks"
            )
        self.K, self.block_size = K, block
        self.nb = K // block
        self.alg = alg or QTermAlgebra()

    def leaf(self, row: int, i: int) -> QTerm:
        return self.alg.x(row, i)

    def block_reduction(self, row: int, b: int) -> Reduction:
        return block_reduction(
            lambda fp, e: self.leaf(row, self.block_size * b + e), self.block_size
        )

    def block_amax(self, row: int, b: int) -> QTerm:
        return self.block_reduction(row, b).canonical(self.alg)

    def block(self, row: int, b: int) -> Dict[str, Any]:
        a = self.alg
        amax = self.block_amax(row, b)
        s, inv = block_chain(a, amax)
        codes = [
            quant_elem(a, self.leaf(row, self.block_size * b + e), inv)
            for e in range(self.block_size)
        ]
        return {"amax": amax, "s": s, "inv": inv, "codes": codes}

    def outputs(self, row: int) -> Dict[str, List[QTerm]]:
        """``{"codes": [K element-value terms], "s": [nb fp32 scale terms]}``."""
        codes: List[QTerm] = []
        ss: List[QTerm] = []
        for b in range(self.nb):
            blk = self.block(row, b)
            codes += blk["codes"]
            ss.append(blk["s"])
        return {"codes": codes, "s": ss}


def quant_fingerprint(block: int = BLOCK) -> str:
    """Identifier of the numerical specification of the class ``fp8bs.quant`` at ``block`` elements per scale: the structural
    hash of block 0's scale term and its first and last code terms. Independent of K, of the input dtype and of the
    scale-factor layout (a block's bits depend on its ``block`` inputs alone; the layout -- the ``K`` or the ``MN`` arrangement
    of the fp32 scales -- is a placement of the store); the block size names the instance (the 32-wide quantizer is another
    class)."""
    s = BSQuantSpec(block, block=block)
    blk = s.block(0, 0)
    parts = [
        "fp8bs.quant",
        str(block),
        qsexpr(blk["s"]),
        qsexpr(blk["codes"][0]),
        qsexpr(blk["codes"][block - 1]),
    ]
    return "fp8bsq-" + hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


# ----------------------------------------------------------------------------- the 32-wide GEMM core
def reduction_kblock(K: int, m: int, n: int, block: int = BLOCK):
    """The block-scale GEMM core at ``block`` elements per scale block for output element (m, n): the dense class
    ``gemm.e4m3.kblock128`` (:func:`flashinfer.gemm.batch_invariant.spec.reduction_blockscale`) at 128, and ``gemm.e4m3.kblock32``
    at 32 -- the same fp8 ``mma32`` step, K cut into ``K / 32`` blocks of ONE step each from +0 (its partial ``P_kb``), the
    partials folded in block order from +0 with ``acc = fma.rn(mul.rn(s_a[m, kb], s_b[n // 32, kb]), P_kb, acc)``. The MoE bodies
    compute exactly this at the class (one accumulator stage per instruction step); the 32-wide core is declared here, the
    dense module keeps the 128-wide class."""
    from ...gemm.batch_invariant.spec import (
        MMA_K_OF,
        MMA_OP_OF,
        kslice_leaf,
        reduction_blockscale,
    )

    check_block(block)
    if block == BLOCK:
        return reduction_blockscale(K, m, n, IN_FMT)
    if K % block:
        raise ValueError(
            f"gemm.e4m3.kblock32: K = {K} is not a whole number of {block}-element blocks"
        )
    op, mk = MMA_OP_OF[IN_FMT], MMA_K_OF[IN_FMT]
    spb = block // mk
    kstep = Digit(
        "kstep",
        spb,
        "fold",
        init=lambda fp: fp.const(0.0),
        step=lambda fp, acc, tile: fp.op(op, acc, tile),
        identity_note="K is a whole number of blocks: no padded step",
    )
    blk = Digit(
        "block",
        K // block,
        "fold",
        init=lambda fp: fp.const(0.0),
        step=lambda fp, acc, partial, kb: fp.fma(
            fp.mul(fp.op("sfa_blk", m, kb), fp.op("sfb_blk", n // block, kb)),
            partial,
            acc,
        ),
        identity_note="fma(s, +0, acc) == acc for finite s: a block of zero operands adds nothing (the scales are finite inputs)",
        indexed=True,
    )
    slice_leaf = kslice_leaf(m, n, IN_FMT)

    def leaf(fp, idx):
        return slice_leaf(fp, {"kstep": idx["block"] * spb + idx["kstep"]})

    return Reduction(f"{GEMM_CLASS_OF[block]}[K={K}]", (kstep, blk), leaf)


def gemm_core_fingerprint(K: int, out_fmt: str = "f32", block: int = BLOCK) -> str:
    """The fingerprint of the GEMM core at ``block``: the dense class's own at 128 (``gemm-e4m3-kblock128-<hash>``), and
    ``gemm-e4m3-kblock32-<hash>`` at 32 over the same parts (K, the formats, the output terms at (0, 0), (1, 1) and a column of
    the second weight scale block)."""
    from ...batch_invariant.fp_interface.term import TermAlgebra as GTermAlgebra
    from ...batch_invariant.fp_interface.term import fingerprint as gfingerprint
    from ...gemm.batch_invariant.spec import output_elem

    check_block(block)
    if block == BLOCK:
        return gemm_fingerprint(K, out_fmt, 1, IN_FMT)
    alg = GTermAlgebra()

    def out(m, n):
        return output_elem(
            alg, reduction_kblock(K, m, n, block).canonical(alg), out_fmt
        )

    parts = [
        f"K={K}",
        "chunks=1",
        f"in={IN_FMT_OF[block]}",
        f"out={out_fmt}",
        "scaled=0",
        gfingerprint(out(0, 0)),
        gfingerprint(out(1, 1)),
        gfingerprint(out(1, block)),
    ]
    return (
        f"gemm-e4m3-kblock{block}-"
        + hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]
    )


# ----------------------------------------------------------------------------- the MoE class
class Fp8BsMoESpec(MoESpec):
    """Symbolic view of the class ``moe.e4m3bs.<act>[H, I, k, E, method]`` for one token: the routing and the combine of
    :class:`.spec.MoESpec`, GEMM1 the bf16 class's chain on the promoted accumulators (the block scales are inside the fold:
    no scale symbol enters the chain), GEMM2 the plain conversion."""

    def __init__(
        self,
        H: int,
        I: int,
        k: int,
        E: int,
        method: RoutingMethod = SOFTMAX_RENORM,
        act: str = "silu",
        params: Sequence[Optional[float]] = (),
        gated: Optional[bool] = None,
        alg: Optional[TermAlgebra] = None,
        block: int = BLOCK,
    ):
        super().__init__(H, I, k, E, method, "bf16", alg, act, params, gated)
        self.block = check_block(block)

    def gemm2(self, r: int = 0, h: int = 0) -> Term:
        """GEMM2 of the block-scale class: ``cvt_bf16(acc2)`` of the promoted accumulator."""
        return self.alg.cvt_out(self.alg.leaf("acc2", r, h), "bf16")

    @property
    def name(self) -> str:
        return (
            CLASS_OF[self.block]
            + "."
            + (
                "swiglu"
                if self.gated and self.act == "silu" and not self.params
                else self.act
            )
        )


def spec_fingerprint_fp8bs(
    H: int,
    I: int,
    k: int,
    E: int,
    method: RoutingMethod = SOFTMAX_RENORM,
    act: str = "silu",
    params: Sequence[Optional[float]] = (),
    gated: Optional[bool] = None,
    block: int = BLOCK,
) -> str:
    """Identifier of the class: the routing weights of slot 0 and slot k-1, the GEMM1 chain (its activation, gating and
    binary32 parameters), the block-scale GEMM core at the block (``gemm.e4m3.kblock128`` / ``kblock32``) at ``K = H`` and ``K =
    I``, the ``fp8bs.quant`` class at the block on the activations, the intermediate and the expert weights (one class, three
    placements), GEMM2's chain and the combine. The scales' values, like the operands, are inputs; the block size
    (``deepseek_fp8_block_size``) is a dimension of the class."""
    s = Fp8BsMoESpec(H, I, k, E, method, act, params, gated, block=block)
    q = quant_fingerprint(block)
    parts = [
        CLASS_OF[block],
        f"H={H}",
        f"I={I}",
        f"k={k}",
        f"E={E}",
        method.name,
        f"act={'glu' if s.gated else 'act'}:{act}",
        f"params={s.params!r}",
        fingerprint(s.weight(0)),
        fingerprint(s.weight(k - 1)),
        fingerprint(s.gemm1()),
        gemm_core_fingerprint(H, "f32", block),
        gemm_core_fingerprint(I, "f32", block),
        q,
        q,
        q,
        fingerprint(s.gemm2()),
        fingerprint(s.combine()),
    ]
    return "moe-" + hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


__all__ = [
    "ACTS",
    "AMAX_IDENTITY_NOTE",
    "BLOCK",
    "BLOCKS",
    "BSQuantSpec",
    "CLASS",
    "CLASS_OF",
    "GEMM_CLASS_OF",
    "IN_FMT_OF",
    "STAGE_K",
    "E4M3_MAX",
    "Fp8BsMoESpec",
    "IN_FMT",
    "PLACEMENT_VS",
    "SCALE_EPS",
    "UNGATED_ACTS",
    "block_amax_placement",
    "block_chain",
    "block_reduction",
    "check_block",
    "elem_digit",
    "gemm_core_fingerprint",
    "inv_term",
    "leaf_value",
    "quant_elem",
    "quant_fingerprint",
    "quant_pre",
    "reduction_kblock",
    "scale_term",
    "spec_fingerprint_fp8bs",
]
