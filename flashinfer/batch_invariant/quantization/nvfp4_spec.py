"""NVFP4 quantization on the canon engine (docs/nvfp4_quant_scoping.md sections 2-3).

Everything numeric lives here once, parameterised by an ``fp`` backend (``TermAlgebra`` symbolic,
``flashinfer.batch_invariant.quantization.reference.ExactF32`` exact CPU, ``flashinfer.batch_invariant.quantization.reference.TorchF32`` vectorised CPU, ``CuteF32``
kernel):

* ``block_reduction`` / ``row_reduction``: the amax reductions as :class:`canon.Reduction` objects
  (``pairwise`` digits combined with ``fmaxc``, the commutativity-normalised max of non-negative values;
  the leaf is ``fabs(x)``);
* ``block_chain``: the per-block scale chain of TransformerEngine's NVFP4 recipe as FlashInfer implements
  it under ``FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH`` (every op correctly rounded or exact);
* ``quant_elem``: the element chain ``cvt_e2m1(mul(x, inv))``;
* ``row_scale_chain``: the per-token global scale and the per-token scale output.

The symbolic spec, the tracer, the CPU references and the CuTe kernel all call these same functions.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from ..fp_interface import exact as fp32
from ..canon import Digit, Reduction
from .qterm import Term, TermAlgebra, sexpr
from .nvfp4_schedule import BLOCK, num_blocks, padded_blocks

C_SIXTH = fp32.to_f32(
    1.0 / 6.0
)  # TE multiplies by the rounded reciprocal of the e2m1 maximum (0x3E2AAAAB)
F32_MAX = fp32.F32_MAX
E4M3_FMT = "e4m3"
E2M1_FMT = "e2m1"
AMAX_IDENTITY_NOTE = (
    "fmaxc(v, +0) == v for every v >= +0; every leaf is fabs(x) >= +0 and never -0"
)


# ----------------------------------------------------------------------------- the reductions
def leaf_value(fp, x):
    """The reduction leaf: ``fabs(x)`` of the (exactly widened) input element."""
    return fp.fabs(x)


def elem_digit() -> Digit:
    return Digit(
        "elem",
        BLOCK,
        "pairwise",
        combine=lambda fp, a, b: fp.fmaxc(a, b),
        identity_note=AMAX_IDENTITY_NOTE,
    )


def blk_digit(K: int) -> Digit:
    return Digit(
        "blk",
        padded_blocks(K),
        "pairwise",
        combine=lambda fp, a, b: fp.fmaxc(a, b),
        identity_note=AMAX_IDENTITY_NOTE,
    )


def block_reduction(leaf_x: Callable) -> Reduction:
    """``amax_b = fmaxc-tree over fabs(x_e)``, ``e`` in 0..15; ``leaf_x(fp, e)`` is element ``e`` of the
    block as an f32 term / value (the caller fixes the row and the block)."""
    return Reduction(
        "nvfp4.block_amax",
        (elem_digit(),),
        lambda fp, idx: leaf_value(fp, leaf_x(fp, idx["elem"])),
    )


def row_reduction(K: int, leaf_x: Callable) -> Reduction:
    """The row amax (per-token class): digits ``elem`` then ``blk`` (radix ``Kbp``; blocks ``>= K/16``
    absent). ``leaf_x(fp, i)`` is element ``i`` of the row. The ``elem`` sub-trees are exactly the
    terms :func:`block_reduction` builds (same leaves, same association)."""
    nb = num_blocks(K)

    def leaf(fp, idx):
        b = idx["blk"]
        return leaf_value(fp, leaf_x(fp, BLOCK * b + idx["elem"])) if b < nb else None

    return Reduction(f"nvfp4.row_amax[K={K}]", (elem_digit(), blk_digit(K)), leaf)


# ----------------------------------------------------------------------------- the chains
def sf_value(fp, amax, gs):
    """``sf_f = mul(amax, mul(gs, 1/6))``: the unrounded block scale (TE's ``vecMax * (SFScaleVal * fp4_max_inv)``)."""
    return fp.mul(amax, fp.mul(gs, fp.const(C_SIXTH)))


def sf_quant(fp, sf_f):
    """``cvt.rn.satfinite.e4m3`` of the block scale, as the f32 value of the stored byte (exact widening)."""
    return fp.cvt_out(sf_f, E4M3_FMT)


def inv_scale(fp, sf_d, gs):
    """``inv = fmin(div(1, mul(sf_d, div(1, gs))), FLT_MAX)``: TE's encode scale, correctly rounded
    (FlashInfer's default path uses ``rcp.approx.ftz`` twice here)."""
    one = fp.const(1.0)
    return fp.fmin(fp.div(one, fp.mul(sf_d, fp.div(one, gs))), fp.const(F32_MAX))


def block_chain(fp, amax, gs):
    """``(sf_d, inv)`` of a block from its amax and the (per-tensor or per-token) global scale."""
    sf_d = sf_quant(fp, sf_value(fp, amax, gs))
    return sf_d, inv_scale(fp, sf_d, gs)


def quant_pre(fp, x, inv):
    """The element chain before the conversion: ``mul(x, inv)`` (the kernel converts eight of these with
    one ``cvt.rn.satfinite.e2m1x2.f32`` pair sequence, the spec with :func:`quant_elem`)."""
    return fp.mul(x, inv)


def quant_elem(fp, x, inv):
    """``q = cvt.rn.satfinite.e2m1(mul(x, inv))``, as the e2m1 value (the caller packs the code)."""
    return fp.cvt_out(quant_pre(fp, x, inv), E2M1_FMT)


def row_scale_chain(fp, ramax, gs_inv):
    """Per-token class: ``gs0 = div(1, gs_inv)``, ``res = fmin(div(gs0, ramax), FLT_MAX)`` (``ramax = 0``
    gives ``FLT_MAX``), ``ges = nzor(res, 1.0)`` (TE: ``rowEncodeScale != 0 ? rowEncodeScale : 1``),
    ``pts = div(1, ges)``. Returns ``(ges, pts)``: the row's global scale and the per-token scale output."""
    one = fp.const(1.0)
    gs0 = fp.div(one, gs_inv)
    res = fp.fmin(fp.div(gs0, ramax), fp.const(F32_MAX))
    ges = fp.nzor(res, one)
    return ges, fp.div(one, ges)


# ----------------------------------------------------------------------------- the classes
@dataclass(frozen=True)
class Variant:
    """``per_token=False``: class ``nvfp4.quant.te`` (the global scale is an input).
    ``per_token=True``: class ``nvfp4.quant.te.pertoken`` (the global scale is derived per row)."""

    per_token: bool = False

    @property
    def name(self) -> str:
        return "nvfp4_quant" + ("_pertoken" if self.per_token else "")


PER_TENSOR = Variant()
PER_TOKEN = Variant(per_token=True)


class QuantSpec:
    """Symbolic (term) view of one row of the class. ``x(row, i)`` is the input element widened to f32
    (exact for bf16 / f16, so the term is the element itself); the global scale is ``sym("gs")``
    (per-tensor) or derived from ``sym("gs_inv")`` and the row amax (per-token)."""

    def __init__(
        self, K: int, variant: Variant = PER_TENSOR, alg: Optional[TermAlgebra] = None
    ):
        self.K, self.variant = K, variant
        self.nb = num_blocks(K)
        self.alg = alg or TermAlgebra()

    def leaf(self, row: int, i: int) -> Term:
        return self.alg.x(row, i)

    def block_reduction(self, row: int, b: int) -> Reduction:
        return block_reduction(lambda fp, e: self.leaf(row, BLOCK * b + e))

    def row_reduction(self, row: int) -> Reduction:
        return row_reduction(self.K, lambda fp, i: self.leaf(row, i))

    def block_amax(self, row: int, b: int) -> Term:
        return self.block_reduction(row, b).canonical(self.alg)

    def row_amax(self, row: int) -> Term:
        return self.row_reduction(row).canonical(self.alg)

    def row_scale(self, row: int):
        """``(gs, pts)``: the scale the blocks of ``row`` are quantised with and the per-token output
        (``None`` for the per-tensor class)."""
        a = self.alg
        if not self.variant.per_token:
            return a.sym("gs"), None
        return row_scale_chain(a, self.row_amax(row), a.sym("gs_inv"))

    def block(self, row: int, b: int, gs: Optional[Term] = None) -> Dict[str, Any]:
        a = self.alg
        gs = self.row_scale(row)[0] if gs is None else gs
        amax = self.block_amax(row, b)
        sf_d, inv = block_chain(a, amax, gs)
        codes = [
            quant_elem(a, self.leaf(row, BLOCK * b + e), inv) for e in range(BLOCK)
        ]
        return {"amax": amax, "sf": sf_d, "inv": inv, "codes": codes}

    def outputs(self, row: int) -> Dict[str, Any]:
        """``{"codes": [K e2m1 value terms], "sf": [nb e4m3 value terms], "pts": term | None}``."""
        gs, pts = self.row_scale(row)
        codes: List[Term] = []
        sfs: List[Term] = []
        for b in range(self.nb):
            blk = self.block(row, b, gs)
            codes += blk["codes"]
            sfs.append(blk["sf"])
        return {"codes": codes, "sf": sfs, "pts": pts}


def class_fingerprint(variant: Variant = PER_TENSOR, K: Optional[int] = None) -> str:
    """The identifier of the class. Per-tensor: the structural hash of block 0's scale term and its first and last code
    terms -- a block's bytes depend on its 16 elements and the global scale alone, so one specification serves every row
    length and the fingerprint takes no dimension. Per-token: the row's global scale is derived from the row amax, whose
    tree spans the row's blocks, so the fingerprint takes ``K`` (the row length is a dimension of that class)."""
    if variant.per_token:
        if K is None:
            raise ValueError(
                "the per-token NVFP4 class's row tree spans the row: fingerprint('nvfp4_quantize', per_token=True) needs K="
            )
        return spec_fingerprint(K, variant)
    s = QuantSpec(BLOCK, variant)
    gs, _ = s.row_scale(0)
    blk = s.block(0, 0, gs)
    parts = [
        variant.name,
        sexpr(blk["sf"]),
        sexpr(blk["codes"][0]),
        sexpr(blk["codes"][BLOCK - 1]),
    ]
    return "nvfp4quant-" + hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def spec_fingerprint(K: int, variant: Variant = PER_TENSOR) -> str:
    """Identifier of the numerical specification of the class for row length ``K``: the structural hash
    of the first and last block's scale and first / last code terms, and the per-token scale term
    (which embeds the whole row tree, hence ``K``)."""
    s = QuantSpec(K, variant)
    gs, pts = s.row_scale(0)
    first, last = s.block(0, 0, gs), s.block(0, s.nb - 1, gs)
    parts = [
        f"K={K}",
        variant.name,
        sexpr(first["sf"]),
        sexpr(first["codes"][0]),
        sexpr(first["codes"][BLOCK - 1]),
        sexpr(last["sf"]),
        sexpr(last["codes"][BLOCK - 1]),
    ]
    if pts is not None:
        parts.append(sexpr(pts))
    return "nvfp4quant-" + hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]
