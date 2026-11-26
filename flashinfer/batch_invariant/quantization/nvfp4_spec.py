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
* ``row_scale_chain``: the per-token global scale and the per-token scale output;
* the 4over6 classes (docs/nvfp4_quant_scoping.md section 9 of the engine repository): ``sf_candidates`` (the two block
  scales), ``recon`` / ``err_term`` / ``err_digit`` (the reconstruction error of a candidate as a register-level fold over
  the block's 16 elements, in index order), ``row_scale_chain_4over6`` and the selection rule ``pick4 = E4 < E6`` with the
  zero rule (``QuantSpec.block`` returns both candidates and the two error terms; the evaluators apply ``select_block``).

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


def inv_scale(fp, sf_d, gs, gs_inv=None):
    """``inv = fmin(div(1, mul(sf_d, div(1, gs))), FLT_MAX)``: TE's encode scale, correctly rounded
    (FlashInfer's default path uses ``rcp.approx.ftz`` twice here)."""
    one = fp.const(1.0)
    gi = (
        fp.div(one, gs) if gs_inv is None else gs_inv
    )  # the same node: a caller may compute it once for several scales
    return fp.fmin(fp.div(one, fp.mul(sf_d, gi)), fp.const(F32_MAX))


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
E2M1_MAX = 6.0
SF4_FACTOR = 1.5  # the "4" candidate's scale is 1.5x the "6" candidate's: the block maximum maps to 4 instead of 6


@dataclass(frozen=True)
class Recipe:
    """FlashInfer's NVFP4 "4over6" recipe (``NVFP44Over6Config``): every block is quantised with the scale that maps its
    maximum to 6 and with 1.5x that scale (the maximum maps to 4; the two largest codes unused, the small values finer),
    and the candidate with the smaller reconstruction error over the block is kept. ``err_mode``: ``"mae"`` (the error of
    an element is ``|d|``) or ``"mse"`` (``d * d``). ``e4m3_max`` (448 or 256): the e4m3 clamp the caller's global scale is
    built with; it enters the term of the per-token variant only (the reconstruction's denominator ``6 * e4m3_max`` and
    the inverse base scale), so the per-tensor classes normalise it to 448."""

    err_mode: str = "mae"
    e4m3_max: int = 448

    def __post_init__(self):
        if self.err_mode not in ("mae", "mse"):
            raise ValueError(f"4over6 err_mode {self.err_mode!r}: 'mae' or 'mse'")
        if self.e4m3_max not in (448, 256):
            raise ValueError(f"4over6 e4m3_max {self.e4m3_max}: 448 or 256")


@dataclass(frozen=True)
class Variant:
    """``per_token=False``: class ``nvfp4.quant.te`` (the global scale is an input).
    ``per_token=True``: class ``nvfp4.quant.te.pertoken`` (the global scale is derived per row).
    ``recipe``: the 4over6 classes ``nvfp4.quant.4over6.<mode>`` / ``.pertoken.e4m3max<N>``."""

    per_token: bool = False
    recipe: Optional[Recipe] = None

    def __post_init__(self):
        if (
            self.recipe is not None
            and not self.per_token
            and self.recipe.e4m3_max != 448
        ):
            # the per-tensor 4over6 term does not contain e4m3_max: one class for both clamps
            object.__setattr__(self, "recipe", Recipe(self.recipe.err_mode, 448))

    @property
    def name(self) -> str:
        n = "nvfp4_quant" + ("_pertoken" if self.per_token else "")
        if self.recipe is not None:
            n += f"_4over6_{self.recipe.err_mode}" + (
                f"_e{self.recipe.e4m3_max}" if self.per_token else ""
            )
        return n

    @property
    def class_name(self) -> str:
        """The registry's class name (``classes.py``)."""
        n = "nvfp4.quant" + (
            ".4over6." + self.recipe.err_mode if self.recipe is not None else ""
        )
        if self.per_token:
            n += ".pertoken" + (
                f".e4m3max{self.recipe.e4m3_max}" if self.recipe is not None else ""
            )
        return n


PER_TENSOR = Variant()
PER_TOKEN = Variant(per_token=True)


# ----------------------------------------------------------------------------- the 4over6 chains
def zsel(fp, e, p):
    """``ZSEL(e, p) = +0 if e == 0 else p``: an exact comparison-and-select (``setp.eq.f32`` + ``selp.f32`` on the device;
    the CPU models compare the binary32 value, so ``-0`` selects too). ``fp.zsel`` on the backends that name it, the opaque
    binary ``op`` on the term algebra."""
    if hasattr(fp, "zsel"):
        return fp.zsel(e, p)
    return fp.op("zsel", e, p)


def sf_candidate_values(fp, amax, gs):
    """The two unrounded block scales of the recipe: ``sf6 = mul(div(amax, 6), gs)`` (FlashInfer divides by 6 exactly here,
    unlike the plain class's ``mul(amax, mul(gs, 1/6))``) and ``sf4 = mul(sf6, 1.5)``. Returns ``(sf4, sf6)``."""
    sf6 = fp.mul(fp.div(amax, fp.const(E2M1_MAX)), gs)
    return fp.mul(sf6, fp.const(SF4_FACTOR)), sf6


def sf_candidates(fp, amax, gs):
    """The two block scales rounded to e4m3 and their encode scales: ``(sfd4, sfd6, inv4, inv6)``."""
    sf4, sf6 = sf_candidate_values(fp, amax, gs)
    sfd4, sfd6 = sf_quant(fp, sf4), sf_quant(fp, sf6)
    return sfd4, sfd6, inv_scale(fp, sfd4, gs), inv_scale(fp, sfd6, gs)


def recon(fp, q, sfd, gs, ramax, e4m3_max: int, gds=None):
    """The reconstruction of one quantised element: per-tensor (``ramax`` None) ``mul(mul(q, sfd), div(1, gs))``; per-token
    ``div(mul(mul(q, sfd), ramax), 6 * e4m3_max)`` (FlashInfer's ``rowAmax > 0`` branch: a row with ``ramax = 0`` is all zero
    and every block of it takes the zero rule). ``gds``: the per-tensor factor ``div(1, gs)`` computed once by the caller
    (the kernel hoists it per block; the same term)."""
    qs = fp.mul(q, sfd)
    if ramax is None:
        return fp.mul(qs, fp.div(fp.const(1.0), gs) if gds is None else gds)
    return fp.div(fp.mul(qs, ramax), fp.const(E2M1_MAX * e4m3_max))


def err_term(fp, d, x, err_mode: str):
    """The element's error: ``diff = add(d, neg(x))`` (``sub.rn``), then ``fabs(diff)`` (mae) or ``mul(diff, diff)`` (mse)."""
    diff = fp.add(d, fp.neg(x))
    return fp.fabs(diff) if err_mode == "mae" else fp.mul(diff, diff)


def err_digit() -> Digit:
    """The block's error sum: a ``fold`` digit over the 16 elements in index order from ``+0`` (the first step is
    ``add(+0, t_0) = t_0`` exactly: every ``t`` is ``>= +0``). A fold digit is register-level in every plan, so a
    placement of a 4over6 class holds a whole block in one thread (``elems_per_thread = 16``)."""
    return Digit(
        "err",
        BLOCK,
        "fold",
        init=lambda fp: fp.const(0.0),
        step=lambda fp, acc, t: fp.add(acc, t),
    )


def block_error(fp, ts):
    """``E = fold(err_digit) over ts``: the canonical chain (``Digit.reduce``: the same function on every backend)."""
    return err_digit().reduce(fp, list(ts))


def row_scale_chain_4over6(fp, ramax, gs_inv):
    """The per-token 4over6 row scale: ``row_scale_chain`` with the recipe's zero rule -- ``ramax = 0`` gives ``ges = 0``
    and ``pts = 0`` (the plain per-token class gives ``pts = 1`` there). Returns ``(ges, pts)``."""
    ges, pts = row_scale_chain(fp, ramax, gs_inv)
    return zsel(fp, ramax, ges), zsel(fp, ramax, pts)


SELECT_RULE = "pick4 = lt(e4, e6) strict (tie -> 6); amax == 0 -> sf 0, codes +0"


def select_block(pick4: bool, zero: bool, blk: Dict[str, Any]):
    """The 4over6 outputs of an evaluated block: ``(sf value, [16 code values])`` -- the "4" candidate when ``pick4``, else the
    "6" one; both ``+0`` when ``zero`` (``amax == 0``). The same rule on every backend (:data:`SELECT_RULE`)."""
    if zero:
        return 0.0, [0.0] * BLOCK
    return (blk["sf4"], blk["codes4"]) if pick4 else (blk["sf6"], blk["codes6"])


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
        if self.variant.recipe is not None:
            return row_scale_chain_4over6(a, self.row_amax(row), a.sym("gs_inv"))
        return row_scale_chain(a, self.row_amax(row), a.sym("gs_inv"))

    def block(self, row: int, b: int, gs: Optional[Term] = None) -> Dict[str, Any]:
        """The block's terms. Plain classes: ``{"amax", "sf", "inv", "codes"}``. 4over6 classes: the two candidates
        ``"sf4" / "sf6"`` (e4m3 values), ``"inv4" / "inv6"``, ``"codes4" / "codes6"`` (16 e2m1 values each), the error terms
        ``"e4" / "e6"`` and ``"amax"``; the outputs are ``codes4, sf4`` when ``e4 < e6`` else ``codes6, sf6``, and the zero
        rule makes every output of a block with ``amax = 0`` the value ``+0`` (``select_block``)."""
        a = self.alg
        gs = self.row_scale(row)[0] if gs is None else gs
        amax = self.block_amax(row, b)
        xs = [self.leaf(row, BLOCK * b + e) for e in range(BLOCK)]
        if self.variant.recipe is None:
            sf_d, inv = block_chain(a, amax, gs)
            return {
                "amax": amax,
                "sf": sf_d,
                "inv": inv,
                "codes": [quant_elem(a, x, inv) for x in xs],
            }
        r = self.variant.recipe
        ramax = self.row_amax(row) if self.variant.per_token else None
        sfd4, sfd6, inv4, inv6 = sf_candidates(a, amax, gs)
        codes4 = [quant_elem(a, x, inv4) for x in xs]
        codes6 = [quant_elem(a, x, inv6) for x in xs]
        t4 = [
            err_term(a, recon(a, q, sfd4, gs, ramax, r.e4m3_max), x, r.err_mode)
            for q, x in zip(codes4, xs, strict=True)
        ]
        t6 = [
            err_term(a, recon(a, q, sfd6, gs, ramax, r.e4m3_max), x, r.err_mode)
            for q, x in zip(codes6, xs, strict=True)
        ]
        e4 = block_error(a, t4)
        e6 = block_error(a, t6)
        return {
            "amax": amax,
            "sf4": sfd4,
            "sf6": sfd6,
            "inv4": inv4,
            "inv6": inv6,
            "codes4": codes4,
            "codes6": codes6,
            "e4": e4,
            "e6": e6,
            "t4": t4,
            "t6": t6,
        }

    def outputs(self, row: int) -> Dict[str, Any]:
        """``{"codes": [K e2m1 value terms], "sf": [nb e4m3 value terms], "pts": term | None}``."""
        gs, pts = self.row_scale(row)
        if self.variant.recipe is not None:
            return {
                "blocks": [self.block(row, b, gs) for b in range(self.nb)],
                "pts": pts,
            }
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
    if variant.recipe is None:
        parts = [
            variant.name,
            sexpr(blk["sf"]),
            sexpr(blk["codes"][0]),
            sexpr(blk["codes"][BLOCK - 1]),
        ]
    else:
        parts = [
            variant.name,
            SELECT_RULE,
            sexpr(blk["sf4"]),
            sexpr(blk["sf6"]),
            sexpr(blk["codes4"][0]),
            sexpr(blk["codes6"][BLOCK - 1]),
            sexpr(blk["e4"]),
            sexpr(blk["e6"]),
        ]
    return "nvfp4quant-" + hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def spec_fingerprint(K: int, variant: Variant = PER_TENSOR) -> str:
    """Identifier of the numerical specification of the class for row length ``K``: the structural hash
    of the first and last block's scale and first / last code terms, and the per-token scale term
    (which embeds the whole row tree, hence ``K``)."""
    s = QuantSpec(K, variant)
    gs, pts = s.row_scale(0)
    first, last = s.block(0, 0, gs), s.block(0, s.nb - 1, gs)
    if variant.recipe is None:
        parts = [
            f"K={K}",
            variant.name,
            sexpr(first["sf"]),
            sexpr(first["codes"][0]),
            sexpr(first["codes"][BLOCK - 1]),
            sexpr(last["sf"]),
            sexpr(last["codes"][BLOCK - 1]),
        ]
    else:
        parts = [
            f"K={K}",
            variant.name,
            SELECT_RULE,
            sexpr(first["sf4"]),
            sexpr(first["sf6"]),
            sexpr(first["codes4"][0]),
            sexpr(first["codes6"][BLOCK - 1]),
            sexpr(first["e4"]),
            sexpr(first["e6"]),
            sexpr(last["e4"]),
        ]
    if pts is not None:
        parts.append(sexpr(pts))
    return "nvfp4quant-" + hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]
