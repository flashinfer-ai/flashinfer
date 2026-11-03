"""MXFP8 quantization on the canon engine.

Everything numeric lives here once, parameterised by an ``fp`` backend (``qterm.TermAlgebra`` symbolic, ``fp_interface.exact``
exact CPU, ``cute_ext.CuteF32`` kernel):

* ``block_reduction``: the per-block amax as a :class:`canon.Reduction` (one ``pairwise`` digit of radix 32 combined
  with ``fmaxc``, the commutativity-normalised max of non-negative values; the leaf is ``fabs(x)`` of the exactly
  widened input element); ``row_reduction`` adds the ``blk`` digit the kernels use for ownership only (it is never
  combined: there is no per-row scale in the MX classes);
* the **recipe constants** (:class:`ScaleRule`, :data:`DEFAULT_RULE`): the normalisation of the amax, the UE8M0
  rounding, the inverse-scale rule and the saturation rule, each a documented, switchable value -- the default is
  TransformerEngine's MXFP8 recipe, which FlashInfer's MX quantizers share up to the two differences recorded in
  the package README (the inverse scale at UE8M0 byte 0, the clamp of infinite inputs); the pinned fingerprints below
  name the default recipe, every other variant is a different class;
* the chains ``scale_norm`` -> ``scale_quant`` -> ``inv_scale`` (``scale_chain``) and ``quant_pre`` -> ``quant_elem``,
  written once and called by the symbolic spec, the exact CPU evaluation and the CuTe kernel.

Stable names: ``MXFP8``, ``FORMATS``, ``ScaleRule``, ``DEFAULT_RULE``, ``BLOCK``,
``block_reduction``, ``block_amax_placement``, ``scale_norm``, ``scale_quant``, ``scale_chain``, ``inv_scale``,
``quant_pre``, ``quant_elem``, ``MXQuantSpec``, ``spec_fingerprint``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from ..fp_interface import exact
from ..canon import Digit, Placement, Reduction
from .qterm import Term, TermAlgebra, sexpr
from .layout import num_blocks, padded_blocks

BLOCK = 32  # elements per UE8M0 scale (sf_vec_size); fixed by the OCP MX formats
AMAX_IDENTITY_NOTE = (
    "fmaxc(v, +0) == v for every v >= +0; every leaf is fabs(x) >= +0 and never -0"
)


# ----------------------------------------------------------------------------- the formats
@dataclass(frozen=True)
class MXFormat:
    """An MX element format: ``elem_fmt`` is the declared ``cvt.rn.satfinite`` conversion (``exact.to_fp8`` / ``exact.to_e2m1``),
    ``elem_max`` its largest finite magnitude, ``emax`` the exponent of its largest power of two."""

    name: str
    elem_fmt: str
    elem_max: float
    emax: int
    elem_bits: int

    @property
    def c_rcp(self) -> float:
        """TE's ``max_norm_rcp``: the binary32 nearest to ``1 / elem_max`` (0x3B124925 for e4m3; above the exact
        reciprocal, and ``elem_max * c_rcp`` rounds to exactly 1.0)."""
        return exact.to_f32(1.0 / self.elem_max)

    @property
    def c_pow2(self) -> float:
        """``2^-emax``: the exact normalisation of OCP's ``floor(log2(amax)) - emax`` rule."""
        return 2.0**-self.emax


MXFP8 = MXFormat("mxfp8", "e4m3", 448.0, 8, 8)
MXFP4 = MXFormat("mxfp4", "e2m1", 6.0, 2, 4)
FORMATS = {"mxfp8": MXFP8, "mxfp4": MXFP4}


# ----------------------------------------------------------------------------- the recipe constants
SCALE_NORMS = ("rcp", "div", "pow2")
SCALE_ROUNDS = exact.UE8M0_ROUNDINGS  # rp (ceil), rz (floor), rn (nearest, software)
INV_RULES = (
    "exact",
)  # FlashInfer's zero sentinel for bytes 0 / 254 is documented, not implemented
SAT_RULES = ("satfinite", "clamp_satfinite")
ZERO_RULES = (
    "keep",
    "positive",
)  # the sign of a zero product: mul keeps it; TE's fma(x, inv, +0) forms turn -0 into +0
DATA_ROUND = (
    "rn"  # cvt.rn.satfinite.e4m3x2.f32: the only hardware rounding of the conversion
)


@dataclass(frozen=True)
class ScaleRule:
    """The switchable constants of the MX recipe.

    ``norm``: how the block amax is normalised before the UE8M0 conversion -- ``"rcp"``: ``mul(amax, c_rcp)`` (TE and
    FlashInfer's CuTe mxfp8; the default), ``"div"``: ``div(amax, elem_max)`` (the correctly rounded quotient),
    ``"pow2"``: ``mul(amax, 2^-emax)`` (exact; with ``round = "rz"`` this is OCP's floor rule, with ``"rp"`` torchao's
    CEIL). ``round``: the UE8M0 rounding, ``"rp"`` (ceil: the hardware mode TE / FlashInfer use), ``"rz"`` (floor),
    ``"rn"`` (nearest power of two, ties up; software). ``inv``: ``"exact"``: ``inv = div(1, sf_d)`` = ``2^(127 - e8)``
    for every byte (TE). ``sat``: ``"satfinite"``: the element conversion's own saturation (TE); ``"clamp_satfinite"``:
    FlashInfer's ``fmax(fmin(y, elem_max), -elem_max)`` before the conversion (the same bits on finite inputs).
    ``zero``: the sign of a zero product -- ``"keep"``: ``mul(x, inv)`` (``-0 * inv = -0``, code 0x80 / nibble 0x8);
    ``"positive"``: ``fma(x, inv, +0)`` (TE's CuTe-DSL backend and its pre-CUDA-13.1 specialized builds: ``-0 + +0 = +0``,
    code 0x00); the two differ only in the sign bit of zero codes."""

    norm: str = "rcp"
    round: str = "rp"
    inv: str = "exact"
    sat: str = "satfinite"
    zero: str = "keep"

    def __post_init__(self):
        if (
            self.norm not in SCALE_NORMS
            or self.round not in SCALE_ROUNDS
            or self.inv not in INV_RULES
            or self.sat not in SAT_RULES
            or self.zero not in ZERO_RULES
        ):
            raise ValueError(f"invalid scale rule {self}")

    @property
    def name(self) -> str:
        s = f"{self.norm}{self.round}"
        if self.inv != "exact":
            s += f"-{self.inv}"
        if self.sat != "satfinite":
            s += "-clamp"
        if self.zero != "keep":
            s += "-pz"
        return s


DEFAULT_RULE = ScaleRule()  # TE's recipe: rcp + rp + exact + satfinite
TE_RULE = DEFAULT_RULE
OCP_FLOOR_RULE = ScaleRule(
    norm="pow2", round="rz"
)  # the OCP MX spec sheet's shared exponent
FI_CLAMP_RULE = ScaleRule(
    sat="clamp_satfinite"
)  # FlashInfer's mxfp8 element path (identical on finite inputs)
TE_FMA_RULE = ScaleRule(
    zero="positive"
)  # TE's fma(x, inv, +0) element path (zero codes always +0)


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
    """The ownership digit of the kernels (never combined: no row scale in the MX classes)."""
    return Digit(
        "blk",
        padded_blocks(K, BLOCK),
        "pairwise",
        combine=lambda fp, a, b: fp.fmaxc(a, b),
        identity_note=AMAX_IDENTITY_NOTE,
    )


def block_reduction(leaf_x: Callable) -> Reduction:
    """``amax_b = fmaxc-tree over fabs(x_e)``, ``e`` in 0..31; ``leaf_x(fp, e)`` is element ``e`` of the block as an f32
    term / value (the caller fixes the row and the block)."""
    return Reduction(
        "mx.block_amax",
        (elem_digit(),),
        lambda fp, idx: leaf_value(fp, leaf_x(fp, idx["elem"])),
    )


def row_reduction(K: int, leaf_x: Callable) -> Reduction:
    """The kernels' reduction: digits ``elem`` then ``blk`` (radix ``Kbp``, blocks ``>= K/32`` absent). The ``elem``
    sub-trees are exactly the terms :func:`block_reduction` builds; the ``blk`` digit only fixes the ownership map
    (``canon.run_steps(..., stop_before="blk")``)."""
    nb = num_blocks(K, BLOCK)

    def leaf(fp, idx):
        b = idx["blk"]
        return leaf_value(fp, leaf_x(fp, BLOCK * b + idx["elem"])) if b < nb else None

    return Reduction(f"mx.row_blocks[K={K}]", (elem_digit(), blk_digit(K)), leaf)


def block_amax_placement(V: int) -> Placement:
    """The placement of the ``elem`` digit a thread holding ``V`` consecutive elements of a block uses:
    ``[(V, reg), (32 / V, lane)]`` (the ``32 / V`` lanes of a block are adjacent lanes; the block amax is finished by
    ``log2(32 / V)`` xor shuffles). ``V`` in {32, 16, 8}. This is the fused kernels' entry point to the canonical tree."""
    assert V in (32, 16, 8), V
    return Placement.make({"elem": [(V, "reg"), (BLOCK // V, "lane")]})


# ----------------------------------------------------------------------------- the chains
def scale_norm(fp, amax, fmt: MXFormat, rule: ScaleRule = DEFAULT_RULE):
    """``v = NORM(amax)``: the value the UE8M0 conversion rounds (section 3.1)."""
    if rule.norm == "rcp":
        return fp.mul(amax, fp.const(fmt.c_rcp))
    if rule.norm == "div":
        return fp.div(amax, fp.const(fmt.elem_max))
    return fp.mul(amax, fp.const(fmt.c_pow2))


def scale_quant(fp, v, rule: ScaleRule = DEFAULT_RULE):
    """``sf_d = cvt_ue8m0_<round>(v)``: the stored UE8M0 byte, as its exactly widened value ``2^(e8 - 127)``."""
    return fp.cvt_out(v, "ue8m0_" + rule.round)


def inv_scale(fp, sf_d, rule: ScaleRule = DEFAULT_RULE):
    """``inv = div(1, sf_d)``: exact for every UE8M0 value (``1 / 2^k`` is a binary32 for ``k`` in [-127, 127]), so the
    kernel executes this very op (one ``div.rn`` per block)."""
    return fp.div(fp.const(1.0), sf_d)


def scale_chain(fp, amax, fmt: MXFormat, rule: ScaleRule = DEFAULT_RULE):
    """``(sf_d, inv)`` of a block from its amax."""
    sf_d = scale_quant(fp, scale_norm(fp, amax, fmt, rule), rule)
    return sf_d, inv_scale(fp, sf_d, rule)


def quant_pre(fp, x, inv, rule: ScaleRule = DEFAULT_RULE):
    """The element chain before the conversion: ``mul(x, inv)`` (``zero = "keep"``) or ``fma(x, inv, +0)`` (``"positive"``:
    the same value, a zero product always ``+0``)."""
    if rule.zero == "keep":
        return fp.mul(x, inv)
    return fp.fma(x, inv, fp.const(0.0))


def saturate(fp, y, fmt: MXFormat, rule: ScaleRule = DEFAULT_RULE):
    """The saturation rule: the identity (the conversion saturates) or FlashInfer's explicit clamp to ``+-elem_max``
    (``max.f32`` as the commutative join ``fmaxc``: its operands, ``fmin(y, elem_max)`` and ``-elem_max``, are never a
    ``+-0`` pair, so the order-sensitive and the commutative max are one function here)."""
    if rule.sat == "satfinite":
        return y
    return fp.fmaxc(fp.fmin(y, fp.const(fmt.elem_max)), fp.const(-fmt.elem_max))


def quant_elem(fp, x, inv, fmt: MXFormat, rule: ScaleRule = DEFAULT_RULE):
    """``q = cvt.rn.satfinite.<elem>(SAT(mul(x, inv)))``, as the element value (the caller packs the code)."""
    return fp.cvt_out(
        saturate(fp, quant_pre(fp, x, inv, rule), fmt, rule), fmt.elem_fmt
    )


# ----------------------------------------------------------------------------- the classes
class MXQuantSpec:
    """Symbolic (term) view of one row of the class ``<fmt>.quant`` under ``rule``. ``x(row, i)`` is the input element
    widened to f32 (exact for bf16 / f16, the identity for fp32, so the term is the element itself: one class per
    format, not per input dtype)."""

    def __init__(
        self,
        K: int,
        fmt: MXFormat = MXFP8,
        rule: ScaleRule = DEFAULT_RULE,
        alg: Optional[TermAlgebra] = None,
    ):
        self.K, self.fmt, self.rule = K, fmt, rule
        self.nb = num_blocks(K, BLOCK)
        self.alg = alg or TermAlgebra()

    def leaf(self, row: int, i: int) -> Term:
        return self.alg.x(row, i)

    def block_reduction(self, row: int, b: int) -> Reduction:
        return block_reduction(lambda fp, e: self.leaf(row, BLOCK * b + e))

    def row_reduction(self, row: int) -> Reduction:
        return row_reduction(self.K, lambda fp, i: self.leaf(row, i))

    def block_amax(self, row: int, b: int) -> Term:
        return self.block_reduction(row, b).canonical(self.alg)

    def block(self, row: int, b: int) -> Dict[str, Any]:
        a = self.alg
        amax = self.block_amax(row, b)
        sf_d, inv = scale_chain(a, amax, self.fmt, self.rule)
        codes = [
            quant_elem(a, self.leaf(row, BLOCK * b + e), inv, self.fmt, self.rule)
            for e in range(BLOCK)
        ]
        return {"amax": amax, "sf": sf_d, "inv": inv, "codes": codes}

    def outputs(self, row: int) -> Dict[str, List[Term]]:
        """``{"codes": [K element-value terms], "sf": [nb UE8M0-value terms]}``."""
        codes: List[Term] = []
        sfs: List[Term] = []
        for b in range(self.nb):
            blk = self.block(row, b)
            codes += blk["codes"]
            sfs.append(blk["sf"])
        return {"codes": codes, "sf": sfs}


# The pinned fingerprints. `mxfp8.quant`: TE's chain exactly -- the fp32(1/448) normaliser, RCEIL cvt.rp.satfinite.ue8m0,
# the exact 2^(127 - e8) inverse at every byte incl. 0, the bare cvt.rn.satfinite e4m3 of an fp32 multiply, -0 products keep
# their sign. The OCP floor rule (`OCP_FLOOR_RULE`: lower
# reconstruction error, but it clips the block maximum) is a documented, unpinned variant.
FINGERPRINT_MXFP8 = "mxquant-mxfp8-9018bd1627b3e72a"
#: ``mxfp4.quant``: the same conventions carried to e2m1 (the binary32 1/6 normaliser, the ceiling UE8M0 conversion, the
#: exact inverse at every byte, the bare ``cvt.rn.satfinite`` e2m1 of an fp32 multiply, -0 products keep their sign): parity
#: with FlashInfer's and TensorRT-LLM's MXFP4 quantizers. (The OCP floor rule measures a lower reconstruction error but clips
#: the block maximum in a third of the blocks; it stays a documented, unpinned constant: ``OCP_FLOOR_RULE``.)
FINGERPRINT_MXFP4 = "mxquant-mxfp4-af5e163620dac20c"
FINGERPRINTS = {"mxfp8": FINGERPRINT_MXFP8, "mxfp4": FINGERPRINT_MXFP4}


def spec_fingerprint(fmt: MXFormat, rule: ScaleRule = DEFAULT_RULE) -> str:
    """Identifier of the numerical specification of the class ``<fmt>.quant``: the structural hash of block 0's scale
    term, first and last code terms, with the format's and the rule's names. Independent of K, of the input dtype and
    of the scale-factor layout (a block's bits depend on its 32 inputs alone; the layout is a placement of the store)."""
    s = MXQuantSpec(BLOCK, fmt, rule)
    blk = s.block(0, 0)
    parts = [
        fmt.name,
        rule.name,
        sexpr(blk["sf"]),
        sexpr(blk["codes"][0]),
        sexpr(blk["codes"][BLOCK - 1]),
    ]
    return (
        f"mxquant-{fmt.name}-"
        + hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]
    )
