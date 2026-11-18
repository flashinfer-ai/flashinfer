# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Generate ``lean/BIKernels/QuantCases.lean`` from the Python side: the quantization classes' conformance cases.

For the NVFP4 classes (``nvfp4_spec`` vs ``Quant.lean``) the file carries, per row length of a small set (one, two and
three-of-four blocks: the padding columns and an absent block): the row amax S-expression of the Python spec, every
placement's kernel term (``canon.trace``: the same ``init_state`` / ``apply_step`` code path the kernel executes at trace
time) with the theorems that the Lean model prints the same canonical S-expression (kernel ``decide``), that the placement
is valid (``PlacementOK``, ``decide``) and the engine's invariance theorem instantiated on it; the block amaxes the kernel
reads before the row digit; per class the scale terms of the first and last block, the first and last code and the
per-token scale; and the e2m1 conversion table (``exact.to_e2m1`` on binary32 values, which ``Quant.e2m1Code`` must
reproduce).

For the MX classes (``mx_spec`` vs ``MXQuant.lean``): the three placements of the block digit (``V`` = 32 / 16 / 8
elements per lane), the block amaxes of one row configuration per block order, a 4-lanes-per-block configuration and the
tile placement, per format and recipe rule the scale terms of blocks 0 and 1 and the codes 0 and 63, and the UE8M0
conversion tables under the three roundings (``MXQuant.ue8m0Byte``).

Every generated theorem is decided by ``decide +kernel`` (the kernel evaluates the terms and the string comparisons; the
elaborator's evaluator overflows on the longer S-expressions), so the file adds nothing to the axiom policy. A change to ``canon.apply_step``, a placement, a spec's chain or a recipe constant that is not
reflected in the Lean model makes ``lake build`` fail; a regenerated file that differs from the committed one fails the
device-free test (``tests/batch_invariant/test_quant_lean.py``), which needs no Lean toolchain.

    python -m flashinfer.batch_invariant.quantization.lean_cases     # rewrite lean/BIKernels/QuantCases.lean
"""

from __future__ import annotations

import math
import os
import random
import struct
from typing import Dict, List, Tuple

from .. import canon
from ..fp_interface import exact
from . import mx_schedule as mxsched
from . import mx_spec as mxspec
from . import nvfp4_schedule as qsched
from . import nvfp4_spec as qspec
from . import qterm as term

LEAN_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lean"
)
CASES_PATH = os.path.join(LEAN_DIR, "BIKernels", "QuantCases.lean")

#: NVFP4: Kbp 1 (padding columns), 2 and 4 with three of four blocks present; small enough for kernel ``decide``.
QUANT_KS = [16, 32, 48]
QUANT_VARIANTS = [qspec.PER_TENSOR, qspec.PER_TOKEN]
LEAN_QVARIANT = {
    qspec.PER_TENSOR: "Quant.Variant.perTensor",
    qspec.PER_TOKEN: "Quant.Variant.perTok",
}
QUANT_OUT_WHICH = ("sf0", "sfL", "code0", "codeL", "pts")

MXQ_K = 64  # two blocks: the chains of block 0 and block 1, codes 0 and 63
MXQ_RULES = [
    (mxspec.DEFAULT_RULE, "MXQuant.Rule.te"),
    (mxspec.OCP_FLOOR_RULE, "MXQuant.Rule.ocpFloor"),
    (mxspec.FI_CLAMP_RULE, "MXQuant.Rule.fiClamp"),
    (mxspec.TE_FMA_RULE, "MXQuant.Rule.teFma"),
    (mxspec.ScaleRule(norm="div"), "MXQuant.Rule.div"),
    (mxspec.ScaleRule(round="rn"), "MXQuant.Rule.rn"),
]
MXQ_FMTS = [(mxspec.MXFP8, "MXQuant.Fmt.mxfp8"), (mxspec.MXFP4, "MXQuant.Fmt.mxfp4")]
MXQ_OUT_WHICH = ("sf0", "sf1", "code0", "codeL")


def lean_factor(radix: int, level: str) -> str:
    bits = radix.bit_length() - 1
    kind = {"reg": ".reg", "lane": ".lane", "warp": ".xchg", "cta": ".xchg"}[level]
    return f"⟨{bits}, {kind}⟩"


def lean_str(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _with_ref(sx: str, sub: str, ref: str) -> str:
    """A Python S-expression as a Lean string expression: the ``def`` reference when it is exactly ``sub`` (the row amax
    every output of a size shares), the literal otherwise."""
    return ref if sx == sub else lean_str(sx)


# ----------------------------------------------------------------------------- NVFP4
def _quant_placements(K: int):
    """The row placements at ``K`` (the per-token space, register staging): every valid per-token configuration."""
    return [
        c for c in qsched.all_valid_configs(K, 2, per_token=True) if c.stage == "regs"
    ]


def quant_cases() -> List[Dict]:
    """Per placement of the NVFP4 quantizer at the declared sizes: the Lean placement, the engine's row-amax term at unit
    (0, 0, 0) (``canon.trace``) and the spec's."""
    out = []
    for K in QUANT_KS:
        sp = qspec.QuantSpec(K, qspec.PER_TOKEN)
        red = sp.row_reduction(0)
        spec_sx = term.sexpr(sp.row_amax(0))
        kb = qsched.padded_blocks(K).bit_length() - 1
        for cfg in _quant_placements(K):
            pl = cfg.placement(K)
            digits = []
            for name in ("elem", "blk"):
                digits.append(
                    "["
                    + ", ".join(lean_factor(f.radix, f.level) for f in pl.of(name))
                    + "]"
                )
            got = canon.trace(canon.plan(red, pl), sp.alg)[(0, 0, 0)]
            out.append(
                dict(
                    K=K,
                    kb=kb,
                    name=cfg.describe(),
                    ps=f"[{', '.join(digits)}]",
                    kernel=term.sexpr(got) if got is not None else "absent",
                    spec=spec_sx,
                )
            )
    return out


def quant_out_cases() -> List[Dict]:
    """Per size and class: the scale terms of the first and last block, the first and last code, and the per-token scale
    (``QuantSpec.outputs(0)``), as Python S-expressions."""
    out = []
    for K in QUANT_KS:
        kb = qsched.padded_blocks(K).bit_length() - 1
        for var in QUANT_VARIANTS:
            sp = qspec.QuantSpec(K, var)
            o = sp.outputs(0)
            out.append(
                dict(
                    K=K,
                    kb=kb,
                    variant=var.name,
                    lean=LEAN_QVARIANT[var],
                    rowamax=term.sexpr(sp.row_amax(0)),
                    sf0=term.sexpr(o["sf"][0]),
                    sfL=term.sexpr(o["sf"][-1]),
                    code0=term.sexpr(o["codes"][0]),
                    codeL=term.sexpr(o["codes"][-1]),
                    pts=term.sexpr(o["pts"]) if o["pts"] is not None else "absent",
                )
            )
    return out


def quant_block_cases() -> List[Tuple[int, str, str]]:
    """The block amax terms the kernel reads before the row digit (``canon.trace(..., stop_before="blk")``) at K = 48 for
    one placement per block order: ``(block, name, sexpr)``, compared with ``Quant.blockAmax``."""
    K = 48
    sp = qspec.QuantSpec(K, qspec.PER_TOKEN)
    red = sp.row_reduction(0)
    out = []
    for cfg in (
        qsched.QuantConfig(4, 32, 16, "lane"),
        qsched.QuantConfig(2, 64, 8, "reg"),
    ):
        p = canon.plan(red, cfg.placement(K))
        states = canon.trace(p, sp.alg, stop_before="blk")
        for (cta, w, lane), st in states.items():
            coords = {"lane": lane, "warp": w, "cta": cta, "grid": 0}
            for key, val in st.items():
                b = p.digit_value("blk", coords, key)
                if val is not None:
                    kname = "_".join("n" if v is None else str(v) for v in key)
                    out.append(
                        (b, f"{cfg.describe()}_l{lane}_k{kname}", term.sexpr(val))
                    )
    seen, uniq = set(), []
    for b, name, sx in out:
        key = (b, name.split("_l")[0])
        if key not in seen:
            seen.add(key)
            uniq.append((b, name, sx))
    return uniq


def _odd_form(v: float) -> Tuple[int, int]:
    """``(a, e)`` with ``v = a * 2^e`` exactly and ``a`` odd (``a = 0`` for zero): the canonical form of a binary value; ``a``
    fits the 24-bit significand, so the Lean model's bounded ``log2`` (``MXQuant.log2Fuel``) covers it."""
    num, den = v.as_integer_ratio()
    e = -(den.bit_length() - 1)
    assert den == 1 << -e
    while num and num % 2 == 0:
        num //= 2
        e += 1
    return num, e


def e2m1_cases() -> List[Tuple[bool, int, int, int]]:
    """``(neg, a, e, code)``: the sign and the exact magnitude ``a * 2^e`` (``a`` odd) of a binary32 input and the code
    ``exact.to_e2m1`` assigns, for the rounding boundaries (+-2 ulps), the magnitudes, saturation, both signs (incl. -0)
    and random values; Lean's ``Quant.e2m1Code`` must agree."""
    rng = random.Random(0)
    vals = []
    for m in (
        list(exact._E2M1_MIDPOINTS)
        + list(exact.E2M1_MAGNITUDES)
        + [6.5, 1e30, 2**-130, 2**-149]
    ):
        bits = struct.unpack("<I", struct.pack("<f", m))[0]
        for d in range(-2, 3):
            v = (
                struct.unpack("<f", struct.pack("<I", (bits + d) & 0xFFFFFFFF))[0]
                if m
                else m
            )
            vals += [v, -v]
    vals += [
        struct.unpack("<f", struct.pack("<f", rng.uniform(-8, 8)))[0] for _ in range(60)
    ]
    out = []
    for v in vals:
        if not math.isfinite(v):
            continue
        a, e = _odd_form(abs(float(v)))
        out.append((math.copysign(1.0, v) < 0, a, e, exact.e2m1_code(exact.to_e2m1(v))))
    return out


# ----------------------------------------------------------------------------- MX
def mxquant_cases() -> List[Dict]:
    """Per placement of the block digit: the Lean placement, the engine's block-amax term at unit (0, 0, 0), the spec's."""
    out = []
    sp = mxspec.MXQuantSpec(32, mxspec.MXFP8)
    red = sp.block_reduction(0, 0)
    spec_sx = term.sexpr(sp.block_amax(0, 0))
    for V in (32, 16, 8):
        pl = mxspec.block_amax_placement(V)
        ps = (
            "[["
            + ", ".join(lean_factor(f.radix, f.level) for f in pl.of("elem"))
            + "]]"
        )
        got = canon.trace(canon.plan(red, pl), sp.alg)[(0, 0, 0)]
        out.append(dict(name=f"V{V}", ps=ps, kernel=term.sexpr(got), spec=spec_sx))
    return out


def mxquant_out_cases() -> List[Dict]:
    """Per format and recipe rule: the scale terms of blocks 0 and 1 and the codes 0 and 63 at K = 64."""
    out = []
    for fmt, lf in MXQ_FMTS:
        for rule, lr in MXQ_RULES:
            sp = mxspec.MXQuantSpec(MXQ_K, fmt, rule)
            o = sp.outputs(0)
            out.append(
                dict(
                    fmt=fmt.name,
                    rule=rule.name,
                    lean_fmt=lf,
                    lean_rule=lr,
                    amax0=term.sexpr(sp.block_amax(0, 0)),
                    amax1=term.sexpr(sp.block_amax(0, 1)),
                    sf0=term.sexpr(o["sf"][0]),
                    sf1=term.sexpr(o["sf"][1]),
                    code0=term.sexpr(o["codes"][0]),
                    codeL=term.sexpr(o["codes"][MXQ_K - 1]),
                )
            )
    return out


def mxquant_block_cases() -> List[Tuple[int, str, str]]:
    """The block amax terms the MX kernel reads before the ownership digit at K = 96 (3 of 4 padded blocks present) for
    one row configuration per block order, a 4-lanes-per-block configuration and the tile placement: ``(block, name,
    sexpr)``, compared with ``MXQuant.blockAmax``."""
    out = []
    cases = (
        (96, mxsched.MXQuantConfig(4, 32, 32, "lane")),
        (96, mxsched.MXQuantConfig(2, 64, 16, "reg")),
        (96, mxsched.MXQuantConfig(16, 4, 8, "lane")),
        (
            32 * 64,
            mxsched.MXQuantConfig(32, 128, 32, mode="tile", tile_warps=16, tile_bpt=2),
        ),
    )
    for K, cfg in cases:
        sp = mxspec.MXQuantSpec(K, mxspec.MXFP8)
        red = sp.row_reduction(0)
        p = canon.plan(red, cfg.placement(K))
        states = canon.trace(p, sp.alg, stop_before="blk")
        for (cta, w, lane), st in states.items():
            coords = {"lane": lane, "warp": w, "cta": cta, "grid": 0}
            for key, val in st.items():
                b = p.digit_value("blk", coords, key)
                if val is not None:
                    kname = "_".join("n" if v is None else str(v) for v in key)
                    out.append(
                        (b, f"{cfg.describe()}_l{lane}_k{kname}", term.sexpr(val))
                    )
    seen, uniq = set(), []
    for b, name, sx in out:
        key = (b, name.split("_l")[0])
        if key not in seen and b < 4:
            seen.add(key)
            uniq.append((b, name, sx))
    return uniq


def ue8m0_cases() -> List[Tuple[int, int, int, int]]:
    """``(rnd, a, e, byte)``: the rounding (0 = rp, 1 = rz, 2 = rn), the exact magnitude ``a * 2^e`` (``a`` odd) of a positive binary32
    and the byte ``exact.to_ue8m0`` assigns, for powers of two, their neighbours, the 1.5 x 2^k ties and neighbours, the
    subnormal and saturation ends, ELEM_MAX x 2^k boundaries and random values; Lean's ``MXQuant.ue8m0Byte`` must agree."""
    rng = random.Random(1)
    vals = []
    for k in [
        -149,
        -140,
        -130,
        -128,
        -127,
        -126,
        -125,
        -120,
        -100,
        -30,
        -10,
        -3,
        -2,
        -1,
        0,
        1,
        2,
        3,
        10,
        30,
        100,
        120,
        125,
        126,
        127,
    ]:
        m = 2.0**k
        bits = struct.unpack("<I", struct.pack("<f", m))[0]
        for d in (-1, 0, 1):
            vals.append(struct.unpack("<f", struct.pack("<I", bits + d))[0])
        if k < 127:
            t = struct.unpack("<I", struct.pack("<f", 1.5 * m))[0]
            for d in (-1, 0, 1):
                vals.append(struct.unpack("<f", struct.pack("<I", t + d))[0])
    vals += (
        [448.0 * 2.0**k for k in (-10, 0, 7)]
        + [6.0 * 2.0**k for k in (-10, 0, 7)]
        + [exact.F32_MAX]
    )
    vals += [
        struct.unpack(
            "<f", struct.pack("<f", rng.uniform(0, 1) * 2.0 ** rng.randint(-140, 127))
        )[0]
        for _ in range(24)
    ]
    out = []
    for v in vals:
        if not (v > 0) or v == float("inf"):
            continue
        a, e = _odd_form(float(v))
        for ri, rnd in enumerate(exact.UE8M0_ROUNDINGS):
            out.append((ri, a, e, exact.ue8m0_byte(exact.to_ue8m0(v, rnd))))
    return out


def _chunks(rows, n):
    return [rows[i : i + n] for i in range(0, len(rows), n)]


# ----------------------------------------------------------------------------- the file
def render() -> str:
    """The content of ``QuantCases.lean``."""
    L = [
        "import BIKernels.Canon",
        "import BIKernels.MXQuant",
        "import BIKernels.Quant",
        "import BIKernels.QuantExport",
        "namespace BIK",
        "-- GENERATED by flashinfer/batch_invariant/quantization/lean_cases.py from the Python engine / specs; do not edit.",
        "set_option maxRecDepth 8192",
        "",
        "def quantRow : Nat → Term := fun i => Term.x i",
        "",
    ]
    # ---- NVFP4
    L.append(
        "/-! The NVFP4 quantization classes (`quantization/nvfp4_spec.py`, `nvfp4_schedule.py` vs `Quant.lean`): the row amax "
        "S-expression per size, every placement's kernel term (the engine's `execC .fmaxc` at unit (0,0,0)) and spec term, "
        "the block amaxes read before the row digit, the output chains per class, and the e2m1 conversion table. -/"
    )
    qcs, qos, qbs = quant_cases(), quant_out_cases(), quant_block_cases()
    qdefs: Dict[int, str] = {}
    for c in qos:
        name = f"quantPy{c['K']}_rowamax"
        if c["K"] not in qdefs:
            qdefs[c["K"]] = name
            L.append(f"def {name} : String := {lean_str(c['rowamax'])}")
    for i, c in enumerate(qcs):
        ref = qdefs[c["K"]]
        assert c["spec"] == [o for o in qos if o["K"] == c["K"]][0]["rowamax"]
        kernel_rhs = ref if c["kernel"] == c["spec"] else lean_str(c["kernel"])
        nb = c["K"] // 16
        L.append(
            f"def quantPl{i} : List (List Factor) := {c['ps']}  -- K={c['K']} {c['name']}"
        )
        L.append(
            f"theorem quant{i}_ok : PlacementOK [4, {c['kb']}] quantPl{i} := by decide +kernel"
        )
        L.append(
            f"theorem quant{i}_kernel : osexpr (execC .fmaxc (flatten quantPl{i}) (Quant.rowLeaf {nb} quantRow) (zeroCoords (flatten quantPl{i}))) = {kernel_rhs} := by decide +kernel"
        )
        L.append(
            f"theorem quant{i}_spec : osexpr (Quant.rowAmax {nb} {c['kb']} quantRow) = {ref} := by decide +kernel"
        )
        L.append(
            f"theorem quant{i}_inv : OEquiv (execC .fmaxc (flatten quantPl{i}) (Quant.rowLeaf {nb} quantRow) (zeroCoords (flatten quantPl{i}))) (Quant.rowAmax {nb} {c['kb']} quantRow) :="
        )
        L.append(
            f"  Quant.row_amax_invariance {nb} {c['kb']} quantRow quantPl{i} _ quant{i}_ok (coordsOK_zero _)"
        )
    L.append(
        "def quantCases : List (Nat × Nat × String × List (List Factor)) := ["
        + ", ".join(
            f'({c["K"]}, {c["kb"]}, "{c["name"]}", quantPl{i})'
            for i, c in enumerate(qcs)
        )
        + "]"
    )
    for c in qos:
        ref, K, kb, v = qdefs[c["K"]], c["K"], c["kb"], c["lean"]
        nb = K // 16
        tag = f"quant{K}_{c['variant']}"
        L.append(
            f"theorem {tag}_sf0 : osexpr (Quant.specSf {v} {nb} {kb} quantRow 0) = {_with_ref(c['sf0'], c['rowamax'], ref)} := by decide +kernel"
        )
        L.append(
            f"theorem {tag}_sfL : osexpr (Quant.specSf {v} {nb} {kb} quantRow {nb - 1}) = {_with_ref(c['sfL'], c['rowamax'], ref)} := by decide +kernel"
        )
        L.append(
            f"theorem {tag}_code0 : osexpr (Quant.specCode {v} {nb} {kb} quantRow 0) = {_with_ref(c['code0'], c['rowamax'], ref)} := by decide +kernel"
        )
        L.append(
            f"theorem {tag}_codeL : osexpr (Quant.specCode {v} {nb} {kb} quantRow {K - 1}) = {_with_ref(c['codeL'], c['rowamax'], ref)} := by decide +kernel"
        )
        if c["pts"] != "absent":
            L.append(
                f"theorem {tag}_pts : osexpr (Quant.specPts {nb} {kb} quantRow) = {_with_ref(c['pts'], c['rowamax'], ref)} := by decide +kernel"
            )
    L.append(
        "def quantOutCases : List (Nat × Nat × String × Quant.Variant) := ["
        + ", ".join(
            f'({c["K"]}, {c["kb"]}, "{c["variant"]}", {c["lean"]})' for c in qos
        )
        + "]"
    )
    for i, (b, name, sx) in enumerate(qbs):
        L.append(
            f"theorem quantblock{i} : osexpr (Quant.blockAmax quantRow {b}) = {lean_str(sx)} := by decide +kernel  -- {name}"
        )
    L.append(
        "def quantBlockCases : List (Nat × String) := ["
        + ", ".join(f'({b}, "{name}")' for b, name, _ in qbs)
        + "]"
    )
    e2 = e2m1_cases()
    L.append(
        "/-- `(neg, a, e, code)`: the Python model `exact.to_e2m1` on binary32 values `+-a * 2^e`; `Quant.e2m1Code` must agree. -/"
    )
    for ci, chunk in enumerate(_chunks(e2, 48)):
        L.append(
            f"def e2m1Table{ci} : List (Bool × Nat × Int × Nat) := ["
            + ", ".join(f"({str(n).lower()}, {a}, {e}, {c})" for n, a, e, c in chunk)
            + "]"
        )
        L.append(
            f"theorem e2m1_model_matches{ci} : e2m1Table{ci}.all (fun ⟨n, a, e, c⟩ => Quant.e2m1Code n a e == c) = true := by decide +kernel"
        )
    # ---- MX
    L.append(
        "/-! The MX quantization classes (`quantization/mx_spec.py` vs `MXQuant.lean`): the block amax S-expression, the three block "
        "placements' kernel terms (the engine's `execC .fmaxc` at unit (0,0,0)), the block amaxes the kernel reads before the "
        "ownership digit, the output chains per format and recipe rule, and the UE8M0 conversion tables. -/"
    )
    mqs, mqo, mqb = mxquant_cases(), mxquant_out_cases(), mxquant_block_cases()
    amax_sx = mqs[0]["spec"]
    L.append(f"def mxqPy_amax0 : String := {lean_str(amax_sx)}")
    L.append(f"def mxqPy_amax1 : String := {lean_str(mqo[0]['amax1'])}")
    L.append(
        "theorem mxq_spec_amax0 : osexpr (MXQuant.blockAmax quantRow 0) = mxqPy_amax0 := by decide +kernel"
    )
    L.append(
        "theorem mxq_spec_amax1 : osexpr (MXQuant.blockAmax quantRow 1) = mxqPy_amax1 := by decide +kernel"
    )
    for i, c in enumerate(mqs):
        assert c["spec"] == amax_sx
        rhs = "mxqPy_amax0" if c["kernel"] == amax_sx else lean_str(c["kernel"])
        L.append(f"def mxqPl{i} : List (List Factor) := {c['ps']}  -- {c['name']}")
        L.append(f"theorem mxq{i}_ok : PlacementOK [5] mxqPl{i} := by decide +kernel")
        L.append(
            f"theorem mxq{i}_kernel : osexpr (execC .fmaxc (flatten mxqPl{i}) (MXQuant.blockLeaf quantRow 0) (zeroCoords (flatten mxqPl{i}))) = {rhs} := by decide +kernel"
        )
        L.append(
            f"theorem mxq{i}_inv : OEquiv (execC .fmaxc (flatten mxqPl{i}) (MXQuant.blockLeaf quantRow 0) (zeroCoords (flatten mxqPl{i}))) (MXQuant.blockAmax quantRow 0) :="
        )
        L.append(
            f"  MXQuant.block_amax_invariance quantRow 0 mxqPl{i} _ mxq{i}_ok (coordsOK_zero _)"
        )
    L.append(
        "def mxqCases : List (String × List (List Factor)) := ["
        + ", ".join(f'("{c["name"]}", mxqPl{i})' for i, c in enumerate(mqs))
        + "]"
    )

    def _mx_ref(sx, amax0, amax1):
        if amax0 in sx:
            return _with_ref(sx, amax0, "mxqPy_amax0")
        return _with_ref(sx, amax1, "mxqPy_amax1")

    for c in mqo:
        tag = f"mxq_{c['fmt']}_{c['rule'].replace('-', '_')}"
        f, r = c["lean_fmt"], c["lean_rule"]
        L.append(
            f"theorem {tag}_sf0 : osexpr (MXQuant.specSf {f} {r} quantRow 0) = {_mx_ref(c['sf0'], c['amax0'], c['amax1'])} := by decide +kernel"
        )
        L.append(
            f"theorem {tag}_sf1 : osexpr (MXQuant.specSf {f} {r} quantRow 1) = {_mx_ref(c['sf1'], c['amax0'], c['amax1'])} := by decide +kernel"
        )
        L.append(
            f"theorem {tag}_code0 : osexpr (MXQuant.specCode {f} {r} quantRow 0) = {_mx_ref(c['code0'], c['amax0'], c['amax1'])} := by decide +kernel"
        )
        L.append(
            f"theorem {tag}_codeL : osexpr (MXQuant.specCode {f} {r} quantRow {MXQ_K - 1}) = {_mx_ref(c['codeL'], c['amax0'], c['amax1'])} := by decide +kernel"
        )
    L.append(
        "def mxqOutCases : List (String × String × MXQuant.Fmt × MXQuant.Rule) := ["
        + ", ".join(
            f'("{c["fmt"]}", "{c["rule"]}", {c["lean_fmt"]}, {c["lean_rule"]})'
            for c in mqo
        )
        + "]"
    )
    for i, (b, name, sx) in enumerate(mqb):
        L.append(
            f"theorem mxqblock{i} : osexpr (MXQuant.blockAmax quantRow {b}) = {lean_str(sx)} := by decide +kernel  -- {name}"
        )
    L.append(
        "def mxqBlockCases : List (Nat × String) := ["
        + ", ".join(f'({b}, "{name}")' for b, name, _ in mqb)
        + "]"
    )
    u8 = ue8m0_cases()
    L.append(
        "/-! `(a, e, byte)` per rounding: the Python model `exact.to_ue8m0` on binary32 magnitudes `a * 2^e`; `MXQuant.ue8m0Byte` must agree. -/"
    )
    for ri, rnd in enumerate(("rp", "rz", "rn")):
        rows = [(a, e, c) for r, a, e, c in u8 if r == ri]
        for ci, chunk in enumerate(_chunks(rows, 48)):
            L.append(
                f"def ue8m0Table_{rnd}{ci} : List (Nat × Int × Nat) := ["
                + ", ".join(f"({a}, {e}, {c})" for a, e, c in chunk)
                + "]"
            )
            L.append(
                f"theorem ue8m0_model_matches_{rnd}{ci} : ue8m0Table_{rnd}{ci}.all (fun ⟨a, e, c⟩ => MXQuant.ue8m0Byte .{rnd} a e == c) = true := by decide +kernel"
            )
    L.append("end BIK")
    return "\n".join(L) + "\n"


def summary() -> str:
    return (
        f"quant lean_cases coverage: nvfp4 placements={len(quant_cases())} output cases={len(quant_out_cases())} "
        f"block cases={len(quant_block_cases())} e2m1 cases={len(e2m1_cases())}; mx placements={len(mxquant_cases())} "
        f"output cases={len(mxquant_out_cases())} block cases={len(mxquant_block_cases())} ue8m0 cases={len(ue8m0_cases())}"
    )


def main(out_path: str = CASES_PATH) -> str:
    content = render()
    with open(out_path, "w") as f:
        f.write(content)
    s = summary()
    print(s)
    return s


if __name__ == "__main__":
    main()
