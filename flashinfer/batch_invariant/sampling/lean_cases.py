# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Generate ``lean/BIKernels/SamplingCases.lean`` from the Python side: the ``lse.row`` class's conformance cases.

Per vocabulary of a small set (one slot, one block, a partial block, two and four blocks) and per placement of the
configuration space (one per thread count and cluster size; the rows per CTA and the vector width do not enter the
placement): the Lean placement with its ``PlacementOK`` theorem (``decide +kernel``), the engine's maximum at unit
(0, 0, 0) (``canon.trace``: the same ``init_state`` / ``apply_step`` code path the kernel executes at trace time) and
the spec's, the engine's sum over its own maximum and the spec's sum, the entropy member's tree ``W`` over the engine's
own maximum and the spec's, and ``lse_end_to_end`` / ``entropy_end_to_end`` instantiated on the placement. Per
vocabulary and temperature form: the spec's first and last probability, first log-softmax element, ``lse`` and
``entropy``.

The S-expressions grow with the vocabulary (the sum's prints the maximum's once per element), so the kernel-decided
string pins (``decide +kernel``) are emitted for the small vocabularies only (:data:`PIN_NS`: the maximum's terms;
:data:`PIN_SUM_NS`: the sum's and the outputs'); every case, pinned or not, is printed by ``lake exe
samplingconformance`` and compared with the Python S-expressions by ``tests/batch_invariant/test_sampling_lean.py``.
The sum's and the outputs' terms are carried for the vocabularies in :data:`SUM_NS` (the largest vocabulary carries
the maximum and the placements only). No ``native_decide``: the file adds nothing to the axiom policy. A change to
``canon.apply_step``, a placement or the spec's chains that is not reflected in the Lean model makes ``lake build``
fail; a regenerated file that differs from the committed one fails the device-free test, which needs no toolchain.

The tensor-parallel protocol (``sampling/shards.py``, ``SamplingTP.lean``) adds, per toy tiling of :data:`TP_CASES`, the
merged maximum and sum of the ranks' pieces (a term identity with the spec's: the maximum and three pieces pinned in the
kernel at N = 300, every case printed and compared), the tiling fact and the protocol theorems instantiated; per
production sharding of :data:`TP_PROD` the tiling fact, the protocol theorem for any row and temperature, and the
layout table printed by the executable against the Python ``ShardLayout`` tables.
    python -m flashinfer.batch_invariant.sampling.lean_cases     # rewrite lean/BIKernels/SamplingCases.lean
"""

from __future__ import annotations

import functools
import os
from typing import Dict, List, Tuple

from .. import canon
from . import schedule as ssched
from . import spec as sspec
from . import shards as sshards
from . import sterm as term

LEAN_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lean"
)
CASES_PATH = os.path.join(LEAN_DIR, "BIKernels", "SamplingCases.lean")

#: The vocabularies: one slot, one block, a partial block, two blocks (a warp exchange, a cluster of 2), four blocks
#: (a warp exchange under a cluster exchange).
LSE_NS = [8, 64, 100, 300, 1000]
#: The vocabularies whose sum and output terms are carried (printed and compared).
SUM_NS = (8, 64, 100, 300)
#: The vocabularies whose maximum terms are pinned in Lean by ``decide +kernel``.
PIN_NS = (8, 64, 100, 300)
#: The vocabularies whose sum and output terms are pinned in Lean by ``decide +kernel``.
PIN_SUM_NS = (8,)
LSE_OUT_WHICH = ("probs0", "probsL", "logp0", "lse", "entropy")
LSE_KINDS = ("maxkernel", "maxspec", "sumkernel", "sumspec", "wkernel", "wspec")


def lean_factor(radix: int, level: str) -> str:
    bits = radix.bit_length() - 1
    kind = {"reg": ".reg", "lane": ".lane", "warp": ".xchg", "cta": ".xchg"}[level]
    return f"⟨{bits}, {kind}⟩"


def lean_str(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _with_refs(sx: str, refs: List[Tuple[str, str]]) -> str:
    """A Python S-expression as a Lean string expression: the named ``def`` wherever one of ``refs`` (``(sub, name)``,
    longest first) occurs, literals in between."""
    parts = _ref_parts(sx, refs)
    return " ++ ".join(lean_str(p) if kind == "lit" else p for kind, p in parts) or '""'


#: Pieces per ``++`` chain of a generated string def: a chain of ~2000 pieces (the entropy tree's string, which repeats
#: the maximum three times per element) overflows Lean's stack while elaborating; the chunked defs keep every chain short.
STR_CHUNK = 128


def _emit_str_def(L: List[str], name: str, sx: str, refs: List[Tuple[str, str]]) -> str:
    """Emit ``def name : String := ...`` for ``_with_refs(sx, refs)``, split into chunked helper defs (``name_0 ++ name_1
    ++ ...``) when the concatenation has more than :data:`STR_CHUNK` pieces. Returns ``name``."""
    parts = _ref_parts(sx, refs)
    exprs = [lean_str(p) if kind == "lit" else p for kind, p in parts] or ['""']
    if len(exprs) <= STR_CHUNK:
        L.append(f"def {name} : String := {' ++ '.join(exprs)}")
        return name
    subs = []
    for i in range(0, len(exprs), STR_CHUNK):
        L.append(
            f"def {name}_{i // STR_CHUNK} : String := {' ++ '.join(exprs[i : i + STR_CHUNK])}"
        )
        subs.append(f"{name}_{i // STR_CHUNK}")
    assert len(subs) <= STR_CHUNK, (name, len(exprs))
    L.append(f"def {name} : String := {' ++ '.join(subs)}")
    return name


def _ref_parts(sx: str, refs: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """The pieces of :func:`_with_refs`: ``("lit", text)`` and ``("ref", def name)`` in order."""
    parts: List[Tuple[str, str]] = [("lit", sx)]
    for sub, name in sorted(refs, key=lambda r: -len(r[0])):
        nxt: List[Tuple[str, str]] = []
        for kind, p in parts:
            if kind != "lit" or sub not in p:
                nxt.append((kind, p))
                continue
            pieces = p.split(sub)
            for i, piece in enumerate(pieces):
                if piece:
                    nxt.append(("lit", piece))
                if i < len(pieces) - 1:
                    nxt.append(("ref", name))
        parts = nxt
    return parts


def lse_placements(N: int):
    """One placement per (threads_per_row, cluster) at register staging; the cluster members at C = 2 and 4 only, so
    the cases stay bounded."""
    seen, out = set(), []
    for cfg in ssched.all_valid_configs(N, 2):
        key = (cfg.threads_per_row, cfg.cluster)
        if cfg.stage == "regs" and cfg.cluster in (1, 2, 4) and key not in seen:
            seen.add(key)
            out.append(cfg)
    return out


@functools.lru_cache(maxsize=None)
def lse_cases() -> Tuple[Dict, ...]:
    """Per placement: the engine's maximum and (for the vocabularies of :data:`SUM_NS`, over its own maximum) sum and
    entropy tree ``W`` at unit (0, 0, 0), and the spec's."""
    out = []
    for N in LSE_NS:
        sp = sspec.LSESpec(N)
        red_m = sp.max_reduction(0)
        m_spec = sp.m(0)
        s_spec = sp.s(0) if N in SUM_NS else None
        kb = ssched.padded_blocks(N).bit_length() - 1
        for cfg in lse_placements(N):
            pl = cfg.placement(N)
            m_k = canon.trace(canon.plan(red_m, pl), sp.alg)[(0, 0, 0)]
            case = dict(
                N=N,
                kb=kb,
                T=cfg.threads_per_row,
                C=cfg.cluster,
                name=cfg.describe(),
                ps="["
                + ", ".join(
                    "["
                    + ", ".join(lean_factor(f.radix, f.level) for f in pl.of(name))
                    + "]"
                    for name in ("block", "lane")
                )
                + "]",
                maxkernel=term.sexpr(m_k),
                maxspec=term.sexpr(m_spec),
                with_sum=N in SUM_NS,
            )
            if N in SUM_NS:
                nm_k, c = sp.alg.neg(m_k), sp.c()
                red_s = sspec.sum_reduction(
                    N,
                    lambda fp, i, nm_k=nm_k, c=c: sspec.weight(
                        fp, sspec.shifted(fp, sp.leaf(0, i), nm_k, c)
                    ),
                )
                s_k = canon.trace(canon.plan(red_s, pl), sp.alg)[(0, 0, 0)]

                def leaf_w(
                    fp, i, nm_k=nm_k, c=c
                ):  # the entropy member's leaf over the engine's own maximum
                    z = sspec.shifted(fp, sp.leaf(0, i), nm_k, c)
                    return sspec.wprod(fp, sspec.weight(fp, z), z)

                w_k = canon.trace(canon.plan(sspec.w_reduction(N, leaf_w), pl), sp.alg)[
                    (0, 0, 0)
                ]
                case.update(
                    sumkernel=term.sexpr(s_k),
                    sumspec=term.sexpr(s_spec),
                    wkernel=term.sexpr(w_k),
                    wspec=term.sexpr(sp.W(0)),
                )
            out.append(case)
    return tuple(out)


@functools.lru_cache(maxsize=None)
def lse_out_cases() -> Tuple[Dict, ...]:
    """Per vocabulary of :data:`SUM_NS` and temperature form: the spec's first / last probability, the first log-softmax
    element, the lse and the entropy of row 0."""
    out = []
    for N in SUM_NS:
        kb = ssched.padded_blocks(N).bit_length() - 1
        for temp in (False, True):
            sp = sspec.LSESpec(N, temperature=temp)
            o = sp.outputs(0)
            out.append(
                dict(
                    N=N,
                    kb=kb,
                    temp=temp,
                    m=term.sexpr(o["m"]),
                    s=term.sexpr(o["s"]),
                    probs0=term.sexpr(o["probs"][0]),
                    probsL=term.sexpr(o["probs"][N - 1]),
                    logp0=term.sexpr(o["logprobs"][0]),
                    lse=term.sexpr(o["lse"]),
                    W=term.sexpr(o["W"]),
                    entropy=term.sexpr(o["entropy"]),
                )
            )
    return tuple(out)


def entropy_chain_sexpr() -> str:
    """The entropy chain ``H = mul(add(lg2(s), neg(div(W, s))), LN2)`` over the symbolic statistics ``W`` and ``s``: the
    string the Lean pin `lse_entropy_chain` and the conformance line ``lsechain`` carry."""
    alg = term.TermAlgebra()
    s, W = alg.sym("s"), alg.sym("W")
    return term.sexpr(sspec.entropy_nat(alg, sspec.lg2(alg, s), W, s))


# ---- the tensor-parallel protocol (`sampling/shards.py` vs `SamplingTP.lean`): toy tilings (two shards; four equal shards; a
# one-slot first shard with a partial-slot-free tail) on the class's toy vocabularies, and the production layouts as tables
#: The toy tilings ``(N, bounds)``.
TP_CASES: List[Tuple[int, List[Tuple[int, int]]]] = [
    (300, [(0, 152), (152, 300)]),
    (1000, [(0, 248), (248, 504), (504, 752), (752, 1000)]),
    (1000, [(0, 8), (8, 504), (504, 1000)]),
]
#: The production shardings ``(N, TP)``: the trainers' vocabularies at their tensor-parallel degrees and Megatron's padded
#: vocabularies (``make_vocab_size_divisible_by=128``: 152064 -> 152576 and 128256 -> 129024 at TP = 8).
TP_PROD: List[Tuple[int, int]] = [
    (32000, 8),
    (32768, 2),
    (128256, 8),
    (151936, 4),
    (152064, 8),
    (262144, 8),
    (152576, 8),
    (129024, 8),
]
#: The toy vocabularies whose merged maximum and pieces are pinned by ``decide +kernel`` (the merged sums' strings are the
#: class's sum strings: carried and compared through the executable, as the row class's are at these vocabularies).
TP_PIN_NS = (300,)


@functools.lru_cache(maxsize=None)
def tp_cases() -> Tuple[Dict, ...]:
    """Per toy tiling: the spec's ``m`` and ``s`` (the merge rebuilds them: a term identity checked on the Python side by
    ``shards.verify_merge`` and proved on the Lean side by ``tp_protocol``) and the first pieces of rank 1 at lanes 0 and
    31 as S-expressions."""
    out = []
    for N, bounds in TP_CASES:
        sp = sspec.LSESpec(N)
        kb = ssched.padded_blocks(N).bit_length() - 1
        red_m = sp.max_reduction(0)
        for red in (red_m, sp.sum_reduction(0)):
            sshards.verify_merge(red, sp.alg, bounds, N)
        lay = sshards.ShardLayout(bounds[1][0], bounds[1][1], N)
        parts = sshards.shard_block_partials(red_m, sp.alg, lay.lo, lay.hi, N)
        pieces = [
            (lane, L, i, term.sexpr(parts[("block", lane, L, i)]))
            for lane in (0, 31)
            for (L, i) in lay.pieces[lane][:3]
        ]
        out.append(
            dict(
                N=N,
                kb=kb,
                bounds=bounds,
                m=term.sexpr(sp.m(0)),
                s=term.sexpr(sp.s(0)),
                pieces=pieces,
            )
        )
    return tuple(out)


def tp_layout_cases() -> List[Tuple[str, int, List[Tuple[int, int]], int, str]]:
    """Per production ``(N, TP)``: the name, ``N``, the bounds, ``kb`` and the literal of every rank's every lane's pieces
    sorted by ``(level, idx)`` -- the string ``toString (SamplingTP.layoutTable kb bounds)`` prints."""
    out = []
    for N, tp in TP_PROD:
        bounds = sshards.tp_shards(N, tp)
        kb = ssched.padded_blocks(N).bit_length() - 1
        ranks = []
        for lo, hi in bounds:
            lay = sshards.ShardLayout(lo, hi, N)
            ranks.append(
                "["
                + ", ".join(
                    "["
                    + ", ".join(f"({L}, {i})" for (L, i) in sorted(lay.pieces[lane]))
                    + "]"
                    for lane in range(32)
                )
                + "]"
            )
        out.append((f"{N}_{tp}", N, bounds, kb, "[" + ", ".join(ranks) + "]"))
    return out


def render() -> str:
    """The content of ``SamplingCases.lean``."""
    L = [
        "import BIKernels.Canon",
        "import BIKernels.QuantExport",
        "import BIKernels.Sampling",
        "import BIKernels.SamplingTP",
        "namespace BIK",
        "-- GENERATED by flashinfer/batch_invariant/sampling/lean_cases.py from the Python engine / spec; do not edit.",
        "set_option maxRecDepth 8192",
        "",
        "/-! The row class `lse.row` (`sampling/spec.py`, `sampling/schedule.py` vs `Sampling.lean`): per placement the "
        "engine's maximum, sum and entropy tree W (the latter two over its own maximum) at unit (0, 0, 0) and the spec's; "
        "per vocabulary and temperature form the spec's first / last probability, first log-softmax element, lse and "
        "entropy, node for node. The string pins are decided in the kernel for the small vocabularies (the entropy chain "
        "once, over symbolic statistics); every case is printed by `lake exe samplingconformance`. -/",
        "def lseRow : Nat → Term := fun i => Term.x i",
        "",
    ]
    lcs, los = lse_cases(), lse_out_cases()
    mdefs: Dict[int, Tuple[str, str]] = {}  # N -> (m def, m sx)
    sdefs: Dict[int, Tuple[str, str]] = {}  # N -> (s def, s sx), N in SUM_NS
    wdefs: Dict[int, Tuple[str, str]] = {}  # N -> (W def, W sx), N in SUM_NS
    for N in LSE_NS:
        c0 = next(c for c in lcs if c["N"] == N)
        mref = f"lsePy{N}_m"
        mdefs[N] = (mref, c0["maxspec"])
        L.append(f"def {mref} : String := {lean_str(c0['maxspec'])}")
        if N in SUM_NS:
            sref = f"lsePy{N}_s"
            sdefs[N] = (sref, c0["sumspec"])
            _emit_str_def(L, sref, c0["sumspec"], [(c0["maxspec"], mref)])
            wref = f"lsePy{N}_W"
            wdefs[N] = (wref, c0["wspec"])
            _emit_str_def(L, wref, c0["wspec"], [(c0["maxspec"], mref)])
    for i, c in enumerate(lcs):
        N, kb = c["N"], c["kb"]
        mref, m_sx = mdefs[N]
        assert c["maxspec"] == m_sx, (N, c["name"])
        L.append(
            f"def lsePl{i} : List (List Factor) := {c['ps']}  -- N={N} {c['name']}"
        )
        L.append(
            f"theorem lse{i}_ok : PlacementOK [{kb}, 5] lsePl{i} := by decide +kernel"
        )
        if N in PIN_NS:
            rhs = mref if c["maxkernel"] == m_sx else lean_str(c["maxkernel"])
            L.append(
                f"theorem lse{i}_max_kernel : osexpr (Sampling.kernelMax {N} {kb} lseRow lsePl{i} (zeroCoords (flatten lsePl{i}))) = {rhs} := by decide +kernel"
            )
            L.append(
                f"theorem lse{i}_max_spec : osexpr (Sampling.specMax {N} {kb} lseRow) = {mref} := by decide +kernel"
            )
        if N in PIN_SUM_NS:
            sref, s_sx = sdefs[N]
            assert c["sumspec"] == s_sx
            rhs = (
                sref
                if c["sumkernel"] == s_sx
                else _with_refs(c["sumkernel"], [(m_sx, mref)])
            )
            L.append(
                f"theorem lse{i}_sum_kernel : osexpr (Sampling.kernelSumOfMax {N} {kb} lseRow none lsePl{i} (zeroCoords (flatten lsePl{i}))) = {rhs} := by decide +kernel"
            )
            L.append(
                f"theorem lse{i}_sum_spec : osexpr (Sampling.specSumOfMax {N} {kb} lseRow none) = {sref} := by decide +kernel"
            )
            wref, w_sx = wdefs[N]
            assert c["wspec"] == w_sx
            rhs = (
                wref
                if c["wkernel"] == w_sx
                else _with_refs(c["wkernel"], [(m_sx, mref)])
            )
            L.append(
                f"theorem lse{i}_w_kernel : osexpr (Sampling.kernelWOfMax {N} {kb} lseRow none lsePl{i} (zeroCoords (flatten lsePl{i}))) = {rhs} := by decide +kernel"
            )
            L.append(
                f"theorem lse{i}_w_spec : osexpr (Sampling.specWOfMax {N} {kb} lseRow none) = {wref} := by decide +kernel"
            )
        L.append(
            f'theorem lse{i}_inv : Sampling.OutputsEquiv (Sampling.kernelOutputs "bf16" {N} {kb} lseRow none lsePl{i} (zeroCoords (flatten lsePl{i}))) (Sampling.specOutputs "bf16" {N} {kb} lseRow none) :='
        )
        L.append(f"  Sampling.lse_end_to_end _ _ _ _ _ _ _ lse{i}_ok (coordsOK_zero _)")
        L.append(
            f"theorem lse{i}_entropy_inv : OEquiv (Sampling.kernelEntropy {N} {kb} lseRow none lsePl{i} (zeroCoords (flatten lsePl{i}))) (Sampling.specEntropy {N} {kb} lseRow none) :="
        )
        L.append(
            f"  Sampling.entropy_end_to_end _ _ _ _ _ _ lse{i}_ok (coordsOK_zero _)"
        )
    L.append(
        "def lseCases : List (Nat × Nat × String × Bool × List (List Factor)) := ["
        + ", ".join(
            f'({c["N"]}, {c["kb"]}, "{c["name"]}", {"true" if c["with_sum"] else "false"}, lsePl{i})'
            for i, c in enumerate(lcs)
        )
        + "]"
    )
    for c in los:
        N, kb = c["N"], c["kb"]
        if N not in PIN_SUM_NS:
            continue
        mref, m_sx = mdefs[N]
        sref, s_sx = sdefs[N]
        wref, w_sx = wdefs[N]
        T = 'some (Term.sym "T")' if c["temp"] else "none"
        tag = f"lse{N}_{'T' if c['temp'] else 'noT'}"
        refs = (
            [(w_sx, wref), (s_sx, sref), (m_sx, mref)]
            if not c["temp"]
            else [(m_sx, mref)]
        )
        L.append(
            f'theorem {tag}_probs0 : osexpr (Sampling.specProb "bf16" {N} {kb} lseRow ({T}) 0) = {_with_refs(c["probs0"], refs)} := by decide +kernel'
        )
        L.append(
            f'theorem {tag}_probsL : osexpr (Sampling.specProb "bf16" {N} {kb} lseRow ({T}) {N - 1}) = {_with_refs(c["probsL"], refs)} := by decide +kernel'
        )
        L.append(
            f'theorem {tag}_logp0 : osexpr (Sampling.specLogp "bf16" {N} {kb} lseRow ({T}) 0) = {_with_refs(c["logp0"], refs)} := by decide +kernel'
        )
        L.append(
            f'theorem {tag}_lse : osexpr (Sampling.specLse "bf16" {N} {kb} lseRow ({T})) = {_with_refs(c["lse"], refs)} := by decide +kernel'
        )
    # the entropy chain over abstract statistics: the full entropy strings (the W tree once, the sum twice, ~5 KB at N = 8)
    # are beyond what kernel `decide`'s string comparison takes without a stack overflow, so the chain is pinned once over
    # `sym` statistics (the W tree is pinned above at N = 8) and every full entropy string is printed and compared
    L.append(
        f'theorem lse_entropy_chain : osexpr (some (Sampling.entropyNat (Sampling.lg2 (Term.sym "s")) (Term.sym "W") (Term.sym "s"))) = {lean_str(entropy_chain_sexpr())} := by decide +kernel'
    )
    L.append(
        "def lseOutCases : List (Nat × Nat × Bool) := ["
        + ", ".join(
            f"({c['N']}, {c['kb']}, {'true' if c['temp'] else 'false'})" for c in los
        )
        + "]"
    )
    # ---- the tensor-parallel protocol (`sampling/shards.py` vs `SamplingTP.lean`)
    L.append("")
    L.append(
        "/-! The tensor-parallel protocol (`sampling/shards.py`, `sampling/tp_kernel.py` vs `SamplingTP.lean`): per toy "
        "tiling the merge of the ranks' pieces followed by the lane tree prints the spec's maximum and sum (the maximum and "
        "three pieces of rank 1 pinned in the kernel at N = 300; every case printed by the executable and compared), the "
        "tiling fact, and the theorems `tpMax_eq_specMax` / `tp_protocol` / `tpOutputs_eq_specOutputs` instantiated; per "
        "production sharding the tiling fact, the protocol theorem for any row and temperature, and the layout table "
        "(`layoutTable`) printed by the executable against the Python `ShardLayout` tables. -/"
    )
    tcs = tp_cases()
    for i, c in enumerate(tcs):
        N, kb = c["N"], c["kb"]
        L.append(
            f"def tpB{i} : List (Nat × Nat) := [{', '.join(f'({lo}, {hi})' for lo, hi in c['bounds'])}]  -- N={N} R={len(c['bounds'])}"
        )
        L.append(
            f"theorem tp{i}_tiling : SamplingTP.tiling tpB{i} 0 {N} = true := by decide +kernel"
        )
        if N in TP_PIN_NS:
            mref, m_sx = mdefs[N]
            assert c["m"] == m_sx, N
            L.append(
                f"theorem tp{i}_max : osexpr (SamplingTP.tpMax {N} {kb} lseRow tpB{i}) = {mref} := by decide +kernel"
            )
            for j, (lane, Lv, idx, sx) in enumerate(c["pieces"]):
                L.append(
                    f"theorem tp{i}_piece{j} : osexpr (SamplingTP.subtree .fmaxc {kb} (SamplingTP.laneLeafMax {N} {kb} lseRow (SamplingTP.natToBits 5 {lane})) {Lv} {idx}) = {lean_str(sx)} := by decide +kernel"
                )
        L.append(
            f"theorem tp{i}_max_eq : SamplingTP.tpMax {N} {kb} lseRow tpB{i} = Sampling.specMax {N} {kb} lseRow :="
        )
        L.append(f"  SamplingTP.tpMax_eq_specMax _ _ _ _ tp{i}_tiling (by decide)")
        L.append(
            f"theorem tp{i}_protocol : SamplingTP.tpSumOfMax {N} {kb} lseRow none tpB{i} = Sampling.specSumOfMax {N} {kb} lseRow none :="
        )
        L.append(f"  SamplingTP.tp_protocol _ _ _ _ _ tp{i}_tiling (by decide)")
        L.append(
            f'theorem tp{i}_outputs : SamplingTP.tpOutputs "bf16" {N} {kb} lseRow none tpB{i} = Sampling.specOutputs "bf16" {N} {kb} lseRow none :='
        )
        L.append(
            f"  SamplingTP.tpOutputs_eq_specOutputs _ _ _ _ _ _ tp{i}_tiling (by decide)"
        )
    L.append(
        "def tpCases : List (Nat × Nat × Nat × List (Nat × Nat)) := ["
        + ", ".join(f"({c['N']}, {c['kb']}, {i}, tpB{i})" for i, c in enumerate(tcs))
        + "]"
    )
    L.append(
        "def tpPieceCases : List (Nat × Nat × Nat × Nat × Nat × Nat) := ["
        + ", ".join(
            f"({c['N']}, {c['kb']}, {i}, {lane}, {Lv}, {idx})"
            for i, c in enumerate(tcs)
            for (lane, Lv, idx, _) in c["pieces"]
        )
        + "]"
    )
    tls = tp_layout_cases()
    for name, N, bounds, kb, _table in tls:
        L.append(
            f"def tpProd_{name} : List (Nat × Nat) := [{', '.join(f'({lo}, {hi})' for lo, hi in bounds)}]"
        )
        L.append(
            f"theorem tpProd_{name}_tiling : SamplingTP.tiling tpProd_{name} 0 {N} = true := by decide +kernel"
        )
        L.append(
            f"theorem tpProd_{name}_protocol (row : Nat → Term) (T : Option Term) : SamplingTP.tpSumOfMax {N} {kb} row T tpProd_{name} = Sampling.specSumOfMax {N} {kb} row T :="
        )
        L.append(f"  SamplingTP.tp_protocol _ _ _ _ _ tpProd_{name}_tiling (by decide)")
    L.append(
        "def tpLayoutCases : List (String × Nat × List (Nat × Nat)) := ["
        + ", ".join(f'("{name}", {kb}, tpProd_{name})' for name, _, _, kb, _ in tls)
        + "]"
    )
    L.append("end BIK")
    return "\n".join(L) + "\n"


def summary() -> str:
    lcs = lse_cases()
    return (
        f"sampling lean_cases coverage: placements={len(lcs)} (with sum={sum(c['with_sum'] for c in lcs)}) "
        f"output cases={len(lse_out_cases()) * len(LSE_OUT_WHICH)} tp tilings={len(tp_cases())} tp layouts={len(tp_layout_cases())}"
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
