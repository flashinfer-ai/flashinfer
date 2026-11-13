# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Generate ``lean/BIKernels/SamplingCases.lean`` from the Python side: the ``lse.row`` class's conformance cases.

Per vocabulary of a small set (one slot, one block, a partial block, two and four blocks) and per placement of the
configuration space (one per thread count and cluster size; the rows per CTA and the vector width do not enter the
placement): the Lean placement with its ``PlacementOK`` theorem (``decide +kernel``), the engine's maximum at unit
(0, 0, 0) (``canon.trace``: the same ``init_state`` / ``apply_step`` code path the kernel executes at trace time) and
the spec's, the engine's sum over its own maximum and the spec's sum, and ``lse_end_to_end`` instantiated on the
placement. Per vocabulary and temperature form: the spec's first and last probability, first log-softmax element and
``lse``.

The S-expressions grow with the vocabulary (the sum's prints the maximum's once per element), so the kernel-decided
string pins (``decide +kernel``) are emitted for the small vocabularies only (:data:`PIN_NS`: the maximum's terms;
:data:`PIN_SUM_NS`: the sum's and the outputs'); every case, pinned or not, is printed by ``lake exe
samplingconformance`` and compared with the Python S-expressions by ``tests/batch_invariant/test_sampling_lean.py``.
The sum's and the outputs' terms are carried for the vocabularies in :data:`SUM_NS` (the largest vocabulary carries
the maximum and the placements only). No ``native_decide``: the file adds nothing to the axiom policy. A change to
``canon.apply_step``, a placement or the spec's chains that is not reflected in the Lean model makes ``lake build``
fail; a regenerated file that differs from the committed one fails the device-free test, which needs no toolchain.

The draw class ``sample.icdf.v1`` (``draw_spec.py`` vs ``Draw.lean``): per toy vocabulary of :data:`DRAW_NS` the total,
the threshold and the inclusive prefixes of the probe elements of row 0 (the first element, the end of the first slot,
the first elements of the second slot and of lane 1 -- a block bit, a lane bit -- and the last), pinned by kernel
``decide`` for the vocabularies of :data:`DRAW_PIN_NS` (the weights are leaves, so the strings stay a few kilobytes) and
printed for every one. The stream (``stream.py`` vs ``Stream.lean``): the blocks of the keys of :data:`STREAM_KEYS`
decided against the Lean model and printed.

    python -m flashinfer.batch_invariant.sampling.lean_cases     # rewrite lean/BIKernels/SamplingCases.lean
"""

from __future__ import annotations

import functools
import os
from typing import Dict, List, Tuple

from .. import canon
from . import draw_spec as dspec
from . import schedule as ssched
from . import spec as sspec
from . import sterm as term
from . import stream

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
LSE_OUT_WHICH = ("probs0", "probsL", "logp0", "lse")
LSE_KINDS = ("maxkernel", "maxspec", "sumkernel", "sumspec")
#: The draw's toy vocabularies: one block (8, 40), two blocks (300: a block bit), four blocks (1000).
DRAW_NS = [8, 40, 300, 1000]
#: The vocabularies whose draw terms are pinned in Lean by ``decide +kernel``.
DRAW_PIN_NS = (8, 40, 300)
#: The stream's generated vectors, ``(seed, offset, block)``: a zero key, one key at three blocks, a negative seed (its
#: 64-bit pattern), a large offset, the first block of the high counter word.
STREAM_KEYS = [
    (0, 0, 0),
    (42, 7, 0),
    (42, 7, 1),
    (42, 7, 5),
    (-1, 1 << 40, 0),
    (1234, 7, 3),
    ((1 << 63) - 1, 3, 1 << 32),
]


def lean_factor(radix: int, level: str) -> str:
    bits = radix.bit_length() - 1
    kind = {"reg": ".reg", "lane": ".lane", "warp": ".xchg", "cta": ".xchg"}[level]
    return f"⟨{bits}, {kind}⟩"


def lean_str(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _with_refs(sx: str, refs: List[Tuple[str, str]]) -> str:
    """A Python S-expression as a Lean string expression: the named ``def`` wherever one of ``refs`` (``(sub, name)``,
    longest first) occurs, literals in between."""
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
    return " ++ ".join(lean_str(p) if kind == "lit" else p for kind, p in parts) or '""'


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
    """Per placement: the engine's maximum and (for the vocabularies of :data:`SUM_NS`, over its own maximum) sum at unit
    (0, 0, 0), and the spec's."""
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
                case.update(sumkernel=term.sexpr(s_k), sumspec=term.sexpr(s_spec))
            out.append(case)
    return tuple(out)


@functools.lru_cache(maxsize=None)
def lse_out_cases() -> Tuple[Dict, ...]:
    """Per vocabulary of :data:`SUM_NS` and temperature form: the spec's first / last probability, the first log-softmax
    element and the lse of row 0."""
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
                )
            )
    return tuple(out)


def draw_probe(N: int) -> List[int]:
    """The probe elements of a vocabulary: the first, the end of the first slot, the first two of the second slot, the
    ends of the first block, the first elements of lane 1 (a lane-bit node), the last."""
    return sorted(
        {0, 7, 8, 9, 255, 256, 257, 8 * 32, 8 * 32 + 9, N - 1} & set(range(N))
    )


@functools.lru_cache(maxsize=None)
def draw_cases() -> Tuple[Dict, ...]:
    """Per vocabulary of :data:`DRAW_NS`: the total, the threshold and the inclusive prefixes of the probe elements of
    row 0 (the probability form: the weights are the row's leaves)."""
    out = []
    for N in DRAW_NS:
        sp = dspec.DrawSpec(N)
        kb = ssched.padded_blocks(N).bit_length() - 1
        out.append(
            dict(
                N=N,
                kb=kb,
                threshold=term.sexpr(sp.threshold(0)),
                total=term.sexpr(sp.total(0)),
                prefixes={i: term.sexpr(sp.prefix_incl(0, i)) for i in draw_probe(N)},
            )
        )
    return tuple(out)


def stream_cases() -> Tuple[Tuple[int, int, int, Tuple[int, int, int, int]], ...]:
    """``(seed, offset, block, words)`` of :data:`STREAM_KEYS`, the keys as their unsigned 64-bit patterns."""
    return tuple(
        (s & stream.MASK64, o & stream.MASK64, n, stream.block(s, o, n))
        for s, o, n in STREAM_KEYS
    )


def render() -> str:
    """The content of ``SamplingCases.lean``."""
    L = [
        "import BIKernels.Canon",
        "import BIKernels.QuantExport",
        "import BIKernels.Sampling",
        "import BIKernels.Stream",
        "import BIKernels.Draw",
        "namespace BIK",
        "-- GENERATED by flashinfer/batch_invariant/sampling/lean_cases.py from the Python engine / spec; do not edit.",
        "set_option maxRecDepth 8192",
        "",
        "/-! The row class `lse.row` (`sampling/spec.py`, `sampling/schedule.py` vs `Sampling.lean`): per placement the "
        "engine's maximum and sum (over its own maximum) at unit (0, 0, 0) and the spec's; per vocabulary and temperature "
        "form the spec's first / last probability, first log-softmax element and lse, node for node. The string pins are "
        "decided in the kernel for the small vocabularies; every case is printed by `lake exe samplingconformance`. -/",
        "def lseRow : Nat → Term := fun i => Term.x i",
        "",
    ]
    lcs, los = lse_cases(), lse_out_cases()
    mdefs: Dict[int, Tuple[str, str]] = {}  # N -> (m def, m sx)
    sdefs: Dict[int, Tuple[str, str]] = {}  # N -> (s def, s sx), N in SUM_NS
    for N in LSE_NS:
        c0 = next(c for c in lcs if c["N"] == N)
        mref = f"lsePy{N}_m"
        mdefs[N] = (mref, c0["maxspec"])
        L.append(f"def {mref} : String := {lean_str(c0['maxspec'])}")
        if N in SUM_NS:
            sref = f"lsePy{N}_s"
            sdefs[N] = (sref, c0["sumspec"])
            L.append(
                f"def {sref} : String := {_with_refs(c0['sumspec'], [(c0['maxspec'], mref)])}"
            )
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
        L.append(
            f'theorem lse{i}_inv : Sampling.OutputsEquiv (Sampling.kernelOutputs "bf16" {N} {kb} lseRow none lsePl{i} (zeroCoords (flatten lsePl{i}))) (Sampling.specOutputs "bf16" {N} {kb} lseRow none) :='
        )
        L.append(f"  Sampling.lse_end_to_end _ _ _ _ _ _ _ lse{i}_ok (coordsOK_zero _)")
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
        T = 'some (Term.sym "T")' if c["temp"] else "none"
        tag = f"lse{N}_{'T' if c['temp'] else 'noT'}"
        refs = [(s_sx, sref), (m_sx, mref)] if not c["temp"] else [(m_sx, mref)]
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
    L.append(
        "def lseOutCases : List (Nat × Nat × Bool) := ["
        + ", ".join(
            f"({c['N']}, {c['kb']}, {'true' if c['temp'] else 'false'})" for c in los
        )
        + "]"
    )
    # ---- the draw (Draw.lean vs sampling/draw_spec.py)
    L.append("")
    L.append(
        "/-! The draw `sample.icdf.v1` (`sampling/draw_spec.py` vs `Draw.lean`): the total, the threshold and the inclusive "
        "prefixes of probe elements of row 0, node for node (the probability form: the weights are the row's leaves). The "
        "string pins are decided in the kernel for the small vocabularies; every case is printed by the conformance "
        "executable. -/"
    )
    L.append("def drawRow : Nat → Term := fun i => Term.x i")
    dcs = draw_cases()
    for c in dcs:
        N, kb = c["N"], c["kb"]
        tref = f"drawPy{N}_total"
        L.append(f"def {tref} : String := {lean_str(c['total'])}")
        if N not in DRAW_PIN_NS:
            continue
        L.append(
            f"theorem draw{N}_total : osexpr (Draw.total {N} {kb} drawRow) = {tref} := by decide +kernel"
        )
        L.append(
            f"theorem draw{N}_threshold : osexpr (Draw.threshold {N} {kb} drawRow) = {_with_refs(c['threshold'], [(c['total'], tref)])} := by decide +kernel"
        )
        for i, sx in c["prefixes"].items():
            L.append(
                f"theorem draw{N}_p{i} : osexpr (Draw.prefixIncl {N} {kb} drawRow {i}) = {lean_str(sx)} := by decide +kernel"
            )
    L.append(
        "def drawCases : List (Nat × Nat × List Nat) := ["
        + ", ".join(
            f"({c['N']}, {c['kb']}, [{', '.join(str(i) for i in c['prefixes'])}])"
            for c in dcs
        )
        + "]"
    )
    # ---- the stream (Stream.lean vs sampling/stream.py)
    L.append("")
    L.append(
        "/-! The stream (`sampling/stream.py` vs `Stream.lean`): the blocks of a few keys, the Python specification's words "
        "decided against the Lean model (the published Random123 / SplitMix64 vectors are decided in `Stream.lean` itself). -/"
    )
    scs = stream_cases()
    for i, (s, o, n, w) in enumerate(scs):
        L.append(
            f"theorem stream{i} : Stream.block {s} {o} {n} = ⟨{w[0]}, {w[1]}, {w[2]}, {w[3]}⟩ := by decide"
        )
    L.append(
        "def streamCases : List (UInt64 × UInt64 × UInt64) := ["
        + ", ".join(f"({s}, {o}, {n})" for s, o, n, _ in scs)
        + "]"
    )
    L.append("end BIK")
    return "\n".join(L) + "\n"


def summary() -> str:
    lcs = lse_cases()
    return (
        f"sampling lean_cases coverage: placements={len(lcs)} (with sum={sum(c['with_sum'] for c in lcs)}) "
        f"output cases={len(lse_out_cases()) * len(LSE_OUT_WHICH)} "
        f"draw cases={sum(len(c['prefixes']) + 2 for c in draw_cases())} stream cases={len(stream_cases())}"
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
