# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Generate ``lean/BIKernels/MoECases.lean`` from the Python side: the bf16 MoE classes' conformance cases.

The file couples the Lean model (``lean/BIKernels/MoE.lean``) to :mod:`.spec` node for node: per case a term the
Python spec builds (:mod:`.mterm` S-expression) and the Lean expression that must print identically
(``MTerm.sexpr``). The chains (the declared exp, sigmoid, tanh, erf and log, the GEMM1 epilogues of every gated and
ungated activation with its parameter sets, the slot-order combine) become theorems decided by ``decide +kernel`` when
their S-expression is at most :data:`THEOREM_MAX_BYTES` (the kernel evaluates the term and the string comparison; the
file adds nothing to the axiom policy); the larger chains (SiTU with a linear scale, PowLU: the log chain inside the exp
chain prints at ~100 KB) and the routing weights of the five named methods (tens to hundreds of KB: the exp chain
repeats per expert) are compared through the ``moeconformance`` executable only
(``tests/batch_invariant/test_moe_lean.py``). A change to a chain, a routing method or a constant that is not
reflected in the Lean model makes ``lake build`` or the conformance comparison fail; a regenerated file that differs
from the committed one fails the device-free test, which needs no Lean toolchain.

    python -m flashinfer.fused_moe.batch_invariant.lean_cases     # rewrite lean/BIKernels/MoECases.lean
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence, Tuple

from . import mterm as term
from . import spec as M

LEAN_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "batch_invariant",
    "lean",
)
CASES_PATH = os.path.join(LEAN_DIR, "BIKernels", "MoECases.lean")

#: A chain whose S-expression is at most this many bytes is a kernel-``decide`` theorem as well as a conformance line;
#: a larger one is a conformance line only (kernel string evaluation of the ~12 KB SiTU and ~100 KB PowLU terms is
#: out of proportion; the executable compares them in milliseconds).
THEOREM_MAX_BYTES = 10_000

#: The routing-weight cases: ``(method, E, selected experts in slot order)``, one row per named method (two for the
#: renormalised softmax: a power-of-two and a padded expert count). The weights of every slot are compared.
ROUTING_CASES: Sequence[Tuple[M.RoutingMethod, int, List[int]]] = (
    (M.SOFTMAX_RENORM, 8, [3, 0]),
    (M.SOFTMAX, 13, [12, 0, 5]),
    (M.SOFTMAX_RENORM, 5, [4, 1, 2]),
    (M.SIGMOID_RENORM, 8, [2, 6]),
    (M.SIGMOID_BIAS_RENORM_SCALED, 8, [1, 7]),
    (M.SIGMOID, 64, [63]),
    # the operator-surface terms: the logits as weights, the sigmoid weights under a selection on the logits, the
    # group-limited DeepSeek-V3 forms (their weight terms are the ungrouped method's: `routingWeights_selection_only`)
    (M.TOPK, 8, [2, 5]),
    (M.TOPK_SIGMOID, 16, [9]),
    (M.SIGMOID_BIAS_RENORM_SCALED.with_groups(4, 2), 16, [3, 9, 12]),
    (M.DEEPSEEK_V3_GROUPED, 64, [17, 40, 5, 63]),
)

#: The gated GEMM1 chains: ``(case name, act, params)`` over ``spec.ACTS`` with one parameter set per form the term can
#: take (``None`` = the scalar FlashInfer's config leaves out). ``swiglu`` (the default SwiGLU) is the ``silu`` row.
GLU_CASES: Sequence[Tuple[str, str, Tuple[Optional[float], ...]]] = (
    ("swiglu", "silu", ()),
    ("glu_gelu", "gelu_tanh", ()),
    ("glu_gelu_erf", "gelu", ()),
    ("glu_swiglu_oai", "swiglu_oai", (1.702, 1.0, 7.0)),
    ("glu_swiglu_step_l7", "swiglu_step", (7.0,)),
    ("glu_swiglu_step_l3", "swiglu_step", (3.0,)),
    ("glu_situ_default", "situ", (4.0, 25.0, None)),
    ("glu_situ_clamp", "situ", (2.0, None, 5.0)),
    ("glu_powlu_default", "powlu", (2.5, None)),
    ("glu_powlu_limit", "powlu", (1.5, 4.0)),
)

#: The ungated GEMM1 chains over ``spec.UNGATED_ACTS`` (no parameters): ``(case name, act)``. ``relu2`` is generated on
#: the ``x`` leaf through ``relu2T`` (the class's own definition), the others on the ``u`` accumulator of ``gemm1``.
ACT_CASES: Sequence[Tuple[str, str]] = (
    ("act_identity", "identity"),
    ("act_relu", "relu"),
    ("act_silu", "silu"),
    ("act_gelu", "gelu"),
)

#: The Lean constructor of ``MoE.GAct`` per gated activation (the parameterised ones take their scalars).
_GACT = {
    "silu": ".silu",
    "gelu_tanh": ".geluTanh",
    "gelu": ".gelu",
    "swiglu_oai": ".swigluOai",
    "swiglu_step": ".swigluStep",
    "situ": ".situ",
    "powlu": ".powlu",
}
#: The Lean constructor of ``MoE.UAct`` per ungated activation.
_UACT = {
    "relu2": ".relu2",
    "identity": ".identity",
    "relu": ".relu",
    "silu": ".silu",
    "gelu": ".gelu",
}


def lean_str(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def lean_const(v: float) -> str:
    """The Lean string of a binary32 constant: Python's ``repr`` of the value ``mterm.sexpr`` prints (``(const <repr>)``),
    so a Lean term built from it prints identically."""
    return lean_str(repr(float(v)))


def lean_opt_const(v: Optional[float]) -> str:
    """An optional scalar: ``none`` for an absent one, ``(some "<repr>")`` otherwise."""
    return "none" if v is None else f"(some {lean_const(v)})"


def lean_gact(act: str, params: Sequence[Optional[float]]) -> str:
    """The ``MoE.GAct`` value of a gated chain: the constructor applied to its binary32 scalars
    (``spec.params_key``: what the Python term's constants are), parenthesised when it takes any."""
    key = M.params_key(params)
    ctor = _GACT[act]
    if act in ("silu", "gelu_tanh", "gelu"):
        assert not key, (act, key)
        return ctor
    if act == "swiglu_oai":
        alpha, beta, limit = key
        assert None not in (alpha, beta, limit), key
        return f"({ctor} {lean_const(alpha)} {lean_const(beta)} {lean_const(limit)})"
    if act == "swiglu_step":
        (limit,) = key
        assert limit is not None, key
        return f"({ctor} {lean_const(limit)})"
    if act == "situ":
        gs, ls, cl = key
        assert gs is not None, key
        return f"({ctor} {lean_const(gs)} {lean_opt_const(ls)} {lean_opt_const(cl)})"
    assert act == "powlu", act
    m, limit = key
    assert m is not None, key
    return f"({ctor} {lean_const(m)} {lean_opt_const(limit)})"


def lean_method(meth: M.RoutingMethod) -> str:
    """The Lean ``MoE.Method`` record: the weight-relevant flags (a selection bias, a selection on the logits and the
    group-limited stage never enter a weight term; ``raw`` marks the logits themselves as the weights)."""

    def b(v: bool) -> str:
        return "true" if v else "false"

    return (
        f"{{ softmax := {b(meth.weight_kind == 'softmax')}, renorm := {b(meth.renormalize)}, "
        f"scaled := {b(meth.scaled)}"
        + (", raw := true" if meth.weight_kind == "logit" else "")
        + " }"
    )


def moe_cases() -> List[Tuple[str, str, str, bool]]:
    """``(name, python_sexpr, lean_term, as_theorem)``: the MoE spec's terms and the Lean model's expressions that
    must print identically. A chain is a theorem in ``MoECases.lean`` when its S-expression is at most
    :data:`THEOREM_MAX_BYTES`; the larger chains and the routing weights (``w_<method>_E<E>_s<slot>``) are compared
    through the conformance executable only."""
    alg = term.TermAlgebra()
    x = alg.leaf("x", 0)
    g, u = alg.leaf("g", 0, 0), alg.leaf("u", 0, 0)
    ys = [alg.leaf("y", r, 0) for r in range(3)]
    ws = [alg.leaf("w", r) for r in range(3)]
    gl, ul = '(.leaf "g" [0, 0])', '(.leaf "u" [0, 0])'
    out: List[Tuple[str, str, str, bool]] = []

    def chain(name: str, t: term.Term, lean: str) -> None:
        sx = term.sexpr(t)
        out.append((name, sx, lean, len(sx) <= THEOREM_MAX_BYTES))

    # the declared building blocks on one leaf
    chain("exp", M.exp_term(alg, x), 'MoE.expT (.leaf "x" [0])')
    chain("sigmoid", M.sigmoid_term(alg, x), 'MoE.sigmoidT (.leaf "x" [0])')
    chain("tanh", M.tanh_term(alg, x), 'MoE.tanhT (.leaf "x" [0])')
    chain("erf_pos", M.erf_pos_term(alg, x), 'MoE.erfPosT (.leaf "x" [0])')
    chain("log", M.log_term(alg, x), 'MoE.logT (.leaf "x" [0])')
    chain("gelu_tanh", M.gelu_tanh_term(alg, x), 'MoE.geluTanhT (.leaf "x" [0])')
    chain("gelu", M.gelu_term(alg, x), 'MoE.geluT (.leaf "x" [0])')
    # the combine
    chain(
        "combine3",
        M.combine_elem(alg, ys, ws),
        'MoE.combineT [.leaf "y" [0, 0], .leaf "y" [1, 0], .leaf "y" [2, 0]] '
        '[.leaf "w" [0], .leaf "w" [1], .leaf "w" [2]]',
    )
    chain(
        "combine1",
        M.combine_elem(alg, ys[:1], ws[:1]),
        'MoE.combineT [.leaf "y" [0, 0]] [.leaf "w" [0]]',
    )
    # the gated GEMM1 chains, as the class builds them (`MoESpec.gemm1` on the (g, u) accumulators of slot 0, column 0)
    for name, act, params in GLU_CASES:
        s_ = M.MoESpec(64, 32, 2, 8, alg=alg, act=act, params=params)
        assert s_.gated, (name, act)
        lean = (
            f"MoE.swigluT {gl} {ul}"
            if name == "swiglu"
            else f"MoE.gluT {lean_gact(act, params)} {gl} {ul}"
        )
        chain(name, s_.gemm1(), lean)
    assert term.sexpr(M.glu_elem(alg, g, u)) == term.sexpr(
        M.MoESpec(64, 32, 2, 8, alg=alg).gemm1()
    )
    # the ungated GEMM1 chains
    chain("relu2", M.act_elem(alg, x, "relu2"), 'MoE.relu2T (.leaf "x" [0])')
    for name, act in ACT_CASES:
        s_ = M.MoESpec(64, 32, 2, 8, alg=alg, act=act, gated=False)
        chain(name, s_.gemm1(), f"MoE.actElemT {_UACT[act]} {ul}")
    # the selection terms of the group-limited stage (selection only: they never enter a weight)
    sg = M.MoESpec(64, 32, 2, 16, M.DEEPSEEK_V3_GROUPED.with_groups(4, 2), alg=alg)
    chain("biased_score", sg.biased_score(3), "MoE.biasedScoreT 3")
    chain("group_score", sg.group_score(3, 1), "MoE.groupScoreT 3 1")
    # the routing weights
    for meth, E, sel in ROUTING_CASES:
        s_ = M.MoESpec(64, 32, len(sel), E, meth, alg=alg)
        _, wt = s_.routing(selected=sel)
        for r in range(len(sel)):
            out.append(
                (
                    f"w_{meth.name}_E{E}_s{r}",
                    term.sexpr(wt[r]),
                    f"(MoE.routingWeights {lean_method(meth)} {E} {sel}).getD {r} default",
                    False,
                )
            )
    return out


def render() -> str:
    """The content of ``MoECases.lean``."""
    L = [
        "import BIKernels.MoE",
        "namespace BIK",
        "-- GENERATED by flashinfer/fused_moe/batch_invariant/lean_cases.py from the Python spec; do not edit.",
        "set_option maxRecDepth 8192",
        "",
        "/-! The bf16 MoE classes (`fused_moe/batch_invariant/spec.py` vs `MoE.lean`): the declared exp, sigmoid, tanh, "
        "erf and log, the GEMM1 chains of every gated and ungated activation (SwiGLU, GeGLU-tanh, GeGLU, SwiGLU with "
        "parameters, SwiGLUStep, SiTU, PowLU; ReLU^2, identity, ReLU, SiLU, GELU), the slot-order combine and the "
        "routing weights print identically (`mterm.sexpr` vs `MTerm.sexpr`). The chains up to "
        f"{THEOREM_MAX_BYTES} bytes are kernel-`decide` theorems; every case is a `moeconformance` line. -/",
    ]
    mcs = moe_cases()
    for name, sx, lt, as_thm in mcs:
        if as_thm:
            L.append(
                f"theorem moe_{name} : ({lt}).sexpr = {lean_str(sx)} := by decide +kernel"
            )
    L.append(
        "def moeCases : List (String × MoE.MTerm) := ["
        + ", ".join(f'("{name}", {lt})' for name, _, lt, _ in mcs)
        + "]"
    )
    L.append("end BIK")
    return "\n".join(L) + "\n"


def summary() -> str:
    mcs = moe_cases()
    chains = [c for c in mcs if not c[0].startswith("w_")]
    return (
        f"moe lean_cases coverage: cases={len(mcs)} chains={len(chains)} "
        f"chain theorems={sum(1 for c in chains if c[3])} "
        f"conformance-only chains={sum(1 for c in chains if not c[3])} "
        f"routing weights={len(mcs) - len(chains)} methods={len({m.name for m, _, _ in ROUTING_CASES})}"
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
