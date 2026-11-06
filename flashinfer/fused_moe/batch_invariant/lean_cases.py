# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Generate ``lean/BIKernels/MoECases.lean`` from the Python side: the bf16 MoE classes' conformance cases.

The file couples the Lean model (``lean/BIKernels/MoE.lean``) to :mod:`.spec` node for node: per case a term the
Python spec builds (:mod:`.mterm` S-expression) and the Lean expression that must print identically
(``MTerm.sexpr``). The chains (the declared exp and sigmoid, the GEMM1 epilogues of the SwiGLU, GeGLU and ReLU^2
classes, the slot-order combine) become theorems decided by ``decide +kernel`` (the kernel evaluates the term and the
string comparison; the file adds nothing to the axiom policy). The routing weights of the five named methods (tens to
hundreds of KB: the exp chain repeats per expert) are compared through the ``moeconformance`` executable only
(``tests/batch_invariant/test_moe_lean.py``). A change to a chain, a routing method or a constant that is not
reflected in the Lean model makes ``lake build`` or the conformance comparison fail; a regenerated file that differs
from the committed one fails the device-free test, which needs no Lean toolchain.

    python -m flashinfer.fused_moe.batch_invariant.lean_cases     # rewrite lean/BIKernels/MoECases.lean
"""

from __future__ import annotations

import os
from typing import List, Sequence, Tuple

from . import mterm as term
from . import spec as M

LEAN_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "batch_invariant",
    "lean",
)
CASES_PATH = os.path.join(LEAN_DIR, "BIKernels", "MoECases.lean")

#: The routing-weight cases: ``(method, E, selected experts in slot order)``, one row per named method (two for the
#: renormalised softmax: a power-of-two and a padded expert count). The weights of every slot are compared.
ROUTING_CASES: Sequence[Tuple[M.RoutingMethod, int, List[int]]] = (
    (M.SOFTMAX_RENORM, 8, [3, 0]),
    (M.SOFTMAX, 13, [12, 0, 5]),
    (M.SOFTMAX_RENORM, 5, [4, 1, 2]),
    (M.SIGMOID_RENORM, 8, [2, 6]),
    (M.SIGMOID_BIAS_RENORM_SCALED, 8, [1, 7]),
    (M.SIGMOID, 64, [63]),
)


def lean_str(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def lean_method(meth: M.RoutingMethod) -> str:
    """The Lean ``MoE.Method`` record: the weight-relevant flags (a selection bias never enters a weight term)."""

    def b(v: bool) -> str:
        return "true" if v else "false"

    return (
        f"{{ softmax := {b(meth.score == 'softmax')}, renorm := {b(meth.renormalize)}, "
        f"scaled := {b(meth.scaled)} }}"
    )


def moe_cases() -> List[Tuple[str, str, str, bool]]:
    """``(name, python_sexpr, lean_term, as_theorem)``: the MoE spec's terms and the Lean model's expressions that
    must print identically. The chains are theorems in ``MoECases.lean``; the routing weights (``w_<method>_E<E>_s<slot>``)
    are compared through the conformance executable only."""
    alg = term.TermAlgebra()
    x = alg.leaf("x", 0)
    g, u = alg.leaf("g", 0, 0), alg.leaf("u", 0, 0)
    ys = [alg.leaf("y", r, 0) for r in range(3)]
    ws = [alg.leaf("w", r) for r in range(3)]
    out: List[Tuple[str, str, str, bool]] = []
    out.append(
        ("exp", term.sexpr(M.exp_term(alg, x)), 'MoE.expT (.leaf "x" [0])', True)
    )
    out.append(
        (
            "sigmoid",
            term.sexpr(M.sigmoid_term(alg, x)),
            'MoE.sigmoidT (.leaf "x" [0])',
            True,
        )
    )
    out.append(
        (
            "swiglu",
            term.sexpr(M.swiglu_elem(alg, g, u)),
            'MoE.swigluT (.leaf "g" [0, 0]) (.leaf "u" [0, 0])',
            True,
        )
    )
    out.append(
        (
            "combine3",
            term.sexpr(M.combine_elem(alg, ys, ws)),
            'MoE.combineT [.leaf "y" [0, 0], .leaf "y" [1, 0], .leaf "y" [2, 0]] '
            '[.leaf "w" [0], .leaf "w" [1], .leaf "w" [2]]',
            True,
        )
    )
    out.append(
        (
            "combine1",
            term.sexpr(M.combine_elem(alg, ys[:1], ws[:1])),
            'MoE.combineT [.leaf "y" [0, 0]] [.leaf "w" [0]]',
            True,
        )
    )
    out.append(
        (
            "gelu_tanh",
            term.sexpr(M.gelu_tanh_term(alg, x)),
            'MoE.geluTanhT (.leaf "x" [0])',
            True,
        )
    )
    out.append(
        (
            "glu_gelu",
            term.sexpr(M.glu_elem(alg, g, u, "gelu_tanh")),
            'MoE.gluT true (.leaf "g" [0, 0]) (.leaf "u" [0, 0])',
            True,
        )
    )
    out.append(
        (
            "relu2",
            term.sexpr(M.act_elem(alg, x, "relu2")),
            'MoE.relu2T (.leaf "x" [0])',
            True,
        )
    )
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
        "/-! The bf16 MoE classes (`fused_moe/batch_invariant/spec.py` vs `MoE.lean`): the declared exp and sigmoid, the "
        "GEMM1 chains (SwiGLU, GeGLU, ReLU^2), the slot-order combine and the routing weights print identically "
        "(`mterm.sexpr` vs `MTerm.sexpr`). -/",
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
    return (
        f"moe lean_cases coverage: cases={len(mcs)} chain theorems={sum(1 for c in mcs if c[3])} "
        f"routing weights={sum(1 for c in mcs if not c[3])} methods={len({m.name for m, _, _ in ROUTING_CASES})}"
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
