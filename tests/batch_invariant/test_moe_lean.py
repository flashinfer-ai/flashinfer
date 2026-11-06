# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The Lean coupling of the bf16 MoE classes (``flashinfer/batch_invariant/lean/``: ``MoE.lean`` and the generated
``MoECases.lean``).

Device-free (every CI job): the committed ``MoECases.lean`` is byte for byte what the Python spec emits, so a change to
a chain, a routing method or a constant that was not propagated to the Lean cases fails here without a toolchain. With
``lake`` installed: ``lake build`` elaborates the model and the generated cases (kernel ``decide`` throughout: the MoE
cases add nothing to the axiom policy ``lake exe axioms`` enforces), ``lake exe moeconformance`` prints the Lean model's
terms, compared here with the Python S-expressions of every case (the chains and the routing weights), and
``lake exe axioms`` enforces the axiom policy of the whole library.
"""

import os
import shutil
import subprocess

import pytest

from flashinfer.fused_moe.batch_invariant import lean_cases

LAKE = shutil.which("lake") or os.path.expanduser("~/.elan/bin/lake")
needs_lake = pytest.mark.skipif(
    not os.path.exists(LAKE), reason="Lean toolchain (elan / lake) not installed"
)


def test_generated_cases_are_current():
    with open(lean_cases.CASES_PATH) as f:
        committed = f.read()
    assert committed == lean_cases.render(), (
        "regenerate with: python -m flashinfer.fused_moe.batch_invariant.lean_cases"
    )
    cases = lean_cases.moe_cases()
    names = [name for name, _, _, _ in cases]
    assert len(names) == len(set(names))
    chains = [name for name, _, _, as_thm in cases if as_thm]
    assert set(chains) == {
        "exp",
        "sigmoid",
        "swiglu",
        "combine3",
        "combine1",
        "gelu_tanh",
        "glu_gelu",
        "relu2",
    }
    for name in chains:
        assert f"theorem moe_{name} :" in committed
    routing = [name for name, _, _, as_thm in cases if not as_thm]
    assert len(routing) >= 2 * len(lean_cases.ROUTING_CASES)
    assert {m.name for m, _, _ in lean_cases.ROUTING_CASES} == {
        "softmax",
        "softmax_renorm",
        "sigmoid",
        "sigmoid_renorm",
        "sigmoid_renorm_bias_scaled",
    }
    assert "sorry" not in committed and "native_decide" not in committed


def _lake(*args, timeout):
    env = dict(
        os.environ,
        ELAN_NO_OVERRIDE_NOTICE="1",
        PATH=os.path.dirname(LAKE) + os.pathsep + os.environ["PATH"],
    )
    return subprocess.run(
        [LAKE, *args],
        cwd=lean_cases.LEAN_DIR,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


@needs_lake
def test_lean_model_matches_python_spec():
    r = _lake("build", timeout=3600)
    assert r.returncode == 0, "Lean proofs failed:\n" + (r.stdout + r.stderr)[-4000:]
    r = _lake("build", "moeconformance", timeout=1800)
    assert r.returncode == 0, (r.stdout + r.stderr)[-4000:]
    r = _lake("exe", "moeconformance", timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    lean = {}
    for line in r.stdout.splitlines():
        tag = line.split(" ", 1)[0]
        if tag == "moe":
            _, name, sx = line.split(" ", 2)
            assert name not in lean, f"duplicate conformance case {name}"
            lean[name] = sx
        else:
            raise AssertionError(f"unexpected conformance line: {line[:80]}")
    checked = 0
    for name, sx, _, _ in lean_cases.moe_cases():
        assert lean[name] == sx, f"MoE model differs from the Python spec: {name}"
        checked += 1
    assert checked == len(lean) == len(lean_cases.moe_cases()) >= 20


@needs_lake
def test_axioms_policy():
    r = _lake("exe", "axioms", timeout=600)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]
    assert "0 with sorry, 0 violations" in r.stdout, r.stdout
