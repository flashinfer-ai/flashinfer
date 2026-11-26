# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The Lean coupling of the mixed MXFP4 x MXFP8 MoE classes (``flashinfer/batch_invariant/lean/``: ``MoEMxfp4w8a.lean`` and the generated
``MoEMxfp4w8aCases.lean``).

Device-free (every CI job): the committed ``MoEMxfp4w8aCases.lean`` is byte for byte what the Python spec emits (the chain
terms and the grouped record space), so a change to the class's chains or its record space that was not propagated to the
Lean cases fails here without a toolchain. With ``lake`` installed: ``lake build`` elaborates the model and the generated
cases (kernel ``decide`` throughout: the file adds nothing to the axiom policy ``lake exe axioms`` enforces), ``lake exe
moemxfp4w8aconformance`` prints the Lean model's terms, compared here with the Python S-expressions, and ``lake exe axioms``
enforces the axiom policy of the whole library.
"""

import os
import shutil
import subprocess

import pytest

from flashinfer.fused_moe.batch_invariant import lean_cases_mxfp4w8a as lean_cases

LAKE = shutil.which("lake") or os.path.expanduser("~/.elan/bin/lake")
needs_lake = pytest.mark.skipif(
    not os.path.exists(LAKE), reason="Lean toolchain (elan / lake) not installed"
)


def test_generated_cases_are_current():
    with open(lean_cases.CASES_PATH) as f:
        committed = f.read()
    assert committed == lean_cases.render(), (
        "regenerate with: python -m flashinfer.fused_moe.batch_invariant.lean_cases_mxfp4w8a"
    )
    names = [name for name, _, _ in lean_cases.cases()]
    assert len(names) == len(set(names)) and {
        "gemm2",
        "gemm1_swiglu",
        "gemm1_gelu_tanh",
        "gemm1_relu2",
        "gemm1_identity",
    } <= set(names)
    for name in names:
        assert f"theorem moe_mxfp4w8a_{name} :" in committed
    assert len(lean_cases.space()) == 72 and len(lean_cases.swap_space()) == 3
    assert (
        "theorem moe_mxfp4w8a_space_size : moeMxfp4w8aSpace.length = 72 := by decide"
        in committed
    )
    assert (
        "theorem moe_mxfp4w8a_swap_space_size : moeMxfp4w8aSwapSpace.length = 3 := by decide"
        in committed
    )
    assert "moe_mxfp4w8a_both_spaces_invariant_k64" in committed
    assert "sorry" not in committed and "native_decide" not in committed


def _lake(*args, timeout):
    from flashinfer.fused_moe.batch_invariant.lean_cases import LEAN_DIR

    env = dict(
        os.environ,
        ELAN_NO_OVERRIDE_NOTICE="1",
        PATH=os.path.dirname(LAKE) + os.pathsep + os.environ["PATH"],
    )
    return subprocess.run(
        [LAKE, *args],
        cwd=LEAN_DIR,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


@needs_lake
def test_lean_model_matches_python_spec():
    r = _lake("build", timeout=3600)
    assert r.returncode == 0, "Lean proofs failed:\n" + (r.stdout + r.stderr)[-4000:]
    r = _lake("build", "moemxfp4w8aconformance", timeout=1800)
    assert r.returncode == 0, (r.stdout + r.stderr)[-4000:]
    r = _lake("exe", "moemxfp4w8aconformance", timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    lean = {}
    size = swap_size = None
    for line in r.stdout.splitlines():
        tag = line.split(" ", 1)[0]
        if tag == "moemxfp4w8a":
            _, name, sx = line.split(" ", 2)
            assert name not in lean, f"duplicate conformance case {name}"
            lean[name] = sx
        elif tag == "moemxfp4w8aspace":
            size = int(line.split(" ", 1)[1])
        elif tag == "moemxfp4w8aswapspace":
            swap_size = int(line.split(" ", 1)[1])
        else:
            raise AssertionError(f"unexpected conformance line: {line[:80]}")
    for name, sx, _ in lean_cases.cases():
        assert lean[name] == sx, (
            f"mixed MXFP4 x MXFP8 MoE model differs from the Python spec: {name}"
        )
    assert (
        len(lean) == len(lean_cases.cases()) and size == len(lean_cases.space()) == 72
    )
    assert swap_size == len(lean_cases.swap_space()) == 3


@needs_lake
def test_axioms_policy():
    r = _lake("exe", "axioms", timeout=600)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]
    assert "0 with sorry, 0 violations" in r.stdout, r.stdout
