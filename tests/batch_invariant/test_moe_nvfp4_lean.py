# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The Lean coupling of the NVFP4 MoE classes (``flashinfer/batch_invariant/lean/``: ``MoENvfp4.lean`` and the generated
``MoENvfp4Cases.lean``).

Device-free (the CI jobs that collect this module): the committed ``MoENvfp4Cases.lean`` is byte for byte what the Python spec emits, so a change
to the W4A4 chains (the row scale, a gated or ungated activation on the scaled accumulators, GEMM2's chain) that was not
propagated to the Lean cases fails here without a toolchain. With ``lake`` installed: ``lake build`` elaborates the model and
the generated cases (kernel ``decide`` throughout: the cases add nothing to the axiom policy), ``lake exe
moenvfp4conformance`` prints the Lean model's terms, compared here with the Python S-expressions of every case, and ``lake
exe axioms`` enforces the axiom policy of the whole library.
"""

import os
import shutil
import subprocess

import pytest

from flashinfer.fused_moe.batch_invariant import lean_cases, lean_cases_nvfp4

LAKE = shutil.which("lake") or os.path.expanduser("~/.elan/bin/lake")
needs_lake = pytest.mark.skipif(
    not os.path.exists(LAKE), reason="Lean toolchain (elan / lake) not installed"
)


def test_generated_cases_are_current():
    with open(lean_cases_nvfp4.CASES_PATH) as f:
        committed = f.read()
    assert committed == lean_cases_nvfp4.render(), (
        "regenerate with: python -m flashinfer.fused_moe.batch_invariant.lean_cases_nvfp4"
    )
    cases = lean_cases_nvfp4.moe_nvfp4_cases()
    names = [name for name, _, _, _ in cases]
    assert len(names) == len(set(names))
    # one gated case per activation and parameter set of the bf16 generator, one ungated case per ungated activation, the
    # row scale and GEMM2
    assert {f"gemm1_{n}" for n, _, _ in lean_cases.GLU_CASES} <= set(names)
    assert {f"gemm1_{n}" for n, _ in lean_cases.ACT_CASES} | {
        "gemm1_act_relu2",
        "row_scale",
        "gemm2",
    } <= set(names)
    theorems = {name for name, _, _, as_thm in cases if as_thm}
    assert {
        "row_scale",
        "gemm2",
        "gemm1_swiglu",
        "gemm1_glu_gelu",
        "gemm1_act_relu2",
        "gemm1_act_identity",
    } <= theorems
    for name in theorems:
        assert f"theorem moe_nvfp4_{name} :" in committed
    # the SiTU chains' SFU form is small enough for kernel theorems; the PowLU chains (the declared log / exp) are too
    # large for one and are conformance lines (the executable), never dropped
    assert {"gemm1_glu_situ_default", "gemm1_glu_situ_clamp"} <= theorems
    large = {"gemm1_glu_powlu_default", "gemm1_glu_powlu_limit"}
    assert large <= set(names) and not (large & theorems)
    for _name, sx, _, as_thm in cases:
        assert as_thm == (len(sx) <= lean_cases.THEOREM_MAX_BYTES)
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
    r = _lake("build", "moenvfp4conformance", timeout=1800)
    assert r.returncode == 0, (r.stdout + r.stderr)[-4000:]
    r = _lake("exe", "moenvfp4conformance", timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    lean = {}
    for line in r.stdout.splitlines():
        tag, name, sx = line.split(" ", 2)
        assert tag == "moenvfp4", line[:80]
        assert name not in lean, f"duplicate conformance case {name}"
        lean[name] = sx
    checked = 0
    for name, sx, _, _ in lean_cases_nvfp4.moe_nvfp4_cases():
        assert lean[name] == sx, f"NVFP4 MoE model differs from the Python spec: {name}"
        checked += 1
    assert checked == len(lean) == len(lean_cases_nvfp4.moe_nvfp4_cases()) >= 17


@needs_lake
def test_axioms_policy():
    r = _lake("exe", "axioms", timeout=600)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]
    assert "0 with sorry, 0 violations" in r.stdout, r.stdout
