# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The Lean coupling of the class ``gemm.bf16`` (``flashinfer/batch_invariant/lean/``).

Device-free (every CI job): the committed ``Cases.lean`` is byte for byte what the Python spec and engine emit, so a
change to the dataflow, the lowering, a knob constraint or the chunk rule that was not propagated to the Lean cases
fails here without a toolchain. With ``lake`` installed: ``lake build`` elaborates the class, its configuration space
and the generated cases (kernel ``decide``; ``native_decide`` for the named exhaustive space checks),
``lake exe conformance`` prints the Lean model's terms, compared here with the Python S-expressions, and
``lake exe axioms`` enforces the axiom policy (no ``sorry``; standard axioms plus ``Lean.ofReduceBool`` for exactly
the named theorems).
"""

import os
import shutil
import subprocess

import pytest

from flashinfer.batch_invariant import lean_cases
from flashinfer.gemm.batch_invariant.spec import MMA_K

LAKE = shutil.which("lake") or os.path.expanduser("~/.elan/bin/lake")
needs_lake = pytest.mark.skipif(
    not os.path.exists(LAKE), reason="Lean toolchain (elan / lake) not installed"
)


def test_generated_cases_are_current():
    with open(lean_cases.CASES_PATH) as f:
        committed = f.read()
    assert committed == lean_cases.render(), (
        "regenerate with: python -m flashinfer.batch_invariant.lean_cases"
    )
    assert "sorry" not in committed
    assert committed.count("native_decide") == len(lean_cases.DECIDED_THEOREMS)
    for name in lean_cases.DECIDED_THEOREMS:
        assert f"theorem {name} " in committed
    with open(os.path.join(lean_cases.LEAN_DIR, "Axioms.lean")) as f:
        axioms = f.read()
    for name in lean_cases.DECIDED_THEOREMS:
        assert f"``BIK.{name}" in axioms, f"{name} is not pinned in Axioms.lean"


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
def test_lean_model_matches_python_engine_and_spec():
    r = _lake("build", timeout=3600)
    assert r.returncode == 0, "Lean proofs failed:\n" + (r.stdout + r.stderr)[-4000:]
    r = _lake("build", "conformance", "axioms", timeout=1800)
    assert r.returncode == 0, (r.stdout + r.stderr)[-4000:]
    r = _lake("exe", "conformance", timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    lean = {}
    for line in r.stdout.splitlines():
        tag = line.split(" ", 1)[0]
        if tag == "gemm":
            _, steps, C, sx = line.split(" ", 3)
            lean[("gemm", int(steps), int(C))] = sx
        elif tag == "gemmloop":
            _, nblk, sx = line.split(" ", 2)
            lean[("gemmloop", int(nblk))] = sx
        elif tag == "gemmout":
            _, steps, fmt, sx = line.split(" ", 3)
            lean[("gemmout", int(steps), fmt)] = sx
        elif tag == "chunks":
            _, N, K, C = line.split(" ")
            lean[("chunks", int(N), int(K))] = int(C)
        else:
            raise AssertionError(f"unexpected conformance line: {line[:80]}")
    checked = 0
    for K, C, sx in lean_cases.spec_cases():
        assert lean[("gemm", K // MMA_K, C)] == sx, (K, C)
        checked += 1
    for nblk, sx in lean_cases.loop_cases()[1]:
        assert lean[("gemmloop", nblk)] == sx, nblk
        checked += 1
    for K, fmt, sx in lean_cases.out_cases():
        assert lean[("gemmout", K // MMA_K, fmt)] == sx, (K, fmt)
        checked += 1
    for N, K, C in lean_cases.chunk_table():
        assert lean[("chunks", N, K)] == C, (N, K)
        checked += 1
    assert checked == len(lean) >= 20


@needs_lake
def test_axioms_policy():
    r = _lake("exe", "axioms", timeout=600)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]
    assert "0 with sorry, 0 violations" in r.stdout, r.stdout
    assert f"{len(lean_cases.DECIDED_THEOREMS)} named table theorems" in r.stdout, (
        r.stdout
    )
