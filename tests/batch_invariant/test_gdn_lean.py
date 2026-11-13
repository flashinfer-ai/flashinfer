# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The Lean coupling of the GDN class ``gdn.bf16.c64`` (``flashinfer/batch_invariant/lean/``: ``Gdn.lean`` and the
generated ``GdnCases.lean``).

Device-free (every CI job): the committed ``GdnCases.lean`` is byte for byte what the Python specification emits, and
the specification's arithmetic agrees with itself on the invariants the Lean file states. With ``lake`` installed:
``lake build`` elaborates the model and the cases (kernel ``decide`` only: the GDN cases add nothing to the axiom policy
``lake exe axioms`` enforces), and ``lake exe gdnconformance`` prints the model's arithmetic, compared here with the
Python specification.
"""

import os
import shutil
import subprocess

import pytest

from flashinfer.batch_invariant.gdn import lean_cases, spec

LAKE = shutil.which("lake") or os.path.expanduser("~/.elan/bin/lake")
needs_lake = pytest.mark.skipif(
    not os.path.exists(LAKE), reason="Lean toolchain (elan / lake) not installed"
)


def test_generated_cases_are_current():
    with open(lean_cases.CASES_PATH) as f:
        committed = f.read()
    assert committed == lean_cases.render(), (
        "regenerate with: python -m flashinfer.batch_invariant.gdn.lean_cases"
    )
    assert "sorry" not in committed and "native_decide" not in committed
    assert len(lean_cases.cases()) == len(lean_cases.POSITIONS)


def test_spec_arithmetic():
    for p in range(0, 5000):
        assert spec.boundary(p) + spec.phase(p) == p
        assert spec.phase(p) < spec.CHUNK
        assert spec.commits(spec.phase(p)) == (spec.phase(p + 1) == 0)
        for n in (1, 15, 16, 64, 200):
            whole, tail = spec.extend_split(spec.phase(p), n)
            assert spec.boundary(p) + whole == spec.boundary(p + n)
            assert tail == spec.phase(p + n)
    for c in range(spec.CHUNK):
        assert spec.next_count(c) == (0 if spec.commits(c) else c + 1)
    assert spec.uses_step_kernel(spec.STEP_MAX_TOKENS - 1)
    assert not spec.uses_step_kernel(spec.STEP_MAX_TOKENS)


def test_fingerprint_is_the_class_receipt():
    import flashinfer.batch_invariant as bi

    fp = bi.fingerprint("chunk_gated_delta_rule")
    assert fp.startswith("gdn-bf16-c64-") and fp == spec.spec_fingerprint()
    assert bi.class_name("chunk_gated_delta_rule") == spec.CLASS
    with pytest.raises(TypeError):
        bi.fingerprint("chunk_gated_delta_rule", N=16)


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
    r = _lake("build", "gdnconformance", timeout=1800)
    assert r.returncode == 0, (r.stdout + r.stderr)[-4000:]
    r = _lake("exe", "gdnconformance", timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    lean = [line for line in r.stdout.splitlines() if line.startswith("gdn")]
    assert lean == lean_cases.conformance_lines()
