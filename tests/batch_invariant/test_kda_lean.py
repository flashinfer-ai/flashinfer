# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The Lean coupling of the KDA class ``kda.bf16.c16`` (``flashinfer/batch_invariant/lean/``: ``Kda.lean`` and the
generated ``KdaCases.lean``).

Device-free (every CI job): the committed ``KdaCases.lean`` is byte for byte what the Python specification emits, and
the specification's arithmetic agrees with itself on the invariants the Lean file states. With ``lake`` installed:
``lake build`` elaborates the model and the cases (kernel ``decide`` only: the KDA cases add nothing to the axiom policy
``lake exe axioms`` enforces), and ``lake exe kdaconformance`` prints the model's arithmetic, compared here with the
Python specification.
"""

import os
import shutil
import subprocess

import pytest

from flashinfer.batch_invariant.kda import lean_cases, spec

LAKE = shutil.which("lake") or os.path.expanduser("~/.elan/bin/lake")
needs_lake = pytest.mark.skipif(
    not os.path.exists(LAKE), reason="Lean toolchain (elan / lake) not installed"
)


def test_append_split_arithmetic():
    """The split that drives every multi-token call: head to the boundary, whole chunks, a tail shorter than a chunk."""
    for count in range(spec.CHUNK):
        for n in range(0, 70):
            head, whole, tail = spec.append_split(count, n)
            assert head + whole + tail == n and min(head, whole, tail) >= 0
            assert whole % spec.CHUNK == 0 and tail < spec.CHUNK and head < spec.CHUNK
            if whole + tail:
                assert (count + head) % spec.CHUNK == 0
            else:
                assert count + head <= spec.CHUNK


def test_generated_cases_are_current():
    with open(lean_cases.CASES_PATH) as f:
        committed = f.read()
    assert committed == lean_cases.render(), (
        "regenerate with: python -m flashinfer.batch_invariant.kda.lean_cases"
    )
    assert "sorry" not in committed and "native_decide" not in committed
    assert len(lean_cases.cases()) == len(lean_cases.POSITIONS)


def test_spec_arithmetic():
    for p in range(0, 5000):
        assert spec.boundary(p) + spec.history_len(p) == p
        assert spec.history_len(p) < spec.CHUNK
        assert spec.commits(spec.history_len(p)) == (spec.history_len(p + 1) == 0)
    for c in range(spec.CHUNK):
        assert spec.next_count(c) == (0 if spec.commits(c) else c + 1)
    for L in (0, 1, 15, 16, 17, 4096, 4097):
        b, tail = spec.prefill_split(L)
        assert b % spec.CHUNK == 0 and b + tail == L and tail < spec.CHUNK


def test_fingerprint_is_the_class_receipt():
    import flashinfer.batch_invariant as bi

    fp = bi.fingerprint("recurrent_kda")
    assert fp.startswith("kda-bf16-c16-engine-") and fp == spec.spec_fingerprint()
    assert bi.class_name("recurrent_kda") == spec.CLASS
    with pytest.raises(TypeError):
        bi.fingerprint("recurrent_kda", N=16)


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
    r = _lake("build", "kdaconformance", timeout=1800)
    assert r.returncode == 0, (r.stdout + r.stderr)[-4000:]
    r = _lake("exe", "kdaconformance", timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    lean = [line for line in r.stdout.splitlines() if line.startswith("kda")]
    assert lean == lean_cases.conformance_lines()
