# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The Lean coupling of the quantization classes (``flashinfer/batch_invariant/lean/``: ``Quant.lean``, ``MXQuant.lean``
and the generated ``QuantCases.lean``).

Device-free (every CI job): the committed ``QuantCases.lean`` is byte for byte what the Python specs and engine emit, so a
change to a dataflow, the engine or a placement that was not propagated to the Lean cases fails here without a
toolchain. With ``lake`` installed: ``lake build`` elaborates the engine theorems, the classes and the generated cases
(kernel ``decide`` throughout: the quantization cases add nothing to the axiom policy ``lake exe axioms`` enforces),
and ``lake exe quantconformance`` prints the Lean model's terms, compared here with the Python S-expressions.
"""

import os
import shutil
import subprocess

import pytest

from flashinfer.batch_invariant.quantization import lean_cases

LAKE = shutil.which("lake") or os.path.expanduser("~/.elan/bin/lake")
needs_lake = pytest.mark.skipif(
    not os.path.exists(LAKE), reason="Lean toolchain (elan / lake) not installed"
)


def test_generated_cases_are_current():
    with open(lean_cases.CASES_PATH) as f:
        committed = f.read()
    assert committed == lean_cases.render(), (
        "regenerate with: python -m flashinfer.batch_invariant.quantization.lean_cases"
    )
    assert len(lean_cases.quant_cases()) >= 6
    assert len(lean_cases.quant_out_cases()) == 6
    assert len(lean_cases.quant_block_cases()) >= 4
    assert len(lean_cases.mxquant_cases()) == 3
    assert len(lean_cases.mxquant_out_cases()) == 12
    assert len(lean_cases.mxquant_block_cases()) >= 8
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
def test_lean_model_matches_python_engine_and_specs():
    r = _lake("build", timeout=3600)
    assert r.returncode == 0, "Lean proofs failed:\n" + (r.stdout + r.stderr)[-4000:]
    r = _lake("build", "quantconformance", timeout=1800)
    assert r.returncode == 0, (r.stdout + r.stderr)[-4000:]
    r = _lake("exe", "quantconformance", timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    lean = {}
    for line in r.stdout.splitlines():
        tag = line.split(" ", 1)[0]
        if tag == "quant":
            _, K, kb, name, kind, sx = line.split(" ", 5)
            lean[("quant", int(K), name, kind)] = sx
        elif tag == "quantout":
            _, K, vname, which, sx = line.split(" ", 4)
            lean[("quantout", int(K), vname, which)] = sx
        elif tag == "quantblock":
            _, b, name, sx = line.split(" ", 3)
            lean[("quantblock", int(b), name)] = sx
        elif tag == "mxq":
            _, name, kind, sx = line.split(" ", 3)
            lean[("mxq", name, kind)] = sx
        elif tag == "mxqout":
            _, fname, rname, which, sx = line.split(" ", 4)
            lean[("mxqout", fname, rname, which)] = sx
        elif tag == "mxqblock":
            _, b, name, sx = line.split(" ", 3)
            lean[("mxqblock", int(b), name)] = sx
        else:
            raise AssertionError(f"unexpected conformance line: {line[:80]}")
    checked = 0
    for c in lean_cases.quant_cases():
        assert lean[("quant", c["K"], c["name"], "spec")] == c["spec"], (
            c["K"],
            c["name"],
        )
        assert lean[("quant", c["K"], c["name"], "kernel")] == c["kernel"], (
            c["K"],
            c["name"],
        )
        checked += 2
    for c in lean_cases.quant_out_cases():
        for which in lean_cases.QUANT_OUT_WHICH:
            assert lean[("quantout", c["K"], c["variant"], which)] == c[which], (
                c["K"],
                c["variant"],
                which,
            )
            checked += 1
    for b, name, sx in lean_cases.quant_block_cases():
        assert lean[("quantblock", b, name)] == sx, (b, name)
        checked += 1
    for c in lean_cases.mxquant_cases():
        assert lean[("mxq", c["name"], "spec")] == c["spec"]
        assert lean[("mxq", c["name"], "kernel")] == c["kernel"], c["name"]
        checked += 2
    for c in lean_cases.mxquant_out_cases():
        for which in lean_cases.MXQ_OUT_WHICH:
            assert lean[("mxqout", c["fmt"], c["rule"], which)] == c[which], (
                c["fmt"],
                c["rule"],
                which,
            )
            checked += 1
    for b, name, sx in lean_cases.mxquant_block_cases():
        assert lean[("mxqblock", b, name)] == sx, (b, name)
        checked += 1
    assert checked >= 100
