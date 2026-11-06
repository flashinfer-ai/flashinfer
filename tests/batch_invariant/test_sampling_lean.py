# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The Lean coupling of the sampling class ``lse.row`` (``flashinfer/batch_invariant/lean/``: ``Sampling.lean`` and the
generated ``SamplingCases.lean``; ``SamplingTP.lean`` and the ``tp*`` cases of the tensor-parallel protocol).

Device-free (every CI job): the committed ``SamplingCases.lean`` is byte for byte what the Python spec and engine emit,
so a change to the dataflow, the engine or a placement that was not propagated to the Lean cases fails here without a
toolchain. With ``lake`` installed: ``lake build`` elaborates the engine theorems, the class and the generated cases
(kernel ``decide`` throughout: the sampling cases add nothing to the axiom policy ``lake exe axioms`` enforces), and
``lake exe samplingconformance`` prints the Lean model's terms, compared here with the Python S-expressions.
"""

import os
import shutil
import subprocess

import pytest

from flashinfer.batch_invariant.sampling import lean_cases

LAKE = shutil.which("lake") or os.path.expanduser("~/.elan/bin/lake")
needs_lake = pytest.mark.skipif(
    not os.path.exists(LAKE), reason="Lean toolchain (elan / lake) not installed"
)


def test_generated_cases_are_current():
    with open(lean_cases.CASES_PATH) as f:
        committed = f.read()
    assert committed == lean_cases.render(), (
        "regenerate with: python -m flashinfer.batch_invariant.sampling.lean_cases"
    )
    cases = lean_cases.lse_cases()
    assert len(cases) >= 10
    assert {c["C"] for c in cases} >= {1, 2, 4}  # single-CTA and cluster placements
    assert len(lean_cases.lse_out_cases()) == 2 * len(lean_cases.SUM_NS)
    assert len(lean_cases.tp_cases()) == len(lean_cases.TP_CASES) == 3
    assert len(lean_cases.tp_layout_cases()) == len(lean_cases.TP_PROD) >= 8
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
def test_lean_model_matches_python_engine_and_spec():
    r = _lake("build", timeout=3600)
    assert r.returncode == 0, "Lean proofs failed:\n" + (r.stdout + r.stderr)[-4000:]
    r = _lake("build", "samplingconformance", timeout=1800)
    assert r.returncode == 0, (r.stdout + r.stderr)[-4000:]
    r = _lake("exe", "samplingconformance", timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    lean = {}
    for line in r.stdout.splitlines():
        tag = line.split(" ", 1)[0]
        if tag == "lse":
            _, N, kb, name, kind, sx = line.split(" ", 5)
            lean[("lse", int(N), name, kind)] = sx
        elif tag == "lseout":
            _, N, temp, which, sx = line.split(" ", 4)
            lean[("lseout", int(N), temp == "true", which)] = sx
        elif tag == "lsechain":
            lean[("lsechain",)] = line.split(" ", 1)[1]
        elif tag == "tp":
            _, N, i, kind, sx = line.split(" ", 4)
            lean[("tp", int(N), int(i), kind)] = sx
        elif tag == "tppiece":
            _, N, i, lane, L, idx, sx = line.split(" ", 6)
            lean[("tppiece", int(N), int(i), int(lane), int(L), int(idx))] = sx
        elif tag == "tplayout":
            _, name, table = line.split(" ", 2)
            lean[("tplayout", name)] = table
        else:
            raise AssertionError(f"unexpected conformance line: {line[:80]}")
    checked = 0
    for c in lean_cases.lse_cases():
        kinds = lean_cases.LSE_KINDS if c["with_sum"] else lean_cases.LSE_KINDS[:2]
        for kind in kinds:
            assert lean[("lse", c["N"], c["name"], kind)] == c[kind], (
                c["N"],
                c["name"],
                kind,
            )
            checked += 1
    for c in lean_cases.lse_out_cases():
        for which in lean_cases.LSE_OUT_WHICH:
            assert lean[("lseout", c["N"], c["temp"], which)] == c[which], (
                c["N"],
                c["temp"],
                which,
            )
            checked += 1
    # the entropy member's chain over symbolic statistics (its full strings are compared above per vocabulary)
    assert lean[("lsechain",)] == lean_cases.entropy_chain_sexpr()
    checked += 1
    # the tensor-parallel protocol: the Lean merge of the ranks' pieces prints the spec's maximum and sum, the pieces are
    # the Python partials' sub-trees, the production layout tables are the Python ShardLayout tables
    for i, c in enumerate(lean_cases.tp_cases()):
        assert lean[("tp", c["N"], i, "max")] == c["m"], (
            "tp merged maximum",
            c["N"],
            i,
        )
        assert lean[("tp", c["N"], i, "sum")] == c["s"], ("tp merged sum", c["N"], i)
        checked += 2
        for lane, L, idx, sx in c["pieces"]:
            assert lean[("tppiece", c["N"], i, lane, L, idx)] == sx, (
                "tp piece",
                c["N"],
                lane,
                L,
                idx,
            )
            checked += 1
    for name, _N, _bounds, _kb, table in lean_cases.tp_layout_cases():
        assert lean[("tplayout", name)] == table, ("tp layout table", name)
        checked += 1
    assert checked == len(lean) >= 60
