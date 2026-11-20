# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The Lean coupling of the sampling class ``lse.row`` (``flashinfer/batch_invariant/lean/``: ``Sampling.lean`` and the
generated ``SamplingCases.lean``), of the draw class ``sample.icdf.v1`` (``Draw.lean``, the ``draw*`` cases) and of the
request-keyed stream (``Stream.lean``, the ``stream*`` cases) and of the support class ``support.row.v1``
(``SupportRow.lean``, the ``sup*`` cases).

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
    assert len(lean_cases.draw_cases()) == len(lean_cases.DRAW_NS) == 4
    assert all(
        len(c["prefixes"]) >= 2 for c in lean_cases.draw_cases()
    )  # N = 8: elements 0 and 7
    assert len(lean_cases.stream_cases()) == len(lean_cases.STREAM_KEYS) >= 6
    assert len(lean_cases.support_cases()) == len(lean_cases.SUPPORT_NS) == 2
    assert all(
        len(c["ks"]) == 6 and len(c["ps"]) == 6 for c in lean_cases.support_cases()
    )
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
        elif tag == "drawt":
            _, N, sx = line.split(" ", 2)
            lean[("drawt", int(N))] = sx
        elif tag == "draw":
            _, N, i, sx = line.split(" ", 3)
            lean[("draw", int(N), int(i))] = sx
        elif tag == "stream":
            _, s, o, n, w0, w1, w2, w3 = line.split(" ")
            lean[("stream", int(s), int(o), int(n))] = (
                int(w0),
                int(w1),
                int(w2),
                int(w3),
            )
        elif tag == "supk":
            _, N, k, pat = line.split(" ")
            lean[("supk", int(N), int(k))] = int(pat)
        elif tag == "supp":
            _, N, tgt, pat = line.split(" ")
            lean[("supp", int(N), int(tgt))] = int(pat)
        elif tag in ("supmass", "suprenorm", "supkeptlogp"):
            _, N, sx = line.split(" ", 2)
            lean[(tag, int(N))] = sx
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
    # the draw: the threshold and the inclusive prefixes of the probe elements, node for node, at every vocabulary
    # (the pinned ones and N = 1000, which only the executable compares)
    for c in lean_cases.draw_cases():
        assert lean[("drawt", c["N"])] == c["threshold"], ("draw threshold", c["N"])
        checked += 1
        for i, sx in c["prefixes"].items():
            assert lean[("draw", c["N"], i)] == sx, ("draw prefix", c["N"], i)
            checked += 1
    # the support: the threshold procedures on the generated pattern lists equal the Python reference at every
    # vocabulary (the small one also pinned in Lean), and the kept mass / renormalisation terms node for node
    for c in lean_cases.support_cases():
        for k, pt in c["ks"]:
            assert lean[("supk", c["N"], k)] == pt, ("support top-k", c["N"], k)
            checked += 1
        for tgt, pt in c["ps"]:
            assert lean[("supp", c["N"], tgt)] == pt, ("support top-p", c["N"], tgt)
            checked += 1
        assert lean[("supmass", c["N"])] == c["mass"], ("support mass", c["N"])
        assert lean[("suprenorm", c["N"])] == c["renorm"], ("support renorm", c["N"])
        assert lean[("supkeptlogp", c["N"])] == c["keptlogp"], (
            "support kept-set log-probability",
            c["N"],
        )
        checked += 3
    # the stream: the Lean model's blocks are the Python specification's words
    for s, o, n, w in lean_cases.stream_cases():
        assert lean[("stream", s, o, n)] == tuple(w), ("stream block", s, o, n)
        checked += 1
    assert checked == len(lean) >= 60
