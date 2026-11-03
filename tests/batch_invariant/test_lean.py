# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The Lean coupling of the classes ``gemm.bf16`` and ``gemm.e4m3`` / ``gemm.e5m2`` (``flashinfer/batch_invariant/lean/``).

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
from flashinfer.gemm.batch_invariant.spec import MMA_K, MMA_K_OF

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


def test_rendering_takes_the_nominal_parts():
    """The dispatch cases are the nominal parts' (``nominal=True``): a device's queried co-resident capacity or L2 size,
    recorded by the API earlier in the same process, never changes what `render` emits, while the device's own dispatch
    does follow its numbers."""
    from flashinfer.gemm.batch_invariant import heuristic as H
    from flashinfer.gemm.batch_invariant.schedule import chunks_for, default_config

    key = ("sm_100", 148)
    shapes = [
        (M, N, K)
        for M in (1, 300, 4096, 16384)
        for N, K in ((4096, 4096), (512, 16384), (14336, 4096))
    ]

    def picks(nominal):
        return [
            default_config(M, N, K, chunks_for(N, K), b, 148, "sm_100", nominal=nominal)
            for M, N, K in shapes
            for b in ("n", "k")
        ]

    before = picks(True)
    saved = (H._QUERIED_CAPACITY.pop(key, None), H._QUERIED_L2.pop(key, None))
    try:
        H.register_capacity("sm_100", 148, 4, 30)
        H.register_capacity("sm_100", 148, 8, 14)
        H.register_l2("sm_100", 148, 32 * 2**20)
        assert H.capacity_for("sm_100", 148)[4] == 30
        assert H.capacity_for("sm_100", 148, nominal=True)[4] == 33
        assert H.l2_for("sm_100", 148) == 32 * 2**20
        assert H.l2_for("sm_100", 148, nominal=True) == H.PART_TABLE["sm_100"].l2_bytes
        assert picks(True) == before
        assert picks(False) != before
    finally:
        H._QUERIED_CAPACITY.pop(key, None)
        H._QUERIED_L2.pop(key, None)
        if saved[0] is not None:
            H._QUERIED_CAPACITY[key] = saved[0]
        if saved[1] is not None:
            H._QUERIED_L2[key] = saved[1]
        H.rank.cache_clear()


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
        if tag in ("gemm", "gemm32"):
            _, steps, C, sx = line.split(" ", 3)
            lean[(tag, int(steps), int(C))] = sx
        elif tag in ("gemmloop", "gemm32loop"):
            _, nblk, sx = line.split(" ", 2)
            lean[(tag, int(nblk))] = sx
        elif tag in ("gemmout", "gemm32out"):
            _, steps, fmt, sx = line.split(" ", 3)
            lean[(tag, int(steps), fmt)] = sx
        elif tag in ("chunks", "chunks32"):
            _, N, K, C = line.split(" ")
            lean[(tag, int(N), int(K))] = int(C)
        else:
            raise AssertionError(f"unexpected conformance line: {line[:80]}")
    checked = 0
    # the bf16 class at the mma16 step and the fp8 classes at the mma32 step (one Lean step for e4m3 and e5m2)
    for in_fmt, p in (("bf16", ""), (lean_cases.FP8_FMT, "32")):
        mk = MMA_K_OF[in_fmt]
        for K, C, sx in lean_cases.spec_cases(in_fmt):
            assert lean[(f"gemm{p}", K // mk, C)] == sx, (in_fmt, K, C)
            checked += 1
        for nblk, sx in lean_cases.loop_cases(in_fmt)[1]:
            assert lean[(f"gemm{p}loop", nblk)] == sx, (in_fmt, nblk)
            checked += 1
        for K, fmt, sx in lean_cases.out_cases(in_fmt):
            assert lean[(f"gemm{p}out", K // mk, fmt)] == sx, (in_fmt, K, fmt)
            checked += 1
        for N, K, C in lean_cases.chunk_table(in_fmt):
            assert lean[(f"chunks{p}", N, K)] == C, (in_fmt, N, K)
            checked += 1
    assert MMA_K_OF["bf16"] == MMA_K
    assert checked == len(lean) >= 40


@needs_lake
def test_axioms_policy():
    r = _lake("exe", "axioms", timeout=600)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]
    assert "0 with sorry, 0 violations" in r.stdout, r.stdout
    assert f"{len(lean_cases.DECIDED_THEOREMS)} named table theorems" in r.stdout, (
        r.stdout
    )
