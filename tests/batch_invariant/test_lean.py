# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The Lean coupling of the classes ``gemm.bf16``, ``gemm.e4m3`` / ``gemm.e5m2``, ``gemm.e4m3.kblock128`` (the block-scale
fp8 class) and ``gemm.mxfp8`` (both instructions' classes) (``flashinfer/batch_invariant/lean/``).

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
    for name in lean_cases.DECIDED_THEOREMS + lean_cases.MODULE_DECIDED_THEOREMS:
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
    kinds = {"gemmbias": "spec", "fp4class": "fp4class", "attn": "attn"}
    # the block-scale fp8 class: the output at (0, 0) and at (3, 129) per (blocks, format), the promotion loop per block count
    kinds.update({"gemmfp8bs": "out", "gemmfp8bs2": "out", "gemmfp8bsloop": "loop"})
    for F in lean_cases.FORMATS:
        kinds[F.tag] = "spec"
        kinds[F.tag + "loop"] = "loop"
        kinds[F.tag + "out"] = "out"
        if F.table is not None:
            kinds[F.chunk_tag] = "chunks"
    lean = {}
    for line in r.stdout.splitlines():
        tag = line.split(" ", 1)[0]
        kind = kinds.get(tag)
        if kind == "spec":
            _, steps, C, sx = line.split(" ", 3)
            lean[(tag, int(steps), int(C))] = sx
        elif kind == "loop":
            _, nblk, sx = line.split(" ", 2)
            lean[(tag, int(nblk))] = sx
        elif kind == "out":
            _, steps, fmt, sx = line.split(" ", 3)
            lean[(tag, int(steps), fmt)] = sx
        elif kind == "chunks":
            _, N, K, C = line.split(" ")
            lean[(tag, int(N), int(K))] = int(C)
        elif kind == "fp4class":
            _, N, K, b = line.split(" ")
            lean[("fp4class", int(N), int(K))] = bool(int(b))
        elif kind == "attn":
            _, name, sx = line.split(" ", 2)
            lean[("attn", name)] = sx
        else:
            raise AssertionError(f"unexpected conformance line: {line[:80]}")
    checked = 0
    # the bf16 class at the mma16 step, the fp8 classes at the mma32 step (one Lean step for e4m3 and e5m2), the MXFP8
    # classes at the mma32mx / mma64mx8 steps (one chunk rule for both, printed once)
    for F in lean_cases.FORMATS:
        in_fmt, mk = F.in_fmt, MMA_K_OF[F.in_fmt]
        for K, C, sx in lean_cases.spec_cases(in_fmt):
            assert lean[(F.tag, K // mk, C)] == sx, (in_fmt, K, C)
            checked += 1
        for nblk, sx in lean_cases.loop_cases(in_fmt)[1]:
            assert lean[(F.tag + "loop", nblk)] == sx, (in_fmt, nblk)
            checked += 1
        for K, fmt, sx in lean_cases.out_cases(in_fmt):
            assert lean[(F.tag + "out", K // mk, fmt)] == sx, (in_fmt, K, fmt)
            checked += 1
        for N, K, C in lean_cases.chunk_table(in_fmt):
            assert lean[(F.chunk_tag, N, K)] == C, (in_fmt, N, K)
            if F.table is not None:
                checked += 1
    # the sm_107 NVFP4 class rule
    for N, K, b in lean_cases.class_table():
        assert lean[("fp4class", N, K)] == b, (N, K)
        checked += 1
    # the biased chain of the bf16 classes
    for K, C, sx in lean_cases.bias_cases():
        assert lean[("gemmbias", K // MMA_K, C)] == sx, (K, C)
        checked += 1
    # the block-scale fp8 class: the spec's output at (0, 0) and at (3, 129) node for node, and the kernel's promotion loop
    # against the engine's per-block traces folded by the block digit
    for KB, fmt, sx0, sx1 in lean_cases.fp8bs_cases():
        assert lean[("gemmfp8bs", KB, fmt)] == sx0, ("e4m3_bs", KB, fmt)
        assert lean[("gemmfp8bs2", KB, fmt)] == sx1, ("e4m3_bs (3, 129)", KB, fmt)
        checked += 2
    for KB, sx in lean_cases.fp8bs_loop_cases():
        assert lean[("gemmfp8bsloop", KB)] == sx, ("e4m3_bs loop", KB)
        checked += 1
    for name, *_, sx in lean_cases.attn_cases():
        assert lean[("attn", name)] == sx, name
        checked += 1
    assert MMA_K_OF["bf16"] == MMA_K
    assert checked == len(lean) >= 80


@needs_lake
def test_axioms_policy():
    r = _lake("exe", "axioms", timeout=600)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]
    assert "0 with sorry, 0 violations" in r.stdout, r.stdout
    n_decided = len(lean_cases.DECIDED_THEOREMS) + len(
        lean_cases.MODULE_DECIDED_THEOREMS
    )
    assert f"{n_decided} named table theorems" in r.stdout, r.stdout
