# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The Lean coupling of the block-scale fp8 MoE classes and their quantizer (``flashinfer/batch_invariant/lean/``:
``FP8BlockQuant.lean``, ``MoEFp8Bs.lean`` and the generated ``MoEFp8BsCases.lean``).

Device-free (every CI job): the committed ``MoEFp8BsCases.lean`` is byte for byte what the Python spec and engine emit, so a
change to the quantizer's chains or placements, the class's chains, the grouped space or the dispatcher that was not
propagated to the Lean cases fails here without a toolchain. With ``lake`` installed: ``lake build`` elaborates the model and
the generated cases (kernel ``decide`` throughout), ``lake exe fp8bsconformance`` prints the Lean model's terms, compared here
with the Python S-expressions of every case, and ``lake exe axioms`` enforces the axiom policy of the whole library.
"""

import os
import shutil
import subprocess

import pytest

from flashinfer.fused_moe.batch_invariant import lean_cases_fp8bs as LC

LAKE = shutil.which("lake") or os.path.expanduser("~/.elan/bin/lake")
needs_lake = pytest.mark.skipif(
    not os.path.exists(LAKE), reason="Lean toolchain (elan / lake) not installed"
)


def test_generated_cases_are_current():
    with open(LC.CASES_PATH) as f:
        committed = f.read()
    assert committed == LC.render(), (
        "regenerate with: python -m flashinfer.fused_moe.batch_invariant.lean_cases_fp8bs"
    )
    assert [c["name"] for c in LC.fp8bsq_cases()] == [
        "V128",
        "V64",
        "V32",
        "V16",
        "V8",
        "V4",
    ]
    assert all(c["kernel"] == c["spec"] for c in LC.fp8bsq_cases()), (
        "every placement's engine term is the canonical block tree"
    )
    out = LC.fp8bsq_out_cases()
    assert (
        out["amax0"] != out["amax1"]
        and out["amax0"] in out["s0"]
        and out["amax1"] in out["s1"]
    )
    assert out["amax0"] in out["code0"] and out["amax1"] in out["codeL"]
    assert (
        out["s0"].startswith("(fmax ")
        and "(div " in out["s0"]
        and "(const 448.0)" in out["s0"]
    )
    assert out["code0"].startswith("(cvt e4m3 (mul ")
    names = [n for n, _, _ in LC.moe_cases()]
    assert names == [
        "e4m3bs_gemm1_silu",
        "e4m3bs_gemm1_relu2",
        "e4m3bs_gemm2",
        "e4m3bs32_gemm1_silu",
        "e4m3bs32_gemm1_relu2",
        "e4m3bs32_gemm2",
    ]
    for name in names:
        assert f"theorem moe_{name} :" in committed
    assert len(LC.space()) == 54 and len(LC.dispatch_cases()) >= 1
    assert len(LC.dispatch_cases(32)) >= 1
    # the 32-wide class: its own quantizer instance (the same chains over 32 leaves), its GEMM core's terms, the per-step loop
    assert [c["name"] for c in LC.fp8bsq32_cases()] == ["V32", "V16", "V8", "V4"]
    assert all(c["kernel"] == c["spec"] for c in LC.fp8bsq32_cases())
    out32 = LC.fp8bsq32_out_cases()
    assert (
        out32["amax0"] != LC.fp8bsq_out_cases()["amax0"]
        and out32["amax0"] in out32["s0"]
    )
    k32 = LC.kblock32_cases()
    assert [(KB, fmt) for KB, fmt, _, _ in k32] == [
        (1, "bf16"),
        (4, "bf16"),
        (8, "f32"),
    ]
    assert "(sfb_blk 1 0)" in k32[0][3] and "(sfb_blk 0 0)" in k32[0][2]
    assert "(mma32 (const 0.0) (ab (a 0 0) (b 0 0)))" in k32[0][2]
    for KB, sx in LC.kblock32_loop_cases():
        assert sx.count("(fma ") == KB
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
def test_lean_model_matches_python_spec_and_engine():
    r = _lake("build", timeout=3600)
    assert r.returncode == 0, "Lean proofs failed:\n" + (r.stdout + r.stderr)[-4000:]
    r = _lake("build", "fp8bsconformance", timeout=1800)
    assert r.returncode == 0, (r.stdout + r.stderr)[-4000:]
    r = _lake("exe", "fp8bsconformance", timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    lean = {}
    for line in r.stdout.splitlines():
        tag = line.split(" ", 1)[0]
        if tag == "fp8bsq":
            _, name, kind, sx = line.split(" ", 3)
            lean[("fp8bsq", name, kind)] = sx
        elif tag == "fp8bsqout":
            _, which, sx = line.split(" ", 2)
            lean[("fp8bsqout", which)] = sx
        elif tag == "moe":
            _, name, sx = line.split(" ", 2)
            lean[("moe", name)] = sx
        elif tag == "fp8bsq32":
            _, name, kind, sx = line.split(" ", 3)
            lean[("fp8bsq32", name, kind)] = sx
        elif tag == "fp8bsq32out":
            _, which, sx = line.split(" ", 2)
            lean[("fp8bsq32out", which)] = sx
        elif tag in ("kblock32", "kblock32b"):
            _, KB, fmt, sx = line.split(" ", 3)
            lean[(tag, int(KB), fmt)] = sx
        elif tag == "kblock32loop":
            _, KB, sx = line.split(" ", 2)
            lean[(tag, int(KB))] = sx
        else:
            raise AssertionError(f"unexpected conformance line: {line[:80]}")
    checked = 0
    for c in LC.fp8bsq_cases():
        assert lean[("fp8bsq", c["name"], "spec")] == c["spec"], c["name"]
        assert lean[("fp8bsq", c["name"], "kernel")] == c["kernel"], c["name"]
        checked += 2
    out = LC.fp8bsq_out_cases()
    for which in LC.FP8BSQ_OUT_WHICH:
        assert lean[("fp8bsqout", which)] == out[which], which
        checked += 1
    for name, sx, _ in LC.moe_cases():
        assert lean[("moe", name)] == sx, (
            f"MoE model differs from the Python spec: {name}"
        )
        checked += 1
    # the 32-wide class
    for c in LC.fp8bsq32_cases():
        assert lean[("fp8bsq32", c["name"], "spec")] == c["spec"], c["name"]
        assert lean[("fp8bsq32", c["name"], "kernel")] == c["kernel"], c["name"]
        checked += 2
    out32 = LC.fp8bsq32_out_cases()
    for which in LC.FP8BSQ_OUT_WHICH:
        assert lean[("fp8bsq32out", which)] == out32[which], which
        checked += 1
    for KB, fmt, sx0, sx1 in LC.kblock32_cases():
        assert lean[("kblock32", KB, fmt)] == sx0, (KB, fmt)
        assert lean[("kblock32b", KB, fmt)] == sx1, (KB, fmt)
        checked += 2
    for KB, sx in LC.kblock32_loop_cases():
        assert lean[("kblock32loop", KB)] == sx, KB
        checked += 1
    assert checked == len(lean) == 19 + 3 + 8 + 4 + 6 + 3


@needs_lake
def test_axioms_policy():
    r = _lake("exe", "axioms", timeout=600)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]
    assert "0 with sorry, 0 violations" in r.stdout, r.stdout
