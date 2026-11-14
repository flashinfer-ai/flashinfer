# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Lean coupling of the ordered BF16 MegaMoE spec and blocked lowering.

The host test detects stale generated cases. With Lean installed, the checked
model reconstructs the five numerical components, preserving operand order,
padding and masked slots, and compares them with the Python spec for all six
plans. These are model proofs, not CUDA race or compiler verification.
"""

import os
import shutil
import subprocess

import pytest

from flashinfer.batch_invariant.fp_interface.term import sexpr
from flashinfer.moe_ep.batch_invariant import lean_cases
from flashinfer.moe_ep.batch_invariant.spec import component_terms

LAKE = shutil.which("lake") or os.path.expanduser("~/.elan/bin/lake")
needs_lake = pytest.mark.skipif(
    not os.path.exists(LAKE), reason="Lean toolchain (elan / lake) not installed"
)


def test_generated_megamoe_cases_are_current():
    with open(lean_cases.CASES_PATH) as source:
        committed = source.read()
    assert committed == lean_cases.render(), (
        "regenerate with: python -m flashinfer.moe_ep.batch_invariant.lean_cases"
    )
    assert "sorry" not in committed and "native_decide" not in committed
    cases = lean_cases.conformance_cases()
    assert {case["early"] for case in cases} == {False, True}
    assert any(case["hidden"] % 64 for case in cases)
    assert any(case["mask"] is not None and not all(case["mask"]) for case in cases)
    assert len(cases) >= 8
    # Clamp signs normalize before the shared numerical chain, including -0.
    assert cases[8]["spec"] == cases[2]["spec"]
    assert lean_cases._clamp(-0.0) == lean_cases._clamp(0.0)
    for case in cases:
        assert len(case["plans"]) == 6
        assert all(value == case["spec"] for value in case["plans"].values())
        assert all(value == case["spec"] for value in case["tiled"].values())
    assert {config.mma_tiler_mnk for case in cases for config in case["configs"]} == {
        config.mma_tiler_mnk for config in lean_cases.ALLOWED_PLAN_CONFIGS
    }


@pytest.mark.parametrize("arch", ("sm_100", "sm_103", "sm_107"))
def test_megamoe_architectures_use_the_proved_ordered_trace(arch):
    # The theorem is parametric in the opaque-atom interpretation. A separate
    # architecture receipt scopes its use; it is not a hardware-equivalence proof.
    for case in lean_cases.conformance_cases():
        terms = component_terms(
            case["hidden"],
            case["intermediate"],
            case["slots"],
            case["experts"],
            case["early"],
            case["clamp"],
            arch,
            valid_routes=case["mask"],
        )
        assert {name: sexpr(value) for name, value in terms.items()} == case["spec"]


def _lake(*args, timeout):
    env = dict(
        os.environ,
        PATH=os.path.dirname(LAKE) + os.pathsep + os.environ["PATH"],
        ELAN_NO_OVERRIDE_NOTICE="1",
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
def test_megamoe_lean_model_matches_python_spec():
    result = _lake("build", "megamoeconformance", "axioms", timeout=1800)
    assert result.returncode == 0, (result.stdout + result.stderr)[-6000:]
    result = _lake("exe", "axioms", timeout=600)
    assert result.returncode == 0, (result.stdout + result.stderr)[-6000:]
    assert "0 with sorry, 0 violations" in result.stdout
    result = _lake("exe", "megamoeconformance", timeout=600)
    assert result.returncode == 0, result.stderr[-4000:]
    actual, tiled, mma = {}, {}, {}
    for line in result.stdout.splitlines():
        if line.startswith("mega "):
            _, case, schedule, carrier, path, component, expression = line.split(" ", 6)
            key = (case, schedule, carrier, path, component)
            assert key not in actual
            actual[key] = expression
        elif line.startswith("mma "):
            _, k, path, expression = line.split(" ", 3)
            mma[(int(k), path)] = expression
        elif line.startswith("megatiled "):
            _, case, geometry, schedule, carrier, component, expression = line.split(
                " ", 6
            )
            key = (case, geometry, schedule, carrier, component)
            assert key not in tiled
            tiled[key] = expression
        else:
            raise AssertionError(f"unexpected conformance output: {line[:100]}")
    checked = 0
    for case in lean_cases.conformance_cases():
        for (schedule, carrier), blocked in case["plans"].items():
            for path, expected in (("flat", case["spec"]), ("blocked", blocked)):
                for component in lean_cases.COMPONENTS:
                    key = (case["name"], schedule, carrier, path, component)
                    assert actual[key] == expected[component], key
                    checked += 1
    assert len(actual) == checked >= 480
    checked = 0
    for case in lean_cases.conformance_cases():
        for (config, (schedule, carrier)), expected in case["tiled"].items():
            for component in lean_cases.COMPONENTS:
                key = (
                    case["name"],
                    lean_cases.tile_name(config),
                    schedule,
                    carrier,
                    component,
                )
                assert tiled[key] == expected[component], key
                checked += 1
    assert len(tiled) == checked >= 2000
    assert mma == lean_cases.mma_strings()
