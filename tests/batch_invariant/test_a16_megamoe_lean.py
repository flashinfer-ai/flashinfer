# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Independent Lean reconstruction of decoded-weight dense-K16 A16 traces."""

from pathlib import Path

from flashinfer.moe_ep.batch_invariant import a16_lean_cases
from flashinfer.moe_ep.batch_invariant.a16_spec import KINDS, PLANS

from .test_megamoe_lean import _lake, needs_lake


def test_generated_a16_megamoe_cases_are_current():
    committed = Path(a16_lean_cases.CASES_PATH).read_text()
    assert committed == a16_lean_cases.render()
    assert "sorry" not in committed and "native_decide" not in committed
    cases = a16_lean_cases.conformance_cases()
    assert len(cases) == 36
    assert {c["params"]["kind"] for c in cases} == set(KINDS)
    assert {c["params"]["arch"] for c in cases} == {"sm_100", "sm_103", "sm_107"}
    assert {c["params"]["output_mode"] for c in cases} == {"finalized", "deferred"}
    assert {c["params"]["handoff_norm"] for c in cases} == {False, True}
    assert {c["variant"].activation for c in cases} == {"swiglu", "situ"}
    assert any(c["params"]["hidden"] % 256 for c in cases)
    for case in cases:
        assert set(case["plans"]) == set(PLANS)
        assert all(value == case["spec"] for value in case["plans"].values())


@needs_lake
def test_a16_megamoe_lean_model_matches_python_and_has_no_added_axioms():
    result = _lake("build", "a16megamoeconformance", "axioms", timeout=1800)
    assert result.returncode == 0, (result.stdout + result.stderr)[-6000:]
    result = _lake("exe", "axioms", timeout=600)
    assert result.returncode == 0, (result.stdout + result.stderr)[-6000:]
    assert "0 with sorry, 0 violations" in result.stdout
    result = _lake("exe", "a16megamoeconformance", timeout=600)
    assert result.returncode == 0, result.stderr[-4000:]
    actual = {}
    for line in result.stdout.splitlines():
        prefix, case, schedule, carrier, lowering, component, expression = line.split(
            " ", 6
        )
        assert prefix == "a16mega"
        key = (case, schedule, carrier, lowering, component)
        assert key not in actual
        actual[key] = expression
    checked = 0
    for case in a16_lean_cases.conformance_cases():
        for schedule, carrier in PLANS:
            for lowering in ("flat", "blocked"):
                for component, expression in case["spec"].items():
                    key = (case["name"], schedule, carrier, lowering, component)
                    assert actual[key] == expression, key
                    checked += 1
    assert len(actual) == checked
    assert checked == 1800
