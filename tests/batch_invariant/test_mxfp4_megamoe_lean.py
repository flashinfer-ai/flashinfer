# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Independent Lean reconstruction of the MXFP4 MegaMoE arithmetic."""

from pathlib import Path

from flashinfer.moe_ep.batch_invariant import mxfp4_lean_cases

from .test_megamoe_lean import _lake, needs_lake


def test_generated_mxfp4_megamoe_cases_are_current():
    committed = Path(mxfp4_lean_cases.CASES_PATH).read_text()
    assert committed == mxfp4_lean_cases.render()
    assert "sorry" not in committed and "native_decide" not in committed
    cases = mxfp4_lean_cases.conformance_cases()
    assert len(cases) == 8
    assert {c["params"]["arch"] for c in cases} == {"sm_100", "sm_103", "sm_107"}
    assert {c["params"]["quant_kind"] for c in cases} == {"mxfp4", "mxfp4_mxfp8"}
    assert {c["params"]["implementation"] for c in cases} == {"standard", "genphase"}
    assert {c["params"]["apply_routing_weights_before_fc2"] for c in cases} == {
        True,
        False,
    }
    assert any(c["params"]["valid_routes"] is not None for c in cases)


@needs_lake
def test_mxfp4_megamoe_lean_model_matches_python_and_has_no_added_axioms():
    result = _lake("build", "mxfp4megamoeconformance", "axioms", timeout=1800)
    assert result.returncode == 0, (result.stdout + result.stderr)[-6000:]
    result = _lake("exe", "axioms", timeout=600)
    assert result.returncode == 0, (result.stdout + result.stderr)[-6000:]
    assert "0 with sorry, 0 violations" in result.stdout
    result = _lake("exe", "mxfp4megamoeconformance", timeout=600)
    assert result.returncode == 0, result.stderr[-4000:]
    actual = {}
    for line in result.stdout.splitlines():
        prefix, case, lowering, component, expression = line.split(" ", 4)
        assert prefix == "mxmega"
        key = (case, lowering, component)
        assert key not in actual
        actual[key] = expression
    checked = 0
    for case in mxfp4_lean_cases.conformance_cases():
        for lowering in ("flat", "blocked"):
            for component, expression in case["spec"].items():
                key = (case["name"], lowering, component)
                assert actual[key] == expression, key
                checked += 1
    assert len(actual) == checked == 240
