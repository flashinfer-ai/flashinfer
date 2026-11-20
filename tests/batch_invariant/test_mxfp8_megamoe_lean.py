# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Independent Lean coupling of all E4M3/E5M2 MegaMoE classes."""

from pathlib import Path

from flashinfer.moe_ep.batch_invariant import mxfp8_lean_cases

from .test_megamoe_lean import _lake, needs_lake


def test_generated_mxfp8_megamoe_cases_are_current():
    committed = Path(mxfp8_lean_cases.CASES_PATH).read_text()
    assert committed == mxfp8_lean_cases.render()
    assert "sorry" not in committed and "native_decide" not in committed
    cases = mxfp8_lean_cases.conformance_cases()
    assert len(cases) == 16
    assert {c["params"]["implementation"] for c in cases} == {
        "legacy",
        "standard",
        "genphase",
    }
    assert {c["params"]["quant_kind"] for c in cases} == {"mxfp8_e4m3", "mxfp8_e5m2"}
    assert all(
        c["params"]["apply_routing_weights_before_fc2"]
        for c in cases
        if c["params"]["implementation"] == "legacy"
    )


@needs_lake
def test_mxfp8_megamoe_lean_reconstructs_all_components_without_new_axioms():
    result = _lake("build", "mxfp8megamoeconformance", "axioms", timeout=1800)
    assert result.returncode == 0, (result.stdout + result.stderr)[-6000:]
    result = _lake("exe", "axioms", timeout=600)
    assert result.returncode == 0, (result.stdout + result.stderr)[-6000:]
    assert "0 with sorry, 0 violations" in result.stdout
    result = _lake("exe", "mxfp8megamoeconformance", timeout=600)
    assert result.returncode == 0, result.stderr[-4000:]
    actual = {}
    for line in result.stdout.splitlines():
        prefix, case, lowering, component, expression = line.split(" ", 4)
        assert prefix == "mx8mega"
        key = (case, lowering, component)
        assert key not in actual
        actual[key] = expression
    checked = 0
    for case in mxfp8_lean_cases.conformance_cases():
        for lowering in ("flat", "blocked"):
            for component, expected in case["spec"].items():
                key = (case["name"], lowering, component)
                assert actual[key] == expected, key
                checked += 1
    assert checked == len(actual) == 480
