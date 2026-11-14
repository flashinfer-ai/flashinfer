# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Actual Lean reconstruction of variants, wire recipes and K partitions."""

from pathlib import Path
from flashinfer.moe_ep.batch_invariant import numerical_lean_cases, wire_lean_cases
from .test_megamoe_lean import _lake, needs_lake


def test_generated_numerical_and_wire_cases_are_current():
    for module in (numerical_lean_cases, wire_lean_cases):
        text = Path(module.CASES_PATH).read_text()
        assert text == module.render()
        assert "sorry" not in text and "native_decide" not in text
    assert len(numerical_lean_cases.conformance_cases()) == 48
    assert len(wire_lean_cases.partition_cases()) == 78


@needs_lake
def test_numerical_variants_and_wire_lean_models_have_no_added_axioms():
    result = _lake("build", "numericalmegamoeconformance", "axioms", timeout=1800)
    assert result.returncode == 0, (result.stdout + result.stderr)[-6000:]
    result = _lake("exe", "axioms", timeout=600)
    assert result.returncode == 0, (result.stdout + result.stderr)[-6000:]
    assert "0 with sorry, 0 violations" in result.stdout
