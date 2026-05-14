"""Reference correctness test for the layernorm trace API."""

from tests.trace.reference_correctness import (
    _run_layernorm_reference_correctness,
    run_reference_case,
)


def test_layernorm_reference_correctness():
    run_reference_case(_run_layernorm_reference_correctness)
