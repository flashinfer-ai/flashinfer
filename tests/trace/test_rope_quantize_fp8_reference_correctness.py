"""Reference correctness test for the rope_quantize_fp8 trace API."""

from tests.trace.reference_correctness import (
    _run_rope_quantize_fp8_reference_correctness,
    run_reference_case,
)


def test_rope_quantize_fp8_reference_correctness():
    run_reference_case(_run_rope_quantize_fp8_reference_correctness)
