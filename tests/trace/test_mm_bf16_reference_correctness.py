"""Reference correctness test for the mm_bf16 trace API."""

from tests.trace.reference_correctness import (
    _run_mm_bf16_reference_correctness,
    run_reference_case,
)


def test_mm_bf16_reference_correctness():
    run_reference_case(_run_mm_bf16_reference_correctness)
