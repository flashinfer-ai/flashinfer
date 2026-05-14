"""Reference correctness test for the silu_and_mul trace API."""

from tests.trace.reference_correctness import (
    _run_silu_and_mul_reference_correctness,
    run_reference_case,
)


def test_silu_and_mul_reference_correctness():
    run_reference_case(_run_silu_and_mul_reference_correctness)
