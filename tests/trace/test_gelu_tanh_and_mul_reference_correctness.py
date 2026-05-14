"""Reference correctness test for the gelu_tanh_and_mul trace API."""

from tests.trace.reference_correctness import (
    _run_gelu_tanh_and_mul_reference_correctness,
    run_reference_case,
)


def test_gelu_tanh_and_mul_reference_correctness():
    run_reference_case(_run_gelu_tanh_and_mul_reference_correctness)
