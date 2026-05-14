"""Reference correctness test for the bmm_bf16 trace API."""

from tests.trace.reference_correctness import (
    _run_bmm_bf16_reference_correctness,
    run_reference_case,
)


def test_bmm_bf16_reference_correctness():
    run_reference_case(_run_bmm_bf16_reference_correctness)
