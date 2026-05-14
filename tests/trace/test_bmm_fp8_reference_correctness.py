"""Reference correctness test for the bmm_fp8 trace API."""

from tests.trace.reference_correctness import (
    _run_bmm_fp8_reference_correctness,
    run_reference_case,
)


def test_bmm_fp8_reference_correctness():
    run_reference_case(_run_bmm_fp8_reference_correctness)
