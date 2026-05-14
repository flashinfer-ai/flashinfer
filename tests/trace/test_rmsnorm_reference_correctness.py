"""Reference correctness test for the rmsnorm trace API."""

from tests.trace.reference_correctness import (
    _run_rmsnorm_reference_correctness,
    run_reference_case,
)


def test_rmsnorm_reference_correctness():
    run_reference_case(_run_rmsnorm_reference_correctness)
