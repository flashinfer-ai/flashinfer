"""Reference correctness test for the gemma_rmsnorm trace API."""

from tests.trace.reference_correctness import (
    _run_gemma_rmsnorm_reference_correctness,
    run_reference_case,
)


def test_gemma_rmsnorm_reference_correctness():
    run_reference_case(_run_gemma_rmsnorm_reference_correctness)
