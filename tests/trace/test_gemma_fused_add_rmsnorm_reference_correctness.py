"""Reference correctness test for the gemma_fused_add_rmsnorm trace API."""

from tests.trace.reference_correctness import (
    _run_gemma_fused_add_rmsnorm_reference_correctness,
    run_reference_case,
)


def test_gemma_fused_add_rmsnorm_reference_correctness():
    run_reference_case(_run_gemma_fused_add_rmsnorm_reference_correctness)
