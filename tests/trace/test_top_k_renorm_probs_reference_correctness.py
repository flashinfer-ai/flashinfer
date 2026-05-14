"""Reference correctness test for the top_k_renorm_probs trace API."""

from tests.trace.reference_correctness import (
    _run_top_k_renorm_probs_reference,
    run_reference_case,
)


def test_top_k_renorm_probs_reference_correctness():
    run_reference_case(_run_top_k_renorm_probs_reference)
