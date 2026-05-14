"""Reference correctness test for the top_p_renorm_probs trace API."""

from tests.trace.reference_correctness import (
    _run_top_p_renorm_probs_reference,
    run_reference_case,
)


def test_top_p_renorm_probs_reference_correctness():
    run_reference_case(_run_top_p_renorm_probs_reference)
