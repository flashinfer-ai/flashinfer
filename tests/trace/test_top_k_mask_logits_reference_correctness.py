"""Reference correctness test for the top_k_mask_logits trace API."""

from tests.trace.reference_correctness import (
    _run_top_k_mask_logits_reference,
    run_reference_case,
)


def test_top_k_mask_logits_reference_correctness():
    run_reference_case(_run_top_k_mask_logits_reference)
