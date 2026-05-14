"""Reference correctness test for the sampling_from_probs trace API."""

from tests.trace.reference_correctness import (
    _run_sampling_from_probs_reference,
    run_reference_case,
)


def test_sampling_from_probs_reference_correctness():
    run_reference_case(_run_sampling_from_probs_reference)
