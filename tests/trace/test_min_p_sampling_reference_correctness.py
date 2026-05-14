"""Reference correctness test for the min_p_sampling trace API."""

from tests.trace.reference_correctness import (
    _run_min_p_sampling_reference,
    run_reference_case,
)


def test_min_p_sampling_reference_correctness():
    run_reference_case(_run_min_p_sampling_reference)
