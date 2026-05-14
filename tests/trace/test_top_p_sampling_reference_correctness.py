"""Reference correctness test for the top_p_sampling trace API."""

from tests.trace.reference_correctness import (
    _run_top_p_sampling_reference_correctness,
    run_reference_case,
)


def test_top_p_sampling_reference_correctness():
    run_reference_case(_run_top_p_sampling_reference_correctness)
