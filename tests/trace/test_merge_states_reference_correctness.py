"""Reference correctness test for the merge_states trace API."""

from tests.trace.reference_correctness import (
    _run_merge_states_reference_correctness,
    run_reference_case,
)


def test_merge_states_reference_correctness():
    run_reference_case(_run_merge_states_reference_correctness)
