"""Reference correctness test for the merge_state trace API."""

from tests.trace.reference_correctness import (
    _run_merge_state_reference_correctness,
    run_reference_case,
)


def test_merge_state_reference_correctness():
    run_reference_case(_run_merge_state_reference_correctness)
