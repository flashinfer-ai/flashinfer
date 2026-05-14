"""Reference correctness test for the multi_level_cascade_run trace API."""

from tests.trace.reference_correctness import (
    _run_multi_level_cascade_run_reference_correctness,
    run_reference_case,
)


def test_multi_level_cascade_run_reference_correctness():
    run_reference_case(_run_multi_level_cascade_run_reference_correctness)
