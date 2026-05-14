"""Reference correctness test for the xqa trace API."""

from tests.trace.reference_correctness import (
    _run_xqa_reference_correctness,
    run_reference_case,
)


def test_xqa_reference_correctness():
    run_reference_case(_run_xqa_reference_correctness)
