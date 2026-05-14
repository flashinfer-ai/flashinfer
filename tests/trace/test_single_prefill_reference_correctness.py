"""Reference correctness test for the single_prefill trace API."""

from tests.trace.reference_correctness import (
    _run_single_prefill,
    run_reference_case,
)


def test_single_prefill_reference_correctness():
    run_reference_case(_run_single_prefill)
