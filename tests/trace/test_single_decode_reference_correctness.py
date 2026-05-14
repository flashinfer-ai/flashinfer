"""Reference correctness test for the single_decode trace API."""

from tests.trace.reference_correctness import (
    _run_single_decode,
    run_reference_case,
)


def test_single_decode_reference_correctness():
    run_reference_case(_run_single_decode)
