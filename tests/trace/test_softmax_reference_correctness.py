"""Reference correctness test for the softmax trace API."""

from tests.trace.reference_correctness import (
    _run_softmax_reference,
    run_reference_case,
)


def test_softmax_reference_correctness():
    run_reference_case(_run_softmax_reference)
