"""Reference correctness test for the chain_speculative_sampling trace API."""

from tests.trace.reference_correctness import (
    _run_chain_speculative_sampling_reference_correctness,
    run_reference_case,
)


def test_chain_speculative_sampling_reference_correctness():
    run_reference_case(_run_chain_speculative_sampling_reference_correctness)
