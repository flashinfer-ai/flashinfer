"""Reference correctness test for the batch_attention_run trace API."""

from tests.trace.reference_correctness import (
    _run_batch_attention_run_reference_correctness,
    run_reference_case,
)


def test_batch_attention_run_reference_correctness():
    run_reference_case(_run_batch_attention_run_reference_correctness)
