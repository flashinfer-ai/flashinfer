"""Reference correctness test for the cudnn_batch_prefill trace API."""

from tests.trace.reference_correctness import (
    _run_cudnn_batch_prefill_reference_correctness,
    run_reference_case,
)


def test_cudnn_batch_prefill_reference_correctness():
    run_reference_case(_run_cudnn_batch_prefill_reference_correctness)
