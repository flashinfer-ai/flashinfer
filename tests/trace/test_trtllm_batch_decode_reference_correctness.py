"""Reference correctness test for the trtllm_batch_decode trace API."""

from tests.trace.reference_correctness import (
    _run_trtllm_batch_decode_reference_correctness,
    run_reference_case,
)


def test_trtllm_batch_decode_reference_correctness():
    run_reference_case(_run_trtllm_batch_decode_reference_correctness)
