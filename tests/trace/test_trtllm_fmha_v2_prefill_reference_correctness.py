"""Reference correctness test for the trtllm_fmha_v2_prefill trace API."""

from tests.trace.reference_correctness import (
    _run_trtllm_fmha_v2_prefill_reference_correctness,
    run_reference_case,
)


def test_trtllm_fmha_v2_prefill_reference_correctness():
    run_reference_case(_run_trtllm_fmha_v2_prefill_reference_correctness)
