"""Reference correctness test for the segment_gemm_run trace API."""

from tests.trace.reference_correctness import (
    _run_segment_gemm_run_reference_correctness,
    run_reference_case,
)


def test_segment_gemm_run_reference_correctness():
    run_reference_case(_run_segment_gemm_run_reference_correctness)
