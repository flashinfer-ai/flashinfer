"""Reference correctness test for the tgv_gemm_sm100 trace API."""

from tests.trace.reference_correctness import (
    _run_tgv_gemm_sm100_reference_correctness,
    run_reference_case,
)


def test_tgv_gemm_sm100_reference_correctness():
    run_reference_case(_run_tgv_gemm_sm100_reference_correctness)
