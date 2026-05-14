"""Reference correctness test for the mxfp4_quantize trace API."""

from tests.trace.reference_correctness import (
    _run_mxfp4_quantize_reference_correctness,
    run_reference_case,
)


def test_mxfp4_quantize_reference_correctness():
    run_reference_case(_run_mxfp4_quantize_reference_correctness)
