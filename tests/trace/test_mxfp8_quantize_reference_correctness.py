"""Reference correctness test for the mxfp8_quantize trace API."""

from tests.trace.reference_correctness import (
    _run_mxfp8_quantize,
    run_reference_case,
)


def test_mxfp8_quantize_reference_correctness():
    run_reference_case(_run_mxfp8_quantize)
