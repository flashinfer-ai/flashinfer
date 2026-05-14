"""Reference correctness test for the nvfp4_quantize trace API."""

from tests.trace.reference_correctness import (
    _run_nvfp4_quantize_reference_correctness,
    run_reference_case,
)


def test_nvfp4_quantize_reference_correctness():
    run_reference_case(_run_nvfp4_quantize_reference_correctness)
