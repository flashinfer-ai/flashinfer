"""Reference correctness test for the fp4_quantize_round_trip trace API."""

from tests.trace.reference_correctness import (
    _run_fp4_quantize_round_trip,
    run_reference_case,
)


def test_fp4_quantize_round_trip_reference_correctness():
    run_reference_case(_run_fp4_quantize_round_trip)
