"""Reference correctness test for the rmsnorm_quant trace API."""

from tests.trace.reference_correctness import (
    _run_rmsnorm_quant,
    run_reference_case,
)


def test_rmsnorm_quant_reference_correctness():
    run_reference_case(_run_rmsnorm_quant)
