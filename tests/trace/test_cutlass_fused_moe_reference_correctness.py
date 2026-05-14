"""Reference correctness test for the cutlass_fused_moe trace API."""

from tests.trace.reference_correctness import (
    _run_cutlass_fused_moe_reference_correctness,
    run_reference_case,
)


def test_cutlass_fused_moe_reference_correctness():
    run_reference_case(_run_cutlass_fused_moe_reference_correctness)
