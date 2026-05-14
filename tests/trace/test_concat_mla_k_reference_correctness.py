"""Reference correctness test for the concat_mla_k trace API."""

from tests.trace.reference_correctness import (
    _run_concat_mla_k_reference_correctness,
    run_reference_case,
)


def test_concat_mla_k_reference_correctness():
    run_reference_case(_run_concat_mla_k_reference_correctness)
