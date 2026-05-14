"""Reference correctness test for the apply_llama31_rope_pos_ids_inplace trace API."""

from tests.trace.reference_correctness import (
    _run_apply_llama31_rope_pos_ids_inplace,
    run_reference_case,
)


def test_apply_llama31_rope_pos_ids_inplace_reference_correctness():
    run_reference_case(_run_apply_llama31_rope_pos_ids_inplace)
