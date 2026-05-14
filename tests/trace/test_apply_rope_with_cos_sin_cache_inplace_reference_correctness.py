"""Reference correctness test for the apply_rope_with_cos_sin_cache_inplace trace API."""

from tests.trace.reference_correctness import (
    _run_apply_rope_with_cos_sin_cache_inplace,
    run_reference_case,
)


def test_apply_rope_with_cos_sin_cache_inplace_reference_correctness():
    run_reference_case(_run_apply_rope_with_cos_sin_cache_inplace)
