"""Reference correctness test for the append_paged_mla_kv_cache trace API."""

from tests.trace.reference_correctness import (
    _run_append_paged_mla_kv_cache_reference_correctness,
    run_reference_case,
)


def test_append_paged_mla_kv_cache_reference_correctness():
    run_reference_case(_run_append_paged_mla_kv_cache_reference_correctness)
