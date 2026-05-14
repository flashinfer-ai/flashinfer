"""Reference correctness test for the pod_with_paged_kv_cache_run trace API."""

from tests.trace.reference_correctness import (
    _run_pod_with_paged_kv_cache_run_reference_correctness,
    run_reference_case,
)


def test_pod_with_paged_kv_cache_run_reference_correctness():
    run_reference_case(_run_pod_with_paged_kv_cache_run_reference_correctness)
