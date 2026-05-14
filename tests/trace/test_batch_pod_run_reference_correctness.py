"""Reference correctness test for the batch_pod_run trace API."""

from tests.trace.reference_correctness import (
    _run_batch_pod_run_reference_correctness,
    run_reference_case,
)


def test_batch_pod_run_reference_correctness():
    run_reference_case(_run_batch_pod_run_reference_correctness)
