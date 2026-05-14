"""Reference correctness test for the block_sparse_run trace API."""

from tests.trace.reference_correctness import (
    _run_block_sparse_run_reference_correctness,
    run_reference_case,
)


def test_block_sparse_run_reference_correctness():
    run_reference_case(_run_block_sparse_run_reference_correctness)
