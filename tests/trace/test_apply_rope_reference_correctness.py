"""Reference correctness test for the apply_rope trace API."""

from tests.trace.reference_correctness import (
    _run_apply_rope,
    run_reference_case,
)


def test_apply_rope_reference_correctness():
    run_reference_case(_run_apply_rope)
