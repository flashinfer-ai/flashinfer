"""Reference correctness test for the merge_state trace API."""

import torch
import pytest

from tests.trace.reference_utils import (
    _assert_finite,
    _check,
)


@pytest.mark.parametrize(
    "shape_kwargs",
    [
        dict(seq_len=16, num_heads=4, head_dim=64),
        dict(seq_len=7, num_heads=2, head_dim=64),
    ],
)
def test_merge_state_reference_correctness(shape_kwargs):
    """flashinfer.merge_state kernel vs reference."""
    import flashinfer
    from flashinfer.trace.templates.cascade import merge_state_trace

    inputs = merge_state_trace.init(**shape_kwargs)
    inputs["v_a"] = inputs["v_a"].to(torch.float16)
    inputs["v_b"] = inputs["v_b"].to(torch.float16)
    _assert_finite(inputs["v_a"], inputs["s_a"], inputs["v_b"], inputs["s_b"])
    v_api, s_api = flashinfer.merge_state(
        inputs["v_a"], inputs["s_a"], inputs["v_b"], inputs["s_b"]
    )
    v_ref, s_ref = merge_state_trace.reference(
        inputs["v_a"], inputs["s_a"], inputs["v_b"], inputs["s_b"]
    )
    _assert_finite(v_api, s_api, v_ref, s_ref)
    _check(merge_state_trace, (v_ref, s_ref), (v_api, s_api), atol=1e-3, rtol=1e-3)
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def test_merge_state_reference_preserves_empty_state():
    """Merging two empty attention states returns the empty state."""
    from flashinfer.trace.templates.cascade import merge_state_trace

    v_a = torch.zeros((1, 1, 64), dtype=torch.float16)
    v_b = torch.zeros_like(v_a)
    s_a = torch.full((1, 1), -torch.inf, dtype=torch.float32)
    s_b = torch.full_like(s_a, -torch.inf)

    v_merged, s_merged = merge_state_trace.reference(v_a, s_a, v_b, s_b)

    torch.testing.assert_close(v_merged, torch.zeros_like(v_merged))
    torch.testing.assert_close(s_merged, torch.full_like(s_merged, -torch.inf))


def test_merge_state_in_place_reference_preserves_empty_state():
    """Merging two empty attention states in place returns the empty state."""
    from flashinfer.trace.templates.cascade import merge_state_in_place_trace

    v = torch.zeros((1, 1, 64), dtype=torch.float16)
    v_other = torch.zeros_like(v)
    s = torch.full((1, 1), -torch.inf, dtype=torch.float32)
    s_other = torch.full_like(s, -torch.inf)

    v_merged, s_merged = merge_state_in_place_trace.reference(v, s, v_other, s_other)

    torch.testing.assert_close(v_merged, torch.zeros_like(v_merged))
    torch.testing.assert_close(s_merged, torch.full_like(s_merged, -torch.inf))


def test_merge_states_reference_preserves_empty_state():
    """Reducing empty attention states returns the empty state."""
    from flashinfer.trace.templates.cascade import merge_states_trace

    v = torch.zeros((1, 2, 1, 64), dtype=torch.float16)
    s = torch.full((1, 2, 1), -torch.inf, dtype=torch.float32)

    v_merged, s_merged = merge_states_trace.reference(v, s)

    torch.testing.assert_close(v_merged, torch.zeros_like(v_merged))
    torch.testing.assert_close(s_merged, torch.full_like(s_merged, -torch.inf))
