"""Reference correctness test for the merge_state trace API."""

import torch
import pytest

from tests.trace.reference_utils import (
    _assert_finite,
    _close,
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
    _close(v_api, v_ref, atol=1e-3, rtol=1e-3)
    _close(s_api, s_ref, atol=1e-3, rtol=1e-3)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
