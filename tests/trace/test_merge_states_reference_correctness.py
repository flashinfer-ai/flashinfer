"""Reference correctness test for the merge_states trace API."""

import torch
import pytest

from tests.trace.reference_utils import (
    _assert_finite,
    _close,
)


@pytest.mark.parametrize(
    "shape_kwargs",
    [
        dict(seq_len=16, num_states=3, num_heads=4, head_dim=64),
        dict(seq_len=7, num_states=2, num_heads=2, head_dim=64),
    ],
)
def test_merge_states_reference_correctness(shape_kwargs):
    """flashinfer.merge_states kernel vs reference."""
    import flashinfer
    from flashinfer.trace.templates.cascade import merge_states_trace

    inputs = merge_states_trace.init(**shape_kwargs)
    inputs["v"] = inputs["v"].to(torch.float16)
    _assert_finite(inputs["v"], inputs["s"])
    v_api, s_api = flashinfer.merge_states(inputs["v"], inputs["s"])
    v_ref, s_ref = merge_states_trace.reference(inputs["v"], inputs["s"])
    _assert_finite(v_api, s_api, v_ref, s_ref)
    _close(v_api, v_ref, atol=1e-3, rtol=1e-3)
    _close(s_api, s_ref, atol=1e-3, rtol=1e-3)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
