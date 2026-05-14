"""Reference correctness test for the merge_state_in_place trace API."""

import torch
import pytest

from tests.trace.reference_utils import (
    _assert_finite,
    _close,
)


@pytest.mark.parametrize(
    "shape_kwargs",
    [
        dict(seq_len=128, num_heads=32, head_dim=128),
        dict(seq_len=17, num_heads=8, head_dim=64),
    ],
)
def test_merge_state_in_place_reference_correctness(shape_kwargs):
    import flashinfer
    from flashinfer.trace.templates.cascade import merge_state_in_place_trace

    # Use fp16 V (matches tests/attention/test_shared_prefix_kernels.py);
    # 1e-3 tolerance is too tight for bf16 (4e-3 per ULP). The init builds
    # bf16 by default, so we cast here.
    inputs = merge_state_in_place_trace.init(**shape_kwargs)
    inputs["v"] = inputs["v"].to(torch.float16)
    inputs["v_other"] = inputs["v_other"].to(torch.float16)
    _assert_finite(inputs["v"], inputs["s"], inputs["v_other"], inputs["s_other"])
    v_api = inputs["v"].clone()
    s_api = inputs["s"].clone()
    flashinfer.merge_state_in_place(v_api, s_api, inputs["v_other"], inputs["s_other"])
    v_ref, s_ref = merge_state_in_place_trace.reference(
        inputs["v"], inputs["s"], inputs["v_other"], inputs["s_other"]
    )
    _assert_finite(v_api, s_api, v_ref, s_ref)
    _close(v_api, v_ref, atol=1e-3, rtol=1e-3)
    _close(s_api, s_ref, atol=1e-3, rtol=1e-3)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
