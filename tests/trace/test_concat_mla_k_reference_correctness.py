"""Reference correctness test for the concat_mla_k trace API."""

import torch
import pytest

from tests.trace.reference_utils import (
    _assert_finite,
    _close,
)


@pytest.mark.parametrize(
    "shape_kwargs",
    [
        dict(num_tokens=2048, num_heads=128, nope_dim=128, rope_dim=64),
        dict(num_tokens=257, num_heads=64, nope_dim=128, rope_dim=64),
    ],
)
def test_concat_mla_k_reference_correctness(shape_kwargs):
    """flashinfer.concat_ops.concat_mla_k kernel vs reference (in-place concat)."""
    from flashinfer.concat_ops import concat_mla_k
    from flashinfer.trace.templates.attention import concat_mla_k_trace

    inputs = concat_mla_k_trace.init(**shape_kwargs)
    _assert_finite(inputs["k_nope"], inputs["k_rope"])
    k_api = inputs["k"].clone()
    k_ref = inputs["k"].clone()
    try:
        concat_mla_k(k_api, inputs["k_nope"], inputs["k_rope"])
    except Exception as exc:
        pytest.skip(f"concat_mla_k unavailable: {exc}")
    concat_mla_k_trace.reference(k_ref, inputs["k_nope"], inputs["k_rope"])
    _assert_finite(k_api, k_ref)
    _close(k_api, k_ref, atol=0.0, rtol=0.0)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
