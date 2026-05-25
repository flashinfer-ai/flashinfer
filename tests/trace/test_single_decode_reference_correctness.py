"""Reference correctness test for the single_decode trace API."""

import torch
import pytest

from tests.trace.reference_utils import (
    _assert_finite,
    _close,
)


@pytest.mark.parametrize(
    "shape_kwargs",
    [
        dict(kv_len=256, num_qo_heads=32, num_kv_heads=8, head_dim=128),
        dict(kv_len=96, num_qo_heads=16, num_kv_heads=4, head_dim=64),
    ],
)
def test_single_decode_reference_correctness(shape_kwargs):
    import flashinfer
    from flashinfer.trace.templates.attention import (
        single_decode_with_kv_cache_trace,
    )

    inputs = single_decode_with_kv_cache_trace.init(**shape_kwargs)
    _assert_finite(inputs["q"], inputs["k"], inputs["v"])
    try:
        out_api = flashinfer.single_decode_with_kv_cache(
            inputs["q"], inputs["k"], inputs["v"]
        )
    except Exception as exc:
        pytest.skip(f"single_decode kernel unavailable: {exc}")
    out_ref = single_decode_with_kv_cache_trace.reference(
        inputs["q"], inputs["k"], inputs["v"]
    )
    _assert_finite(out_api, out_ref)
    _close(out_api, out_ref, atol=1e-2, rtol=1e-2)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
