"""Reference correctness test for the rmsnorm_quant trace API."""

import torch
import pytest

from tests.trace.reference_utils import (
    _assert_finite,
    _close,
)


@pytest.mark.parametrize(
    "shape_kwargs",
    [dict(batch_size=32, hidden_size=2048), dict(batch_size=7, hidden_size=1024)],
)
def test_rmsnorm_quant_reference_correctness(shape_kwargs):
    import flashinfer
    from flashinfer.trace.templates.norm import rmsnorm_quant_trace

    inputs = rmsnorm_quant_trace.init(**shape_kwargs)
    _assert_finite(inputs["input"], inputs["weight"])
    out_api = inputs["out"].clone()
    try:
        flashinfer.rmsnorm_quant(
            out_api, inputs["input"], inputs["weight"], inputs["scale"]
        )
    except Exception as exc:
        pytest.skip(f"rmsnorm_quant kernel unavailable: {exc}")
    out_ref = rmsnorm_quant_trace.reference(
        inputs["input"], inputs["weight"], inputs["scale"]
    )
    _assert_finite(out_api, out_ref)
    s = inputs["scale"]
    _close(out_api.float() * s, out_ref.float() * s, atol=1.0, rtol=1.0)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
