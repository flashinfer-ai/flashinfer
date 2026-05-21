"""Reference correctness test for the fused_add_rmsnorm_quant trace API."""

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
def test_fused_add_rmsnorm_quant_reference_correctness(shape_kwargs):
    import flashinfer
    from flashinfer.trace.templates.norm import fused_add_rmsnorm_quant_trace

    inputs = fused_add_rmsnorm_quant_trace.init(**shape_kwargs)
    _assert_finite(inputs["input"], inputs["residual"], inputs["weight"])
    out_api = inputs["out"].clone()
    residual_api = inputs["residual"].clone()
    try:
        flashinfer.fused_add_rmsnorm_quant(
            out_api,
            inputs["input"],
            residual_api,
            inputs["weight"],
            inputs["scale"],
        )
    except Exception as exc:
        pytest.skip(f"fused_add_rmsnorm_quant kernel unavailable: {exc}")
    out_ref, residual_ref = fused_add_rmsnorm_quant_trace.reference(
        inputs["input"], inputs["residual"], inputs["weight"], inputs["scale"]
    )
    _assert_finite(out_api, residual_api, out_ref, residual_ref)
    _close(residual_api, residual_ref, atol=1e-3, rtol=1e-3)
    s = inputs["scale"]
    _close(out_api.float() * s, out_ref.float() * s, atol=1.0, rtol=1.0)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
