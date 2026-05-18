"""Reference correctness test for the fused_add_rmsnorm trace API."""

import torch
import pytest

from tests.trace.reference_utils import (
    _assert_finite,
    _close,
)


@pytest.mark.parametrize(
    "shape_kwargs",
    [dict(batch_size=8, hidden_size=256), dict(batch_size=3, hidden_size=320)],
)
def test_fused_add_rmsnorm_reference_correctness(shape_kwargs):
    """flashinfer.fused_add_rmsnorm kernel vs reference.

    The kernel mutates input (→ norm output) and residual (→ residual + input).
    The trace reference returns the normalized output only; we compare that
    against the mutated input and verify the residual update by hand.
    """
    import flashinfer
    from flashinfer.trace.templates.norm import fused_add_rmsnorm_trace

    inputs = fused_add_rmsnorm_trace.init(**shape_kwargs)
    x_orig, res_orig = inputs["input"].clone(), inputs["residual"].clone()
    _assert_finite(x_orig, res_orig, inputs["weight"])
    x_api = inputs["input"].clone()
    res_api = inputs["residual"].clone()
    flashinfer.fused_add_rmsnorm(x_api, res_api, inputs["weight"], eps=1e-6)
    ref_norm = fused_add_rmsnorm_trace.reference(x_orig, res_orig, inputs["weight"])
    _assert_finite(x_api, res_api, ref_norm)
    _close(x_api, ref_norm, atol=1e-3, rtol=1e-3)
    _close(res_api, res_orig + x_orig, atol=1e-3, rtol=1e-3)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
