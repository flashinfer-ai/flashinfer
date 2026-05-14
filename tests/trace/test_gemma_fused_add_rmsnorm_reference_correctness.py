"""Reference correctness test for the gemma_fused_add_rmsnorm trace API."""

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
def test_gemma_fused_add_rmsnorm_reference_correctness(shape_kwargs):
    """flashinfer.gemma_fused_add_rmsnorm kernel vs reference.

    Same in-place mutation pattern as fused_add_rmsnorm; reference returns
    only the normalized output.
    """
    import flashinfer
    from flashinfer.trace.templates.norm import gemma_fused_add_rmsnorm_trace

    inputs = gemma_fused_add_rmsnorm_trace.init(**shape_kwargs)
    x_orig, res_orig = inputs["input"].clone(), inputs["residual"].clone()
    _assert_finite(x_orig, res_orig, inputs["weight"])
    x_api = inputs["input"].clone()
    res_api = inputs["residual"].clone()
    flashinfer.gemma_fused_add_rmsnorm(x_api, res_api, inputs["weight"], eps=1e-6)
    ref_norm = gemma_fused_add_rmsnorm_trace.reference(
        x_orig, res_orig, inputs["weight"]
    )
    _assert_finite(x_api, res_api, ref_norm)
    _close(x_api, ref_norm, atol=1e-3, rtol=1e-3)
    _close(res_api, res_orig + x_orig, atol=1e-3, rtol=1e-3)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
