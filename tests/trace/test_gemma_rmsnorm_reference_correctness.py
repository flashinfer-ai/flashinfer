"""Reference correctness test for the gemma_rmsnorm trace API."""

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
def test_gemma_rmsnorm_reference_correctness(shape_kwargs):
    """flashinfer.gemma_rmsnorm kernel vs reference."""
    import flashinfer
    from flashinfer.trace.templates.norm import gemma_rmsnorm_trace

    inputs = gemma_rmsnorm_trace.init(**shape_kwargs)
    _assert_finite(inputs["input"], inputs["weight"])
    api = flashinfer.gemma_rmsnorm(inputs["input"], inputs["weight"], eps=1e-6)
    ref = gemma_rmsnorm_trace.reference(inputs["input"], inputs["weight"])
    _assert_finite(api, ref)
    _close(api, ref, atol=1e-3, rtol=1e-3)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
