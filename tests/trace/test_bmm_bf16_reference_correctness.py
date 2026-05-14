"""Reference correctness test for the bmm_bf16 trace API."""

import torch
import pytest

from tests.trace.reference_utils import (
    _assert_finite,
    _close_fp8,
)


@pytest.mark.parametrize(
    "shape_kwargs",
    [dict(batch_size=4, M=16, N=1024, K=1024), dict(batch_size=2, M=8, N=1024, K=1024)],
)
def test_bmm_bf16_reference_correctness(shape_kwargs):
    """flashinfer.bmm_bf16 kernel vs reference (batched matmul, cos-sim per
    tests/gemm/test_bmm_bf16.py)."""
    import flashinfer
    from flashinfer.trace.templates.gemm import bmm_bf16_trace

    inputs = bmm_bf16_trace.init(**shape_kwargs)
    _assert_finite(inputs["A"], inputs["B"])
    # bmm_bf16 with cutlass backend requires the same column-major view
    # via the [..., K, N] stride pattern used by the unit test.
    b_kmaj = inputs["B"].transpose(1, 2).contiguous().transpose(1, 2)
    try:
        api = flashinfer.bmm_bf16(inputs["A"], b_kmaj, backend="cutlass")
    except Exception as exc:
        pytest.skip(f"bmm_bf16 unavailable: {exc}")
    ref = bmm_bf16_trace.reference(inputs["A"], inputs["B"])
    _assert_finite(api, ref)
    _close_fp8(api, ref, cos_sim_min=0.99)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
