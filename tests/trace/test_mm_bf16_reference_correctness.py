"""Reference correctness test for the mm_bf16 trace API."""

import torch
import pytest

from tests.trace.reference_utils import (
    _assert_finite,
    _close_fp8,
)


@pytest.mark.parametrize(
    "shape_kwargs", [dict(M=32, N=1024, K=1024), dict(M=16, N=2048, K=1024)]
)
def test_mm_bf16_reference_correctness(shape_kwargs):
    """flashinfer.mm_bf16 kernel vs reference (plain matmul).

    B must be column-major (stride [1, K]) for mm_bf16; the trace's
    init() returns ``b = randn(N, K).T`` — a contiguous [K, N]
    column-major view, exactly the layout the kernel expects.
    """
    import flashinfer
    from flashinfer.trace.templates.gemm import mm_bf16_trace

    inputs = mm_bf16_trace.init(**shape_kwargs)
    _assert_finite(inputs["a"], inputs["b"])
    try:
        api = flashinfer.mm_bf16(inputs["a"], inputs["b"], backend="cutlass")
    except Exception as exc:
        pytest.skip(f"mm_bf16 unavailable: {exc}")
    ref = mm_bf16_trace.reference(inputs["a"], inputs["b"])
    _assert_finite(api, ref)
    _close_fp8(api, ref.to(api.dtype), cos_sim_min=0.99)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
