"""Reference correctness test for the mm_tf32 trace API."""

import torch
import pytest

from tests.trace.reference_utils import (
    _assert_finite,
    _check,
)


@pytest.mark.parametrize(
    "shape_kwargs", [dict(M=32, N=256, K=4096), dict(M=16, N=2048, K=1024)]
)
def test_mm_tf32_reference_correctness(shape_kwargs):
    """flashinfer.mm_tf32 kernel vs reference (plain matmul).

    B must be column-major (stride [1, K]) for mm_tf32; the trace's
    init() returns ``b = randn(N, K).T`` — a contiguous [K, N]
    column-major view, exactly the layout the kernel expects.
    """
    import flashinfer
    from flashinfer.trace.templates.gemm import mm_tf32_trace

    major, minor = torch.cuda.get_device_capability()
    if not flashinfer.mm_tf32.is_compute_capability_supported(major * 10 + minor):
        pytest.skip(f"mm_tf32 is not supported on SM{major}{minor}")
    inputs = mm_tf32_trace.init(**shape_kwargs)
    _assert_finite(inputs["a"], inputs["b"])
    allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        api = flashinfer.mm_tf32(inputs["a"], inputs["b"], backend="auto")
    finally:
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    ref = mm_tf32_trace.reference(inputs["a"], inputs["b"])
    _assert_finite(api, ref)
    _check(
        mm_tf32_trace,
        ref.to(api.dtype),
        api,
        max_mismatch_pct=100.0,
        min_cos_sim=0.99,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
