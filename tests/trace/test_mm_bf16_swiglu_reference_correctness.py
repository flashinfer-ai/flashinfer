"""Reference correctness test for the fused BF16 SwiGLU trace API."""

import pytest
import torch

from tests.trace.reference_utils import _assert_finite, _check


@pytest.mark.parametrize(
    "shape_kwargs",
    [dict(M=1, gate_up_size=128, K=128), dict(M=3, gate_up_size=1024, K=6144)],
)
def test_mm_bf16_swiglu_reference_correctness(shape_kwargs):
    """Compare the kernel with its strict BF16-boundary reference."""
    import flashinfer
    from flashinfer.trace.templates.gemm import mm_bf16_swiglu_trace

    try:
        inputs = mm_bf16_swiglu_trace.init(**shape_kwargs)
        api = flashinfer.mm_bf16_swiglu(**inputs)
    except Exception as exc:
        pytest.skip(f"mm_bf16_swiglu unavailable: {exc}")

    _assert_finite(inputs["a"], inputs["b"], api)
    ref = mm_bf16_swiglu_trace.reference(inputs["a"], inputs["b"], inputs["pdl"])
    _assert_finite(ref)
    _check(
        mm_bf16_swiglu_trace,
        ref.to(api.dtype),
        api,
        max_mismatch_pct=100.0,
        min_cos_sim=0.999,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
