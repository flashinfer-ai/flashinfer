"""Reference correctness test for the bmm_fp8 trace API."""

import torch
import pytest

from tests.trace.reference_utils import (
    _check,
    _skip_if_not_sm100_or_103,
)


@pytest.mark.parametrize("with_bias", [False, True])
@pytest.mark.parametrize(
    "shape_kwargs",
    [
        dict(device="cuda", batch_size=4, M=16, N=1024, K=1024),
        dict(device="cuda", batch_size=2, M=8, N=1024, K=1024),
    ],
)
def test_bmm_fp8_reference_correctness(shape_kwargs, with_bias):
    """flashinfer.bmm_fp8 kernel vs reference (per-tensor FP8 BMM).

    Matches tests/gemm/test_bmm_fp8.py: cos_sim > 0.99.
    """
    import flashinfer
    from flashinfer.trace.templates.gemm import (
        bmm_fp8_bias_trace,
        bmm_fp8_trace,
        bmm_fp8_trace_dispatch,
    )

    _skip_if_not_sm100_or_103()
    inputs = bmm_fp8_trace.init(**shape_kwargs)
    b_fp8_kmaj = inputs["B"].transpose(1, 2).contiguous().transpose(1, 2)
    # On the scale of the output, so a dropped bias fails the cosine check.
    bias = (
        torch.randn(shape_kwargs["N"], device="cuda", dtype=torch.bfloat16)
        * shape_kwargs["K"] ** 0.5
        if with_bias
        else None
    )
    try:
        api = flashinfer.bmm_fp8(
            inputs["A"],
            b_fp8_kmaj,
            inputs["A_scale"],
            inputs["B_scale"],
            dtype=torch.bfloat16,
        )
    except Exception as exc:
        pytest.skip(f"bmm_fp8 unavailable: {exc}")
    if with_bias:
        # Outside the skip above, so a failing bias path fails the test.
        api = flashinfer.bmm_fp8(
            inputs["A"],
            b_fp8_kmaj,
            inputs["A_scale"],
            inputs["B_scale"],
            dtype=torch.bfloat16,
            bias=bias,
        )
    # A biased call gets its own definition; an unbiased one keeps the original.
    template = bmm_fp8_trace_dispatch(bias=bias)
    assert template is (bmm_fp8_bias_trace if with_bias else bmm_fp8_trace)
    bias_kwargs = {"bias": bias} if with_bias else {}
    ref = template.reference(
        inputs["A"],
        inputs["B"],
        inputs["A_scale"],
        inputs["B_scale"],
        dtype=torch.bfloat16,
        **bias_kwargs,
    )
    _check(template, ref, api, max_mismatch_pct=100.0, min_cos_sim=0.99)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
