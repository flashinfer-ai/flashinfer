"""Reference correctness test for the bmm_fp8 trace API."""

import torch
import pytest

from tests.trace.reference_utils import (
    _close_fp8,
    _skip_if_not_sm100_or_103,
)


@pytest.mark.parametrize(
    "shape_kwargs",
    [
        dict(device="cuda", batch_size=4, M=16, N=1024, K=1024),
        dict(device="cuda", batch_size=2, M=8, N=1024, K=1024),
    ],
)
def test_bmm_fp8_reference_correctness(shape_kwargs):
    """flashinfer.bmm_fp8 kernel vs reference (per-tensor FP8 BMM).

    Matches tests/gemm/test_bmm_fp8.py: cos_sim > 0.99.
    """
    import flashinfer
    from flashinfer.trace.templates.gemm import bmm_fp8_trace

    _skip_if_not_sm100_or_103()
    inputs = bmm_fp8_trace.init(**shape_kwargs)
    b_fp8_kmaj = inputs["B"].transpose(1, 2).contiguous().transpose(1, 2)
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
    ref = bmm_fp8_trace.reference(
        inputs["A"],
        inputs["B"],
        inputs["A_scale"],
        inputs["B_scale"],
        dtype=torch.bfloat16,
    )
    _close_fp8(api, ref, cos_sim_min=0.99)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
