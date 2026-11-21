"""Reference correctness test for the gemv_bf16_fp4_gather trace API."""

import pytest
import torch

from tests.trace.reference_utils import _cc, _check


@pytest.mark.parametrize("M", [1, 5])
def test_gemv_bf16_fp4_gather_reference_correctness(M):
    if _cc()[0] != 12:
        pytest.skip("gemv_bf16_fp4_gather requires SM120/121")
    from flashinfer import gemv_bf16_fp4_gather
    from flashinfer.trace.templates.gemm import gemv_bf16_fp4_gather_trace as tpl

    inputs = tpl.init(M=M, J=1024, K=2048, R=8192)
    ref_out = inputs["out"].clone()
    api_out = gemv_bf16_fp4_gather(**inputs)
    torch.cuda.synchronize()
    tpl.reference(**{**inputs, "out": ref_out})
    _check(tpl, ref_out, api_out)
