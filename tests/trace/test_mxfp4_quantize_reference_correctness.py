"""Reference correctness test for the mxfp4_quantize trace API."""

import torch
import pytest

from tests.test_helpers.utils_fp4 import pack_fp4_values, ref_fp4_quant
from tests.trace.reference_utils import _skip_if_not_sm100


@pytest.mark.parametrize(
    "shape_kwargs", [dict(device="cuda", M=64, K=128), dict(device="cuda", M=17, K=256)]
)
def test_mxfp4_quantize_reference_correctness(shape_kwargs):
    """Check MXFP4 payload, logical scale bytes, and decoding exactly."""
    import flashinfer

    # fp4_quantize compiles on SM90+ but only produces correct output on
    # SM100+ — on Hopper the kernel silently returns near-zero garbage.
    _skip_if_not_sm100()
    from flashinfer.trace.templates.quantize import mxfp4_quantize_trace

    inputs = mxfp4_quantize_trace.init(**shape_kwargs)
    api_packed, api_scales = flashinfer.mxfp4_quantize(inputs["a"])
    api_dq = flashinfer.mxfp4_dequantize(api_packed, api_scales)
    q_ref, sf_ref = ref_fp4_quant(inputs["a"], 1.0, 32, sf_use_ue8m0=True)
    torch.testing.assert_close(api_packed, pack_fp4_values(q_ref), rtol=0, atol=0)
    m, k = inputs["a"].shape
    rows = torch.arange(m, device=api_scales.device).unsqueeze(1)
    cols = torch.arange(k // 32, device=api_scales.device).unsqueeze(0)
    padded_cols = ((k // 32 + 3) // 4) * 4
    offsets = (
        (rows // 128) * (128 * padded_cols)
        + (cols // 4) * 512
        + (rows % 32) * 16
        + ((rows % 128) // 32) * 4
        + cols % 4
    )
    torch.testing.assert_close(
        api_scales.reshape(-1)[offsets],
        (sf_ref.view(torch.int32) >> 23).to(torch.uint8),
        rtol=0,
        atol=0,
    )
    dq_ref = (q_ref.reshape(m, k // 32, 32) * sf_ref.unsqueeze(-1)).reshape(m, k)
    torch.testing.assert_close(
        api_dq.float().view(torch.int32),
        dq_ref.cpu().view(torch.int32),
        rtol=0,
        atol=0,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
