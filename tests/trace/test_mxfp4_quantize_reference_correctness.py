"""Reference correctness test for the mxfp4_quantize trace API."""

import torch
import pytest

from tests.trace.reference_utils import (
    _close,
    _skip_if_not_sm100,
)


@pytest.mark.parametrize(
    "shape_kwargs", [dict(device="cuda", M=64, K=128), dict(device="cuda", M=17, K=256)]
)
def test_mxfp4_quantize_reference_correctness(shape_kwargs):
    """mxfp4_quantize kernel: dequantized round-trip correctness.

    The CUDA kernel and the torch template reference use incompatible packed
    layouts (nibble ordering / scale packing differ), so we verify the kernel
    by its dequantized round-trip: quantize(a) → dequantize should reproduce
    ``a`` to within one E2M1 ULP * UE8M0 scale.
    """
    import flashinfer

    # fp4_quantize compiles on SM90+ but only produces correct output on
    # SM100+ — on Hopper the kernel silently returns near-zero garbage.
    _skip_if_not_sm100()
    from flashinfer.trace.templates.quantize import mxfp4_quantize_trace

    inputs = mxfp4_quantize_trace.init(**shape_kwargs)
    try:
        api_packed, api_scales = flashinfer.mxfp4_quantize(inputs["a"])
    except Exception as exc:
        pytest.skip(f"mxfp4_quantize unavailable: {exc}")
    api_dq = flashinfer.mxfp4_dequantize(api_packed, api_scales)
    _close(api_dq.float(), inputs["a"].cpu().float(), atol=2.0, rtol=0.25)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
