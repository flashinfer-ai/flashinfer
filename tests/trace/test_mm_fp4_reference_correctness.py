"""Reference correctness test for the mm_fp4 trace API."""

import pytest
import torch

from tests.trace.reference_utils import (
    _assert_finite,
    _check,
)


def _quantized_operands(m, n, k, fmt):
    """Quantize random A [m, k] and B [n, k] the way mm_fp4's callers do."""
    from flashinfer import SfLayout, mxfp4_quantize, nvfp4_quantize

    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    if fmt == "mxfp4":
        a_fp4, a_sf = mxfp4_quantize(a)
        b_fp4, b_sf = mxfp4_quantize(b)
        return a_fp4, b_fp4, a_sf, b_sf, None, 32
    g_a = (448 * 6) / a.float().abs().max()
    g_b = (448 * 6) / b.float().abs().max()
    a_fp4, a_sf = nvfp4_quantize(
        a, g_a, sfLayout=SfLayout.layout_128x4, do_shuffle=False
    )
    b_fp4, b_sf = nvfp4_quantize(
        b, g_b, sfLayout=SfLayout.layout_128x4, do_shuffle=False
    )
    return a_fp4, b_fp4, a_sf, b_sf, 1.0 / (g_a * g_b), 16


@pytest.mark.parametrize("fmt", ["nvfp4", "mxfp4"])
@pytest.mark.parametrize("m,n,k", [(1, 256, 512), (130, 640, 2048), (130, 256, 544)])
def test_mm_fp4_reference_correctness(fmt, m, n, k):
    """flashinfer.mm_fp4 kernel vs reference (E2M1 decode, 128x4 unswizzle, matmul).

    The inputs come from the quantizers. So the reference is checked against
    the scale layout they write, not a layout built in the test.
    m = 130 pads the scale rows to 256, and k = 544 pads the scale columns
    (to 36 for nvfp4, 20 for mxfp4).
    """
    import flashinfer
    from flashinfer.trace.templates.gemm import mm_fp4_trace
    from flashinfer.utils import BackendSupportedError, LibraryError

    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("mm_fp4 requires SM100 or newer")
    torch.manual_seed(0)
    a_fp4, b_fp4, a_sf, b_sf, alpha, block_size = _quantized_operands(m, n, k, fmt)
    # Only a missing backend or library is a skip; any other error fails the test.
    try:
        api = flashinfer.mm_fp4(
            a_fp4,
            b_fp4.T,
            a_sf,
            b_sf.T,
            alpha,
            torch.bfloat16,
            block_size=block_size,
            use_nvfp4=fmt == "nvfp4",
        )
    except (BackendSupportedError, LibraryError) as exc:
        pytest.skip(f"mm_fp4 ({fmt}) unavailable: {exc}")
    ref = mm_fp4_trace.reference(
        a_fp4, b_fp4.T, a_sf, b_sf.T, alpha=alpha, block_size=block_size
    ).float()
    _assert_finite(api, ref)
    # Every element must be within tolerance. With atol/rtol given, _check
    # drops the cosine test, and by default the FP4 check allows any share of
    # mismatches.
    _check(
        mm_fp4_trace,
        ref.to(api.dtype),
        api,
        atol=2e-2,
        rtol=2e-2,
        max_mismatch_pct=0.0,
    )
    torch.cuda.synchronize()
