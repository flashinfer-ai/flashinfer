"""Unit tests for the cuTile backend of flashinfer.gemm.gemm_fp8_nt_groupwise.

The cuTile path lives in `flashinfer.cutile.fp8_gemm.gemm_fp8_nt_groupwise_cutile`
and is wired into `flashinfer.gemm.gemm_fp8_nt_groupwise` with `backend="cutile"`.
Companion to `test_groupwise_scaled_gemm_fp8.py`, scoped to the cuTile-only
quirks:

* Supports `scale_major_mode == "K"` only in v1
* Supports `scale_granularity_mnk == (1, 128, 128)` only in v1
* Requires SM >= 100 (Blackwell) and the cuda-tile python package
"""

import math

import pytest
import torch
from einops import einsum

from flashinfer.gemm import gemm_fp8_nt_groupwise
from flashinfer.testing.utils import dequantize_fp8, quantize_fp8
from flashinfer.utils import is_sm100a_supported


def _cutile_available() -> bool:
    try:
        import cuda.tile  # noqa: F401
    except Exception:
        return False
    return True


@pytest.mark.parametrize("m", [128, 256, 512])
@pytest.mark.parametrize("n", [2048, 4096, 7168])
@pytest.mark.parametrize("k", [2048, 7168])
def test_gemm_fp8_nt_groupwise_cutile(m, n, k):
    """cuTile FP8 W8A8 GEMM must agree with the fp32 dequant reference within atol/rtol = 1e-2."""
    if not _cutile_available():
        pytest.skip("cuda-tile not installed in this environment.")
    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("cuTile path requires SM >= 100")

    torch.random.manual_seed(0)
    tile_size = 128

    a_val = torch.randn((m, k), dtype=torch.float, device="cuda")
    b_val = torch.randn((n, k), dtype=torch.float, device="cuda") / math.sqrt(k)

    # K-major scale layout matches the cuTile v1 expectation.
    a_scale_shape = (m, k // tile_size)
    b_scale_shape = (n // tile_size, k // tile_size)
    a_tile_shape = (1, tile_size)
    b_tile_shape = (tile_size, tile_size)
    scale_major_mode = "K"

    a_fp8, a_scale = quantize_fp8(a_val, a_scale_shape, a_tile_shape, scale_major_mode)
    b_fp8, b_scale = quantize_fp8(b_val, b_scale_shape, b_tile_shape, scale_major_mode)

    a_dequant = dequantize_fp8(a_fp8, a_scale, scale_major_mode)
    b_dequant = dequantize_fp8(b_fp8, b_scale, scale_major_mode)
    ref_c = einsum(a_dequant, b_dequant, "m k, n k -> m n").to(torch.bfloat16)

    c = gemm_fp8_nt_groupwise(
        a=a_fp8,
        b=b_fp8,
        a_scale=a_scale,
        b_scale=b_scale,
        scale_major_mode=scale_major_mode,
        mma_sm=1,
        out_dtype=torch.bfloat16,
        backend="cutile",
    )
    torch.testing.assert_close(c, ref_c, atol=1e-2, rtol=1e-2)


def test_gemm_fp8_nt_groupwise_cutile_rejects_mn_scale_major():
    """The v1 cuTile path only supports K-major scales; MN-major must raise."""
    if not _cutile_available():
        pytest.skip("cuda-tile not installed in this environment.")
    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("cuTile path requires SM >= 100")

    torch.random.manual_seed(0)
    m, n, k = 128, 1024, 2048
    tile_size = 128

    a_val = torch.randn((m, k), dtype=torch.float, device="cuda")
    b_val = torch.randn((n, k), dtype=torch.float, device="cuda")

    a_scale_shape = (k // tile_size, m)
    b_scale_shape = (k // tile_size, n // tile_size)
    a_tile_shape = (1, tile_size)
    b_tile_shape = (tile_size, tile_size)

    a_fp8, a_scale = quantize_fp8(a_val, a_scale_shape, a_tile_shape, "MN")
    b_fp8, b_scale = quantize_fp8(b_val, b_scale_shape, b_tile_shape, "MN")

    # The @backend_requirement decorator raises ValueError before reaching the
    # cuTile module's own NotImplementedError.
    with pytest.raises(ValueError, match="scale_major_mode='K' only"):
        gemm_fp8_nt_groupwise(
            a=a_fp8,
            b=b_fp8,
            a_scale=a_scale,
            b_scale=b_scale,
            scale_major_mode="MN",
            mma_sm=1,
            out_dtype=torch.bfloat16,
            backend="cutile",
        )


if __name__ == "__main__":
    pytest.main([__file__])
