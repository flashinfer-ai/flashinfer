"""Tests for SM120 MXFP8 GEMM (issue #2728)."""

import pytest
import torch
import torch.nn.functional as F

from flashinfer import mm_mxfp8, SfLayout
from flashinfer.fp8_quantization import mxfp8_quantize
from flashinfer.utils import get_compute_capability


def _is_sm120_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        cc = get_compute_capability(torch.device("cuda"))
        return cc[0] == 12
    except RuntimeError:
        return False


def _skip_if_not_sm120():
    if not _is_sm120_available():
        pytest.skip("Requires SM12x GPU")


def _prepare_mxfp8(a_bf16, b_bf16, swizzled: bool):
    sflayout = SfLayout.layout_128x4 if swizzled else SfLayout.layout_linear
    a_fp8, a_sf = mxfp8_quantize(a_bf16, sf_swizzle_layout=sflayout)
    b_fp8, b_sf = mxfp8_quantize(b_bf16, sf_swizzle_layout=sflayout)
    if not swizzled:
        m, k = a_bf16.shape
        n = b_bf16.shape[0]
        a_sf = a_sf.view(m, k // 32)
        b_sf = b_sf.view(n, k // 32).t()
    return a_fp8, b_fp8, a_sf, b_sf


# Swizzled (layout_128x4) scale: mxfp8_quantize pads the scale buffer to pad_up(M, 128)
# rows internally, so arbitrary M is supported.
@pytest.mark.parametrize("m", [1, 17, 100, 128, 256, 512, 1024])
@pytest.mark.parametrize("n", [128, 256, 512, 1024])
@pytest.mark.parametrize("k", [128, 256, 512, 1024])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16])
def test_mm_mxfp8_sm120_swizzled(m, n, k, out_dtype):
    _skip_if_not_sm120()

    a = torch.randn([m, k], device="cuda", dtype=torch.bfloat16)
    b = torch.randn([n, k], device="cuda", dtype=torch.bfloat16)

    a_fp8, b_fp8, a_sf, b_sf = _prepare_mxfp8(a, b, swizzled=True)
    reference = torch.mm(a, b.T)

    result = mm_mxfp8(
        a_fp8, b_fp8.T, a_sf, b_sf, out_dtype=out_dtype, backend="cutlass"
    )

    assert result.shape == (m, n)
    assert result.dtype == out_dtype
    assert torch.isfinite(result).all(), "Output contains NaN/Inf"

    cos_sim = F.cosine_similarity(
        reference.reshape(-1).float(), result.reshape(-1).float(), dim=0
    ).item()
    assert cos_sim > 0.99, f"cos_sim={cos_sim:.4f} < 0.99 for M={m},N={n},K={k}"


def test_mm_mxfp8_sm120_tactic_num():
    """Verify tactic count for SM120 MXFP8 GEMM."""
    _skip_if_not_sm120()
    from flashinfer.jit.gemm import gen_gemm_sm120_module_cutlass_mxfp8

    module = gen_gemm_sm120_module_cutlass_mxfp8().build_and_load()
    num_tactics = module.mxfp8_gemm_tactic_num()
    # SM120 has 5 tile configs (128x32x128, 128x64x128, 128x128x128, 256x128x128, 128x256x128)
    # and each config can swap AB to compute Output^T = Weight^T Activations^T
    assert num_tactics == 10, f"Expected 10 tactics, got {num_tactics}"


def test_mm_mxfp8_sm120_auto_tactic():
    """Verify SM120 MXFP8 produces correct results (tactic auto-selected)."""
    _skip_if_not_sm120()
    from flashinfer.jit.gemm import gen_gemm_sm120_module_cutlass_mxfp8

    m, n, k = 256, 256, 256
    a = torch.randn([m, k], device="cuda", dtype=torch.bfloat16)
    b = torch.randn([n, k], device="cuda", dtype=torch.bfloat16)
    a_fp8, b_fp8, a_sf, b_sf = _prepare_mxfp8(a, b, swizzled=True)
    reference = torch.mm(a, b.T)

    module = gen_gemm_sm120_module_cutlass_mxfp8().build_and_load()
    num_tactics = module.mxfp8_gemm_tactic_num()
    assert num_tactics > 0

    result = mm_mxfp8(
        a_fp8,
        b_fp8.T,
        a_sf,
        b_sf,
        out_dtype=torch.bfloat16,
        backend="cutlass",
    )
    assert result.shape == (m, n)
    assert torch.isfinite(result).all()
    cos_sim = F.cosine_similarity(
        reference.reshape(-1).float(), result.reshape(-1).float(), dim=0
    ).item()
    assert cos_sim > 0.98, f"cos_sim={cos_sim:.4f}"


def test_mm_mxfp8_sm120_rejects_linear_scales():
    """SM120 CUTLASS MXFP8 must reject non-swizzled (2D) scale tensors."""
    _skip_if_not_sm120()

    m, n, k = 128, 128, 128
    a = torch.randn([m, k], device="cuda", dtype=torch.bfloat16)
    b = torch.randn([n, k], device="cuda", dtype=torch.bfloat16)
    a_fp8, b_fp8, a_sf, b_sf = _prepare_mxfp8(a, b, swizzled=False)

    with pytest.raises((RuntimeError, ValueError)):
        mm_mxfp8(
            a_fp8, b_fp8.T, a_sf, b_sf, out_dtype=torch.bfloat16, backend="cutlass"
        )
