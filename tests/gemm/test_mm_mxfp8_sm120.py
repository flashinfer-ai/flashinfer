"""Tests for SM120/SM121 MXFP8 GEMM."""

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


def test_probe_mxfp8_gemm_tactics_sm12x():
    """_probe_mxfp8_gemm_tactics returns only device-compatible tactics.

    SM120 (B100/B200, ~256 KB smem/SM) should support all 10 tactics.
    SM121 (GB10/DGX Spark, ~99 KB smem/SM opt-in) should support only
    tactics 0-5 (small CTA tiles); tactics 6-9 (256x128, 128x256) need
    ~99 KB with StageCount<2> and fail on SM121.
    """
    _skip_if_not_sm120()

    from flashinfer.gemm.gemm_base import _probe_mxfp8_gemm_tactics
    from flashinfer.jit.gemm import gen_gemm_sm120_module_cutlass_mxfp8

    raw_module = gen_gemm_sm120_module_cutlass_mxfp8().build_and_load()
    valid = _probe_mxfp8_gemm_tactics(raw_module, torch.cuda.current_device())

    # At least the three small-tile configs (128x32, 128x64, 128x128) x 2 variants
    # must be supported on any SM12x device.
    assert len(valid) >= 6, f"Expected at least 6 valid tactics, got {len(valid)}: {valid}"

    # All returned tactics must be within range.
    total = raw_module.mxfp8_gemm_tactic_num()
    assert all(0 <= t < total for t in valid), f"Out-of-range tactic in {valid}"

    # Device-specific expectations.
    cc = get_compute_capability(torch.device("cuda"))
    if cc == (12, 1):
        # SM121: large tiles (tactics 6-9) should not be valid.
        assert 6 not in valid, "Tactic 6 (256x128 non-swapped) should not work on SM121"
        assert 7 not in valid, "Tactic 7 (256x128 swapped) should not work on SM121"
        assert 8 not in valid, "Tactic 8 (128x256 non-swapped) should not work on SM121"
        assert 9 not in valid, "Tactic 9 (128x256 swapped) should not work on SM121"
        assert valid == list(range(6)), f"SM121 expected tactics [0..5], got {valid}"
    elif cc == (12, 0):
        # SM120 data-centre: all 10 tactics should be valid.
        assert valid == list(range(10)), f"SM120 expected all 10 tactics, got {valid}"
