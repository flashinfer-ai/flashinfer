"""
Tests for flashinfer.fused_rmsnorm_silu() — fused RMSNorm + SiLU activation.

Requires SM80+ GPU and nvidia-cudnn-frontend.

The kernel is tuned/optimized for WAN VAE problem sizes on B200 (SM100).
Other SM80+ GPUs use a conservative fallback heuristic.

Coverage:
  - bf16 output:  8 C × 5 token counts = 40 configs
  - FP8 output:   8 C × 5 token counts = 40 configs (SM89+)
  - NVFP4 output: 8 C × 5 token counts = 40 configs (SM100+ only)
  - Pre-allocated output tensor
  - Output shape/dtype/NaN/Inf sanity
  - L2Norm ↔ RMSNorm epsilon equivalence
"""

import pytest
import torch
import torch.nn.functional as F

import flashinfer


# ============================================================
# Reference & helper functions
# ============================================================


def rmsnorm_silu_reference(x, weight, eps, output_dtype=None):
    """PyTorch reference: RMSNorm + SiLU in float32.

    Args:
        output_dtype: if None, returns in x.dtype (bf16). Pass torch.float32
                      for a float32 result (avoids bf16 intermediate rounding,
                      important for FP8/NVFP4 reference accuracy).
    """
    rms = torch.sqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + eps)
    normed = (x.float() / rms) * weight.float()
    result = F.silu(normed)
    if output_dtype is not None:
        return result.to(output_dtype)
    return result.to(x.dtype)


# FP4 E2M1 lookup table (nibble index → float value)
# Encoding: 1 sign bit, 2 exponent bits, 1 mantissa bit
_FP4_E2M1_TABLE = [
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,  # positive (nibbles 0-7)
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,  # negative (nibbles 8-15)
]


def _unpack_fp4_nibbles(packed_bytes, num_tokens, C):
    """Unpack FP4 packed bytes into a (num_tokens, C) int tensor of nibble values.

    Vectorized on GPU — no CPU data transfers.
    """
    packed = packed_bytes.view(num_tokens, C // 2)
    nibbles = torch.empty(num_tokens, C, dtype=torch.int32, device=packed_bytes.device)
    nibbles[:, 0::2] = packed.int() & 0x0F  # lo nibble → even columns
    nibbles[:, 1::2] = (packed.int() >> 4) & 0x0F  # hi nibble → odd columns
    return nibbles


def _quantize_to_fp4_reference(values_f32, C):
    """Quantize float32 values to FP4 E2M1 nibbles using the same algorithm as the kernel.

    Matches the CuDNN BlockScaleRowHelper quantization:
      1. amax = max(|block of 16 elements|)
      2. scale = max(amax / 6.0, FLT_MIN)       — float32, NOT rounded to FP8
      3. quantized = round_to_nearest_fp4(value / scale)

    Quantizes by magnitude first, then applies the sign bit.  This matches
    the hardware's nv_fp4x2_e2m1 which preserves sign.
    """
    BLOCK_SIZE = 16
    FP4_MAX = 6.0
    FLT_MIN = 1.17549435082228750796873653722224568e-38
    num_tokens = values_f32.shape[0]
    num_blocks = C // BLOCK_SIZE
    device = values_f32.device

    fp4_positive = torch.tensor(_FP4_E2M1_TABLE[:8], dtype=torch.float32)
    nibbles = torch.zeros(num_tokens, C, dtype=torch.int32, device=device)

    for b in range(num_blocks):
        col_start = b * BLOCK_SIZE
        col_end = col_start + BLOCK_SIZE
        block_vals = values_f32[:, col_start:col_end].cpu().float()

        # Step 1-2: compute scale from amax (same as kernel, in float32)
        amax = block_vals.abs().max(dim=1, keepdim=True).values
        scale = torch.clamp(amax / FP4_MAX, min=FLT_MIN)

        # Step 3: quantize by magnitude, then apply sign
        scaled = block_vals / scale
        magnitudes = scaled.abs()
        signs = (scaled < 0).int()

        # Find closest positive FP4 value (nibbles 0-7) by magnitude
        diffs = (magnitudes.unsqueeze(2) - fp4_positive.unsqueeze(0).unsqueeze(0)).abs()
        mag_nibbles = diffs.argmin(dim=2)

        # Apply sign: negative values get +8 (bit 3 = sign bit)
        block_nibbles = mag_nibbles + signs * 8
        nibbles[:, col_start:col_end] = block_nibbles.to(device)

    return nibbles


def _get_sm():
    """Return compute capability as major*10+minor (e.g. 80, 89, 90, 100)."""
    if not torch.cuda.is_available():
        return 0
    major, minor = torch.cuda.get_device_capability(0)
    return major * 10 + minor


# ============================================================
# Problem sizes — full WAN VAE sweep
# ============================================================

C_VALUES = [64, 128, 160, 256, 320, 512, 640, 1024]
TOKEN_VALUES = [1560, 6240, 24960, 99840, 399360]


# ============================================================
# Test 1: bf16 output (40 configs)
# ============================================================


@pytest.mark.parametrize("C", C_VALUES)
@pytest.mark.parametrize("num_tokens", TOKEN_VALUES)
def test_fused_rmsnorm_silu_bf16_output(C, num_tokens):
    """Test bf16 RMSNorm+SiLU output against PyTorch reference.

    Tolerance: atol=2e-2, rtol=2e-2 (bf16 has ~8-bit mantissa).
    Requires 0 mismatches.
    """
    torch.manual_seed(42)

    x = torch.randn(num_tokens, C, dtype=torch.bfloat16, device="cuda") * 5.0 + 5.0
    w = torch.rand(C, dtype=torch.bfloat16, device="cuda") * 1.5 + 0.5
    eps = 1e-6

    out = flashinfer.fused_rmsnorm_silu(x, w, eps)
    ref = rmsnorm_silu_reference(x, w, eps)

    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)


# ============================================================
# Test 2: FP8 E4M3 output (40 configs)
# ============================================================


@pytest.mark.parametrize("C", C_VALUES)
@pytest.mark.parametrize("num_tokens", TOKEN_VALUES)
def test_fused_rmsnorm_silu_fp8_output(C, num_tokens):
    """Test FP8 E4M3 output against float32 reference.

    Reference is computed in float32 (no bf16 intermediate) then quantized
    to FP8, matching the kernel's float32 → FP8 path.  This avoids
    double-rounding that would artificially inflate max_diff.

    Tolerance: atol=0.125, rtol=0.125 (~1 FP8 ULP).
    Requires 0 mismatches.
    """
    torch.manual_seed(42)

    x = torch.randn(num_tokens, C, dtype=torch.bfloat16, device="cuda") * 5.0 + 5.0
    w = torch.rand(C, dtype=torch.bfloat16, device="cuda") * 1.5 + 0.5

    out_fp8 = torch.empty(num_tokens, C, dtype=torch.float8_e4m3fn, device="cuda")
    result = flashinfer.fused_rmsnorm_silu(x, w, 1e-6, out=out_fp8)

    assert result is out_fp8
    assert result.dtype == torch.float8_e4m3fn
    assert not torch.isnan(result.float()).any()

    # Reference in float32 directly (skip bf16 intermediate to match kernel path)
    ref_f32 = rmsnorm_silu_reference(x, w, 1e-6, output_dtype=torch.float32)
    ref_fp8 = ref_f32.clamp(-448.0, 448.0).to(torch.float8_e4m3fn)

    mismatches = ~torch.isclose(result.float(), ref_fp8.float(), atol=0.125, rtol=0.125)
    assert mismatches.sum().item() == 0, (
        f"FP8 C={C}, tokens={num_tokens}: "
        f"{mismatches.sum().item()}/{result.numel()} mismatches "
        f"(max_diff={(result.float() - ref_fp8.float()).abs().max().item():.6e})"
    )


# ============================================================
# Test 3: NVFP4 E2M1 output (40 configs, SM100+ only)
# ============================================================


@pytest.mark.parametrize("C", C_VALUES)
@pytest.mark.parametrize("num_tokens", TOKEN_VALUES)
def test_fused_rmsnorm_silu_nvfp4_output(C, num_tokens):
    """Test NVFP4 E2M1 output via nibble-level comparison.

    Calls the public ``fused_rmsnorm_silu`` API with a ``torch.uint8`` output
    tensor, then compares FP4 nibble indices directly: quantize the float32
    reference using the same block-scale algorithm as the kernel, then verify
    every nibble matches within ±1 (one FP4 ULP).  This isolates kernel
    correctness from inherent FP4 quantization error.

    Requires 0 mismatches (nibble_diff > 1).
    """
    if _get_sm() < 100:
        pytest.skip("NVFP4 output requires SM100+ (Blackwell)")

    torch.manual_seed(42)
    x = torch.randn(num_tokens, C, dtype=torch.bfloat16, device="cuda") * 5.0 + 5.0
    w = torch.rand(C, dtype=torch.bfloat16, device="cuda") * 1.5 + 0.5
    eps = 1e-6

    out_nvfp4 = torch.empty(num_tokens * C // 2, dtype=torch.uint8, device="cuda")
    result = flashinfer.fused_rmsnorm_silu(x, w, eps, out=out_nvfp4)

    # Basic shape / dtype checks
    assert result is out_nvfp4
    assert out_nvfp4.dtype == torch.uint8
    assert out_nvfp4.shape == (num_tokens * C // 2,)
    assert out_nvfp4.any(), "NVFP4 output is all zeros — kernel may not have executed"

    # ---- Nibble-level comparison ----
    ref_f32 = rmsnorm_silu_reference(x, w, eps, output_dtype=torch.float32)

    z_packed = out_nvfp4.view(num_tokens, C // 2)
    kernel_nibbles = _unpack_fp4_nibbles(z_packed, num_tokens, C)
    ref_nibbles = _quantize_to_fp4_reference(ref_f32, C)

    nibble_diff = (kernel_nibbles - ref_nibbles).abs()
    mismatches = nibble_diff > 1
    num_mismatches = mismatches.sum().item()
    max_nibble_diff = nibble_diff.max().item()

    assert num_mismatches == 0, (
        f"NVFP4 C={C}, tokens={num_tokens}: "
        f"{num_mismatches}/{num_tokens * C} nibbles differ by >1 ULP "
        f"(max_nibble_diff={max_nibble_diff})"
    )


# ============================================================
# Test 4: Pre-allocated output tensor
# ============================================================


@pytest.mark.parametrize("C", [64, 256, 512, 1024])
def test_fused_rmsnorm_silu_preallocated_out(C):
    """Test that passing a pre-allocated out tensor writes in-place."""
    torch.manual_seed(42)
    num_tokens = 1560

    x = torch.randn(num_tokens, C, dtype=torch.bfloat16, device="cuda") * 5.0 + 5.0
    w = torch.rand(C, dtype=torch.bfloat16, device="cuda") * 1.5 + 0.5
    out = torch.empty_like(x)

    result = flashinfer.fused_rmsnorm_silu(x, w, 1e-6, out=out)
    ref = rmsnorm_silu_reference(x, w, 1e-6)

    assert result is out, "Should return the same tensor when out is provided"
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)


# ============================================================
# Test 5: Output properties (shape, dtype, no NaN/Inf)
# ============================================================


def test_fused_rmsnorm_silu_output_properties():
    """Test basic output properties: shape, dtype, no NaN/Inf."""
    x = torch.randn(1560, 512, dtype=torch.bfloat16, device="cuda") * 5.0 + 5.0
    w = torch.ones(512, dtype=torch.bfloat16, device="cuda")

    out = flashinfer.fused_rmsnorm_silu(x, w, 1e-6)

    assert out.shape == x.shape
    assert out.dtype == x.dtype
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


# ============================================================
# Test 6: L2Norm ↔ RMSNorm epsilon equivalence
# ============================================================


def test_fused_rmsnorm_silu_l2norm_equivalence():
    """Verify L2Norm(eps) ≡ RMSNorm(eps/C) with appropriate gamma.

    The WAN VAE uses L2 normalization (F.normalize), which is equivalent to
    RMSNorm with adjusted epsilon:

        RMSNorm(x, eps/C) = x / sqrt(mean(x²) + eps/C)
                           = x * sqrt(C) / sqrt(sum(x²) + eps)
                           = sqrt(C) * L2Norm(x, eps)

    So: SiLU(L2Norm(x) * sqrt(C) * w) = SiLU(RMSNorm(x, eps/C) * w)

    The sqrt(C) factor is absorbed by the RMSNorm ↔ L2Norm conversion,
    so the gamma passed to the kernel is just ``w`` (not ``w * sqrt(C)``).
    """
    torch.manual_seed(42)
    C = 512
    num_tokens = 1560
    eps = 1e-6

    x = torch.randn(num_tokens, C, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(C, dtype=torch.bfloat16, device="cuda")

    # L2Norm + SiLU:  SiLU( x / ||x|| * sqrt(C) * weight )
    l2norm = torch.sqrt(torch.sum(x.float() ** 2, dim=-1, keepdim=True) + eps)
    scale = C**0.5
    l2_out = F.silu((x.float() / l2norm) * scale * weight.float()).to(x.dtype)

    # RMSNorm(eps/C) + SiLU:  SiLU( RMSNorm(x, eps/C) * weight )
    # RMSNorm(x, eps/C) already includes the sqrt(C) factor, so gamma = weight.
    rms_out = flashinfer.fused_rmsnorm_silu(x, weight, eps / C)

    # The cudnn_frontend epsilon equivalence test uses atol=1e-2, rtol=1e-2,
    # but that test compares two PyTorch-only computations. Here we compare a
    # PyTorch reference against the actual kernel, which introduces additional
    # float32 → bf16 rounding, so we need the slightly wider bf16 tolerance.
    torch.testing.assert_close(l2_out, rms_out, atol=2e-2, rtol=2e-2)
