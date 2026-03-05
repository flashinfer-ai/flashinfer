"""
Tests for flashinfer.fused_rmsnorm_silu() — fused RMSNorm + SiLU activation.

On SM100+ (Blackwell): exercises the cuDNN OSS engine path.
On other GPUs: exercises the PyTorch fallback path.
"""

import pytest
import torch
import torch.nn.functional as F

import flashinfer


def rmsnorm_silu_reference(x, weight, eps, output_dtype=None):
    """PyTorch reference: RMSNorm + SiLU in float32."""
    rms = torch.sqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + eps)
    normed = (x.float() / rms) * weight.float()
    result = F.silu(normed)
    if output_dtype is not None:
        return result.to(output_dtype)
    return result.to(x.dtype)


# FP4 E2M1 lookup table for dequantization
_FP4_E2M1_TABLE = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
]


def dequantize_nvfp4_block(packed_bytes, scale_row_fp8, num_tokens, C):
    """Dequantize NVFP4 packed output using FP8 block scales from workspace."""
    BLOCK_SIZE = 16
    scale_f32 = scale_row_fp8.view(torch.float8_e4m3fn).float()
    output = torch.zeros(num_tokens, C, dtype=torch.float32, device=packed_bytes.device)

    for col_byte in range(C // 2):
        even_col = col_byte * 2
        odd_col = col_byte * 2 + 1
        byte_val = packed_bytes[:, col_byte].int()
        lo_nibble = byte_val & 0x0F
        hi_nibble = (byte_val >> 4) & 0x0F

        lo_float = torch.tensor([_FP4_E2M1_TABLE[v] for v in lo_nibble.cpu().tolist()],
                                dtype=torch.float32, device=packed_bytes.device)
        hi_float = torch.tensor([_FP4_E2M1_TABLE[v] for v in hi_nibble.cpu().tolist()],
                                dtype=torch.float32, device=packed_bytes.device)

        even_block = even_col // BLOCK_SIZE
        odd_block = odd_col // BLOCK_SIZE
        output[:, even_col] = lo_float * scale_f32[:, even_block]
        output[:, odd_col] = hi_float * scale_f32[:, odd_block]

    return output


def _is_sm100_plus():
    if not torch.cuda.is_available():
        return False
    prop = torch.cuda.get_device_properties(0)
    return prop.major >= 10


# VAE problem sizes
C_VALUES = [64, 128, 160, 256, 320, 512, 640, 1024]
TOKEN_VALUES_SMALL = [1560, 6240, 24960, 99840, 399360]


@pytest.mark.parametrize("C", C_VALUES)
@pytest.mark.parametrize("num_tokens", TOKEN_VALUES_SMALL)
def test_fused_rmsnorm_silu_correctness(C, num_tokens):
    """Test fused RMSNorm+SiLU against PyTorch reference."""
    torch.manual_seed(42)

    x = (torch.randn(num_tokens, C, dtype=torch.bfloat16, device="cuda") * 5.0 + 5.0)
    w = (torch.rand(C, dtype=torch.bfloat16, device="cuda") * 1.5 + 0.5)
    eps = 1e-6

    out = flashinfer.fused_rmsnorm_silu(x, w, eps)
    ref = rmsnorm_silu_reference(x, w, eps)

    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("C", [64, 256, 512, 1024])
def test_fused_rmsnorm_silu_preallocated_out(C):
    """Test with pre-allocated output tensor."""
    torch.manual_seed(42)
    num_tokens = 1560

    x = torch.randn(num_tokens, C, dtype=torch.bfloat16, device="cuda") * 5.0 + 5.0
    w = torch.rand(C, dtype=torch.bfloat16, device="cuda") * 1.5 + 0.5
    out = torch.empty_like(x)

    result = flashinfer.fused_rmsnorm_silu(x, w, 1e-6, out=out)
    ref = rmsnorm_silu_reference(x, w, 1e-6)

    assert result is out, "Should return the same tensor when out is provided"
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)


def test_fused_rmsnorm_silu_output_properties():
    """Test basic output properties: shape, dtype, no NaN/Inf."""
    x = torch.randn(1560, 512, dtype=torch.bfloat16, device="cuda") * 5.0 + 5.0
    w = torch.ones(512, dtype=torch.bfloat16, device="cuda")

    out = flashinfer.fused_rmsnorm_silu(x, w, 1e-6)

    assert out.shape == x.shape
    assert out.dtype == x.dtype
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


@pytest.mark.parametrize("C", [64, 256, 512, 1024])
@pytest.mark.parametrize("num_tokens", [1560, 6240])
def test_fused_rmsnorm_silu_fp8_output(C, num_tokens):
    """Test FP8 E4M3 output: pass out=float8_e4m3fn tensor."""
    torch.manual_seed(42)

    x = (torch.randn(num_tokens, C, dtype=torch.bfloat16, device="cuda") * 5.0 + 5.0)
    w = (torch.rand(C, dtype=torch.bfloat16, device="cuda") * 1.5 + 0.5)

    # FP8 output
    out_fp8 = torch.empty(num_tokens, C, dtype=torch.float8_e4m3fn, device="cuda")
    result = flashinfer.fused_rmsnorm_silu(x, w, 1e-6, out=out_fp8)

    assert result is out_fp8
    assert result.dtype == torch.float8_e4m3fn
    assert not torch.isnan(result.float()).any()

    # Compare against bf16 reference quantized to FP8
    ref_bf16 = rmsnorm_silu_reference(x, w, 1e-6)
    ref_fp8 = ref_bf16.float().clamp(-448.0, 448.0).to(torch.float8_e4m3fn)

    # FP8 has 3-bit mantissa — allow 1 ULP tolerance
    torch.testing.assert_close(
        result.float(), ref_fp8.float(), atol=0.125, rtol=0.125
    )


@pytest.mark.parametrize("C", [64, 256, 512])
@pytest.mark.parametrize("num_tokens", [1560, 6240])
def test_fused_rmsnorm_silu_nvfp4_output(C, num_tokens):
    """Test NVFP4 E2M1 output: pass out=uint8 tensor (packed FP4)."""
    torch.manual_seed(42)

    x = (torch.randn(num_tokens, C, dtype=torch.bfloat16, device="cuda") * 5.0 + 5.0)
    w = (torch.rand(C, dtype=torch.bfloat16, device="cuda") * 1.5 + 0.5)

    # NVFP4 output: packed, 2 elements per byte
    out_nvfp4 = torch.empty(num_tokens * C // 2, dtype=torch.uint8, device="cuda")
    result = flashinfer.fused_rmsnorm_silu(x, w, 1e-6, out=out_nvfp4)

    assert result is out_nvfp4
    assert result.dtype == torch.uint8
    assert result.shape == (num_tokens * C // 2,)

    # Read scale_row from workspace to dequantize
    # The engine auto-allocates scale_row in workspace; for a basic sanity check,
    # verify the output is not all zeros (kernel executed and wrote data)
    assert result.any(), "NVFP4 output is all zeros — kernel may not have executed"


def test_fused_rmsnorm_silu_l2norm_equivalence():
    """Verify L2Norm(eps) ≡ RMSNorm(eps/C) with weight adjustment.

    The WAN VAE uses L2Norm (F.normalize), which is equivalent to
    RMSNorm with eps_cudnn = eps_l2 / C.
    """
    torch.manual_seed(42)
    C = 512
    num_tokens = 1560
    eps = 1e-6

    x = torch.randn(num_tokens, C, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(C, dtype=torch.bfloat16, device="cuda")

    # L2Norm + SiLU
    l2norm = torch.sqrt(torch.sum(x.float() ** 2, dim=-1, keepdim=True) + eps)
    scale = C ** 0.5
    l2_out = F.silu((x.float() / l2norm) * scale * weight.float()).to(x.dtype)

    # RMSNorm(eps/C) + SiLU (what cuDNN computes)
    rms_out = flashinfer.fused_rmsnorm_silu(x, weight * scale, eps / C)

    torch.testing.assert_close(l2_out, rms_out, atol=2e-2, rtol=2e-2)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available, skipping")
        exit(0)

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"SM100+: {_is_sm100_plus()}")
    print()

    passed = 0
    failed = 0
    for C in C_VALUES:
        for tokens in TOKEN_VALUES_SMALL:
            try:
                torch.manual_seed(42)
                x = (torch.randn(tokens, C, dtype=torch.bfloat16, device="cuda") * 5.0 + 5.0)
                w = (torch.rand(C, dtype=torch.bfloat16, device="cuda") * 1.5 + 0.5)
                out = flashinfer.fused_rmsnorm_silu(x, w, 1e-6)
                ref = rmsnorm_silu_reference(x, w, 1e-6)
                max_diff = (out.float() - ref.float()).abs().max().item()
                mismatches = (~torch.isclose(out.float(), ref.float(), atol=2e-2, rtol=2e-2)).sum().item()
                status = "PASS" if mismatches == 0 else "FAIL"
                if mismatches == 0:
                    passed += 1
                else:
                    failed += 1
                print(f"  C={C:>4}, tokens={tokens:>6}: {status}  "
                      f"max_diff={max_diff:.3e}  mismatches={mismatches}/{out.numel()}")
            except Exception as e:
                failed += 1
                print(f"  C={C:>4}, tokens={tokens:>6}: ERROR  {e}")

    print(f"\n--- bf16 Results: {passed} passed, {failed} failed ---")

    # FP8 sweep
    print("\n--- FP8 Output ---")
    fp8_passed = 0
    fp8_failed = 0
    for C in [64, 256, 512, 1024]:
        for tokens in [1560, 6240]:
            try:
                torch.manual_seed(42)
                x = (torch.randn(tokens, C, dtype=torch.bfloat16, device="cuda") * 5.0 + 5.0)
                w = (torch.rand(C, dtype=torch.bfloat16, device="cuda") * 1.5 + 0.5)
                out_fp8 = torch.empty(tokens, C, dtype=torch.float8_e4m3fn, device="cuda")
                flashinfer.fused_rmsnorm_silu(x, w, 1e-6, out=out_fp8)
                ref = rmsnorm_silu_reference(x, w, 1e-6)
                ref_fp8 = ref.float().clamp(-448, 448).to(torch.float8_e4m3fn)
                max_diff = (out_fp8.float() - ref_fp8.float()).abs().max().item()
                status = "PASS" if max_diff < 0.5 else "FAIL"
                if max_diff < 0.5:
                    fp8_passed += 1
                else:
                    fp8_failed += 1
                print(f"  FP8 C={C:>4}, tokens={tokens:>6}: {status}  max_diff={max_diff:.3e}")
            except Exception as e:
                fp8_failed += 1
                print(f"  FP8 C={C:>4}, tokens={tokens:>6}: ERROR  {e}")

    print(f"\n--- FP8 Results: {fp8_passed} passed, {fp8_failed} failed ---")

    # NVFP4 sweep
    print("\n--- NVFP4 Output ---")
    nvfp4_passed = 0
    nvfp4_failed = 0
    for C in [64, 256, 512]:
        for tokens in [1560, 6240]:
            try:
                torch.manual_seed(42)
                x = (torch.randn(tokens, C, dtype=torch.bfloat16, device="cuda") * 5.0 + 5.0)
                w = (torch.rand(C, dtype=torch.bfloat16, device="cuda") * 1.5 + 0.5)
                out_nvfp4 = torch.empty(tokens * C // 2, dtype=torch.uint8, device="cuda")
                flashinfer.fused_rmsnorm_silu(x, w, 1e-6, out=out_nvfp4)
                has_data = out_nvfp4.any().item()
                status = "PASS" if has_data else "FAIL"
                if has_data:
                    nvfp4_passed += 1
                else:
                    nvfp4_failed += 1
                print(f"  NVFP4 C={C:>4}, tokens={tokens:>6}: {status}  has_data={has_data}")
            except Exception as e:
                nvfp4_failed += 1
                print(f"  NVFP4 C={C:>4}, tokens={tokens:>6}: ERROR  {e}")

    print(f"\n--- NVFP4 Results: {nvfp4_passed} passed, {nvfp4_failed} failed ---")

    total = passed + failed + fp8_passed + fp8_failed + nvfp4_passed + nvfp4_failed
    total_pass = passed + fp8_passed + nvfp4_passed
    print(f"\n--- Total: {total_pass}/{total} passed ---")
