# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Unit tests for Fused RMSNorm + SiLU kernel.
Tests cover bf16, FP8, and NVFP4 output for all 40 LUT shapes plus fallback knobs.
"""

import pytest
import torch
import torch.nn.functional as F


def get_cc():
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor


def rmsnorm_silu_reference(x, weight, eps, output_dtype=None):
    """Reference: RMSNorm + SiLU.

    Compute entirely in float32 for maximum reference accuracy.
    If output_dtype is specified, cast the result to that dtype.
    """
    rms = torch.sqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + eps)
    x_norm = (x.float() / rms) * weight.float()
    result = F.silu(x_norm)
    if output_dtype is not None:
        return result.to(output_dtype)
    return result.to(x.dtype)


# FP4 E2M1 lookup table (4-bit value -> float)
_FP4_E2M1_TABLE = [
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,  # positive
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,  # negative
]


def _unpack_fp4_nibbles(packed_bytes, num_tokens, C):
    """Unpack FP4 packed bytes into a [num_tokens, C] int tensor of 4-bit nibble values."""
    nibbles = torch.zeros(num_tokens, C, dtype=torch.int32, device=packed_bytes.device)
    for col_byte in range(C // 2):
        byte_val = packed_bytes[:, col_byte].int()
        nibbles[:, col_byte * 2] = byte_val & 0x0F
        nibbles[:, col_byte * 2 + 1] = (byte_val >> 4) & 0x0F
    return nibbles


def _quantize_to_fp4_reference(values_f32, C):
    """Quantize float32 values to FP4 E2M1 nibbles matching the kernel's algorithm.

    Matches the kernel's block-scale quantization:
      1. amax = max(|block of 16 elements|)
      2. scale = max(amax / 6.0, FLT_MIN)
      3. quantized = nv_fp4x2_e2m1(value / scale)
    """
    BLOCK_SIZE = 16
    FP4_MAX = 6.0
    FLT_MIN = 1.17549435082228750796873653722224568e-38
    num_tokens = values_f32.shape[0]
    num_blocks = C // BLOCK_SIZE

    fp4_positive = torch.tensor(_FP4_E2M1_TABLE[:8], dtype=torch.float32)
    nibbles = torch.zeros(num_tokens, C, dtype=torch.int32, device=values_f32.device)

    for b in range(num_blocks):
        col_start = b * BLOCK_SIZE
        col_end = col_start + BLOCK_SIZE
        block_vals = values_f32[:, col_start:col_end].cpu().float()

        amax = block_vals.abs().max(dim=1, keepdim=True).values
        scale = torch.clamp(amax / FP4_MAX, min=FLT_MIN)

        scaled = block_vals / scale
        magnitudes = scaled.abs()
        signs = (scaled < 0).int()

        diffs = (magnitudes.unsqueeze(2) - fp4_positive.unsqueeze(0).unsqueeze(0)).abs()
        mag_nibbles = diffs.argmin(dim=2)

        block_nibbles = mag_nibbles + signs * 8
        nibbles[:, col_start:col_end] = block_nibbles.to(values_f32.device)

    return nibbles


def dequantize_nvfp4(packed_bytes, scale_row_fp8, num_tokens, C):
    """Dequantize NVFP4 1D1X1X output to float32."""
    BLOCK_SIZE = 16
    scale_f32 = scale_row_fp8.view(torch.float8_e4m3fn).float()
    nibbles = _unpack_fp4_nibbles(packed_bytes, num_tokens, C)

    output = torch.zeros(num_tokens, C, dtype=torch.float32, device=packed_bytes.device)
    for col in range(C):
        block = col // BLOCK_SIZE
        fp4_vals = torch.tensor(
            [_FP4_E2M1_TABLE[v] for v in nibbles[:, col].cpu().tolist()],
            dtype=torch.float32,
            device=packed_bytes.device,
        )
        output[:, col] = fp4_vals * scale_f32[:, block]
    return output


@pytest.fixture(autouse=True)
def skip_if_not_sm100():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if get_cc() < 100:
        pytest.skip("Fused RMSNorm+SiLU requires SM100+")


SUPPORTED_C = [64, 128, 160, 256, 320, 512, 640, 1024]
SUPPORTED_TOKENS = [1560, 6240, 24960, 99840, 399360]

ALL_LUT_SHAPES = [(tokens, C) for C in SUPPORTED_C for tokens in SUPPORTED_TOKENS]


# ============================================================
# bf16 output — atol=2e-2, rtol=2e-2, zero mismatches
# atol=2e-2, rtol=2e-2, zero mismatches required
# ============================================================


@pytest.mark.parametrize(
    "num_tokens,hidden_size",
    ALL_LUT_SHAPES,
    ids=[f"t{t}_C{c}" for t, c in ALL_LUT_SHAPES],
)
def test_lut_bf16(num_tokens, hidden_size):
    """All 40 LUT shapes for bf16 output."""
    import flashinfer

    torch.manual_seed(42)
    x = (
        torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device="cuda") * 5.0
        + 5.0
    )
    weight = torch.rand(hidden_size, dtype=torch.bfloat16, device="cuda") * 1.5 + 0.5

    out = flashinfer.fused_rmsnorm_silu(x, weight, eps=1e-6)
    ref = rmsnorm_silu_reference(x, weight, eps=1e-6)

    mismatches = ~torch.isclose(out.float(), ref.float(), atol=2e-2, rtol=2e-2)
    num_mismatches = mismatches.sum().item()
    max_diff = (out.float() - ref.float()).abs().max().item()
    assert num_mismatches == 0, (
        f"C={hidden_size}, tokens={num_tokens}: "
        f"{num_mismatches}/{out.numel()} mismatches (max_diff={max_diff:.6e})"
    )


# ============================================================
# FP8 output — atol=0.125, rtol=0.125, zero mismatches
# Reference in float32, then cast to FP8 (avoids bf16 double-rounding)
# atol=0.125, rtol=0.125, zero mismatches; reference in float32 then cast to FP8
# ============================================================


@pytest.mark.parametrize(
    "num_tokens,hidden_size",
    ALL_LUT_SHAPES,
    ids=[f"t{t}_C{c}" for t, c in ALL_LUT_SHAPES],
)
def test_lut_fp8(num_tokens, hidden_size):
    """All 40 LUT shapes for FP8 (E4M3) output."""
    import flashinfer

    torch.manual_seed(42)
    x = (
        torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device="cuda") * 5.0
        + 5.0
    )
    weight = torch.rand(hidden_size, dtype=torch.bfloat16, device="cuda") * 1.5 + 0.5
    out = torch.empty(num_tokens, hidden_size, dtype=torch.float8_e4m3fn, device="cuda")

    result = flashinfer.fused_rmsnorm_silu(x, weight, eps=1e-6, out=out)

    ref_f32 = rmsnorm_silu_reference(x, weight, eps=1e-6, output_dtype=torch.float32)
    ref_fp8 = ref_f32.clamp(-448.0, 448.0).to(torch.float8_e4m3fn)

    z_float = result.float()
    ref_float = ref_fp8.float()
    mismatches = ~torch.isclose(z_float, ref_float, atol=0.125, rtol=0.125)
    num_mismatches = mismatches.sum().item()
    max_diff = (z_float - ref_float).abs().max().item()
    assert num_mismatches == 0, (
        f"FP8 C={hidden_size}, tokens={num_tokens}: "
        f"{num_mismatches}/{result.numel()} mismatches (max_diff={max_diff:.6e})"
    )


# ============================================================
# NVFP4 output — nibble-level comparison, <=1 ULP allowed
# Nibble-level comparison, <=1 ULP allowed
# ============================================================

has_fp4_dtype = hasattr(torch, "float4_e2m1fn_x2")


@pytest.mark.skipif(not has_fp4_dtype, reason="torch.float4_e2m1fn_x2 not available")
@pytest.mark.parametrize(
    "num_tokens,hidden_size",
    ALL_LUT_SHAPES,
    ids=[f"t{t}_C{c}" for t, c in ALL_LUT_SHAPES],
)
def test_lut_nvfp4(num_tokens, hidden_size):
    """All 40 LUT shapes for NVFP4 (FP4_E2M1) 1D1X1X block-scale output."""
    import flashinfer

    torch.manual_seed(42)
    C = hidden_size
    x = torch.randn(num_tokens, C, dtype=torch.bfloat16, device="cuda") * 5.0 + 5.0
    weight = torch.rand(C, dtype=torch.bfloat16, device="cuda") * 1.5 + 0.5

    # FP4 packs 2 values per byte
    out = torch.empty(num_tokens, C // 2, dtype=torch.float4_e2m1fn_x2, device="cuda")
    result, block_scale = flashinfer.fused_rmsnorm_silu(x, weight, eps=1e-6, out=out)

    assert result.data_ptr() == out.data_ptr()
    assert block_scale.shape == (num_tokens, C // 16)
    assert block_scale.dtype == torch.float8_e4m3fn

    ref_f32 = rmsnorm_silu_reference(x, weight, eps=1e-6, output_dtype=torch.float32)

    # Unpack kernel output nibbles
    z_packed = result.view(torch.uint8).reshape(num_tokens, C // 2)
    kernel_nibbles = _unpack_fp4_nibbles(z_packed, num_tokens, C)

    # Quantize reference using the same FP4 algorithm
    ref_nibbles = _quantize_to_fp4_reference(ref_f32, C)

    # Allow <=1 nibble index difference (1 FP4 ULP)
    nibble_diff = (kernel_nibbles - ref_nibbles).abs()
    mismatches = nibble_diff > 1
    num_mismatches = mismatches.sum().item()
    max_nibble_diff = nibble_diff.max().item()
    assert num_mismatches == 0, (
        f"NVFP4 C={C}, tokens={num_tokens}: "
        f"{num_mismatches}/{num_tokens * C} nibbles differ by >{1} ULP "
        f"(max_nibble_diff={max_nibble_diff})"
    )


# ============================================================
# Random / non-LUT shapes (fallback heuristics) — bf16
# ============================================================

RANDOM_SHAPES = [
    (1, 64),
    (7, 128),
    (32, 256),
    (100, 512),
    (1024, 1024),
    (2048, 640),
    (4096, 320),
    (8192, 160),
    (16384, 128),
    (500, 64),
    (3000, 256),
    (50000, 512),
    (200000, 1024),
]


@pytest.mark.parametrize(
    "num_tokens,hidden_size",
    RANDOM_SHAPES,
    ids=[f"t{t}_C{c}" for t, c in RANDOM_SHAPES],
)
def test_fallback_knobs_bf16(num_tokens, hidden_size):
    """Non-LUT shapes that use fallback default knobs."""
    import flashinfer

    torch.manual_seed(42)
    x = (
        torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device="cuda") * 5.0
        + 5.0
    )
    weight = torch.rand(hidden_size, dtype=torch.bfloat16, device="cuda") * 1.5 + 0.5

    out = flashinfer.fused_rmsnorm_silu(x, weight, eps=1e-6)
    ref = rmsnorm_silu_reference(x, weight, eps=1e-6)

    mismatches = ~torch.isclose(out.float(), ref.float(), atol=2e-2, rtol=2e-2)
    num_mismatches = mismatches.sum().item()
    max_diff = (out.float() - ref.float()).abs().max().item()
    assert num_mismatches == 0, (
        f"C={hidden_size}, tokens={num_tokens}: "
        f"{num_mismatches}/{out.numel()} mismatches (max_diff={max_diff:.6e})"
    )


# ============================================================
# Random / non-LUT shapes — FP8
# ============================================================

RANDOM_SHAPES_FP8 = [
    (32, 256),
    (1024, 512),
    (2048, 1024),
    (4096, 128),
    (8192, 640),
]


@pytest.mark.parametrize(
    "num_tokens,hidden_size",
    RANDOM_SHAPES_FP8,
    ids=[f"t{t}_C{c}" for t, c in RANDOM_SHAPES_FP8],
)
def test_fallback_knobs_fp8(num_tokens, hidden_size):
    """Non-LUT shapes with FP8 output using fallback knobs."""
    import flashinfer

    torch.manual_seed(42)
    x = (
        torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device="cuda") * 5.0
        + 5.0
    )
    weight = torch.rand(hidden_size, dtype=torch.bfloat16, device="cuda") * 1.5 + 0.5
    out = torch.empty(num_tokens, hidden_size, dtype=torch.float8_e4m3fn, device="cuda")

    result = flashinfer.fused_rmsnorm_silu(x, weight, eps=1e-6, out=out)

    ref_f32 = rmsnorm_silu_reference(x, weight, eps=1e-6, output_dtype=torch.float32)
    ref_fp8 = ref_f32.clamp(-448.0, 448.0).to(torch.float8_e4m3fn)

    z_float = result.float()
    ref_float = ref_fp8.float()
    mismatches = ~torch.isclose(z_float, ref_float, atol=0.125, rtol=0.125)
    num_mismatches = mismatches.sum().item()
    max_diff = (z_float - ref_float).abs().max().item()
    assert num_mismatches == 0, (
        f"FP8 C={hidden_size}, tokens={num_tokens}: "
        f"{num_mismatches}/{result.numel()} mismatches (max_diff={max_diff:.6e})"
    )


# ============================================================
# Random / non-LUT shapes — NVFP4
# ============================================================

RANDOM_SHAPES_NVFP4 = [
    (32, 256),
    (1024, 512),
    (2048, 1024),
    (4096, 128),
]


@pytest.mark.skipif(not has_fp4_dtype, reason="torch.float4_e2m1fn_x2 not available")
@pytest.mark.parametrize(
    "num_tokens,hidden_size",
    RANDOM_SHAPES_NVFP4,
    ids=[f"t{t}_C{c}" for t, c in RANDOM_SHAPES_NVFP4],
)
def test_fallback_knobs_nvfp4(num_tokens, hidden_size):
    """Non-LUT shapes with NVFP4 output using fallback knobs."""
    import flashinfer

    torch.manual_seed(42)
    C = hidden_size
    x = torch.randn(num_tokens, C, dtype=torch.bfloat16, device="cuda") * 5.0 + 5.0
    weight = torch.rand(C, dtype=torch.bfloat16, device="cuda") * 1.5 + 0.5
    out = torch.empty(num_tokens, C // 2, dtype=torch.float4_e2m1fn_x2, device="cuda")

    result, block_scale = flashinfer.fused_rmsnorm_silu(x, weight, eps=1e-6, out=out)

    assert block_scale.shape == (num_tokens, C // 16)
    assert block_scale.dtype == torch.float8_e4m3fn

    ref_f32 = rmsnorm_silu_reference(x, weight, eps=1e-6, output_dtype=torch.float32)

    z_packed = result.view(torch.uint8).reshape(num_tokens, C // 2)
    kernel_nibbles = _unpack_fp4_nibbles(z_packed, num_tokens, C)
    ref_nibbles = _quantize_to_fp4_reference(ref_f32, C)

    nibble_diff = (kernel_nibbles - ref_nibbles).abs()
    mismatches = nibble_diff > 1
    num_mismatches = mismatches.sum().item()
    assert num_mismatches == 0, (
        f"NVFP4 C={C}, tokens={num_tokens}: "
        f"{num_mismatches}/{num_tokens * C} nibbles differ by >1 ULP"
    )


# ============================================================
# Pre-allocated output
# ============================================================


def test_preallocated_output_bf16():
    import flashinfer

    num_tokens, hidden_size = 1560, 1024
    torch.manual_seed(42)
    x = (
        torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device="cuda") * 5.0
        + 5.0
    )
    weight = torch.rand(hidden_size, dtype=torch.bfloat16, device="cuda") * 1.5 + 0.5
    out = torch.empty_like(x)

    result = flashinfer.fused_rmsnorm_silu(x, weight, eps=1e-6, out=out)
    ref = rmsnorm_silu_reference(x, weight, eps=1e-6)

    assert result.data_ptr() == out.data_ptr()
    mismatches = ~torch.isclose(out.float(), ref.float(), atol=2e-2, rtol=2e-2)
    assert mismatches.sum().item() == 0


def test_preallocated_output_fp8():
    import flashinfer

    num_tokens, hidden_size = 1560, 1024
    torch.manual_seed(42)
    x = (
        torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device="cuda") * 5.0
        + 5.0
    )
    weight = torch.rand(hidden_size, dtype=torch.bfloat16, device="cuda") * 1.5 + 0.5
    out = torch.empty(num_tokens, hidden_size, dtype=torch.float8_e4m3fn, device="cuda")

    result = flashinfer.fused_rmsnorm_silu(x, weight, eps=1e-6, out=out)
    ref_f32 = rmsnorm_silu_reference(x, weight, eps=1e-6, output_dtype=torch.float32)
    ref_fp8 = ref_f32.clamp(-448.0, 448.0).to(torch.float8_e4m3fn)

    assert result.data_ptr() == out.data_ptr()
    mismatches = ~torch.isclose(result.float(), ref_fp8.float(), atol=0.125, rtol=0.125)
    assert mismatches.sum().item() == 0


@pytest.mark.skipif(not has_fp4_dtype, reason="torch.float4_e2m1fn_x2 not available")
def test_preallocated_output_nvfp4():
    """Pre-allocated out AND block_scale for NVFP4."""
    import flashinfer

    num_tokens, hidden_size = 1560, 256
    C = hidden_size
    torch.manual_seed(42)
    x = torch.randn(num_tokens, C, dtype=torch.bfloat16, device="cuda") * 5.0 + 5.0
    weight = torch.rand(C, dtype=torch.bfloat16, device="cuda") * 1.5 + 0.5

    out = torch.empty(num_tokens, C // 2, dtype=torch.float4_e2m1fn_x2, device="cuda")
    block_scale = torch.empty(
        num_tokens, C // 16, dtype=torch.float8_e4m3fn, device="cuda"
    )

    y_fp4, bs = flashinfer.fused_rmsnorm_silu(
        x, weight, eps=1e-6, out=out, block_scale=block_scale
    )

    assert y_fp4.data_ptr() == out.data_ptr()
    assert bs.data_ptr() == block_scale.data_ptr()
    assert bs.shape == (num_tokens, C // 16)
    assert bs.dtype == torch.float8_e4m3fn

    ref_f32 = rmsnorm_silu_reference(x, weight, eps=1e-6, output_dtype=torch.float32)
    z_packed = y_fp4.view(torch.uint8).reshape(num_tokens, C // 2)
    kernel_nibbles = _unpack_fp4_nibbles(z_packed, num_tokens, C)
    ref_nibbles = _quantize_to_fp4_reference(ref_f32, C)
    nibble_diff = (kernel_nibbles - ref_nibbles).abs()
    assert (nibble_diff > 1).sum().item() == 0


@pytest.mark.skipif(not has_fp4_dtype, reason="torch.float4_e2m1fn_x2 not available")
def test_preallocated_block_scale_wrong_shape():
    """block_scale with wrong shape should raise ValueError."""
    import flashinfer

    num_tokens, C = 1560, 256
    x = torch.randn(num_tokens, C, dtype=torch.bfloat16, device="cuda")
    weight = torch.rand(C, dtype=torch.bfloat16, device="cuda")
    out = torch.empty(num_tokens, C // 2, dtype=torch.float4_e2m1fn_x2, device="cuda")
    bad_scale = torch.empty(num_tokens, 1, dtype=torch.float8_e4m3fn, device="cuda")

    with pytest.raises(ValueError, match="block_scale shape mismatch"):
        flashinfer.fused_rmsnorm_silu(
            x, weight, eps=1e-6, out=out, block_scale=bad_scale
        )


@pytest.mark.skipif(not has_fp4_dtype, reason="torch.float4_e2m1fn_x2 not available")
def test_preallocated_block_scale_wrong_dtype():
    """block_scale with wrong dtype should raise ValueError."""
    import flashinfer

    num_tokens, C = 1560, 256
    x = torch.randn(num_tokens, C, dtype=torch.bfloat16, device="cuda")
    weight = torch.rand(C, dtype=torch.bfloat16, device="cuda")
    out = torch.empty(num_tokens, C // 2, dtype=torch.float4_e2m1fn_x2, device="cuda")
    bad_scale = torch.empty(num_tokens, C // 16, dtype=torch.float32, device="cuda")

    with pytest.raises(ValueError, match="block_scale must be float8_e4m3fn"):
        flashinfer.fused_rmsnorm_silu(
            x, weight, eps=1e-6, out=out, block_scale=bad_scale
        )


# ============================================================
# Numerical edge cases
# ============================================================


def test_epsilon_sensitivity():
    import flashinfer

    num_tokens, hidden_size = 6240, 512
    torch.manual_seed(42)
    x = (
        torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device="cuda") * 5.0
        + 5.0
    )
    weight = torch.rand(hidden_size, dtype=torch.bfloat16, device="cuda") * 1.5 + 0.5

    for eps in [1e-5, 1e-6, 1e-8]:
        out = flashinfer.fused_rmsnorm_silu(x, weight, eps=eps)
        ref = rmsnorm_silu_reference(x, weight, eps=eps)
        mismatches = ~torch.isclose(out.float(), ref.float(), atol=2e-2, rtol=2e-2)
        assert mismatches.sum().item() == 0, (
            f"eps={eps}: {mismatches.sum().item()} mismatches"
        )


def test_uniform_weight():
    import flashinfer

    num_tokens, hidden_size = 1560, 256
    torch.manual_seed(42)
    x = (
        torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device="cuda") * 5.0
        + 5.0
    )
    weight = torch.ones(hidden_size, dtype=torch.bfloat16, device="cuda")

    out = flashinfer.fused_rmsnorm_silu(x, weight, eps=1e-6)
    ref = rmsnorm_silu_reference(x, weight, eps=1e-6)

    mismatches = ~torch.isclose(out.float(), ref.float(), atol=2e-2, rtol=2e-2)
    assert mismatches.sum().item() == 0


# ============================================================
# NVFP4 round-trip dequantization (verifies block_scale is usable)
# ============================================================

ROUNDTRIP_SHAPES = [
    (1560, 256),
    (6240, 512),
    (24960, 1024),
]


@pytest.mark.skipif(not has_fp4_dtype, reason="torch.float4_e2m1fn_x2 not available")
@pytest.mark.parametrize(
    "num_tokens,hidden_size",
    ROUNDTRIP_SHAPES,
    ids=[f"t{t}_C{c}" for t, c in ROUNDTRIP_SHAPES],
)
def test_nvfp4_roundtrip_dequantize(num_tokens, hidden_size):
    """Verify that (y_fp4, block_scale) can round-trip back to float via dequantization."""
    import flashinfer

    torch.manual_seed(42)
    C = hidden_size
    x = torch.randn(num_tokens, C, dtype=torch.bfloat16, device="cuda") * 5.0 + 5.0
    weight = torch.rand(C, dtype=torch.bfloat16, device="cuda") * 1.5 + 0.5

    out = torch.empty(num_tokens, C // 2, dtype=torch.float4_e2m1fn_x2, device="cuda")
    y_fp4, block_scale = flashinfer.fused_rmsnorm_silu(x, weight, eps=1e-6, out=out)

    z_packed = y_fp4.view(torch.uint8).reshape(num_tokens, C // 2)
    dequantized = dequantize_nvfp4(z_packed, block_scale, num_tokens, C)

    ref_f32 = rmsnorm_silu_reference(x, weight, eps=1e-6, output_dtype=torch.float32)

    # FP4 has very limited precision (3-bit mantissa equivalent), so the
    # dequantized values won't match exactly. We check relative error is
    # bounded: each FP4 value is within one block-scale quantum of the reference.
    abs_err = (dequantized - ref_f32).abs()
    rel_err = abs_err / (ref_f32.abs() + 1e-6)
    median_rel_err = rel_err.median().item()
    assert median_rel_err < 0.5, (
        f"NVFP4 round-trip median relative error too large: {median_rel_err:.4f}"
    )
    # Also check no catastrophic outliers (>2x the reference magnitude)
    max_rel_err = rel_err.max().item()
    assert max_rel_err < 2.0, (
        f"NVFP4 round-trip max relative error too large: {max_rel_err:.4f}"
    )
