# Copyright (c) 2025 by FlashInfer team.
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
Unit tests for Fused Add + RMSNorm + FP4 Quantization using CuTe-DSL backend.
"""

import pytest
import torch

from flashinfer.cute_dsl.utils import is_cute_dsl_available
from tests.test_helpers.utils_fp4 import cast_from_fp4


def get_cc():
    """Get CUDA compute capability."""
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor


def llama_rms_norm(x, w, eps=1e-6):
    """Reference RMSNorm implementation (LLaMA style)."""
    orig_dtype = x.dtype
    x = x.float()
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    x = x * w.float()
    x = x.to(orig_dtype)
    return x


def dequantize_fp4_output(
    y_fp4: torch.Tensor,
    block_scale: torch.Tensor,
    block_size: int,
    global_scale: torch.Tensor | None = None,
):
    """
    Dequantize packed FP4 tensor using the associated block scales.

    If global_scale is provided, the dequantized values are divided by global_scale
    to reverse the scaling applied during quantization.
    """
    # View as uint8 for bitwise operations in cast_from_fp4
    # (float4_e2m1fn_x2 and uint8 have the same memory layout)
    y_fp4_float = cast_from_fp4(y_fp4.view(torch.uint8))
    if y_fp4_float.dim() == 2:
        b, hidden_size = y_fp4_float.shape
        assert hidden_size % block_size == 0
        y_fp4_float = y_fp4_float.view(b, hidden_size // block_size, block_size)
        if block_scale.dtype == torch.uint8:
            scales = torch.pow(2.0, block_scale.int() - 127).unsqueeze(-1)
        else:
            scales = block_scale.float().unsqueeze(-1)
        result = (y_fp4_float * scales).reshape(b, hidden_size)
    elif y_fp4_float.dim() == 3:
        b, s, hidden_size = y_fp4_float.shape
        assert hidden_size % block_size == 0
        y_fp4_float = y_fp4_float.view(b, s, hidden_size // block_size, block_size)
        if block_scale.dtype == torch.uint8:
            scales = torch.pow(2.0, block_scale.int() - 127).unsqueeze(-1)
        else:
            scales = block_scale.float().unsqueeze(-1)
        result = (y_fp4_float * scales).reshape(b, s, hidden_size)
    else:
        raise ValueError(f"Unsupported FP4 output rank: {y_fp4_float.dim()}")

    # Reverse global scale if it was applied during quantization
    if global_scale is not None:
        result = result / global_scale.item()

    return result


def compute_global_scale(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """
    Compute global scale for NVFP4 quantization of add+rmsnorm output.

    global_scale = (FP8_E4M3_MAX * FP4_E2M1_MAX) / max_abs(rmsnorm(x + residual, weight))

    This ensures the dynamic range of the output fits within the FP4 range.
    """
    FLOAT4_E2M1_MAX = 6.0
    FLOAT8_E4M3_MAX = float(torch.finfo(torch.float8_e4m3fn).max)

    # Compute reference add+RMSNorm output
    h = x + residual
    ref_output = llama_rms_norm(h, weight, eps=eps)
    tensor_amax = torch.abs(ref_output).max().to(torch.float32)
    global_scale = torch.tensor(
        [FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / tensor_amax.item()],
        dtype=torch.float32,
        device=x.device,
    )
    return global_scale


def assert_close_with_tiered_tolerance(
    actual: torch.Tensor,
    expected: torch.Tensor,
    tight_rtol: float = 0.1,
    tight_atol: float = 0.1,
    loose_rtol: float = 0.5,
    loose_atol: float = 2.0,
    tight_pct: float = 0.99,
    msg: str = "",
):
    """
    Two-tiered tolerance check for quantized outputs.

    - tight_pct (e.g., 99%) of elements must be within tight tolerance
    - 100% of elements must be within loose tolerance

    This handles the expected quantization noise where most elements match closely
    but a few outliers may differ more due to rounding boundary effects.
    """
    diff = (actual - expected).abs()
    rel_diff = diff / (expected.abs() + 1e-8)

    # Check 1: tight_pct of elements within tight tolerance
    within_tight = (diff <= tight_atol) | (rel_diff <= tight_rtol)
    tight_pct_actual = within_tight.float().mean().item()
    assert tight_pct_actual >= tight_pct, (
        f"{msg}: Only {tight_pct_actual * 100:.1f}% of elements within tight tolerance "
        f"(rtol={tight_rtol}, atol={tight_atol}), expected {tight_pct * 100:.0f}%"
    )

    # Check 2: 100% of elements within loose tolerance
    within_loose = (diff <= loose_atol) | (rel_diff <= loose_rtol)
    if not within_loose.all():
        max_diff = diff.max().item()
        max_rel = rel_diff.max().item()
        raise AssertionError(
            f"{msg}: Max diff {max_diff:.4f} (rel: {max_rel:.4f}) exceeds loose tolerance "
            f"(rtol={loose_rtol}, atol={loose_atol})"
        )


def requires_cute_dsl():
    """Check if CuTe-DSL is available."""
    return is_cute_dsl_available()


def requires_blackwell():
    """Check if running on Blackwell GPU."""
    return get_cc() >= 100


cute_dsl_available = pytest.mark.skipif(
    not requires_cute_dsl(), reason="CuTe-DSL not available"
)

blackwell_required = pytest.mark.skipif(
    not requires_blackwell(),
    reason="FP4 quantization requires Blackwell GPU (SM100+)",
)


@cute_dsl_available
@blackwell_required
class TestAddRMSNormFP4QuantCuteDSL:
    """Tests for CuTe-DSL Add + RMSNorm + FP4 Quantization."""

    @pytest.mark.parametrize(
        "batch_size", [1, 4, 16, 32, 7, 13, 128, 512, 1000, 8192, 16384]
    )
    @pytest.mark.parametrize(
        "hidden_size", [64, 128, 256, 512, 1024, 1536, 2048, 4096, 8192]
    )
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("eps", [1e-5, 1e-6])
    def test_add_rmsnorm_fp4quant_2d(self, batch_size, hidden_size, dtype, eps):
        """Test fused Add + RMSNorm + FP4 quantization with 2D input."""
        from flashinfer.cute_dsl.add_rmsnorm_fp4quant import (
            add_rmsnorm_fp4quant,
        )

        torch.manual_seed(42)
        block_size = 16

        x = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
        r = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
        weight = torch.randn(hidden_size, device="cuda", dtype=dtype)

        y_fp4 = torch.empty(
            batch_size, hidden_size // 2, device="cuda", dtype=torch.float4_e2m1fn_x2
        )
        block_scale = torch.empty(
            batch_size,
            hidden_size // block_size,
            device="cuda",
            dtype=torch.float8_e4m3fn,
        )

        # Compute reference BEFORE kernel call (kernel modifies r in-place)
        h_ref = x + r
        ref_rmsnorm = llama_rms_norm(h_ref, weight, eps=eps)

        add_rmsnorm_fp4quant(
            x, r, weight, y_fp4, block_scale, eps=eps, block_size=block_size
        )

        # Verify output shapes
        assert y_fp4.shape == (batch_size, hidden_size // 2)
        assert block_scale.shape == (batch_size, hidden_size // block_size)
        assert y_fp4.dtype == torch.float4_e2m1fn_x2
        assert block_scale.dtype == torch.float8_e4m3fn

        # Dequantize FP4 output for value-level comparison
        # Tolerance based on separate FP4 roundtrip test (rtol=0.3, atol=0.5)
        y_dequant = dequantize_fp4_output(y_fp4, block_scale, block_size)
        assert_close_with_tiered_tolerance(
            y_dequant,
            ref_rmsnorm.float(),
            tight_rtol=0.3,
            tight_atol=0.5,
            loose_rtol=0.5,
            loose_atol=2.0,
            tight_pct=0.99,
        )

    @pytest.mark.parametrize("batch_size", [1, 4, 3, 7, 128])
    @pytest.mark.parametrize("seq_len", [16, 64, 128, 37])
    @pytest.mark.parametrize("hidden_size", [128, 256, 1536, 4096, 8192])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_add_rmsnorm_fp4quant_3d(self, batch_size, seq_len, hidden_size, dtype):
        """Test fused Add + RMSNorm + FP4 quantization with 3D input."""
        from flashinfer.cute_dsl.add_rmsnorm_fp4quant import (
            add_rmsnorm_fp4quant,
        )

        torch.manual_seed(42)
        block_size = 16
        eps = 1e-5

        x = torch.randn(batch_size, seq_len, hidden_size, device="cuda", dtype=dtype)
        r = torch.randn(batch_size, seq_len, hidden_size, device="cuda", dtype=dtype)
        weight = torch.randn(hidden_size, device="cuda", dtype=dtype)

        y_fp4 = torch.empty(
            batch_size,
            seq_len,
            hidden_size // 2,
            device="cuda",
            dtype=torch.float4_e2m1fn_x2,
        )
        block_scale = torch.empty(
            batch_size,
            seq_len,
            hidden_size // block_size,
            device="cuda",
            dtype=torch.float8_e4m3fn,
        )

        # Compute reference BEFORE kernel call (kernel modifies r in-place)
        h_ref = x + r
        ref_rmsnorm = llama_rms_norm(h_ref, weight, eps=eps)

        add_rmsnorm_fp4quant(
            x, r, weight, y_fp4, block_scale, eps=eps, block_size=block_size
        )

        # Verify output shapes
        assert y_fp4.shape == (batch_size, seq_len, hidden_size // 2)
        assert block_scale.shape == (batch_size, seq_len, hidden_size // block_size)
        assert y_fp4.dtype == torch.float4_e2m1fn_x2
        assert block_scale.dtype == torch.float8_e4m3fn

        # Tolerance based on separate FP4 roundtrip test (rtol=0.3, atol=0.5)
        y_dequant = dequantize_fp4_output(y_fp4, block_scale, block_size)
        assert_close_with_tiered_tolerance(
            y_dequant,
            ref_rmsnorm.float(),
            tight_rtol=0.3,
            tight_atol=0.5,
            loose_rtol=0.5,
            loose_atol=2.0,
            tight_pct=0.99,
        )

    @pytest.mark.parametrize(
        "batch_size,hidden_size",
        [
            (512, 4096),
            (1024, 4096),
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.float16])
    def test_large_batch(self, batch_size, hidden_size, dtype):
        """Test with large batch sizes."""
        from flashinfer.cute_dsl.add_rmsnorm_fp4quant import (
            add_rmsnorm_fp4quant,
        )

        torch.manual_seed(42)
        block_size = 16
        eps = 1e-6

        x = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
        r = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
        weight = torch.randn(hidden_size, device="cuda", dtype=dtype)

        y_fp4 = torch.empty(
            batch_size, hidden_size // 2, device="cuda", dtype=torch.float4_e2m1fn_x2
        )
        block_scale = torch.empty(
            batch_size,
            hidden_size // block_size,
            device="cuda",
            dtype=torch.float8_e4m3fn,
        )

        # Reference computation (sample first 10 rows for speed)
        # Compute BEFORE kernel call (kernel modifies r in-place)
        h_ref = x[:10] + r[:10]
        ref_rmsnorm = llama_rms_norm(h_ref, weight, eps=eps)

        add_rmsnorm_fp4quant(
            x, r, weight, y_fp4, block_scale, eps=eps, block_size=block_size
        )

        y_dequant = dequantize_fp4_output(y_fp4[:10], block_scale[:10], block_size)

        torch.testing.assert_close(
            y_dequant,
            ref_rmsnorm.float(),
            rtol=0.3,
            atol=0.5,
        )


@cute_dsl_available
@blackwell_required
class TestAddRMSNormFP4QuantMXFP4:
    """Tests for MXFP4 format (block_size=32, UE8M0 scales)."""

    @pytest.mark.parametrize("batch_size", [1, 4, 16, 7, 128, 8192])
    @pytest.mark.parametrize("hidden_size", [128, 256, 512, 1536, 2048, 4096])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_mxfp4_basic(self, batch_size, hidden_size, dtype):
        """Test MXFP4 format (block_size=32, UE8M0 scales)."""
        from flashinfer.cute_dsl.add_rmsnorm_fp4quant import (
            add_rmsnorm_fp4quant,
        )

        torch.manual_seed(42)
        block_size = 32  # MXFP4
        eps = 1e-6

        x = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
        r = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
        weight = torch.randn(hidden_size, device="cuda", dtype=dtype)

        y_fp4 = torch.empty(
            batch_size, hidden_size // 2, device="cuda", dtype=torch.float4_e2m1fn_x2
        )
        # UE8M0 scale factors are returned as uint8
        block_scale = torch.empty(
            batch_size, hidden_size // block_size, device="cuda", dtype=torch.uint8
        )

        # Compute reference BEFORE kernel call (kernel modifies r in-place)
        h_ref = x + r
        ref_rmsnorm = llama_rms_norm(h_ref, weight, eps=eps)

        add_rmsnorm_fp4quant(
            x,
            r,
            weight,
            y_fp4,
            block_scale,
            eps=eps,
            block_size=block_size,
            scale_format="ue8m0",
        )

        # Verify output shapes
        assert y_fp4.shape == (batch_size, hidden_size // 2)
        assert block_scale.shape == (batch_size, hidden_size // block_size)
        assert y_fp4.dtype == torch.float4_e2m1fn_x2
        assert block_scale.dtype == torch.uint8

        # Dequantize FP4 output
        # MXFP4 uses power-of-2 scales which can introduce more quantization error
        y_dequant = dequantize_fp4_output(y_fp4, block_scale, block_size)
        torch.testing.assert_close(
            y_dequant,
            ref_rmsnorm.float(),
            rtol=0.3,
            atol=0.7,
        )


@cute_dsl_available
@blackwell_required
class TestVsSeparateFlashInfer:
    """Tests comparing fused kernel output against reference RMSNorm computation."""

    @pytest.mark.parametrize("batch_size", [4, 16, 128, 512])
    @pytest.mark.parametrize("hidden_size", [256, 512, 1024, 4096, 8192])
    @pytest.mark.parametrize("dtype", [torch.float16])
    def test_fused_vs_separate(self, batch_size, hidden_size, dtype):
        """
        Compare fused kernel output with reference torch.add + rmsnorm.

        Note: We compare against the reference RMSNorm output after dequantization

        """
        from flashinfer.cute_dsl.add_rmsnorm_fp4quant import (
            add_rmsnorm_fp4quant,
        )
        from flashinfer.norm import rmsnorm

        torch.manual_seed(42)
        block_size = 16
        eps = 1e-6

        x = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
        r = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
        weight = torch.randn(hidden_size, device="cuda", dtype=dtype)

        # Reference: torch.add + rmsnorm (compute BEFORE kernel call since kernel modifies r)
        h_ref = x + r
        y_ref = rmsnorm(h_ref, weight, eps=eps)

        # Fused kernel
        y_fp4_fused = torch.empty(
            batch_size, hidden_size // 2, device="cuda", dtype=torch.float4_e2m1fn_x2
        )
        block_scale_fused = torch.empty(
            batch_size,
            hidden_size // block_size,
            device="cuda",
            dtype=torch.float8_e4m3fn,
        )

        add_rmsnorm_fp4quant(
            x, r, weight, y_fp4_fused, block_scale_fused, eps=eps, block_size=block_size
        )

        # Verify output shapes
        assert y_fp4_fused.shape == (batch_size, hidden_size // 2)
        assert block_scale_fused.shape == (batch_size, hidden_size // block_size)

        # Dequantize fused output and compare to reference
        y_fused_dequant = dequantize_fp4_output(
            y_fp4_fused, block_scale_fused, block_size
        )

        # Value-level comparison against reference RMSNorm output
        assert_close_with_tiered_tolerance(
            y_fused_dequant,
            y_ref.float(),
            tight_rtol=0.3,
            tight_atol=0.5,
            loose_rtol=0.5,
            loose_atol=2.0,
            tight_pct=0.99,
        )


@cute_dsl_available
@blackwell_required
class TestFusedVsSeparateFP4Quantize:
    """
    Tests comparing fused Add+RMSNorm+FP4Quant against separate add + RMSNorm + fp4_quantize.

    This validates that the fused kernel applies global_scale identically to the
    standalone fp4_quantize function.
    """

    @pytest.mark.parametrize("batch_size", [1, 4, 16, 128])
    @pytest.mark.parametrize("hidden_size", [64, 256, 512, 1024, 2048, 4096])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_nvfp4_fused_matches_separate(self, batch_size, hidden_size, dtype):
        """
        Compare fused kernel against separate add + RMSNorm + fp4_quantize for NVFP4.

        This test verifies that the fused kernel applies global_scale identically
        to the standalone fp4_quantize function, by comparing:
        1. The packed FP4 output bytes
        2. The block scale factors
        """
        from flashinfer.cute_dsl.add_rmsnorm_fp4quant import add_rmsnorm_fp4quant
        from flashinfer import fp4_quantize

        torch.manual_seed(42)
        block_size = 16  # NVFP4
        eps = 1e-6

        x = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
        r = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
        weight = torch.randn(hidden_size, device="cuda", dtype=dtype)

        # Compute global_scale for NVFP4
        global_scale = compute_global_scale(x, r, weight, eps=eps)

        # === Separate path: add + RMSNorm + fp4_quantize ===
        # Compute BEFORE kernel call (kernel modifies r in-place)
        h_ref = x + r
        y_rmsnorm = llama_rms_norm(h_ref, weight, eps=eps)

        # === Fused kernel path ===
        y_fp4_fused = torch.empty(
            batch_size, hidden_size // 2, device="cuda", dtype=torch.float4_e2m1fn_x2
        )
        block_scale_fused = torch.empty(
            batch_size,
            hidden_size // block_size,
            device="cuda",
            dtype=torch.float8_e4m3fn,
        )
        add_rmsnorm_fp4quant(
            x,
            r,
            weight,
            y_fp4_fused,
            block_scale_fused,
            global_scale=global_scale,
            eps=eps,
            block_size=block_size,
            is_sf_swizzled_layout=False,  # Use unswizzled for easier comparison
        )
        y_fp4_separate, block_scale_separate = fp4_quantize(
            y_rmsnorm,
            global_scale,
            sf_vec_size=block_size,
            sf_use_ue8m0=False,  # E4M3 for NVFP4
            is_sf_swizzled_layout=False,
        )

        # === Compare FP4 packed outputs ===
        # View as uint8 for comparison (float4_e2m1fn_x2 doesn't support == operator)
        fp4_match = (
            (y_fp4_fused.view(torch.uint8) == y_fp4_separate.view(torch.uint8))
            .float()
            .mean()
            .item()
        )
        assert fp4_match > 0.95, (
            f"FP4 output mismatch: only {fp4_match * 100:.1f}% of bytes match"
        )

        # === Compare block scales ===
        scale_fused = block_scale_fused.to(torch.float32)
        scale_separate = (
            block_scale_separate.view(torch.float8_e4m3fn)
            .view(batch_size, -1)
            .to(torch.float32)
        )

        scale_match = (scale_fused == scale_separate).float().mean().item()
        assert scale_match > 0.95, (
            f"Block scale mismatch: only {scale_match * 100:.1f}% of scales match"
        )

        # === Also verify dequantized values are close ===
        y_fused_dequant = dequantize_fp4_output(
            y_fp4_fused, block_scale_fused, block_size, global_scale
        )
        y_separate_dequant = dequantize_fp4_output(
            y_fp4_separate,
            block_scale_separate.view(torch.float8_e4m3fn).view(batch_size, -1),
            block_size,
            global_scale,
        )

        # Two-tiered tolerance: 99% within tight tolerance, 100% within loose tolerance
        assert_close_with_tiered_tolerance(
            y_fused_dequant,
            y_separate_dequant,
            tight_rtol=0.3,
            tight_atol=0.5,
            loose_rtol=0.5,
            loose_atol=2.0,
            tight_pct=0.99,
            msg="Dequantized outputs from fused and separate paths should match closely",
        )

    @pytest.mark.parametrize("batch_size", [1, 4, 16, 128])
    @pytest.mark.parametrize("hidden_size", [128, 256, 512, 1024, 2048, 4096])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_mxfp4_fused_matches_separate(self, batch_size, hidden_size, dtype):
        """
        Compare fused kernel against separate add + RMSNorm + fp4_quantize for MXFP4.

        MXFP4 uses block_size=32, UE8M0 scales, and no global_scale (global_scale=1.0).
        """
        from flashinfer.cute_dsl.add_rmsnorm_fp4quant import add_rmsnorm_fp4quant
        from flashinfer import fp4_quantize

        torch.manual_seed(42)
        block_size = 32  # MXFP4
        eps = 1e-6

        x = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
        r = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
        weight = torch.randn(hidden_size, device="cuda", dtype=dtype)

        # MXFP4 uses global_scale=1.0
        global_scale_val = torch.tensor(1.0, dtype=torch.float32, device="cuda")

        # === Separate path: add + RMSNorm + fp4_quantize ===
        # Compute BEFORE kernel call (kernel modifies r in-place)
        h_ref = x + r
        y_rmsnorm = llama_rms_norm(h_ref, weight, eps=eps)

        # === Fused kernel path ===
        y_fp4_fused = torch.empty(
            batch_size, hidden_size // 2, device="cuda", dtype=torch.float4_e2m1fn_x2
        )
        block_scale_fused = torch.empty(
            batch_size, hidden_size // block_size, device="cuda", dtype=torch.uint8
        )
        add_rmsnorm_fp4quant(
            x,
            r,
            weight,
            y_fp4_fused,
            block_scale_fused,
            eps=eps,
            block_size=block_size,
            scale_format="ue8m0",
            is_sf_swizzled_layout=False,
        )
        y_fp4_separate, block_scale_separate = fp4_quantize(
            y_rmsnorm,
            global_scale_val,
            sf_vec_size=block_size,
            sf_use_ue8m0=True,  # UE8M0 for MXFP4
            is_sf_swizzled_layout=False,
        )

        # === Compare FP4 packed outputs ===
        # View as uint8 for comparison (float4_e2m1fn_x2 doesn't support == operator)
        fp4_match = (
            (y_fp4_fused.view(torch.uint8) == y_fp4_separate.view(torch.uint8))
            .float()
            .mean()
            .item()
        )
        assert fp4_match > 0.95, (
            f"FP4 output mismatch: only {fp4_match * 100:.1f}% of bytes match"
        )

        # === Compare block scales ===
        scale_fused = block_scale_fused
        scale_separate = block_scale_separate.view(batch_size, -1)

        scale_match = (scale_fused == scale_separate).float().mean().item()
        assert scale_match > 0.95, (
            f"Block scale mismatch: only {scale_match * 100:.1f}% of scales match"
        )

        # === Also verify dequantized values are close ===
        # MXFP4 has larger errors due to power-of-2 scale constraints
        y_fused_dequant = dequantize_fp4_output(
            y_fp4_fused, block_scale_fused, block_size
        )
        y_separate_dequant = dequantize_fp4_output(
            y_fp4_separate, scale_separate, block_size
        )

        # Two-tiered tolerance: 99% within tight tolerance, 100% within loose tolerance
        assert_close_with_tiered_tolerance(
            y_fused_dequant,
            y_separate_dequant,
            tight_rtol=0.3,
            tight_atol=0.5,
            loose_rtol=0.5,
            loose_atol=2.0,
            tight_pct=0.99,
            msg="Dequantized outputs from fused and separate paths should match closely",
        )

    @pytest.mark.parametrize("batch_size", [1, 16, 64])
    @pytest.mark.parametrize("hidden_size", [256, 1024, 4096])
    def test_global_scale_value_consistency(self, batch_size, hidden_size):
        """
        Verify that the global_scale value correctly scales the block scales.

        When global_scale is applied:
        - block_scale_with_gs = global_scale * max_abs / FP4_MAX
        - This should be approximately global_scale times larger than without global_scale
        """
        from flashinfer.cute_dsl.add_rmsnorm_fp4quant import add_rmsnorm_fp4quant

        torch.manual_seed(42)
        block_size = 16  # NVFP4
        eps = 1e-6
        dtype = torch.float16

        x = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
        r = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
        weight = torch.randn(hidden_size, device="cuda", dtype=dtype)

        # Run with computed global_scale
        global_scale = compute_global_scale(x, r, weight, eps=eps)

        # Clone r since kernel modifies it in-place and we need to run twice
        r_clone = r.clone()

        y_fp4_gs, block_scale_gs = add_rmsnorm_fp4quant(
            x,
            r,
            weight,
            global_scale=global_scale,
            eps=eps,
            block_size=block_size,
            is_sf_swizzled_layout=False,
        )

        # Run without global_scale (global_scale=1.0)
        global_scale_one = torch.tensor([1.0], dtype=torch.float32, device="cuda")

        y_fp4_no_gs, block_scale_no_gs = add_rmsnorm_fp4quant(
            x,
            r_clone,
            weight,
            global_scale=global_scale_one,
            eps=eps,
            block_size=block_size,
            is_sf_swizzled_layout=False,
        )

        # The block scales with global_scale should be approximately global_scale times
        # larger than without (since block_scale = global_scale * max_abs / FP4_MAX)
        scale_gs = block_scale_gs.to(torch.float32)
        scale_no_gs = block_scale_no_gs.to(torch.float32)

        # Compute ratio where both are non-zero
        non_zero_mask = (scale_no_gs > 0) & (scale_gs > 0)
        if non_zero_mask.sum() > 0:
            ratio = (scale_gs[non_zero_mask] / scale_no_gs[non_zero_mask]).mean().item()
            expected_ratio = global_scale.item()

            # Allow some tolerance due to FP8 quantization
            assert abs(ratio - expected_ratio) / expected_ratio < 0.2, (
                f"Block scale ratio {ratio:.2f} doesn't match expected global_scale {expected_ratio:.2f}"
            )


@cute_dsl_available
@blackwell_required
class TestLargeHiddenSize:
    """Tests for large hidden sizes (16K, 32K) that use cluster synchronization.

    These hidden sizes trigger the cluster sync code path in the CuTe-DSL kernel.
    Uses fewer batch sizes to keep test time reasonable, and samples rows for
    value comparison since full dequantization is slow for large tensors.
    """

    @pytest.mark.parametrize("batch_size", [1, 16, 128, 1024])
    @pytest.mark.parametrize("hidden_size", [16384, 32768])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_large_hidden_nvfp4(self, batch_size, hidden_size, dtype):
        """Test NVFP4 format with large hidden sizes (cluster sync path)."""
        from flashinfer.cute_dsl.add_rmsnorm_fp4quant import (
            add_rmsnorm_fp4quant,
        )

        torch.manual_seed(42)
        block_size = 16
        eps = 1e-6

        x = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
        r = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
        weight = torch.randn(hidden_size, device="cuda", dtype=dtype)

        # Sample first few rows for value comparison (full dequant is slow)
        # Compute BEFORE kernel call (kernel modifies r in-place)
        num_check = min(10, batch_size)
        h_ref = x[:num_check] + r[:num_check]
        ref_rmsnorm = llama_rms_norm(h_ref, weight, eps=eps)

        y_fp4 = torch.empty(
            batch_size, hidden_size // 2, device="cuda", dtype=torch.float4_e2m1fn_x2
        )
        block_scale = torch.empty(
            batch_size,
            hidden_size // block_size,
            device="cuda",
            dtype=torch.float8_e4m3fn,
        )

        # Run kernel
        add_rmsnorm_fp4quant(
            x, r, weight, y_fp4, block_scale, eps=eps, block_size=block_size
        )

        # Verify output shapes
        assert y_fp4.shape == (batch_size, hidden_size // 2)
        assert block_scale.shape == (batch_size, hidden_size // block_size)
        assert y_fp4.dtype == torch.float4_e2m1fn_x2
        assert block_scale.dtype == torch.float8_e4m3fn

        y_dequant = dequantize_fp4_output(
            y_fp4[:num_check], block_scale[:num_check], block_size
        )

        torch.testing.assert_close(
            y_dequant,
            ref_rmsnorm.float(),
            rtol=0.3,
            atol=0.5,
        )

    @pytest.mark.parametrize("batch_size", [1, 16, 128, 1024])
    @pytest.mark.parametrize("hidden_size", [16384, 32768])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_large_hidden_mxfp4(self, batch_size, hidden_size, dtype):
        """Test MXFP4 format with large hidden sizes (cluster sync path)."""
        from flashinfer.cute_dsl.add_rmsnorm_fp4quant import (
            add_rmsnorm_fp4quant,
        )

        torch.manual_seed(42)
        block_size = 32  # MXFP4
        eps = 1e-6

        x = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
        r = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
        weight = torch.randn(hidden_size, device="cuda", dtype=dtype)

        # Sample first few rows for value comparison (full dequant is slow)
        # Compute BEFORE kernel call (kernel modifies r in-place)
        num_check = min(10, batch_size)
        h_ref = x[:num_check] + r[:num_check]
        ref_rmsnorm = llama_rms_norm(h_ref, weight, eps=eps)

        y_fp4 = torch.empty(
            batch_size, hidden_size // 2, device="cuda", dtype=torch.float4_e2m1fn_x2
        )
        block_scale = torch.empty(
            batch_size, hidden_size // block_size, device="cuda", dtype=torch.uint8
        )

        # Run kernel
        add_rmsnorm_fp4quant(
            x,
            r,
            weight,
            y_fp4,
            block_scale,
            eps=eps,
            block_size=block_size,
            scale_format="ue8m0",
        )

        # Verify output shapes
        assert y_fp4.shape == (batch_size, hidden_size // 2)
        assert block_scale.shape == (batch_size, hidden_size // block_size)
        assert y_fp4.dtype == torch.float4_e2m1fn_x2
        assert block_scale.dtype == torch.uint8

        y_dequant = dequantize_fp4_output(
            y_fp4[:num_check], block_scale[:num_check], block_size
        )

        torch.testing.assert_close(
            y_dequant,
            ref_rmsnorm.float(),
            rtol=0.3,
            atol=0.7,
        )


def unswizzle_sf(
    sf: torch.Tensor, row: int, col: int, scaling_vector_size: int = 16
) -> torch.Tensor:
    """
    Unswizzle scale factors from 128x4 tile swizzled layout to row-major layout.

    The swizzle pattern uses 128x4 tiles where scales are arranged as:
    [m_tile][k_tile][outer_m (32)][inner_m (4)][inner_k (4)]

    Parameters
    ----------
    sf : torch.Tensor
        Swizzled scale factor tensor.
    row : int
        Number of rows (batch_size).
    col : int
        Number of columns (hidden_size).
    scaling_vector_size : int
        Block size for quantization (16 for NVFP4, 32 for MXFP4).

    Returns
    -------
    torch.Tensor
        Unswizzled scale factors in row-major layout, shape (row, col // scaling_vector_size).
    """
    factor = scaling_vector_size * 4
    num_m_tiles = (row + 128 - 1) // 128
    num_k_tiles = (col + factor - 1) // factor
    # SF layout [num_m_tiles, num_k_tiles, 32 (m_tile column major), 4 (m_tile column major), 4(k_tile)]
    sf_reshaped = sf.view(num_m_tiles, num_k_tiles, 32, 4, 4)
    sf_unswizzle = sf_reshaped.transpose(1, 3)
    sf_unswizzle = sf_unswizzle.reshape(num_m_tiles * 32 * 4, num_k_tiles * 4)
    sf_unswizzle_sliced = sf_unswizzle[:row, : (col // scaling_vector_size)]
    return sf_unswizzle_sliced.contiguous()


@pytest.mark.skipif(not is_cute_dsl_available(), reason="CuTe-DSL not available")
@pytest.mark.skipif(get_cc() < 100, reason="Requires SM100+")
class TestSwizzledScaleFactors:
    """Tests for swizzled scale factor output layout."""

    @pytest.mark.parametrize("batch_size", [1, 16, 128, 256])
    @pytest.mark.parametrize("hidden_size", [512, 1024, 2048, 4096])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_nvfp4_swizzled_vs_unswizzled(self, batch_size, hidden_size, dtype):
        """
        Test that swizzled output, when unswizzled, matches the non-swizzled output.
        Uses NVFP4 format (block_size=16, E4M3 scales).
        """
        from flashinfer.cute_dsl import add_rmsnorm_fp4quant

        block_size = 16
        torch.manual_seed(42)

        x = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
        r = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
        weight = torch.randn(hidden_size, device="cuda", dtype=dtype)

        # Clone r since kernel modifies it in-place and we need to run twice
        r_clone = r.clone()

        # Non-swizzled output
        y_fp4_ref = torch.empty(
            batch_size, hidden_size // 2, device="cuda", dtype=torch.float4_e2m1fn_x2
        )
        block_scale_ref = torch.empty(
            batch_size,
            hidden_size // block_size,
            device="cuda",
            dtype=torch.float8_e4m3fn,
        )

        # Swizzled output - allocate padded buffer for swizzle
        factor = block_size * 4
        num_m_tiles = (batch_size + 128 - 1) // 128
        num_k_tiles = (hidden_size + factor - 1) // factor
        swizzled_size = num_m_tiles * num_k_tiles * 32 * 4 * 4  # 128x4 tile pattern
        y_fp4_swizzled = torch.empty(
            batch_size, hidden_size // 2, device="cuda", dtype=torch.float4_e2m1fn_x2
        )
        block_scale_swizzled = torch.empty(
            swizzled_size, device="cuda", dtype=torch.float8_e4m3fn
        )

        # Run kernels
        add_rmsnorm_fp4quant(
            x,
            r,
            weight,
            y_fp4_ref,
            block_scale_ref,
            block_size=block_size,
            is_sf_swizzled_layout=False,
        )
        add_rmsnorm_fp4quant(
            x,
            r_clone,
            weight,
            y_fp4_swizzled,
            block_scale_swizzled,
            block_size=block_size,
            is_sf_swizzled_layout=True,
        )

        # Unswizzle and compare
        block_scale_unswizzled = unswizzle_sf(
            block_scale_swizzled.view(torch.uint8), batch_size, hidden_size, block_size
        ).view(torch.float8_e4m3fn)

        # FP4 values should be identical (view as uint8 for comparison)
        torch.testing.assert_close(
            y_fp4_swizzled.view(torch.uint8), y_fp4_ref.view(torch.uint8)
        )

        # Scale factors should match after unswizzling
        torch.testing.assert_close(
            block_scale_unswizzled.view(torch.uint8), block_scale_ref.view(torch.uint8)
        )

    @pytest.mark.parametrize("batch_size", [1, 16, 128, 256])
    @pytest.mark.parametrize("hidden_size", [512, 1024, 2048, 4096])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_mxfp4_swizzled_vs_unswizzled(self, batch_size, hidden_size, dtype):
        """
        Test that swizzled output, when unswizzled, matches the non-swizzled output.
        Uses MXFP4 format (block_size=32, UE8M0 scales).
        """
        from flashinfer.cute_dsl import add_rmsnorm_fp4quant

        block_size = 32
        torch.manual_seed(42)

        x = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
        r = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
        weight = torch.randn(hidden_size, device="cuda", dtype=dtype)

        # Clone r since kernel modifies it in-place and we need to run twice
        r_clone = r.clone()

        # Non-swizzled output
        y_fp4_ref = torch.empty(
            batch_size, hidden_size // 2, device="cuda", dtype=torch.float4_e2m1fn_x2
        )
        block_scale_ref = torch.empty(
            batch_size, hidden_size // block_size, device="cuda", dtype=torch.uint8
        )

        # Swizzled output - allocate padded buffer for swizzle
        factor = block_size * 4
        num_m_tiles = (batch_size + 128 - 1) // 128
        num_k_tiles = (hidden_size + factor - 1) // factor
        swizzled_size = num_m_tiles * num_k_tiles * 32 * 4 * 4  # 128x4 tile pattern
        y_fp4_swizzled = torch.empty(
            batch_size, hidden_size // 2, device="cuda", dtype=torch.float4_e2m1fn_x2
        )
        block_scale_swizzled = torch.empty(
            swizzled_size, device="cuda", dtype=torch.uint8
        )

        # Run kernels
        add_rmsnorm_fp4quant(
            x,
            r,
            weight,
            y_fp4_ref,
            block_scale_ref,
            block_size=block_size,
            is_sf_swizzled_layout=False,
        )
        add_rmsnorm_fp4quant(
            x,
            r_clone,
            weight,
            y_fp4_swizzled,
            block_scale_swizzled,
            block_size=block_size,
            is_sf_swizzled_layout=True,
        )

        # Unswizzle and compare
        block_scale_unswizzled = unswizzle_sf(
            block_scale_swizzled, batch_size, hidden_size, block_size
        )

        # FP4 values should be identical (view as uint8 for comparison)
        torch.testing.assert_close(
            y_fp4_swizzled.view(torch.uint8), y_fp4_ref.view(torch.uint8)
        )

        # Scale factors should match after unswizzling
        torch.testing.assert_close(block_scale_unswizzled, block_scale_ref)


@cute_dsl_available
@blackwell_required
class TestAutoAllocation:
    """Tests for automatic output tensor allocation when y_fp4 and block_scale are None."""

    @pytest.mark.parametrize("batch_size", [1, 16, 128])
    @pytest.mark.parametrize("hidden_size", [256, 1024, 4096])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_auto_allocation_2d_nvfp4(self, batch_size, hidden_size, dtype):
        """Test auto-allocation with 2D input and NVFP4 format."""
        from flashinfer.cute_dsl.add_rmsnorm_fp4quant import add_rmsnorm_fp4quant

        torch.manual_seed(42)
        block_size = 16
        eps = 1e-6

        x = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
        r = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
        weight = torch.randn(hidden_size, device="cuda", dtype=dtype)

        # Reference computation (before kernel call since kernel modifies r in-place)
        h_ref = x + r
        ref_rmsnorm = llama_rms_norm(h_ref, weight, eps=eps)

        # Call without providing y_fp4 and block_scale
        y_fp4, block_scale = add_rmsnorm_fp4quant(
            x, r, weight, eps=eps, block_size=block_size
        )

        # Verify output shapes
        assert y_fp4.shape == (batch_size, hidden_size // 2)
        assert block_scale.shape == (batch_size, hidden_size // block_size)

        # Verify output dtypes
        assert y_fp4.dtype == torch.float4_e2m1fn_x2
        assert block_scale.dtype == torch.float8_e4m3fn

        # Dequantize and verify values
        y_dequant = dequantize_fp4_output(y_fp4, block_scale, block_size)
        torch.testing.assert_close(
            y_dequant,
            ref_rmsnorm.float(),
            rtol=0.3,
            atol=0.5,
        )

    @pytest.mark.parametrize("batch_size", [1, 4, 16])
    @pytest.mark.parametrize("seq_len", [16, 64])
    @pytest.mark.parametrize("hidden_size", [256, 1024])
    @pytest.mark.parametrize("dtype", [torch.float16])
    def test_auto_allocation_3d_nvfp4(self, batch_size, seq_len, hidden_size, dtype):
        """Test auto-allocation with 3D input and NVFP4 format."""
        from flashinfer.cute_dsl.add_rmsnorm_fp4quant import add_rmsnorm_fp4quant

        torch.manual_seed(42)
        block_size = 16
        eps = 1e-6

        x = torch.randn(batch_size, seq_len, hidden_size, device="cuda", dtype=dtype)
        r = torch.randn(batch_size, seq_len, hidden_size, device="cuda", dtype=dtype)
        weight = torch.randn(hidden_size, device="cuda", dtype=dtype)

        # Reference computation (before kernel call since kernel modifies r in-place)
        h_ref = x + r
        ref_rmsnorm = llama_rms_norm(h_ref, weight, eps=eps)

        # Call without providing y_fp4 and block_scale
        y_fp4, block_scale = add_rmsnorm_fp4quant(
            x, r, weight, eps=eps, block_size=block_size
        )

        # Verify output shapes
        assert y_fp4.shape == (batch_size, seq_len, hidden_size // 2)
        assert block_scale.shape == (batch_size, seq_len, hidden_size // block_size)

        # Verify output dtypes
        assert y_fp4.dtype == torch.float4_e2m1fn_x2
        assert block_scale.dtype == torch.float8_e4m3fn

        # Dequantize and verify values
        y_dequant = dequantize_fp4_output(y_fp4, block_scale, block_size)
        torch.testing.assert_close(
            y_dequant,
            ref_rmsnorm.float(),
            rtol=0.3,
            atol=0.5,
        )

    @pytest.mark.parametrize("batch_size", [1, 16, 128])
    @pytest.mark.parametrize("hidden_size", [256, 1024])
    @pytest.mark.parametrize("dtype", [torch.float16])
    def test_auto_allocation_mxfp4(self, batch_size, hidden_size, dtype):
        """Test auto-allocation with MXFP4 format (block_size=32, UE8M0 scales)."""
        from flashinfer.cute_dsl.add_rmsnorm_fp4quant import add_rmsnorm_fp4quant

        torch.manual_seed(42)
        block_size = 32
        eps = 1e-6

        x = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
        r = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
        weight = torch.randn(hidden_size, device="cuda", dtype=dtype)

        # Reference computation (before kernel call since kernel modifies r in-place)
        h_ref = x + r
        ref_rmsnorm = llama_rms_norm(h_ref, weight, eps=eps)

        # Call without providing y_fp4 and block_scale
        y_fp4, block_scale = add_rmsnorm_fp4quant(
            x, r, weight, eps=eps, block_size=block_size, scale_format="ue8m0"
        )

        # Verify output shapes
        assert y_fp4.shape == (batch_size, hidden_size // 2)
        assert block_scale.shape == (batch_size, hidden_size // block_size)

        # Verify output dtypes
        assert y_fp4.dtype == torch.float4_e2m1fn_x2
        assert block_scale.dtype == torch.uint8  # UE8M0 uses uint8

        # Dequantize and verify values
        y_dequant = dequantize_fp4_output(y_fp4, block_scale, block_size)
        torch.testing.assert_close(
            y_dequant,
            ref_rmsnorm.float(),
            rtol=0.3,
            atol=0.7,
        )

    @pytest.mark.parametrize("batch_size", [16, 128])
    @pytest.mark.parametrize("hidden_size", [512, 1024])
    @pytest.mark.parametrize("dtype", [torch.float16])
    def test_auto_allocation_swizzled(self, batch_size, hidden_size, dtype):
        """Test auto-allocation with swizzled scale factor layout."""
        from flashinfer.cute_dsl.add_rmsnorm_fp4quant import add_rmsnorm_fp4quant

        torch.manual_seed(42)
        block_size = 16
        eps = 1e-6

        x = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
        r = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
        weight = torch.randn(hidden_size, device="cuda", dtype=dtype)

        # Clone r since kernel modifies it in-place and we need to run twice
        r_clone = r.clone()

        # Call without providing y_fp4 and block_scale, with swizzled layout
        y_fp4, block_scale = add_rmsnorm_fp4quant(
            x, r, weight, eps=eps, block_size=block_size, is_sf_swizzled_layout=True
        )

        # Verify output shapes
        assert y_fp4.shape == (batch_size, hidden_size // 2)
        # Swizzled layout has different shape
        factor = block_size * 4
        num_m_tiles = (batch_size + 127) // 128
        num_k_tiles = (hidden_size + factor - 1) // factor
        expected_swizzled_size = num_m_tiles * num_k_tiles * 32 * 4 * 4
        assert block_scale.shape == (expected_swizzled_size,)

        # Verify output dtypes
        assert y_fp4.dtype == torch.float4_e2m1fn_x2
        assert block_scale.dtype == torch.float8_e4m3fn

        # Unswizzle and compare with non-swizzled version
        y_fp4_ref = torch.empty(
            batch_size, hidden_size // 2, device="cuda", dtype=torch.float4_e2m1fn_x2
        )
        block_scale_ref = torch.empty(
            batch_size,
            hidden_size // block_size,
            device="cuda",
            dtype=torch.float8_e4m3fn,
        )
        add_rmsnorm_fp4quant(
            x,
            r_clone,
            weight,
            y_fp4_ref,
            block_scale_ref,
            eps=eps,
            block_size=block_size,
        )

        # FP4 values should be identical (view as uint8 for comparison)
        torch.testing.assert_close(y_fp4.view(torch.uint8), y_fp4_ref.view(torch.uint8))

        # Unswizzle and compare scales
        block_scale_unswizzled = unswizzle_sf(
            block_scale.view(torch.uint8), batch_size, hidden_size, block_size
        ).view(torch.float8_e4m3fn)
        torch.testing.assert_close(
            block_scale_unswizzled.view(torch.uint8), block_scale_ref.view(torch.uint8)
        )

    @pytest.mark.parametrize("batch_size", [16, 128])
    @pytest.mark.parametrize("hidden_size", [512, 1024])
    def test_auto_allocation_matches_preallocated(self, batch_size, hidden_size):
        """Test that auto-allocation produces same results as pre-allocated tensors."""
        from flashinfer.cute_dsl.add_rmsnorm_fp4quant import add_rmsnorm_fp4quant

        torch.manual_seed(42)
        block_size = 16
        eps = 1e-6
        dtype = torch.float16

        x = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
        r = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
        weight = torch.randn(hidden_size, device="cuda", dtype=dtype)

        # Clone r since kernel modifies it in-place and we need to run twice
        r_clone = r.clone()

        # Pre-allocated version
        y_fp4_pre = torch.empty(
            batch_size, hidden_size // 2, device="cuda", dtype=torch.float4_e2m1fn_x2
        )
        block_scale_pre = torch.empty(
            batch_size,
            hidden_size // block_size,
            device="cuda",
            dtype=torch.float8_e4m3fn,
        )
        add_rmsnorm_fp4quant(
            x, r, weight, y_fp4_pre, block_scale_pre, eps=eps, block_size=block_size
        )

        # Auto-allocated version
        y_fp4_auto, block_scale_auto = add_rmsnorm_fp4quant(
            x, r_clone, weight, eps=eps, block_size=block_size
        )

        # Results should be identical (view as uint8 for comparison)
        torch.testing.assert_close(
            y_fp4_auto.view(torch.uint8), y_fp4_pre.view(torch.uint8)
        )
        torch.testing.assert_close(
            block_scale_auto.view(torch.uint8), block_scale_pre.view(torch.uint8)
        )


@cute_dsl_available
@blackwell_required
class TestResidualInPlaceUpdate:
    """Tests to verify that the residual tensor is updated in-place with input + residual."""

    @pytest.mark.parametrize("batch_size", [1, 4, 16, 128, 512])
    @pytest.mark.parametrize("hidden_size", [256, 512, 1024, 4096])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_residual_inplace_update_2d(self, batch_size, hidden_size, dtype):
        """Test that residual is updated in-place for 2D input."""
        from flashinfer.cute_dsl.add_rmsnorm_fp4quant import add_rmsnorm_fp4quant

        torch.manual_seed(42)
        block_size = 16
        eps = 1e-6

        x = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
        r = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
        weight = torch.randn(hidden_size, device="cuda", dtype=dtype)

        # Store original values before kernel call
        r_original = r.clone()
        expected_residual = x + r_original

        # Call kernel - residual should be modified in-place
        y_fp4, block_scale = add_rmsnorm_fp4quant(
            x, r, weight, eps=eps, block_size=block_size
        )

        # Verify residual is updated in-place to input + original_residual
        torch.testing.assert_close(
            r,
            expected_residual,
            rtol=0,
            atol=0,
            msg="Residual should be exactly input + original_residual",
        )

    @pytest.mark.parametrize("batch_size", [1, 4, 16])
    @pytest.mark.parametrize("seq_len", [16, 64, 128])
    @pytest.mark.parametrize("hidden_size", [256, 1024, 4096])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_residual_inplace_update_3d(self, batch_size, seq_len, hidden_size, dtype):
        """Test that residual is updated in-place for 3D input."""
        from flashinfer.cute_dsl.add_rmsnorm_fp4quant import add_rmsnorm_fp4quant

        torch.manual_seed(42)
        block_size = 16
        eps = 1e-6

        x = torch.randn(batch_size, seq_len, hidden_size, device="cuda", dtype=dtype)
        r = torch.randn(batch_size, seq_len, hidden_size, device="cuda", dtype=dtype)
        weight = torch.randn(hidden_size, device="cuda", dtype=dtype)

        # Store original values before kernel call
        r_original = r.clone()
        expected_residual = x + r_original

        # Call kernel - residual should be modified in-place
        y_fp4, block_scale = add_rmsnorm_fp4quant(
            x, r, weight, eps=eps, block_size=block_size
        )

        # Verify residual is updated in-place to input + original_residual
        torch.testing.assert_close(
            r,
            expected_residual,
            rtol=0,
            atol=0,
            msg="Residual should be exactly input + original_residual",
        )

    @pytest.mark.parametrize("batch_size", [1, 16, 128])
    @pytest.mark.parametrize("hidden_size", [256, 1024, 2048])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_residual_inplace_update_mxfp4(self, batch_size, hidden_size, dtype):
        """Test that residual is updated in-place for MXFP4 format (block_size=32)."""
        from flashinfer.cute_dsl.add_rmsnorm_fp4quant import add_rmsnorm_fp4quant

        torch.manual_seed(42)
        block_size = 32  # MXFP4
        eps = 1e-6

        x = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
        r = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
        weight = torch.randn(hidden_size, device="cuda", dtype=dtype)

        # Store original values before kernel call
        r_original = r.clone()
        expected_residual = x + r_original

        # Call kernel - residual should be modified in-place
        y_fp4, block_scale = add_rmsnorm_fp4quant(
            x, r, weight, eps=eps, block_size=block_size, scale_format="ue8m0"
        )

        # Verify residual is updated in-place to input + original_residual
        torch.testing.assert_close(
            r,
            expected_residual,
            rtol=0,
            atol=0,
            msg="Residual should be exactly input + original_residual",
        )

    @pytest.mark.parametrize("batch_size", [1, 16, 128])
    @pytest.mark.parametrize("hidden_size", [16384, 32768])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_residual_inplace_update_large_hidden(self, batch_size, hidden_size, dtype):
        """Test residual in-place update with large hidden sizes (cluster sync path)."""
        from flashinfer.cute_dsl.add_rmsnorm_fp4quant import add_rmsnorm_fp4quant

        torch.manual_seed(42)
        block_size = 16
        eps = 1e-6

        x = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
        r = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
        weight = torch.randn(hidden_size, device="cuda", dtype=dtype)

        # Store original values before kernel call
        r_original = r.clone()
        expected_residual = x + r_original

        # Call kernel - residual should be modified in-place
        y_fp4, block_scale = add_rmsnorm_fp4quant(
            x, r, weight, eps=eps, block_size=block_size
        )

        # Verify residual is updated in-place to input + original_residual
        torch.testing.assert_close(
            r,
            expected_residual,
            rtol=0,
            atol=0,
            msg="Residual should be exactly input + original_residual",
        )

    @pytest.mark.parametrize("batch_size", [16, 128])
    @pytest.mark.parametrize("hidden_size", [512, 1024])
    def test_residual_used_for_rmsnorm(self, batch_size, hidden_size):
        """
        Test that the updated residual (input + original_residual) is used for RMSNorm.

        This verifies the correct sequence of operations:
        1. residual = input + residual (in-place)
        2. output = RMSNorm(residual) * weight
        3. quantize output to FP4
        """
        from flashinfer.cute_dsl.add_rmsnorm_fp4quant import add_rmsnorm_fp4quant

        torch.manual_seed(42)
        block_size = 16
        eps = 1e-6
        dtype = torch.float16

        x = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
        r = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
        weight = torch.randn(hidden_size, device="cuda", dtype=dtype)

        # Store original residual and compute expected values
        r_original = r.clone()
        expected_h = x + r_original
        expected_rmsnorm = llama_rms_norm(expected_h, weight, eps=eps)

        # Call kernel
        y_fp4, block_scale = add_rmsnorm_fp4quant(
            x, r, weight, eps=eps, block_size=block_size
        )

        # Verify residual is updated
        torch.testing.assert_close(r, expected_h, rtol=0, atol=0)

        # Verify FP4 output matches RMSNorm of the updated residual
        y_dequant = dequantize_fp4_output(y_fp4, block_scale, block_size)
        assert_close_with_tiered_tolerance(
            y_dequant,
            expected_rmsnorm.float(),
            tight_rtol=0.3,
            tight_atol=0.5,
            loose_rtol=0.5,
            loose_atol=2.0,
            tight_pct=0.99,
            msg="FP4 output should match RMSNorm of updated residual",
        )

    @pytest.mark.parametrize("batch_size", [16, 128])
    @pytest.mark.parametrize("hidden_size", [512, 1024])
    def test_residual_inplace_with_preallocated_outputs(self, batch_size, hidden_size):
        """Test residual in-place update when using pre-allocated output tensors."""
        from flashinfer.cute_dsl.add_rmsnorm_fp4quant import add_rmsnorm_fp4quant

        torch.manual_seed(42)
        block_size = 16
        eps = 1e-6
        dtype = torch.float16

        x = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
        r = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
        weight = torch.randn(hidden_size, device="cuda", dtype=dtype)

        # Pre-allocate output tensors
        y_fp4 = torch.empty(
            batch_size, hidden_size // 2, device="cuda", dtype=torch.float4_e2m1fn_x2
        )
        block_scale = torch.empty(
            batch_size,
            hidden_size // block_size,
            device="cuda",
            dtype=torch.float8_e4m3fn,
        )

        # Store original residual
        r_original = r.clone()
        expected_residual = x + r_original

        # Call kernel with pre-allocated outputs
        add_rmsnorm_fp4quant(
            x, r, weight, y_fp4, block_scale, eps=eps, block_size=block_size
        )

        # Verify residual is updated in-place
        torch.testing.assert_close(
            r,
            expected_residual,
            rtol=0,
            atol=0,
            msg="Residual should be updated in-place even with pre-allocated outputs",
        )

    @pytest.mark.parametrize("batch_size", [16, 128])
    @pytest.mark.parametrize("hidden_size", [512, 1024])
    def test_residual_inplace_swizzled_layout(self, batch_size, hidden_size):
        """Test residual in-place update with swizzled scale factor layout."""
        from flashinfer.cute_dsl.add_rmsnorm_fp4quant import add_rmsnorm_fp4quant

        torch.manual_seed(42)
        block_size = 16
        eps = 1e-6
        dtype = torch.float16

        x = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
        r = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
        weight = torch.randn(hidden_size, device="cuda", dtype=dtype)

        # Store original residual
        r_original = r.clone()
        expected_residual = x + r_original

        # Call kernel with swizzled layout
        y_fp4, block_scale = add_rmsnorm_fp4quant(
            x, r, weight, eps=eps, block_size=block_size, is_sf_swizzled_layout=True
        )

        # Verify residual is updated in-place
        torch.testing.assert_close(
            r,
            expected_residual,
            rtol=0,
            atol=0,
            msg="Residual should be updated in-place with swizzled layout",
        )

    def test_residual_not_aliased_with_input(self):
        """Test that the kernel handles non-aliased input and residual correctly."""
        from flashinfer.cute_dsl.add_rmsnorm_fp4quant import add_rmsnorm_fp4quant

        torch.manual_seed(42)
        batch_size = 64
        hidden_size = 1024
        block_size = 16
        eps = 1e-6
        dtype = torch.float16

        x = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
        r = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
        weight = torch.randn(hidden_size, device="cuda", dtype=dtype)

        # Ensure x and r are separate tensors (not views of each other)
        assert x.data_ptr() != r.data_ptr()

        # Store originals
        x_original = x.clone()
        r_original = r.clone()
        expected_residual = x_original + r_original

        # Call kernel
        y_fp4, block_scale = add_rmsnorm_fp4quant(
            x, r, weight, eps=eps, block_size=block_size
        )

        # Verify x is unchanged
        torch.testing.assert_close(
            x, x_original, rtol=0, atol=0, msg="Input tensor should not be modified"
        )

        # Verify r is updated
        torch.testing.assert_close(
            r,
            expected_residual,
            rtol=0,
            atol=0,
            msg="Residual should be updated in-place",
        )


@cute_dsl_available
@blackwell_required
class TestOutputBothSFLayouts:
    """Tests for output_both_sf_layouts=True which returns both swizzled and unswizzled SFs."""

    @pytest.mark.parametrize("batch_size", [1, 16, 128, 256])
    @pytest.mark.parametrize("hidden_size", [256, 512, 1024, 4096])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_nvfp4_both_sf_layouts_basic(self, batch_size, hidden_size, dtype):
        """
        Test that output_both_sf_layouts=True returns 3 tensors and both SFs are correct.
        Uses NVFP4 format (block_size=16, E4M3 scales).
        """
        from flashinfer.cute_dsl import add_rmsnorm_fp4quant

        block_size = 16
        eps = 1e-6
        torch.manual_seed(42)

        x = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
        r = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
        weight = torch.randn(hidden_size, device="cuda", dtype=dtype)

        # Clone r since kernel modifies it in-place
        r_clone1 = r.clone()

        # Call with output_both_sf_layouts=True
        result = add_rmsnorm_fp4quant(
            x,
            r,
            weight,
            eps=eps,
            block_size=block_size,
            output_both_sf_layouts=True,
        )

        # Should return 3 tensors
        assert len(result) == 3, f"Expected 3 tensors, got {len(result)}"
        y_fp4_both, block_scale_swizzled, block_scale_unswizzled = result

        # Verify shapes
        assert y_fp4_both.shape == (batch_size, hidden_size // 2)
        assert block_scale_unswizzled.shape == (batch_size, hidden_size // block_size)

        # Swizzled layout should be 1D
        factor = block_size * 4
        num_m_tiles = (batch_size + 127) // 128
        num_k_tiles = (hidden_size + factor - 1) // factor
        expected_swizzled_size = num_m_tiles * num_k_tiles * 32 * 4 * 4
        assert block_scale_swizzled.shape == (expected_swizzled_size,)

        # Verify dtypes
        assert y_fp4_both.dtype == torch.float4_e2m1fn_x2
        assert block_scale_swizzled.dtype == torch.float8_e4m3fn
        assert block_scale_unswizzled.dtype == torch.float8_e4m3fn

        # Compare against separate calls with is_sf_swizzled_layout=False
        y_fp4_unswizzled_only, block_scale_ref_unswizzled = add_rmsnorm_fp4quant(
            x,
            r_clone1,
            weight,
            eps=eps,
            block_size=block_size,
            is_sf_swizzled_layout=False,
        )

        # FP4 values should be identical
        assert torch.equal(
            y_fp4_both.view(torch.uint8), y_fp4_unswizzled_only.view(torch.uint8)
        ), "FP4 outputs should be bitwise identical"

        # Unswizzled SF should match the reference unswizzled
        assert torch.equal(
            block_scale_unswizzled.view(torch.uint8),
            block_scale_ref_unswizzled.view(torch.uint8),
        ), "Unswizzled scale factors should be bitwise identical"

        # Verify swizzled SF is correct by unswizzling and comparing
        block_scale_from_swizzled = unswizzle_sf(
            block_scale_swizzled.view(torch.uint8), batch_size, hidden_size, block_size
        ).view(torch.float8_e4m3fn)
        assert torch.equal(
            block_scale_from_swizzled.view(torch.uint8),
            block_scale_ref_unswizzled.view(torch.uint8),
        ), "Unswizzled-from-swizzled scale factors should match reference"

    @pytest.mark.parametrize("batch_size", [1, 16, 128, 256])
    @pytest.mark.parametrize("hidden_size", [256, 512, 1024, 4096])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_mxfp4_both_sf_layouts_basic(self, batch_size, hidden_size, dtype):
        """
        Test that output_both_sf_layouts=True returns 3 tensors for MXFP4 format.
        Uses MXFP4 format (block_size=32, UE8M0 scales).
        """
        from flashinfer.cute_dsl import add_rmsnorm_fp4quant

        block_size = 32
        eps = 1e-6
        torch.manual_seed(42)

        x = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
        r = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
        weight = torch.randn(hidden_size, device="cuda", dtype=dtype)

        # Clone r since kernel modifies it in-place
        r_clone1 = r.clone()

        # Call with output_both_sf_layouts=True
        result = add_rmsnorm_fp4quant(
            x,
            r,
            weight,
            eps=eps,
            block_size=block_size,
            scale_format="ue8m0",
            output_both_sf_layouts=True,
        )

        # Should return 3 tensors
        assert len(result) == 3, f"Expected 3 tensors, got {len(result)}"
        y_fp4_both, block_scale_swizzled, block_scale_unswizzled = result

        # Verify shapes
        assert y_fp4_both.shape == (batch_size, hidden_size // 2)
        assert block_scale_unswizzled.shape == (batch_size, hidden_size // block_size)

        # Verify dtypes (UE8M0 uses uint8)
        assert y_fp4_both.dtype == torch.float4_e2m1fn_x2
        assert block_scale_swizzled.dtype == torch.uint8
        assert block_scale_unswizzled.dtype == torch.uint8

        # Compare against separate calls with is_sf_swizzled_layout=False
        y_fp4_unswizzled_only, block_scale_ref_unswizzled = add_rmsnorm_fp4quant(
            x,
            r_clone1,
            weight,
            eps=eps,
            block_size=block_size,
            scale_format="ue8m0",
            is_sf_swizzled_layout=False,
        )

        # FP4 values should be identical
        assert torch.equal(
            y_fp4_both.view(torch.uint8), y_fp4_unswizzled_only.view(torch.uint8)
        ), "FP4 outputs should be bitwise identical"

        # Unswizzled SF should match the reference unswizzled
        assert torch.equal(block_scale_unswizzled, block_scale_ref_unswizzled), (
            "Unswizzled scale factors should be bitwise identical"
        )

        # Verify swizzled SF is correct by unswizzling and comparing
        block_scale_from_swizzled = unswizzle_sf(
            block_scale_swizzled, batch_size, hidden_size, block_size
        )
        assert torch.equal(block_scale_from_swizzled, block_scale_ref_unswizzled), (
            "Unswizzled-from-swizzled scale factors should match reference"
        )

    @pytest.mark.parametrize("batch_size", [1, 16, 128])
    @pytest.mark.parametrize("hidden_size", [512, 1024, 4096])
    @pytest.mark.parametrize("dtype", [torch.float16])
    def test_both_sf_layouts_consistency(self, batch_size, hidden_size, dtype):
        """
        Test that unswizzling the swizzled SF matches the unswizzled SF.
        This verifies internal consistency of the dual output.
        """
        from flashinfer.cute_dsl import add_rmsnorm_fp4quant

        block_size = 16
        eps = 1e-6
        torch.manual_seed(42)

        x = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
        r = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
        weight = torch.randn(hidden_size, device="cuda", dtype=dtype)

        # Call with output_both_sf_layouts=True
        y_fp4, block_scale_swizzled, block_scale_unswizzled = add_rmsnorm_fp4quant(
            x,
            r,
            weight,
            eps=eps,
            block_size=block_size,
            output_both_sf_layouts=True,
        )

        # Unswizzle the swizzled SF
        block_scale_unswizzled_from_swizzled = unswizzle_sf(
            block_scale_swizzled.view(torch.uint8), batch_size, hidden_size, block_size
        ).view(torch.float8_e4m3fn)

        # Should match the directly returned unswizzled SF
        assert torch.equal(
            block_scale_unswizzled_from_swizzled.view(torch.uint8),
            block_scale_unswizzled.view(torch.uint8),
        ), "Unswizzled-from-swizzled SF should match directly returned unswizzled SF"

    @pytest.mark.parametrize("batch_size", [1, 4, 8])
    @pytest.mark.parametrize("seq_len", [16, 64, 128])
    @pytest.mark.parametrize("hidden_size", [256, 1024])
    @pytest.mark.parametrize("dtype", [torch.float16])
    def test_both_sf_layouts_3d_input(self, batch_size, seq_len, hidden_size, dtype):
        """Test output_both_sf_layouts=True with 3D input tensors."""
        from flashinfer.cute_dsl import add_rmsnorm_fp4quant

        block_size = 16
        eps = 1e-6
        torch.manual_seed(42)

        x = torch.randn(batch_size, seq_len, hidden_size, device="cuda", dtype=dtype)
        r = torch.randn(batch_size, seq_len, hidden_size, device="cuda", dtype=dtype)
        weight = torch.randn(hidden_size, device="cuda", dtype=dtype)

        # Reference computation before kernel call
        h_ref = x + r
        ref_rmsnorm = llama_rms_norm(h_ref, weight, eps=eps)

        # Call with output_both_sf_layouts=True
        result = add_rmsnorm_fp4quant(
            x,
            r,
            weight,
            eps=eps,
            block_size=block_size,
            output_both_sf_layouts=True,
        )

        assert len(result) == 3
        y_fp4, block_scale_swizzled, block_scale_unswizzled = result

        # Verify shapes
        assert y_fp4.shape == (batch_size, seq_len, hidden_size // 2)
        assert block_scale_unswizzled.shape == (
            batch_size,
            seq_len,
            hidden_size // block_size,
        )

        # Verify dtypes
        assert y_fp4.dtype == torch.float4_e2m1fn_x2
        assert block_scale_swizzled.dtype == torch.float8_e4m3fn
        assert block_scale_unswizzled.dtype == torch.float8_e4m3fn

        # Dequantize using unswizzled SF and verify values
        y_dequant = dequantize_fp4_output(y_fp4, block_scale_unswizzled, block_size)
        assert_close_with_tiered_tolerance(
            y_dequant,
            ref_rmsnorm.float(),
            tight_rtol=0.3,
            tight_atol=0.5,
            loose_rtol=0.5,
            loose_atol=2.0,
            tight_pct=0.99,
        )

    @pytest.mark.parametrize("batch_size", [16, 128])
    @pytest.mark.parametrize("hidden_size", [512, 1024])
    @pytest.mark.parametrize("dtype", [torch.float16])
    def test_both_sf_layouts_with_global_scale(self, batch_size, hidden_size, dtype):
        """Test output_both_sf_layouts=True with global_scale applied."""
        from flashinfer.cute_dsl import add_rmsnorm_fp4quant

        block_size = 16
        eps = 1e-6
        torch.manual_seed(42)

        x = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
        r = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
        weight = torch.randn(hidden_size, device="cuda", dtype=dtype)

        # Compute global_scale
        global_scale = compute_global_scale(x, r, weight, eps=eps)

        # Clone r since kernel modifies it in-place
        r_clone = r.clone()

        # Call with output_both_sf_layouts=True and global_scale
        result = add_rmsnorm_fp4quant(
            x,
            r,
            weight,
            global_scale=global_scale,
            eps=eps,
            block_size=block_size,
            output_both_sf_layouts=True,
        )

        assert len(result) == 3
        y_fp4_both, block_scale_swizzled, block_scale_unswizzled = result

        # Compare with is_sf_swizzled_layout=False (unswizzled only)
        y_fp4_ref, block_scale_ref = add_rmsnorm_fp4quant(
            x,
            r_clone,
            weight,
            global_scale=global_scale,
            eps=eps,
            block_size=block_size,
            is_sf_swizzled_layout=False,
        )

        # FP4 values should be identical
        assert torch.equal(y_fp4_both.view(torch.uint8), y_fp4_ref.view(torch.uint8)), (
            "FP4 outputs should be bitwise identical"
        )

        # Unswizzled SF should match
        assert torch.equal(
            block_scale_unswizzled.view(torch.uint8),
            block_scale_ref.view(torch.uint8),
        ), "Unswizzled scale factors should be bitwise identical"

    @pytest.mark.parametrize("batch_size", [16, 128])
    @pytest.mark.parametrize("hidden_size", [512, 1024])
    def test_both_sf_layouts_with_preallocated_tensors(self, batch_size, hidden_size):
        """Test output_both_sf_layouts=True with pre-allocated output tensors."""
        from flashinfer.cute_dsl import add_rmsnorm_fp4quant

        block_size = 16
        eps = 1e-6
        dtype = torch.float16
        torch.manual_seed(42)

        x = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
        r = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
        weight = torch.randn(hidden_size, device="cuda", dtype=dtype)

        # Pre-allocate output tensors
        y_fp4 = torch.empty(
            batch_size, hidden_size // 2, device="cuda", dtype=torch.float4_e2m1fn_x2
        )

        # Swizzled scale factors
        factor = block_size * 4
        num_m_tiles = (batch_size + 127) // 128
        num_k_tiles = (hidden_size + factor - 1) // factor
        swizzled_size = num_m_tiles * num_k_tiles * 32 * 4 * 4
        block_scale_swizzled = torch.empty(
            swizzled_size, device="cuda", dtype=torch.float8_e4m3fn
        )

        # Unswizzled scale factors
        block_scale_unswizzled = torch.empty(
            batch_size,
            hidden_size // block_size,
            device="cuda",
            dtype=torch.float8_e4m3fn,
        )

        # Clone r for comparison
        r_clone = r.clone()

        # Call with pre-allocated tensors
        result = add_rmsnorm_fp4quant(
            x,
            r,
            weight,
            y_fp4=y_fp4,
            block_scale=block_scale_swizzled,
            block_scale_unswizzled=block_scale_unswizzled,
            eps=eps,
            block_size=block_size,
            output_both_sf_layouts=True,
        )

        assert len(result) == 3
        y_fp4_out, block_scale_swizzled_out, block_scale_unswizzled_out = result

        # Verify the returned tensors are the same as pre-allocated
        assert y_fp4_out.data_ptr() == y_fp4.data_ptr()
        assert block_scale_swizzled_out.data_ptr() == block_scale_swizzled.data_ptr()
        assert (
            block_scale_unswizzled_out.data_ptr() == block_scale_unswizzled.data_ptr()
        )

        # Compare with auto-allocated version
        y_fp4_auto, block_scale_swizzled_auto, block_scale_unswizzled_auto = (
            add_rmsnorm_fp4quant(
                x,
                r_clone,
                weight,
                eps=eps,
                block_size=block_size,
                output_both_sf_layouts=True,
            )
        )

        # FP4 and unswizzled SF should be identical
        assert torch.equal(y_fp4.view(torch.uint8), y_fp4_auto.view(torch.uint8)), (
            "FP4 outputs should be bitwise identical"
        )
        assert torch.equal(
            block_scale_unswizzled.view(torch.uint8),
            block_scale_unswizzled_auto.view(torch.uint8),
        ), "Unswizzled scale factors should be bitwise identical"

        # For swizzled SF, compare by unswizzling (to avoid padding differences)
        block_scale_from_swizzled = unswizzle_sf(
            block_scale_swizzled.view(torch.uint8), batch_size, hidden_size, block_size
        ).view(torch.float8_e4m3fn)
        block_scale_from_swizzled_auto = unswizzle_sf(
            block_scale_swizzled_auto.view(torch.uint8),
            batch_size,
            hidden_size,
            block_size,
        ).view(torch.float8_e4m3fn)
        assert torch.equal(
            block_scale_from_swizzled.view(torch.uint8),
            block_scale_from_swizzled_auto.view(torch.uint8),
        ), "Unswizzled-from-swizzled scale factors should match"

    @pytest.mark.parametrize("batch_size", [1, 16, 128])
    @pytest.mark.parametrize("hidden_size", [16384, 32768])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_both_sf_layouts_large_hidden(self, batch_size, hidden_size, dtype):
        """Test output_both_sf_layouts=True with large hidden sizes (cluster sync path)."""
        from flashinfer.cute_dsl import add_rmsnorm_fp4quant

        block_size = 16
        eps = 1e-6
        torch.manual_seed(42)

        x = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
        r = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
        weight = torch.randn(hidden_size, device="cuda", dtype=dtype)

        # Sample first few rows for value comparison (full dequant is slow)
        num_check = min(10, batch_size)
        h_ref = x[:num_check] + r[:num_check]
        ref_rmsnorm = llama_rms_norm(h_ref, weight, eps=eps)

        # Clone r for comparison
        r_clone1 = r.clone()

        # Call with output_both_sf_layouts=True
        result = add_rmsnorm_fp4quant(
            x,
            r,
            weight,
            eps=eps,
            block_size=block_size,
            output_both_sf_layouts=True,
        )

        assert len(result) == 3
        y_fp4_both, block_scale_swizzled, block_scale_unswizzled = result

        # Compare with separate call using is_sf_swizzled_layout=False
        y_fp4_unswizzled, block_scale_ref_unswizzled = add_rmsnorm_fp4quant(
            x,
            r_clone1,
            weight,
            eps=eps,
            block_size=block_size,
            is_sf_swizzled_layout=False,
        )

        # FP4 values should be identical
        assert torch.equal(
            y_fp4_both.view(torch.uint8), y_fp4_unswizzled.view(torch.uint8)
        ), "FP4 outputs should be bitwise identical"

        # Unswizzled scale factors should match
        assert torch.equal(
            block_scale_unswizzled.view(torch.uint8),
            block_scale_ref_unswizzled.view(torch.uint8),
        ), "Unswizzled scale factors should be bitwise identical"

        # Verify swizzled SF by unswizzling and comparing
        block_scale_from_swizzled = unswizzle_sf(
            block_scale_swizzled.view(torch.uint8), batch_size, hidden_size, block_size
        ).view(torch.float8_e4m3fn)
        assert torch.equal(
            block_scale_from_swizzled.view(torch.uint8),
            block_scale_ref_unswizzled.view(torch.uint8),
        ), "Unswizzled-from-swizzled scale factors should match reference"

        # Verify dequantized values
        y_dequant = dequantize_fp4_output(
            y_fp4_both[:num_check], block_scale_unswizzled[:num_check], block_size
        )
        torch.testing.assert_close(
            y_dequant,
            ref_rmsnorm.float(),
            rtol=0.3,
            atol=0.5,
        )

    @pytest.mark.parametrize("batch_size", [16, 128])
    @pytest.mark.parametrize("hidden_size", [512, 1024])
    def test_both_sf_layouts_residual_inplace(self, batch_size, hidden_size):
        """Test that residual is updated in-place when output_both_sf_layouts=True."""
        from flashinfer.cute_dsl import add_rmsnorm_fp4quant

        block_size = 16
        eps = 1e-6
        dtype = torch.float16
        torch.manual_seed(42)

        x = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
        r = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
        weight = torch.randn(hidden_size, device="cuda", dtype=dtype)

        # Store original residual
        r_original = r.clone()
        expected_residual = x + r_original

        # Call kernel
        y_fp4, block_scale_swizzled, block_scale_unswizzled = add_rmsnorm_fp4quant(
            x, r, weight, eps=eps, block_size=block_size, output_both_sf_layouts=True
        )

        # Verify residual is updated in-place
        assert torch.equal(r, expected_residual), (
            "Residual should be updated in-place with output_both_sf_layouts=True"
        )

    def test_is_sf_swizzled_layout_ignored_when_output_both(self):
        """
        Test that is_sf_swizzled_layout is effectively ignored when output_both_sf_layouts=True.
        Both True and False values should produce identical results.
        """
        from flashinfer.cute_dsl import add_rmsnorm_fp4quant

        batch_size = 64
        hidden_size = 1024
        block_size = 16
        eps = 1e-6
        dtype = torch.float16
        torch.manual_seed(42)

        x = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
        r = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
        weight = torch.randn(hidden_size, device="cuda", dtype=dtype)

        # Clone r for both calls
        r_clone = r.clone()

        # Call with is_sf_swizzled_layout=False and output_both_sf_layouts=True
        result1 = add_rmsnorm_fp4quant(
            x,
            r,
            weight,
            eps=eps,
            block_size=block_size,
            is_sf_swizzled_layout=False,
            output_both_sf_layouts=True,
        )

        # Call with is_sf_swizzled_layout=True and output_both_sf_layouts=True
        result2 = add_rmsnorm_fp4quant(
            x,
            r_clone,
            weight,
            eps=eps,
            block_size=block_size,
            is_sf_swizzled_layout=True,
            output_both_sf_layouts=True,
        )

        # Both should return 3 tensors
        assert len(result1) == 3
        assert len(result2) == 3

        y_fp4_1, swizzled_1, unswizzled_1 = result1
        y_fp4_2, swizzled_2, unswizzled_2 = result2

        # FP4 outputs should be identical
        assert torch.equal(y_fp4_1.view(torch.uint8), y_fp4_2.view(torch.uint8)), (
            "FP4 outputs should be bitwise identical regardless of is_sf_swizzled_layout"
        )

        # Unswizzled outputs should be identical
        assert torch.equal(
            unswizzled_1.view(torch.uint8), unswizzled_2.view(torch.uint8)
        ), (
            "Unswizzled outputs should be bitwise identical regardless of is_sf_swizzled_layout"
        )

        # For swizzled outputs, compare by unswizzling (to avoid padding differences)
        swizzled_1_unswizzled = unswizzle_sf(
            swizzled_1.view(torch.uint8), batch_size, hidden_size, block_size
        )
        swizzled_2_unswizzled = unswizzle_sf(
            swizzled_2.view(torch.uint8), batch_size, hidden_size, block_size
        )
        assert torch.equal(swizzled_1_unswizzled, swizzled_2_unswizzled), (
            "Swizzled outputs should be identical regardless of is_sf_swizzled_layout"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
