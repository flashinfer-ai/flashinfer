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
Unit tests for Fused RMSNorm + FP4 Quantization using CuTe-DSL backend.
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
    y_fp4: torch.Tensor, block_scale: torch.Tensor, block_size: int
):
    """
    Dequantize packed FP4 tensor using the associated block scales.

    Handles both 2D inputs shaped [B, H/2] and 3D inputs shaped [B, S, H/2].
    """
    y_fp4_float = cast_from_fp4(y_fp4)
    if y_fp4_float.dim() == 2:
        b, hidden_size = y_fp4_float.shape
        assert hidden_size % block_size == 0
        y_fp4_float = y_fp4_float.view(b, hidden_size // block_size, block_size)
        # Handle different scale dtype (E4M3 vs UE8M0)
        if block_scale.dtype == torch.uint8:
            # UE8M0: scale = 2^(ue8m0 - 127)
            scales = torch.pow(2.0, block_scale.int() - 127).unsqueeze(-1)
        else:
            scales = block_scale.float().unsqueeze(-1)
        return (y_fp4_float * scales).reshape(b, hidden_size)
    elif y_fp4_float.dim() == 3:
        b, s, hidden_size = y_fp4_float.shape
        assert hidden_size % block_size == 0
        y_fp4_float = y_fp4_float.view(b, s, hidden_size // block_size, block_size)
        if block_scale.dtype == torch.uint8:
            scales = torch.pow(2.0, block_scale.int() - 127).unsqueeze(-1)
        else:
            scales = block_scale.float().unsqueeze(-1)
        return (y_fp4_float * scales).reshape(b, s, hidden_size)
    else:
        raise ValueError(f"Unsupported FP4 output rank: {y_fp4_float.dim()}")


def requires_cute_dsl():
    """Check if CuTe-DSL is available."""
    return is_cute_dsl_available()


def requires_hopper_or_later():
    """Check if running on Hopper (SM90+) or later GPU."""
    return get_cc() >= 90


def requires_blackwell():
    """Check if running on Blackwell GPU."""
    return get_cc() >= 100


# Skip conditions
cute_dsl_available = pytest.mark.skipif(
    not requires_cute_dsl(), reason="CuTe-DSL not available"
)

hopper_required = pytest.mark.skipif(
    not requires_hopper_or_later(),
    reason="CuTe-DSL kernel requires Hopper (SM90+) or later GPU",
)

blackwell_required = pytest.mark.skipif(
    not requires_blackwell(),
    reason="FP4 quantization requires Blackwell GPU (SM100+)",
)


@cute_dsl_available
@blackwell_required
class TestRMSNormFP4QuantCuteDSL:
    """Tests for CuTe-DSL RMSNorm + FP4 Quantization."""

    @pytest.mark.parametrize(
        "batch_size", [1, 4, 16, 32, 7, 13, 33, 100, 128, 8192, 16384]
    )
    @pytest.mark.parametrize(
        "hidden_size", [64, 128, 256, 512, 1024, 1536, 2048, 4096, 8192]
    )
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("eps", [1e-5, 1e-6])
    def test_rmsnorm_fp4quant_2d(self, batch_size, hidden_size, dtype, eps):
        """Test fused RMSNorm + FP4 quantization with 2D input."""
        from flashinfer.cute_dsl.rmsnorm_fp4quant import rmsnorm_fp4quant_cute_dsl

        torch.manual_seed(42)
        block_size = 16

        # Create input tensors
        x = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
        weight = torch.randn(hidden_size, device="cuda", dtype=dtype)

        # Allocate output tensors
        y_fp4 = torch.empty(
            batch_size, hidden_size // 2, device="cuda", dtype=torch.uint8
        )
        block_scale = torch.empty(
            batch_size,
            hidden_size // block_size,
            device="cuda",
            dtype=torch.float8_e4m3fn,
        )

        # Run fused kernel
        rmsnorm_fp4quant_cute_dsl(
            x, weight, y_fp4, block_scale, eps=eps, block_size=block_size
        )

        # Verify output shapes
        assert y_fp4.shape == (batch_size, hidden_size // 2)
        assert block_scale.shape == (batch_size, hidden_size // block_size)

        # Verify output dtypes
        assert y_fp4.dtype == torch.uint8
        assert block_scale.dtype == torch.float8_e4m3fn

        # Reference computation
        ref_rmsnorm = llama_rms_norm(x, weight, eps=eps)

        # Dequantize FP4 output for value-level comparison
        # Tolerance based on separate FP4 roundtrip test (rtol=0.3, atol=0.5)
        y_dequant = dequantize_fp4_output(y_fp4, block_scale, block_size)
        torch.testing.assert_close(
            y_dequant,
            ref_rmsnorm.float(),
            rtol=0.3,
            atol=0.5,
        )

    @pytest.mark.parametrize("batch_size", [1, 4, 3, 7, 128])
    @pytest.mark.parametrize("seq_len", [16, 64, 128, 37, 99])
    @pytest.mark.parametrize("hidden_size", [128, 256, 1536, 4096, 8192])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_rmsnorm_fp4quant_3d(self, batch_size, seq_len, hidden_size, dtype):
        """Test fused RMSNorm + FP4 quantization with 3D input."""
        from flashinfer.cute_dsl.rmsnorm_fp4quant import rmsnorm_fp4quant_cute_dsl

        torch.manual_seed(42)
        block_size = 16
        eps = 1e-5

        # Create input tensors
        x = torch.randn(batch_size, seq_len, hidden_size, device="cuda", dtype=dtype)
        weight = torch.randn(hidden_size, device="cuda", dtype=dtype)

        # Allocate output tensors
        y_fp4 = torch.empty(
            batch_size, seq_len, hidden_size // 2, device="cuda", dtype=torch.uint8
        )
        block_scale = torch.empty(
            batch_size,
            seq_len,
            hidden_size // block_size,
            device="cuda",
            dtype=torch.float8_e4m3fn,
        )

        # Run fused kernel
        rmsnorm_fp4quant_cute_dsl(
            x, weight, y_fp4, block_scale, eps=eps, block_size=block_size
        )

        # Verify output shapes
        assert y_fp4.shape == (batch_size, seq_len, hidden_size // 2)
        assert block_scale.shape == (batch_size, seq_len, hidden_size // block_size)

        # Verify output dtypes
        assert y_fp4.dtype == torch.uint8
        assert block_scale.dtype == torch.float8_e4m3fn

        # Reference computation
        ref_rmsnorm = llama_rms_norm(x, weight, eps=eps)

        # Dequantize FP4 output for value-level comparison
        # Tolerance based on separate FP4 roundtrip test (rtol=0.3, atol=0.5)
        y_dequant = dequantize_fp4_output(y_fp4, block_scale, block_size)
        torch.testing.assert_close(
            y_dequant,
            ref_rmsnorm.float(),
            rtol=0.3,
            atol=0.5,
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
        from flashinfer.cute_dsl.rmsnorm_fp4quant import rmsnorm_fp4quant_cute_dsl

        torch.manual_seed(42)
        block_size = 16
        eps = 1e-6

        x = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
        weight = torch.randn(hidden_size, device="cuda", dtype=dtype)

        y_fp4 = torch.empty(
            batch_size, hidden_size // 2, device="cuda", dtype=torch.uint8
        )
        block_scale = torch.empty(
            batch_size,
            hidden_size // block_size,
            device="cuda",
            dtype=torch.float8_e4m3fn,
        )

        # Should complete without error
        rmsnorm_fp4quant_cute_dsl(
            x, weight, y_fp4, block_scale, eps=eps, block_size=block_size
        )

        # Reference computation (sample first 10 rows for speed)
        ref_rmsnorm = llama_rms_norm(x[:10], weight, eps=eps)
        y_dequant = dequantize_fp4_output(y_fp4[:10], block_scale[:10], block_size)

        torch.testing.assert_close(
            y_dequant,
            ref_rmsnorm.float(),
            rtol=0.3,
            atol=0.5,
        )


@cute_dsl_available
@blackwell_required
class TestRMSNormFP4QuantMXFP4:
    """Tests for MXFP4 format (block_size=32, UE8M0 scales)."""

    @pytest.mark.parametrize("batch_size", [1, 4, 16, 7, 25, 128, 8192])
    @pytest.mark.parametrize("hidden_size", [128, 256, 512, 1536, 2048, 4096])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_mxfp4_basic(self, batch_size, hidden_size, dtype):
        """Test MXFP4 format (block_size=32, UE8M0 scales)."""
        from flashinfer.cute_dsl.rmsnorm_fp4quant import rmsnorm_fp4quant_cute_dsl

        torch.manual_seed(42)
        block_size = 32  # MXFP4
        eps = 1e-6

        x = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
        weight = torch.randn(hidden_size, device="cuda", dtype=dtype)

        y_fp4 = torch.empty(
            batch_size, hidden_size // 2, device="cuda", dtype=torch.uint8
        )
        # UE8M0 scale factors are returned as uint8
        block_scale = torch.empty(
            batch_size, hidden_size // block_size, device="cuda", dtype=torch.uint8
        )

        rmsnorm_fp4quant_cute_dsl(
            x,
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
        assert y_fp4.dtype == torch.uint8
        assert block_scale.dtype == torch.uint8

        # Reference computation
        ref_rmsnorm = llama_rms_norm(x, weight, eps=eps)

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
class TestSeparateFlashInferComparison:
    """Tests comparing CuTe-DSL fused kernel output against reference RMSNorm."""

    @pytest.mark.parametrize("batch_size", [4, 16, 128, 512, 8192])
    @pytest.mark.parametrize("hidden_size", [256, 512, 1024, 1536, 4096, 8192])
    @pytest.mark.parametrize("dtype", [torch.float16])
    def test_fused_vs_separate(self, batch_size, hidden_size, dtype):
        """
        Compare CuTe-DSL fused output with reference RMSNorm.

        We compare the dequantized output against the reference RMSNorm,
        rather than comparing bitwise with separate fp4_quantize (which uses
        different scaling approaches).
        """
        from flashinfer.cute_dsl.rmsnorm_fp4quant import rmsnorm_fp4quant_cute_dsl
        from flashinfer.norm import rmsnorm

        torch.manual_seed(42)
        block_size = 16
        eps = 1e-6

        x = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
        weight = torch.randn(hidden_size, device="cuda", dtype=dtype)

        # Fused CuTe-DSL kernel
        y_fp4_fused = torch.empty(
            batch_size, hidden_size // 2, device="cuda", dtype=torch.uint8
        )
        block_scale_fused = torch.empty(
            batch_size,
            hidden_size // block_size,
            device="cuda",
            dtype=torch.float8_e4m3fn,
        )
        rmsnorm_fp4quant_cute_dsl(
            x, weight, y_fp4_fused, block_scale_fused, eps=eps, block_size=block_size
        )

        # Reference: RMSNorm only
        y_ref = rmsnorm(x, weight, eps=eps)

        # Verify output shapes
        assert y_fp4_fused.shape == (batch_size, hidden_size // 2)
        assert block_scale_fused.shape == (batch_size, hidden_size // block_size)

        # Dequantize fused output and compare to reference
        y_fused_dequant = dequantize_fp4_output(
            y_fp4_fused, block_scale_fused, block_size
        )

        # Value-level comparison against reference RMSNorm output
        torch.testing.assert_close(
            y_fused_dequant,
            y_ref.float(),
            rtol=0.3,
            atol=0.5,
        )


@cute_dsl_available
@blackwell_required
class TestCuDNNComparison:
    """Tests comparing CuTe-DSL backend with cuDNN backend."""

    @pytest.mark.parametrize("batch_size", [1, 4, 16, 7, 25, 128, 8192, 16384])
    @pytest.mark.parametrize("hidden_size", [128, 256, 512, 1536, 4096, 8192])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_cudnn_vs_cute_dsl(self, batch_size, hidden_size, dtype):
        """Compare CuTe-DSL output with cuDNN backend."""
        from flashinfer.cute_dsl.rmsnorm_fp4quant import rmsnorm_fp4quant_cute_dsl
        from flashinfer.norm import rmsnorm_fp4quant, CUDNN_AVAILABLE

        if not CUDNN_AVAILABLE:
            pytest.skip("cuDNN not available")

        try:
            import cudnn

            if cudnn.backend_version() < 91800:
                pytest.skip("cuDNN version too old for FP4")
        except Exception:
            pytest.skip("cuDNN import failed")

        torch.manual_seed(42)
        block_size = 16
        eps = 1e-6

        x = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
        weight = torch.randn(hidden_size, device="cuda", dtype=dtype)

        # cuDNN backend
        y_fp4_cudnn = torch.empty(
            batch_size, hidden_size // 2, device="cuda", dtype=torch.uint8
        )
        block_scale_cudnn = torch.empty(
            batch_size,
            hidden_size // block_size,
            device="cuda",
            dtype=torch.float8_e4m3fn,
        )
        rmsnorm_fp4quant(
            x, weight, y_fp4_cudnn, block_scale_cudnn, eps=eps, block_size=block_size
        )

        # CuTe-DSL backend
        y_fp4_cute = torch.empty(
            batch_size, hidden_size // 2, device="cuda", dtype=torch.uint8
        )
        block_scale_cute = torch.empty(
            batch_size,
            hidden_size // block_size,
            device="cuda",
            dtype=torch.float8_e4m3fn,
        )
        rmsnorm_fp4quant_cute_dsl(
            x, weight, y_fp4_cute, block_scale_cute, eps=eps, block_size=block_size
        )

        # Compare outputs - allow some tolerance due to implementation differences
        fp4_match = (y_fp4_cudnn == y_fp4_cute).float().mean()
        scale_match = (
            (block_scale_cudnn.view(torch.uint8) == block_scale_cute.view(torch.uint8))
            .float()
            .mean()
        )

        # Should have high match ratio (>85%)
        assert fp4_match > 0.85, f"FP4 match ratio too low: {fp4_match:.2%}"
        assert scale_match > 0.85, f"Scale match ratio too low: {scale_match:.2%}"


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
        from flashinfer.cute_dsl.rmsnorm_fp4quant import rmsnorm_fp4quant_cute_dsl

        torch.manual_seed(42)
        block_size = 16
        eps = 1e-6

        x = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
        weight = torch.randn(hidden_size, device="cuda", dtype=dtype)

        y_fp4 = torch.empty(
            batch_size, hidden_size // 2, device="cuda", dtype=torch.uint8
        )
        block_scale = torch.empty(
            batch_size,
            hidden_size // block_size,
            device="cuda",
            dtype=torch.float8_e4m3fn,
        )

        # Run kernel
        rmsnorm_fp4quant_cute_dsl(
            x, weight, y_fp4, block_scale, eps=eps, block_size=block_size
        )

        # Verify output shapes
        assert y_fp4.shape == (batch_size, hidden_size // 2)
        assert block_scale.shape == (batch_size, hidden_size // block_size)
        assert y_fp4.dtype == torch.uint8
        assert block_scale.dtype == torch.float8_e4m3fn

        # Sample first few rows for value comparison (full dequant is slow)
        num_check = min(10, batch_size)
        ref_rmsnorm = llama_rms_norm(x[:num_check], weight, eps=eps)
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
        from flashinfer.cute_dsl.rmsnorm_fp4quant import rmsnorm_fp4quant_cute_dsl

        torch.manual_seed(42)
        block_size = 32  # MXFP4
        eps = 1e-6

        x = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
        weight = torch.randn(hidden_size, device="cuda", dtype=dtype)

        y_fp4 = torch.empty(
            batch_size, hidden_size // 2, device="cuda", dtype=torch.uint8
        )
        block_scale = torch.empty(
            batch_size, hidden_size // block_size, device="cuda", dtype=torch.uint8
        )

        # Run kernel
        rmsnorm_fp4quant_cute_dsl(
            x,
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
        assert y_fp4.dtype == torch.uint8
        assert block_scale.dtype == torch.uint8

        # Sample first few rows for value comparison (full dequant is slow)
        num_check = min(10, batch_size)
        ref_rmsnorm = llama_rms_norm(x[:num_check], weight, eps=eps)
        y_dequant = dequantize_fp4_output(
            y_fp4[:num_check], block_scale[:num_check], block_size
        )

        torch.testing.assert_close(
            y_dequant,
            ref_rmsnorm.float(),
            rtol=0.3,
            atol=0.7,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
