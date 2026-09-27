"""
Copyright (c) 2024 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import pytest
import torch
import torch.nn.functional as F

import flashinfer
from flashinfer.gemm import fp8_blockscale_gemm_sm90
from flashinfer.testing.utils import per_block_cast_to_fp8, per_token_cast_to_fp8
from flashinfer.utils import (
    get_compute_capability,
    has_flashinfer_jit_cache,
    is_sm90a_supported,
)
from flashinfer.jit.gemm import gen_fp8_blockscale_gemm_sm90_module


@pytest.fixture(
    autouse=not has_flashinfer_jit_cache(),
    scope="module",
)
def warmup_jit():
    """Warm up JIT compilation for FP8 block-scale GEMM if not cached."""
    if is_sm90a_supported(torch.device("cuda:0")):
        jit_specs = [gen_fp8_blockscale_gemm_sm90_module()]
        flashinfer.jit.build_jit_specs(jit_specs, verbose=False)
    yield


def _dequant_1x128(x_fp8, scale):
    """Dequantize with per-token (M, K // 128) scales."""
    return x_fp8.float() * scale.float().repeat_interleave(128, dim=1)


def _dequant_128x128(x_fp8, scale):
    """Dequantize with per-block (ceil(N / 128), K // 128) scales."""
    n, k = x_fp8.shape
    scale = scale.float().repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
    return x_fp8.float() * scale[:n, :k]


def _rel_err(out, ref):
    return ((out.float() - ref).norm() / ref.norm()).item()


@pytest.mark.parametrize("m", [1, 16, 32, 64, 128])
@pytest.mark.parametrize("n", [128, 256, 512, 1024, 4096])
@pytest.mark.parametrize("k", [256, 512, 1024, 4096])
@pytest.mark.parametrize("input_dtype", [torch.bfloat16])
@pytest.mark.parametrize("weight_dtype", [torch.bfloat16])
def test_fp8_blockscale_gemm_sm90(m, n, k, input_dtype, weight_dtype):
    """Test FP8 block-scale GEMM with swapAB optimization.

    This test focuses on the usage: BF16 inputs with internal quantization.
    The kernel automatically handles FP8 quantization with proper block-scale computation.
    """
    compute_capability = get_compute_capability(torch.device("cuda"))
    if compute_capability[0] < 9:
        pytest.skip("FP8 block-scale GEMM requires SM90 (Hopper) or later")

    if not is_sm90a_supported(torch.device("cuda")):
        pytest.skip("FP8 block-scale GEMM requires SM90a (Hopper) support")

    # K must be divisible by 128 (block size requirement)
    if k % 128 != 0:
        pytest.skip("K must be divisible by 128 for block-scale GEMM")

    device = "cuda"
    torch.manual_seed(42)

    # Create BF16 inputs
    input = torch.randn(m, k, device=device, dtype=input_dtype)
    weight = torch.randn(n, k, device=device, dtype=weight_dtype)

    # Compute reference result
    reference = torch.matmul(input, weight.T)

    # Run FP8 block-scale GEMM
    output = fp8_blockscale_gemm_sm90(input, weight)

    # Verify output shape
    assert output.shape == (m, n), f"Expected shape {(m, n)}, got {output.shape}"
    assert output.dtype == torch.bfloat16, f"Expected BF16 output, got {output.dtype}"

    # Check correctness
    cos_sim = F.cosine_similarity(
        reference.flatten().float(), output.flatten().float(), dim=0
    )
    assert cos_sim > 0.99, f"Cosine similarity {cos_sim} is too low (expected > 0.99)"


@pytest.mark.parametrize("m", [1, 32, 128])
@pytest.mark.parametrize("n", [1024, 4096])
@pytest.mark.parametrize("k", [512, 4096])
@pytest.mark.parametrize(
    "input_dtype,weight_dtype",
    [
        (
            torch.bfloat16,
            torch.bfloat16,
        ),  # Both BF16 (for testing internal quantization)
        (torch.bfloat16, torch.float8_e4m3fn),  # BF16 input + FP8 weight
    ],
)
def test_fp8_blockscale_gemm_dtypes(m, n, k, input_dtype, weight_dtype):
    """Test the 2 recommended dtype combinations with proper FP8 quantization.

    Uses quantization from flashinfer.testing.utils:
    - per_token_cast_to_fp8: 1x128 block quantization for the input
    - per_block_cast_to_fp8: 128x128 block quantization for the weight,
      giving the (N // 128, K // 128) weight scales the kernel expects
    """
    compute_capability = get_compute_capability(torch.device("cuda"))
    if compute_capability[0] < 9:
        pytest.skip("FP8 block-scale GEMM requires SM90 (Hopper) or later")

    if not is_sm90a_supported(torch.device("cuda")):
        pytest.skip("FP8 block-scale GEMM requires SM90a (Hopper) support")

    if k % 128 != 0:
        pytest.skip("K must be divisible by 128 for block-scale GEMM")

    device = "cuda"
    torch.manual_seed(42)
    fp8_info = torch.finfo(torch.float8_e4m3fn)
    fp8_max = fp8_info.max

    # Create BF16 data for reference
    input_bf16 = (
        (torch.rand(m, k, dtype=torch.bfloat16, device=device) - 0.5) * 2 * fp8_max
    )
    weight_bf16 = (
        (torch.rand(n, k, dtype=torch.bfloat16, device=device) - 0.5) * 2 * fp8_max
    )

    # Quantize input
    if input_dtype == torch.float8_e4m3fn:
        input_tensor, input_scale = per_token_cast_to_fp8(input_bf16)
    else:
        input_tensor, input_scale = input_bf16, None

    # Quantize weight
    if weight_dtype == torch.float8_e4m3fn:
        weight_tensor, weight_scale = per_block_cast_to_fp8(weight_bf16)
    else:
        weight_tensor, weight_scale = weight_bf16, None

    # Compute reference
    reference = torch.matmul(input_bf16, weight_bf16.T)

    # Run FP8 block-scale GEMM
    output = fp8_blockscale_gemm_sm90(
        input_tensor, weight_tensor, input_scale, weight_scale
    )

    # Verify output properties
    assert output.shape == (m, n), f"Expected shape {(m, n)}, got {output.shape}"
    assert output.dtype == torch.bfloat16, f"Expected BF16 output, got {output.dtype}"

    # Check correctness
    cos_sim = F.cosine_similarity(
        reference.flatten().float(), output.flatten().float(), dim=0
    )

    threshold = 0.99

    assert cos_sim > threshold, (
        f"Cosine similarity {cos_sim:.4f} too low for "
        f"{input_dtype} + {weight_dtype} (expected > {threshold})"
    )


@pytest.mark.parametrize("m", [7, 32, 128])
@pytest.mark.parametrize("n", [1024, 4096])
@pytest.mark.parametrize("k", [512, 4096])
@pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_fp8_blockscale_gemm_w8a8(m, n, k, input_dtype):
    """Test W8A8 (FP8+FP8) GEMM with 1x128 input and 128x128 weight scales.

    This test demonstrates full FP8 quantization for both activations and weights.
    """
    compute_capability = get_compute_capability(torch.device("cuda"))
    if compute_capability[0] < 9:
        pytest.skip("FP8 block-scale GEMM requires SM90 (Hopper) or later")

    if not is_sm90a_supported(torch.device("cuda")):
        pytest.skip("FP8 block-scale GEMM requires SM90a (Hopper) support")

    device = "cuda"
    # m, n, k = 64, 2048, 4096
    torch.manual_seed(42)
    fp8_info = torch.finfo(torch.float8_e4m3fn)
    fp8_max = fp8_info.max

    # Create BF16 inputs for reference (no normalization)
    # Raw randn values work well with FP8 quantization without causing numerical issues
    input_bf16 = (
        (torch.rand(m, k, dtype=torch.bfloat16, device=device) - 0.5) * 2 * fp8_max
    )
    weight_bf16 = (
        (torch.rand(n, k, dtype=torch.bfloat16, device=device) - 0.5) * 2 * fp8_max
    )

    # Quantize the input with 1x128 scales and the weight with 128x128 scales
    input_fp8, input_scale = per_token_cast_to_fp8(input_bf16)
    weight_fp8, weight_scale = per_block_cast_to_fp8(weight_bf16)

    # Verify scale shapes
    assert input_scale.shape == (m, k // 128), (
        f"Expected input scale shape ({m}, {k // 128}), got {input_scale.shape}"
    )
    assert weight_scale.shape == (n // 128, k // 128), (
        f"Expected weight scale shape ({n // 128}, {k // 128}), got {weight_scale.shape}"
    )
    assert input_scale.min() > 0, "Input scale should be positive"
    assert weight_scale.min() > 0, "Weight scale should be positive"

    if input_dtype == torch.float8_e4m3fn:
        output = fp8_blockscale_gemm_sm90(
            input_fp8, weight_fp8, input_scale, weight_scale
        )
        # Dequantize FP8 tensors to create reference (tests kernel correctness, not quantization)
        input_dequant = _dequant_1x128(input_fp8, input_scale)
    else:
        output = fp8_blockscale_gemm_sm90(input_bf16, weight_fp8, None, weight_scale)
        input_dequant = input_bf16.float()

    weight_dequant = _dequant_128x128(weight_fp8, weight_scale)

    reference = torch.matmul(input_dequant, weight_dequant.T)

    # Use cosine similarity (same metric as BF16+FP8 tests)
    cos_sim = F.cosine_similarity(
        reference.flatten().float(), output.flatten().float(), dim=0
    )

    assert cos_sim > 0.99, (
        f"W8A8 cosine similarity {cos_sim:.4f} too low (expected > 0.99)"
    )


@pytest.mark.parametrize(
    "m,n,k",
    [
        (1, 4096, 4096),
        (8, 4096, 4096),
        (128, 4096, 4096),
        (16, 8192, 8192),
        (32, 2048, 4096),
    ],
)
def test_fp8_blockscale_gemm_shapes(m, n, k):
    """Test various common shapes used in LLM inference."""
    compute_capability = get_compute_capability(torch.device("cuda"))
    if compute_capability[0] < 9:
        pytest.skip("FP8 block-scale GEMM requires SM90 (Hopper) or later")

    if not is_sm90a_supported(torch.device("cuda")):
        pytest.skip("FP8 block-scale GEMM requires SM90a (Hopper) support")

    if k % 128 != 0:
        pytest.skip("K must be divisible by 128")

    device = "cuda"
    torch.manual_seed(42)

    input = torch.randn(m, k, device=device, dtype=torch.bfloat16)
    weight = torch.randn(n, k, device=device, dtype=torch.bfloat16)

    reference = torch.matmul(input, weight.T)
    output = fp8_blockscale_gemm_sm90(input, weight)

    cos_sim = F.cosine_similarity(
        reference.flatten().float(), output.flatten().float(), dim=0
    )
    assert cos_sim > 0.99, f"Shape ({m}, {n}, {k}): cosine similarity {cos_sim} too low"


def test_fp8_blockscale_gemm_error_handling():
    """Test that proper errors are raised for invalid inputs."""
    compute_capability = get_compute_capability(torch.device("cuda"))
    if compute_capability[0] < 9:
        pytest.skip("FP8 block-scale GEMM requires SM90 (Hopper) or later")

    if not is_sm90a_supported(torch.device("cuda")):
        pytest.skip("FP8 block-scale GEMM requires SM90a (Hopper) support")

    device = "cuda"
    m, n, k = 16, 256, 256

    # Test: K not divisible by 128
    input = torch.randn(m, 127, device=device, dtype=torch.bfloat16)
    weight = torch.randn(n, 127, device=device, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="divisible by block size"):
        fp8_blockscale_gemm_sm90(input, weight)

    # Test: FP16 not supported
    input = torch.randn(m, k, device=device, dtype=torch.float16)
    weight = torch.randn(n, k, device=device, dtype=torch.float16)
    with pytest.raises(ValueError, match="FP8.*or BF16"):
        fp8_blockscale_gemm_sm90(input, weight)

    # Test: FP8 weight without scale (naive conversion)
    input_bf16 = torch.randn(m, k, device=device, dtype=torch.bfloat16)
    weight_bf16 = torch.randn(n, k, device=device, dtype=torch.bfloat16)
    weight_fp8_naive = weight_bf16.to(torch.float8_e4m3fn)
    with pytest.raises(ValueError, match="weight_scale is required when weight is FP8"):
        fp8_blockscale_gemm_sm90(input_bf16, weight_fp8_naive, None, None)

    # Test: BF16 input with scale (should raise error)
    input = torch.randn(m, k, device=device, dtype=torch.bfloat16)
    weight = torch.randn(n, k, device=device, dtype=torch.bfloat16)
    fake_scale = torch.ones(m, k // 128, device=device, dtype=torch.float32)
    with pytest.raises(ValueError, match="input_scale should not be provided for BF16"):
        fp8_blockscale_gemm_sm90(input, weight, input_scale=fake_scale)

    # Test: Wrong scale shape for FP8 input
    input_bf16 = torch.randn(m, k, device=device, dtype=torch.bfloat16)
    input_fp8, _ = per_token_cast_to_fp8(input_bf16)
    weight = torch.randn(n, k, device=device, dtype=torch.bfloat16)
    wrong_scale = torch.ones(m, k // 64, device=device, dtype=torch.float32)
    with pytest.raises(ValueError):
        fp8_blockscale_gemm_sm90(input_fp8, weight, input_scale=wrong_scale)

    # Test: FP8 input + BF16 weight is NOT supported
    input_bf16 = torch.randn(m, k, device=device, dtype=torch.bfloat16)
    input_fp8, input_scale = per_token_cast_to_fp8(input_bf16)
    weight = torch.randn(n, k, device=device, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="FP8 input.*BF16 weight.*not supported"):
        fp8_blockscale_gemm_sm90(input_fp8, weight, input_scale, None)


def _input_scale_in_layout(scale, layout):
    """Return per-token (M, K // 128) scales with the same values in another layout."""
    m, k_blocks = scale.shape
    if layout == "row_major":  # per_token_cast_to_fp8 output
        return scale.contiguous()
    if layout == "m_major_tma_aligned":  # strides (1, ceil(M / 4) * 4), read as is
        aligned_m = (m + 3) // 4 * 4
        buf = torch.zeros(k_blocks, aligned_m, dtype=scale.dtype, device=scale.device)
        buf[:, :m] = scale.T
        return buf.T[:m]
    if layout == "m_major_unaligned":  # strides (1, M)
        return scale.T.contiguous().T
    if layout == "strided":
        buf = torch.zeros(m, 2 * k_blocks, dtype=scale.dtype, device=scale.device)
        buf[:, ::2] = scale
        return buf[:, ::2]
    raise ValueError(layout)


@pytest.mark.parametrize(
    "m,n,k",
    [
        (1, 256, 512),
        (4, 256, 512),  # M == K // 128
        (7, 192, 1024),  # M % 4 != 0, N % 128 != 0
        (32, 256, 4096),  # M == K // 128
        (130, 1024, 1024),
    ],
)
@pytest.mark.parametrize(
    "input_layout",
    ["bf16", "row_major", "m_major_tma_aligned", "m_major_unaligned", "strided"],
)
@pytest.mark.parametrize("weight_scale_contiguous", [True, False])
def test_fp8_blockscale_gemm_scale_layouts(
    m, n, k, input_layout, weight_scale_contiguous
):
    """Every supported input/weight scale layout must match an fp32 reference.

    Magnitudes vary per 1x128 block so that a scale read from the wrong
    position changes the result.
    """
    compute_capability = get_compute_capability(torch.device("cuda"))
    if compute_capability[0] < 9:
        pytest.skip("FP8 block-scale GEMM requires SM90 (Hopper) or later")

    if not is_sm90a_supported(torch.device("cuda")):
        pytest.skip("FP8 block-scale GEMM requires SM90a (Hopper) support")

    device = "cuda"
    torch.manual_seed(0)

    def randn_varied(rows, cols):
        mag = torch.exp2(
            torch.randint(-3, 4, (rows, cols // 128), device=device).float()
        )
        x = torch.randn(rows, cols, device=device) * mag.repeat_interleave(128, dim=1)
        return x.to(torch.bfloat16)

    input_bf16 = randn_varied(m, k)
    weight_fp8, weight_scale = per_block_cast_to_fp8(randn_varied(n, k))
    weight_dequant = _dequant_128x128(weight_fp8, weight_scale)
    if not weight_scale_contiguous:
        weight_scale = weight_scale.T.contiguous().T

    if input_layout == "bf16":
        output = fp8_blockscale_gemm_sm90(input_bf16, weight_fp8, None, weight_scale)
        reference = input_bf16.float() @ weight_dequant.T
        # Includes the error of the kernel's internal FP8 input quantization.
        tol = 5e-2
    else:
        input_fp8, input_scale = per_token_cast_to_fp8(input_bf16)
        output = fp8_blockscale_gemm_sm90(
            input_fp8,
            weight_fp8,
            _input_scale_in_layout(input_scale, input_layout),
            weight_scale,
        )
        reference = _dequant_1x128(input_fp8, input_scale) @ weight_dequant.T
        tol = 1e-2

    err = _rel_err(output, reference)
    assert err < tol, f"relative error {err:.4f} (tolerance {tol})"


@pytest.mark.parametrize("m,n,k", [(7, 256, 512), (64, 1024, 1024)])
def test_fp8_blockscale_gemm_unsupported_scale_layouts(m, n, k):
    """Scale layouts the kernel cannot read must raise instead of being misread."""
    compute_capability = get_compute_capability(torch.device("cuda"))
    if compute_capability[0] < 9:
        pytest.skip("FP8 block-scale GEMM requires SM90 (Hopper) or later")

    if not is_sm90a_supported(torch.device("cuda")):
        pytest.skip("FP8 block-scale GEMM requires SM90a (Hopper) support")

    device = "cuda"
    torch.manual_seed(0)
    input_bf16 = torch.randn(m, k, device=device, dtype=torch.bfloat16)
    weight_bf16 = torch.randn(n, k, device=device, dtype=torch.bfloat16)
    input_fp8, input_scale = per_token_cast_to_fp8(input_bf16)
    weight_fp8, weight_scale = per_block_cast_to_fp8(weight_bf16)

    # Per-token (1x128) weight scales: the kernel only applies 128x128 scales.
    weight_fp8_tok, weight_scale_tok = per_token_cast_to_fp8(weight_bf16)
    with pytest.raises(ValueError, match="Per-token .*weight_scale.*not supported"):
        fp8_blockscale_gemm_sm90(input_bf16, weight_fp8_tok, None, weight_scale_tok)
    with pytest.raises(ValueError, match="Per-token .*weight_scale.*not supported"):
        fp8_blockscale_gemm_sm90(
            input_fp8, weight_fp8_tok, input_scale, weight_scale_tok
        )

    # Other weight scale shapes and dtypes.
    with pytest.raises(ValueError, match="weight_scale shape mismatch"):
        fp8_blockscale_gemm_sm90(
            input_bf16, weight_fp8, None, weight_scale.repeat_interleave(2, dim=1)
        )
    with pytest.raises(ValueError, match="weight_scale must be float32"):
        fp8_blockscale_gemm_sm90(
            input_bf16, weight_fp8, None, weight_scale.to(torch.bfloat16)
        )

    # Input scales must be (M, K // 128), including not the transposed shape.
    with pytest.raises(ValueError, match="pass input_scale.T"):
        fp8_blockscale_gemm_sm90(input_fp8, weight_fp8, input_scale.T, weight_scale)
    with pytest.raises(ValueError, match="input_scale shape mismatch"):
        fp8_blockscale_gemm_sm90(
            input_fp8,
            weight_fp8,
            input_scale.repeat_interleave(2, dim=1),
            weight_scale,
        )
    with pytest.raises(ValueError, match="input_scale must be float32"):
        fp8_blockscale_gemm_sm90(
            input_fp8, weight_fp8, input_scale.to(torch.bfloat16), weight_scale
        )


def test_fp8_blockscale_gemm_output_buffer():
    """Test providing pre-allocated output buffer."""
    compute_capability = get_compute_capability(torch.device("cuda"))
    if compute_capability[0] < 9:
        pytest.skip("FP8 block-scale GEMM requires SM90 (Hopper) or later")

    if not is_sm90a_supported(torch.device("cuda")):
        pytest.skip("FP8 block-scale GEMM requires SM90a (Hopper) support")

    device = "cuda"
    m, n, k = 16, 256, 256
    torch.manual_seed(42)

    input = torch.randn(m, k, device=device, dtype=torch.bfloat16)
    weight = torch.randn(n, k, device=device, dtype=torch.bfloat16)

    # Pre-allocate output
    output = torch.empty(m, n, device=device, dtype=torch.bfloat16)

    # Run GEMM with pre-allocated output
    result = fp8_blockscale_gemm_sm90(input, weight, out=output)

    # Verify result is the same buffer
    assert result is output

    # Verify correctness
    reference = torch.matmul(input, weight.T)
    cos_sim = F.cosine_similarity(
        reference.flatten().float(), output.flatten().float(), dim=0
    )
    assert cos_sim > 0.99


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
