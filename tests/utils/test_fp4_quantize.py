import functools
import os
from dataclasses import dataclass

import pytest
import torch
from tests.test_helpers.utils_fp4 import (
    cast_from_fp4,
    nvfp4_global_decode_scale_te,
    nvfp4_global_encode_scale_te,
    ref_fp4_quant,
    ref_fp4_quant_4over6_te,
    ref_fp4_quant_te,
)

from flashinfer import (
    block_scale_interleave,
    e2m1_and_ufp8sf_scale_to_float,
    fp4_quantize,
    mxfp4_quantize,
    mxfp4_dequantize,
    nvfp4_quantize,
    nvfp4_batched_quantize,
    scaled_fp4_grouped_quantize,
    silu_and_mul_scaled_nvfp4_experts_quantize,
    silu_and_mul,
    SfLayout,
)
from flashinfer.quantization.nvfp4_quantization_utils import (
    NVFP44Over6Config,
)
from flashinfer.quantization.fp4_quantization import NVFP4_QUANT_ENV_VARS
from flashinfer.utils import (
    is_sm100a_supported,
    is_sm110a_supported,
    is_sm12x_supported,
)

pytestmark = pytest.mark.long_running


def _is_fp4_supported(device: torch.device) -> bool:
    """Check if FP4 quantization is supported on this device."""
    return (
        is_sm100a_supported(device)
        or is_sm110a_supported(device)
        or is_sm12x_supported(device)
    )


DTYPES = [torch.float16, torch.bfloat16]
# The batch dimension doesn't need to be multiple of 128
SHAPES = [(128, 64), (256, 128), (120, 64), (200, 256), (2048, 2048)]
BATCH_SHAPES = [
    (1, 256, 128),
    (2, 128, 64),
    (3, 256, 128),
    (1, 120, 64),
    (128, 2048, 2048),
]
SEEDS = [42]
CUDA_DEVICES = ["cuda:0"]

FLOAT4_E2M1_MAX = 6.0
FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max

BLOCK_SIZE = 16
FP4_BACKENDS = ["cuda", "cute-dsl"]


def swizzle_sf(
    unswizzled_sf: torch.Tensor,
    original_row: int,
    original_col: int,
    scaling_vector_size: int = 16,
) -> torch.Tensor:
    """
    Inverse of `unswizzle_sf`. Converts an unswizzled tensor back to swizzled form.

    Args:
        unswizzled_sf: Tensor of shape [row, col // scaling_vector_size].
        original_row: Original row dimension (e.g., 120).
        original_col: Original column dimension (e.g., 64).
        scaling_vector_size: Scaling factor (default 16).

    Returns:
        Swizzled tensor of shape [padded_row, padded_col // scaling_vector_size].
    """
    unswizzled_sf = unswizzled_sf.contiguous()
    factor = scaling_vector_size * 4
    padded_row = ((original_row + 128 - 1) // 128) * 128  # Next multiple of 128
    padded_col = ((original_col + factor - 1) // factor) * factor  # Next multiple of 64

    # Pad the input tensor to [padded_row, padded_col // scaling_vector_size]
    pad_rows = padded_row - original_row
    pad_cols = (padded_col - original_col) // scaling_vector_size
    padded_sf = torch.nn.functional.pad(
        unswizzled_sf,
        (0, pad_cols, 0, pad_rows),
        mode="constant",
        value=0,
    ).contiguous()

    # Reshape and transpose to reverse unswizzle_sf
    num_m_tiles = padded_row // 128
    num_k_tiles = padded_col // factor
    sf_reshaped = padded_sf.view(num_m_tiles, 4, 32, num_k_tiles, 4)  # Reverse reshape
    sf_swizzled = sf_reshaped.transpose(
        1, 3
    )  # Reverse transpose [num_m_tiles, num_k_tiles, 32, 4, 4]
    sf_swizzled = sf_swizzled.reshape(
        padded_row, padded_col // scaling_vector_size
    )  # Flatten to [128, 64]

    return sf_swizzled.contiguous()


def unswizzle_sf(
    sf: torch.Tensor, row: int, col: int, scaling_vector_size: int = 16
) -> torch.Tensor:
    factor = scaling_vector_size * 4
    num_m_tiles = (row + 128 - 1) // 128
    num_k_tiles = (col + factor - 1) // factor
    # SF layout [num_m_tiles, num_k_tiles, 32 (m_tile column major), 4 (m_tile column major), 4(k_tile)]
    sf_reshaped = sf.view(num_m_tiles, num_k_tiles, 32, 4, 4)
    sf_unswizzle = sf_reshaped.transpose(1, 3)
    sf_unswizzle = sf_unswizzle.reshape(num_m_tiles * 32 * 4, num_k_tiles * 4)
    sf_unswizzle_sliced = sf_unswizzle[:row, : (col // scaling_vector_size)]
    return sf_unswizzle_sliced.contiguous()


@pytest.mark.parametrize("backend", FP4_BACKENDS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("sf_use_ue8m0", [False, True])
@pytest.mark.parametrize("is_swizzled", [False, True])
@torch.inference_mode()
def test_fp4_quantization(
    backend: str,
    dtype: torch.dtype,
    shape: tuple[int, int],
    seed: int,
    device: str,
    sf_use_ue8m0: bool,
    is_swizzled: bool,
) -> None:
    if not _is_fp4_supported(torch.device(device)):
        pytest.skip("Nvfp4 Requires compute capability >= 10 and CUDA >= 12.8")
    if backend == "cute-dsl":
        if not _is_cute_dsl_available():
            pytest.skip("CuTe-DSL not available")
    torch.set_default_device(device)
    torch.manual_seed(seed)
    m, n = shape
    sf_vec_size = 32 if sf_use_ue8m0 else 16
    x = torch.randn((m, n), dtype=dtype)
    tensor_amax = torch.abs(x).max().to(torch.float32)
    if sf_use_ue8m0:
        global_scale = torch.tensor(1.0, dtype=torch.float32)
    else:
        global_scale = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / tensor_amax
    out_ref, scale_ref = ref_fp4_quant(x, global_scale, sf_vec_size, sf_use_ue8m0)
    out, out_scale = fp4_quantize(
        x, global_scale, sf_vec_size, sf_use_ue8m0, is_swizzled, backend=backend
    )
    assert n % sf_vec_size == 0, f"cols needs to be {sf_vec_size} divisible"
    if sf_use_ue8m0:
        out_scale = (out_scale.to(torch.int32) << 23).view(torch.float32)
    else:
        out_scale = out_scale.view(torch.float8_e4m3fn).to(torch.float32)
    if is_swizzled:
        scale_ans = unswizzle_sf(
            out_scale.reshape(-1, n // sf_vec_size), m, n, sf_vec_size
        )
    else:
        scale_ans = out_scale
    out_ans = cast_from_fp4(out).reshape(m, n)
    torch.testing.assert_close(out_ans, out_ref, rtol=1e0, atol=1e-1)
    torch.testing.assert_close(scale_ans, scale_ref, rtol=1e-1, atol=1e-1)


@pytest.mark.parametrize("backend", FP4_BACKENDS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_scale_swizzling(
    backend: str,
    dtype: torch.dtype,
    shape: tuple[int, int],
    seed: int,
    device: str,
) -> None:
    if not _is_fp4_supported(torch.device("cuda")):
        pytest.skip("Nvfp4 Requires compute capability >= 10 and CUDA >= 12.8")
    if backend == "cute-dsl" and not _is_cute_dsl_available():
        pytest.skip("CuTe-DSL not available")
    torch.set_default_device(device)
    torch.manual_seed(seed)
    m, n = shape
    x = torch.randn((m, n), dtype=dtype)
    tensor_amax = torch.abs(x).max().to(torch.float32)
    global_scale = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / tensor_amax

    _, unswizzled_scale = fp4_quantize(
        x, global_scale, BLOCK_SIZE, False, False, backend=backend
    )
    _, swizzled_scale = fp4_quantize(
        x, global_scale, BLOCK_SIZE, False, True, backend=backend
    )
    assert n % BLOCK_SIZE == 0, f"cols needs to be {BLOCK_SIZE} divisible"
    recovered_unswizzled_scale = unswizzle_sf(
        swizzle_sf(unswizzled_scale, m, n),
        m,
        n,
    )

    # We don't expect the following since padding:
    # swizzle_sf(unswizzled_scale) == swizzled_scale
    ref_unswizzled_scale = unswizzle_sf(swizzled_scale, m, n)
    assert_equal = functools.partial(torch.testing.assert_close, rtol=0, atol=0)
    assert_equal(recovered_unswizzled_scale, unswizzled_scale)
    assert_equal(ref_unswizzled_scale, unswizzled_scale)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_block_scale_interleave(
    shape: tuple[int, int],
    seed: int,
    device: str,
) -> None:
    """Test the block_scale_interleave function directly."""
    if not _is_fp4_supported(torch.device("cuda")):
        pytest.skip("Nvfp4 Requires compute capability >= 10 and CUDA >= 12.8")
    torch.set_default_device(device)
    torch.manual_seed(seed)

    m, n = shape
    sf_vec_size = BLOCK_SIZE

    # Create a test scale factors tensor with uint8 dtype
    # The shape should be [m, n // sf_vec_size] for scale factors
    scale_shape = (m, n // sf_vec_size)
    unswizzled_sf = torch.randint(0, 256, scale_shape, dtype=torch.uint8, device=device)

    # Test the swizzling function
    swizzled_sf = block_scale_interleave(unswizzled_sf)

    # Compare against the reference implementation
    ref_swizzled_sf = swizzle_sf(unswizzled_sf, m, n, sf_vec_size)

    # Basic checks
    assert swizzled_sf.dtype == torch.uint8, f"Expected uint8, got {swizzled_sf.dtype}"
    assert swizzled_sf.device == unswizzled_sf.device, "Device mismatch"

    # Check that the output has the expected padded shape
    factor = sf_vec_size * 4
    padded_row = ((m + 128 - 1) // 128) * 128  # Next multiple of 128
    padded_col = ((n + factor - 1) // factor) * factor  # Next multiple of 64
    expected_shape = (padded_row, padded_col // sf_vec_size)
    expected_size = expected_shape[0] * expected_shape[1]

    assert expected_size == swizzled_sf.shape[0], (
        f"Expected size {expected_size}, got {swizzled_sf.shape[0]}"
    )
    assert_equal = functools.partial(torch.testing.assert_close, rtol=0, atol=0)
    assert_equal(swizzled_sf.reshape(expected_shape), ref_swizzled_sf)


@pytest.mark.parametrize("backend", FP4_BACKENDS)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("sf_use_ue8m0", [True, False])
@torch.inference_mode()
def test_e2m1_dequantization(
    backend: str,
    shape: tuple[int, int],
    seed: int,
    device: str,
    sf_use_ue8m0: bool,
) -> None:
    """Test roundtrip: fp4_quantize -> e2m1_and_ufp8sf_scale_to_float."""
    if not _is_fp4_supported(torch.device("cuda")):
        pytest.skip("Nvfp4 Requires compute capability >= 10 and CUDA >= 12.8")
    if backend == "cute-dsl" and not _is_cute_dsl_available():
        pytest.skip("CuTe-DSL not available")
    torch.set_default_device(device)
    torch.manual_seed(seed)

    # Create a reasonable test tensor
    m, n = shape
    x = torch.randn((m, n), dtype=torch.float16)

    # Calculate global scale as in the other tests
    tensor_amax = torch.abs(x).max().to(torch.float32)
    global_scale = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / tensor_amax

    # Test with default common settings
    is_sf_swizzled_layout = True
    block_size = 32 if sf_use_ue8m0 else 16

    # Step 1: Quantize with fp4_quantize
    quantized_tensor, scale_factors = fp4_quantize(
        x,
        global_scale,
        block_size,
        sf_use_ue8m0,
        is_sf_swizzled_layout,
        backend=backend,
    )

    # Step 2: Dequantize with e2m1_and_ufp8sf_scale_to_float
    ufp8_type = 0 if sf_use_ue8m0 else 1
    dequantized_tensor = e2m1_and_ufp8sf_scale_to_float(
        quantized_tensor,
        scale_factors,
        1 / global_scale,
        sf_vec_size=block_size,
        ufp8_type=ufp8_type,
        is_sf_swizzled_layout=is_sf_swizzled_layout,
    )

    # Move back to device for comparison
    dequantized_tensor = dequantized_tensor.to(device)
    x_float32 = x.to(torch.float32)

    # Step 3: Compare results
    assert dequantized_tensor.shape == x.shape, (
        f"Shape mismatch: expected {x.shape}, got {dequantized_tensor.shape}"
    )
    assert dequantized_tensor.dtype == torch.float32, (
        f"Expected float32, got {dequantized_tensor.dtype}"
    )

    # Check for invalid values
    assert not torch.isnan(dequantized_tensor).any(), (
        "Dequantized tensor contains NaN values"
    )
    assert not torch.isinf(dequantized_tensor).any(), (
        "Dequantized tensor contains Inf values"
    )

    # Compare with original - should be reasonably close since FP4 is designed to preserve important values
    torch.testing.assert_close(
        dequantized_tensor,
        x_float32,
        rtol=0.3,
        atol=0.5,  # Reasonable tolerance for FP4 quantization
        msg="Quantize -> dequantize roundtrip failed",
    )


# =============================================================================
# MXFP4 Quantization Tests (Both Backends)
# =============================================================================

MXFP4_SHAPES = [
    # K must be a multiple of 128 so K/32 is a multiple of 4 (CUDA reshape
    # constraint for swizzled layout).
    # Small M with swizzled layout: padded_M >> M (row padding dominance)
    (1, 128),  # padded_M=128, 127 padding rows
    (1, 1024),  # padded_M=128, large K
    (3, 256),  # padded_M=128, odd M
    (16, 128),  # padded_M=128, 112 padding rows
    (64, 128),  # padded_M=128, 64 padding rows
    # Standard sizes
    (128, 128),
    (256, 128),
    (512, 256),
    (128, 1024),
    (1024, 2048),
    # Large K (column loop path in swizzled kernel)
    (128, 16384),
]
MXFP4_BACKENDS = ["cuda", "cute-dsl"]


def _is_cute_dsl_available():
    """Check if CuTe-DSL is available."""
    try:
        from flashinfer.cute_dsl import is_cute_dsl_available

        return is_cute_dsl_available()
    except ImportError:
        return False


@pytest.mark.parametrize("backend", MXFP4_BACKENDS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", MXFP4_SHAPES)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_mxfp4_quantize_roundtrip(
    backend: str,
    dtype: torch.dtype,
    shape: tuple[int, int],
    device: str,
) -> None:
    """Test MXFP4 quantization roundtrip for both backends."""
    if not _is_fp4_supported(torch.device(device)):
        pytest.skip("Nvfp4 Requires compute capability >= 10 and CUDA >= 12.8")
    if backend == "cute-dsl" and not _is_cute_dsl_available():
        pytest.skip("CuTe-DSL not available")

    torch.set_default_device(device)
    torch.manual_seed(42)

    m, n = shape
    x = torch.randn((m, n), dtype=dtype)

    # Test specified backend
    quant_out, scale_out = mxfp4_quantize(x, backend=backend)

    # Basic shape checks
    assert quant_out.shape == (m, n // 2), (
        f"Expected shape ({m}, {n // 2}), got {quant_out.shape}"
    )
    assert quant_out.dtype == torch.uint8, f"Expected uint8, got {quant_out.dtype}"
    assert scale_out.dtype == torch.uint8, f"Expected uint8, got {scale_out.dtype}"

    # Check roundtrip with mxfp4_dequantize
    dq_out = mxfp4_dequantize(quant_out, scale_out)

    # Verify no NaN/Inf
    assert not torch.isnan(dq_out).any(), "Dequantized tensor contains NaN"
    assert not torch.isinf(dq_out).any(), "Dequantized tensor contains Inf"

    # Verify roundtrip is reasonably accurate
    torch.testing.assert_close(
        dq_out.cpu().to(torch.float32),
        x.cpu().to(torch.float32),
        rtol=0.3,
        atol=0.5,
        msg=f"{backend} MXFP4 quantize -> dequantize roundtrip failed",
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", MXFP4_SHAPES)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_mxfp4_quantize_backend_parity(
    dtype: torch.dtype,
    shape: tuple[int, int],
    device: str,
) -> None:
    """Test that CUDA and CuTe-DSL backends produce matching results."""
    if not _is_fp4_supported(torch.device(device)):
        pytest.skip("Nvfp4 Requires compute capability >= 10 and CUDA >= 12.8")
    if not _is_cute_dsl_available():
        pytest.skip("CuTe-DSL not available")

    torch.set_default_device(device)
    torch.manual_seed(42)

    m, n = shape
    x = torch.randn((m, n), dtype=dtype)

    # Get results from both backends
    quant_cuda, scale_cuda = mxfp4_quantize(x, backend="cuda")
    quant_cute, scale_cute = mxfp4_quantize(x, backend="cute-dsl")

    # Shape should match
    assert quant_cuda.shape == quant_cute.shape, "Quantized output shape mismatch"
    assert scale_cuda.shape == scale_cute.shape, "Scale output shape mismatch"

    # Dequantize both and compare
    dq_cuda = mxfp4_dequantize(quant_cuda, scale_cuda)
    dq_cute = mxfp4_dequantize(quant_cute, scale_cute)

    # Compute detailed error statistics
    dq_cuda_f32 = dq_cuda.cpu().to(torch.float32)
    dq_cute_f32 = dq_cute.cpu().to(torch.float32)
    abs_diff = (dq_cuda_f32 - dq_cute_f32).abs()
    rel_diff = abs_diff / (dq_cuda_f32.abs() + 1e-8)

    # Print diagnostic info on failure
    max_abs_diff = abs_diff.max().item()
    mean_abs_diff = abs_diff.mean().item()
    max_rel_diff = rel_diff.max().item()
    mean_rel_diff = rel_diff.mean().item()

    # Check quantized data match
    quant_match_pct = (quant_cuda == quant_cute).float().mean().item() * 100
    scale_match_pct = (scale_cuda == scale_cute).float().mean().item() * 100

    error_msg = (
        f"CUDA and CuTe-DSL backends differ after dequantization:\n"
        f"  Shape: {shape}, dtype: {dtype}\n"
        f"  Quantized match: {quant_match_pct:.1f}%, Scale match: {scale_match_pct:.1f}%\n"
        f"  Abs diff - max: {max_abs_diff:.6f}, mean: {mean_abs_diff:.6f}\n"
        f"  Rel diff - max: {max_rel_diff:.6f}, mean: {mean_rel_diff:.6f}\n"
        f"  CUDA dq range: [{dq_cuda_f32.min().item():.4f}, {dq_cuda_f32.max().item():.4f}]\n"
        f"  CuTe dq range: [{dq_cute_f32.min().item():.4f}, {dq_cute_f32.max().item():.4f}]"
    )

    # Verify high agreement between backends
    # For FP4 quantization, we expect >95% exact match due to minor rounding differences
    assert quant_match_pct > 95.0, (
        f"Quantized values should match >95%, got {quant_match_pct:.1f}%"
    )
    assert scale_match_pct > 95.0, (
        f"Scale factors should match >95%, got {scale_match_pct:.1f}%"
    )

    torch.testing.assert_close(
        dq_cuda_f32,
        dq_cute_f32,
        rtol=0,
        atol=0,
        msg=error_msg,
    )


MXFP4_SF_LAYOUTS = [
    SfLayout.layout_128x4,
    SfLayout.layout_8x4,
    SfLayout.layout_linear,
]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", MXFP4_SHAPES)
@pytest.mark.parametrize("sf_layout", MXFP4_SF_LAYOUTS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_mxfp4_quantize_layout_backend_parity(
    dtype: torch.dtype,
    shape: tuple[int, int],
    sf_layout: SfLayout,
    device: str,
) -> None:
    """Test that CUDA and CuTe-DSL backends agree across MXFP4 SF layouts.

    Uses the low-level fp4_quantize API to exercise the sf_layout knob, since
    the high-level mxfp4_quantize() hardcodes 128x4 on both backends.
    """
    if not _is_fp4_supported(torch.device(device)):
        pytest.skip("Nvfp4 Requires compute capability >= 10 and CUDA >= 12.8")
    if not _is_cute_dsl_available():
        pytest.skip("CuTe-DSL not available")

    torch.set_default_device(device)
    torch.manual_seed(42)

    m, n = shape
    x = torch.randn((m, n), dtype=dtype)

    is_sf_swizzled_layout = sf_layout != SfLayout.layout_linear
    is_sf_8x4_layout = sf_layout == SfLayout.layout_8x4

    # MXFP4 uses sf_use_ue8m0=True and sf_vec_size=32. global_scale is unused
    # in the MXFP4 path but the kernel signature still expects a value.
    global_scale = torch.tensor([1.0], dtype=torch.float32, device=device)

    quant_cuda, scale_cuda = fp4_quantize(
        x,
        global_scale,
        sf_vec_size=32,
        sf_use_ue8m0=True,
        is_sf_swizzled_layout=is_sf_swizzled_layout,
        is_sf_8x4_layout=is_sf_8x4_layout,
        backend="cuda",
    )
    quant_cute, scale_cute = fp4_quantize(
        x,
        global_scale,
        sf_vec_size=32,
        sf_use_ue8m0=True,
        is_sf_swizzled_layout=is_sf_swizzled_layout,
        is_sf_8x4_layout=is_sf_8x4_layout,
        backend="cute-dsl",
    )

    assert quant_cuda.shape == quant_cute.shape, (
        f"Quantized output shape mismatch for {sf_layout.name}"
    )
    # Scale buffers may have different physical sizes (cute-dsl pads M to its
    # row-tile size; CUDA returns the unpadded length). Compare the leading
    # CUDA-sized prefix, which covers the valid data.
    n_compare = min(scale_cuda.numel(), scale_cute.numel())

    quant_match_pct = (quant_cuda == quant_cute).float().mean().item() * 100
    scale_match_pct = (
        scale_cuda.flatten()[:n_compare] == scale_cute.flatten()[:n_compare]
    ).float().mean().item() * 100
    assert quant_match_pct > 95.0, (
        f"Quantized values should match >95%, got {quant_match_pct:.1f}% "
        f"(layout={sf_layout.name})"
    )
    assert scale_match_pct > 95.0, (
        f"Scale factors should match >95%, got {scale_match_pct:.1f}% "
        f"(layout={sf_layout.name})"
    )


# =============================================================================
# NVFP4 Quantization Tests (Both Backends)
# =============================================================================

NVFP4_SHAPES = [
    # K must be a multiple of 64 so K/16 is a multiple of 4 (CUDA reshape
    # constraint for swizzled layout).
    # Small M with swizzled layout: padded_M >> M (row padding dominance)
    (1, 64),  # padded_M=128, 127 padding rows
    (1, 1024),  # padded_M=128, large K
    (3, 128),  # padded_M=128 (128x4) or 8 (8x4), odd M
    (16, 64),  # padded_M=128, 112 padding rows
    (64, 128),  # padded_M=128, 64 padding rows
    # Standard sizes
    (128, 64),
    (256, 128),
    (512, 256),
    (128, 1024),
    (1024, 2048),
    # Large K (column loop path in swizzled kernel)
    (128, 16384),
]
NVFP4_BACKENDS = ["cuda", "cute-dsl"]
NVFP4_SF_LAYOUTS = [SfLayout.layout_128x4, SfLayout.layout_8x4, SfLayout.layout_linear]
# Roundtrip test only for layouts the dequantizer supports (128x4 and linear)
NVFP4_ROUNDTRIP_SF_LAYOUTS = [SfLayout.layout_128x4, SfLayout.layout_linear]


@dataclass(frozen=True, kw_only=True)
class NVFP44Over6TestConfig(NVFP44Over6Config):
    id: str


NVFP4_TE_REFERENCE_CONFIGS = [
    None,
    NVFP44Over6TestConfig(
        id="4over6-mae-e4m3-448-exact",
        e4m3_max=448,
        err_mode="MAE",
    ),
    NVFP44Over6TestConfig(
        id="4over6-mae-e4m3-448-fp16",
        e4m3_max=448,
        err_mode="MAE",
        err_use_fast_math=True,
    ),
    NVFP44Over6TestConfig(
        id="4over6-mae-e4m3-256-exact",
        e4m3_max=256,
        err_mode="MAE",
    ),
    NVFP44Over6TestConfig(
        id="4over6-mae-e4m3-256-fp16",
        e4m3_max=256,
        err_mode="MAE",
        err_use_fast_math=True,
    ),
    NVFP44Over6TestConfig(
        id="4over6-mse-e4m3-448-exact",
        e4m3_max=448,
        err_mode="MSE",
    ),
    NVFP44Over6TestConfig(
        id="4over6-mse-e4m3-448-fp16",
        e4m3_max=448,
        err_mode="MSE",
        err_use_fast_math=True,
    ),
    NVFP44Over6TestConfig(
        id="4over6-mse-e4m3-256-exact",
        e4m3_max=256,
        err_mode="MSE",
    ),
    NVFP44Over6TestConfig(
        id="4over6-mse-e4m3-256-fp16",
        e4m3_max=256,
        err_mode="MSE",
        err_use_fast_math=True,
    ),
]
NVFP4_DEFAULT_4OVER6_CONFIGS = [
    None,
    NVFP44Over6TestConfig(
        id="4over6-mae-e4m3-448",
        e4m3_max=448,
        err_mode="MAE",
        err_use_fast_math=False,
    ),
]


def _te_ref_scale_bytes_for_layout(
    scale_ref: torch.Tensor,
    sf_layout: SfLayout,
) -> torch.Tensor:
    scale_ref = scale_ref.view(torch.uint8)

    if sf_layout == SfLayout.layout_linear:
        return scale_ref
    if sf_layout == SfLayout.layout_128x4:
        rows = ((scale_ref.shape[0] + 127) // 128) * 128
        cols = ((scale_ref.shape[1] + 3) // 4) * 4
        return block_scale_interleave(scale_ref).reshape(rows, cols)
    if sf_layout == SfLayout.layout_8x4:
        rows = ((scale_ref.shape[0] + 7) // 8) * 8
        cols = ((scale_ref.shape[1] + 3) // 4) * 4
        # Vectorized 8x4 swizzle: flat_offset =
        #   m_tile * (cols//4) * 32 + k_tile * 32 + inner_m * 4 + inner_k
        row_idx = torch.arange(scale_ref.shape[0], device=scale_ref.device).unsqueeze(1)
        col_idx = torch.arange(scale_ref.shape[1], device=scale_ref.device).unsqueeze(0)
        flat_offset = (
            (row_idx // 8) * (cols // 4) * 32
            + (col_idx // 4) * 32
            + (row_idx % 8) * 4
            + col_idx % 4
        )
        expected = torch.zeros(
            rows * cols,
            dtype=torch.uint8,
            device=scale_ref.device,
        )
        expected[flat_offset.reshape(-1)] = scale_ref.reshape(-1)
        return expected.view(rows, cols)
    raise ValueError(f"Unknown scale-factor layout: {sf_layout}")


def _te_ref_fp4_bytes(q_ref: torch.Tensor) -> torch.Tensor:
    q_abs = torch.abs(q_ref)
    q_code = torch.zeros_like(q_abs, dtype=torch.uint8)
    q_code[q_abs == 0.0] = 0
    q_code[q_abs == 0.5] = 1
    q_code[q_abs == 1.0] = 2
    q_code[q_abs == 1.5] = 3
    q_code[q_abs == 2.0] = 4
    q_code[q_abs == 3.0] = 5
    q_code[q_abs == 4.0] = 6
    q_code[q_abs == 6.0] = 7
    q_code = q_code | (torch.signbit(q_ref).to(torch.uint8) << 3)
    q_pair = q_code.reshape(q_ref.shape[0], q_ref.shape[1] // 2, 2)
    return q_pair.select(-1, 0) | (q_pair.select(-1, 1) << 4)


@pytest.fixture(autouse=True)
def set_nvfp4_quant_env():
    """Set NVFP4 quantization env vars for one test."""
    env_names = NVFP4_QUANT_ENV_VARS
    original_values = {name: os.environ.get(name, None) for name in env_names}

    def _set_bool_env(name: str, value: bool | None):
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = "1" if value else "0"

    def _set_str_env(name: str, value: str | None):
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value

    def _set_env(
        nvfp4_4over6_config: NVFP44Over6TestConfig | None = None,
        disable_quant_fast_math: bool | None = None,
    ):
        use_4over6 = nvfp4_4over6_config is not None
        nvfp4_4over6_err_mode = None
        nvfp4_4over6_err_use_fast_math = None
        e4m3_max_is_256 = None
        if nvfp4_4over6_config is not None:
            nvfp4_4over6_err_mode = nvfp4_4over6_config.err_mode_name
            nvfp4_4over6_err_use_fast_math = nvfp4_4over6_config.err_use_fast_math
            e4m3_max_is_256 = nvfp4_4over6_config.e4m3_max == 256
        _set_bool_env("FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH", disable_quant_fast_math)
        _set_bool_env("FLASHINFER_NVFP4_4OVER6", use_4over6)
        _set_str_env("FLASHINFER_NVFP4_4OVER6_ERR_MODE", nvfp4_4over6_err_mode)
        _set_bool_env(
            "FLASHINFER_NVFP4_4OVER6_ERR_USE_FAST_MATH",
            nvfp4_4over6_err_use_fast_math,
        )
        _set_bool_env("FLASHINFER_NVFP4_4OVER6_E4M3_USE_256", e4m3_max_is_256)

    _set_env()
    yield _set_env

    for name, value in original_values.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", NVFP4_SHAPES)
@pytest.mark.parametrize("sf_layout", NVFP4_SF_LAYOUTS)
@pytest.mark.parametrize("init_data", ["random", "boundary", "zeros", "maxes"])
@pytest.mark.parametrize("per_token_activation", [False, True])
@pytest.mark.parametrize(
    "nvfp4_4over6_config",
    NVFP4_TE_REFERENCE_CONFIGS,
    ids=lambda config: "nvfp4" if config is None else config.id,
)
@pytest.mark.parametrize("backend", NVFP4_BACKENDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_nvfp4_quantize_te_reference(
    backend: str,
    dtype: torch.dtype,
    shape: tuple[int, int],
    sf_layout: SfLayout,
    init_data: str,
    per_token_activation: bool,
    nvfp4_4over6_config: NVFP44Over6TestConfig | None,
    device: str,
    set_nvfp4_quant_env,
) -> None:
    """NVFP4 quantization should match the Python reference bitwise."""
    if not _is_fp4_supported(torch.device(device)):
        pytest.skip("Nvfp4 Requires compute capability >= 10 and CUDA >= 12.8")
    if backend == "cute-dsl" and not _is_cute_dsl_available():
        pytest.skip("CuTe-DSL not available")

    torch.set_default_device(device)
    torch.manual_seed(42)

    m, n = shape
    if init_data == "random":
        x = torch.randn((m, n), dtype=dtype)
        if m > 1:
            x[0].zero_()
    elif init_data == "boundary":
        base = torch.linspace(-12.0, 12.0, steps=n // 2, dtype=torch.float32)
        eps = torch.full_like(base, 1e-3)
        eps = torch.maximum(eps, 1e-4 * torch.ones_like(base))
        row = torch.empty(n, dtype=torch.float32)
        row[0::2] = base - eps
        row[1::2] = base + eps
        x = row.unsqueeze(0).repeat(m, 1).to(dtype=dtype)
    elif init_data == "zeros":
        x = torch.zeros((m, n), dtype=dtype)
    elif init_data == "maxes":
        x = torch.full((m, n), torch.finfo(dtype).max, dtype=dtype)
    else:
        raise ValueError(f"Unknown init_data: {init_data}")

    expected_per_token_scale = None
    if per_token_activation:
        global_amax = torch.abs(x).max(dim=1).values.to(torch.float32)
        expected_per_token_scale = torch.where(
            global_amax == 0,
            torch.zeros_like(global_amax),
            nvfp4_global_decode_scale_te(global_amax, nvfp4_4over6_config),
        )
        per_token_global_scale_inv = nvfp4_global_decode_scale_te(
            torch.ones((), dtype=torch.float32, device=x.device),
            nvfp4_4over6_config,
        )
    else:
        global_amax = torch.abs(x).max().to(torch.float32)
        global_scale = nvfp4_global_encode_scale_te(
            global_amax,
            nvfp4_4over6_config,
        )

    def _run_quantize(expected_per_token_scale=None):
        if per_token_activation:
            q_out, scale_out, per_token_scale = nvfp4_quantize(
                x,
                per_token_global_scale_inv,
                sfLayout=sf_layout,
                per_token_activation=True,
                backend=backend,
            )
            if expected_per_token_scale is not None:
                torch.testing.assert_close(
                    per_token_scale,
                    expected_per_token_scale,
                    rtol=0,
                    atol=0,
                )
        else:
            q_out, scale_out = nvfp4_quantize(
                x,
                global_scale,
                sfLayout=sf_layout,
                backend=backend,
            )
        return q_out, scale_out

    if nvfp4_4over6_config is not None:
        q_ref, scale_ref, expected_per_token_scale, _ = ref_fp4_quant_4over6_te(
            x,
            global_amax,
            per_token_rowwise=per_token_activation,
            nvfp4_4over6_config=nvfp4_4over6_config,
        )
        expected_scale = _te_ref_scale_bytes_for_layout(scale_ref, sf_layout)
    else:
        q_ref, scale_ref = ref_fp4_quant_te(
            x,
            global_amax,
            per_token_rowwise=per_token_activation,
        )
        expected_scale = _te_ref_scale_bytes_for_layout(scale_ref, sf_layout)

    set_nvfp4_quant_env(
        nvfp4_4over6_config=nvfp4_4over6_config, disable_quant_fast_math=True
    )
    q_out, scale_out = _run_quantize(expected_per_token_scale)
    torch.testing.assert_close(q_out, _te_ref_fp4_bytes(q_ref), rtol=0, atol=0)
    torch.testing.assert_close(scale_out, expected_scale, rtol=0, atol=0)


@pytest.mark.parametrize("backend", NVFP4_BACKENDS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", NVFP4_SHAPES)
@pytest.mark.parametrize("sf_layout", NVFP4_ROUNDTRIP_SF_LAYOUTS)
@pytest.mark.parametrize("per_token_activation", [False, True])
@pytest.mark.parametrize(
    "nvfp4_4over6_config",
    NVFP4_DEFAULT_4OVER6_CONFIGS,
    ids=lambda config: "nvfp4" if config is None else config.id,
)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_nvfp4_quantize_roundtrip(
    backend: str,
    dtype: torch.dtype,
    shape: tuple[int, int],
    sf_layout: SfLayout,
    per_token_activation: bool,
    nvfp4_4over6_config: NVFP44Over6TestConfig | None,
    device: str,
    set_nvfp4_quant_env,
) -> None:
    """Test NVFP4 quantization roundtrip for both backends and layouts."""
    if not _is_fp4_supported(torch.device(device)):
        pytest.skip("Nvfp4 Requires compute capability >= 10 and CUDA >= 12.8")
    if backend == "cute-dsl" and not _is_cute_dsl_available():
        pytest.skip("CuTe-DSL not available")

    set_nvfp4_quant_env(nvfp4_4over6_config=nvfp4_4over6_config)

    torch.set_default_device(device)
    torch.manual_seed(42)

    m, n = shape
    x = torch.randn((m, n), dtype=dtype)

    tensor_amax = torch.abs(x).max().to(torch.float32)
    global_scale = nvfp4_global_encode_scale_te(
        tensor_amax,
        nvfp4_4over6_config,
    )
    per_token_global_scale_inv = nvfp4_global_decode_scale_te(
        torch.ones((), dtype=torch.float32, device=device),
        nvfp4_4over6_config,
    )

    if per_token_activation:
        quant_out, scale_out, per_token_scale = nvfp4_quantize(
            x,
            per_token_global_scale_inv,
            sfLayout=sf_layout,
            backend=backend,
            per_token_activation=True,
        )
        dequant_global_scale = torch.ones(1, dtype=torch.float32, device=device)
    else:
        quant_out, scale_out = nvfp4_quantize(
            x, global_scale, sfLayout=sf_layout, backend=backend
        )
        dequant_global_scale = nvfp4_global_decode_scale_te(
            tensor_amax,
            nvfp4_4over6_config,
        )

    # Basic shape checks
    assert quant_out.shape == (m, n // 2), (
        f"Expected shape ({m}, {n // 2}), got {quant_out.shape}"
    )
    assert quant_out.dtype == torch.uint8, f"Expected uint8, got {quant_out.dtype}"
    assert scale_out.dtype == torch.uint8, f"Expected uint8, got {scale_out.dtype}"

    is_swizzled = sf_layout != SfLayout.layout_linear

    # Dequantize round-trip
    dq_out = e2m1_and_ufp8sf_scale_to_float(
        quant_out,
        scale_out,
        dequant_global_scale,
        sf_vec_size=16,
        ufp8_type=1,
        is_sf_swizzled_layout=is_swizzled,
    )
    dq_out = dq_out.to(device)
    if per_token_activation:
        dq_out *= per_token_scale.view(-1, 1)

    # Verify no NaN/Inf
    assert not torch.isnan(dq_out).any(), "Dequantized tensor contains NaN"
    assert not torch.isinf(dq_out).any(), "Dequantized tensor contains Inf"

    # Verify roundtrip is reasonably accurate
    torch.testing.assert_close(
        dq_out.to(torch.float32),
        x.to(torch.float32),
        rtol=0.3,
        atol=0.5,
        msg=f"{backend} {sf_layout.name} NVFP4 quantize -> dequantize roundtrip failed",
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", NVFP4_SHAPES)
@pytest.mark.parametrize("sf_layout", NVFP4_SF_LAYOUTS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_nvfp4_quantize_backend_parity(
    dtype: torch.dtype,
    shape: tuple[int, int],
    sf_layout: SfLayout,
    device: str,
) -> None:
    """Test that CUDA and CuTe-DSL backends produce matching results for NVFP4."""
    if not _is_fp4_supported(torch.device(device)):
        pytest.skip("Nvfp4 Requires compute capability >= 10 and CUDA >= 12.8")
    if not _is_cute_dsl_available():
        pytest.skip("CuTe-DSL not available")

    torch.set_default_device(device)
    torch.manual_seed(42)

    m, n = shape
    x = torch.randn((m, n), dtype=dtype)

    tensor_amax = torch.abs(x).max().to(torch.float32)
    global_scale = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / tensor_amax

    # Get results from both backends
    quant_cuda, scale_cuda = nvfp4_quantize(
        x, global_scale, sfLayout=sf_layout, backend="cuda"
    )
    quant_cute, scale_cute = nvfp4_quantize(
        x, global_scale, sfLayout=sf_layout, backend="cute-dsl"
    )

    # Shape should match
    assert quant_cuda.shape == quant_cute.shape, (
        f"Quantized output shape mismatch for {sf_layout.name}"
    )
    assert scale_cuda.shape == scale_cute.shape, (
        f"Scale output shape mismatch for {sf_layout.name}"
    )

    # Quantized FP4 values should match exactly (layout-independent)
    quant_match_pct = (quant_cuda == quant_cute).float().mean().item() * 100
    assert quant_match_pct > 95.0, (
        f"Quantized values should match >95%, got {quant_match_pct:.1f}% "
        f"(layout={sf_layout.name})"
    )

    # Scale factors should match exactly (layout-specific indexing)
    scale_match_pct = (scale_cuda == scale_cute).float().mean().item() * 100
    assert scale_match_pct > 95.0, (
        f"Scale factors should match >95%, got {scale_match_pct:.1f}% "
        f"(layout={sf_layout.name})"
    )

    # For layouts that support dequantization, also compare dequantized values
    is_swizzled = sf_layout != SfLayout.layout_linear
    can_dequantize = sf_layout in (SfLayout.layout_128x4, SfLayout.layout_linear)

    if can_dequantize:
        dq_cuda = (
            e2m1_and_ufp8sf_scale_to_float(
                quant_cuda,
                scale_cuda,
                1 / global_scale,
                sf_vec_size=16,
                ufp8_type=1,
                is_sf_swizzled_layout=is_swizzled,
            )
            .to(device)
            .to(torch.float32)
        )
        dq_cute = (
            e2m1_and_ufp8sf_scale_to_float(
                quant_cute,
                scale_cute,
                1 / global_scale,
                sf_vec_size=16,
                ufp8_type=1,
                is_sf_swizzled_layout=is_swizzled,
            )
            .to(device)
            .to(torch.float32)
        )

        abs_diff = (dq_cuda - dq_cute).abs()
        rel_diff = abs_diff / (dq_cuda.abs() + 1e-8)

        error_msg = (
            f"CUDA and CuTe-DSL backends differ after dequantization:\n"
            f"  Shape: {shape}, dtype: {dtype}, layout: {sf_layout.name}\n"
            f"  Quantized match: {quant_match_pct:.1f}%, Scale match: {scale_match_pct:.1f}%\n"
            f"  Abs diff - max: {abs_diff.max().item():.6f}, mean: {abs_diff.mean().item():.6f}\n"
            f"  Rel diff - max: {rel_diff.max().item():.6f}, mean: {rel_diff.mean().item():.6f}\n"
            f"  CUDA dq range: [{dq_cuda.min().item():.4f}, {dq_cuda.max().item():.4f}]\n"
            f"  CuTe dq range: [{dq_cute.min().item():.4f}, {dq_cute.max().item():.4f}]"
        )

        torch.testing.assert_close(
            dq_cuda,
            dq_cute,
            rtol=0,
            atol=0,
            msg=error_msg,
        )


NVFP4_FP8_SHAPES = [(128, 64), (256, 128), (512, 256), (128, 1024)]


@pytest.mark.parametrize("shape", NVFP4_FP8_SHAPES)
@pytest.mark.parametrize("sf_layout", NVFP4_SF_LAYOUTS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_nvfp4_quantize_fp8_input_cute_dsl(
    shape: tuple[int, int],
    sf_layout: SfLayout,
    device: str,
) -> None:
    """Test CuTe-DSL NVFP4 quantization with FP8 E4M3 input."""
    if not _is_fp4_supported(torch.device(device)):
        pytest.skip("Nvfp4 Requires compute capability >= 10 and CUDA >= 12.8")
    if not _is_cute_dsl_available():
        pytest.skip("CuTe-DSL not available")

    torch.set_default_device(device)
    torch.manual_seed(42)

    m, n = shape
    x_fp32 = torch.randn((m, n), dtype=torch.float32)
    x_fp8 = x_fp32.to(torch.float8_e4m3fn)

    tensor_amax = torch.abs(x_fp8.float()).max().to(torch.float32)
    global_scale = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / tensor_amax

    quant_out, scale_out = nvfp4_quantize(
        x_fp8, global_scale, sfLayout=sf_layout, backend="cute-dsl"
    )

    assert quant_out.shape == (m, n // 2), (
        f"Expected shape ({m}, {n // 2}), got {quant_out.shape}"
    )
    assert quant_out.dtype == torch.uint8, f"Expected uint8, got {quant_out.dtype}"
    assert scale_out.dtype == torch.uint8, f"Expected uint8, got {scale_out.dtype}"

    assert not torch.all(quant_out == 0), "All quantized values are zero"
    assert not torch.all(scale_out == 0), "All scale factors are zero"

    is_swizzled = sf_layout != SfLayout.layout_linear
    can_dequantize = sf_layout in (SfLayout.layout_128x4, SfLayout.layout_linear)

    if can_dequantize:
        dq_out = (
            e2m1_and_ufp8sf_scale_to_float(
                quant_out,
                scale_out,
                1 / global_scale,
                sf_vec_size=16,
                ufp8_type=1,
                is_sf_swizzled_layout=is_swizzled,
            )
            .to(device)
            .to(torch.float32)
        )
        assert not torch.isnan(dq_out).any(), "Dequantized tensor contains NaN"
        assert not torch.isinf(dq_out).any(), "Dequantized tensor contains Inf"

        # The FP8→FP4 path (matching CUDA cvt_warp_fp8_to_fp4) pre-scales input
        # by 6/global_scale before quantization. Standard dequant (e2m1 * sf / gs)
        # therefore reconstructs x_fp8 * 6/gs, not x_fp8.
        expected = x_fp8.float() * (6.0 / global_scale.item())
        torch.testing.assert_close(
            dq_out,
            expected,
            rtol=0.3,
            atol=0.5,
            msg=f"CuTe-DSL FP8 input NVFP4 roundtrip failed (layout={sf_layout.name})",
        )


@pytest.mark.parametrize("shape", NVFP4_FP8_SHAPES)
@pytest.mark.parametrize("sf_layout", NVFP4_SF_LAYOUTS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_nvfp4_quantize_fp8_backend_parity(
    shape: tuple[int, int],
    sf_layout: SfLayout,
    device: str,
) -> None:
    """Test CUDA and CuTe-DSL backends produce matching results for FP8 input."""
    if not _is_fp4_supported(torch.device(device)):
        pytest.skip("Nvfp4 Requires compute capability >= 10 and CUDA >= 12.8")
    if not _is_cute_dsl_available():
        pytest.skip("CuTe-DSL not available")

    torch.set_default_device(device)
    torch.manual_seed(42)

    m, n = shape
    x_fp32 = torch.randn((m, n), dtype=torch.float32)
    x_fp8 = x_fp32.to(torch.float8_e4m3fn)

    tensor_amax = torch.abs(x_fp8.float()).max().to(torch.float32)
    global_scale = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / tensor_amax

    quant_cuda, scale_cuda = nvfp4_quantize(
        x_fp8, global_scale, sfLayout=sf_layout, backend="cuda"
    )
    quant_cute, scale_cute = nvfp4_quantize(
        x_fp8, global_scale, sfLayout=sf_layout, backend="cute-dsl"
    )

    assert quant_cuda.shape == quant_cute.shape, (
        f"Quantized output shape mismatch for FP8 input, {sf_layout.name}"
    )
    assert scale_cuda.shape == scale_cute.shape, (
        f"Scale output shape mismatch for FP8 input, {sf_layout.name}"
    )

    quant_match_pct = (quant_cuda == quant_cute).float().mean().item() * 100
    assert quant_match_pct > 95.0, (
        f"FP8 quantized values should match >95%, got {quant_match_pct:.1f}% "
        f"(layout={sf_layout.name})"
    )

    scale_match_pct = (scale_cuda == scale_cute).float().mean().item() * 100
    assert scale_match_pct > 95.0, (
        f"FP8 scale factors should match >95%, got {scale_match_pct:.1f}% "
        f"(layout={sf_layout.name})"
    )


# =============================================================================
# NVFP4 TMA Kernel Tests
# =============================================================================

NVFP4_TMA_SHAPES = [
    # Shapes that trigger TMA: log2(M)+log2(K) >= 25 and K % 512 == 0
    (4096, 8192),  # log2sum=25, smallest TMA case
    (8192, 4096),  # log2sum=25
    (16384, 2048),  # log2sum=25
    (32768, 1024),  # log2sum=25
    (16384, 4096),  # log2sum=26
    (8192, 8192),  # log2sum=26
]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", NVFP4_TMA_SHAPES)
@pytest.mark.parametrize("sf_layout", NVFP4_SF_LAYOUTS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_nvfp4_quantize_tma_backend_parity(
    dtype: torch.dtype,
    shape: tuple[int, int],
    sf_layout: SfLayout,
    device: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that TMA-based CuTe-DSL kernel matches the CUDA backend for large problems."""
    if not _is_fp4_supported(torch.device(device)):
        pytest.skip("Nvfp4 Requires compute capability >= 10 and CUDA >= 12.8")
    if not _is_cute_dsl_available():
        pytest.skip("CuTe-DSL not available")

    # TMA is disabled by default (flashinfer#3905); force it on so this test
    # still exercises the CuTe-DSL TMA kernel.
    monkeypatch.setenv("FLASHINFER_NVFP4_QUANTIZE_USE_TMA", "1")

    torch.set_default_device(device)
    torch.manual_seed(42)

    m, n = shape
    x = torch.randn((m, n), dtype=dtype)

    tensor_amax = torch.abs(x).max().to(torch.float32)
    global_scale = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / tensor_amax

    quant_cuda, scale_cuda = nvfp4_quantize(
        x, global_scale, sfLayout=sf_layout, backend="cuda"
    )
    quant_cute, scale_cute = nvfp4_quantize(
        x, global_scale, sfLayout=sf_layout, backend="cute-dsl"
    )

    assert quant_cuda.shape == quant_cute.shape, (
        f"TMA quantized output shape mismatch for {sf_layout.name}"
    )
    assert scale_cuda.shape == scale_cute.shape, (
        f"TMA scale output shape mismatch for {sf_layout.name}"
    )

    quant_match_pct = (quant_cuda == quant_cute).float().mean().item() * 100
    assert quant_match_pct > 95.0, (
        f"TMA quantized values should match >95%, got {quant_match_pct:.1f}% "
        f"(shape={shape}, layout={sf_layout.name})"
    )

    scale_match_pct = (scale_cuda == scale_cute).float().mean().item() * 100
    assert scale_match_pct > 95.0, (
        f"TMA scale factors should match >95%, got {scale_match_pct:.1f}% "
        f"(shape={shape}, layout={sf_layout.name})"
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("batch_shape", BATCH_SHAPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_nvfp4_batched_quantize(
    dtype: torch.dtype,
    batch_shape: tuple[int, int, int],
    seed: int,
    device: str,
) -> None:
    """Test nvfp4_batched_quantize function."""
    if not _is_fp4_supported(torch.device(device)):
        pytest.skip("Nvfp4 Requires compute capability of 10 or above")
    torch.set_default_device(device)
    torch.manual_seed(seed)

    b, m, n = batch_shape
    x = torch.randn(batch_shape, dtype=dtype)
    tensor_amax = torch.abs(x).max().to(torch.float32)
    global_scale = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / tensor_amax

    # Test the batched quantization
    out, out_scale = nvfp4_batched_quantize(x, global_scale)

    # Basic shape checks
    assert out.shape == (
        b,
        m,
        n // 2,
    ), f"Expected shape {(b, m, n // 2)}, got {out.shape}"
    assert out.dtype == torch.uint8, f"Expected uint8, got {out.dtype}"
    assert out_scale.dtype == torch.uint8, f"Expected uint8, got {out_scale.dtype}"

    # Compare with single tensor quantization for each batch
    for i in range(b):
        single_out, single_scale = fp4_quantize(x[i], global_scale, 16, False, True)
        torch.testing.assert_close(out[i], single_out, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(
            out_scale[i], single_scale.flatten(), rtol=1e-5, atol=1e-5
        )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("batch_shape", BATCH_SHAPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_scaled_fp4_grouped_quantize(
    dtype: torch.dtype,
    batch_shape: tuple[int, int, int],
    seed: int,
    device: str,
) -> None:
    """Test scaled_fp4_grouped_quantize function."""
    if not _is_fp4_supported(torch.device(device)):
        pytest.skip("Nvfp4 Requires compute capability of 10 or above")
    torch.set_default_device(device)
    torch.manual_seed(seed)

    b, m, n = batch_shape
    x = torch.randn(batch_shape, dtype=dtype)
    tensor_amax = torch.abs(x).amax(dim=(1, 2)).to(torch.float32)
    global_scale = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / tensor_amax
    mask = torch.randint(low=1, high=m + 1, size=(b,), dtype=torch.int32, device=device)
    out, out_scale = scaled_fp4_grouped_quantize(x, mask, global_scale)
    out = out.permute(2, 0, 1)
    out_scale = out_scale.permute(5, 2, 4, 0, 1, 3)
    # Basic shape checks
    assert out.shape == (
        b,
        m,
        n // 2,
    ), f"Expected shape {(b, m, n // 2)}, got {out.shape}"
    assert out.dtype == torch.uint8, f"Expected uint8, got {out.dtype}"
    assert out_scale.dtype == torch.float8_e4m3fn, (
        f"Expected uint8, got {out_scale.dtype}"
    )

    # Compare with single tensor quantization for each batch
    for i in range(b):
        single_out, single_scale = fp4_quantize(x[i], global_scale[i], 16, False, True)
        torch.testing.assert_close(
            out[i][: mask[i]], single_out[: mask[i]], rtol=1e-5, atol=1e-5
        )
        scale_ref = unswizzle_sf(single_scale.view(torch.float8_e4m3fn), m, n)
        scale_ans = unswizzle_sf(out_scale[i], m, n)
        torch.testing.assert_close(scale_ref[: mask[i]], scale_ans[: mask[i]])


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("batch_shape", BATCH_SHAPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize(
    "nvfp4_4over6_config",
    NVFP4_DEFAULT_4OVER6_CONFIGS,
    ids=lambda config: "nvfp4" if config is None else config.id,
)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_silu_and_mul_scaled_nvfp4_experts_quantize(
    dtype: torch.dtype,
    batch_shape: tuple[int, int, int],
    seed: int,
    nvfp4_4over6_config: NVFP44Over6TestConfig | None,
    device: str,
    set_nvfp4_quant_env,
) -> None:
    """Test silu_and_mul_nvfp4_batched_quantize function."""
    if not _is_fp4_supported(torch.device(device)):
        pytest.skip("Nvfp4 Requires compute capability of 10 or above")
    set_nvfp4_quant_env(nvfp4_4over6_config=nvfp4_4over6_config)
    torch.set_default_device(device)
    torch.manual_seed(seed)

    b, m, n = batch_shape
    x = torch.randn((b, m, n * 2), dtype=dtype)
    mask = torch.randint(low=1, high=m + 1, size=(b,), dtype=torch.int32, device=device)
    ref_y = silu_and_mul(x)

    tensor_amax = ref_y.abs().amax(dim=(1, 2)).to(torch.float32)
    global_scale = nvfp4_global_encode_scale_te(
        tensor_amax,
        nvfp4_4over6_config,
    )

    out, out_scale = silu_and_mul_scaled_nvfp4_experts_quantize(x, mask, global_scale)

    # Basic shape checks
    out = out.permute(2, 0, 1)
    out_scale = out_scale.permute(5, 2, 4, 0, 1, 3)
    assert out.shape == (b, m, n // 2), f"Expected shape {(b, m, n)}, got {out.shape}"
    assert out.dtype == torch.uint8, f"Expected uint8, got {out.dtype}"
    assert out_scale.dtype == torch.float8_e4m3fn, (
        f"Expected uint8, got {out_scale.dtype}"
    )

    # Compare with single tensor quantization for each batch
    for i in range(b):
        x_silu_mul = silu_and_mul(x[i])
        single_out, single_scale = fp4_quantize(
            x_silu_mul, global_scale[i], 16, False, True
        )
        torch.testing.assert_close(
            out[i][: mask[i]], single_out[: mask[i]], rtol=1e-5, atol=1e-5
        )

        scale_ref = unswizzle_sf(single_scale.view(torch.float8_e4m3fn), m, n)
        scale_ans = unswizzle_sf(out_scale[i], m, n)
        torch.testing.assert_close(scale_ref[: mask[i]], scale_ans[: mask[i]])


@pytest.mark.parametrize("m", [128, 256, 384, 512, 1024, 1152, 2048])
@pytest.mark.parametrize("scale_dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_nvfp4_quantize_global_scale_dtype_regression(m: int, scale_dtype: torch.dtype):
    """Regression for issue #3398.

    The CUDA nvfp4 quantize kernel reads `global_scale` as float32. A bf16/fp16
    global scale (e.g. ``(448*6) / x.abs().max()`` inherits x's bf16 dtype) used to
    be misread byte-wise, silently producing all-zero scale factors for a range of
    M (e.g. 384..1024 at K=2048) and under-scaled values elsewhere. The Python
    wrapper now normalizes the global scale to float32, so every dtype must produce
    valid, correctly-scaled scale factors.

    Guards against cosine-only checks: we assert (a) scale factors are NOT all-zero
    and (b) the dequant magnitude matches the input (catches the under-scaling that
    a cosine-only test misses).
    """
    device = torch.device("cuda")
    if not _is_fp4_supported(device):
        pytest.skip("Nvfp4 requires compute capability >= 10 and CUDA >= 12.8")
    torch.manual_seed(0)
    k = 2048
    x = torch.randn(m, k, device=device, dtype=torch.bfloat16) * 5
    # global scale deliberately built in `scale_dtype` (bf16 reproduces the bug)
    global_scale = ((448.0 * 6.0) / x.abs().max()).to(scale_dtype)

    fp4, sf = nvfp4_quantize(
        x,
        global_scale,
        sfLayout=SfLayout.layout_128x4,
        do_shuffle=False,
        backend="cuda",
    )
    sf_u8 = sf.view(torch.uint8).reshape(-1)
    assert (sf_u8 != 0).any(), (
        f"All scale-factor bytes are zero for m={m}, scale_dtype={scale_dtype} "
        "(issue #3398: global scale misread)."
    )

    # Round-trip and check magnitude (not just direction).
    deq = e2m1_and_ufp8sf_scale_to_float(
        fp4.cpu(),
        sf_u8.cpu(),
        (1.0 / global_scale.float()).reshape(1).cpu(),
        sf_vec_size=16,
        ufp8_type=1,
        is_sf_swizzled_layout=True,
    ).to(device)
    xf = x.float()
    mask = xf.abs() > 0.5
    ratio = (deq[mask].abs() / xf[mask].abs()).median().item()
    assert 0.5 < ratio < 2.0, (
        f"Dequant magnitude off (median |deq/x|={ratio:.3f}) for m={m}, "
        f"scale_dtype={scale_dtype} -- global scale not applied correctly."
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
