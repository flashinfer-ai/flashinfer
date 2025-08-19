import functools

import pytest
import torch
from utils_fp4 import cast_from_fp4, recover_swizzled_scales, ref_fp4_quant

from flashinfer import (
    block_scale_interleave,
    e2m1_and_ufp8sf_scale_to_float,
    fp4_quantize,
    mxfp4_quantize,
    mxfp4_dequantize,
)
from flashinfer.utils import is_sm100a_supported

DTYPES = [torch.float16, torch.bfloat16]
# The batch dimension doesn't need to be multiple of 128
SHAPES = [(128, 64), (256, 128), (120, 64), (200, 256)]
SEEDS = [42]
CUDA_DEVICES = ["cuda:0"]

FLOAT4_E2M1_MAX = 6.0
FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max

BLOCK_SIZE = 16


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


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("sf_use_ue8m0", [False, True])
@pytest.mark.parametrize("is_swizzled", [False, True])
@torch.inference_mode()
def test_fp4_quantization(
    dtype: torch.dtype,
    shape: tuple[int, int],
    seed: int,
    device: str,
    sf_use_ue8m0: bool,
    is_swizzled: bool,
) -> None:
    if not is_sm100a_supported(torch.device(device)):
        pytest.skip("Nvfp4 Requires compute capability of 10 or above")
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
        x, global_scale, sf_vec_size, sf_use_ue8m0, is_swizzled
    )
    assert n % sf_vec_size == 0, f"cols needs to be {sf_vec_size} divisible"
    if sf_use_ue8m0:
        out_scale = (out_scale.to(torch.int32) << 23).view(torch.float32)
    else:
        out_scale = out_scale.view(torch.float8_e4m3fn).to(torch.float32)
    if is_swizzled:
        scale_ans = recover_swizzled_scales(
            out_scale.reshape(-1, n // sf_vec_size),
            m,
            n,
            sf_vec_size,
        )
    else:
        scale_ans = out_scale
    out_ans = cast_from_fp4(out).reshape(m, n)
    torch.testing.assert_close(out_ans, out_ref, rtol=1e0, atol=1e-1)
    torch.testing.assert_close(scale_ans, scale_ref, rtol=1e-1, atol=1e-1)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_scale_swizzling(
    dtype: torch.dtype,
    shape: tuple[int, int],
    seed: int,
    device: str,
) -> None:
    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("Nvfp4 Requires compute capability of 10 or above")
    torch.set_default_device(device)
    torch.manual_seed(seed)
    m, n = shape
    x = torch.randn((m, n), dtype=dtype)
    tensor_amax = torch.abs(x).max().to(torch.float32)
    global_scale = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / tensor_amax

    _, unswizzled_scale = fp4_quantize(x, global_scale, BLOCK_SIZE, False, False)
    _, swizzled_scale = fp4_quantize(x, global_scale, BLOCK_SIZE, False, True)
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
    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("Nvfp4 Requires compute capability of 10 or above")
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


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("sf_use_ue8m0", [True, False])
@torch.inference_mode()
def test_e2m1_dequantization(
    shape: tuple[int, int],
    seed: int,
    device: str,
    sf_use_ue8m0: bool,
) -> None:
    """Test roundtrip: fp4_quantize -> e2m1_and_ufp8sf_scale_to_float."""
    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("Nvfp4 Requires compute capability of 10 or above")
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
        x, global_scale, block_size, sf_use_ue8m0, is_sf_swizzled_layout
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


def test_mxfp4_quantize_roundtrip():
    x = torch.randn((128, 64), device="cuda", dtype=torch.bfloat16) / 10

    quant_a, sfs = mxfp4_quantize(x)
    dq_a = mxfp4_dequantize(quant_a, sfs)

    torch.testing.assert_close(
        dq_a.cpu().to(torch.float32),
        x.cpu().to(torch.float32),
        rtol=0.3,
        atol=0.5,
        msg="Quantize -> dequantize mxfp4 roundtrip failed",
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
