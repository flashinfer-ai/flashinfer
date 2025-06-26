import functools

import pytest
import torch

from flashinfer import fp4_quantize, fp4_swizzle_blockscale
from flashinfer.utils import is_sm100a_supported

DTYPES = [torch.float16, torch.bfloat16]
# The batch dimension doesn't need to be multiple of 128
SHAPES = [(128, 64), (256, 128), (120, 64), (200, 256)]
SEEDS = [42]
CUDA_DEVICES = ["cuda:0"]

FLOAT4_E2M1_MAX = 6.0
FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max

# E2M1 to float
# 0111 -> 6
# 0110 -> 4
# 0101 -> 3
# 0100 -> 2
# 0011 -> 1.5
# 0010 -> 1
# 0001 -> 0.5
# 0000 -> 0
E2M1_TO_FLOAT32 = [
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
]
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


def cast_from_fp4(x, m, n):
    # The fp4 values are packed in uint8 as [v_1st | v_2nd]
    v_2nd = x & 0xF
    v_1st = (x >> 4) & 0xF
    c = torch.stack((v_2nd, v_1st), dim=-1)
    out = torch.tensor([E2M1_TO_FLOAT32[x] for x in c.flatten()])
    out = out.reshape(m, n).to(torch.float32)
    return out


def cast_to_fp4(x):
    sign = torch.sign(x)
    x = torch.abs(x)
    x[(x >= 0.0) & (x <= 0.25)] = 0.0
    x[(x > 0.25) & (x < 0.75)] = 0.5
    x[(x >= 0.75) & (x <= 1.25)] = 1.0
    x[(x > 1.25) & (x < 1.75)] = 1.5
    x[(x >= 1.75) & (x <= 2.5)] = 2.0
    x[(x > 2.5) & (x < 3.5)] = 3.0
    x[(x >= 3.5) & (x <= 5.0)] = 4.0
    x[x > 5.0] = 6.0
    return x * sign


def get_reciprocal(x):
    if isinstance(x, torch.Tensor):
        return torch.where(x == 0, torch.tensor(0.0, dtype=x.dtype), 1.0 / x)
    elif isinstance(x, (float, int)):
        return 0.0 if x == 0 else 1.0 / x
    else:
        raise TypeError("Input must be a float, int, or a torch.Tensor.")


def ref_nvfp4_quant(x, global_scale):
    assert global_scale.dtype == torch.float32
    assert x.ndim == 2
    m, n = x.shape
    x = torch.reshape(x, (m, n // BLOCK_SIZE, BLOCK_SIZE))
    vec_max = torch.max(torch.abs(x), dim=-1, keepdim=True)[0].to(torch.float32)
    scale = global_scale * (vec_max * get_reciprocal(FLOAT4_E2M1_MAX))
    scale = scale.to(torch.float8_e4m3fn).to(torch.float32)
    output_scale = get_reciprocal(scale * get_reciprocal(global_scale))

    scaled_x = x.to(torch.float32) * output_scale
    clipped_x = torch.clamp(scaled_x, -6.0, 6.0).reshape(m, n)
    return cast_to_fp4(clipped_x), scale.squeeze(-1)


def recover_swizzled_scales(scale, m, n):
    round_up = lambda x, y: (x + y - 1) // y * y
    rounded_m = round_up(m, 128)
    scale_n = n // BLOCK_SIZE
    rounded_n = round_up(scale_n, 4)
    # Recover the swizzled scaling factor to linear layout
    tmp = torch.reshape(scale, (1, rounded_m // 128, rounded_n // 4, 32, 4, 4))
    tmp = torch.permute(tmp, (0, 1, 4, 3, 2, 5))
    result = torch.reshape(tmp, (rounded_m, rounded_n)).to(torch.float32)
    return result[:m, :scale_n]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_fp4_quantization(
    dtype: torch.dtype,
    shape: tuple[int, int],
    seed: int,
    device: str,
) -> None:
    if not is_sm100a_supported(torch.device(device)):
        pytest.skip("Nvfp4 Requires compute capability of 10 or above")
    torch.set_default_device(device)
    torch.manual_seed(seed)
    m, n = shape
    x = torch.randn((m, n), dtype=dtype)
    tensor_amax = torch.abs(x).max().to(torch.float32)
    global_scale = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / tensor_amax
    out_ref, scale_ref = ref_nvfp4_quant(x, global_scale)

    out, out_scale = fp4_quantize(x, global_scale, BLOCK_SIZE, False)
    assert n % BLOCK_SIZE == 0, f"cols needs to be {BLOCK_SIZE} divisible"
    scale_ans = recover_swizzled_scales(
        out_scale.reshape(-1, n // BLOCK_SIZE).view(torch.float8_e4m3fn), m, n
    )
    out_ans = cast_from_fp4(out, m, n)
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
def test_fp4_swizzle_blockscale(
    shape: tuple[int, int],
    seed: int,
    device: str,
) -> None:
    """Test the fp4_swizzle_blockscale function directly."""
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
    swizzled_sf = fp4_swizzle_blockscale(unswizzled_sf, m, n, sf_vec_size)

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

    assert (
        swizzled_sf.shape == expected_shape
    ), f"Expected shape {expected_shape}, got {swizzled_sf.shape}"
    assert_equal = functools.partial(torch.testing.assert_close, rtol=0, atol=0)
    assert_equal(swizzled_sf, ref_swizzled_sf)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
