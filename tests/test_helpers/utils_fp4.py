import torch

import flashinfer.utils as utils

FLOAT4_E2M1_MAX = 6.0

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


def cast_from_fp4(x):
    # The fp4 values are packed in uint8 as [v_1st | v_2nd]
    v_2nd = x & 0xF
    v_1st = (x >> 4) & 0xF
    c = torch.stack((v_2nd, v_1st), dim=-1)
    new_shape = c.shape[:-2] + (
        c.shape[-2] * c.shape[-1],
    )  # fuse the dim added by stack
    lookup_table = torch.tensor(E2M1_TO_FLOAT32, device=c.device)
    out = lookup_table[c.to(torch.long)].reshape(new_shape).to(torch.float32)
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


def ref_fp4_quant(x, global_scale, block_size, sf_use_ue8m0=False):
    assert isinstance(global_scale, (float, int)) or global_scale.dtype == torch.float32

    sliced_shape = x.shape[:-1] + (x.shape[-1] // block_size, block_size)
    sliced_x = torch.reshape(x, sliced_shape)
    vec_max = torch.max(torch.abs(sliced_x), dim=-1, keepdim=True)[0].to(torch.float32)
    scale = global_scale * (vec_max * get_reciprocal(FLOAT4_E2M1_MAX))
    if sf_use_ue8m0:
        scale = (scale.view(torch.int32) + 0x007FFFFF) & 0x7F800000
        scale = scale.view(torch.float32)
    else:
        scale = scale.to(torch.float8_e4m3fn).to(torch.float32)
    output_scale = get_reciprocal(scale * get_reciprocal(global_scale))

    scaled_x = sliced_x.to(torch.float32) * output_scale
    clipped_x = torch.clamp(scaled_x, -6.0, 6.0).reshape(x.shape)
    return cast_to_fp4(clipped_x), scale.squeeze(-1)


def recover_swizzled_scales(scale, m, n, block_size, sf_start_index=0):
    assert sf_start_index + m <= scale.shape[0]
    full_m = scale.shape[0]
    scale_n = n // block_size
    rounded_n = utils.round_up(scale_n, 4)
    # Recover the swizzled scaling factor to linear layout
    tmp = torch.reshape(scale, (1, full_m // 128, rounded_n // 4, 32, 4, 4))
    tmp = torch.permute(tmp, (0, 1, 4, 3, 2, 5))
    result = torch.reshape(tmp, (full_m, rounded_n)).to(torch.float32)
    return result[sf_start_index : sf_start_index + m, :scale_n]


def create_nvfp4_kv(shape, device):
    """Create random NVFP4 KV data directly.

    Args:
        shape: (..., head_dim//2) for packed data, where leading dims are e.g.
               (total_num_pages, page_size, num_kv_heads, head_dim//2).
        device: torch device.

    Returns:
        packed: uint8 tensor of given shape, random with bits 3 and 7 cleared.
        sf: uint8 tensor of shape (*shape[:-1], shape[-1]//8), random from [32, 40, 48, 56]
            (FP8 e4m3 encoding of 0.125, 0.25, 0.5, 1.0).
        global_scale: scalar tensor, 1.0.
    """
    packed = torch.randint(0, 256, shape, dtype=torch.uint8, device=device)
    packed &= 0x77  # clear bit 3 (0x08) and bit 7 (0x80)

    # head_dim//2 packed bytes → head_dim FP4 values; one SF per 16 FP4 values → head_dim//16 SFs
    sf_shape = (*shape[:-1], shape[-1] // 8)
    sf_choices = torch.tensor(
        [56, 48, 40, 32], dtype=torch.uint8, device=device
    )  # 1.0, 0.5, 0.25, 0.125 in FP8 e4m3
    sf_idx = torch.randint(0, 4, sf_shape, device=device)
    sf = sf_choices[sf_idx]

    return packed, sf, torch.tensor(1.0, device=device)


def nvfp4_to_float(x, sf, global_sf):
    """Dequantize NVFP4 (packed uint8 + FP8 SF) back to float32.

    x:  (..., head_dim//2) uint8 packed FP4
    sf: (..., head_dim//16) uint8 FP8 scale factors, one per 16 FP4 elements
    """
    from flashinfer.fp4_quantization import e2m1_and_ufp8sf_scale_to_float

    x_flat = x.reshape(-1, x.shape[-1])
    sf_flat = sf.reshape(-1, sf.shape[-1])
    x_dq = e2m1_and_ufp8sf_scale_to_float(
        x_flat, sf_flat, global_sf, sf_vec_size=16, is_sf_swizzled_layout=False
    )
    return x_dq.reshape(*x.shape[:-1], -1).to(x.device)
