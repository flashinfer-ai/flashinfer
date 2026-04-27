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


def cast_to_fp4x2(x):
    """Quantize a tensor to FP4 E2M1 and store in a byte tensor."""
    result = torch.zeros_like(x, dtype=torch.uint8)
    result[(x >= 0.0) & (x <= 0.25)] = 0
    result[(x > 0.25) & (x < 0.75)] = 1
    result[(x >= 0.75) & (x <= 1.25)] = 2
    result[(x > 1.25) & (x < 1.75)] = 3
    result[(x >= 1.75) & (x <= 2.5)] = 4
    result[(x > 2.5) & (x < 3.5)] = 5
    result[(x >= 3.5) & (x <= 5.0)] = 6
    result[x > 5.0] = 7

    result[(x >= -0.25) & (x < -0.0)] = 8
    result[(x < -0.25) & (x > -0.75)] = 9
    result[(x <= -0.75) & (x >= -1.25)] = 10
    result[(x < -1.25) & (x > -1.75)] = 11
    result[(x <= -1.75) & (x >= -2.5)] = 12
    result[(x < -2.5) & (x > -3.5)] = 13
    result[(x <= -3.5) & (x >= -5.0)] = 14
    result[x < -5.0] = 15

    return result[:, ::2] + result[:, 1::2] * 16


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


def nvfp4_global_encode_scale_te(global_amax: torch.Tensor) -> torch.Tensor:
    global_amax = global_amax.to(torch.float32)
    float4_e2m1_max = torch.tensor(6.0, device=global_amax.device, dtype=torch.float32)
    float8_e4m3_max = torch.tensor(
        448.0, device=global_amax.device, dtype=torch.float32
    )
    global_encode_scale = torch.div(float8_e4m3_max * float4_e2m1_max, global_amax)
    global_encode_scale = torch.min(
        global_encode_scale,
        torch.tensor(
            torch.finfo(torch.float32).max,
            device=global_encode_scale.device,
            dtype=torch.float32,
        ),
    )
    if global_encode_scale.numel() == 1:
        if global_encode_scale == torch.tensor(
            0.0, device=global_amax.device, dtype=torch.float32
        ):
            global_encode_scale = torch.tensor(
                1.0, device=global_amax.device, dtype=torch.float32
            )
    else:
        global_encode_scale = torch.where(
            global_encode_scale == 0.0,
            torch.ones_like(global_encode_scale),
            global_encode_scale,
        )
    return global_encode_scale


def nvfp4_global_decode_scale_te(global_amax: torch.Tensor) -> torch.Tensor:
    return torch.div(1.0, nvfp4_global_encode_scale_te(global_amax))


def _quantize_blockwise_reference(
    x: torch.Tensor,
    global_amax: torch.Tensor,
    tile_len_x: int,
    tile_len_y: int,
    *,
    per_token_rowwise: bool = False,
):
    """Subset of TE's NVFP4 _quantize_blockwise_reference used by tests."""
    if x.ndim != 2:
        raise ValueError(
            f"_quantize_blockwise_reference expects a 2D tensor, got {x.ndim}D with shape"
            f" {x.shape}"
        )
    assert tile_len_x == 16 and tile_len_y == 1

    m, n = x.shape
    x = x.view(m, n // tile_len_x, tile_len_x)
    vec_max = torch.amax(torch.abs(x), dim=-1, keepdim=True).to(torch.float32)

    FLOAT4_E2M1_MAX = torch.tensor(6.0, device=x.device, dtype=torch.float32)
    FLOAT8_E4M3_MAX = torch.tensor(448.0, device=x.device, dtype=torch.float32)

    if per_token_rowwise:
        global_amax = global_amax.to(torch.float32).view(m, 1, 1)
    else:
        global_amax = global_amax.to(torch.float32)

    global_encode_scale = nvfp4_global_encode_scale_te(global_amax)
    global_decode_scale = torch.div(1.0, global_encode_scale)
    global_encode_scale_multiplier = global_encode_scale * torch.reciprocal(
        FLOAT4_E2M1_MAX
    )

    decode_scale = vec_max * global_encode_scale_multiplier
    decode_scale = torch.min(
        decode_scale,
        torch.tensor(
            torch.finfo(torch.float32).max,
            device=decode_scale.device,
            dtype=torch.float32,
        ),
    )
    decode_scale = torch.clamp(decode_scale, min=-FLOAT8_E4M3_MAX, max=FLOAT8_E4M3_MAX)
    decode_scale = decode_scale.to(torch.float8_e4m3fn)

    encode_scale = torch.min(
        torch.div(1.0, decode_scale.to(torch.float32) * global_decode_scale),
        torch.tensor(
            torch.finfo(torch.float32).max,
            device=decode_scale.device,
            dtype=torch.float32,
        ),
    )

    scaled_x = x.to(torch.float32) * encode_scale
    clipped_x = torch.clamp(scaled_x, -FLOAT4_E2M1_MAX, FLOAT4_E2M1_MAX).reshape(m, n)

    return cast_to_fp4x2(clipped_x), decode_scale.squeeze(-1)


def ref_fp4_quant_te(
    x: torch.Tensor,
    global_amax: torch.Tensor,
    block_size: int = 16,
    *,
    per_token_rowwise: bool = False,
):
    """NVFP4 reference mirroring TE's Python reference quantizer.

    Returns packed E2M1 bytes and E4M3 block scales in linear layout.
    """
    return _quantize_blockwise_reference(
        x,
        global_amax,
        block_size,
        1,
        per_token_rowwise=per_token_rowwise,
    )


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
