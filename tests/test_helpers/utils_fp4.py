import torch

from flashinfer.quantization.nvfp4_quantization_utils import (
    NVFP44Over6Config,
    NVFP44Over6ErrMode,
    nvfp4_e4m3_max,
)
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


def nvfp4_global_encode_scale_te(
    global_amax: torch.Tensor,
    nvfp4_4over6_config: NVFP44Over6Config | None = None,
) -> torch.Tensor:
    """Return the effective NVFP4 global encode scale."""
    global_amax = global_amax.to(torch.float32)
    float4_e2m1_max = torch.tensor(6.0, device=global_amax.device, dtype=torch.float32)
    float8_e4m3_max = torch.tensor(
        nvfp4_e4m3_max(nvfp4_4over6_config),
        device=global_amax.device,
        dtype=torch.float32,
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


def nvfp4_global_decode_scale_te(
    global_amax: torch.Tensor,
    nvfp4_4over6_config: NVFP44Over6Config | None = None,
) -> torch.Tensor:
    return torch.div(
        1.0,
        nvfp4_global_encode_scale_te(global_amax, nvfp4_4over6_config),
    )


def ref_fp4_quant_te(
    x: torch.Tensor,
    global_amax: torch.Tensor,
    block_size: int = 16,
    *,
    per_token_rowwise: bool = False,
):
    """NVFP4 reference mirroring TE's Python reference quantizer.

    Returns unpacked E2M1 values and E4M3 block scales in linear layout.
    """
    if x.ndim != 2:
        raise ValueError(
            f"ref_fp4_quant_te expects a 2D tensor, got {x.ndim}D with shape {x.shape}"
        )
    assert block_size == 16

    m, n = x.shape
    x = x.view(m, n // block_size, block_size)
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

    return cast_to_fp4(clipped_x), decode_scale.squeeze(-1)


def _ref_fp4_quant_te_with_decode_scale(
    x_blocks: torch.Tensor,
    decode_scale: torch.Tensor,
    global_decode_scale: torch.Tensor,
) -> torch.Tensor:
    """TE-style FP4 quantization with a caller-provided E4M3 decode scale."""
    max_float32 = torch.tensor(
        torch.finfo(torch.float32).max,
        device=x_blocks.device,
        dtype=torch.float32,
    )
    decode_scale = decode_scale.to(torch.float32)
    encode_scale = torch.minimum(
        torch.div(1.0, decode_scale * global_decode_scale),
        max_float32,
    )
    scaled_x = x_blocks.to(torch.float32) * encode_scale
    clipped_x = torch.clamp(scaled_x, -FLOAT4_E2M1_MAX, FLOAT4_E2M1_MAX)
    return cast_to_fp4(clipped_x)


def _ref_nvfp4_4over6_fp16_candidate(
    q: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Decode E2M1 x E4M3 with fp16 product semantics.

    The CuTe-DSL implementation uses PTX to decode E2M1x2 and E4M3 to f16,
    multiply in f16x2, then widen the product to f32 for error accumulation.
    """
    # Stage 1: express the decoded E2M1 input as sign and integer significand:
    # q = (-1)^q_sign * q_sig * 2^-1.
    q_float = q.to(torch.float32)
    q_sign = (q_float < 0).to(torch.int32)
    q_sig = (torch.abs(q_float) * 2).to(torch.int32)

    # Stage 2: decode the E4M3 scale byte as sign and integer significand:
    # scale = (-1)^scale_sign * scale_sig * 2^scale_exp2.
    scale_code = scale.contiguous().view(torch.uint8).to(torch.int32)
    scale_sign = scale_code >> 7
    scale_exp_field = (scale_code >> 3) & 0xF
    scale_mantissa = scale_code & 0x7
    scale_sig = torch.where(
        scale_exp_field == 0,
        scale_mantissa,
        scale_mantissa + 8,
    )
    scale_exp2 = torch.where(
        scale_exp_field == 0,
        scale_exp_field - 9,
        scale_exp_field - 10,
    )

    # Stage 3: multiply the dyadic significands exactly in integer space:
    # product = (-1)^product_sign * product_sig * 2^product_exp2.
    product_sign = q_sign ^ scale_sign
    product_sig = q_sig * scale_sig
    product_exp2 = scale_exp2 - 1

    # Stage 4: pack that exact dyadic product into fp16 bits. These products
    # are exactly representable in fp16, so RN does not need a tie path here.
    log2_sig = torch.zeros_like(product_sig)
    for threshold in (2, 4, 8, 16, 32, 64, 128, 256):
        log2_sig = log2_sig + (product_sig >= threshold).to(torch.int32)

    floor_exp = log2_sig + product_exp2
    normal_bits = ((floor_exp + 15) << 10) | (
        torch.bitwise_left_shift(product_sig, 10 - log2_sig) - 1024
    )
    subnormal_bits = torch.bitwise_left_shift(product_sig, product_exp2 + 24)
    magnitude_bits = torch.where(floor_exp < -14, subnormal_bits, normal_bits)
    prod_bits = (product_sign << 15) | magnitude_bits
    prod_bits = torch.where(product_sig == 0, product_sign << 15, prod_bits)
    prod_bits = torch.where(
        (scale_code & 0x7F) == 0x7F,
        torch.full_like(prod_bits, 0x7E00),
        prod_bits,
    )

    # Stage 5: decode the fp16 bit pattern to fp32 without using fp16 math.
    sign_f32 = torch.where(
        (prod_bits & 0x8000) != 0,
        torch.tensor(-1.0, device=prod_bits.device, dtype=torch.float32),
        torch.tensor(1.0, device=prod_bits.device, dtype=torch.float32),
    )
    fp16_exp = (prod_bits >> 10) & 0x1F
    fp16_frac = prod_bits & 0x3FF
    normal_f32 = torch.ldexp((fp16_frac + 1024).to(torch.float32), fp16_exp - 25)
    subnormal_f32 = torch.ldexp(fp16_frac.to(torch.float32), fp16_exp - 24)
    return sign_f32 * torch.where(fp16_exp == 0, subnormal_f32, normal_f32)


def ref_fp4_quant_4over6_te(
    x: torch.Tensor,
    global_amax: torch.Tensor,
    block_size: int = 16,
    per_token_rowwise: bool = False,
    nvfp4_4over6_config: NVFP44Over6Config | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """NVFP4 4over6 reference for FlashInfer's candidate-error contract.

    Returns unpacked E2M1 values, E4M3 block scales in linear layout, the
    effective global/per-token decode scale, and the chosen candidate mask.
    """
    if x.ndim != 2:
        raise ValueError(
            f"ref_fp4_quant_4over6_te expects a 2D tensor, got {x.ndim}D with shape {x.shape}"
        )
    assert block_size == 16
    if nvfp4_4over6_config is None:
        nvfp4_4over6_config = NVFP44Over6Config()
    nvfp4_4over6_err_mode = nvfp4_4over6_config.err_mode

    m, n = x.shape
    x_blocks = x.view(m, n // block_size, block_size).to(torch.float32)
    vec_max = torch.amax(torch.abs(x_blocks), dim=-1, keepdim=True)

    e2m1_max = torch.tensor(FLOAT4_E2M1_MAX, device=x.device, dtype=torch.float32)

    float8_e4m3_max = torch.tensor(
        nvfp4_e4m3_max(nvfp4_4over6_config),
        device=x.device,
        dtype=torch.float32,
    )

    if per_token_rowwise:
        global_amax = global_amax.to(torch.float32).view(m)
        global_encode_scale = nvfp4_global_encode_scale_te(
            global_amax, nvfp4_4over6_config
        )
        global_decode_scale = torch.where(
            global_amax == 0.0,
            torch.zeros_like(global_amax),
            torch.div(1.0, global_encode_scale),
        )
        global_encode_scale = global_encode_scale.view(m, 1, 1)
        global_decode_scale_blocks = global_decode_scale.view(m, 1, 1)
        error_row_amax = global_amax.view(m, 1)
        error_global_decode_scale = None
        per_token_scale = global_decode_scale
    else:
        global_amax = global_amax.to(torch.float32)
        global_encode_scale = nvfp4_global_encode_scale_te(
            global_amax, nvfp4_4over6_config
        )
        global_decode_scale = torch.div(1.0, global_encode_scale)
        global_decode_scale_blocks = global_decode_scale
        error_row_amax = None
        error_global_decode_scale = global_decode_scale
        per_token_scale = global_decode_scale.reshape(())

    # Candidate scale construction follows the original 4over6 reference.
    sf6_high_precision = torch.where(
        vec_max == 0.0,
        torch.zeros_like(vec_max),
        torch.div(vec_max, e2m1_max) * global_encode_scale,
    )
    sf4_high_precision = sf6_high_precision * 1.5
    float8_e4m3_clamp_max = torch.tensor(448.0, device=x.device, dtype=torch.float32)
    sf4_high_precision = torch.clamp(
        sf4_high_precision, min=-float8_e4m3_clamp_max, max=float8_e4m3_clamp_max
    )
    sf6_high_precision = torch.clamp(
        sf6_high_precision, min=-float8_e4m3_clamp_max, max=float8_e4m3_clamp_max
    )
    sf4_fp8 = sf4_high_precision.to(torch.float8_e4m3fn)
    sf6_fp8 = sf6_high_precision.to(torch.float8_e4m3fn)
    sf4 = sf4_fp8.to(torch.float32)
    sf6 = sf6_fp8.to(torch.float32)

    # Candidate FP4 quantization follows the TE reference expression exactly:
    # encode_scale = min(1 / (fp32(fp8_scale) * global_decode_scale), fp32_max),
    # then clamp to E2M1 range and map with cast_to_fp4.
    q4 = _ref_fp4_quant_te_with_decode_scale(
        x_blocks,
        sf4_fp8,
        global_decode_scale_blocks,
    )
    q6 = _ref_fp4_quant_te_with_decode_scale(
        x_blocks,
        sf6_fp8,
        global_decode_scale_blocks,
    )

    err4 = torch.zeros((m, n // block_size), dtype=torch.float32, device=x.device)
    err6 = torch.zeros((m, n // block_size), dtype=torch.float32, device=x.device)
    if nvfp4_4over6_config.err_use_fast_math:
        original_scaled = x_blocks * global_encode_scale
        candidate4_scaled = _ref_nvfp4_4over6_fp16_candidate(q4, sf4_fp8)
        candidate6_scaled = _ref_nvfp4_4over6_fp16_candidate(q6, sf6_fp8)
        for i in range(block_size):
            diff4 = candidate4_scaled[:, :, i] - original_scaled[:, :, i]
            diff6 = candidate6_scaled[:, :, i] - original_scaled[:, :, i]
            if nvfp4_4over6_err_mode == NVFP44Over6ErrMode.MSE:
                err4 += diff4 * diff4
                err6 += diff6 * diff6
            else:
                err4 += torch.abs(diff4)
                err6 += torch.abs(diff6)
    else:
        # Per-token strict scoring uses the row amax that is already computed
        # online. Per-tensor strict scoring intentionally uses the normal
        # global-decode-scale expression to avoid an additional tensor amax.
        denom = e2m1_max * float8_e4m3_max
        sf4 = sf4.squeeze(-1)
        sf6 = sf6.squeeze(-1)
        for i in range(block_size):
            val4 = q4[:, :, i] * sf4
            if per_token_rowwise:
                val4 = val4 * error_row_amax
                val4 = val4 / denom
            else:
                val4 = val4 * error_global_decode_scale
            diff4 = val4 - x_blocks[:, :, i]
            val6 = q6[:, :, i] * sf6
            if per_token_rowwise:
                val6 = val6 * error_row_amax
                val6 = val6 / denom
            else:
                val6 = val6 * error_global_decode_scale
            diff6 = val6 - x_blocks[:, :, i]
            if nvfp4_4over6_err_mode == NVFP44Over6ErrMode.MSE:
                err4 += diff4 * diff4
                err6 += diff6 * diff6
            else:
                err4 += torch.abs(diff4)
                err6 += torch.abs(diff6)
    pick_four = err4 < err6

    q_ref = torch.where(pick_four.unsqueeze(-1), q4, q6).reshape(m, n)
    scale_ref = torch.where(
        pick_four,
        sf4_fp8.view(torch.uint8).squeeze(-1),
        sf6_fp8.view(torch.uint8).squeeze(-1),
    )
    return q_ref, scale_ref, per_token_scale, pick_four


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
