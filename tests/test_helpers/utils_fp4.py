"""Independent FP4 quantization references for bit-exact, zero-tolerance tests.

``ref_fp4_quant`` covers CUDA and CuTe DSL ``fp4_quantize``/``nvfp4_quantize``
standard-fast NVFP4 and ``fp4_quantize``/``mxfp4_quantize`` MXFP4 with UE8M0
scales. ``ref_fp4_quant_te`` covers AMAX-based NVFP4, per tensor or per token,
with explicit standard/4over6 and strict/fast arithmetic configuration.
Compare packed payload and scale bytes; per-token decode scales compare FP32
bits. These helpers do not call production quantizers. Fast arithmetic uses
an independent PTX reciprocal primitive and requires CUDA.
"""

import functools

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
    -0.0,
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
    """Quantize to unpacked E2M1 values and linear FP32 block scales.

    The E4M3 branch is the bit-exact standard-fast NVFP4 reference for
    ``nvfp4_quantize`` and ``fp4_quantize`` with 16-element blocks, for both
    CUDA and CuTe DSL. CUDA is required for the independent PTX reciprocal. Inputs are widened to
    FP32 without additional rounding; arbitrary leading dimensions are
    preserved. The supplied encode scale does not provide the original
    tensor AMAX needed for TE per-tensor parity. No environment is read and
    no strict or 4over6 mode is inferred.

    The UE8M0 branch is the bit-exact FP16/BF16 MXFP4 reference for
    ``mxfp4_quantize`` and ``fp4_quantize`` with 32-element blocks, for both
    CUDA and CuTe DSL. It ignores the global scale, rounds block scales
    upward, and converts E2M1 values with nearest-even rounding.
    Both backends preserve FP32 subnormals. Returned FP32 scales use FlashInfer's
    host-decode convention: UE8M0 byte zero becomes zero. Compare packed
    payloads and scale bytes with zero tolerance, including signed zeros.
    """
    assert isinstance(global_scale, (float, int)) or global_scale.dtype == torch.float32

    sliced_shape = x.shape[:-1] + (x.shape[-1] // block_size, block_size)
    sliced_x = x.to(torch.float32).reshape(sliced_shape)
    vec_max = sliced_x.abs().amax(dim=-1, keepdim=True)
    if sf_use_ue8m0:
        if block_size != 32 or x.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("MXFP4 reference requires FP16/BF16 and block size 32")
        six = torch.full((), 6.0, dtype=torch.float32, device=x.device)
        normalized_max = vec_max * _ref_rcp_approx_ftz(six)

        # Saturating UE8M0 conversion rounds toward +inf, including its
        # minimum-scale boundary at 2**-127. This is not nearest rounding.
        bits = normalized_max.contiguous().view(torch.int32)
        exponent = (bits >> 23) & 0xFF
        mantissa = bits & 0x7FFFFF
        round_up = (mantissa != 0) & ((exponent != 0) | (mantissa > 0x400000))
        exponent = (exponent + round_up.to(torch.int32)).clamp(max=254)
        inverse = ((254 - exponent) << 23).view(torch.float32)
        inverse = torch.where(exponent == 0, 0.0, inverse)
        scaled = sliced_x * inverse
        q = _ref_e2m1_values(scaled).reshape_as(x)
        return q, (exponent << 23).view(torch.float32).squeeze(-1)
    else:
        global_scale = torch.as_tensor(
            global_scale, dtype=torch.float32, device=x.device
        )
        q, sf, _ = _ref_nvfp4_quant_blocks(
            sliced_x,
            vec_max,
            global_scale,
            _ref_rcp_approx_ftz(global_scale),
            None,
            False,
            None,
        )
        return q.reshape_as(x), sf.view(torch.float8_e4m3fn).float()


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
    per_token_rowwise: bool = False,
    nvfp4_4over6_config: NVFP44Over6Config | None = None,
    disable_fp4_quant_fast_math: bool = True,
    global_scale_inv: torch.Tensor | None = None,
):
    """AMAX-based NVFP4 reference for standard or configured 4over6 quantization.

    ``None`` selects standard NVFP4; a resolved config selects 4over6. Returns
    unpacked E2M1 values, linear E4M3 scale bytes, the global/row decode scale,
    and the chosen-four mask (all false for standard NVFP4).

    Strict arithmetic retains the TE-style AMAX contract. Fast arithmetic
    models FlashInfer's approximate reciprocals and requires CUDA. The
    quantization arithmetic flag is independent of the recipe's candidate
    error mode. Signed zeros follow the FP4 conversion; 4over6 zero-AMAX
    blocks return zero bytes.

    With ``per_token_rowwise=True``, pass the original row AMAX. The optional
    scalar ``global_scale_inv`` preserves the per-token API's normalization;
    otherwise it is 1 / (6 * recipe E4M3 maximum). Per-tensor calls derive
    the encode scale from the supplied tensor AMAX. No environment is read.
    """
    if x.ndim != 2 or x.shape[1] % block_size:
        raise ValueError("AMAX NVFP4 reference requires a 2D block-aligned tensor")
    assert block_size == 16
    if nvfp4_4over6_config is not None and not isinstance(
        nvfp4_4over6_config, NVFP44Over6Config
    ):
        raise TypeError("Expected a resolved NVFP44Over6Config or None")
    if not isinstance(disable_fp4_quant_fast_math, bool):
        raise TypeError("disable_fp4_quant_fast_math must be bool")
    m, n = x.shape
    blocks = x.float().reshape(m, n // block_size, block_size)
    block_amax = blocks.abs().amax(dim=-1, keepdim=True)
    global_amax = torch.as_tensor(global_amax, dtype=torch.float32, device=x.device)
    if per_token_rowwise:
        global_amax = global_amax.reshape(m, 1, 1)
        if global_scale_inv is None:
            global_scale_inv = 1.0 / (6.0 * nvfp4_e4m3_max(nvfp4_4over6_config))
        global_scale_inv = torch.as_tensor(
            global_scale_inv, dtype=torch.float32, device=x.device
        )
        if global_scale_inv.numel() != 1:
            raise ValueError("Per-token normalization requires a scalar inverse scale")
        maximum = torch.finfo(torch.float32).max
        if disable_fp4_quant_fast_math:
            global_scale = torch.div(
                torch.ones_like(global_scale_inv), global_scale_inv
            )
            encode = torch.div(global_scale, global_amax).clamp(max=maximum)
            encode = torch.where(encode == 0, 1.0, encode)
            output_decode = torch.div(torch.ones_like(encode), encode)
        else:
            output_decode = global_amax * global_scale_inv
            encode = _ref_rcp_approx_ftz(output_decode)
        output_decode = torch.where(global_amax == 0, 0.0, output_decode).reshape(m)
        encode = torch.where(
            global_amax == 0,
            maximum if nvfp4_4over6_config is None else 0.0,
            encode,
        )
        error_row_amax = global_amax.reshape(m, 1)
    else:
        if global_scale_inv is not None:
            raise ValueError("global_scale_inv applies only to per-token quantization")
        if global_amax.numel() != 1:
            raise ValueError("Per-tensor quantization requires a scalar AMAX")
        encode = nvfp4_global_encode_scale_te(global_amax, nvfp4_4over6_config)
        error_row_amax = None

    if disable_fp4_quant_fast_math:
        decode = torch.div(torch.ones_like(encode), encode)
    else:
        decode = _ref_rcp_approx_ftz(encode)
    if not per_token_rowwise:
        output_decode = decode.reshape(())
    q, sf, pick_four = _ref_nvfp4_quant_blocks(
        blocks,
        block_amax,
        encode,
        decode,
        nvfp4_4over6_config,
        disable_fp4_quant_fast_math,
        error_row_amax,
    )
    return q.reshape_as(x), sf, output_decode, pick_four


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


@functools.lru_cache(maxsize=1)
def _rcp_approx_ftz_kernel():
    # Keep Triton optional for the pre-existing pure-PyTorch references.
    import triton
    import triton.language as tl

    @triton.jit
    def kernel(X, Y, size: tl.constexpr, BLOCK: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        value = tl.load(X + offsets, offsets < size, other=1.0)
        result = tl.inline_asm_elementwise(
            "rcp.approx.ftz.f32 $0, $1;",
            constraints="=f,f",
            args=[value],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        tl.store(Y + offsets, result, offsets < size)

    return kernel


def _ref_rcp_approx_ftz(x):
    """Evaluate the PTX reciprocal primitive independently of CuTe quantizers."""
    if x.device.type != "cuda" or x.dtype != torch.float32:
        raise ValueError("Approximate reciprocal reference requires CUDA FP32 input")
    x = x.contiguous()
    output = torch.empty_like(x)
    if x.numel():
        _rcp_approx_ftz_kernel()[((x.numel() + 255) // 256,)](x, output, x.numel(), 256)
    return output


def _ref_e2m1_values(x):
    # PTX E2M1 saturates either sign of NaN to +6 and preserves signed zeros.
    finite_or_inf = torch.copysign(cast_to_fp4(x.abs()), x)
    return torch.where(torch.isnan(x), 6.0, finite_or_inf)


def pack_fp4_values(q_ref):
    """Pack unpacked E2M1 values into bytes, with the first value in low bits."""
    if q_ref.ndim != 2 or q_ref.shape[1] % 2:
        raise ValueError("FP4 packing requires a 2D tensor with even row width")
    q_abs = torch.abs(q_ref)
    q_code = torch.zeros_like(q_abs, dtype=torch.uint8)
    for code, value in enumerate(E2M1_TO_FLOAT32[:8]):
        q_code = torch.where(q_abs == value, code, q_code)
    q_code = q_code | (torch.signbit(q_ref).to(torch.uint8) << 3)
    return q_code[:, 0::2] | (q_code[:, 1::2] << 4)


def _ref_nvfp4_quant_blocks(
    blocks,
    block_amax,
    encode,
    decode,
    nvfp4_4over6_config,
    disable_fp4_quant_fast_math,
    error_row_amax,
):
    """Shared candidate construction and conversion for both scale contracts."""
    if disable_fp4_quant_fast_math:
        if nvfp4_4over6_config is None:
            sf6_high = block_amax * (encode * (1.0 / 6.0))
        else:
            six = torch.full((), 6.0, dtype=torch.float32, device=blocks.device)
            sf6_high = torch.div(block_amax, six) * encode
    else:
        six = torch.full((), 6.0, dtype=torch.float32, device=blocks.device)
        sf6_high = encode * (block_amax * _ref_rcp_approx_ftz(six))
    sf6 = sf6_high.clamp(0, 448).to(torch.float8_e4m3fn)

    def quantize(sf):
        if disable_fp4_quant_fast_math:
            multiplier = torch.div(
                torch.ones_like(sf, dtype=torch.float32), sf.float() * decode
            ).clamp(max=torch.finfo(torch.float32).max)
        else:
            multiplier = _ref_rcp_approx_ftz(sf.float() * decode)
        if nvfp4_4over6_config is None:
            if disable_fp4_quant_fast_math:
                # Nonzero blocks with underflowed SF retain the clamped
                # multiplier; only zero-AMAX blocks are zeroed in strict mode.
                multiplier = torch.where(block_amax == 0, 0.0, multiplier)
            else:
                multiplier = torch.where(sf.float() == 0, 0.0, multiplier)
        return _ref_e2m1_values(blocks * multiplier)

    q6 = quantize(sf6)
    if nvfp4_4over6_config is None:
        return (
            q6,
            sf6.view(torch.uint8).squeeze(-1),
            torch.zeros_like(block_amax.squeeze(-1), dtype=torch.bool),
        )
    else:
        sf4_fp8 = (sf6_high * 1.5).clamp(0, 448).to(torch.float8_e4m3fn)
        sf6_fp8 = sf6
        q4 = quantize(sf4_fp8)
        m, num_blocks, block_size = blocks.shape
        nvfp4_4over6_err_mode = nvfp4_4over6_config.err_mode
        per_token_rowwise = error_row_amax is not None
        e2m1_max = torch.tensor(6.0, dtype=torch.float32, device=blocks.device)
        float8_e4m3_max = torch.tensor(
            nvfp4_e4m3_max(nvfp4_4over6_config),
            dtype=torch.float32,
            device=blocks.device,
        )
        sf4 = sf4_fp8.to(torch.float32)
        sf6 = sf6_fp8.to(torch.float32)
        err4 = torch.zeros((m, num_blocks), dtype=torch.float32, device=blocks.device)
        err6 = torch.zeros_like(err4)
        if nvfp4_4over6_config.err_use_fast_math:
            original_scaled = blocks * encode
            candidate4_scaled = _ref_nvfp4_4over6_fp16_candidate(q4, sf4_fp8)
            candidate6_scaled = _ref_nvfp4_4over6_fp16_candidate(q6, sf6_fp8)
            for i in range(block_size):
                diff4 = candidate4_scaled[:, :, i] - original_scaled[:, :, i]
                diff6 = candidate6_scaled[:, :, i] - original_scaled[:, :, i]
                if nvfp4_4over6_err_mode == NVFP44Over6ErrMode.MSE:
                    err4 += diff4 * diff4
                    err6 += diff6 * diff6
                elif nvfp4_4over6_err_mode == NVFP44Over6ErrMode.MAE:
                    err4 += torch.abs(diff4)
                    err6 += torch.abs(diff6)
                else:
                    raise ValueError(
                        f"Unsupported 4over6 error mode: {nvfp4_4over6_err_mode}"
                    )
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
                    val4 = val4 * decode
                diff4 = val4 - blocks[:, :, i]
                val6 = q6[:, :, i] * sf6
                if per_token_rowwise:
                    val6 = val6 * error_row_amax
                    val6 = val6 / denom
                else:
                    val6 = val6 * decode
                diff6 = val6 - blocks[:, :, i]
                if nvfp4_4over6_err_mode == NVFP44Over6ErrMode.MSE:
                    err4 += diff4 * diff4
                    err6 += diff6 * diff6
                elif nvfp4_4over6_err_mode == NVFP44Over6ErrMode.MAE:
                    err4 += torch.abs(diff4)
                    err6 += torch.abs(diff6)
                else:
                    raise ValueError(
                        f"Unsupported 4over6 error mode: {nvfp4_4over6_err_mode}"
                    )
        pick_four = err4 < err6

        q_ref = torch.where(pick_four.unsqueeze(-1), q4, q6)
        scale_ref = torch.where(
            pick_four,
            sf4_fp8.view(torch.uint8).squeeze(-1),
            sf6_fp8.view(torch.uint8).squeeze(-1),
        )
        # The 4over6 kernel exits before candidate formation for zero blocks,
        # including negative-zero inputs.
        q_ref = torch.where(block_amax == 0, 0.0, q_ref)
        scale_ref = torch.where(block_amax.squeeze(-1) == 0, 0, scale_ref)
        return q_ref, scale_ref, pick_four


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
