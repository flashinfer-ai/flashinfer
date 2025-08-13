"""
Copyright (c) 2025 by FlashInfer team.

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

import math
from enum import Enum, auto
from itertools import product

import pytest
import torch
from einops import einsum, rearrange

from flashinfer.fp4_quantization import (
    _pad_scale_factors,
    get_fp4_quantization_module,
)
from flashinfer.gemm import group_gemm_mxfp4_nt_groupwise


class QuantMode(Enum):
    MXFP4 = auto()
    MXFP8_E4M3 = auto()
    MXFP8_E5M2 = auto()


def swizzle_blockscale(
    unswizzled_sf: torch.Tensor, b: int, m: int, n: int, sf_vec_size: int = 32
) -> torch.Tensor:
    r"""Swizzle block scale tensor for MXFP4/MXFP8 format.

    This function swizzles the block scale tensor to optimize memory access patterns
    for FP4 operations. The output needs to be padded in the m dimension to be a multiple of 128.

    Args:
        unswizzled_sf (torch.Tensor): Input tensor with dtype uint8.
        b (int): Batch dimension.
        m (int): M dimension.
        n (int): N dimension.
        sf_vec_size (int, optional): Scale factor vector size. Defaults to 32.

    Returns:
        torch.Tensor: Swizzled tensor with the same shape as input.
    """
    assert unswizzled_sf.dtype == torch.uint8, (
        f"Input dtype must be uint8, got {unswizzled_sf.dtype}"
    )
    assert unswizzled_sf.ndim == 3, f"Input must be 3D, got {unswizzled_sf.ndim}"
    assert unswizzled_sf.shape[0] == b, (
        f"Batch dimension must equal b, got {unswizzled_sf.shape[0]} != {b}"
    )
    padded_input_sf_chunked = [
        _pad_scale_factors(unswizzled_sf[i], m, n, sf_vec_size) for i in range(b)
    ]
    padded_input_sf = torch.stack(padded_input_sf_chunked)
    out = get_fp4_quantization_module().nvfp4_block_scale_interleave_sm100(
        padded_input_sf
    )
    out = out.view(padded_input_sf.shape)
    return out


# Vanilla implementation only for unit test
def quantize_e2m1(x):
    r"""
    Quantizes a tensor to FP4.

    Args:
        x (torch.Tensor): The input tensor.

    Returns:
        torch.Tensor: The quantized tensor.
    """
    assert x.shape[-1] % 2 == 0
    x = x.clamp(-6, 6)
    x_sign_bit = torch.lt(x, 0)
    x_abs = torch.abs(x)
    log_x_quant = torch.floor(torch.log2(x_abs)).clamp(0, 2)
    x_quant_e_fp32 = torch.exp2(log_x_quant)
    m_scale = 2
    x_quant_m_scaled_fp32 = torch.round(x_abs * m_scale / x_quant_e_fp32)
    mask = torch.ge(x_quant_m_scaled_fp32, m_scale)
    x_quant_data_raw_e = log_x_quant + mask
    x_quant_data_raw_m = x_quant_m_scaled_fp32 - mask * m_scale
    x_quant_data_raw = (
        x_sign_bit * 8 + x_quant_data_raw_e * m_scale + x_quant_data_raw_m
    ).to(torch.uint8)
    x_quant_data = x_quant_data_raw[..., ::2] + x_quant_data_raw[..., 1::2] * 16
    return x_quant_data


# Vanilla implementation only for unit test
def dequantize_e2m1(x):
    r"""
    Dequantizes a tensor from FP4.

    Args:
        x (torch.Tensor): The input tensor.

    Returns:
        torch.Tensor: The dequantized tensor.
    """
    x_quant_data_raw_1 = x % 16
    x_quant_data_raw_2 = x // 16
    x_quant_data_raw = torch.stack(
        [x_quant_data_raw_1, x_quant_data_raw_2], dim=-1
    ).flatten(start_dim=-2)
    x_sign_bit = x_quant_data_raw // 8
    x = x_quant_data_raw % 8
    m_scale = 2
    x_quant_data_raw_e = x // m_scale
    x_quant_data_raw_m = x % m_scale
    mask = torch.gt(x_quant_data_raw_e, 0).to(torch.float32)
    log_x_quant = x_quant_data_raw_e - mask
    x_quant_m_scaled_fp32 = x_quant_data_raw_m + mask * m_scale
    x_dequant_abs = x_quant_m_scaled_fp32 / m_scale * torch.exp2(log_x_quant)
    x_dequant = (0.5 - x_sign_bit) * 2 * x_dequant_abs
    return x_dequant


def gemm_mxfp8_mxfp4_nt_groupwise_ref(
    A, B, As, Bs, tile_size, n, k, output_dtype=torch.bfloat16
):
    r"""
    A: (m, k), torch.float8_e4m3fn or torch.float8_e5m2
    B: (n // 2, k), e2m1 packed as torch.uint8
    A_scale: (m, k // tile_size), ue8m0 saved as torch.uint8
    B_scale: (n, k // tile_size), ue8m0 saved as torch.uint8
    """
    ue8m0_bias = 127
    A_f32 = A.to(torch.float32)
    B_f32 = dequantize_e2m1(B)
    A_f32_reshape = rearrange(A_f32, "m (k b) -> m k b", b=tile_size)
    A_f32_scale_reshape = A_f32_reshape * rearrange(
        torch.exp2(As.to(torch.float32) - ue8m0_bias), "m k -> m k 1"
    )
    A_f32_scale = rearrange(A_f32_scale_reshape, "m k b -> m (k b)")[:, :k]
    B_f32_reshape = rearrange(B_f32, "n (k b) -> n k b", b=tile_size)
    B_f32_scale_reshape = B_f32_reshape * rearrange(
        torch.exp2(Bs.to(torch.float32) - ue8m0_bias), "n k -> n k 1"
    )
    B_f32_scale = rearrange(B_f32_scale_reshape, "n k b -> n (k b)")[:n, :k]
    return einsum(A_f32_scale, B_f32_scale, "m k, n k -> m n").to(output_dtype)


def quantize_tensor(x, tile_size, n_padded, k_padded, quant_mode):
    r"""
    Quantizes a tensor to MXFP4 or MXFP8.

    Args:
        x (torch.Tensor): The input tensor.
        tile_size (int): The tile size.
        n_padded (int): The padded n dimension, None if not needed.
        k_padded (int): The padded k dimension.
        quant_mode (QuantMode): The quantization mode.

    Returns:
        tuple: A tuple containing the quantized tensor and the
               calculated scales.
    """
    # 1. Initial Setup
    ue8m0_bias = 127
    if quant_mode == QuantMode.MXFP8_E4M3:
        fp8_info = torch.finfo(torch.float8_e4m3fn)
        quant_amax = torch.tensor(fp8_info.max, dtype=torch.float32, device=x.device)
    elif quant_mode == QuantMode.MXFP8_E5M2:
        fp8_info = torch.finfo(torch.float8_e5m2)
        quant_amax = torch.tensor(fp8_info.max, dtype=torch.float32, device=x.device)
    elif quant_mode == QuantMode.MXFP4:
        quant_amax = torch.tensor(6, dtype=torch.float32, device=x.device)
    else:
        raise ValueError(f"Unsupported quantization mode: {quant_mode}")
    if n_padded is not None and x.shape[-2] != n_padded:
        x = torch.cat(
            [
                x,
                torch.zeros(
                    (*x.shape[:-2], n_padded - x.shape[-2], x.shape[-1]),
                    dtype=x.dtype,
                    device=x.device,
                ),
            ],
            dim=-2,
        )
    if x.shape[-1] != k_padded:
        x = torch.cat(
            [
                x,
                torch.zeros(
                    (*x.shape[:-1], k_padded - x.shape[-1]),
                    dtype=x.dtype,
                    device=x.device,
                ),
            ],
            dim=-1,
        )

    # 2. Tiling and Scale Calculation
    x_tiled = x.unflatten(-1, (-1, tile_size))
    x_tiled_abs = x_tiled.abs()
    log2_x_scale = (
        torch.floor(torch.log2(x_tiled_abs.amax(dim=-1)))
        - torch.floor(torch.log2(quant_amax))
    ).clamp(-ue8m0_bias, ue8m0_bias)

    # 3. Final Quantization
    # Divide the original tensor by the broadcasted scales
    x_tiled_quant = (
        torch.exp2(torch.log2(x_tiled_abs) - log2_x_scale[..., None]).clamp(
            0, quant_amax
        )
        * x_tiled.sign()
    )
    x_quant = x_tiled_quant.flatten(-2, -1)

    # Convert the result to the target format
    if quant_mode == QuantMode.MXFP8_E4M3:
        x_quant_data = x_quant.to(torch.float8_e4m3fn)
    elif quant_mode == QuantMode.MXFP8_E5M2:
        x_quant_data = x_quant.to(torch.float8_e5m2)
    elif quant_mode == QuantMode.MXFP4:
        x_quant_data = quantize_e2m1(x_quant)
    else:
        raise ValueError(f"Unsupported quantization mode: {quant_mode}")
    x_scale_data = (log2_x_scale + ue8m0_bias).to(torch.uint8)

    return x_quant_data, x_scale_data


@pytest.mark.parametrize("m", [4, 128, 256, 512, 4096, 8192])
@pytest.mark.parametrize("n", [128, 256, 512, 2879, 4096, 8192])
@pytest.mark.parametrize("k", [128, 256, 512, 2880, 4096, 8192])
@pytest.mark.parametrize("group_size", [1, 2, 4, 8])
@pytest.mark.parametrize("fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16])
def test_mxfp8_mxfp4_groupwise_group_gemm(
    m,
    n,
    k,
    group_size,
    fp8_dtype,
    out_dtype,
):
    torch.random.manual_seed(0)
    tile_size = 32
    alignment_n = 8
    alignment_k = 128

    a_val = torch.randn((group_size * m, k), dtype=torch.float32, device="cuda")
    b_val = torch.randn(
        (group_size, n, k), dtype=torch.float32, device="cuda"
    ) / math.sqrt(k)
    n_padded = (n + alignment_n - 1) // alignment_n * alignment_n
    k_padded = (k + alignment_k - 1) // alignment_k * alignment_k

    if fp8_dtype == torch.float8_e4m3fn:
        a_quant_mode = QuantMode.MXFP8_E4M3
    elif fp8_dtype == torch.float8_e5m2:
        a_quant_mode = QuantMode.MXFP8_E5M2
    else:
        raise ValueError(f"Unsupported FP8 dtype: {fp8_dtype}")
    a_fp8, a_scale = quantize_tensor(a_val, tile_size, None, k_padded, a_quant_mode)
    b_fp4, b_scale = quantize_tensor(
        b_val, tile_size, n_padded, k_padded, QuantMode.MXFP4
    )

    a_scale_swizzled = swizzle_blockscale(
        a_scale.unflatten(0, (group_size, m)), group_size, m, k_padded, tile_size
    ).flatten(0, 1)
    b_scale_swizzled = swizzle_blockscale(
        b_scale, group_size, n_padded, k_padded, tile_size
    )

    group_arange = torch.arange(0, group_size + 1, dtype=torch.int32, device="cuda")
    m_indptr = group_arange * m

    # Pad a_scale_swizzled according to the function compute_sm100_cutlass_group_gemm_args
    # in group_gemm_mxfp4_groupwise_sm100.cuh
    alignment_m_sf = 128
    m_indptr_padded = (
        (m_indptr + group_arange * (alignment_m_sf - 1))
        // alignment_m_sf
        * alignment_m_sf
    )
    m_sf = m_indptr_padded[1:] - m_indptr_padded[:-1]
    a_scale_chunked = a_scale_swizzled.chunk(group_size, dim=0)
    a_scale_chunked = [
        torch.cat(
            [
                x,
                torch.zeros(
                    m_sf[i] - x.shape[0], *x.shape[1:], dtype=x.dtype, device=x.device
                ),
            ]
        )
        for i, x in enumerate(a_scale_chunked)
    ]
    a_scale_swizzled = torch.cat(a_scale_chunked)

    out_ref = torch.empty((group_size * m, n), dtype=out_dtype, device="cuda")
    for i in range(group_size):
        out_ref[m * i : m * (i + 1)] = gemm_mxfp8_mxfp4_nt_groupwise_ref(
            a_fp8[m * i : m * (i + 1)],
            b_fp4[i],
            a_scale[m * i : m * (i + 1)],
            b_scale[i],
            tile_size,
            n,
            k,
            out_dtype,
        )

    mma_sm_list = [1, 2]
    tile_m_list = [128]
    tile_n_list = [64, 128, 192, 256]
    tile_k_list = [128, 256]
    swap_ab_list = [True, False]
    for mma_sm, tile_m, tile_n, tile_k, swap_ab in product(
        mma_sm_list, tile_m_list, tile_n_list, tile_k_list, swap_ab_list
    ):
        out = group_gemm_mxfp4_nt_groupwise(
            a_fp8,
            b_fp4,
            a_scale_swizzled,
            b_scale_swizzled,
            m_indptr,
            mma_sm=mma_sm,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            swap_ab=swap_ab,
            out_dtype=out_dtype,
        )[:, :n]
        torch.testing.assert_close(out, out_ref, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    for fp8_dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
        for out_dtype in [torch.bfloat16, torch.float16]:
            test_mxfp8_mxfp4_groupwise_group_gemm(
                4, 2879, 2880, 2, fp8_dtype, out_dtype
            )
