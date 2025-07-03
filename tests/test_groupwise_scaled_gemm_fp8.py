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

import pytest
import torch
from einops import einsum, rearrange, reduce, repeat

from flashinfer.gemm import (
    gemm_fp8_nt_blockscaled,
    gemm_fp8_nt_groupwise,
    group_gemm_fp8_nt_groupwise,
)


def gemm_fp8_nt_blockscaled_ref(
    A, B, As, Bs, block_size, scale_major_mode, output_dtype=torch.float16
):
    r"""
    A: (m, k)
    B: (n, k)
    A_scale: (k // block_size, m // block_size)
    B_scale: (k // block_size, n // block_size)
    """
    A_f32 = A.to(torch.float32)
    B_f32 = B.to(torch.float32)
    A_f32_reshape = rearrange(
        A_f32, "(m b) (k c) -> m k b c", b=block_size, c=block_size
    )
    if scale_major_mode == "K":
        A_f32_scale_reshape = A_f32_reshape * rearrange(As, "m k -> m k 1 1")
    else:
        A_f32_scale_reshape = A_f32_reshape * rearrange(As, "k m -> m k 1 1")
    A_f32_scale = rearrange(A_f32_scale_reshape, "m k b c -> (m b) (k c)")
    B_f32_reshape = rearrange(
        B_f32, "(n b) (k c) -> n k b c", b=block_size, c=block_size
    )
    if scale_major_mode == "K":
        B_f32_scale_reshape = B_f32_reshape * rearrange(Bs, "n k -> n k 1 1")
    else:
        B_f32_scale_reshape = B_f32_reshape * rearrange(Bs, "k n -> n k 1 1")
    B_f32_scale = rearrange(B_f32_scale_reshape, "n k b c -> (n b) (k c)")
    C_f32 = einsum(A_f32_scale, B_f32_scale, "m k, n k -> m n")
    return C_f32.to(output_dtype)


def gemm_fp8_nt_groupwise_ref(
    A, B, As, Bs, block_size, scale_major_mode, output_dtype=torch.float16
):
    r"""
    A: (m, k)
    B: (n, k)
    A_scale: (k // block_size, m)
    B_scale: (k // block_size, n // block_size)
    """
    A_f32 = A.to(torch.float32)
    B_f32 = B.to(torch.float32)
    A_f32_reshape = rearrange(A_f32, "m (k b) -> m k b", b=block_size)
    if scale_major_mode == "K":
        A_f32_scale_reshape = A_f32_reshape * rearrange(As, "m k -> m k 1")
    else:
        A_f32_scale_reshape = A_f32_reshape * rearrange(As, "k m -> m k 1")
    A_f32_scale = rearrange(A_f32_scale_reshape, "m k b -> m (k b)")
    B_f32_reshape = rearrange(
        B_f32, "(n b) (k c) -> n k b c", b=block_size, c=block_size
    )
    if scale_major_mode == "K":
        B_f32_scale_reshape = B_f32_reshape * rearrange(Bs, "n k -> n k 1 1")
    else:
        B_f32_scale_reshape = B_f32_reshape * rearrange(Bs, "k n -> n k 1 1")
    B_f32_scale = rearrange(B_f32_scale_reshape, "n k b c -> (n b) (k c)")
    return einsum(A_f32_scale, B_f32_scale, "m k, n k -> m n").to(output_dtype)


def quantize_fp8(x, scale_shape, tile_shape, scale_major_mode):
    """
    Quantizes a 2D or 3D tensor to FP8.

    Args:
        x (torch.Tensor): The 2D or 3D input tensor.
        scale_shape (tuple): The shape of the scale tensor.
        tile_shape (tuple): The shape of the tiles.
        scale_major_mode (str): The tiling order, "K" for row-major like,
                                or another value for column-major like.

    Returns:
        tuple: A tuple containing the quantized FP8 tensor and the
               calculated float32 scales.
    """
    # 1. Assertions and Initial Setup
    ndim = x.ndim
    assert ndim in [2, 3], f"x.ndim must be 2 or 3, but got {ndim}"
    assert ndim == len(scale_shape) == len(tile_shape)

    fp8_info = torch.finfo(torch.float8_e4m3fn)
    fp8_amax = torch.tensor(fp8_info.max, device=x.device, dtype=torch.float32)

    # 2. Tiling and Scale Calculation
    if ndim == 2:
        s0, s1 = scale_shape
        t0, t1 = tile_shape
        if scale_major_mode == "K":
            # Tile x and find the max absolute value in each tile
            x_tiled = rearrange(x, "(s0 t0) (s1 t1) -> s0 s1 t0 t1", s0=s0, s1=s1)
            abs_max = reduce(x_tiled.abs(), "s0 s1 t0 t1 -> s0 s1", "max")
            x_scale = abs_max / fp8_amax

            # Broadcast scales back to the original tensor shape
            scales_repeated = repeat(x_scale, "s0 s1 -> (s0 t0) (s1 t1)", t0=t0, t1=t1)
        else:
            # Handle column-major tiling
            x_tiled = rearrange(x, "(s1 t0) (s0 t1) -> s0 s1 t0 t1", s0=s0, s1=s1)
            abs_max = reduce(x_tiled.abs(), "s0 s1 t0 t1 -> s0 s1", "max")
            x_scale = abs_max / fp8_amax

            # Permute scale axes before repeating to match layout
            scales_permuted = rearrange(x_scale, "s0 s1 -> s1 s0")
            scales_repeated = repeat(
                scales_permuted, "s1 s0 -> (s1 t0) (s0 t1)", t0=t0, t1=t1
            )

    elif ndim == 3:
        s0, s1, s2 = scale_shape
        t0, t1, t2 = tile_shape
        if scale_major_mode == "K":
            # Tile x and find the max absolute value in each tile
            x_tiled = rearrange(
                x, "(s0 t0) (s1 t1) (s2 t2) -> s0 s1 s2 t0 t1 t2", s0=s0, s1=s1, s2=s2
            )
            abs_max = reduce(x_tiled.abs(), "s0 s1 s2 t0 t1 t2 -> s0 s1 s2", "max")
            x_scale = abs_max / fp8_amax

            # Broadcast scales back to the original tensor shape
            scales_repeated = repeat(
                x_scale, "s0 s1 s2 -> (s0 t0) (s1 t1) (s2 t2)", t0=t0, t1=t1, t2=t2
            )
        else:
            # Handle layout where the last two axes are swapped
            x_tiled = rearrange(
                x, "(s0 t0) (s2 t1) (s1 t2) -> s0 s1 s2 t0 t1 t2", s0=s0, s1=s1, s2=s2
            )
            abs_max = reduce(x_tiled.abs(), "s0 s1 s2 t0 t1 t2 -> s0 s1 s2", "max")
            x_scale = abs_max / fp8_amax

            # Permute scale axes before repeating to match layout
            scales_permuted = rearrange(x_scale, "s0 s1 s2 -> s0 s2 s1")
            scales_repeated = repeat(
                scales_permuted,
                "s0 s2 s1 -> (s0 t0) (s2 t1) (s1 t2)",
                t0=t0,
                t1=t1,
                t2=t2,
            )

    # 3. Final Quantization
    # Divide the original tensor by the broadcasted scales
    x_fp32 = x / (scales_repeated + 1e-8)

    # Convert the result to the target FP8 format
    x_fp8 = x_fp32.to(torch.float8_e4m3fn)

    return x_fp8, x_scale


@pytest.mark.parametrize("m", [128, 256, 512, 4096, 8192])
@pytest.mark.parametrize("n", [128, 256, 512, 4096, 8192])
@pytest.mark.parametrize("k", [128, 256, 512, 4096, 8192])
@pytest.mark.parametrize("scale_major_mode", ["MN", "K"])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16])
def test_fp8_blockscale_gemm(
    m,
    n,
    k,
    scale_major_mode,
    out_dtype,
):
    torch.random.manual_seed(0)
    tile_size = 128

    a_val = torch.randn((m, k), dtype=torch.float, device="cuda")
    b_val = torch.randn((n, k), dtype=torch.float, device="cuda") / math.sqrt(k)

    if scale_major_mode == "K":
        a_scale_shape = (m // tile_size, k // tile_size)
        b_scale_shape = (n // tile_size, k // tile_size)
    else:
        a_scale_shape = (k // tile_size, m // tile_size)
        b_scale_shape = (k // tile_size, n // tile_size)
    a_tile_shape = (tile_size, tile_size)
    b_tile_shape = (tile_size, tile_size)

    a_fp8, a_scale = quantize_fp8(a_val, a_scale_shape, a_tile_shape, scale_major_mode)
    b_fp8, b_scale = quantize_fp8(b_val, b_scale_shape, b_tile_shape, scale_major_mode)

    c = gemm_fp8_nt_blockscaled(
        a_fp8, b_fp8, a_scale, b_scale, scale_major_mode, out_dtype=out_dtype
    )
    ref_c = gemm_fp8_nt_blockscaled_ref(
        a_fp8,
        b_fp8,
        a_scale,
        b_scale,
        tile_size,
        scale_major_mode,
        out_dtype,
    )
    torch.testing.assert_close(c, ref_c, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("m", [128, 256, 512, 4096, 8192])
@pytest.mark.parametrize("n", [128, 256, 512, 4096, 8192])
@pytest.mark.parametrize("k", [128, 256, 512, 4096, 8192])
@pytest.mark.parametrize("scale_major_mode", ["MN", "K"])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16])
def test_fp8_groupwise_gemm(
    m,
    n,
    k,
    scale_major_mode,
    out_dtype,
):
    torch.random.manual_seed(0)
    tile_size = 128

    a_val = torch.randn((m, k), dtype=torch.float, device="cuda")
    b_val = torch.randn((n, k), dtype=torch.float, device="cuda") / math.sqrt(k)

    if scale_major_mode == "K":
        a_scale_shape = (m, k // tile_size)
        b_scale_shape = (n // tile_size, k // tile_size)
    else:
        a_scale_shape = (k // tile_size, m)
        b_scale_shape = (k // tile_size, n // tile_size)
    a_tile_shape = (1, tile_size)
    b_tile_shape = (tile_size, tile_size)

    a_fp8, a_scale = quantize_fp8(a_val, a_scale_shape, a_tile_shape, scale_major_mode)
    b_fp8, b_scale = quantize_fp8(b_val, b_scale_shape, b_tile_shape, scale_major_mode)

    c = gemm_fp8_nt_groupwise(
        a_fp8, b_fp8, a_scale, b_scale, scale_major_mode, out_dtype=out_dtype
    )
    ref_c = gemm_fp8_nt_groupwise_ref(
        a_fp8,
        b_fp8,
        a_scale,
        b_scale,
        tile_size,
        scale_major_mode,
        out_dtype,
    )
    torch.testing.assert_close(c, ref_c, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("m", [4, 128, 256, 512, 4096, 8192])
@pytest.mark.parametrize("n", [128, 256, 512, 4096, 8192])
@pytest.mark.parametrize("k", [128, 256, 512, 4096, 8192])
@pytest.mark.parametrize("group_size", [1, 2, 4, 8])
@pytest.mark.parametrize("scale_major_mode", ["MN", "K"])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16])
def test_fp8_groupwise_group_gemm(
    m,
    n,
    k,
    group_size,
    scale_major_mode,
    out_dtype,
):
    torch.random.manual_seed(0)
    tile_size = 128

    a_val = torch.randn((group_size * m, k), dtype=torch.float, device="cuda")
    b_val = torch.randn(
        (group_size, n, k), dtype=torch.float, device="cuda"
    ) / math.sqrt(k)

    if scale_major_mode == "K":
        a_scale_shape = (group_size * m, k // tile_size)
        b_scale_shape = (group_size, n // tile_size, k // tile_size)
    else:
        a_scale_shape = (k // tile_size, m * group_size)
        b_scale_shape = (group_size, k // tile_size, n // tile_size)
    a_tile_shape = (1, tile_size)
    b_tile_shape = (1, tile_size, tile_size)

    a_fp8, a_scale = quantize_fp8(a_val, a_scale_shape, a_tile_shape, scale_major_mode)
    b_fp8, b_scale = quantize_fp8(b_val, b_scale_shape, b_tile_shape, scale_major_mode)

    m_indptr = torch.arange(0, group_size + 1, dtype=torch.int32, device="cuda") * m

    out = group_gemm_fp8_nt_groupwise(
        a_fp8,
        b_fp8,
        a_scale,
        b_scale,
        m_indptr,
        scale_major_mode=scale_major_mode,
        out_dtype=out_dtype,
    )
    for i in range(group_size):
        a_scale_i = (
            a_scale[m * i : m * (i + 1)]
            if scale_major_mode == "K"
            else a_scale[::, m * i : m * (i + 1)]
        )
        ref_c = gemm_fp8_nt_groupwise_ref(
            a_fp8[m * i : m * (i + 1)],
            b_fp8[i],
            a_scale_i,
            b_scale[i],
            tile_size,
            scale_major_mode,
            out_dtype,
        )
        torch.testing.assert_close(
            out[m * i : m * (i + 1)], ref_c, atol=1e-2, rtol=1e-2
        )


if __name__ == "__main__":
    test_fp8_blockscale_gemm(8192, 8192, 8192, "MN", torch.bfloat16)
    test_fp8_groupwise_gemm(8192, 8192, 8192, "K", torch.bfloat16)
    test_fp8_groupwise_group_gemm(4, 128, 256, 2, "MN", torch.bfloat16)
