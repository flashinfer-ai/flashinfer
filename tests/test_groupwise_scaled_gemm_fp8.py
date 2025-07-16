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
from einops import einsum, rearrange

from flashinfer.gemm import (
    batch_deepgemm_fp8_nt_groupwise,
    gemm_fp8_nt_blockscaled,
    gemm_fp8_nt_groupwise,
    group_deepgemm_fp8_nt_groupwise,
    group_gemm_fp8_nt_groupwise,
)
from flashinfer.utils import quantize_fp8, dequantize_fp8


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


@pytest.mark.parametrize("m_per_group", [128, 256])
@pytest.mark.parametrize("n", [128, 512, 4096])
@pytest.mark.parametrize("k", [128, 512, 4096])
@pytest.mark.parametrize("group_size", [1, 4, 8])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16])
def test_fp8_groupwise_group_deepgemm(
    m_per_group,
    n,
    k,
    group_size,
    out_dtype,
):
    torch.random.manual_seed(0)
    m = m_per_group * group_size
    a = torch.randn((m, k), device="cuda", dtype=torch.float32)
    b = torch.randn((group_size, n, k), device="cuda", dtype=torch.float32)
    m_indptr = torch.empty((m,), device="cuda", dtype=torch.int32)
    a_fp8, a_scale = quantize_fp8(a, (m, k // 128), (1, 128), "K")
    b_fp8, b_scale = quantize_fp8(
        b, (group_size, n // 128, k // 128), (1, 128, 128), "K"
    )
    a_dequant = dequantize_fp8(a_fp8, a_scale, "K")
    b_dequant = dequantize_fp8(b_fp8, b_scale, "K")

    ref = torch.empty((m, n), device="cuda", dtype=out_dtype)

    for i in range(group_size):
        r = slice(i * m_per_group, (i + 1) * m_per_group)
        m_indptr[r] = i
        ref[r] = a_dequant[r] @ b_dequant[i].t()

    out = group_deepgemm_fp8_nt_groupwise(
        a_fp8,
        b_fp8,
        a_scale,
        b_scale,
        m_indptr,
        out_dtype=out_dtype,
    )
    torch.testing.assert_close(out, ref, atol=3e-2, rtol=3e-2)


@pytest.mark.parametrize("m", [128, 256, 512, 1024])
@pytest.mark.parametrize("nk", [(128, 512), (512, 128), (4096, 7168), (7168, 2048)])
@pytest.mark.parametrize("group_size", [1, 4, 8, 64, 128, 256])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16])
def test_fp8_groupwise_batch_deepgemm_masked(
    m,
    nk,
    group_size,
    out_dtype,
):
    torch.random.manual_seed(0)
    n, k = nk
    a = torch.randn((group_size, m, k), device="cuda", dtype=torch.float32)
    b = torch.randn((group_size, n, k), device="cuda", dtype=torch.float32)
    masked_m = torch.randint(0, m, (group_size,), device="cuda", dtype=torch.int32)

    a_fp8, a_scale = quantize_fp8(a, (group_size, m, k // 128), (1, 1, 128), "K")
    b_fp8, b_scale = quantize_fp8(
        b, (group_size, n // 128, k // 128), (1, 128, 128), "K"
    )

    a_dequant = dequantize_fp8(a_fp8, a_scale, "K")
    b_dequant = dequantize_fp8(b_fp8, b_scale, "K")
    ref = torch.einsum("bmk,bnk->bmn", a_dequant, b_dequant).to(out_dtype)

    expected_m = min(int(masked_m.float().mean()) + 1, m)

    out = batch_deepgemm_fp8_nt_groupwise(
        a_fp8,
        b_fp8,
        a_scale,
        b_scale,
        masked_m,
        expected_m,
        out_dtype=out_dtype,
    )
    for i in range(group_size):
        torch.testing.assert_close(
            out[i][: masked_m[i]], ref[i][: masked_m[i]], atol=3e-2, rtol=3e-2
        )


if __name__ == "__main__":
    test_fp8_blockscale_gemm(8192, 8192, 8192, "MN", torch.bfloat16)
    test_fp8_groupwise_gemm(8192, 8192, 8192, "K", torch.bfloat16)
    test_fp8_groupwise_group_gemm(4, 128, 256, 2, "MN", torch.bfloat16)
    test_fp8_groupwise_group_deepgemm(256, 128, 128, 4, torch.bfloat16)
    test_fp8_groupwise_batch_deepgemm_masked(256, (128, 512), 8, torch.bfloat16)
