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
from einops import einsum

from flashinfer.gemm import (
    batch_deepgemm_fp8_nt_groupwise,
    gemm_fp8_nt_blockscaled,
    gemm_fp8_nt_groupwise,
    group_deepgemm_fp8_nt_groupwise,
    group_gemm_fp8_nt_groupwise,
)
from flashinfer.testing.utils import dequantize_fp8, quantize_fp8


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

    a_dequant = dequantize_fp8(a_fp8, a_scale, scale_major_mode)
    b_dequant = dequantize_fp8(b_fp8, b_scale, scale_major_mode)
    ref_c = einsum(a_dequant, b_dequant, "m k, n k -> m n").to(out_dtype)

    c = gemm_fp8_nt_blockscaled(
        a_fp8, b_fp8, a_scale, b_scale, scale_major_mode, out_dtype=out_dtype
    )
    torch.testing.assert_close(c, ref_c, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("m", [128, 256, 512, 4096, 8192])
@pytest.mark.parametrize("n", [128, 256, 512, 4096, 8192])
@pytest.mark.parametrize("k", [128, 256, 512, 4096, 8192])
@pytest.mark.parametrize("scale_major_mode", ["MN", "K"])
@pytest.mark.parametrize("backend", ["cutlass", "trtllm"])
def test_fp8_groupwise_gemm(
    m,
    n,
    k,
    scale_major_mode,
    backend,
):
    if backend == "trtllm":
        if scale_major_mode != "MN":
            pytest.skip("trtllm only supports MN scale_major_mode")
        if k < 256:
            pytest.skip("k < 256")

    torch.random.manual_seed(0)
    tile_size = 128
    out_dtype = torch.bfloat16

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

    a_dequant = dequantize_fp8(a_fp8, a_scale, scale_major_mode)
    b_dequant = dequantize_fp8(b_fp8, b_scale, scale_major_mode)
    ref_c = einsum(a_dequant, b_dequant, "m k, n k -> m n").to(out_dtype)

    if backend == "trtllm":
        b_scale = b_scale.t().contiguous()

    c = gemm_fp8_nt_groupwise(
        a_fp8,
        b_fp8,
        a_scale,
        b_scale,
        scale_major_mode,
        out_dtype=out_dtype,
        backend=backend,
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

    a_dequant = dequantize_fp8(a_fp8, a_scale, scale_major_mode)
    b_dequant = dequantize_fp8(b_fp8, b_scale, scale_major_mode)

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
    ref_c = (
        einsum(
            a_dequant.view((group_size, m, k)),
            b_dequant,
            "b m k, b n k -> b m n",
        )
        .view((group_size * m, n))
        .to(out_dtype)
    )
    torch.testing.assert_close(out, ref_c, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("m", [128, 256, 512, 1024])
@pytest.mark.parametrize("nk", [(128, 512), (512, 128), (4096, 7168), (7168, 2048)])
@pytest.mark.parametrize("group_size", [1, 4, 8, 64, 128, 256])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16])
def test_fp8_groupwise_group_deepgemm(
    m,
    nk,
    group_size,
    out_dtype,
):
    torch.random.manual_seed(0)
    m_per_group = m // group_size
    if m_per_group < 128:
        return
    n, k = nk
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
    test_fp8_groupwise_group_deepgemm(256, (128, 512), 4, torch.bfloat16)
    test_fp8_groupwise_batch_deepgemm_masked(256, (128, 512), 8, torch.bfloat16)
