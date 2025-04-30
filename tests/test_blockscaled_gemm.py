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

import pytest
import torch

import flashinfer
from flashinfer.gemm import gemm_fp8_nt_blockscaled


def native_w8a8_block_fp8_matmul(A, B, As, Bs, block_size, output_dtype=torch.float16):
    """Matrix multiplication with block-wise quantization using native torch."""
    A = A.to(torch.float32)
    B = B.to(torch.float32)
    assert A.shape[-1] == B.shape[-1]
    assert B.ndim == 2 and B.is_contiguous() and Bs.ndim == 2
    assert len(block_size) == 2
    block_n, block_k = block_size[0], block_size[1]
    assert (A.shape[-1] + block_k - 1) // block_k == As.shape[-1]
    assert A.shape[:-1] == As.shape[:-1]

    M = A.numel() // A.shape[-1]
    N, K = B.shape
    origin_C_shape = A.shape[:-1] + (N,)
    A = A.reshape(M, A.shape[-1])
    As = As.reshape(M, As.shape[-1])
    n_tiles = (N + block_n - 1) // block_n
    k_tiles = (K + block_k - 1) // block_k
    assert n_tiles == Bs.shape[0]
    assert k_tiles == Bs.shape[1]

    C_shape = (M, N)
    C = torch.zeros(C_shape, dtype=torch.float32, device=A.device)

    A_tiles = [A[:, i * block_k : min((i + 1) * block_k, K)] for i in range(k_tiles)]
    B_tiles = [
        [
            B[
                j * block_n : min((j + 1) * block_n, N),
                i * block_k : min((i + 1) * block_k, K),
            ]
            for i in range(k_tiles)
        ]
        for j in range(n_tiles)
    ]
    C_tiles = [C[:, j * block_n : min((j + 1) * block_n, N)] for j in range(n_tiles)]
    As_tiles = [As[:, i : i + 1] for i in range(k_tiles)]

    for i in range(k_tiles):
        for j in range(n_tiles):
            a = A_tiles[i]
            b = B_tiles[j][i]
            c = C_tiles[j]
            s = As_tiles[i] * Bs[j][i]
            c[:, :] += torch.matmul(a, b.t()) * s

    C = C.reshape(origin_C_shape).to(output_dtype)
    return C


@pytest.mark.parametrize("m", [128, 256, 512, 4096, 8192])
@pytest.mark.parametrize("n", [128, 256, 512, 4096, 8192])
@pytest.mark.parametrize("k", [128, 256, 512, 4096, 8192])
@pytest.mark.parametrize("out_dtype", [torch.float32])
def test_fp8_blockscale_gemm(
    m,
    n,
    k,
    out_dtype,
):
    torch.random.manual_seed(0)
    tile_size = 128
    factor_for_scale = 0.01
    fp8_info = torch.finfo(torch.float8_e4m3fn)
    fp8_max, fp8_min = fp8_info.max, fp8_info.min

    a_fp32 = (torch.randn((m, k), device="cuda", dtype=torch.float)) * 2 * fp8_max
    a_fp8 = a_fp32.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)

    b_fp32 = (torch.randn((n, k), device="cuda", dtype=torch.float)) * 2 * fp8_max
    b_fp8 = b_fp32.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)

    a_scale = (
        torch.ones((k // tile_size, m), dtype=torch.float32, device="cuda")
        * factor_for_scale
    )
    b_scale = (
        torch.ones((n // tile_size, k // tile_size), dtype=torch.float32, device="cuda")
        * factor_for_scale
    )

    c = gemm_fp8_nt_blockscaled(a_fp8, b_fp8, a_scale, b_scale, out_dtype=out_dtype)
    out_dtype = torch.float
    a_scale_naive = torch.transpose(a_scale, 0, 1).contiguous()
    ref_c = native_w8a8_block_fp8_matmul(
        a_fp8.to("cpu"),
        b_fp8.to("cpu"),
        a_scale_naive.to("cpu"),
        b_scale.to("cpu"),
        [tile_size, tile_size],
        out_dtype,
    )
    print(c, ref_c)
    torch.testing.assert_close(c.cpu(), ref_c.to(c.dtype), atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    test_fp8_blockscale_gemm(8192, 8192, 8192, torch.bfloat16)
