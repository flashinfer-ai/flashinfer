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
from flashinfer.gemm import is_cuda_tile_available
from flashinfer.testing.utils import dequantize_fp8, quantize_fp8
from flashinfer.utils import get_compute_capability

pytestmark = pytest.mark.solo


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
    compute_capability = get_compute_capability(torch.device(device="cuda"))
    if compute_capability[0] not in [10, 11, 12]:
        pytest.skip(
            "gemm_fp8_nt_blockscaled is only supported on SM100/103/107, SM110, and SM120/121 GPUs."
        )
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
@pytest.mark.parametrize("backend", ["cutlass", "trtllm", "cutile"])
def test_fp8_groupwise_gemm(
    m,
    n,
    k,
    scale_major_mode,
    backend,
):
    compute_capability = get_compute_capability(torch.device(device="cuda"))
    if backend == "trtllm":
        if compute_capability[0] != 10:
            pytest.skip(
                "gemm_fp8_nt_groupwise is only supported on SM100, SM103, SM107 in trtllm backend."
            )
        if scale_major_mode != "MN":
            pytest.skip("trtllm only supports MN scale_major_mode")
        if k < 256:
            pytest.skip("k < 256")
    if backend == "cutlass" and compute_capability[0] not in [10, 11, 12]:
        pytest.skip(
            "gemm_fp8_nt_groupwise with cutlass backend is only supported on SM100/103/107, SM110, and SM120/121 GPUs."
        )
    if backend == "cutile":
        if compute_capability[0] not in [10, 11, 12]:
            pytest.skip(
                "gemm_fp8_nt_groupwise with cuTile backend is only supported on SM100+ GPUs."
            )
        if scale_major_mode != "K":
            pytest.skip(
                "gemm_fp8_nt_groupwise with cuTile backend currently supports scale_major_mode='K' only."
            )
        if not is_cuda_tile_available():
            pytest.skip(
                "cuda-tile / tileiras compiler not available in this environment."
            )
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


@pytest.mark.parametrize("m", [1, 4, 16, 32])
@pytest.mark.parametrize("n", [128, 256])
@pytest.mark.parametrize("k", [256])
@pytest.mark.parametrize("scale_major_mode", ["MN", "K"])
def test_fp8_groupwise_gemm_small_batch_size(m, n, k, scale_major_mode):
    compute_capability = get_compute_capability(torch.device(device="cuda"))
    if compute_capability[0] != 10:
        pytest.skip(
            "Small-batch gemm_fp8_nt_groupwise dispatch is only relevant on SM100/103."
        )
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

    c = gemm_fp8_nt_groupwise(
        a_fp8,
        b_fp8,
        a_scale,
        b_scale,
        scale_major_mode,
        out_dtype=out_dtype,
        backend="cutlass",
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
    compute_capability = get_compute_capability(torch.device(device="cuda"))
    if group_size > 1 and compute_capability[0] in [
        12,
    ]:
        pytest.skip(
            "group_gemm_fp8_nt_groupwise has correctness issues for num_groups > 1 on SM120/121"
        )
    if compute_capability[0] not in [10, 12]:
        pytest.skip(
            "group_gemm_fp8_nt_groupwise is only supported on SM100/103/107, and SM120/121 GPUs."
        )
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
    compute_capability = get_compute_capability(torch.device(device="cuda"))
    if compute_capability[0] != 10:
        pytest.skip(
            "group_deepgemm_fp8_nt_groupwise is only supported on SM100, SM103, SM107 in trtllm backend."
        )
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
@pytest.mark.parametrize("backend", ["deepgemm", "cake"])
def test_fp8_groupwise_batch_deepgemm_masked(
    m,
    nk,
    group_size,
    out_dtype,
    backend,
):
    compute_capability = get_compute_capability(torch.device(device="cuda"))
    if compute_capability[0] != 10:
        pytest.skip(
            "batch_deepgemm_fp8_nt_groupwise is only supported on SM100, SM103, SM107."
        )
    if backend == "cake" and compute_capability[1] not in (0, 3):
        pytest.skip("Cake batch DeepGEMM FP8 is only supported on SM100 and SM103.")
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
        backend=backend,
    )
    for i in range(group_size):
        torch.testing.assert_close(
            out[i][: masked_m[i]], ref[i][: masked_m[i]], atol=3e-2, rtol=3e-2
        )


@pytest.mark.parametrize(
    ("nk", "expected_m", "masked_m_values"),
    [
        ((6144, 7168), 1228, (1057, 1325, 833, 1197, 1203, 1235)),
        ((6144, 7168), 24, (21, 15, 20, 14, 20, 22)),
        ((7168, 3072), 24, (20, 24, 16, 17, 24, 21)),
        ((4096, 4096), 24, (16, 18, 17, 20, 25, 16)),
        ((4096, 2048), 24, (21, 18, 24, 20, 20, 16)),
    ],
)
def test_fp8_groupwise_batch_deepgemm_cake_deepgemm_benchmark_shapes(
    nk,
    expected_m,
    masked_m_values,
):
    compute_capability = get_compute_capability(torch.device(device="cuda"))
    if compute_capability not in ((10, 0), (10, 3)):
        pytest.skip("Cake batch DeepGEMM FP8 is only supported on SM100 and SM103.")

    torch.random.manual_seed(0)
    group_size, m = 6, 4096
    n, k = nk
    a_fp8 = torch.randint(
        -2, 3, (group_size, m, k), device="cuda", dtype=torch.int8
    ).to(torch.float8_e4m3fn)
    b_fp8 = torch.randint(
        -2, 3, (group_size, n, k), device="cuda", dtype=torch.int8
    ).to(torch.float8_e4m3fn)
    a_scale = torch.ones((group_size, m, k // 128), device="cuda", dtype=torch.float32)
    b_scale = torch.ones(
        (group_size, n // 128, k // 128), device="cuda", dtype=torch.float32
    )
    masked_m = torch.tensor(masked_m_values, device="cuda", dtype=torch.int32)

    out = batch_deepgemm_fp8_nt_groupwise(
        a_fp8,
        b_fp8,
        a_scale,
        b_scale,
        masked_m,
        expected_m,
        out_dtype=torch.bfloat16,
        backend="cake",
    )
    for i, valid_rows in enumerate(masked_m_values):
        for row in {0, valid_rows - 1}:
            reference = torch.mv(b_fp8[i].float(), a_fp8[i, row].float())
            torch.testing.assert_close(
                out[i, row].float(), reference, atol=0.1, rtol=0.1
            )


@pytest.mark.parametrize(
    ("group_size", "m", "nk", "masked_m_values"),
    [
        (
            64,
            256,
            (7168, 2048),
            tuple(200 + (index % 4) * 16 for index in range(64)),
        ),
        (1, 8192, (7168, 2048), (32,)),
        (1, 16384, (4096, 7168), (64,)),
        (8, 1024, (4096, 7168), (397, 653, 935, 441, 733, 555, 725, 320)),
    ],
)
def test_fp8_groupwise_batch_deepgemm_cake_packed_m224_routes(
    group_size,
    m,
    nk,
    masked_m_values,
):
    compute_capability = get_compute_capability(torch.device(device="cuda"))
    if compute_capability not in ((10, 0), (10, 3)):
        pytest.skip("Cake batch DeepGEMM FP8 is only supported on SM100 and SM103.")

    torch.random.manual_seed(0)
    n, k = nk
    a_fp8 = torch.randint(
        -2, 3, (group_size, m, k), device="cuda", dtype=torch.int8
    ).to(torch.float8_e4m3fn)
    b_fp8 = torch.randint(
        -2, 3, (group_size, n, k), device="cuda", dtype=torch.int8
    ).to(torch.float8_e4m3fn)
    a_scale = torch.ones((group_size, m, k // 128), device="cuda", dtype=torch.float32)
    b_scale = torch.ones(
        (group_size, n // 128, k // 128), device="cuda", dtype=torch.float32
    )
    masked_m = torch.tensor(masked_m_values, device="cuda", dtype=torch.int32)
    expected_m = min(sum(masked_m_values) // group_size + 1, m)

    out = batch_deepgemm_fp8_nt_groupwise(
        a_fp8,
        b_fp8,
        a_scale,
        b_scale,
        masked_m,
        expected_m,
        out_dtype=torch.bfloat16,
        backend="cake",
    )
    groups_to_check = range(group_size) if group_size <= 8 else (0, 1, 2, 3, 63)
    for group in groups_to_check:
        valid_rows = masked_m_values[group]
        for row in {0, min(223, valid_rows - 1), valid_rows - 1}:
            reference = torch.mv(b_fp8[group].float(), a_fp8[group, row].float())
            torch.testing.assert_close(
                out[group, row].float(), reference, atol=0.1, rtol=0.1
            )


@pytest.mark.parametrize(
    ("m", "nk", "expected_m"),
    [
        (256, (4096, 7168), 1),
        (256, (4096, 7168), 8),
        (256, (7168, 2048), 1),
        (256, (7168, 2048), 8),
        (512, (4096, 7168), 8),
        (512, (7168, 2048), 8),
    ],
)
def test_fp8_groupwise_batch_deepgemm_cake_native_packed_scales(
    m,
    nk,
    expected_m,
):
    compute_capability = get_compute_capability(torch.device(device="cuda"))
    if compute_capability not in ((10, 0), (10, 3)):
        pytest.skip("Cake batch DeepGEMM FP8 is only supported on SM100 and SM103.")

    torch.random.manual_seed(0)
    group_size = 64
    n, k = nk
    a_fp8 = torch.randint(
        -2, 3, (group_size, m, k), device="cuda", dtype=torch.int8
    ).to(torch.float8_e4m3fn)
    b_fp8 = torch.randint(
        -2, 3, (group_size, n, k), device="cuda", dtype=torch.int8
    ).to(torch.float8_e4m3fn)
    packed_cols = k // 512
    packed_one = 0x7F7F7F7F
    a_scale_storage = torch.full(
        (group_size, packed_cols, m),
        packed_one,
        device="cuda",
        dtype=torch.int32,
    )
    b_scale_storage = torch.full(
        (group_size, packed_cols, n),
        packed_one,
        device="cuda",
        dtype=torch.int32,
    )
    a_scale = a_scale_storage.as_strided(
        (group_size, m, packed_cols), (m * packed_cols, 1, m)
    )
    b_scale = b_scale_storage.as_strided(
        (group_size, n, packed_cols), (n * packed_cols, 1, n)
    )
    masked_m = torch.full(
        (group_size,), expected_m, device="cuda", dtype=torch.int32
    )

    out = batch_deepgemm_fp8_nt_groupwise(
        a_fp8,
        b_fp8,
        a_scale,
        b_scale,
        masked_m,
        expected_m,
        out_dtype=torch.bfloat16,
        backend="cake",
    )
    for group in (0, 1, 31, 63):
        for row in (0, expected_m - 1):
            reference = torch.mv(b_fp8[group].float(), a_fp8[group, row].float())
            torch.testing.assert_close(
                out[group, row].float(), reference, atol=0.1, rtol=0.1
            )


@pytest.mark.parametrize("m", [128, 512])
@pytest.mark.parametrize("n", [256, 4096])
@pytest.mark.parametrize("k", [256, 2048])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16])
def test_gemm_fp8_nt_groupwise_cutile_out_dtypes(m, n, k, out_dtype):
    """cuTile FP8 groupwise GEMM correctness across supported output dtypes.

    ``gemm_fp8_nt_groupwise`` (all backends) is contract-restricted to bf16 /
    fp16 output by ``_validate_fp8_output_dtype`` in ``gemm_base.py``, so we
    do not parametrize over fp32 here. The cuTile kernel itself supports
    fp32 store, but matching the function-level contract avoids divergence
    from the other backends.
    """
    compute_capability = get_compute_capability(torch.device(device="cuda"))
    if compute_capability[0] not in [10, 11, 12]:
        pytest.skip("cuTile fp8 backend requires SM100+ GPUs.")
    if not is_cuda_tile_available():
        pytest.skip("cuda-tile / tileiras compiler not available in this environment.")

    torch.random.manual_seed(0)
    tile_size = 128
    scale_major_mode = "K"

    a_val = torch.randn((m, k), dtype=torch.float, device="cuda")
    b_val = torch.randn((n, k), dtype=torch.float, device="cuda") / math.sqrt(k)

    a_scale_shape = (m, k // tile_size)
    b_scale_shape = (n // tile_size, k // tile_size)
    a_tile_shape = (1, tile_size)
    b_tile_shape = (tile_size, tile_size)

    a_fp8, a_scale = quantize_fp8(a_val, a_scale_shape, a_tile_shape, scale_major_mode)
    b_fp8, b_scale = quantize_fp8(b_val, b_scale_shape, b_tile_shape, scale_major_mode)

    a_dequant = dequantize_fp8(a_fp8, a_scale, scale_major_mode)
    b_dequant = dequantize_fp8(b_fp8, b_scale, scale_major_mode)
    ref_c = einsum(a_dequant, b_dequant, "m k, n k -> m n").to(out_dtype)

    c = gemm_fp8_nt_groupwise(
        a=a_fp8,
        b=b_fp8,
        a_scale=a_scale,
        b_scale=b_scale,
        scale_major_mode=scale_major_mode,
        mma_sm=1,
        out_dtype=out_dtype,
        backend="cutile",
    )
    torch.testing.assert_close(c, ref_c, atol=1e-2, rtol=1e-2)


def test_gemm_fp8_nt_groupwise_cutile_rejects_mn_scale_major():
    """The v1 cuTile fp8 path only supports K-major scales; MN-major must raise."""
    compute_capability = get_compute_capability(torch.device("cuda"))
    if compute_capability[0] not in [10, 11, 12]:
        pytest.skip("cuTile fp8 backend requires SM100+ GPUs.")
    if not is_cuda_tile_available():
        pytest.skip("cuda-tile / tileiras compiler not available in this environment.")

    torch.random.manual_seed(0)
    m, n, k = 128, 1024, 2048
    tile_size = 128

    a_val = torch.randn((m, k), dtype=torch.float, device="cuda")
    b_val = torch.randn((n, k), dtype=torch.float, device="cuda")

    a_scale_shape = (k // tile_size, m)
    b_scale_shape = (k // tile_size, n // tile_size)
    a_tile_shape = (1, tile_size)
    b_tile_shape = (tile_size, tile_size)

    a_fp8, a_scale = quantize_fp8(a_val, a_scale_shape, a_tile_shape, "MN")
    b_fp8, b_scale = quantize_fp8(b_val, b_scale_shape, b_tile_shape, "MN")

    # The @backend_requirement decorator raises ValueError before reaching the
    # cuTile module's own NotImplementedError.
    with pytest.raises(ValueError, match="scale_major_mode='K' only"):
        gemm_fp8_nt_groupwise(
            a=a_fp8,
            b=b_fp8,
            a_scale=a_scale,
            b_scale=b_scale,
            scale_major_mode="MN",
            mma_sm=1,
            out_dtype=torch.bfloat16,
            backend="cutile",
        )


if __name__ == "__main__":
    test_fp8_blockscale_gemm(8192, 8192, 8192, "MN", torch.bfloat16)
    test_fp8_groupwise_gemm(8192, 8192, 8192, "K", backend="cutlass")
    test_fp8_groupwise_group_gemm(4, 128, 256, 2, "MN", torch.bfloat16)
    test_fp8_groupwise_group_deepgemm(256, (128, 512), 4, torch.bfloat16)
    test_fp8_groupwise_batch_deepgemm_masked(256, (128, 512), 8, torch.bfloat16)
