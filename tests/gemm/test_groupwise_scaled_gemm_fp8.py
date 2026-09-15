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
    group_gemm_fp8_nt_groupwise_contiguous,
)
from flashinfer.gemm import is_cuda_tile_available
from flashinfer.cute_dsl import is_cute_dsl_available
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


@pytest.mark.parametrize("backend", ["trtllm", "cutile"])
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
    backend,
):
    """Grouped FP8 groupwise GEMM must match the reference across group sizes and scale modes."""
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
    if backend == "cutile":
        # cuTile group GEMM backend (added in #3426); mirror the capability
        # guards used by test_fp8_groupwise_gemm's cuTile branch.
        if compute_capability[0] not in [10, 11, 12]:
            pytest.skip(
                "group_gemm_fp8_nt_groupwise cuTile backend requires SM100+ GPUs."
            )
        if scale_major_mode != "K":
            pytest.skip(
                "group_gemm_fp8_nt_groupwise cuTile backend supports scale_major_mode='K' only."
            )
        if not is_cuda_tile_available():
            pytest.skip(
                "cuda-tile / tileiras compiler not available in this environment."
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
        backend=backend,
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


@pytest.mark.parametrize("segment_alignment", [128, 256])
@pytest.mark.parametrize("m", [256, 512])
@pytest.mark.parametrize("n", [256])
@pytest.mark.parametrize("k", [256])
@pytest.mark.parametrize("group_size", [4])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16])
def test_fp8_groupwise_group_gemm_cutile_tma_fast_path(
    m,
    n,
    k,
    group_size,
    segment_alignment,
    out_dtype,
):
    """cuTile group GEMM: exercise the ``segment_alignment>=128`` aligned-segment TMA path.

    ``group_gemm_fp8_nt_groupwise(backend="cutile")`` uses the gather kernel for the
    default ``segment_alignment=1`` and only takes the faster aligned-segment TMA
    path (which dispatches into ``ragged_block_scaled_bmm``) when the caller
    guarantees ``segment_alignment`` is a multiple of 128 AND ``block_n == block_k``
    AND ``K % block_k == 0`` (see ``gemm_fp8_nt_groupwise_cutile.py``). Every other
    group-GEMM test leaves ``segment_alignment`` at its default, so that branch had
    no coverage. Equal groups of ``m`` rows (``m`` a multiple of
    ``segment_alignment``) make every ``m_indptr`` offset ``segment_alignment``-
    aligned, satisfying the fast path's caller contract.
    """
    compute_capability = get_compute_capability(torch.device(device="cuda"))
    if compute_capability[0] not in [10, 11, 12]:
        pytest.skip("group_gemm_fp8_nt_groupwise cuTile backend requires SM100+ GPUs.")
    if group_size > 1 and compute_capability[0] == 12:
        # Public-op validator raises for num_groups > 1 on SM120/121; mirror the
        # skip in test_fp8_groupwise_group_gemm so this errors nowhere.
        pytest.skip(
            "group_gemm_fp8_nt_groupwise has correctness issues for num_groups > 1 on SM120/121"
        )
    if not is_cuda_tile_available():
        pytest.skip("cuda-tile / tileiras compiler not available in this environment.")
    # Offsets are i*m; they are segment_alignment-aligned iff m is. Guard the
    # contract so a mismatched parametrization can never silently miscompile.
    assert m % segment_alignment == 0

    torch.random.manual_seed(0)
    tile_size = 128
    scale_major_mode = "K"  # cuTile group GEMM supports K-major scales only.

    a_val = torch.randn((group_size * m, k), dtype=torch.float, device="cuda")
    b_val = torch.randn(
        (group_size, n, k), dtype=torch.float, device="cuda"
    ) / math.sqrt(k)

    a_scale_shape = (group_size * m, k // tile_size)
    b_scale_shape = (group_size, n // tile_size, k // tile_size)
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
        backend="cutile",
        segment_alignment=segment_alignment,
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


@pytest.mark.parametrize("backend", ["trtllm", "cutile"])
@pytest.mark.parametrize(
    "seg_sizes",
    [
        (128, 0, 4, 256),  # zero-length group in the middle
        (0, 256, 128, 4),  # zero-length leading group
        (4, 128, 256, 0),  # zero-length trailing group
    ],
)
@pytest.mark.parametrize("n", [256])
@pytest.mark.parametrize("k", [256])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16])
def test_fp8_groupwise_group_gemm_uneven_segments(
    seg_sizes,
    n,
    k,
    out_dtype,
    backend,
):
    """Grouped FP8 GEMM with non-uniform + zero-length groups.

    Every other group-GEMM test builds equal-size groups via
    ``m_indptr = arange(0, group_size+1) * m``, so the persistent-kernel
    rewrite's uneven / zero-length segment handling is untested. Here the group
    sizes differ and one group is empty (at the front, middle, and back across
    parametrizations). The reference is a per-group loop because rows can no
    longer be reshaped to ``(group_size, m, k)``; the empty group contributes no
    output rows.
    """
    compute_capability = get_compute_capability(torch.device(device="cuda"))
    group_size = len(seg_sizes)
    if group_size > 1 and compute_capability[0] == 12:
        pytest.skip(
            "group_gemm_fp8_nt_groupwise has correctness issues for num_groups > 1 on SM120/121"
        )
    if compute_capability[0] not in [10, 12]:
        pytest.skip(
            "group_gemm_fp8_nt_groupwise is only supported on SM100/103/107, and SM120/121 GPUs."
        )
    if backend == "cutile":
        if compute_capability[0] not in [10, 11, 12]:
            pytest.skip(
                "group_gemm_fp8_nt_groupwise cuTile backend requires SM100+ GPUs."
            )
        if not is_cuda_tile_available():
            pytest.skip(
                "cuda-tile / tileiras compiler not available in this environment."
            )

    torch.random.manual_seed(0)
    tile_size = 128
    # cuTile group GEMM supports K-major scales only; use K for both backends so
    # the two paths are compared on identical inputs.
    scale_major_mode = "K"
    total_m = sum(seg_sizes)

    a_val = torch.randn((total_m, k), dtype=torch.float, device="cuda")
    b_val = torch.randn(
        (group_size, n, k), dtype=torch.float, device="cuda"
    ) / math.sqrt(k)

    a_scale_shape = (total_m, k // tile_size)
    b_scale_shape = (group_size, n // tile_size, k // tile_size)
    a_tile_shape = (1, tile_size)
    b_tile_shape = (1, tile_size, tile_size)

    a_fp8, a_scale = quantize_fp8(a_val, a_scale_shape, a_tile_shape, scale_major_mode)
    b_fp8, b_scale = quantize_fp8(b_val, b_scale_shape, b_tile_shape, scale_major_mode)

    a_dequant = dequantize_fp8(a_fp8, a_scale, scale_major_mode)
    b_dequant = dequantize_fp8(b_fp8, b_scale, scale_major_mode)

    # Prefix-sum the (non-uniform) segment sizes into the m_indptr the op expects.
    offs = [0]
    for s in seg_sizes:
        offs.append(offs[-1] + s)
    m_indptr = torch.tensor(offs, dtype=torch.int32, device="cuda")

    out = group_gemm_fp8_nt_groupwise(
        a_fp8,
        b_fp8,
        a_scale,
        b_scale,
        m_indptr,
        scale_major_mode=scale_major_mode,
        out_dtype=out_dtype,
        backend=backend,
    )

    ref_c = torch.empty((total_m, n), dtype=out_dtype, device="cuda")
    for g in range(group_size):
        start, end = offs[g], offs[g + 1]
        if end == start:
            continue  # empty group -> no rows to fill
        ref_c[start:end] = einsum(
            a_dequant[start:end], b_dequant[g], "m k, n k -> m n"
        ).to(out_dtype)
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


def _assert_fp8_groupwise_group_cute_dsl(
    group_counts, n, k, use_non_default_stream=False, use_out=False
):
    compute_capability = get_compute_capability(torch.device(device="cuda"))
    if compute_capability not in [(10, 0), (10, 3)]:
        pytest.skip("The contiguous grouped CuTe-DSL kernel requires SM100 or SM103")
    if not is_cute_dsl_available():
        pytest.skip("nvidia-cutlass-dsl is not available")

    torch.random.manual_seed(0)
    group_counts = torch.tensor(group_counts, device="cuda")
    group_size = group_counts.numel()
    m = int(group_counts.sum())

    a = torch.randn((m, k), device="cuda", dtype=torch.float32)
    b = torch.randn((group_size, n, k), device="cuda", dtype=torch.float32)
    m_indices = torch.repeat_interleave(
        torch.arange(group_size, device="cuda", dtype=torch.int32), group_counts
    )
    a_fp8, a_scale = quantize_fp8(a, (m, k // 128), (1, 128), "K")
    b_fp8, b_scale = quantize_fp8(
        b, (group_size, n // 128, k // 128), (1, 128, 128), "K"
    )

    a_dequant = dequantize_fp8(a_fp8, a_scale, "K")
    b_dequant = dequantize_fp8(b_fp8, b_scale, "K")
    ref = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    row_start = 0
    for group, group_count in enumerate(group_counts.tolist()):
        row_end = row_start + group_count
        ref[row_start:row_end] = (
            a_dequant[row_start:row_end] @ b_dequant[group].t()
        ).to(torch.bfloat16)
        row_start = row_end

    output = torch.empty_like(ref) if use_out else None
    if use_non_default_stream:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        stream_context = torch.cuda.stream(stream)
    else:
        stream = torch.cuda.current_stream()
        stream_context = torch.cuda.stream(stream)
    with stream_context:
        out = group_gemm_fp8_nt_groupwise_contiguous(
            a_fp8,
            b_fp8,
            a_scale,
            b_scale,
            m_indices,
            out=output,
            validate_indices=True,
        )
        if use_out:
            assert out is output
    stream.synchronize()

    torch.testing.assert_close(out, ref, atol=3e-2, rtol=3e-2)


@pytest.mark.parametrize("n,k", [(256, 128), (256, 4096)])
@pytest.mark.parametrize("use_out", [False, True])
def test_fp8_groupwise_group_cute_dsl_current_stream(n, k, use_out):
    _assert_fp8_groupwise_group_cute_dsl(
        [128, 256, 128, 256], n, k, use_non_default_stream=True, use_out=use_out
    )


@pytest.mark.parametrize(
    "group_counts,n,k",
    [
        ([1], 128, 128),
        ([64], 256, 384),
        ([128, 1], 384, 1024),
        ([128, 64], 128, 128),
        ([0, 128, 1], 256, 384),
        ([128, 128, 1, 0], 256, 1024),
    ],
)
@pytest.mark.parametrize("use_out", [False, True])
def test_fp8_groupwise_group_cute_dsl_partial_final_m(group_counts, n, k, use_out):
    _assert_fp8_groupwise_group_cute_dsl(group_counts, n, k, use_out=use_out)


def _require_grouped_cute_dsl(device="cuda"):
    """Skip unsupported environments before importing the optional kernel."""
    if get_compute_capability(torch.device(device)) not in [(10, 0), (10, 3)]:
        pytest.skip("Requires SM100 or SM103")
    if not is_cute_dsl_available():
        pytest.skip("nvidia-cutlass-dsl is not available")


def _grouped_cute_dsl_inputs(m=256, n=128, k=128, device="cuda"):
    """Small inputs with two experts and unit scales for integration tests."""
    _require_grouped_cute_dsl(device)
    a = torch.randn(m, k, device=device).to(torch.float8_e4m3fn)
    b = torch.randn(2, n, k, device=device).to(torch.float8_e4m3fn)
    a_scale = torch.ones(m, k // 128, device=device)
    b_scale = torch.ones(2, n // 128, k // 128, device=device)
    indices = (torch.arange(m, device=device) >= 128).to(torch.int32)
    return a, b, a_scale, b_scale, indices


@pytest.mark.parametrize("backend", ["deepgemm", "cute_dsl"])
@pytest.mark.parametrize("use_out", [False, True])
def test_grouped_cute_dsl_empty(backend, use_out):
    inputs = _grouped_cute_dsl_inputs(m=0)
    supplied = torch.empty(0, 128, dtype=torch.bfloat16, device="cuda")
    api = (
        group_deepgemm_fp8_nt_groupwise
        if backend == "deepgemm"
        else group_gemm_fp8_nt_groupwise_contiguous
    )
    kwargs = {} if backend == "deepgemm" else {"validate_indices": True}
    out = api(
        *inputs,
        out=supplied if use_out else None,
        **kwargs,
    )
    assert out.shape == (0, 128)
    if use_out:
        assert out is supplied


@pytest.mark.parametrize(
    "case,match",
    [
        ("negative", "0 <= index"),
        ("too_large", "0 <= index"),
        ("unsorted", "sorted"),
        ("mid_tile", "aligned to 128"),
    ],
)
def test_grouped_cute_dsl_rejects_indices(case, match):
    inputs = _grouped_cute_dsl_inputs()
    indices = inputs[-1]
    if case == "negative":
        indices[0] = -1
    elif case == "too_large":
        indices[-1] = 2
    elif case == "unsorted":
        indices[:128], indices[128:] = 1, 0
    else:
        indices[64:] = 1
    with pytest.raises(ValueError, match=match):
        group_gemm_fp8_nt_groupwise_contiguous(*inputs, validate_indices=True)


@pytest.mark.parametrize(
    "case,match",
    [
        ("n", "multiple of 128"),
        ("k", "multiple of 128"),
        ("scale_shape", "a_scale.shape"),
        ("scale_dtype", "torch.float32"),
        ("noncontiguous", "contiguous"),
        ("alignment", "16-byte aligned"),
        ("index_shape", "one-dimensional"),
        ("device", "same CUDA device"),
        ("output_dtype", "bfloat16"),
        ("granularity", "scale_granularity_mnk"),
        ("a_rank", "shape"),
        ("b_rank", "shape"),
        ("a_dtype", "float8_e4m3fn"),
        ("b_dtype", "float8_e4m3fn"),
        ("k_mismatch", "same K"),
        ("index_dtype", "int32"),
        ("index_count", "one expert index per row"),
        ("output_shape", "out.shape"),
    ],
)
def test_grouped_cute_dsl_rejects_metadata(case, match):
    inputs = list(
        _grouped_cute_dsl_inputs(
            n=192 if case == "n" else 128, k=192 if case == "k" else 128
        )
    )
    kwargs = {}
    if case == "scale_shape":
        inputs[2] = torch.ones(256, 2, device="cuda")
    elif case == "scale_dtype":
        inputs[2] = inputs[2].half()
    elif case == "noncontiguous":
        inputs[2] = torch.ones(256, 2, device="cuda")[:, :1]
    elif case == "alignment":
        inputs[2] = torch.ones(257, device="cuda")[1:].view(256, 1)
    elif case == "index_shape":
        inputs[4] = inputs[4].view(256, 1)
    elif case == "device":
        inputs[2] = inputs[2].cpu()
    elif case == "output_dtype":
        kwargs["out_dtype"] = torch.float16
    elif case == "granularity":
        kwargs["scale_granularity_mnk"] = (1, 64, 128)
    elif case == "a_rank":
        inputs[0] = inputs[0].unsqueeze(0)
    elif case == "b_rank":
        inputs[1] = inputs[1][0]
    elif case == "a_dtype":
        inputs[0] = inputs[0].half()
    elif case == "b_dtype":
        inputs[1] = inputs[1].half()
    elif case == "k_mismatch":
        inputs[1] = inputs[1][..., :64].contiguous()
    elif case == "index_dtype":
        inputs[4] = inputs[4].long()
    elif case == "index_count":
        inputs[4] = inputs[4][:-1]
    elif case == "output_shape":
        kwargs["out"] = torch.empty(256, 256, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match=match):
        group_gemm_fp8_nt_groupwise_contiguous(*inputs, **kwargs)


@pytest.mark.parametrize(
    "api",
    [
        group_deepgemm_fp8_nt_groupwise,
        group_gemm_fp8_nt_groupwise_contiguous,
    ],
)
@pytest.mark.parametrize("skip_check", [False, True])
def test_grouped_cute_dsl_has_no_backend_selector(api, skip_check):
    inputs = _grouped_cute_dsl_inputs()
    with pytest.raises(TypeError, match="unexpected keyword argument 'backend'"):
        api(*inputs, backend="auto", skip_check=skip_check)


def test_grouped_cute_dsl_requires_package(monkeypatch):
    import flashinfer.gemm.gemm_base as gb

    inputs = _grouped_cute_dsl_inputs()
    monkeypatch.setattr(gb, "CUTE_DSL_AVAILABLE", False)
    with pytest.raises(ValueError, match="requires nvidia-cutlass-dsl"):
        group_gemm_fp8_nt_groupwise_contiguous(*inputs)


def test_grouped_cute_dsl_requires_arch(monkeypatch):
    import flashinfer.gemm.gemm_base as gb

    inputs = _grouped_cute_dsl_inputs()
    checked = []

    def reject(device):
        checked.append(device)
        raise ValueError("installed DSL cannot compile this architecture")

    monkeypatch.setattr(gb, "_check_cute_dsl_arch", reject)
    with pytest.raises(ValueError, match="cannot compile this architecture"):
        group_gemm_fp8_nt_groupwise_contiguous(*inputs)
    assert checked == [inputs[0].device]


def test_grouped_cute_dsl_graph():
    """Opt-in value checks run before capture; normal execution can be replayed."""
    inputs = _grouped_cute_dsl_inputs()
    out = torch.empty(256, 128, dtype=torch.bfloat16, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        group_gemm_fp8_nt_groupwise_contiguous(*inputs, out=out, validate_indices=True)
    stream.synchronize()
    expected = out.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        with pytest.raises(
            ValueError, match="validate routing before CUDA graph capture"
        ):
            group_gemm_fp8_nt_groupwise_contiguous(
                *inputs, out=out, validate_indices=True
            )
        group_gemm_fp8_nt_groupwise_contiguous(*inputs, out=out)
    out.zero_()
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out, expected, atol=3e-2, rtol=3e-2)


@pytest.mark.parametrize("k", [128, 4096])
def test_grouped_cute_dsl_launch_stream(monkeypatch, k):
    """Observe the actual DSL launch stream after warming compilation."""
    inputs = _grouped_cute_dsl_inputs(k=k)
    import flashinfer.gemm.kernels.grouped_gemm_contiguous_blackwell as mod

    group_gemm_fp8_nt_groupwise_contiguous(*inputs)
    torch.cuda.synchronize()
    key = (inputs[0].device.index, True, 128, k, 2)
    compiled = mod._COMPILED[key]
    launched = []

    def record_launch(*args):
        launched.append(int(args[-1]))
        return compiled(*args)

    monkeypatch.setitem(mod._COMPILED, key, record_launch)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        out = group_gemm_fp8_nt_groupwise_contiguous(*inputs)
    stream.synchronize()
    assert launched == [stream.cuda_stream]
    ref = torch.cat(
        [
            inputs[0][:128].float() @ inputs[1][0].float().T,
            inputs[0][128:].float() @ inputs[1][1].float().T,
        ]
    ).bfloat16()
    torch.testing.assert_close(out, ref, atol=3e-2, rtol=3e-2)


def test_grouped_cute_dsl_noncurrent_device(monkeypatch):
    """Compile and launch on the input device, preserving the caller's context."""
    if torch.cuda.device_count() < 2:
        pytest.skip("Requires two CUDA devices")
    initial = torch.cuda.current_device()
    target = (initial + 1) % torch.cuda.device_count()
    inputs = _grouped_cute_dsl_inputs(m=257, device=f"cuda:{target}")
    import flashinfer.gemm.kernels.grouped_gemm_contiguous_blackwell as mod

    key = (target, False, 128, 128, 2)
    monkeypatch.delitem(mod._COMPILED, key, raising=False)
    compiled_on = []
    compile_fn = mod.cute.compile

    def record_compile(*args, **kwargs):
        compiled_on.append(torch.cuda.current_device())
        return compile_fn(*args, **kwargs)

    monkeypatch.setattr(mod.cute, "compile", record_compile)
    out = group_gemm_fp8_nt_groupwise_contiguous(*inputs)
    torch.cuda.synchronize(target)
    # HardwareInfo may compile an occupancy probe in addition to the GEMM.
    assert compiled_on and set(compiled_on) == {target}
    assert torch.cuda.current_device() == initial
    ref = torch.cat(
        [
            inputs[0][:128].float() @ inputs[1][0].float().T,
            inputs[0][128:].float() @ inputs[1][1].float().T,
        ]
    ).bfloat16()
    torch.testing.assert_close(out, ref, atol=3e-2, rtol=3e-2)


@pytest.mark.parametrize("n,k", [(128, 128), (256, 128), (128, 384), (256, 4096)])
@pytest.mark.parametrize("groups", [1, 2])
@pytest.mark.parametrize("reverse", [False, True])
def test_grouped_cute_dsl_reuses_m_specializations(monkeypatch, n, k, groups, reverse):
    """Reuse aligned/tail artifacts across growth, shrinkage and singleton M."""
    _require_grouped_cute_dsl()
    import flashinfer.gemm.kernels.grouped_gemm_contiguous_blackwell as mod

    monkeypatch.setattr(mod, "_COMPILED", {})
    compile_fn = mod.cute.compile
    builds = []

    def record_compile(kernel, *args, **kwargs):
        if isinstance(kernel, mod.BlockwiseContiguousGroupedGemmKernel):
            builds.append(kernel)
        return compile_fn(kernel, *args, **kwargs)

    monkeypatch.setattr(mod.cute, "compile", record_compile)
    sizes = [1, 65, 129, 257, 513, 128, 256, 384, 512, 127, 1, 256]
    if reverse:
        sizes.reverse()
    for m in sizes:
        inputs = list(_grouped_cute_dsl_inputs(m=m, n=n, k=k))
        if groups == 1:
            inputs[1] = inputs[1][:1]
            inputs[3] = inputs[3][:1]
            inputs[4].zero_()
        out = group_gemm_fp8_nt_groupwise_contiguous(*inputs, validate_indices=True)
        if groups == 1:
            ref = (inputs[0].float() @ inputs[1][0].float().T).bfloat16()
        else:
            ref = torch.cat(
                [
                    inputs[0][:128].float() @ inputs[1][0].float().T,
                    inputs[0][128:].float() @ inputs[1][1].float().T,
                ]
            ).bfloat16()
        torch.testing.assert_close(out, ref, atol=3e-2, rtol=3e-2)
    assert len(builds) == 2
    assert len(mod._COMPILED) == 2


@pytest.mark.parametrize("k", [128, 4096])
def test_grouped_cute_dsl_concurrent_cache_miss(monkeypatch, k):
    """Two callers observing the same cold key compile once and both run correctly."""
    from concurrent.futures import ThreadPoolExecutor
    import threading

    inputs = _grouped_cute_dsl_inputs(k=k)
    import flashinfer.gemm.kernels.grouped_gemm_contiguous_blackwell as mod

    producer = torch.cuda.current_stream()
    initial_lookups = threading.Barrier(2, timeout=60)
    local = threading.local()

    class ConcurrentMissCache(dict):
        def get(self, key, default=None):
            value = super().get(key, default)
            if not getattr(local, "looked_up", False):
                local.looked_up = True
                # Both callers see the missing entry before either can compile.
                initial_lookups.wait()
            return value

    monkeypatch.setattr(mod, "_COMPILED", ConcurrentMissCache())
    compile_fn = mod.cute.compile
    builds = []

    def record_compile(kernel, *args, **kwargs):
        if isinstance(kernel, mod.BlockwiseContiguousGroupedGemmKernel):
            builds.append(kernel)
        return compile_fn(kernel, *args, **kwargs)

    monkeypatch.setattr(mod.cute, "compile", record_compile)

    def call():
        with torch.cuda.device(inputs[0].device):
            stream = torch.cuda.Stream()
            stream.wait_stream(producer)
            with torch.cuda.stream(stream):
                out = group_gemm_fp8_nt_groupwise_contiguous(*inputs)
            stream.synchronize()
            return out

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(call) for _ in range(2)]
        outputs = [future.result(timeout=120) for future in futures]
    ref = torch.cat(
        [
            inputs[0][:128].float() @ inputs[1][0].float().T,
            inputs[0][128:].float() @ inputs[1][1].float().T,
        ]
    ).bfloat16()
    for out in outputs:
        torch.testing.assert_close(out, ref, atol=3e-2, rtol=3e-2)
    assert len(builds) == 1
    assert len(mod._COMPILED) == 1


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
    compute_capability = get_compute_capability(torch.device(device="cuda"))
    if compute_capability[0] != 10:
        pytest.skip(
            "batch_deepgemm_fp8_nt_groupwise is only supported on SM100, SM103, SM107."
        )
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
