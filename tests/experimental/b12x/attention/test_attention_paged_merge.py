from __future__ import annotations

import torch

import cutlass

from b12x.attention.paged._forward import (
    _build_merge_kernel,
    _to_kernel_tensor,
    _torch_to_cutlass_dtype,
)
from b12x.attention.paged.merge import (
    LagunaVerifierMergeKernel,
    PagedPersistentMergeKernel,
)
from b12x._lib.compiler import compile as b12x_compile
from b12x._lib.utils import current_cuda_stream

from b12x.testing.reference.helpers import require_b12x


def test_b1_regular_decode_graph_uses_four_merge_warps() -> None:
    b1 = _build_merge_kernel(
        torch.bfloat16,
        128,
        1,
        1,
        True,
        True,
        False,
        False,
        False,
    )
    b2 = _build_merge_kernel(
        torch.bfloat16,
        128,
        2,
        1,
        True,
        True,
        False,
        False,
        False,
    )

    assert b1.bdx * b1.bdy // 32 == 4
    assert b2.bdx * b2.bdy // 32 == 3


def _merge_reference_base2(
    partial_o: torch.Tensor,
    partial_lse: torch.Tensor,
    merge_indptr: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    total_rows = merge_indptr.numel() - 1
    num_heads = partial_o.shape[1]
    head_dim = partial_o.shape[2]
    out = torch.empty(
        total_rows, num_heads, head_dim, dtype=torch.float32, device=partial_o.device
    )
    lse = torch.empty(
        num_heads, total_rows, dtype=torch.float32, device=partial_o.device
    )

    for row_idx in range(total_rows):
        start_idx = int(merge_indptr[row_idx].item())
        end_idx = int(merge_indptr[row_idx + 1].item())
        if start_idx == end_idx:
            out[row_idx].zero_()
            lse[:, row_idx] = -torch.inf
            continue
        if end_idx == start_idx + 1:
            out[row_idx] = partial_o[start_idx].to(torch.float32)
            lse[:, row_idx] = partial_lse[start_idx]
            continue

        row_lse = partial_lse[start_idx:end_idx].to(torch.float32)
        row_out = partial_o[start_idx:end_idx].to(torch.float32)
        lse_max = row_lse.max(dim=0).values
        weights = torch.pow(2.0, row_lse - lse_max)
        weight_sum = weights.sum(dim=0)
        out[row_idx] = (row_out * weights[:, :, None]).sum(dim=0) / weight_sum[:, None]
        lse[:, row_idx] = torch.log2(weight_sum) + lse_max
    return out, lse


def _make_merge_problem(
    *,
    dtype: torch.dtype = torch.bfloat16,
    counts: list[int] | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    device = "cuda"
    counts = [0, 1, 3, 1, 4] if counts is None else counts
    num_heads = 3
    head_dim = 256
    nnz = sum(counts)
    partial_o = (torch.randn(nnz, num_heads, head_dim, device=device) / 4).to(dtype)
    partial_lse = (
        torch.randn(nnz, num_heads, dtype=torch.float32, device=device) * 2 - 1
    )
    merge_indptr_list = [0]
    for count in counts:
        merge_indptr_list.append(merge_indptr_list[-1] + count)
    merge_indptr = torch.tensor(merge_indptr_list, dtype=torch.int32, device=device)
    output = torch.full(
        (len(counts), num_heads, head_dim), -77.0, dtype=dtype, device=device
    )
    lse = torch.full(
        (num_heads, len(counts)), -99.0, dtype=torch.float32, device=device
    )
    total_rows_ptr = torch.tensor([len(counts)], dtype=torch.int32, device=device)
    cache_seqlens = torch.tensor(
        [max(count, 1) * 64 for count in counts], dtype=torch.int32, device=device
    )
    kv_chunk_size_ptr = torch.tensor([64], dtype=torch.int32, device=device)
    return (
        partial_o,
        partial_lse,
        merge_indptr,
        output,
        lse,
        total_rows_ptr,
        cache_seqlens,
        kv_chunk_size_ptr,
    )


def _make_regular_decode_graph_merge_problem(
    *,
    dtype: torch.dtype = torch.bfloat16,
    counts: list[int],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    (
        partial_o,
        partial_lse,
        merge_indptr,
        output,
        lse,
        total_rows_ptr,
        cache_seqlens,
        kv_chunk_size_ptr,
    ) = _make_merge_problem(dtype=dtype, counts=counts)
    max_chunks_per_req = max(counts)
    fixed_partial_o = torch.zeros(
        (len(counts) * max_chunks_per_req, partial_o.shape[1], partial_o.shape[2]),
        dtype=partial_o.dtype,
        device=partial_o.device,
    )
    fixed_partial_lse = torch.zeros(
        (len(counts) * max_chunks_per_req, partial_lse.shape[1]),
        dtype=partial_lse.dtype,
        device=partial_lse.device,
    )
    for row_idx, count in enumerate(counts):
        if count <= 0:
            continue
        src_start = int(merge_indptr[row_idx].item())
        src_end = int(merge_indptr[row_idx + 1].item())
        dst_start = row_idx * max_chunks_per_req
        fixed_partial_o[dst_start : dst_start + count].copy_(
            partial_o[src_start:src_end]
        )
        fixed_partial_lse[dst_start : dst_start + count].copy_(
            partial_lse[src_start:src_end]
        )
    return (
        fixed_partial_o,
        fixed_partial_lse,
        merge_indptr,
        output,
        lse,
        total_rows_ptr,
        cache_seqlens,
        kv_chunk_size_ptr,
    )


def _run_merge_kernel(
    kernel: PagedPersistentMergeKernel,
    partial_o: torch.Tensor,
    partial_lse: torch.Tensor,
    merge_indptr: torch.Tensor,
    cache_seqlens: torch.Tensor,
    kv_chunk_size_ptr: torch.Tensor,
    output: torch.Tensor,
    lse: torch.Tensor,
    total_rows_ptr: torch.Tensor | None,
) -> None:
    args = (
        _to_kernel_tensor(partial_o, _torch_to_cutlass_dtype(partial_o.dtype)),
        _to_kernel_tensor(partial_lse, cutlass.Float32),
        _to_kernel_tensor(merge_indptr, cutlass.Int32, assumed_align=4),
        _to_kernel_tensor(cache_seqlens, cutlass.Int32, assumed_align=4),
        _to_kernel_tensor(kv_chunk_size_ptr, cutlass.Int32, assumed_align=4),
        _to_kernel_tensor(output, _torch_to_cutlass_dtype(output.dtype)),
        _to_kernel_tensor(lse, cutlass.Float32),
        None
        if total_rows_ptr is None
        else _to_kernel_tensor(total_rows_ptr, cutlass.Int32, assumed_align=4),
        current_cuda_stream(),
    )
    compiled = b12x_compile(kernel, *args)
    compiled(*args)


def _run_laguna_verify_merge_kernel(
    kernel: LagunaVerifierMergeKernel,
    partial_o: torch.Tensor,
    partial_lse: torch.Tensor,
    cache_seqlens: torch.Tensor,
    output: torch.Tensor,
    lse: torch.Tensor,
) -> None:
    args = (
        _to_kernel_tensor(partial_o, cutlass.BFloat16),
        _to_kernel_tensor(partial_lse, cutlass.Float32),
        _to_kernel_tensor(cache_seqlens, cutlass.Int32, assumed_align=4),
        _to_kernel_tensor(output, cutlass.BFloat16),
        _to_kernel_tensor(lse, cutlass.Float32),
        current_cuda_stream(),
    )
    compiled = b12x_compile(kernel, *args)
    compiled(*args)


@torch.inference_mode()
def test_paged_persistent_merge_matches_reference() -> None:
    require_b12x()
    (
        partial_o,
        partial_lse,
        merge_indptr,
        output,
        lse,
        total_rows_ptr,
        cache_seqlens,
        kv_chunk_size_ptr,
    ) = _make_merge_problem()
    kernel = PagedPersistentMergeKernel(
        cutlass.BFloat16,
        cutlass.BFloat16,
        head_dim=256,
        persistent_ctas=2,
    )

    _run_merge_kernel(
        kernel,
        partial_o,
        partial_lse,
        merge_indptr,
        cache_seqlens,
        kv_chunk_size_ptr,
        output,
        lse,
        total_rows_ptr,
    )
    torch.cuda.synchronize()

    ref_out, ref_lse = _merge_reference_base2(partial_o, partial_lse, merge_indptr)
    assert torch.allclose(output.to(torch.float32), ref_out, atol=2e-2, rtol=2e-2)
    assert torch.allclose(lse, ref_lse, atol=2e-3, rtol=2e-3)


@torch.inference_mode()
def test_paged_persistent_merge_respects_dynamic_total_rows() -> None:
    require_b12x()
    (
        partial_o,
        partial_lse,
        merge_indptr,
        output,
        lse,
        total_rows_ptr,
        cache_seqlens,
        kv_chunk_size_ptr,
    ) = _make_merge_problem()
    total_rows_ptr[0] = 3
    kernel = PagedPersistentMergeKernel(
        cutlass.BFloat16,
        cutlass.BFloat16,
        head_dim=256,
        persistent_ctas=2,
    )

    output_before = output.clone()
    lse_before = lse.clone()
    _run_merge_kernel(
        kernel,
        partial_o,
        partial_lse,
        merge_indptr,
        cache_seqlens,
        kv_chunk_size_ptr,
        output,
        lse,
        total_rows_ptr,
    )
    torch.cuda.synchronize()

    ref_out, ref_lse = _merge_reference_base2(partial_o, partial_lse, merge_indptr)
    assert torch.allclose(
        output[:3].to(torch.float32), ref_out[:3], atol=2e-2, rtol=2e-2
    )
    assert torch.allclose(lse[:, :3], ref_lse[:, :3], atol=2e-3, rtol=2e-3)
    assert torch.equal(output[3:], output_before[3:])
    assert torch.equal(lse[:, 3:], lse_before[:, 3:])


@torch.inference_mode()
def test_paged_persistent_merge_handles_more_than_one_partial_per_ty() -> None:
    require_b12x()
    (
        partial_o,
        partial_lse,
        merge_indptr,
        output,
        lse,
        total_rows_ptr,
        cache_seqlens,
        kv_chunk_size_ptr,
    ) = _make_merge_problem(counts=[8, 7, 5])
    kernel = PagedPersistentMergeKernel(
        cutlass.BFloat16,
        cutlass.BFloat16,
        head_dim=256,
        persistent_ctas=2,
    )

    _run_merge_kernel(
        kernel,
        partial_o,
        partial_lse,
        merge_indptr,
        cache_seqlens,
        kv_chunk_size_ptr,
        output,
        lse,
        total_rows_ptr,
    )
    torch.cuda.synchronize()

    ref_out, ref_lse = _merge_reference_base2(partial_o, partial_lse, merge_indptr)
    assert torch.allclose(output.to(torch.float32), ref_out, atol=2e-2, rtol=2e-2)
    assert torch.allclose(lse, ref_lse, atol=2e-3, rtol=2e-3)


@torch.inference_mode()
def test_paged_persistent_merge_regular_decode_graph_matches_reference() -> None:
    require_b12x()
    counts = [1, 3, 2, 4]
    (
        partial_o,
        partial_lse,
        merge_indptr,
        output,
        lse,
        total_rows_ptr,
        cache_seqlens,
        kv_chunk_size_ptr,
    ) = _make_regular_decode_graph_merge_problem(counts=counts)
    kernel = PagedPersistentMergeKernel(
        cutlass.BFloat16,
        cutlass.BFloat16,
        head_dim=256,
        persistent_ctas=2,
        direct_grid=True,
        regular_decode_graph=True,
    )

    _run_merge_kernel(
        kernel,
        partial_o,
        partial_lse,
        merge_indptr,
        cache_seqlens,
        kv_chunk_size_ptr,
        output,
        lse,
        total_rows_ptr,
    )
    torch.cuda.synchronize()

    compact_o = []
    compact_lse = []
    max_chunks_per_req = max(counts)
    for row_idx, count in enumerate(counts):
        row_start = row_idx * max_chunks_per_req
        row_end = row_start + count
        compact_o.append(partial_o[row_start:row_end])
        compact_lse.append(partial_lse[row_start:row_end])
    ref_out, ref_lse = _merge_reference_base2(
        torch.cat(compact_o, dim=0), torch.cat(compact_lse, dim=0), merge_indptr
    )
    assert torch.allclose(output.to(torch.float32), ref_out, atol=2e-2, rtol=2e-2)
    assert torch.allclose(lse, ref_lse, atol=2e-3, rtol=2e-3)


@torch.inference_mode()
def test_laguna_verifier_merge_matches_reference() -> None:
    require_b12x()
    torch.manual_seed(117)
    rows = 8
    num_heads = 24
    head_dim = 128
    chunks_per_row = 4
    partial_o = (
        torch.randn(
            rows * chunks_per_row,
            num_heads,
            head_dim,
            device="cuda",
        )
        / 4
    ).to(torch.bfloat16)
    partial_lse = (
        torch.randn(
            rows * chunks_per_row,
            num_heads,
            dtype=torch.float32,
            device="cuda",
        )
        * 2
        - 1
    )
    cache_seqlens = torch.tensor([256], dtype=torch.int32, device="cuda")
    output = torch.full(
        (rows, num_heads, head_dim),
        -77.0,
        dtype=torch.bfloat16,
        device="cuda",
    )
    lse = torch.full(
        (num_heads, rows),
        -99.0,
        dtype=torch.float32,
        device="cuda",
    )
    kernel = LagunaVerifierMergeKernel(rows * num_heads)

    _run_laguna_verify_merge_kernel(
        kernel,
        partial_o,
        partial_lse,
        cache_seqlens,
        output,
        lse,
    )
    torch.cuda.synchronize()

    merge_indptr = torch.arange(
        0,
        (rows + 1) * chunks_per_row,
        chunks_per_row,
        dtype=torch.int32,
        device="cuda",
    )
    ref_out, ref_lse = _merge_reference_base2(
        partial_o,
        partial_lse,
        merge_indptr,
    )
    assert torch.allclose(output.to(torch.float32), ref_out, atol=2e-2, rtol=2e-2)
    assert torch.allclose(lse, ref_lse, atol=2e-3, rtol=2e-3)
