import pytest
import torch

import flashinfer
from flashinfer.page import get_page_module

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def _make_indptr(lengths, device="cuda:0"):
    if not isinstance(lengths, torch.Tensor):
        lengths = torch.tensor(lengths, dtype=torch.int32, device=device)
    else:
        lengths = lengths.to(device=device, dtype=torch.int32)
    indptr = torch.empty(lengths.numel() + 1, dtype=torch.int32, device=device)
    indptr[0] = 0
    indptr[1:] = torch.cumsum(lengths, dim=0)
    return lengths, indptr


def _batch_indices_positions_ref(append_indptr, seq_lens):
    append_indptr_cpu = append_indptr.cpu()
    seq_lens_cpu = seq_lens.cpu()
    nnz = int(append_indptr_cpu[-1].item())
    batch_indices = torch.empty(nnz, dtype=torch.int32)
    positions = torch.empty(nnz, dtype=torch.int32)
    for batch_idx in range(seq_lens_cpu.numel()):
        batch_start = int(append_indptr_cpu[batch_idx].item())
        batch_end = int(append_indptr_cpu[batch_idx + 1].item())
        seq_len = int(seq_lens_cpu[batch_idx].item())
        for offset in range(batch_start, batch_end):
            batch_indices[offset] = batch_idx
            positions[offset] = offset + seq_len - batch_end
    return batch_indices.to(append_indptr.device), positions.to(append_indptr.device)


def _get_batch_indices_positions_cuda(append_indptr, seq_lens):
    nnz = int(append_indptr[-1].item())
    batch_indices = torch.full(
        (nnz,), -1, dtype=torch.int32, device=append_indptr.device
    )
    positions = torch.full((nnz,), -2, dtype=torch.int32, device=append_indptr.device)
    get_page_module().get_batch_indices_positions_cuda(
        append_indptr, seq_lens, batch_indices, positions
    )
    return batch_indices, positions


@pytest.mark.parametrize(
    ("append_lengths", "seq_lens"),
    [
        ([1, 2, 3, 4], [5, 5, 5, 5]),
        ([45, 8, 25, 22], [45, 8, 25, 22]),
        ([0, 3, 0, 5, 1], [7, 10, 2, 8, 6]),
        ([7], [11]),
        ([0, 0, 0], [3, 4, 5]),
        ([], []),
    ],
)
def test_get_batch_indices_positions_cuda(append_lengths, seq_lens):
    _, append_indptr = _make_indptr(append_lengths)
    seq_lens = torch.tensor(seq_lens, dtype=torch.int32, device="cuda:0")

    batch_indices, positions = _get_batch_indices_positions_cuda(
        append_indptr, seq_lens
    )
    ref_batch_indices, ref_positions = _batch_indices_positions_ref(
        append_indptr, seq_lens
    )

    torch.testing.assert_close(batch_indices, ref_batch_indices)
    torch.testing.assert_close(positions, ref_positions)


def test_get_batch_indices_positions_cuda_many_rows():
    append_lengths = torch.arange(1024, dtype=torch.int32, device="cuda:0") % 9
    seq_lens = append_lengths + (
        torch.arange(1024, dtype=torch.int32, device="cuda:0") % 17
    )
    _, append_indptr = _make_indptr(append_lengths)

    batch_indices, positions = _get_batch_indices_positions_cuda(
        append_indptr, seq_lens
    )
    ref_batch_indices, ref_positions = _batch_indices_positions_ref(
        append_indptr, seq_lens
    )

    torch.testing.assert_close(batch_indices, ref_batch_indices)
    torch.testing.assert_close(positions, ref_positions)


def test_get_batch_indices_positions_cuda_rejects_int64_inputs():
    append_lengths, append_indptr = _make_indptr([1, 2, 3])
    seq_lens = append_lengths.to(torch.int64)
    nnz = int(append_indptr[-1].item())
    batch_indices = torch.empty((nnz,), dtype=torch.int32, device="cuda:0")
    positions = torch.empty((nnz,), dtype=torch.int32, device="cuda:0")

    with pytest.raises(Exception):
        get_page_module().get_batch_indices_positions_cuda(
            append_indptr, seq_lens, batch_indices, positions
        )


def test_append_paged_kv_cache_with_cuda_batch_indices_positions():
    torch.manual_seed(0)
    append_lengths, append_indptr = _make_indptr([5, 2, 7])
    seq_lens = append_lengths.clone()
    page_size = 4
    num_kv_heads = 2
    head_dim = 8
    nnz = int(append_indptr[-1].item())

    num_pages_per_req = (seq_lens + page_size - 1) // page_size
    kv_indptr = torch.empty(seq_lens.numel() + 1, dtype=torch.int32, device="cuda:0")
    kv_indptr[0] = 0
    kv_indptr[1:] = torch.cumsum(num_pages_per_req, dim=0)
    num_pages = int(kv_indptr[-1].item())
    kv_indices = torch.arange(num_pages, dtype=torch.int32, device="cuda:0")
    kv_last_page_len = ((seq_lens - 1) % page_size + 1).to(torch.int32)

    append_key = torch.randn(
        nnz, num_kv_heads, head_dim, dtype=torch.float16, device="cuda:0"
    )
    append_value = torch.randn_like(append_key)
    paged_kv_cache = torch.zeros(
        num_pages,
        2,
        page_size,
        num_kv_heads,
        head_dim,
        dtype=torch.float16,
        device="cuda:0",
    )
    batch_indices, positions = _get_batch_indices_positions_cuda(
        append_indptr, seq_lens
    )

    flashinfer.append_paged_kv_cache(
        append_key,
        append_value,
        batch_indices,
        positions,
        paged_kv_cache,
        kv_indices,
        kv_indptr,
        kv_last_page_len,
    )

    batch_indices_cpu = batch_indices.cpu()
    positions_cpu = positions.cpu()
    kv_indptr_cpu = kv_indptr.cpu()
    kv_indices_cpu = kv_indices.cpu()
    for i in range(nnz):
        batch_idx = int(batch_indices_cpu[i].item())
        pos = int(positions_cpu[i].item())
        page_iter = int(kv_indptr_cpu[batch_idx].item()) + pos // page_size
        page = int(kv_indices_cpu[page_iter].item())
        entry = pos % page_size
        torch.testing.assert_close(paged_kv_cache[page, 0, entry], append_key[i])
        torch.testing.assert_close(paged_kv_cache[page, 1, entry], append_value[i])


@pytest.mark.parametrize("contiguous", [True, False])
def test_append_paged_kv_cache(contiguous):
    nnz_kv = 100
    num_kv_heads = 32
    head_dim = 128

    if contiguous:
        k_append = torch.randn(nnz_kv, num_kv_heads, head_dim).half().to(0)
        v_append = torch.randn(nnz_kv, num_kv_heads, head_dim).half().to(0)
    else:
        kv_append = torch.randn(nnz_kv, 2, num_kv_heads, head_dim).half().to(0)
        k_append = kv_append[:, 0]
        v_append = kv_append[:, 1]
    # 45 + 8 + 25 + 22 = nnz_kv
    kv_append_length = torch.tensor([45, 8, 25, 22], dtype=torch.int32, device="cuda:0")
    kv_append_indptr = torch.cat(
        [torch.zeros(1).int().to(0), torch.cumsum(kv_append_length, dim=0)]
    ).int()

    max_num_pages = 1000
    page_size = 16
    paged_kv_cache = (
        torch.randn(max_num_pages, 2, page_size, num_kv_heads, head_dim).half().to(0)
    )
    num_pages_per_req = torch.tensor([3, 1, 2, 2], dtype=torch.int32, device="cuda:0")
    kv_page_indptr = torch.cat(
        [torch.zeros(1).int().to(0), torch.cumsum(num_pages_per_req, dim=0)]
    ).int()
    # use first 8 pages in the paged-kv
    kv_page_indices = torch.arange(8, dtype=torch.int32, device="cuda:0")
    # 45 = (3 - 1) * 16 + 13
    # 8 = (1 - 1) * 16 + 8
    # 25 = (2 - 1) * 16 + 9
    # 22 = (2 - 1) * 16 + 6
    kv_last_page_len = torch.tensor([13, 8, 9, 6], dtype=torch.int32, device="cuda:0")
    batch_indices, positions = flashinfer.get_batch_indices_positions(
        kv_append_indptr,
        flashinfer.get_seq_lens(kv_page_indptr, kv_last_page_len, page_size),
        nnz_kv,
    )

    flashinfer.append_paged_kv_cache(
        k_append,
        v_append,
        batch_indices,
        positions,
        paged_kv_cache,
        kv_page_indices,
        kv_page_indptr,
        kv_last_page_len,
    )
