import pytest
import torch

import flashinfer


def test_append_mla_paged_kv_cache():
    nnz_kv = 100
    ckv_dim = 512
    kpe_dim = 64

    ckv_append = torch.randn(nnz_kv, ckv_dim).half().to(0)
    kpe_append = torch.randn(nnz_kv, kpe_dim).half().to(0)
    # 45 + 8 + 25 + 22 = nnz_kv
    kv_append_length = torch.tensor([45, 8, 25, 22], dtype=torch.int32, device="cuda:0")
    kv_append_indptr = torch.cat(
        [torch.zeros(1).int().to(0), torch.cumsum(kv_append_length, dim=0)]
    ).int()

    max_num_pages = 1000
    page_size = 16
    ckv_cache = torch.zeros(max_num_pages, page_size, ckv_dim).half().to(0)
    kpe_cache = torch.zeros(max_num_pages, page_size, kpe_dim).half().to(0)
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
    flashinfer.append_paged_mla_kv_cache(
        ckv_append,
        kpe_append,
        batch_indices,
        positions,
        ckv_cache,
        kpe_cache,
        kv_page_indices,
        kv_page_indptr,
        kv_last_page_len,
    )

    # 45 + 8 + 25 + 22 = nnz_kv
    ckv_cache = ckv_cache.view(-1, ckv_dim)
    kpe_cache = kpe_cache.view(-1, kpe_dim)
    assert torch.all(torch.isclose(ckv_append[:45], ckv_cache[:45]))
    assert torch.all(torch.isclose(kpe_append[:45], kpe_cache[:45]))
    assert bool(torch.all(ckv_cache[45:48] == 0))
    assert bool(torch.all(kpe_cache[45:48] == 0))

    assert torch.all(torch.isclose(kpe_append[45:53], kpe_cache[48:56]))
    assert torch.all(torch.isclose(ckv_append[45:53], ckv_cache[48:56]))
    assert bool(torch.all(ckv_cache[56:64] == 0))
    assert bool(torch.all(kpe_cache[56:64] == 0))

    assert torch.all(torch.isclose(kpe_append[53:78], kpe_cache[64:89]))
    assert torch.all(torch.isclose(ckv_append[53:78], ckv_cache[64:89]))
    assert bool(torch.all(ckv_cache[89:96] == 0))
    assert bool(torch.all(kpe_cache[89:96] == 0))

    assert torch.all(torch.isclose(kpe_append[78:100], kpe_cache[96:118]))
    assert torch.all(torch.isclose(ckv_append[78:100], ckv_cache[96:118]))
    assert bool(torch.all(ckv_cache[118:] == 0))
    assert bool(torch.all(kpe_cache[118:] == 0))


if __name__ == "__main__":
    test_append_mla_paged_kv_cache()
