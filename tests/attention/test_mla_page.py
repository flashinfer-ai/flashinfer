import math
from typing import List

import pytest
import torch

import flashinfer

CKV_DIM = 512
KPE_DIM = 64


def calculate_last_page_len(kv_len: List[int], page_size: int):
    return [len % page_size if len % page_size != 0 else page_size for len in kv_len]


kv_len_configs = [
    [45],
    [4096],
    [45, 8, 25],
    [45, 8, 25, 22],
    [45, 8, 25, 22, 400],
    [45, 8, 25, 22, 100],
]


@pytest.mark.parametrize("kv_len", kv_len_configs)
@pytest.mark.parametrize("page_size", [1, 16, 64])
def test_append_mla_paged_kv_cache(kv_len, page_size):
    nnz_kv = sum(kv_len)
    ckv_append = torch.randn(nnz_kv, CKV_DIM, dtype=torch.float16, device="cuda:0")
    kpe_append = torch.randn(nnz_kv, KPE_DIM, dtype=torch.float16, device="cuda:0")
    num_pages_per_req = torch.tensor(
        [math.ceil(len / page_size) for len in kv_len],
        dtype=torch.int32,
        device="cuda:0",
    )
    kv_append_length = torch.tensor(kv_len, dtype=torch.int32, device="cuda:0")
    kv_append_indptr = torch.cat(
        [torch.zeros(1).int().to(0), torch.cumsum(kv_append_length, dim=0)]
    ).int()

    max_num_pages = sum(num_pages_per_req)
    ckv_cache = torch.zeros(
        max_num_pages, page_size, CKV_DIM, dtype=torch.float16, device="cuda:0"
    )
    kpe_cache = torch.zeros(
        max_num_pages, page_size, KPE_DIM, dtype=torch.float16, device="cuda:0"
    )
    kv_page_indptr = torch.cat(
        [torch.zeros(1).int().to(0), torch.cumsum(num_pages_per_req, dim=0)]
    ).int()
    kv_page_indices = torch.arange(
        sum(num_pages_per_req), dtype=torch.int32, device="cuda:0"
    )
    kv_last_page_len = torch.tensor(
        calculate_last_page_len(kv_len, page_size), dtype=torch.int32, device="cuda:0"
    )
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

    ckv_cache = ckv_cache.view(-1, CKV_DIM)
    kpe_cache = kpe_cache.view(-1, KPE_DIM)

    acc_kv_len = 0
    acc_padding_kv_len = 0
    for i in range(len(kv_len)):
        assert torch.all(
            torch.isclose(
                kpe_append[acc_kv_len : acc_kv_len + kv_len[i]],
                kpe_cache[acc_padding_kv_len : acc_padding_kv_len + kv_len[i]],
            )
        )
        assert torch.all(
            torch.isclose(
                ckv_append[acc_kv_len : acc_kv_len + kv_len[i]],
                ckv_cache[acc_padding_kv_len : acc_padding_kv_len + kv_len[i]],
            )
        )
        assert bool(
            torch.all(
                ckv_cache[
                    acc_padding_kv_len + kv_len[i] : acc_padding_kv_len
                    + num_pages_per_req[i] * page_size
                ]
                == 0
            )
        )
        assert bool(
            torch.all(
                kpe_cache[
                    acc_padding_kv_len + kv_len[i] : acc_padding_kv_len
                    + num_pages_per_req[i] * page_size
                ]
                == 0
            )
        )
        acc_kv_len += kv_len[i]
        acc_padding_kv_len += num_pages_per_req[i] * page_size


if __name__ == "__main__":
    test_append_mla_paged_kv_cache([45], 64)
    test_append_mla_paged_kv_cache([4096], 64)
    test_append_mla_paged_kv_cache([45, 8, 25], 64)
    test_append_mla_paged_kv_cache([45, 8, 25, 22], 64)
    test_append_mla_paged_kv_cache([45, 8, 25, 22, 400], 128)
    test_append_mla_paged_kv_cache([45, 8, 25, 22, 100], 16)
