import numpy
import pytest
import torch

import flashinfer


@pytest.mark.parametrize("batch_size", [12, 17])
@pytest.mark.parametrize("kv_len", [54, 97])
@pytest.mark.parametrize("qo_len", [37, 17])
@pytest.mark.parametrize("page_size", [1, 8, 16])
@pytest.mark.parametrize("num_kv_heads", [4])
@pytest.mark.parametrize("num_qo_heads", [4, 32])
@pytest.mark.parametrize("head_dim", [128])
def test_batch_prefill_with_paged_kv_cache(
    batch_size,
    kv_len,
    qo_len,
    page_size,
    num_kv_heads,
    num_qo_heads,
    head_dim,
):
    q = torch.randn(batch_size * qo_len, num_qo_heads, head_dim).to(0).half()
    q_indptr = torch.arange(0, batch_size + 1).to(0).int() * qo_len
    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size
    kv_data = (
        torch.randn(total_num_pages, 2, num_kv_heads, page_size, head_dim).to(0).half()
    )
    kv_indptr = torch.arange(0, batch_size + 1).to(0).int() * num_pages_per_seq
    kv_indices = torch.arange(0, total_num_pages).to(0).int()
    kv_last_page_len = torch.full(
        (batch_size,), (kv_len - 1) % page_size + 1, dtype=torch.int32
    ).to(0)

    o = flashinfer.ops.batch_prefill_with_paged_kv_cache(
        q,
        q_indptr,
        kv_data,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        page_size,
    )

    for i in range(batch_size):
        qi = q[q_indptr[i] : q_indptr[i + 1]]
        ki = torch.cat(
            [
                kv_data[kv_indptr[i] : kv_indptr[i + 1] - 1, 0]
                .permute(0, 2, 1, 3)
                .reshape(-1, num_kv_heads, head_dim),
                kv_data[kv_indptr[i + 1] - 1, 0, :, : kv_last_page_len[i]]
                .permute(1, 0, 2)
                .reshape(-1, num_kv_heads, head_dim),
            ],
            dim=0,
        )
        vi = torch.cat(
            [
                kv_data[kv_indptr[i] : kv_indptr[i + 1] - 1, 1]
                .permute(0, 2, 1, 3)
                .reshape(-1, num_kv_heads, head_dim),
                kv_data[kv_indptr[i + 1] - 1, 1, :, : kv_last_page_len[i]]
                .permute(1, 0, 2)
                .reshape(-1, num_kv_heads, head_dim),
            ],
            dim=0,
        )
        o_ref_i = flashinfer.ops.single_prefill_with_kv_cache(qi, ki, vi, True)
        o_i_np = o[q_indptr[i] : q_indptr[i + 1]].cpu().numpy()
        o_ref_i_np = o_ref_i.cpu().numpy()
        numpy.testing.assert_allclose(o_i_np, o_ref_i_np, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    test_batch_prefill_with_paged_kv_cache(12, 54, 37, 8, 8, 8, 128)
    test_batch_prefill_with_paged_kv_cache(12, 54, 37, 1, 8, 8, 128)
