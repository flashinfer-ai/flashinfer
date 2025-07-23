import math

import numpy as np
import pytest
import torch

import flashinfer


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("s_qo", [8, 17, 1024])
@pytest.mark.parametrize("s_kv", [8, 32])
@pytest.mark.parametrize("page_size", [8, 32, 64])
@pytest.mark.parametrize("num_kv_heads", [1, 4])
@pytest.mark.parametrize("num_qo_heads", [4])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("return_lse", [True])
@pytest.mark.parametrize("is_cuda_graph_compatible", [False, True])
def test_cudnn_prefill(
    batch_size,
    s_qo,
    s_kv,
    page_size,
    num_kv_heads,
    num_qo_heads,
    head_dim,
    causal,
    return_lse,
    is_cuda_graph_compatible,
):
    if s_qo > s_kv:
        pytest.skip("s_qo > s_kv, skipping test as causal")

    # test set up basics
    seed = 0
    torch.manual_seed(seed)
    device = "cuda:0"

    actual_seq_lens_q = torch.randint(
        1, s_qo + 1, (batch_size, 1, 1, 1), dtype=torch.int32
    )
    actual_seq_lens_kv = torch.randint(
        1, s_kv + 1, (batch_size, 1, 1, 1), dtype=torch.int32
    )

    cumsum_s_qo = torch.sum(actual_seq_lens_q)
    q = torch.randn(
        cumsum_s_qo, num_qo_heads, head_dim, device=device, dtype=torch.bfloat16
    )

    # Initialize KV Cache
    num_pages_per_seq = (s_kv + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size

    kv_cache_shape = (total_num_pages, 2, num_kv_heads, page_size, head_dim)
    kv_cache = torch.randn(size=kv_cache_shape, dtype=torch.bfloat16).to(device)
    kv_cache = kv_cache.as_strided(
        kv_cache.shape,
        (
            2 * page_size * num_kv_heads * head_dim,
            page_size * num_kv_heads * head_dim,
            head_dim,
            num_kv_heads * head_dim,
            1,
        ),
    )

    actual_seq_lens_q_device = actual_seq_lens_q.to(device)
    actual_seq_lens_kv_device = actual_seq_lens_kv.to(device)
    qo_indptr = (
        torch.cat(
            [
                torch.tensor([0], device=device),
                torch.cumsum(actual_seq_lens_q_device.view(-1), dim=0),
            ]
        )
        .int()
        .to(device)
    )

    kv_indptr = (
        torch.cat(
            [
                torch.tensor([0], device=device),
                torch.cumsum(
                    (actual_seq_lens_kv_device.flatten() + page_size - 1) // page_size,
                    dim=0,
                ),
            ]
        )
        .int()
        .to(device)
    )

    # kv_indices
    kv_indices = torch.zeros(kv_indptr[-1], device=device, dtype=torch.int32)
    for i in range(len(kv_indptr) - 1):
        start_idx = kv_indptr[i]
        end_idx = kv_indptr[i + 1]
        kv_indices[start_idx:end_idx] = torch.arange(
            i * num_pages_per_seq,
            i * num_pages_per_seq + (end_idx - start_idx),
            device=device,
        )

    # kv_last_page_len
    kv_last_page_len = (
        torch.where(
            actual_seq_lens_kv_device.flatten() % page_size == 0,
            torch.full((batch_size,), page_size, device=device),
            actual_seq_lens_kv_device.flatten() % page_size,
        )
        .int()
        .to(device)
    )

    # Workspace buffer
    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)

    if is_cuda_graph_compatible:
        wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            workspace_buffer,
            "HND",
            backend="cudnn",
            use_cuda_graph=is_cuda_graph_compatible,
            qo_indptr_buf=qo_indptr,
            paged_kv_indptr_buf=kv_indptr,
            paged_kv_indices_buf=kv_indices,
            paged_kv_last_page_len_buf=kv_last_page_len,
        )
    else:
        wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            workspace_buffer,
            "HND",
            backend="cudnn",
            use_cuda_graph=is_cuda_graph_compatible,
        )
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        pos_encoding_mode="NONE",
        causal=causal,
        q_data_type=torch.bfloat16,
    )

    output, lse = wrapper.run(q, kv_cache, return_lse=return_lse)

    wrapper_ref = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer,
        "HND",
        use_cuda_graph=is_cuda_graph_compatible,
        qo_indptr_buf=qo_indptr,
        paged_kv_indptr_buf=kv_indptr,
        paged_kv_indices_buf=kv_indices,
        paged_kv_last_page_len_buf=kv_last_page_len,
    )
    wrapper_ref.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        pos_encoding_mode="NONE",
        causal=causal,
        q_data_type=torch.bfloat16,
    )

    output_ref, lse_ref = wrapper.run(q, kv_cache, return_lse=True)

    torch.testing.assert_close(output, output_ref)
    torch.testing.assert_close(lse, lse_ref)


if __name__ == "__main__":
    test_cudnn_prefill(4, 17, 32, 32, 4, 4, 128, True, True, True)
