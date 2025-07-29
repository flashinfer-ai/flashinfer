import math

import numpy as np
import pytest
import torch

import flashinfer


@pytest.mark.parametrize("batch_size", [4, 8, 17, 64])
@pytest.mark.parametrize("s_kv", [8, 40, 1024])
@pytest.mark.parametrize("page_size", [1, 8])
@pytest.mark.parametrize("num_kv_heads", [4])
@pytest.mark.parametrize("num_qo_heads", [4, 32])
@pytest.mark.parametrize("is_cuda_graph_compatible", [False, True])
def test_cudnn_decode(
    batch_size,
    s_kv,
    page_size,
    num_kv_heads,
    num_qo_heads,
    is_cuda_graph_compatible,
):

    # test set up basics
    seed = 0
    torch.manual_seed(seed)
    device = "cuda:0"

    s_qo = 1
    head_dim = 128

    # Initialize Q tensor
    # Since the number of tokens is 1, batch size is the token count
    q = torch.randn(
        batch_size, num_qo_heads, head_dim, device=device, dtype=torch.bfloat16
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

    # Now initialize the page tables
    block_tables = torch.tensor(
        [
            [k + i * num_pages_per_seq for k in range(num_pages_per_seq)]
            for i in range(batch_size)
        ],
        dtype=torch.int,
        device=device,
    )

    # Actual sequence lengths (should be randomized across batches. )
    actual_seq_lens_kv = torch.randint(
        1, s_kv + 1, (batch_size, 1, 1, 1), dtype=torch.int32
    )

    workspace_buffer_size = math.ceil(
        (
            batch_size * s_qo * num_qo_heads * head_dim * 4
            + batch_size * s_qo * num_qo_heads * 4
        )
        / (1024 * 1024)
    ) * (1024 * 1024)

    workspace_buffer_size = max(workspace_buffer_size, 128 * 1024 * 1024)

    workspace_buffer = torch.empty(
        workspace_buffer_size, dtype=torch.int8, device=device
    )

    actual_seq_lens_kv_device = actual_seq_lens_kv.to(device)

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

    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer, "HND", backend="cudnn"
    )
    wrapper.plan(
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        block_tables=block_tables,
        q_data_type=torch.bfloat16,
    )

    output = wrapper.run(q, kv_cache)

    wrapper_ref = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace_buffer, "HND")
    wrapper_ref.plan(
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        q_data_type=torch.bfloat16,
    )

    output_ref = wrapper_ref.run(q, kv_cache)

    torch.testing.assert_close(output, output_ref, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    test_cudnn_decode(4, 40, 1, 4, 4, True)
