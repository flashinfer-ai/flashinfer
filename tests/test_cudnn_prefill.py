import pytest
import torch

import flashinfer


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("s_qo", [8, 17, 700])
@pytest.mark.parametrize("s_kv", [8, 32, 1066])
@pytest.mark.parametrize("page_size", [8, 16, 64])
@pytest.mark.parametrize("num_kv_heads", [1, 4])
@pytest.mark.parametrize("num_qo_heads", [4])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("return_lse", [True, False])
@pytest.mark.parametrize("is_cuda_graph_compatible", [True])
def test_cudnn_prefill(
    batch_size,
    s_qo,
    s_kv,
    page_size,
    num_kv_heads,
    num_qo_heads,
    causal,
    return_lse,
    is_cuda_graph_compatible,
):
    head_dim = 128
    if s_qo > s_kv:
        pytest.skip("s_qo > s_kv, skipping test")

    # test set up basics
    seed = 1
    torch.manual_seed(seed)
    device = "cuda:0"

    actual_seq_lens_q = torch.randint(
        1, s_qo + 1, (batch_size, 1, 1, 1), dtype=torch.int32, device=device
    )
    actual_seq_lens_kv = torch.randint(
        s_qo, s_kv + 1, (batch_size, 1, 1, 1), dtype=torch.int32, device=device
    )

    cumsum_s_qo = torch.sum(actual_seq_lens_q)
    q = torch.randn(
        cumsum_s_qo, num_qo_heads, head_dim, device=device, dtype=torch.bfloat16
    )

    q_indptr = torch.cat(
        [
            torch.tensor([0], device=device),
            torch.cumsum(actual_seq_lens_q.view(-1), dim=0) * head_dim * num_qo_heads,
        ]
    ).int()

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
    k_cache_view = kv_cache[:, 0, :, :, :]
    v_cache_view = kv_cache[:, 1, :, :, :]

    v_cache = v_cache_view.as_strided(
        v_cache_view.shape,
        (2 * page_size * num_kv_heads * head_dim, head_dim, num_kv_heads * head_dim, 1),
    )
    k_cache = k_cache_view.as_strided(
        k_cache_view.shape,
        (2 * page_size * num_kv_heads * head_dim, head_dim, num_kv_heads * head_dim, 1),
    )

    kv_indptr = torch.cat(
        [
            torch.tensor([0], device=device),
            torch.cumsum(
                (actual_seq_lens_kv.flatten() + page_size - 1) // page_size,
                dim=0,
            ),
        ]
    ).int()

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
    kv_last_page_len = torch.where(
        actual_seq_lens_kv.flatten() % page_size == 0,
        torch.full((batch_size,), page_size, device=device),
        actual_seq_lens_kv.flatten() % page_size,
    ).int()

    # Now initialize the page tables
    block_tables = torch.tensor(
        [
            [k + i * num_pages_per_seq for k in range(num_pages_per_seq)]
            for i in range(batch_size)
        ],
        dtype=torch.int,
        device=device,
    )

    # Initialize scale
    scale = float(1.0 / (head_dim**0.5))

    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)

    wrapper_cudnn = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer, "NHD", backend="cudnn"
    )
    wrapper_cudnn.plan(
        q_indptr,
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
        seq_lens=actual_seq_lens_kv,
        seq_lens_q=actual_seq_lens_q,
        sm_scale=scale,
        max_token_per_sequence=s_qo,
        max_sequence_kv=s_kv,
        block_tables=block_tables,
    )

    output = wrapper_cudnn.run(q, (k_cache, v_cache))

    qo_indptr = torch.cat(
        [
            torch.tensor([0], device=device),
            torch.cumsum(actual_seq_lens_q.view(-1), dim=0),
        ]
    ).int()

    # Workspace buffer
    workspace_buffer_ref = torch.empty(
        128 * 1024 * 1024, dtype=torch.int8, device=device
    )

    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer_ref, "HND"
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

    output_ref = wrapper.run(q, kv_cache)

    torch.testing.assert_close(output, output_ref, atol=2e-3, rtol=1e-2)
