import math

import numpy as np
import pytest
import torch

import flashinfer


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("s_qo", [8, 37, 1024])
@pytest.mark.parametrize("s_kv", [8, 54, 2048])
@pytest.mark.parametrize("page_size", [1, 2, 8, 128])
@pytest.mark.parametrize("num_kv_heads", [1, 3])
@pytest.mark.parametrize("num_qo_heads", [4, 3])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("causal", [True])
@pytest.mark.parametrize("return_lse", [True])
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
):
    # test set up basics
    seed = 0
    torch.manual_seed(seed)
    device = "cuda:0"

    actual_seq_lens_q = torch.randint(
        1, 1 + 1, (batch_size, 1, 1, 1), dtype=torch.int32
    )
    actual_seq_lens_kv = torch.randint(
        1, 1 + 1, (batch_size, 1, 1, 1), dtype=torch.int32
    )

    cumsum_s_qo = torch.sum(actual_seq_lens_q)
    print(f"cumsum_s_qo: {cumsum_s_qo}")
    print(f"num_qo_heads: {num_qo_heads}")
    print(f"num_kv_heads: {num_kv_heads}")

    print(f"actual_seq_lens_kv: {actual_seq_lens_kv.flatten()}")

    q = torch.randn(cumsum_s_qo, num_qo_heads, head_dim, device=device).half()

    # q[:,1,:] = 1.0

    print(f"q.shape: {q.shape}, q.stride: {q.stride()}")
    # print(q[:,:,0:12])

    # Initialize KV Cache
    num_pages_per_seq = (s_kv + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size

    print(f"total_num_pages: {total_num_pages}")

    # k_cache = torch.randn(total_num_pages, page_size, num_kv_heads, head_dim, device=device).half()
    # v_cache_experimental = torch.randn(total_num_pages, page_size, num_kv_heads, head_dim, device=device).half()

    # v_cache = v_cache_experimental.view(total_num_pages, page_size, num_kv_heads, head_dim).permute(0, 2, 1, 3)
    # v_cache = v_cache_experimental

    # PARTIALLY INTERLEAVED KV CACHE
    kv_cache_shape = (total_num_pages, 2, page_size, num_kv_heads, head_dim)
    kv_cache = torch.randn(size=kv_cache_shape).half().to(device)
    k_cache = kv_cache[:, 0, :, :, :]

    v_cache_view = kv_cache[:, 1, :, :, :]

    # for i in range(total_num_pages):
    #     k_cache[i,:,:,:] = i
    #     v_cache_view[i,:,:,:] = i + 1000

    v_cache = v_cache_view.permute(0, 2, 1, 3)

    print(f"kv_cache.shape: {kv_cache.shape}, kv_cache.stride: {kv_cache.stride()}")
    print(f"v_cache.shape: {v_cache.shape}, v_cache.stride: {v_cache.stride()}")
    print(
        f"v_cache_view.shape: {v_cache_view.shape}, v_cache_view.stride: {v_cache_view.stride()}"
    )

    # END PARTIALLY INTERLEAVED KV CACHE

    # Now initialize the page tables
    block_tables = torch.tensor(
        [
            [k + i * num_pages_per_seq for k in range(num_pages_per_seq)]
            for i in range(batch_size)
        ],
        dtype=torch.int,
        device=device,
    )

    print(block_tables)
    print(
        f"block_tables.shape: {block_tables.shape}, block_tables.stride: {block_tables.stride()}"
    )

    # Initialize scale
    scale = float(1.0 / (head_dim**0.5))

    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)

    output, lse = flashinfer.prefill.cudnn_batch_prefill_with_kv_cache(
        batch_size,
        s_qo,
        q,
        k_cache,
        v_cache,
        scale,
        workspace_buffer,
        actual_seq_lens_q,
        actual_seq_lens_kv,
        block_tables,
        num_pages_per_seq,
        causal,
        return_lse,
        use_cuda_graph=False,
    )
    torch.cuda.synchronize()

    print("output")
    print(f"output.shape: {output.shape}, output.stride: {output.stride()}")
    print(output[0:3, 0:3, 0:10])

    torch.save(output, "output.pt")

    torch.save(v_cache, "v_cache.pt")
    torch.save(v_cache_view, "v_cache_view.pt")
    torch.save(k_cache, "k_cache.pt")
    torch.save(kv_cache, "kv_cache.pt")

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
                torch.cumsum(actual_seq_lens_kv_device.view(-1), dim=0),
            ]
        )
        .int()
        .to(device)
    )

    kv_indices = torch.arange(total_num_pages, device=device).int().to(device)
    kv_last_page_len = (
        torch.full((batch_size,), page_size, device=device).int().to(device)
    )
    workspace_buffer_ref = torch.empty(
        128 * 1024 * 1024, dtype=torch.int8, device=device
    )

    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer_ref, "NHD"
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
        causal=True,
        q_data_type=torch.float16,
    )

    output_ref = wrapper.run(q, kv_cache)

    torch.cuda.synchronize()

    torch.save(output_ref, "output_ref.pt")

    # print(output_ref)
