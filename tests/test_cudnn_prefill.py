import math

import numpy as np
import pytest
import torch

import flashinfer


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("s_qo", [8, 37, 1024])
@pytest.mark.parametrize("s_kv", [8, 54, 2048])
@pytest.mark.parametrize("page_size", [1, 8, 128])
@pytest.mark.parametrize("num_kv_heads", [1, 4])
@pytest.mark.parametrize("num_qo_heads", [4])
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

    # Initialize Q tensor
    shape = (batch_size, s_qo, num_qo_heads, head_dim)

    q_unstrided = torch.randn(
        (batch_size * s_qo * num_qo_heads * head_dim), device=device
    ).half()
    q_unstrided[2048:] = 0

    # cudnn expects shape B, H, S, D
    # In order to go from one head to other D elements to be crossed
    # D, , 1
    # In order to go from one s to other s, h * d elements to be crossed
    # D, H * D, 1
    # In order to go from one batch to other batch, s * h * d elements to be crossed
    # S*H*D, D, H*D, 1
    q = q_unstrided.as_strided(
        (batch_size, num_qo_heads, s_qo, head_dim),
        (s_qo * num_qo_heads * head_dim, head_dim, num_qo_heads * head_dim, 1),
    )

    # assert q.is_contiguous()
    # q = q_unstrided.as_strided((batch_size, num_qo_heads, s_qo, head_dim),
    #                         (s_qo * num_qo_heads * head_dim, s_qo * head_dim, head_dim, 1))

    # Initialize KV Cache
    num_pages_per_seq = (s_kv + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size

    # kv_cache_shape = (total_num_pages, 2, num_kv_heads, page_size, head_dim)
    kv_cache = torch.ones(size=kv_cache_shape).half().to(device)
    # k_cache = kv_cache[:, 0, :, :, :]
    # v_cache = kv_cache[:, 1, :, :, :]
    k_cache = torch.randn(
        total_num_pages, num_kv_heads, page_size, head_dim, device=device
    ).half()
    k_cache = k_cache.as_strided(
        (total_num_pages, num_kv_heads, page_size, head_dim),
        (page_size * num_kv_heads * head_dim, head_dim, num_kv_heads * head_dim, 1),
    )

    print(f"k_cache.shape: {k_cache.shape}, k_cache.stride: {k_cache.stride()}")

    v_cache = torch.randn(
        total_num_pages, num_kv_heads, page_size, head_dim, device=device
    ).half()
    v_cache = v_cache.as_strided(
        (total_num_pages, num_kv_heads, page_size, head_dim),
        (page_size * num_kv_heads * head_dim, head_dim, num_kv_heads * head_dim, 1),
    )

    print(f"v_cache.shape: {v_cache.shape}, v_cache.stride: {v_cache.stride()}")

    # k_cache = k_cache_unstrided.as_strided((total_num_pages, page_size, num_kv_heads, head_dim), (page_size * num_kv_heads * head_dim, head_dim, num_kv_heads * head_dim, 1))
    # v_cache = v_cache_unstrided.as_strided((total_num_pages, page_size, num_kv_heads, head_dim), (page_size * num_kv_heads * head_dim, head_dim, num_kv_heads * head_dim, 1))

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

    # Actual sequence lengths (should be randomized across batches)
    actual_seq_lens_q = torch.randint(
        1, 1 + 1, (batch_size, 1, 1, 1), dtype=torch.int32
    )
    actual_seq_lens_kv = torch.randint(
        1, 1 + 1, (batch_size, 1, 1, 1), dtype=torch.int32
    )

    print(f"actual_seq_lens_q: {actual_seq_lens_q}")
    print(f"actual_seq_lens_kv: {actual_seq_lens_kv}")

    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)

    output, lse = flashinfer.prefill.cudnn_batch_prefill_with_kv_cache(
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

    # cumsum_s_qo = torch.sum(actual_seq_lens_q)
    # print(f"cumsum_s_qo: {cumsum_s_qo}")

    # output = output.as_strided(
    #     (batch_size, num_qo_heads, s_qo, head_dim),
    #     (num_qo_heads * s_qo * head_dim, head_dim, num_qo_heads * head_dim, 1),
    # )

    output0 = output.reshape(batch_size * s_qo, num_qo_heads, head_dim)
    print(f"output0.shape: {output0.shape}, output0.stride: {output0.stride()}")
    print(output0)
    output1 = output.as_strided(
        (batch_size, num_qo_heads, s_qo, head_dim),
        (s_qo * num_qo_heads * head_dim, head_dim, s_qo * head_dim, 1),
    )
    print(f"output1.shape: {output1.shape}, output1.stride: {output1.stride()}")
    print(output1)

    # output = output.as_strided(
    #     (batch_size, num_qo_heads, s_qo, head_dim),
    #     (num_qo_heads * s_qo * head_dim, s_qo * head_dim, head_dim, 1),
    # )

    torch.save(output, "output.pt")

    import csv

    output_np = output.cpu().numpy()
    with open("cudnn_prefill_output.csv", "w", newline="") as f:
        csv.writer(f).writerows([[float(x)] for x in output_np.flatten()])

    workspace_buffer_ref = torch.empty(
        128 * 1024 * 1024, dtype=torch.int8, device=device
    )

    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer_ref, "NHD"
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

    q_flat = q.reshape(-1, num_qo_heads, head_dim)
    # print(f"q_flat.shape: {q_flat.shape}, q_flat.stride: {q_flat.stride()}")
    # print(q)

    # print(f"q data: {hex(q.data_ptr())}, q_flat.data: {hex(q_flat.data_ptr())}")

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

    print(f"qo_indptr: {qo_indptr}")
    print(f"kv_indptr: {kv_indptr}")

    kv_indices = torch.arange(total_num_pages, device=device).int().to(device)
    kv_last_page_len = (
        torch.full((batch_size,), page_size, device=device).int().to(device)
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

    kv_cache = kv_cache.reshape(total_num_pages, 2, page_size, num_kv_heads, head_dim)

    # assert q_flat.is_contiguous()

    output_ref = wrapper.run(q_flat, kv_cache)

    print(
        f"output_ref.shape: {output_ref.shape}, output_ref.stride: {output_ref.stride()}"
    )
    print(output_ref)

    # torch.testing.assert_close(output.flatten(), output_ref.flatten(), rtol=1e-2, atol=1e-2)
