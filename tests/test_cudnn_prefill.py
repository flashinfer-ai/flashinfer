import math

import numpy as np
import pytest
import torch

import flashinfer


@pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
@pytest.mark.parametrize("s_qo", [8, 16, 37, 17, 32, 64, 128, 1024])
@pytest.mark.parametrize("s_kv", [8, 16, 32, 54, 97, 128, 2048])
@pytest.mark.parametrize("page_size", [1, 8, 16, 32, 128])
@pytest.mark.parametrize("num_kv_heads", [1, 4, 8])
@pytest.mark.parametrize("num_qo_heads", [1, 4, 8, 32])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("causal", [True, False])
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
    q = torch.randn(batch_size, num_qo_heads, s_qo, head_dim).to(device).half()
    # Create view with desired strides [1, h*d, d, h*s*d]
    q = q.as_strided(
        (batch_size, num_qo_heads, s_qo, head_dim),
        (num_qo_heads * s_qo * head_dim, head_dim, num_qo_heads * head_dim, 1),
    )

    # Initialize KV Cache
    num_pages_per_seq = (s_kv + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size

    # kv_cache_shape = (total_num_pages, num_kv_heads, page_size, head_dim)
    # k_cache = torch.randn(size=kv_cache_shape).half().to(device)
    # v_cache = torch.randn(size=kv_cache_shape).half().to(device)
    # print(f"k_cache.shape: {k_cache.shape}, stride: {k_cache.stride()}")
    # print(f"v_cache.shape: {v_cache.shape}, stride: {v_cache.stride()}")

    kv_cache_shape = (total_num_pages, 2, num_kv_heads, page_size, head_dim)
    kv_cache = torch.ones(size=kv_cache_shape).half().to(device)
    k_cache = kv_cache[:, 0, :, :, :]
    v_cache = kv_cache[:, 1, :, :, :]

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
        1, s_qo + 1, (batch_size, 1, 1, 1), dtype=torch.int32
    )
    actual_seq_lens_kv = torch.randint(
        1, s_kv + 1, (batch_size, 1, 1, 1), dtype=torch.int32
    )

    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)

    output = flashinfer.prefill.cudnn_batch_prefill_with_kv_cache(
        q.contiguous(),
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
    )

    import csv

    output_np = output.cpu().numpy()
    with open("cudnn_prefill_output.csv", "w", newline="") as f:
        csv.writer(f).writerows([[float(x)] for x in output_np.flatten()])

    # Check if any value in output is nan
    if torch.isnan(output).any():
        print("WARNING: NaN values detected in output tensor")
        nan_count = torch.isnan(output).sum().item()
        print(f"Number of NaN values: {nan_count}")
        print(f"Total elements in output: {output.numel()}")
        print(f"Percentage of NaN values: {nan_count/output.numel()*100:.2f}%")
    else:
        print("No NaN values detected in output tensor")

    torch.cuda.synchronize()

    # wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace_buffer, "HND")
    # kv_indptr = (
    #     torch.cat(
    #         [
    #             torch.tensor([0], device=device),
    #             torch.cumsum(actual_seq_lens_kv_device.view(-1) // page_size, dim=0),
    #         ]
    #     )
    #     .int()
    #     .to(device)
    # )
    # kv_indices = torch.arange(total_num_pages, device=device).int().to(device)
    # kv_last_page_len = (
    #     torch.full((batch_size,), page_size, device=device).int().to(device)
    # )

    # wrapper.plan(
    #     kv_indptr,
    #     kv_indices,
    #     kv_last_page_len,
    #     num_qo_heads,
    #     num_kv_heads,
    #     head_dim,
    #     page_size,
    #     pos_encoding_mode="NONE",
    #     data_type=torch.float16,
    #     q_data_type=torch.float16,
    # )

    # output_ref = wrapper.run(q.contiguous(), kv_cache)
    # torch.testing.assert_close(output, output_ref, rtol=1e-2, atol=1e-2)
