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
    print(
        f"Running test with batch_size: {batch_size}, s_qo: {s_qo}, s_kv: {s_kv}, page_size: {page_size}, num_kv_heads: {num_kv_heads}, num_qo_heads: {num_qo_heads}, head_dim: {head_dim}, causal: {causal}, return_lse: {return_lse}"
    )
    # test set up basics
    seed = 0
    torch.manual_seed(seed)
    device = "cuda:0"

    # Initialize Q tensor
    shape = (batch_size, num_qo_heads, s_qo, head_dim)
    strides = (num_qo_heads * s_qo * head_dim, head_dim, num_qo_heads * head_dim, 1)

    q = torch.randn(shape, device=device).half()

    # Create view with desired strides [1, h*d, d, h*s*d]
    q = q.as_strided(shape, strides)

    # Initialize KV Cache
    num_pages_per_seq = (s_kv + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size

    kv_cache_shape = (total_num_pages, 2, num_kv_heads, page_size, head_dim)
    kv_cache = torch.randn(size=kv_cache_shape).half().to(device)
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

    import csv

    # print(f"output: {output[2, 0:4, 4:8, 0:3]}")
    # print(f"output: {output[2, 2:4, 0:8, 0:3]}")
    # print(f"output.shape: {output.shape}, output.stride: {output.stride()}, output.data_ptr: {hex(output.data_ptr())}")

    output = output.as_strided(
        (batch_size, num_qo_heads, s_qo, head_dim),
        (num_qo_heads * s_qo * head_dim, s_qo * head_dim, head_dim, 1),
    )
    # print(f"output: {output[2, 0:4, 4:8, 0:3]}")
    # print(f"output: {output[2, 2:4, 0:8, 0:3]}")
    # print(f"output.shape: {output.shape}, output.stride: {output.stride()}, output.data_ptr: {hex(output.data_ptr())}")

    output_np = output.cpu().numpy()
    with open("cudnn_prefill_output.csv", "w", newline="") as f:
        csv.writer(f).writerows([[float(x)] for x in output_np.flatten()])

    # workspace_buffer_ref = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)

    # wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace_buffer_ref, "NHD")

    # actual_seq_lens_kv_device = actual_seq_lens_kv.to(device)

    # q_flat = q.reshape(-1, q.shape[1], q.shape[3])
    # print(f"q_flat.shape: {q_flat.shape}, q_flat.stride: {q_flat.stride()}")

    # print(f"q data: {hex(q.data_ptr())}, q_flat.data: {hex(q_flat.data_ptr())}")

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
    #     actual_seq_lens_q,
    #     kv_indptr,
    #     kv_indices,
    #     kv_last_page_len,
    #     num_qo_heads,
    #     num_kv_heads,
    #     head_dim,
    #     page_size,
    #     pos_encoding_mode="NONE",
    #     causal=True,
    #     q_data_type=torch.float16,
    # )

    # output_ref = wrapper.run(q_flat.contiguous(), kv_cache)

    # output_ref_np = output_ref.cpu().numpy()
    # with open("cudnn_prefill_output_ref.csv", "w", newline="") as f:
    #     csv.writer(f).writerows([[float(x)] for x in output_ref_np.flatten()])

    # print(f"output: {output.shape} @ {output.stride()}")

    # output_flat = output.reshape(-1, output.shape[1], output.shape[3])
    # print(f"output_flat.shape: {output_flat.shape}, output_flat.stride: {output_flat.stride()}")

    # print(f"output data: {hex(output.data_ptr())}, output_flat.data: {hex(output_flat.data_ptr())}")

    # print(f"output_ref: {output_ref.shape} @ {output_ref.stride()}")
    # torch.testing.assert_close(output_flat, output_ref, rtol=1e-2, atol=1e-2)
