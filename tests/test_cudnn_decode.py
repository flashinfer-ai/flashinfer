import math

import numpy as np
import pytest
import torch

import flashinfer


@pytest.mark.parametrize("batch_size", [4, 8])
@pytest.mark.parametrize("s_qo", [1])
@pytest.mark.parametrize("s_kv", [64, 128, 2048])
@pytest.mark.parametrize("page_size", [1, 8, 16])
@pytest.mark.parametrize("num_kv_heads", [4])
@pytest.mark.parametrize("num_qo_heads", [4, 32, 64])
@pytest.mark.parametrize("head_dim", [128])
def test_cudnn_decode(
    batch_size, s_qo, s_kv, page_size, num_kv_heads, num_qo_heads, head_dim
):

    # test set up basics
    seed = 0
    torch.manual_seed(seed)
    device = "cuda:0"

    # Initialize Q tensor
    q = torch.randn(batch_size, num_qo_heads, s_qo, head_dim).to(device).half()

    # Initialize KV Cache
    num_pages_per_seq = (s_kv + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size

    kv_cache_shape = (total_num_pages, num_kv_heads, page_size, head_dim)
    k_cache = torch.randn(size=kv_cache_shape).half().to(device)
    v_cache = torch.randn(size=kv_cache_shape).half().to(device)

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

    # Actual sequence lengths (should be randomized across batches. )
    actual_seq_lens_kv = torch.randint(
        1, s_kv, (batch_size, 1, 1, 1), dtype=torch.int32
    )
    actual_seq_lens_q = torch.randint(
        1, s_qo + 1, (batch_size, 1, 1, 1), dtype=torch.int32
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

    output = flashinfer.decode.cudnn_batch_decode_with_kv_cache(
        q.contiguous(),
        k_cache,
        v_cache,
        scale,
        workspace_buffer,
        actual_seq_lens_q,
        actual_seq_lens_kv,
        block_tables,
        num_pages_per_seq,
    )

    # import csv
    # output_flat = output.cpu().numpy().flatten()
    # with open('cudnn_decode_output.csv', 'w', newline='') as f:
    #     csv.writer(f).writerows([[x] for x in output_flat])
    torch.cuda.synchronize()
