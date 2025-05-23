import math

import numpy as np
import pytest
import torch

import flashinfer


@pytest.mark.parametrize("batch_size", [4, 8])
@pytest.mark.parametrize("s_qo", [37, 17, 1024])
@pytest.mark.parametrize("s_kv", [54, 97, 2048])
@pytest.mark.parametrize("page_size", [1, 8, 16])
@pytest.mark.parametrize("num_kv_heads", [4])
@pytest.mark.parametrize("num_qo_heads", [4, 32])
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

    # Actual sequence lengths (should be randomized across batches)
    actual_seq_lens_q = torch.randint(1, s_qo, (batch_size, 1, 1, 1), dtype=torch.int32)
    actual_seq_lens_kv = torch.randint(
        1, s_kv, (batch_size, 1, 1, 1), dtype=torch.int32
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

    torch.cuda.synchronize()
