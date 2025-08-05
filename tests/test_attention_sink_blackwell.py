"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import einops
import pytest
import torch
from sink_attention_reference import sink_attention_unified

import flashinfer


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("batch_size", [1, 4, 16])
@pytest.mark.parametrize("page_size", [32])
@pytest.mark.parametrize("seq_len", [32, 128, 1024])
@pytest.mark.parametrize("num_qo_heads", [32])
@pytest.mark.parametrize("num_kv_heads", [8, 32])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_blackwell_trtllm_gen_decode_attention_sink(
    dtype,
    batch_size,
    page_size,
    seq_len,
    num_qo_heads,
    num_kv_heads,
    head_dim,
):
    seed = 0
    torch.manual_seed(seed)
    device = "cuda:0"

    seq_lens = torch.full((batch_size,), seq_len, dtype=torch.int32, device=device)

    blocks_per_seq = (seq_lens + page_size - 1) // page_size
    max_num_blocks_per_seq = torch.max(blocks_per_seq).item()

    # Generate unique block IDs for all sequences
    block_tables = torch.arange(
        (batch_size * max_num_blocks_per_seq), dtype=torch.int32, device=device
    ).reshape(batch_size, max_num_blocks_per_seq)

    # Create separate K and V caches
    num_tokens = seq_len * batch_size
    num_blocks = (num_tokens + page_size - 1) // page_size
    q = torch.randn(
        batch_size,
        num_qo_heads,
        head_dim,
        dtype=dtype,
        device=device,
    )

    k_cache = torch.randn(
        num_blocks, num_kv_heads, page_size, head_dim, dtype=dtype, device=device
    )
    v_cache = torch.randn(
        num_blocks, num_kv_heads, page_size, head_dim, dtype=dtype, device=device
    )

    sink = torch.rand(num_qo_heads, device=device, dtype=torch.float32) * 5

    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)

    output = flashinfer.decode.trtllm_batch_decode_with_kv_cache(
        q.contiguous(),
        (k_cache, v_cache),
        workspace_buffer,
        block_tables,
        seq_lens,
        seq_len,
        1.0,  # bmm1_scale
        1.0,  # bmm2_scale
        -1,  # window_left
        out_dtype=dtype,
        sinks=sink,
    )

    k = einops.rearrange(
        k_cache,
        "(b num_pages_per_b) h p d -> b (num_pages_per_b p) h d",
        num_pages_per_b=max_num_blocks_per_seq,
    )
    v = einops.rearrange(
        v_cache,
        "(b num_pages_per_b) h p d -> b (num_pages_per_b p) h d",
        num_pages_per_b=max_num_blocks_per_seq,
    )

    o_ref = sink_attention_unified(
        q,
        k,
        v,
        sink,
        -1,
        False,
        1.0,
        mode="incremental",
    )

    if dtype == torch.float16:
        atol, rtol = 1e-3, 1e-3
    elif dtype == torch.bfloat16:
        atol, rtol = 1e-2, 1e-2
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    torch.testing.assert_close(o_ref, output, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("batch_size", [1, 4, 16])
@pytest.mark.parametrize("page_size", [32])
@pytest.mark.parametrize("seq_len", [32, 128, 1024])
@pytest.mark.parametrize("num_qo_heads", [32])
@pytest.mark.parametrize("num_kv_heads", [8, 32])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_blackwell_trtllm_gen_context_attention_sink(
    dtype,
    batch_size,
    page_size,
    seq_len,
    num_qo_heads,
    num_kv_heads,
    head_dim,
):
    seed = 0
    torch.manual_seed(seed)
    device = "cuda:0"

    seq_lens = torch.full((batch_size,), seq_len, dtype=torch.int32, device=device)

    blocks_per_seq = (seq_lens + page_size - 1) // page_size
    max_num_blocks_per_seq = torch.max(blocks_per_seq).item()

    # Generate unique block IDs for all sequences
    block_tables = torch.arange(
        (batch_size * max_num_blocks_per_seq), dtype=torch.int32, device=device
    ).reshape(batch_size, max_num_blocks_per_seq)

    # Create separate K and V caches
    num_tokens = seq_len * batch_size
    num_blocks = (num_tokens + page_size - 1) // page_size
    q = torch.randn(
        num_tokens,
        num_qo_heads,
        head_dim,
        dtype=dtype,
        device=device,
    )

    k_cache = torch.randn(
        num_blocks, num_kv_heads, page_size, head_dim, dtype=dtype, device=device
    )
    v_cache = torch.randn(
        num_blocks, num_kv_heads, page_size, head_dim, dtype=dtype, device=device
    )

    sink = torch.rand(num_qo_heads, device=device, dtype=torch.float32) * 5

    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)
    q_indptr = (
        torch.arange(0, batch_size + 1, dtype=torch.int32, device=device) * seq_len
    )
    kv_indptr = (
        torch.arange(0, num_blocks + 1, dtype=torch.int32, device=device) * page_size
    )

    output = flashinfer.prefill.trtllm_batch_context_with_kv_cache(
        q.contiguous(),
        (k_cache, v_cache),
        workspace_buffer,
        block_tables,
        seq_lens,
        seq_len,
        seq_len,
        1.0,  # bmm1_scale
        1.0,  # bmm2_scale
        batch_size,
        q_indptr,
        kv_indptr,
        -1,  # window_left
        out_dtype=dtype,
        sinks=sink,
    )

    k = einops.rearrange(
        k_cache,
        "num_pages h p d -> (num_pages p) h d",
    )
    v = einops.rearrange(
        v_cache,
        "num_pages h p d -> (num_pages p) h d",
    )

    print(q.shape, k.shape, v.shape)

    o_ref = sink_attention_unified(
        q,
        k,
        v,
        sink,
        -1,
        True,
        1.0,
        mode="prefill",
        batch_size=batch_size,
    )

    if dtype == torch.float16:
        atol, rtol = 1e-3, 1e-3
    elif dtype == torch.bfloat16:
        atol, rtol = 1e-2, 1e-2
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    torch.testing.assert_close(o_ref, output, atol=atol, rtol=rtol)
