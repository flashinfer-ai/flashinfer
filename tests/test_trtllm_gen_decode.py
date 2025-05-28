import math

import numpy as np
import pytest
import torch

import flashinfer


def to_float8(x, dtype=torch.float8_e4m3fn):
    finfo = torch.finfo(dtype)
    min_val, max_val = x.aminmax()
    amax = torch.maximum(min_val.abs(), max_val.abs()).clamp(min=1e-12)
    scale = finfo.max / amax * 0.1
    x_scl_sat = (x * scale).clamp(min=finfo.min, max=finfo.max)
    return x_scl_sat.to(dtype), scale.float().reciprocal()


@pytest.mark.parametrize("kv_layout", ["HND"])  # trtllm-gen only support HND
@pytest.mark.parametrize("batch_size", [4, 8])
@pytest.mark.parametrize("page_size", [16, 32, 64])
@pytest.mark.parametrize("num_kv_heads", [4])
@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"])
def test_trtllm_batch_decode(
    kv_layout, batch_size, page_size, num_kv_heads, kv_cache_dtype
):
    # Set up test parameters
    seed = 0
    torch.manual_seed(seed)
    device = "cuda:0"
    head_dim = 128
    HEAD_GRP_SIZE = 8
    num_qo_heads = num_kv_heads * HEAD_GRP_SIZE
    batch_size = batch_size
    MAX_SEQ_LEN = 128

    # Initialize tensors
    num_tokens = MAX_SEQ_LEN * batch_size
    num_blocks = num_tokens // page_size
    dtype = torch.float16

    scale = float(1.0 / (head_dim**0.5))
    q = torch.randn(2, num_qo_heads, head_dim).to(0).half()

    # Sequence lengths and block tables
    seq_lens = [MAX_SEQ_LEN for _ in range(batch_size)]
    seq_lens[-1] = MAX_SEQ_LEN
    max_seq_len = max(seq_lens)
    seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int, device=device)

    max_num_blocks_per_seq = (max_seq_len + page_size - 1) // page_size
    block_tables = torch.tensor(
        [
            [k + i * max_num_blocks_per_seq for k in range(max_num_blocks_per_seq)]
            for i in range(batch_size)
        ],
        dtype=torch.int,
        device=device,
    )

    # Create interleaved KV cache
    kv_cache_shape = (num_blocks, 2, num_kv_heads, page_size, head_dim)
    kv_cache = torch.randn(size=kv_cache_shape).half()
    k_scale = v_scale = 1.0

    if kv_cache_dtype.startswith("fp8"):
        kv_cache, _ = to_float8(kv_cache)

    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)

    output = flashinfer.decode.trtllm_batch_decode_with_kv_cache(
        q.contiguous(),
        kv_cache,
        workspace_buffer,
        num_qo_heads,
        num_kv_heads,
        scale,
        block_tables,
        seq_lens_tensor,
        page_size,
        max_seq_len,
        kv_cache_dtype,
        k_scale,
        v_scale,
    )

    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace_buffer, kv_layout)
    kv_indptr = (
        torch.cat(
            [
                torch.tensor([0], device=device),
                torch.cumsum(seq_lens_tensor // page_size, dim=0),
            ]
        )
        .int()
        .to(device)
    )
    kv_indices = torch.arange(num_blocks, device=device).int().to(device)

    print(f"kv_indptr: {kv_indptr}")
    print(f"kv_indices: {kv_indices}")

    kv_last_page_len = (
        torch.full((batch_size,), page_size, device=device).int().to(device)
    )

    wrapper.plan(
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        pos_encoding_mode="NONE",
        data_type=torch.float16 if kv_cache_dtype == "auto" else torch.float8_e4m3fn,
        q_data_type=torch.float16,
    )

    output_ref = wrapper.run(q.contiguous(), kv_cache)
    torch.testing.assert_close(output, output_ref, rtol=1e-2, atol=1e-2)
