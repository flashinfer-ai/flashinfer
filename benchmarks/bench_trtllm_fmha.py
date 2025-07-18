import numpy as np
import torch
import triton

import flashinfer

page_size = 16
num_kv_heads = 4
num_qo_heads = 32
head_dim = 128

workspace_buffer = torch.empty(1024 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")


def bench_trtllm_fmha(batch_size, seq_len, kv_cache_dtype):
    torch.manual_seed(42)
    seq_lens = torch.full((batch_size,), seq_len, device="cuda:0", dtype=torch.int32)
    seq_lens_blocks = torch.ceil(seq_lens / page_size).int()
    kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int, device="cuda:0")
    kv_indptr[1:] = torch.cumsum(seq_lens_blocks, dim=0)
    last_page_len = (seq_lens - (seq_lens_blocks - 1) * page_size).int()
    last_page_len[last_page_len == 0] = page_size
    num_blocks = kv_indptr[-1].item()
    kv_indices = torch.arange(num_blocks, dtype=torch.int32, device="cuda:0")

    q = torch.rand(batch_size, num_qo_heads, head_dim, device="cuda:0").to(
        torch.bfloat16
    )
    kv_data = torch.randn(
        num_blocks, 2, num_kv_heads, page_size, head_dim, device="cuda:0"
    ).to(torch.float8_e4m3fn if kv_cache_dtype == "fp8" else torch.float16)

    wrapper = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer, "HND", backend="trtllm-gen"
    )
    wrapper.plan(
        kv_indptr,
        kv_indices,
        last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        pos_encoding_mode="NONE",
        q_data_type=q.dtype,
        kv_data_type=kv_data.dtype,
    )
    # add one warmup here
    wrapper.run(q, kv_data)
    torch.cuda.synchronize()

    ms = triton.testing.do_bench_cudagraph(
        lambda: wrapper.run(q, kv_data),
        rep=4,
    )
    io = q.numel() * q.element_size() + kv_data.numel() * kv_data.element_size()
    print(
        f"batch_size={batch_size}, seq_len={seq_len}, num_qo_heads={num_qo_heads}, num_kv_heads={num_kv_heads}, head_dim={head_dim}, page_size={page_size}"
    )
    print(f"execution time: {ms}ms")
    print(f"memory bandwidth: {io / ms / 1024 / 1024 :.2f} GB/s")


def to_float8(x, dtype=torch.float8_e4m3fn):
    finfo = torch.finfo(dtype)
    min_val, max_val = x.aminmax()
    amax = torch.maximum(min_val.abs(), max_val.abs()).clamp(min=1e-12)
    scale = finfo.max / amax * 0.1
    x_scl_sat = (x * scale).clamp(min=finfo.min, max=finfo.max)
    return x_scl_sat.to(dtype), scale.float().reciprocal()


def bench_trtllm_fmha_wrapper(
    kv_layout,
    batch_size,
    max_seq_len,
    page_size,
    num_kv_heads,
    q_dtype,
    head_grp_size,
    kv_cache_dtype,
    window_left,
):
    torch.manual_seed(42)
    device = "cuda:0"
    head_dim = 128
    num_qo_heads = num_kv_heads * head_grp_size
    batch_size = batch_size

    # Initialize tensors
    num_tokens = max_seq_len * batch_size
    num_blocks = (num_tokens + page_size - 1) // page_size

    dtype_map = {
        "half": torch.float16,
        "bf16": torch.bfloat16,
        "fp8": torch.float8_e4m3fn,
    }

    q = torch.randn(batch_size, num_qo_heads, head_dim, device=device).to(
        dtype_map[q_dtype]
    )

    # Sequence lengths and block tables
    seq_lens = torch.full((batch_size,), max_seq_len)
    seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int, device=device)

    blocks_per_seq = [(seq_len + page_size - 1) // page_size for seq_len in seq_lens]

    # Generate random but unique block IDs for all sequences
    total_blocks_needed = sum(blocks_per_seq)
    all_block_ids = torch.randperm(
        total_blocks_needed, device=device
    )  # Random permutation

    kv_cache_shape = (num_blocks, 2, num_kv_heads, page_size, head_dim)
    kv_cache = torch.randn(size=kv_cache_shape).to(q.dtype)

    if kv_cache_dtype.startswith("fp8") and q_dtype != "fp8":
        kv_cache, _ = to_float8(kv_cache)

    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)

    blocks_per_seq = (seq_lens_tensor + page_size - 1) // page_size

    # Compute kv_indptr as cumulative sum of blocks per sequence
    kv_indptr = (
        torch.cat(
            [torch.tensor([0], device=device), torch.cumsum(blocks_per_seq, dim=0)]
        )
        .int()
        .to(device)
    )

    kv_indices = all_block_ids.int()

    # Calculate last page lengths
    kv_last_page_len = seq_lens_tensor % page_size
    kv_last_page_len[kv_last_page_len == 0] = page_size

    # trtllm-gen
    wrapper = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer, kv_layout, backend="trtllm-gen"
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
        data_type=kv_cache.dtype,
        q_data_type=q.dtype,
        window_left=window_left,
    )

    # add one warmup here
    wrapper.run(q, kv_cache)
    torch.cuda.synchronize()

    ms = triton.testing.do_bench_cudagraph(
        lambda: wrapper.run(q, kv_cache),
        rep=4,
    )
    io = q.numel() * q.element_size() + kv_cache.numel() * kv_cache.element_size()
    print(
        f"batch_size={batch_size}, seq_len={max_seq_len}, num_qo_heads={num_qo_heads}, num_kv_heads={num_kv_heads}, head_dim={head_dim}, page_size={page_size}"
    )
    print(f"execution time: {ms}ms")
    print(f"memory bandwidth: {io / ms / 1024 / 1024 :.2f} GB/s")


if __name__ == "__main__":
    for batch_size in [4, 128, 256]:
        for seq_len in [1024, 4096, 8192, 16384]:
            bench_trtllm_fmha_wrapper(
                kv_layout="HND",
                batch_size=batch_size,
                max_seq_len=seq_len,
                page_size=16,
                num_kv_heads=4,
                q_dtype="bf16",
                head_grp_size=1,
                kv_cache_dtype="auto",
                window_left=-1,
            )
