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


if __name__ == "__main__":
    for batch_size in [4, 8, 16, 32, 64, 128, 256, 512]:
        for seq_len in [1024, 4096, 8192, 16384]:
            for kv_cache_dtype in ["fp8", "auto"]:
                bench_trtllm_fmha(batch_size, seq_len, kv_cache_dtype)
