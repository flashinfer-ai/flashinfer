import numpy as np
import torch
import triton

import flashinfer

page_size = 16
num_kv_heads = 4
num_qo_heads = 32
head_dim = 128

scale = float(1.0 / (head_dim**0.5))
k_scale = v_scale = 1.0

workspace_buffer = torch.empty(1024 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")


def bench_trtllm_fmha(batch_size, seq_len, kv_cache_dtype):
    np.random.seed(42)
    seq_lens = torch.full((batch_size,), seq_len)
    seq_lens_blocks = torch.ceil(seq_lens / page_size).int()
    kv_indptr = torch.cat([torch.tensor([0]), torch.cumsum(seq_lens_blocks, 0)], dim=0)
    kv_indptr = kv_indptr.int()
    last_page_len = seq_lens - (seq_lens_blocks - 1) * page_size
    last_page_len = last_page_len.int()
    num_blocks = kv_indptr[-1].item()
    max_num_blocks_per_seq = (seq_len + page_size - 1) // page_size
    base_blocks = torch.arange(batch_size * max_num_blocks_per_seq, dtype=torch.int32)
    block_tables = base_blocks.reshape(batch_size, max_num_blocks_per_seq).to(0)
    seq_lens_gpu = seq_lens.int().to(0).contiguous()

    q = torch.rand(batch_size, num_qo_heads, head_dim).half().to(0)
    kv_data = (
        torch.randn(num_blocks, 2, num_kv_heads, page_size, head_dim)
        .to(0)
        .to(torch.float8_e4m3fn if kv_cache_dtype == "fp8" else torch.float16)
    )

    ms = triton.testing.do_bench_cudagraph(
        lambda: flashinfer.decode.trtllm_batch_decode_with_kv_cache(
            q,
            kv_data,
            workspace_buffer,
            num_qo_heads,
            num_kv_heads,
            scale,
            block_tables,
            seq_lens_gpu,
            page_size,
            seq_len,
            kv_cache_dtype,
            k_scale,
            v_scale,
        ),
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
