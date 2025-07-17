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


def bench_trtllm_gen_mla(batch_size, seq_len, dtype):
    np.random.seed(42)
    seq_lens = torch.full((batch_size,), seq_len, device="cuda:0", dtype=torch.int32)
    seq_lens_blocks = torch.ceil(seq_lens / page_size).int()
    kv_indptr = torch.cat([torch.tensor([0]), torch.cumsum(seq_lens_blocks, 0)], dim=0)
    kv_indptr = kv_indptr.int()
    last_page_len = seq_lens - (seq_lens_blocks - 1) * page_size
    last_page_len = last_page_len.int()
    num_blocks = kv_indptr[-1].item()
    max_num_blocks_per_seq = (seq_len + page_size - 1) // page_size
    base_blocks = torch.arange(batch_size * max_num_blocks_per_seq, dtype=torch.int32)
    block_tables = base_blocks.reshape(batch_size, max_num_blocks_per_seq).to(0)

    q = torch.rand(batch_size, num_qo_heads, head_dim, device="cuda:0").to(dtype)
    kv_cache = torch.randn(
        size=(num_blocks, page_size, 512 + 64), device="cuda:0"
    ).to(dtype)

    # add one warmup here
    flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
        query=q,
        kv_cache=kv_cache,
        workspace_buffer=workspace_buffer,
        qk_nope_head_dim=128,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        block_tables=block_tables,
        seq_lens=seq_lens,
        block_size=page_size,
        max_seq_len=seq_len,
        bmm1_scale=scale,
        bmm2_scale=1.0,
    )
    torch.cuda.synchronize()

    ms = triton.testing.do_bench_cudagraph(
        lambda: flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
            query=q,
            kv_cache=kv_cache,
            workspace_buffer=workspace_buffer,
            qk_nope_head_dim=128,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            block_tables=block_tables,
            seq_lens=seq_lens,
            block_size=page_size,
            max_seq_len=seq_len,
            bmm1_scale=scale,
            bmm2_scale=1.0,
        ),
        rep=4,
    )
    io = q.numel() * q.element_size() + kv_cache.numel() * kv_cache.element_size()
    print(
        f"batch_size={batch_size}, seq_len={seq_len}, num_qo_heads={num_qo_heads}, num_kv_heads={num_kv_heads}, head_dim={head_dim}, page_size={page_size}"
    )
    print(f"execution time: {ms}ms")
    print(f"memory bandwidth: {io / ms / 1024 / 1024 :.2f} GB/s")


if __name__ == "__main__":
    for batch_size in [4, 8, 16, 32, 64, 128, 256, 512]:
        for seq_len in [1024, 4096, 8192, 16384]:
            for dtype in [torch.float8_e4m3fn, torch.float16]:
                bench_trtllm_gen_mla(batch_size, seq_len, dtype)
