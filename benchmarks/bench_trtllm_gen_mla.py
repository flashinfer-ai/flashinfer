import numpy as np
import torch
import triton

import flashinfer
from flashinfer.testing.utils import bench_gpu_time, bench_gpu_time_with_cudagraph

num_q_heads = 128
num_kv_heads = 1
qk_nope_head_dim = 128
qk_rope_head_dim = 64
kv_lora_rank = 512


def bench_trtllm_mla(batch_size, q_len_per_request, seq_len, page_size, dtype):
    torch.manual_seed(42)
    device = "cuda:0"

    # Initialize tensors
    query = torch.randn(
        batch_size,
        q_len_per_request,
        num_q_heads,
        kv_lora_rank + qk_rope_head_dim,
        device=device,
    ).to(dtype)

    num_tokens = seq_len * batch_size
    num_blocks = (num_tokens + page_size - 1) // page_size

    # Sequence lengths and block tables
    seq_lens = [torch.randint(1, seq_len, (1,)).item() for _ in range(batch_size)]
    seq_lens[-1] = seq_len
    max_seq_len = max(seq_lens)
    seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int, device=device)

    blocks_per_seq = (seq_lens_tensor + page_size - 1) // page_size
    max_num_blocks_per_seq = blocks_per_seq.max().item()

    # Generate random but unique block IDs for all sequences
    total_blocks_needed = sum(blocks_per_seq)
    all_block_ids = torch.randperm(
        total_blocks_needed, device=device
    )  # Random permutation

    # Generate unique block IDs for all sequences
    block_id = 0
    block_tables = torch.zeros(
        (batch_size, max_num_blocks_per_seq), dtype=torch.int, device=device
    )

    # Populate block tables and track block assignments
    block_id = 0
    for i in range(batch_size):
        num_blocks_needed = blocks_per_seq[i]
        block_tables[i, :num_blocks_needed] = all_block_ids[
            block_id : block_id + num_blocks_needed
        ]
        block_id += num_blocks_needed

    # Create interleaved KV cache
    # Allocate more than needed blocks, block_id is just enough, to mimick real-world cases
    kv_cache = torch.randn(
        size=(num_blocks, page_size, kv_lora_rank + qk_rope_head_dim), device=device
    ).to(dtype)
    # (num_blocks, 1, page_size, kv_lora_rank + qk_rope_head_dim)

    # Allocate workspace buffer
    # todo(Yingyi): calculate the actual size of workspace buffer
    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)

    # Run decode-MLA
    # warmup
    flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
        query=query,
        kv_cache=kv_cache.unsqueeze(1),
        workspace_buffer=workspace_buffer,
        qk_nope_head_dim=qk_nope_head_dim,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        block_tables=block_tables,
        seq_lens=seq_lens_tensor,
        max_seq_len=max_seq_len,
        bmm1_scale=1.0 / ((128 + 64) ** 0.5),
        bmm2_scale=1.0,
    )
    # benchmark
    measurements = bench_gpu_time_with_cudagraph(
        lambda: flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
            query=query,
            kv_cache=kv_cache.unsqueeze(1),
            workspace_buffer=workspace_buffer,
            qk_nope_head_dim=qk_nope_head_dim,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            block_tables=block_tables,
            seq_lens=seq_lens_tensor,
            max_seq_len=max_seq_len,
            bmm1_scale=1.0 / ((128 + 64) ** 0.5),
            bmm2_scale=1.0,
        ),
        dry_run_time_ms=100,
        repeat_time_ms=1000,
    )
    io = (
        query.numel() * query.element_size()
        + kv_cache.numel() * kv_cache.element_size()
    )
    ms = np.median(measurements)
    flops = (
        2
        * batch_size
        * num_q_heads
        * (2 * kv_lora_rank + qk_rope_head_dim)
        * seq_len
        * q_len_per_request
    )
    print(
        f"batch_size={batch_size}, q_len_per_request={q_len_per_request}, seq_len={seq_len}, num_q_heads={num_q_heads}, num_kv_heads={num_kv_heads}, qk_nope_head_dim={qk_nope_head_dim}, qk_rope_head_dim={qk_rope_head_dim}, kv_lora_rank={kv_lora_rank}, page_size={page_size}"
    )
    print(f"execution time: {ms} ms")
    print(f"memory bandwidth: {io / ms / 1024 / 1024 :.2f} GB/s")
    print(f"FLOPs: {flops * 1e-9 / ms:.2f} TFLOPs/s")


if __name__ == "__main__":
    for dtype in [torch.bfloat16, torch.float8_e4m3fn]:
        for page_size in [32, 64]:
            for batch_size in [1, 2, 4, 16, 32, 64, 128, 256, 512, 768, 1024]:
                for seq_len in [1024, 4096, 8192]:
                    for q_len_per_request in [1, 2, 4, 8, 16]:
                        bench_trtllm_mla(
                            batch_size, q_len_per_request, seq_len, page_size, dtype
                        )
