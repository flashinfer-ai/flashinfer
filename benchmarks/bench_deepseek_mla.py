"""
Copyright (c) 2024 by FlashInfer team.

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

import torch
import triton
import math
import flashinfer
from flashinfer.cute_dsl.mla import BatchMLAPagedAttentionWrapperCuteDSL



def bench_deepseek_mla_decode(batch_size, seq_len, num_heads, backend):
    head_dim_ckv = 512
    head_dim_kpe = 64
    page_size = 128
    q_nope = torch.randn(
        batch_size * 1, num_heads, head_dim_ckv, dtype=torch.bfloat16, device="cuda"
    )
    q_pe = torch.zeros(
        batch_size * 1, num_heads, head_dim_kpe, dtype=torch.bfloat16, device="cuda"
    )
    ckv = torch.randn(
        batch_size * seq_len, 1, head_dim_ckv, dtype=torch.bfloat16, device="cuda"
    )
    kpe = torch.zeros(
        batch_size * seq_len, 1, head_dim_kpe, dtype=torch.bfloat16, device="cuda"
    )
    sm_scale = 1.0 / ((head_dim_ckv + head_dim_kpe) ** 0.5)
    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8).to(0)
    wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(
        workspace_buffer, backend=backend
    )
    q_indptr = torch.arange(0, batch_size + 1).to(0).int()
    kv_indptr = torch.arange(0, batch_size + 1).to(0).int() * seq_len
    kv_indices = torch.arange(0, batch_size * seq_len).to(0).int()
    kv_lens = torch.full((batch_size,), seq_len, dtype=torch.int32).to(0)
    wrapper.plan(
        q_indptr,
        kv_indptr,
        kv_indices,
        kv_lens,
        num_heads,
        head_dim_ckv,
        head_dim_kpe,
        page_size,
        False,  # causal
        sm_scale,
        q_nope.dtype,
        ckv.dtype,
    )
    # o_input = torch.empty_like(q_nope)
    # lse_input = torch.empty((batch_size, num_heads), dtype=torch.float32, device="cuda")
    o = wrapper.run(q_nope, q_pe, ckv, kpe, return_lse=False)

    ms = triton.testing.do_bench(
        lambda: wrapper.run(q_nope, q_pe, ckv, kpe, return_lse=False),
        warmup=100,
        rep=1000,
    )

    io = sum([_.numel() * _.element_size() for _ in [q_nope, q_pe, ckv, kpe, o]])
    flops = 2 * batch_size * num_heads * (2 * head_dim_ckv + head_dim_kpe) * seq_len

    print(f"Config: batch_size={batch_size}, seq_len={seq_len}, num_heads={num_heads}")
    print(f"Memory bandwidth: {io * 1e-6 / ms:.2f} GB/s")
    print(f"FLOPs: {flops * 1e-9 / ms:.2f} TFLOPs")

def bench_deepseek_mla_decode_dsl(batch_size, seq_len, num_heads):
    head_dim_ckv = 512
    head_dim_kpe = 64
    page_size = 128
    q_nope = torch.randn(
        batch_size * 1, num_heads, head_dim_ckv, dtype=torch.bfloat16, device="cuda"
    )
    q_pe = torch.randn(
        batch_size * 1, num_heads, head_dim_kpe, dtype=torch.bfloat16, device="cuda"
    )
    pages_num = math.ceil(seq_len / page_size)
    ckv = torch.randn(
        batch_size * pages_num, page_size, head_dim_ckv, dtype=torch.bfloat16, device="cuda"
    )
    kpe = torch.randn(
        batch_size * pages_num, page_size, head_dim_kpe, dtype=torch.bfloat16, device="cuda"
    )
    sm_scale = 1.0 / ((head_dim_ckv + head_dim_kpe) ** 0.5)
    
    # Calculate workspace size
    # workspace_size = flashinfer.mla_cutedsl.BlackwellMultiLatentAttentionForward.get_workspace_size(
    #     num_heads, head_dim_ckv, batch_size, -1, flashinfer.mla_cutedsl.cutlass.Float32
    # )
    workspace_buffer = torch.empty(1, dtype=torch.int8, device="cuda")
    
    # Create wrapper and initialize
    wrapper = BatchMLAPagedAttentionWrapperCuteDSL(workspace_buffer)
    
    # Create indptr tensors for the wrapper
    qo_indptr = torch.arange(0, batch_size + 1, dtype=torch.int32, device="cuda")
    kv_indptr = torch.arange(0, batch_size + 1, dtype=torch.int32, device="cuda") * seq_len
    kv_indices = torch.arange(batch_size * seq_len, dtype=torch.int32, device="cuda")
    kv_lens = torch.full((batch_size,), seq_len, dtype=torch.int32, device="cuda")
    
    # Plan the computation
    wrapper.plan(
        qo_indptr=qo_indptr,
        kv_indptr=kv_indptr,
        kv_indices=kv_indices,
        kv_len_arr=kv_lens,
        num_heads=num_heads,
        head_dim_ckv=head_dim_ckv,
        head_dim_kpe=head_dim_kpe,
        page_size=page_size,
        causal=True,  # causal
        sm_scale=sm_scale,
        q_data_type=q_nope.dtype,
        kv_data_type=ckv.dtype,
    )
    # Run the computation once to warm up
    o, lse = wrapper.run(
        q_nope=q_nope,
        q_pe=q_pe,
        ckv_cache=ckv,
        kpe_cache=kpe,
        return_lse=True
    )

    # Benchmark the computation
    ms = triton.testing.do_bench(
        lambda: wrapper.run(
            q_nope=q_nope,
            q_pe=q_pe,
            ckv_cache=ckv,
            kpe_cache=kpe,
            return_lse=True,
        ),
        warmup=100,
        rep=1000,
    )

    io = sum([_.numel() * _.element_size() for _ in [q_nope, q_pe, ckv, kpe, o]])
    flops = 2 * batch_size * num_heads * (2 * head_dim_ckv + head_dim_kpe) * seq_len

    print(f"Config: batch_size={batch_size}, seq_len={seq_len}, num_heads={num_heads}")
    print(f"Memory bandwidth: {io * 1e-6 / ms:.2f} GB/s")
    print(f"FLOPs: {flops * 1e-9 / ms:.2f} TFLOPs")


def bench_deepseek_mla_decode_trtllm(batch_size, seq_len, num_heads):
    # Deepseek attention config (decode-MLA)
    qk_nope_head_dim = 128
    qk_rope_head_dim = 64
    kv_lora_rank = 512
    page_size = 64
    q_len_per_request = 1
    
    # Initialize query tensor [batch_size, q_len_per_request, num_heads, kv_lora_rank + qk_rope_head_dim]
    query = torch.randn(
        batch_size,
        q_len_per_request,
        num_heads,
        kv_lora_rank + qk_rope_head_dim,
        dtype=torch.bfloat16,
        device="cuda"
    )
    
    # Calculate number of blocks needed
    num_tokens = seq_len * batch_size
    num_blocks = (num_tokens + page_size - 1) // page_size
    
    # Create sequence lengths (all sequences have the same length for simplicity)
    seq_lens = torch.full((batch_size,), seq_len, dtype=torch.int32, device="cuda")
    max_seq_len = seq_len
    
    # Calculate blocks per sequence
    blocks_per_seq = (seq_lens + page_size - 1) // page_size
    max_num_blocks_per_seq = blocks_per_seq.max().item()
    
    # Generate block tables
    total_blocks_needed = sum(blocks_per_seq)
    all_block_ids = torch.randperm(total_blocks_needed, device="cuda")
    
    block_tables = torch.zeros(
        (batch_size, max_num_blocks_per_seq), dtype=torch.int32, device="cuda"
    )
    
    block_id = 0
    for i in range(batch_size):
        num_blocks_needed = blocks_per_seq[i]
        block_tables[i, :num_blocks_needed] = all_block_ids[
            block_id : block_id + num_blocks_needed
        ]
        block_id += num_blocks_needed
    
    # Create KV cache [num_blocks, page_size, kv_lora_rank + qk_rope_head_dim]
    kv_cache = torch.randn(
        num_blocks, page_size, kv_lora_rank + qk_rope_head_dim, 
        dtype=torch.bfloat16, device="cuda"
    )
    
    # Allocate workspace buffer
    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device="cuda")
    
    # Calculate scale factors
    sm_scale = 1.0 / ((qk_nope_head_dim + qk_rope_head_dim) ** 0.5)
    
    # Dynamic scale tensors (set to None for static scaling)
    bmm1_scale_log2_tensor = None
    bmm2_scale_tensor = None
    
    # Run the computation once to warm up
    output = flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
        query=query,
        kv_cache=kv_cache.unsqueeze(1),  # Add dimension for [num_blocks, 1, page_size, head_dim]
        workspace_buffer=workspace_buffer,
        qk_nope_head_dim=qk_nope_head_dim,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        block_tables=block_tables,
        seq_lens=seq_lens,
        max_seq_len=max_seq_len,
        bmm1_scale=sm_scale,
        bmm2_scale=1.0,
        bmm1_scale_log2_tensor=bmm1_scale_log2_tensor,
        bmm2_scale_tensor=bmm2_scale_tensor,
    )

    # Benchmark the computation
    ms = triton.testing.do_bench(
        lambda: flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
            query=query,
            kv_cache=kv_cache.unsqueeze(1),
            workspace_buffer=workspace_buffer,
            qk_nope_head_dim=qk_nope_head_dim,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            block_tables=block_tables,
            seq_lens=seq_lens,
            max_seq_len=max_seq_len,
            bmm1_scale=sm_scale,
            bmm2_scale=1.0,
            bmm1_scale_log2_tensor=bmm1_scale_log2_tensor,
            bmm2_scale_tensor=bmm2_scale_tensor,
        ),
        warmup=100,
        rep=1000,
    )

    io = sum([_.numel() * _.element_size() for _ in [query, kv_cache, output]])
    flops = 2 * batch_size * num_heads * (2 * kv_lora_rank + qk_rope_head_dim) * seq_len

    print(f"Config: batch_size={batch_size}, seq_len={seq_len}, num_heads={num_heads}")
    print(f"Memory bandwidth: {io * 1e-6 / ms:.2f} GB/s")
    print(f"FLOPs: {flops * 1e-9 / ms:.2f} TFLOPs")


if __name__ == "__main__":
    # for seq_len in [1024, 2048, 8192]:
    #     for batch_size in [64, 128, 768]:
    #         for num_heads in [128]:
    #             bench_deepseek_mla_decode(batch_size, seq_len, num_heads, "auto")
    
    print("\n=== CuteDSL Benchmark ===")
    for seq_len in [1024, 2048, 8192]:
        for batch_size in [64, 128, 768]:
            for num_heads in [128]:
                bench_deepseek_mla_decode_dsl(batch_size, seq_len, num_heads)
    
    # print("\n=== TensorRT-LLM MLA Benchmark ===")
    # for seq_len in [1024, 2048, 8192]:
    #     for batch_size in [64, 128, 768]:
    #         for num_heads in [128]:
    #             bench_deepseek_mla_decode_trtllm(batch_size, seq_len, num_heads)
