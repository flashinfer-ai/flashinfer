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

import math

import torch
import triton

from flashinfer.cute_dsl.attention import BatchMLAPagedAttentionWrapperCuteDSL


def bench_mla_decode_cutedsl(batch_size, seq_len, num_heads):
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
        batch_size * pages_num,
        page_size,
        head_dim_ckv,
        dtype=torch.bfloat16,
        device="cuda",
    )
    kpe = torch.randn(
        batch_size * pages_num,
        page_size,
        head_dim_kpe,
        dtype=torch.bfloat16,
        device="cuda",
    )
    sm_scale = 1.0 / ((head_dim_ckv + head_dim_kpe) ** 0.5)

    workspace_buffer = torch.empty(1, dtype=torch.int8, device="cuda")
    wrapper = BatchMLAPagedAttentionWrapperCuteDSL(workspace_buffer)

    qo_indptr = torch.arange(0, batch_size + 1, dtype=torch.int32, device="cuda")
    kv_indptr = (
        torch.arange(0, batch_size + 1, dtype=torch.int32, device="cuda") * seq_len
    )
    kv_indices = torch.arange(batch_size * seq_len, dtype=torch.int32, device="cuda")
    kv_lens = torch.full((batch_size,), seq_len, dtype=torch.int32, device="cuda")

    wrapper.plan(
        qo_indptr=qo_indptr,
        kv_indptr=kv_indptr,
        kv_indices=kv_indices,
        kv_len_arr=kv_lens,
        num_heads=num_heads,
        head_dim_ckv=head_dim_ckv,
        head_dim_kpe=head_dim_kpe,
        page_size=page_size,
        causal=True,
        sm_scale=sm_scale,
        q_data_type=q_nope.dtype,
        kv_data_type=ckv.dtype,
    )
    o, lse = wrapper.run(
        q_nope=q_nope, q_pe=q_pe, ckv_cache=ckv, kpe_cache=kpe, return_lse=True
    )

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


if __name__ == "__main__":
    for seq_len in [1024, 2048, 8192]:
        for batch_size in [64, 128, 768]:
            for num_heads in [128]:
                bench_mla_decode_cutedsl(batch_size, seq_len, num_heads)
