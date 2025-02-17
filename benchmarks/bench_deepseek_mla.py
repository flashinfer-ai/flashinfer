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

import flashinfer


def bench_deepseek_mla_decode(batch_size, seq_len, num_heads):
    head_dim_ckv = 512
    head_dim_kpe = 64
    page_size = 1
    q_nope = torch.randn(
        batch_size * 1, num_heads, head_dim_ckv, dtype=torch.half, device="cuda"
    )
    q_pe = torch.zeros(
        batch_size * 1, num_heads, head_dim_kpe, dtype=torch.half, device="cuda"
    )
    ckv = torch.randn(
        batch_size * seq_len, 1, head_dim_ckv, dtype=torch.half, device="cuda"
    )
    kpe = torch.zeros(
        batch_size * seq_len, 1, head_dim_kpe, dtype=torch.half, device="cuda"
    )
    sm_scale = 1.0 / ((head_dim_ckv + head_dim_kpe) ** 0.5)
    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8).to(0)
    wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(
        workspace_buffer, backend="fa2"
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
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            o = wrapper.run(q_nope, q_pe, ckv, kpe, return_lse=False)

    torch.cuda.current_stream().wait_stream(s)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        o = wrapper.run(q_nope, q_pe, ckv, kpe, return_lse=False)

    ms = triton.testing.do_bench(
        # lambda: wrapper.run(q_nope, q_pe, ckv, kpe),
        lambda: g.replay(),
        warmup=100,
        rep=1000,
    )

    io = sum([_.numel() * _.element_size() for _ in [q_nope, q_pe, ckv, kpe, o]])

    print(f"Config: batch_size={batch_size}, seq_len={seq_len}, num_heads={num_heads}")
    print(f"Memory bandwidth: {io * 1e-6 / ms:.2f} GB/s")


if __name__ == "__main__":
    # bench_deepseek_mla_decode(768, 1024, 16)
    # bench_deepseek_mla_decode(768, 1024, 32)
    # bench_deepseek_mla_decode(768, 1024, 64)
    # bench_deepseek_mla_decode(768, 2048, 16)
    # bench_deepseek_mla_decode(768, 2048, 32)
    # bench_deepseek_mla_decode(768, 2048, 64)
    bench_deepseek_mla_decode(1, 1024, 16)
    bench_deepseek_mla_decode(1, 1024, 32)
    bench_deepseek_mla_decode(1, 1024, 64)
    bench_deepseek_mla_decode(1, 2048, 16)
    bench_deepseek_mla_decode(1, 2048, 32)
    bench_deepseek_mla_decode(1, 2048, 64)
    bench_deepseek_mla_decode(132, 1024, 16)
    bench_deepseek_mla_decode(1, 32768, 64)
    bench_deepseek_mla_decode(32, 32768, 16)
    bench_deepseek_mla_decode(64, 32768, 16)
