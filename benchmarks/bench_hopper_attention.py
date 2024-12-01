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


def bench_single_prefill(seq_len, num_heads, causal, head_dim):
    num_qo_heads = num_kv_heads = num_heads
    q = torch.randn(seq_len, num_qo_heads, head_dim, dtype=torch.half, device="cuda")
    k = torch.randn(seq_len, num_kv_heads, head_dim, dtype=torch.half, device="cuda")
    v = torch.randn(seq_len, num_kv_heads, head_dim, dtype=torch.half, device="cuda")

    sm80_ms, sm90_ms = (
        triton.testing.do_bench(
            lambda: flashinfer.single_prefill_with_kv_cache_return_lse(
                q, k, v, causal=causal, backend=backend
            ),
            warmup=100,
            rep=1000,
        )
        for backend in ["fa2", "fa3"]
    )

    def flops(ms):
        if causal:
            return seq_len * seq_len * num_qo_heads * head_dim * 2 / ms / 1e9
        else:
            return seq_len * seq_len * num_qo_heads * head_dim * 4 / ms / 1e9

    print(
        f"bench_single_prefill (seq_len={seq_len}, num_heads={num_heads}, causal={causal}, head_dim={head_dim}), fa2-template: {flops(sm80_ms):.3f} TFLOPs/s, fa3-template: {flops(sm90_ms):.3f} TFLOPs/s"
    )


def bench_batch_ragged_prefill(batch_size, num_heads, seq_len, causal, head_dim):
    num_qo_heads = num_kv_heads = num_heads
    q = torch.randn(
        batch_size * seq_len, num_qo_heads, head_dim, dtype=torch.half, device="cuda"
    )
    k = torch.randn(
        batch_size * seq_len, num_kv_heads, head_dim, dtype=torch.half, device="cuda"
    )
    v = torch.randn(
        batch_size * seq_len, num_kv_heads, head_dim, dtype=torch.half, device="cuda"
    )

    sm80_wrapper, sm90_wrapper = (
        flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
            torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device="cuda:0"),
            kv_layout="NHD",
            backend=backend,
        )
        for backend in ["fa2", "fa3"]
    )

    qo_indptr = torch.arange(0, batch_size * seq_len + 1, seq_len).int()
    kv_indptr = torch.arange(0, batch_size * seq_len + 1, seq_len).int()

    for wrapper in [sm80_wrapper, sm90_wrapper]:
        wrapper.plan(
            qo_indptr,
            kv_indptr,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            causal=causal,
        )

    sm80_ms, sm90_ms = (
        triton.testing.do_bench(
            lambda: wrapper.run(q, k, v),
            warmup=100,
            rep=1000,
        )
        for wrapper in [sm80_wrapper, sm90_wrapper]
    )

    def flops(ms):
        if causal:
            return (
                batch_size * seq_len * seq_len * num_qo_heads * head_dim * 2 / ms / 1e9
            )
        else:
            return (
                batch_size * seq_len * seq_len * num_qo_heads * head_dim * 4 / ms / 1e9
            )

    print(
        f"bench_batch_ragged_prefill (batch_size={batch_size}, num_heads={num_heads}, seq_len={seq_len}, causal={causal}, head_dim={head_dim}), fa2-template: {flops(sm80_ms):.3f} TFLOPs/s, fa3-template: {flops(sm90_ms):.3f} TFLOPs/s"
    )


def bench_batch_paged_prefill(
    page_size, batch_size, num_heads, seq_len, causal, head_dim
):
    num_qo_heads = num_kv_heads = num_heads
    q = torch.randn(
        batch_size * seq_len, num_qo_heads, head_dim, dtype=torch.half, device="cuda"
    )
    k = torch.randn(
        batch_size * seq_len // page_size,
        page_size,
        num_kv_heads,
        head_dim,
        dtype=torch.half,
        device="cuda",
    )
    v = torch.randn(
        batch_size * seq_len // page_size,
        page_size,
        num_kv_heads,
        head_dim,
        dtype=torch.half,
        device="cuda",
    )

    sm80_wrapper, sm90_wrapper = (
        flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device="cuda:0"),
            kv_layout="NHD",
            backend=backend,
        )
        for backend in ["fa2", "fa3"]
    )

    qo_indptr = torch.arange(0, batch_size * seq_len + 1, seq_len).int()
    kv_indptr = torch.arange(
        0, batch_size * (seq_len // page_size) + 1, (seq_len // page_size)
    ).int()
    kv_indices = torch.arange(0, batch_size * (seq_len // page_size)).int()
    last_page_len = torch.ones(batch_size, dtype=torch.int32) * page_size

    for wrapper in [sm80_wrapper, sm90_wrapper]:
        wrapper.plan(
            qo_indptr,
            kv_indptr,
            kv_indices,
            last_page_len,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,  # page_size
            causal=causal,
        )

    sm80_ms, sm90_ms = (
        triton.testing.do_bench(
            lambda: wrapper.run(q, (k, v)),
            warmup=100,
            rep=1000,
        )
        for wrapper in [sm80_wrapper, sm90_wrapper]
    )

    def flops(ms):
        if causal:
            return (
                batch_size * seq_len * seq_len * num_qo_heads * head_dim * 2 / ms / 1e9
            )
        else:
            return (
                batch_size * seq_len * seq_len * num_qo_heads * head_dim * 4 / ms / 1e9
            )

    print(
        f"bench_batch_paged_prefill (page_size={page_size} batch_size={batch_size}, num_heads={num_heads}, seq_len={seq_len}, causal={causal}, head_dim={head_dim}), fa2-template: {flops(sm80_ms):.3f} TFLOPs/s, fa3-template: {flops(sm90_ms):.3f} TFLOPs/s"
    )


if __name__ == "__main__":
    bench_batch_paged_prefill(1, 128, 32, 1024, True, 128)
    bench_batch_paged_prefill(1, 64, 32, 2048, True, 128)
    bench_batch_paged_prefill(1, 32, 32, 4096, True, 128)
    bench_batch_paged_prefill(1, 16, 32, 8192, True, 128)
    bench_batch_paged_prefill(1, 1, 32, 32768, True, 128)
    bench_batch_paged_prefill(16, 128, 32, 1024, True, 128)
    bench_batch_paged_prefill(16, 64, 32, 2048, True, 128)
    bench_batch_paged_prefill(16, 32, 32, 4096, True, 128)
    bench_batch_paged_prefill(16, 16, 32, 8192, True, 128)
    bench_batch_paged_prefill(16, 1, 32, 32768, True, 128)
    bench_batch_ragged_prefill(128, 32, 1024, True, 128)
    bench_batch_ragged_prefill(64, 32, 2048, True, 128)
    bench_batch_ragged_prefill(32, 32, 4096, True, 128)
    bench_batch_ragged_prefill(16, 32, 8192, True, 128)
    bench_batch_ragged_prefill(1, 32, 32768, True, 128)
