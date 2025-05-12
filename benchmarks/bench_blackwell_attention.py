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

import torch
from triton.testing import do_bench

import flashinfer


def bench_fmha_blackwell(
    batch_size,
    qkv_len,
    num_heads,
    head_dim,
    causal,
    dtype,
):
    q = torch.randn(
        batch_size * qkv_len, num_heads, head_dim, dtype=dtype, device="cuda"
    )
    k = torch.randn(
        batch_size * qkv_len, num_heads, head_dim, dtype=dtype, device="cuda"
    )
    v = torch.randn(
        batch_size * qkv_len, num_heads, head_dim, dtype=dtype, device="cuda"
    )

    qo_segment_offsets = (
        torch.arange(0, batch_size + 1, device="cuda", dtype=torch.int32) * qkv_len
    )
    kv_segment_offsets = (
        torch.arange(0, batch_size + 1, device="cuda", dtype=torch.int32) * qkv_len
    )

    o, lse = flashinfer.prefill.fmha_varlen(
        q, k, v, qo_segment_offsets, kv_segment_offsets, causal=causal
    )

    ms = do_bench(
        lambda: flashinfer.prefill.fmha_varlen(
            q,
            k,
            v,
            qo_segment_offsets,
            kv_segment_offsets,
            causal=causal,
        )
    )

    def flops(ms):
        if causal:
            return batch_size * qkv_len * qkv_len * num_heads * head_dim * 2 / ms / 1e9
        else:
            return batch_size * qkv_len * qkv_len * num_heads * head_dim * 4 / ms / 1e9

    print(
        f"bench_fmha_blackwell (batch_size={batch_size}, qkv_len={qkv_len}, num_heads={num_heads}, head_dim={head_dim}, causal={causal}), flops: {flops(ms):.3f} TFLOPs/s"
    )


if __name__ == "__main__":
    bench_fmha_blackwell(128, 512, 32, 128, False, torch.bfloat16)
    bench_fmha_blackwell(64, 1024, 32, 128, False, torch.bfloat16)
    bench_fmha_blackwell(32, 2048, 32, 128, False, torch.bfloat16)
    bench_fmha_blackwell(16, 4096, 32, 128, False, torch.bfloat16)
    bench_fmha_blackwell(8, 8192, 32, 128, False, torch.bfloat16)
    bench_fmha_blackwell(4, 16384, 32, 128, False, torch.bfloat16)
    bench_fmha_blackwell(2, 32768, 32, 128, False, torch.bfloat16)
    bench_fmha_blackwell(1, 65536, 32, 128, False, torch.bfloat16)

    bench_fmha_blackwell(128, 512, 32, 128, True, torch.bfloat16)
    bench_fmha_blackwell(64, 1024, 32, 128, True, torch.bfloat16)
    bench_fmha_blackwell(32, 2048, 32, 128, True, torch.bfloat16)
    bench_fmha_blackwell(16, 4096, 32, 128, True, torch.bfloat16)
    bench_fmha_blackwell(8, 8192, 32, 128, True, torch.bfloat16)
    bench_fmha_blackwell(4, 16384, 32, 128, True, torch.bfloat16)
    bench_fmha_blackwell(2, 32768, 32, 128, True, torch.bfloat16)
    bench_fmha_blackwell(1, 65536, 32, 128, True, torch.bfloat16)
