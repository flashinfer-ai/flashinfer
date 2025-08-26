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

import numpy as np
import torch

import flashinfer
from flashinfer.testing.utils import bench_gpu_time

from flashinfer.cute_dsl.prefill import BatchPrefillCuteDSLWrapper


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
    wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        torch.empty(128 * 1024 * 1024, dtype=dtype, device="cuda"),
        kv_layout="NHD",
        backend="cutlass",
    )
    wrapper.plan(
        qo_segment_offsets,
        kv_segment_offsets,
        num_heads,
        num_heads,
        head_dim,
        head_dim_vo=head_dim,
        causal=causal,
        q_data_type=dtype,
        kv_data_type=dtype,
    )
    o = wrapper.run(q, k, v)
    measurements = bench_gpu_time(
        lambda: wrapper.run(q, k, v),
        dry_run_time_ms=100,
        repeat_time_ms=1000,
    )
    ms = np.median(measurements)

    def flops(ms):
        if causal:
            return batch_size * qkv_len * qkv_len * num_heads * head_dim * 2 / ms / 1e9
        else:
            return batch_size * qkv_len * qkv_len * num_heads * head_dim * 4 / ms / 1e9

    def io(ms):
        mem_size = (
            q.numel() * q.element_size()
            + k.numel() * k.element_size()
            + v.numel() * v.element_size()
            + o.numel() * o.element_size()
        )
        return mem_size / ms / 1e6

    print(
        f"bench_fmha_blackwell (batch_size={batch_size}, qkv_len={qkv_len}, num_heads={num_heads}, head_dim={head_dim}, causal={causal}), flops: {flops(ms):.3f} TFLOPs/s, io: {io(ms):.3f} GB/s"
    )


def bench_fmha_cutedsl(
    batch_size,
    qkv_len,
    num_heads,
    head_dim,
    causal,
    dtype,
    sm_scale=None,
):
    if sm_scale is None:
        sm_scale = 1.0 / (head_dim**0.5)

    q = torch.randn(
        batch_size * qkv_len, num_heads, head_dim, dtype=dtype, device="cuda"
    )
    k = torch.randn(
        batch_size * qkv_len, num_heads, head_dim, dtype=dtype, device="cuda"
    )
    v = torch.randn(
        batch_size * qkv_len, num_heads, head_dim, dtype=dtype, device="cuda"
    )

    qo_indptr = (
        torch.arange(0, batch_size + 1, device="cuda", dtype=torch.int32) * qkv_len
    )
    kv_indptr = (
        torch.arange(0, batch_size + 1, device="cuda", dtype=torch.int32) * qkv_len
    )

    wrapper = BatchPrefillCuteDSLWrapper(
        torch.empty(128 * 1024 * 1024, device="cuda", dtype=torch.uint8),
    )
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        num_heads,
        num_heads,
        head_dim,
        head_dim_vo=head_dim,
        causal=causal,
        sm_scale=sm_scale,
        q_data_type=dtype,
        kv_data_type=dtype,
    )
    o = wrapper.run(q, k, v)
    measurements = bench_gpu_time(
        lambda: wrapper.run(q, k, v),
        dry_run_time_ms=100,
        repeat_time_ms=1000,
    )
    ms = np.median(measurements)

    def flops(ms):
        if causal:
            return batch_size * qkv_len * qkv_len * num_heads * head_dim * 2 / ms / 1e9
        else:
            return batch_size * qkv_len * qkv_len * num_heads * head_dim * 4 / ms / 1e9

    def io(ms):
        mem_size = (
            q.numel() * q.element_size()
            + k.numel() * k.element_size()
            + v.numel() * v.element_size()
            + o.numel() * o.element_size()
        )
        return mem_size / ms / 1e6

    print(
        f"bench_fmha_cutedsl (batch_size={batch_size}, qkv_len={qkv_len}, num_heads={num_heads}, head_dim={head_dim}, causal={causal}), flops: {flops(ms):.3f} TFLOPs/s, io: {io(ms):.3f} GB/s"
    )


if __name__ == "__main__":
    # bench_fmha_blackwell(128, 512, 32, 128, False, torch.bfloat16)
    # bench_fmha_blackwell(64, 1024, 32, 128, False, torch.bfloat16)
    # bench_fmha_blackwell(32, 2048, 32, 128, False, torch.bfloat16)
    # bench_fmha_blackwell(16, 4096, 32, 128, False, torch.bfloat16)
    # bench_fmha_blackwell(8, 8192, 32, 128, False, torch.bfloat16)
    # bench_fmha_blackwell(4, 16384, 32, 128, False, torch.bfloat16)
    # bench_fmha_blackwell(2, 32768, 32, 128, False, torch.bfloat16)
    # bench_fmha_blackwell(1, 65536, 32, 128, False, torch.bfloat16)

    # bench_fmha_blackwell(128, 512, 32, 128, True, torch.bfloat16)
    # bench_fmha_blackwell(64, 1024, 32, 128, True, torch.bfloat16)
    # bench_fmha_blackwell(32, 2048, 32, 128, True, torch.bfloat16)
    # bench_fmha_blackwell(16, 4096, 32, 128, True, torch.bfloat16)
    # bench_fmha_blackwell(8, 8192, 32, 128, True, torch.bfloat16)
    # bench_fmha_blackwell(4, 16384, 32, 128, True, torch.bfloat16)
    # bench_fmha_blackwell(2, 32768, 32, 128, True, torch.bfloat16)
    # bench_fmha_blackwell(1, 65536, 32, 128, True, torch.bfloat16)

    bench_fmha_cutedsl(128, 512, 32, 128, True, torch.bfloat16)
