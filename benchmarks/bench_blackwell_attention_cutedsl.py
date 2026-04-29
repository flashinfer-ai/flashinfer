# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import sys

import numpy as np
import torch

import flashinfer
from flashinfer.cute_dsl.utils import is_cute_dsl_available
from flashinfer.testing.utils import bench_gpu_time
from flashinfer.utils import is_sm100a_supported

if not is_cute_dsl_available():
    print("Skipping: nvidia-cutlass-dsl package not installed")
    sys.exit(0)

from flashinfer.cute_dsl.attention import BatchPrefillCuteDSLWrapper


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
        enable_cupti=True,
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
        enable_cupti=True,
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
    if not is_sm100a_supported(torch.device("cuda")):
        print("Skipping: requires SM100+")
        sys.exit(0)

    configs = [
        (128, 512, 32, 128, True, torch.bfloat16),
        (64, 1024, 32, 128, True, torch.bfloat16),
        (32, 2048, 32, 128, True, torch.bfloat16),
        (16, 4096, 32, 128, True, torch.bfloat16),
        (8, 8192, 32, 128, True, torch.bfloat16),
        (4, 16384, 32, 128, True, torch.bfloat16),
        (2, 32768, 32, 128, True, torch.bfloat16),
        (1, 65536, 32, 128, True, torch.bfloat16),
    ]

    print("=== CUTLASS (via BatchPrefillWithRaggedKVCacheWrapper) ===")
    for cfg in configs:
        bench_fmha_blackwell(*cfg)
    print()
    print("=== CuTe DSL (via BatchPrefillCuteDSLWrapper) ===")
    for cfg in configs:
        bench_fmha_cutedsl(*cfg)
