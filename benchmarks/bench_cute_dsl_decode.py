# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Benchmark the cute-dsl GQA decode backend against fa2/trtllm-gen.

Runs paged batch decode through ``flashinfer.BatchDecodeWithPagedKVCacheWrapper``
with backend="cute-dsl" alongside the same workload through backend="fa2" so
they can be compared side-by-side.

Example::

    python benchmarks/bench_cute_dsl_decode.py
"""

import sys

import numpy as np
import torch

import flashinfer
from flashinfer.cute_dsl.utils import is_cute_dsl_available
from flashinfer.testing.utils import bench_gpu_time
from flashinfer.utils import is_sm100a_supported


if not torch.cuda.is_available() or not is_sm100a_supported(torch.device("cuda")):
    print("Skipping: cute-dsl GQA decode requires Blackwell (SM100a)")
    sys.exit(0)

if not is_cute_dsl_available():
    print("Skipping: nvidia-cutlass-dsl not installed")
    sys.exit(0)


def _build_inputs(batch_size, seq_len, page_size, num_qo_heads, num_kv_heads, head_dim, dtype):
    pages_per_seq = (seq_len + page_size - 1) // page_size
    total_pages = pages_per_seq * batch_size
    q = torch.randn(
        batch_size, num_qo_heads, head_dim, device="cuda", dtype=dtype
    )
    kv = torch.randn(
        total_pages,
        2,
        page_size,
        num_kv_heads,
        head_dim,
        device="cuda",
        dtype=dtype,
    )
    kv_indptr = (
        torch.arange(batch_size + 1, device="cuda", dtype=torch.int32) * pages_per_seq
    )
    kv_indices = torch.arange(total_pages, device="cuda", dtype=torch.int32)
    last_page_len = torch.full(
        (batch_size,), (seq_len - 1) % page_size + 1, device="cuda", dtype=torch.int32
    )
    return q, kv, kv_indptr, kv_indices, last_page_len


def _bench(wrapper, q, kv):
    return np.median(
        bench_gpu_time(
            lambda: wrapper.run(q, kv),
            dry_run_time_ms=50,
            repeat_time_ms=200,
            enable_cupti=True,
        )
    )


def bench_one(
    batch_size, seq_len, page_size, num_qo_heads, num_kv_heads, head_dim, dtype
):
    q, kv, kv_indptr, kv_indices, last_page_len = _build_inputs(
        batch_size, seq_len, page_size, num_qo_heads, num_kv_heads, head_dim, dtype
    )

    results = {}
    for backend in ("fa2", "cute-dsl"):
        workspace = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device="cuda")
        wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            workspace, kv_layout="NHD", backend=backend
        )
        wrapper.plan(
            kv_indptr,
            kv_indices,
            last_page_len,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            q_data_type=dtype,
            kv_data_type=dtype,
        )
        # warmup compile (cute-dsl), then time
        wrapper.run(q, kv)
        results[backend] = _bench(wrapper, q, kv)

    io_bytes = q.numel() * q.element_size() + kv.numel() * kv.element_size()
    print(
        f"b={batch_size:4d} s={seq_len:6d} p={page_size:3d} "
        f"h_q={num_qo_heads} h_kv={num_kv_heads} d={head_dim} dtype={dtype}"
    )
    for backend, ms in results.items():
        bw = io_bytes / ms / 1e9
        print(f"  {backend:>9s}: {ms * 1000:8.3f} us  {bw:7.1f} GB/s")


if __name__ == "__main__":
    num_qo_heads = 32
    num_kv_heads = 4
    head_dim = 128
    page_size = 16

    for dtype in (torch.bfloat16,):
        for batch_size in (1, 4, 16, 64):
            for seq_len in (1024, 4096, 16384):
                bench_one(
                    batch_size,
                    seq_len,
                    page_size,
                    num_qo_heads,
                    num_kv_heads,
                    head_dim,
                    dtype,
                )
