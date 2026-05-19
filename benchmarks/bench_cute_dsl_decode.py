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


def _build_inputs(batch_size, prediction, seq_len, page_size, num_qo_heads, num_kv_heads, head_dim, dtype):
    pages_per_seq = (seq_len + page_size - 1) // page_size
    total_pages = pages_per_seq * batch_size
    q = torch.randn(
        batch_size * prediction, num_qo_heads, head_dim, device="cuda", dtype=dtype
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


def _bench(wrapper, q, kv, o):
    return np.mean(
        bench_gpu_time(
            lambda q, kv, o: wrapper.run(q, kv, out=o, enable_pdl=True),
            dry_run_iters=10,
            repeat_iters=50,
            enable_cupti=False,
            use_cuda_graph=True,
            input_args=(q, kv, o),
            cold_l2_cache=True,
        )
    )


def bench_one(
    batch_size, prediction, seq_len, page_size, num_qo_heads, num_kv_heads, head_dim, dtype
):
    print(
        f"b={batch_size:2d} mtp={prediction - 1} s={seq_len:5d} pg={page_size:2d} "
        f"h_q={num_qo_heads} h_kv={num_kv_heads} d={head_dim} dtype={dtype}"
    )

    q, kv, kv_indptr, kv_indices, last_page_len = _build_inputs(
        batch_size, prediction, seq_len, page_size, num_qo_heads, num_kv_heads, head_dim, dtype
    )
    io_bytes = q.nbytes * 2 + kv.nbytes

    for backend in ("fa2", "trtllm-gen", "cute-dsl"):
        if backend == "fa2" and prediction > 1:
            continue # fa2 IMAs in this case

        # trtllm-gen backend expects workspace to be zero-init for semaphores
        workspace_alloc = torch.zeros if backend == "trtllm-gen" else torch.empty
        workspace = workspace_alloc(64 * 1024 * 1024, dtype=torch.uint8, device="cuda")

        # cute-dsl backend expects out to be zero-init if provided
        out_alloc = torch.zeros_like if backend == "cute-dsl" else torch.empty_like
        out = out_alloc(q)

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
            q_len_per_req=prediction,
        )

        ms = _bench(wrapper, q, kv, out)
        bw = io_bytes / ms / 1e9
        print(f"  {backend:>11s}: {ms * 1000:8.3f} us  {bw:7.1f} GB/s")


if __name__ == "__main__":
    num_qo_heads = 64
    num_kv_heads = 8
    page_size = 16

    for dtype in (torch.bfloat16,):
        for head_dim in (128,):
            for prediction in (1, 4):
                for batch_size in (1, 8, 64):
                    for seq_len in (1024, 4096, 16384):
                        bench_one(
                            batch_size,
                            prediction,
                            seq_len,
                            page_size,
                            num_qo_heads,
                            num_kv_heads,
                            head_dim,
                            dtype,
                        )
