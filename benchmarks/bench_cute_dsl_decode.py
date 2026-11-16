# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Benchmark the cute-dsl GQA decode backend against fa2/trtllm-gen.

Runs paged batch decode through ``flashinfer.BatchDecodeWithPagedKVCacheWrapper``
with backend="cute-dsl" alongside the same workload through backend="fa2" so
they can be compared side-by-side.

Example::

    python benchmarks/bench_cute_dsl_decode.py

B200 Results::

b= 1 mtp=0 s= 1024 pg=16 h_q=64 h_kv=8 d=128 dtype=torch.bfloat16
          fa2:   13.051 us      0.3 TB/s
   trtllm-gen:    6.086 us      0.7 TB/s
     cute-dsl:    4.449 us      1.0 TB/s
b= 1 mtp=0 s= 4096 pg=16 h_q=64 h_kv=8 d=128 dtype=torch.bfloat16
          fa2:   17.929 us      0.9 TB/s
   trtllm-gen:    8.077 us      2.1 TB/s
     cute-dsl:    8.204 us      2.0 TB/s
b= 1 mtp=0 s=16384 pg=16 h_q=64 h_kv=8 d=128 dtype=torch.bfloat16
          fa2:   44.855 us      1.5 TB/s
   trtllm-gen:   17.410 us      3.9 TB/s
     cute-dsl:   14.782 us      4.5 TB/s
b= 8 mtp=0 s= 1024 pg=16 h_q=64 h_kv=8 d=128 dtype=torch.bfloat16
          fa2:   27.957 us      1.2 TB/s
   trtllm-gen:   10.262 us      3.3 TB/s
     cute-dsl:    8.631 us      3.9 TB/s
b= 8 mtp=0 s= 4096 pg=16 h_q=64 h_kv=8 d=128 dtype=torch.bfloat16
          fa2:   86.036 us      1.6 TB/s
   trtllm-gen:   24.685 us      5.4 TB/s
     cute-dsl:   23.062 us      5.8 TB/s
b= 8 mtp=0 s=16384 pg=16 h_q=64 h_kv=8 d=128 dtype=torch.bfloat16
          fa2:  303.641 us      1.8 TB/s
   trtllm-gen:   78.237 us      6.9 TB/s
     cute-dsl:   77.279 us      7.0 TB/s
b=64 mtp=0 s= 1024 pg=16 h_q=64 h_kv=8 d=128 dtype=torch.bfloat16
          fa2:  163.317 us      1.7 TB/s
   trtllm-gen:   43.196 us      6.3 TB/s
     cute-dsl:   44.837 us      6.0 TB/s
b=64 mtp=0 s= 4096 pg=16 h_q=64 h_kv=8 d=128 dtype=torch.bfloat16
          fa2:  598.808 us      1.8 TB/s
   trtllm-gen:  155.587 us      6.9 TB/s
     cute-dsl:  151.897 us      7.1 TB/s
b=64 mtp=0 s=16384 pg=16 h_q=64 h_kv=8 d=128 dtype=torch.bfloat16
          fa2: 2352.243 us      1.8 TB/s
   trtllm-gen:  607.291 us      7.1 TB/s
     cute-dsl:  599.133 us      7.2 TB/s
b= 1 mtp=3 s= 1024 pg=16 h_q=64 h_kv=8 d=128 dtype=torch.bfloat16
   trtllm-gen:    7.234 us      0.6 TB/s
     cute-dsl:    5.364 us      0.8 TB/s
b= 1 mtp=3 s= 4096 pg=16 h_q=64 h_kv=8 d=128 dtype=torch.bfloat16
   trtllm-gen:   10.311 us      1.6 TB/s
     cute-dsl:   10.598 us      1.6 TB/s
b= 1 mtp=3 s=16384 pg=16 h_q=64 h_kv=8 d=128 dtype=torch.bfloat16
   trtllm-gen:   18.622 us      3.6 TB/s
     cute-dsl:   16.848 us      4.0 TB/s
b= 8 mtp=3 s= 1024 pg=16 h_q=64 h_kv=8 d=128 dtype=torch.bfloat16
   trtllm-gen:   11.381 us      3.0 TB/s
     cute-dsl:   10.379 us      3.3 TB/s
b= 8 mtp=3 s= 4096 pg=16 h_q=64 h_kv=8 d=128 dtype=torch.bfloat16
   trtllm-gen:   25.358 us      5.3 TB/s
     cute-dsl:   25.004 us      5.4 TB/s
b= 8 mtp=3 s=16384 pg=16 h_q=64 h_kv=8 d=128 dtype=torch.bfloat16
   trtllm-gen:   80.188 us      6.7 TB/s
     cute-dsl:   84.371 us      6.4 TB/s
b=64 mtp=3 s= 1024 pg=16 h_q=64 h_kv=8 d=128 dtype=torch.bfloat16
   trtllm-gen:   49.976 us      5.5 TB/s
     cute-dsl:   49.739 us      5.6 TB/s
b=64 mtp=3 s= 4096 pg=16 h_q=64 h_kv=8 d=128 dtype=torch.bfloat16
   trtllm-gen:  163.325 us      6.6 TB/s
     cute-dsl:  163.233 us      6.6 TB/s
b=64 mtp=3 s=16384 pg=16 h_q=64 h_kv=8 d=128 dtype=torch.bfloat16
   trtllm-gen:  640.689 us      6.7 TB/s
     cute-dsl:  671.059 us      6.4 TB/s
"""

import sys

import numpy as np
import torch

import flashinfer
from flashinfer.cute_dsl.utils import is_cute_dsl_available
from flashinfer.testing import bench_gpu_time
from flashinfer.utils import is_sm100a_supported


if not torch.cuda.is_available() or not is_sm100a_supported(torch.device("cuda")):
    print("Skipping: cute-dsl GQA decode requires Blackwell (SM100a)")
    sys.exit(0)

if not is_cute_dsl_available():
    print("Skipping: nvidia-cutlass-dsl not installed")
    sys.exit(0)


def _build_inputs(
    batch_size,
    prediction,
    seq_len,
    page_size,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    dtype,
):
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
            use_cuda_graph=True,
            input_args=(q, kv, o),
            cold_l2_cache=True,
        )
    )


def bench_one(
    batch_size,
    prediction,
    seq_len,
    page_size,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    dtype,
):
    print(
        f"b={batch_size:2d} mtp={prediction - 1} s={seq_len:5d} pg={page_size:2d} "
        f"h_q={num_qo_heads} h_kv={num_kv_heads} d={head_dim} dtype={dtype}"
    )

    q, kv, kv_indptr, kv_indices, last_page_len = _build_inputs(
        batch_size,
        prediction,
        seq_len,
        page_size,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        dtype,
    )
    io_bytes = q.nbytes * 2 + kv.nbytes

    for backend in ("fa2", "trtllm-gen", "cute-dsl"):
        if backend == "fa2" and prediction > 1:
            continue  # fa2 IMAs in this case

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
        print(f"  {backend:>11s}: {ms * 1000:8.3f} us  {bw:7.1f} TB/s")


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
