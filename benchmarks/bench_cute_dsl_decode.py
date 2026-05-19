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
          fa2:   12.243 us      0.3 GB/s
   trtllm-gen:    6.301 us      0.7 GB/s
     cute-dsl:    4.862 us      0.9 GB/s
b= 1 mtp=0 s= 4096 pg=16 h_q=64 h_kv=8 d=128 dtype=torch.bfloat16
          fa2:   18.184 us      0.9 GB/s
   trtllm-gen:    8.128 us      2.1 GB/s
     cute-dsl:   11.002 us      1.5 GB/s
b= 1 mtp=0 s=16384 pg=16 h_q=64 h_kv=8 d=128 dtype=torch.bfloat16
          fa2:   44.368 us      1.5 GB/s
   trtllm-gen:   17.782 us      3.8 GB/s
     cute-dsl:   18.137 us      3.7 GB/s
b= 8 mtp=0 s= 1024 pg=16 h_q=64 h_kv=8 d=128 dtype=torch.bfloat16
          fa2:   28.691 us      1.2 GB/s
   trtllm-gen:   10.577 us      3.2 GB/s
     cute-dsl:    8.952 us      3.8 GB/s
b= 8 mtp=0 s= 4096 pg=16 h_q=64 h_kv=8 d=128 dtype=torch.bfloat16
          fa2:   85.133 us      1.6 GB/s
   trtllm-gen:   24.139 us      5.6 GB/s
     cute-dsl:   22.920 us      5.9 GB/s
b= 8 mtp=0 s=16384 pg=16 h_q=64 h_kv=8 d=128 dtype=torch.bfloat16
          fa2:  300.016 us      1.8 GB/s
   trtllm-gen:   78.230 us      6.9 GB/s
     cute-dsl:   77.682 us      6.9 GB/s
b=64 mtp=0 s= 1024 pg=16 h_q=64 h_kv=8 d=128 dtype=torch.bfloat16
          fa2:  159.955 us      1.7 GB/s
   trtllm-gen:   43.420 us      6.2 GB/s
     cute-dsl:   45.634 us      5.9 GB/s
b=64 mtp=0 s= 4096 pg=16 h_q=64 h_kv=8 d=128 dtype=torch.bfloat16
          fa2:  589.830 us      1.8 GB/s
   trtllm-gen:  156.094 us      6.9 GB/s
     cute-dsl:  152.400 us      7.1 GB/s
b=64 mtp=0 s=16384 pg=16 h_q=64 h_kv=8 d=128 dtype=torch.bfloat16
          fa2: 2314.402 us      1.9 GB/s
   trtllm-gen:  606.548 us      7.1 GB/s
     cute-dsl:  598.255 us      7.2 GB/s
b= 1 mtp=3 s= 1024 pg=16 h_q=64 h_kv=8 d=128 dtype=torch.bfloat16
   trtllm-gen:    6.703 us      0.6 GB/s
     cute-dsl:    6.693 us      0.6 GB/s
b= 1 mtp=3 s= 4096 pg=16 h_q=64 h_kv=8 d=128 dtype=torch.bfloat16
   trtllm-gen:   10.593 us      1.6 GB/s
     cute-dsl:   11.871 us      1.4 GB/s
b= 1 mtp=3 s=16384 pg=16 h_q=64 h_kv=8 d=128 dtype=torch.bfloat16
   trtllm-gen:   19.029 us      3.5 GB/s
     cute-dsl:   21.917 us      3.1 GB/s
b= 8 mtp=3 s= 1024 pg=16 h_q=64 h_kv=8 d=128 dtype=torch.bfloat16
   trtllm-gen:   11.857 us      2.9 GB/s
     cute-dsl:   11.616 us      3.0 GB/s
b= 8 mtp=3 s= 4096 pg=16 h_q=64 h_kv=8 d=128 dtype=torch.bfloat16
   trtllm-gen:   25.721 us      5.3 GB/s
     cute-dsl:   29.520 us      4.6 GB/s
b= 8 mtp=3 s=16384 pg=16 h_q=64 h_kv=8 d=128 dtype=torch.bfloat16
   trtllm-gen:   80.154 us      6.7 GB/s
     cute-dsl:   90.475 us      5.9 GB/s
b=64 mtp=3 s= 1024 pg=16 h_q=64 h_kv=8 d=128 dtype=torch.bfloat16
   trtllm-gen:   49.531 us      5.6 GB/s
     cute-dsl:   63.087 us      4.4 GB/s
b=64 mtp=3 s= 4096 pg=16 h_q=64 h_kv=8 d=128 dtype=torch.bfloat16
   trtllm-gen:  162.000 us      6.7 GB/s
     cute-dsl:  180.057 us      6.0 GB/s
b=64 mtp=3 s=16384 pg=16 h_q=64 h_kv=8 d=128 dtype=torch.bfloat16
   trtllm-gen:  621.074 us      6.9 GB/s
     cute-dsl:  690.734 us      6.2 GB/s
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
