"""Paged FA2 attention planned from host index tensors: kernel time with a cold L2 cache.

The wrappers are ordinary (no CUDA graph) and ``plan()`` receives the page table as
host tensors, so ``plan()`` decides where the index arrays live on the device.

    python benchmarks/bench_packed_index_buffers.py [--output results.json]
"""

import argparse
import json
import time

import numpy as np
import torch

import flashinfer
from flashinfer.testing import bench_gpu_time

NUM_QO_HEADS, NUM_KV_HEADS, HEAD_DIM, PAGE_SIZE = 32, 8, 128, 16
SHAPES = ((1, 64), (1, 512), (1, 2048), (8, 512), (64, 2048))


def bench_case(kind, batch_size, kv_len, dtype, device):
    pages = (kv_len + PAGE_SIZE - 1) // PAGE_SIZE
    kv_indptr = torch.arange(batch_size + 1, dtype=torch.int32) * pages
    kv_indices = torch.randperm(batch_size * pages).int()
    kv_last_page_len = torch.full(
        (batch_size,), (kv_len - 1) % PAGE_SIZE + 1, dtype=torch.int32
    )
    qo_indptr = torch.arange(batch_size + 1, dtype=torch.int32)
    kv_data = torch.randn(
        batch_size * pages,
        2,
        PAGE_SIZE,
        NUM_KV_HEADS,
        HEAD_DIM,
        device=device,
        dtype=dtype,
    )
    q = torch.randn(batch_size, NUM_QO_HEADS, HEAD_DIM, device=device, dtype=dtype)
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)
    common = (NUM_QO_HEADS, NUM_KV_HEADS, HEAD_DIM, PAGE_SIZE)
    if kind == "prefill":
        wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            workspace, "NHD", backend="fa2"
        )

        def plan():
            wrapper.plan(
                qo_indptr,
                kv_indptr,
                kv_indices,
                kv_last_page_len,
                *common,
                causal=True,
                q_data_type=dtype,
                kv_data_type=dtype,
            )
    else:
        wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            workspace, "NHD", use_tensor_cores=kind == "decode_tc", backend="fa2"
        )

        def plan():
            wrapper.plan(
                kv_indptr,
                kv_indices,
                kv_last_page_len,
                *common,
                q_data_type=dtype,
                kv_data_type=dtype,
            )

    plan_ns = []
    for i in range(250):
        start = time.perf_counter_ns()
        plan()
        torch.cuda.synchronize()
        if i >= 50:
            plan_ns.append(time.perf_counter_ns() - start)
    times_ms = bench_gpu_time(
        lambda: wrapper.run(q, kv_data),
        dry_run_iters=20,
        repeat_iters=200,
        enable_cupti=True,
        cold_l2_cache=True,
    )
    times_ms = np.asarray(times_ms, dtype=float).ravel()
    run_us = float(np.median(times_ms) if times_ms.size > 2 else times_ms[0]) * 1e3
    return dict(run_us=run_us, plan_us=float(np.median(plan_ns)) / 1e3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    device = torch.device("cuda:0")
    torch.manual_seed(0)
    results = {}
    for dtype in (torch.float16, torch.bfloat16):
        for kind in ("decode", "decode_tc", "prefill"):
            for batch_size, kv_len in SHAPES:
                name = f"{kind}/{str(dtype).split('.')[-1]}/{batch_size}x{kv_len}"
                results[name] = bench_case(kind, batch_size, kv_len, dtype, device)
                print(
                    f"{name}: run {results[name]['run_us']:.2f} us, "
                    f"plan {results[name]['plan_us']:.1f} us",
                    flush=True,
                )
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f)


if __name__ == "__main__":
    main()
