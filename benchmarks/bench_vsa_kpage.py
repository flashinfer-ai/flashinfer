# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0

"""Measure KPage layout building separately from warm, reused execution.

python benchmarks/bench_vsa_kpage.py --block-size 64
python benchmarks/bench_vsa_kpage.py --block-size 128
"""

import argparse
import json
import statistics
import time

import torch

from flashinfer import BlockSparseAttentionWrapper
from flashinfer.testing import bench_gpu_time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block-size", type=int, choices=(64, 128), default=128)
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--topk", type=int, default=16)
    parser.add_argument("--kv-valid", type=int, default=40)
    args = parser.parse_args()
    block, seq, heads = args.block_size, args.seq_len, args.heads
    if seq <= 0 or seq % block or heads < 1 or not 0 < args.kv_valid <= block:
        parser.error(
            "require positive aligned sequence/head counts and 0 < kv-valid <= block-size"
        )
    nb = seq // block
    if not 1 <= args.topk <= nb:
        parser.error("topk must be between 1 and the number of KV blocks")
    torch.manual_seed(42)
    q, k, v = [
        torch.randn(seq, heads, 128, device="cuda", dtype=torch.bfloat16)
        for _ in range(3)
    ]
    idx = (
        torch.stack(
            [torch.randperm(nb, device="cuda")[: args.topk] for _ in range(heads * nb)]
        )
        .reshape(heads, nb, args.topk)
        .int()
    )
    counts = torch.full((heads, nb), args.topk, device="cuda", dtype=torch.int32)
    lengths = torch.full((nb,), args.kv_valid, device="cuda", dtype=torch.int32)
    wrapper = BlockSparseAttentionWrapper(
        torch.empty(8 * 1024 * 1024, device="cuda", dtype=torch.uint8),
        backend=f"vsa_sm100_blk{block}",
    )

    def prepare():
        wrapper.plan(
            None,
            None,
            seq,
            seq,
            block,
            block,
            heads,
            heads,
            128,
            q_data_type=torch.bfloat16,
            o_data_type=torch.bfloat16,
            q2k_indices=idx,
            q2k_num=counts,
            kv_block_lens=lengths,
            kpage=True,
        )

    prepare()
    out = torch.empty_like(q)
    lse = torch.empty((seq, heads), dtype=torch.float32, device="cuda")

    def execute():
        return wrapper.run(q, k, v, out=out, lse=lse, return_lse=True)

    execute()  # Exclude first JIT from both reported measurements.
    torch.cuda.synchronize()
    build_ms = []
    for _ in range(5):
        start = time.perf_counter()
        prepare()
        torch.cuda.synchronize()
        build_ms.append((time.perf_counter() - start) * 1000)
    gpu_ms = bench_gpu_time(execute, enable_cupti=True)
    print(
        json.dumps(
            {
                "gpu": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                **vars(args),
                "warm_plan_wall_ms": statistics.median(build_ms),
                "reused_run_gpu_ms": statistics.median(gpu_ms),
                "timing": "warm JIT; synchronized wall time for plan; GPU time for run, CUPTI with CUDA-event fallback; preallocated outputs; query rows all valid",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
