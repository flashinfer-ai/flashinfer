"""
Benchmark flashinfer.dsa_indexer.dsa_indexer_topk against the dense DSA indexer
(deep_gemm.fp8_mqa_logits materializing [num_q, seq_kv] logits + flashinfer.top_k).

Inputs are synthetic; the prefix is simply kv[:12288]. In serving, placing the
previous chunk's top-k in the prefix gives a tighter seed. The dense peak includes
the flashinfer.top_k workspace; "logits GiB" is the logits matrix alone.

    python benchmarks/bench_dsa_indexer_topk.py --num-q 8192 --seq-kv 262144 1048576
"""

import argparse

import numpy as np
import torch

import deep_gemm
import flashinfer
from flashinfer.dsa_indexer import dsa_indexer_topk
from flashinfer.testing import bench_gpu_time


def peak_gib(fn):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    fn()
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - base) / 2**30


def bench(fn):
    return np.median(bench_gpu_time(fn, dry_run_iters=3, repeat_iters=10))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-q", type=int, default=8192)
    parser.add_argument(
        "--seq-kv", type=int, nargs="+", default=[262144, 524288, 1048576]
    )
    parser.add_argument("--prefix", type=int, default=12288)
    args = parser.parse_args()
    top_k, num_q, prefix_len = 2048, args.num_q, args.prefix
    print(
        " seq_kv  dense ms  fused ms  speedup  dense peak GiB  logits GiB  fused peak GiB"
    )
    for seq_kv in args.seq_kv:
        assert seq_kv % 4 == 0 and seq_kv - num_q + 1 >= prefix_len
        q = (torch.randn(num_q, 32, 128, device="cuda") * 0.5).to(torch.float8_e4m3fn)
        kv = (torch.randn(seq_kv, 128, device="cuda") * 0.5).to(torch.float8_e4m3fn)
        kv_scales = torch.rand(seq_kv, device="cuda") + 0.5
        weights = torch.randn(num_q, 32, device="cuda") * 0.1
        # Causal chunked prefill: query i is token seq_kv - num_q + i.
        ks = torch.zeros(num_q, dtype=torch.int32, device="cuda")
        ke = torch.arange(
            seq_kv - num_q + 1, seq_kv + 1, dtype=torch.int32, device="cuda"
        )
        prefix_ke = torch.full_like(ke, prefix_len)
        prefix_kv = (kv[:prefix_len], kv_scales[:prefix_len])

        def dense():
            logits = deep_gemm.fp8_mqa_logits(q, (kv, kv_scales), weights, ks, ke)
            return flashinfer.top_k(logits, top_k)

        def fused():
            prefix = deep_gemm.fp8_mqa_logits(
                q, prefix_kv, weights, ks, prefix_ke, clean_logits=False
            )
            return dsa_indexer_topk(q, kv, kv_scales, weights, prefix, ke, top_k)

        fused_ms, fused_gib = bench(fused), peak_gib(fused)
        logits_gib = 4 * num_q * seq_kv / 2**30
        torch.cuda.empty_cache()
        try:
            dense_ms, dense_gib = bench(dense), peak_gib(dense)
            speedup = f"{dense_ms / fused_ms:7.2f}x"
            dense_ms, dense_gib = f"{dense_ms:9.2f}", f"{dense_gib:15.1f}"
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            dense_ms, dense_gib, speedup = f"{'OOM':>9}", f"{'OOM':>15}", " " * 8
        print(
            f"{seq_kv:7d} {dense_ms} {fused_ms:9.2f} {speedup} {dense_gib} "
            f"{logits_gib:11.1f} {fused_gib:15.1f}"
        )


if __name__ == "__main__":
    main()
