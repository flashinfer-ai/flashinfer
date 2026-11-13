"""Whole-call LiteTopK vs DeepGEMM logits + FlashInfer ragged top-k.

Reports cold-L2 GPU span (first activity to last; CUDA-event fallback).
Compilation and input allocation are excluded. LiteTopK reuses caller-owned
workspace/output; its reported memory includes those retained buffers plus any
extra allocation during the call. Dense peak includes logits and top-k scratch.

    python benchmarks/bench_dsa_indexer_topk.py --num-q 8192 --heads 32 --top-k 2048
"""

import argparse
import statistics

import torch

from flashinfer.dsa_indexer import dsa_indexer_topk, dsa_indexer_topk_workspace
from flashinfer.testing import bench_gpu_time


def peak_bytes(fn):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() - base


def bench(fn):
    fn()  # Compile and initialize outside the measured region.
    torch.cuda.synchronize()
    return statistics.median(
        bench_gpu_time(
            fn,
            dry_run_iters=3,
            repeat_iters=10,
            enable_cupti=True,
            use_cuda_graph=False,
            cold_l2_cache=True,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-q", type=int, default=8192)
    parser.add_argument("--heads", type=int, choices=[32, 64], default=32)
    parser.add_argument("--top-k", type=int, choices=[512, 1024, 2048], default=2048)
    parser.add_argument("--cand-cap", type=int, default=12288)
    parser.add_argument(
        "--seq-kv", type=int, nargs="+", default=[262144, 524288, 1048576]
    )
    args = parser.parse_args()
    if args.num_q < 1:
        parser.error("--num-q must be positive")
    if not 4096 <= args.cand_cap <= 12288 or args.cand_cap % 4:
        parser.error("--cand-cap must be in [4096, 12288] and a multiple of 4")
    if any(n < args.num_q + args.top_k - 1 or n > 2**20 or n % 4 for n in args.seq_kv):
        parser.error("--seq-kv must be a multiple of 4 in [num_q + top_k - 1, 2**20]")
    import deep_gemm
    import flashinfer

    print(
        f"GPU: {torch.cuda.get_device_name()}, rows={args.num_q}, heads={args.heads}, k={args.top_k}"
    )
    print(
        "Cold L2; whole-call GPU span; CUPTI with CUDA-event fallback; no compile time."
    )
    print(
        " seq_kv  dense ms  fused ms  speedup  dense peak GiB  logits GiB  fused scratch+out GiB"
    )
    torch.manual_seed(0)
    for seq_kv in args.seq_kv:
        q = torch.randn(args.num_q, args.heads, 128, device="cuda").to(
            torch.float8_e4m3fn
        )
        kv = torch.randn(seq_kv, 128, device="cuda").to(torch.float8_e4m3fn)
        scales = torch.ones(seq_kv, device="cuda")
        weights = torch.randn(args.num_q, args.heads, device="cuda") / args.heads
        starts = torch.zeros(args.num_q, dtype=torch.int32, device="cuda")
        ends = torch.arange(
            seq_kv - args.num_q + 1, seq_kv + 1, dtype=torch.int32, device="cuda"
        )
        workspace = dsa_indexer_topk_workspace(
            q.device, args.num_q, args.heads, seq_kv, cand_cap=args.cand_cap
        )
        out = torch.empty(args.num_q, args.top_k, dtype=torch.int32, device="cuda")

        def fused(workspace=workspace, out=out):
            return dsa_indexer_topk(
                q,
                kv,
                scales,
                weights,
                ends,
                args.top_k,
                cu_start=starts,
                cand_cap=args.cand_cap,
                workspace=workspace,
                out=out,
            )

        fused_ms = bench(fused)
        fused_gib = (
            sum(t.nbytes for t in workspace) + out.nbytes + peak_bytes(fused)
        ) / 2**30
        # Free the fused retained buffers before measuring dense memory headroom.
        del fused, workspace, out
        torch.cuda.empty_cache()

        # Recent DeepGEMM uses packed MX scales on SM100 and leaves logits
        # beyond each row's valid span uninitialized. The ragged selector masks
        # those tails without allocating a full [rows, keys] boolean mask.
        if hasattr(deep_gemm, "fp8_fp4_mqa_logits"):
            q_scales = torch.full(
                (args.num_q, args.heads), 0x7F7F7F7F, dtype=torch.int32, device="cuda"
            )
            k_scales = torch.full(
                (seq_kv,), 0x7F7F7F7F, dtype=torch.int32, device="cuda"
            )

            def dense_logits():
                return deep_gemm.fp8_fp4_mqa_logits(
                    (q, q_scales),
                    (kv, k_scales),
                    weights,
                    starts,
                    ends,
                    seq_kv,
                    logits_dtype=torch.float32,
                )
        else:

            def dense_logits():
                return deep_gemm.fp8_mqa_logits(q, (kv, scales), weights, starts, ends)

        def dense():
            return flashinfer.top_k_ragged_transform(
                dense_logits(), starts, ends, args.top_k
            )

        try:
            dense_ms = bench(dense)
            dense_gib = peak_bytes(dense) / 2**30
            speedup = f"{dense_ms / fused_ms:7.2f}x"
            dense_time, dense_memory = f"{dense_ms:9.3f}", f"{dense_gib:15.3f}"
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            dense_time, dense_memory, speedup = f"{'OOM':>9}", f"{'OOM':>15}", " " * 8
        logits_gib = 4 * args.num_q * seq_kv / 2**30
        print(
            f"{seq_kv:7d} {dense_time} {fused_ms:9.3f} {speedup} {dense_memory} "
            f"{logits_gib:11.3f} {fused_gib:22.3f}"
        )


if __name__ == "__main__":
    main()
