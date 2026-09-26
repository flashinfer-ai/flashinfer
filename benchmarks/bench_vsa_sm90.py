"""Benchmark SM90 VSA against the generic block-sparse backends.

The default shape mirrors the MiniMax-H3 workload from PR #4944.

Example:
    python benchmarks/bench_vsa_sm90.py --dtype fp16
"""

import argparse
import statistics
import time

import torch

from flashinfer import BlockSparseAttentionWrapper
from flashinfer.testing import bench_gpu_time


def build_bsr(MB: int, NB: int, k: int, device):
    """Random BSR with exactly k blocks per row (VSA top-k style)."""
    indptr = torch.zeros(MB + 1, dtype=torch.int32)
    parts = []
    for i in range(MB):
        col = torch.randperm(NB, device="cpu")[:k].sort().values.to(torch.int32)
        indptr[i + 1] = indptr[i] + k
        parts.append(col)
    return indptr.to(device), torch.cat(parts).to(device)


def main():
    """Parse benchmark options and report timings for each selected backend."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=109632)
    ap.add_argument("--heads", type=int, default=7)
    ap.add_argument("--kv-heads", type=int, default=7)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--blk", type=int, default=64)
    ap.add_argument("--topk", type=int, default=64)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    ap.add_argument("--backends", default="vsa_sm90_blk64,fa2,auto,fa3")
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    device = torch.device("cuda")
    torch.manual_seed(42)

    M = N = args.seq
    R = C = args.blk
    MB = (M + R - 1) // R
    NB = (N + C - 1) // C
    H, KVH, D = args.heads, args.kv_heads, args.head_dim

    q = torch.randn(M, H, D, dtype=dtype, device=device)
    k = torch.randn(N, KVH, D, dtype=dtype, device=device)
    v = torch.randn(N, KVH, D, dtype=dtype, device=device)
    indptr, indices = build_bsr(MB, NB, args.topk, device)
    nnz = int(indices.numel())
    flops = 4 * nnz * R * C * H * D  # 2*QK^T + 2*PV over selected blocks

    print(
        f"workload: seq={M} H={H}/{KVH} D={D} blk={R} MB={MB} NB={NB} "
        f"topk={args.topk} nnz={nnz} dtype={args.dtype}"
    )
    print(
        f"device: {torch.cuda.get_device_name(0)}  ({flops / 1e12:.2f} TFLOP of sparse work)"
    )
    workspace = torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device=device)

    for backend in args.backends.split(","):
        try:
            wrapper = BlockSparseAttentionWrapper(workspace, backend=backend)
            wrapper.plan(indptr, indices, M, N, R, C, H, KVH, D, q_data_type=dtype)
            wrapper.run(q, k, v)
            torch.cuda.synchronize()

            times = bench_gpu_time(
                wrapper.run,
                input_args=(q, k, v),
                enable_cupti=True,
                dry_run_iters=10,
                repeat_iters=args.iters,
            )
            med = statistics.median(times)
            # wall clock incl. host-side plan-consumed metadata & launch overhead
            t0 = time.perf_counter()
            for _ in range(20):
                wrapper.run(q, k, v)
            torch.cuda.synchronize()
            wall = (time.perf_counter() - t0) / 20 * 1e3

            print(
                f"  {backend:>6s}: gpu {med:9.3f} ms | {flops / (med * 1e-3) / 1e12:8.1f} TFLOPS "
                f"| wall {wall:9.3f} ms | max {max(times):9.3f} ms"
            )
        except Exception as e:
            print(f"  {backend:>6s}: FAILED — {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
