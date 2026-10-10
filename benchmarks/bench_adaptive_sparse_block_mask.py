"""Benchmark adaptive sparse block-mask selection against a PyTorch pipeline."""

from __future__ import annotations

import argparse
import statistics

import torch

from flashinfer.sparse import adaptive_sparse_block_mask
from flashinfer.testing import bench_gpu_time


def _samples_ms(fn, *, enable_cupti: bool) -> list[float]:
    return bench_gpu_time(
        fn,
        enable_cupti=enable_cupti,
        dry_run_iters=5,
        repeat_iters=15,
        cold_l2_cache=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--q-blocks", type=int, default=64)
    parser.add_argument("--k-blocks", type=int, default=1024)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--cuda-events", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda")
    logits = torch.randn(
        args.batch_size,
        args.num_heads,
        args.q_blocks,
        args.k_blocks,
        dtype=torch.bfloat16,
        device=device,
    )
    length = args.k_blocks * args.block_size
    q_length = args.q_blocks * args.block_size
    q_lens = torch.full((args.batch_size,), q_length, dtype=torch.int32, device=device)
    kv_lens = torch.full((args.batch_size,), length, dtype=torch.int32, device=device)

    # alpha=1 makes every row's adaptive budget equal, allowing an equivalent
    # allocation-free torch.topk + scatter baseline.
    budget = (
        args.k_blocks
        if args.k_blocks < 56
        else int(args.k_blocks * (0.2 if args.k_blocks < 160 else 0.1)) + 30
    )
    budget = min(max(budget, 1), args.k_blocks)
    values = torch.empty(*logits.shape[:-1], budget, dtype=logits.dtype, device=device)
    indices = torch.empty(*logits.shape[:-1], budget, dtype=torch.int64, device=device)
    torch_mask = torch.empty_like(logits, dtype=torch.bool)
    kernel_mask = torch.empty_like(logits, dtype=torch.bool)
    forced = torch.zeros_like(torch_mask)
    forced[..., : min(4, args.k_blocks)] = 1
    rows = torch.arange(args.q_blocks, device=device)
    offset = max(args.k_blocks - args.q_blocks, 0)
    for delta in range(4):
        columns = (rows + offset - delta).clamp(0, args.k_blocks - 1)
        forced[:, :, rows, columns] = 1

    def torch_baseline() -> None:
        torch.topk(logits, budget, dim=-1, out=(values, indices))
        torch.ge(logits, values[..., -1:], out=torch_mask)
        torch_mask.bitwise_or_(forced)

    def flashinfer_kernel() -> None:
        adaptive_sparse_block_mask(
            logits,
            q_lens,
            kv_lens,
            kv_lens,
            block_size=args.block_size,
            alpha=1.0,
            out=kernel_mask,
        )

    torch_baseline()
    flashinfer_kernel()
    torch.cuda.synchronize()
    torch.testing.assert_close(kernel_mask, torch_mask)

    enable_cupti = not args.cuda_events
    kernel_times: list[float] = []
    baseline_times: list[float] = []
    timing_order: list[str] = []
    timing_batches = (
        (
            ("flashinfer", flashinfer_kernel, kernel_times),
            ("torch", torch_baseline, baseline_times),
        ),
        (
            ("torch", torch_baseline, baseline_times),
            ("flashinfer", flashinfer_kernel, kernel_times),
        ),
    )
    for batch in timing_batches:
        timing_order.append("->".join(name for name, _, _ in batch))
        for _, fn, samples in batch:
            samples.extend(_samples_ms(fn, enable_cupti=enable_cupti))

    kernel_ms = statistics.median(kernel_times)
    baseline_ms = statistics.median(baseline_times)
    print(f"gpu={torch.cuda.get_device_name(device)}")
    print(
        f"shape=B{args.batch_size},H{args.num_heads},Qb{args.q_blocks},"
        f"Kb{args.k_blocks},budget={budget}"
    )
    print(f"timing_order={','.join(timing_order)}")
    print(f"adaptive_cuda_ms={kernel_ms:.6f}")
    print(f"torch_topk_pipeline_ms={baseline_ms:.6f}")
    print(f"speedup={baseline_ms / kernel_ms:.3f}x")


if __name__ == "__main__":
    main()
