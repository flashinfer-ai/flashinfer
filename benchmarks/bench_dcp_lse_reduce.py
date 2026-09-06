"""Benchmark decode-CP A2A + LSE reduce under torchrun.

Example:
  torchrun --standalone --nproc-per-node=4 \
    benchmarks/bench_dcp_lse_reduce.py --label fused
"""

import argparse
import json
import os
import statistics
from typing import Callable

import torch
import torch.distributed as dist

from flashinfer.comm import (
    decode_cp_a2a_lse_reduce,
    decode_cp_a2a_lse_reduce_create_workspace,
)


def _measure(
    fn: Callable[[], None],
    *,
    warmup: int,
    samples: int,
    launches_per_sample: int,
) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(samples):
        start.record()
        for _ in range(launches_per_sample):
            fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000 / launches_per_sample)
    return times


def _max_rank_median(times_us: list[float], world_size: int) -> float:
    local = torch.tensor(times_us, dtype=torch.float64, device="cuda")
    gathered = [torch.empty_like(local) for _ in range(world_size)]
    dist.all_gather(gathered, local)
    per_rank_medians = [statistics.median(tensor.cpu().tolist()) for tensor in gathered]
    return max(per_rank_medians)


def _reference_a2a_lse_reduce(
    partial_o: torch.Tensor,
    partial_lse: torch.Tensor,
    *,
    cp_rank: int,
    cp_size: int,
) -> torch.Tensor:
    """All-gather reference for the unfused A2A + LSE-reduction baseline."""
    all_o = [torch.empty_like(partial_o) for _ in range(cp_size)]
    all_lse = [torch.empty_like(partial_lse) for _ in range(cp_size)]
    dist.all_gather(all_o, partial_o)
    dist.all_gather(all_lse, partial_lse)
    recv_o = torch.stack([tensor[..., cp_rank, :] for tensor in all_o], dim=-2)
    recv_lse = torch.stack([tensor[..., cp_rank] for tensor in all_lse], dim=-1)
    recv_lse = torch.where(
        torch.isnan(recv_lse) | torch.isposinf(recv_lse),
        torch.full_like(recv_lse, float("-inf")),
        recv_lse,
    )
    lse_max = recv_lse.max(dim=-1, keepdim=True).values
    lse_max = torch.where(torch.isneginf(lse_max), torch.zeros_like(lse_max), lse_max)
    weights = torch.exp2(recv_lse - lse_max)
    denom = weights.sum(dim=-1, keepdim=True)
    output = (recv_o.float() * weights.unsqueeze(-1)).sum(dim=-2) / denom.clamp_min(1e-20)
    output = torch.where(denom == 0, torch.zeros_like(output), output)
    return output.to(partial_o.dtype)


def _benchmark_case(
    *,
    label: str,
    batch: int,
    local_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    samples: int,
    launches_per_sample: int,
) -> None:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.manual_seed(2026 + rank)

    partial_o = torch.randn(
        batch,
        local_heads,
        world_size,
        head_dim,
        dtype=dtype,
        device="cuda",
    )
    partial_lse = torch.randn(
        batch,
        local_heads,
        world_size,
        dtype=torch.float32,
        device="cuda",
    )
    workspace = decode_cp_a2a_lse_reduce_create_workspace(
        max_tokens=batch,
        local_heads=local_heads,
        cp_size=world_size,
        head_dim=head_dim,
        dtype=dtype,
        group=dist.group.WORLD,
    )

    def eager_call() -> None:
        decode_cp_a2a_lse_reduce(
            partial_o,
            partial_lse,
            workspace,
            cp_rank=rank,
            cp_size=world_size,
        )

    def baseline_call() -> None:
        _reference_a2a_lse_reduce(
            partial_o,
            partial_lse,
            cp_rank=rank,
            cp_size=world_size,
        )

    dist.barrier()
    eager_us = _measure(
        eager_call,
        warmup=10,
        samples=samples,
        launches_per_sample=launches_per_sample,
    )

    dist.barrier()
    baseline_us = _measure(
        baseline_call,
        warmup=10,
        samples=samples,
        launches_per_sample=launches_per_sample,
    )

    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _graph_output = decode_cp_a2a_lse_reduce(
            partial_o,
            partial_lse,
            workspace,
            cp_rank=rank,
            cp_size=world_size,
        )
    graph_us = _measure(
        graph.replay,
        warmup=10,
        samples=samples,
        launches_per_sample=launches_per_sample,
    )

    result = {
        "label": label,
        "world_size": world_size,
        "batch": batch,
        "local_heads": local_heads,
        "head_dim": head_dim,
        "dtype": str(dtype).removeprefix("torch."),
        "eager_max_rank_median_us": _max_rank_median(eager_us, world_size),
        "baseline_max_rank_median_us": _max_rank_median(baseline_us, world_size),
        "graph_max_rank_median_us": _max_rank_median(graph_us, world_size),
    }
    if rank == 0:
        print("DCP_LSE_BENCH " + json.dumps(result, sort_keys=True), flush=True)
    dist.barrier()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--launches-per-sample", type=int, default=50)
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    # Eagerly create the NCCL communicator before the symmetric-memory
    # rendezvous. This registers the group's host communicator for this
    # concrete CUDA device on released PyTorch builds.
    dist.init_process_group("nccl", device_id=device)
    warmup = torch.zeros(1, device=device)
    dist.all_reduce(warmup)
    try:
        for batch, local_heads, head_dim in [
            (1, 2, 64),
            (1, 8, 128),
            (4, 8, 128),
            (16, 8, 128),
        ]:
            _benchmark_case(
                label=args.label,
                batch=batch,
                local_heads=local_heads,
                head_dim=head_dim,
                dtype=torch.bfloat16,
                samples=args.samples,
                launches_per_sample=args.launches_per_sample,
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
