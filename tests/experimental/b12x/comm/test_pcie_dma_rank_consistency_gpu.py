#!/usr/bin/env python3
"""Four-GPU gate for rank-identical compressed PCIe-DMA all-reduce outputs.

Run inside the target image after installing the overlay:

    python test_pcie_dma_rank_consistency_gpu.py --mode ag
    python test_pcie_dma_rank_consistency_gpu.py --mode ring
    python test_pcie_dma_rank_consistency_gpu.py --mode a2a
    python test_pcie_dma_rank_consistency_gpu.py --mode i8
    python test_pcie_dma_rank_consistency_gpu.py --mode i8_ring
    python test_pcie_dma_rank_consistency_gpu.py --mode i8_a2a
    python test_pcie_dma_rank_consistency_gpu.py --mode mx
    python test_pcie_dma_rank_consistency_gpu.py --mode mx_ring
    python test_pcie_dma_rank_consistency_gpu.py --mode mx_a2a
"""

from __future__ import annotations

import argparse
import os
import socket

import torch
import torch.distributed as dist

from b12x.comm.pcie._dma_preparation import (
    plan as dma_plan,
    prepared_call,
    query_from_runtime,
)
from b12x.preparation import PreparationSession


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def make_input(rank: int, device: torch.device, iteration: int = 0) -> torch.Tensor:
    rows = 512
    hidden = 6144
    values = torch.arange(rows * hidden, device=device, dtype=torch.float32)
    values = values.reshape(rows, hidden)
    return (torch.sin(values * 0.001 + rank * 0.31 + iteration * 0.17) * 0.5).to(
        torch.bfloat16
    )


def assert_output(
    output: torch.Tensor,
    reference: torch.Tensor,
    *,
    rank: int,
    world: int,
    mode: str,
    label: str,
) -> None:
    rank0_output = output.clone() if rank == 0 else torch.empty_like(output)
    dist.broadcast(rank0_output, src=0)
    if not torch.equal(output, rank0_output):
        mismatch = (output.float() - rank0_output.float()).abs()
        raise AssertionError(
            f"mode={mode} {label} rank={rank} output differs from rank0: "
            f"count={int(torch.count_nonzero(mismatch).item())} "
            f"max_abs={float(mismatch.max().item())}"
        )
    torch.testing.assert_close(
        output.float(), reference, rtol=1.5e-1, atol=6e-2 * world
    )


def worker(rank: int, world: int, port: int, mode: str) -> None:
    os.environ["B12X_PCIE_DMA_FP8"] = mode
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world,
    )
    from b12x.comm.pcie.pcie_dma import PCIeDmaAllReduce

    ring = PCIeDmaAllReduce(
        exchange_group=dist.group.WORLD, device=device, max_bytes=512 * 6144 * 2, fp8=mode
    )
    session = PreparationSession(device=device, autotune=False)
    result = None
    try:
        query = query_from_runtime(ring, surface="DmaAllReduce.all_reduce", call={})
        prime_inp = torch.zeros(world * 128, dtype=torch.bfloat16, device=device)
        prime_out = torch.empty_like(prime_inp)
        declaration = dma_plan(query, runtime=ring)
        request = declaration.request(
            name="dma",
            prepare_call=lambda state: prepared_call(
                state, inp=prime_inp, out=prime_out
            ),
        )
        result = session.prepare((request,))
        plan = declaration
        inp = make_input(rank, device)
        reference = inp.float()
        dist.all_reduce(reference)
        output = torch.empty_like(inp)
        ring.all_reduce(inp, plan=plan, out=output)
        torch.cuda.synchronize(device)
        assert_output(output, reference, rank=rank, world=world, mode=mode, label="eager")

        graph_input = make_input(rank, device, 1)
        graph_output = torch.empty_like(graph_input)
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            ring.all_reduce(graph_input, plan=plan, out=graph_output)
        for iteration in range(2, 5):
            graph_input.copy_(make_input(rank, device, iteration))
            graph_reference = graph_input.float()
            dist.all_reduce(graph_reference)
            graph.replay()
            torch.cuda.synchronize(device)
            assert_output(
                graph_output, graph_reference, rank=rank, world=world, mode=mode,
                label=f"graph-replay-{iteration}",
            )
        dist.barrier()
    finally:
        if result is not None:
            result.close()
        session.close()
        ring.close()
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        required=True,
        choices=(
            "ag",
            "ring",
            "a2a",
            "i8",
            "i8_ring",
            "i8_a2a",
            "mx",
            "mx_ring",
            "mx_a2a",
        ),
    )
    args = parser.parse_args()
    world = 4
    if not torch.cuda.is_available() or torch.cuda.device_count() < world:
        raise RuntimeError("this gate requires four visible CUDA GPUs")
    mp.spawn(worker, args=(world, free_port(), args.mode), nprocs=world, join=True)


if __name__ == "__main__":
    main()
