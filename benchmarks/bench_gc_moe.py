# SPDX-License-Identifier: Apache-2.0
"""Latency of the experimental GC MoE forward pass on one rank.

Sweeps tokens per rank over one staged workspace. Weights and the workspace
are built once, outside the timed region; each point re-stages the tokens.
``--cuda-graph`` times the replay path a serving stack actually runs.

    python benchmarks/bench_gc_moe.py --tokens 1,8,64,256
"""

import argparse
import os
import socket
import statistics

import torch
import torch.distributed as dist

import flashinfer
from flashinfer.testing.utils import bench_gpu_time


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def checkpoint_weights(backend, experts, hidden, intermediate, device, seed=0):
    generator = torch.Generator(device="cpu").manual_seed(seed)

    def payload(rows, k):
        return torch.randint(
            -128, 128, (experts, rows, k // 2), dtype=torch.int8, generator=generator
        ).to(device)

    def scales(rows, k):
        if backend == "sm90":
            return torch.ones((experts, rows, k // 32), device=device)
        if backend == "sm100":
            # SM100 reads MN-major packed scales: [E, N, K/128] with stride(-2) == 1.
            return torch.full(
                (experts, k // 128, rows), 0x7F7F7F7F, dtype=torch.int32, device=device
            ).transpose(-1, -2)
        return torch.full(
            (experts, rows, k // 128), 0x7F7F7F7F, dtype=torch.int32, device=device
        )

    return (
        (payload(2 * intermediate, hidden), scales(2 * intermediate, hidden)),
        (payload(hidden, intermediate), scales(hidden, intermediate)),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-experts", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--intermediate-hidden", type=int, default=1024)
    parser.add_argument("--num-topk", type=int, default=4)
    parser.add_argument(
        "--capacity", type=int, default=256, help="max tokens per rank staged per call"
    )
    parser.add_argument("--tokens", type=str, default="1,8,64,256")
    parser.add_argument(
        "--cuda-graph", action="store_true", help="time the captured replay path"
    )
    args = parser.parse_args()
    token_counts = [int(value) for value in args.tokens.split(",") if value.strip()]

    if not torch.cuda.is_available():
        raise SystemExit("requires a CUDA device")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{free_port()}",
        world_size=1,
        rank=0,
        device_id=device,
    )
    generator = torch.Generator(device="cpu").manual_seed(0)
    samples = (torch.randn(args.capacity, args.hidden, generator=generator) * 0.5).to(
        device
    )
    samples = samples.to(torch.float8_e4m3fn)
    routes = (
        torch.stack(
            [
                torch.randperm(args.num_experts, generator=generator)[: args.num_topk]
                for _ in range(args.capacity)
            ]
        )
        .to(torch.int64)
        .to(device)
    )
    route_weights = torch.rand(args.capacity, args.num_topk, generator=generator).to(
        device
    )

    with flashinfer.GCMoEWorkspace(
        dist.group.WORLD,
        args.num_experts,
        args.capacity,
        args.num_topk,
        args.hidden,
        args.intermediate_hidden,
    ) as workspace:
        checkpoint = checkpoint_weights(
            workspace.backend,
            args.num_experts,
            args.hidden,
            args.intermediate_hidden,
            device,
        )
        l1_weights, l2_weights = flashinfer.transform_gc_moe_weights(*checkpoint)
        y = torch.empty(args.capacity, args.hidden, dtype=torch.bfloat16, device=device)

        def stage(tokens):
            workspace.x[:tokens].copy_(samples[:tokens])
            if workspace.x_sf.dtype == torch.float32:
                workspace.x_sf[:tokens].fill_(1.0)
            else:
                workspace.x_sf[:tokens].fill_(0x7F7F7F7F)
            workspace.topk_idx[:tokens].copy_(routes[:tokens])
            workspace.topk_weights[:tokens].copy_(route_weights[:tokens])

        def run(tokens):
            return lambda: flashinfer.gc_moe(
                y[:tokens], l1_weights, l2_weights, workspace
            )

        print(
            f"backend={workspace.backend} experts={args.num_experts} "
            f"hidden={args.hidden} intermediate={args.intermediate_hidden} "
            f"topk={args.num_topk} graph={args.cuda_graph}"
        )
        print(f"{'tokens':>7} {'latency_us':>12} {'us_per_token':>13}")
        for tokens in token_counts:
            if tokens > args.capacity:
                print(f"{tokens:>7} {'skipped':>12} {'> capacity':>13}")
                continue
            stage(tokens)
            run(tokens)()
            torch.cuda.synchronize()
            latency_ms = bench_gpu_time(
                run(tokens), use_cuda_graph=args.cuda_graph, cold_l2_cache=False
            )
            latency_us = statistics.median(latency_ms) * 1e3
            print(f"{tokens:>7} {latency_us:>12.2f} {latency_us / tokens:>13.2f}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
