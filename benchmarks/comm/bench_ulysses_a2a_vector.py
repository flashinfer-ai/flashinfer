"""Benchmark aligned and odd-width Ulysses all-to-all rows on one NVLink node.

Run with ``torchrun --standalone --nproc_per_node=8``. Set ``--include-dir``
to a complete alternate include tree to compare two header revisions.
"""

import argparse
import datetime
import json
import os
from pathlib import Path
import statistics

import torch
import torch.distributed as dist

from flashinfer.comm import UlyssesCommunicator
from flashinfer.jit import env as jit_env


def reference_scatter(x, world, rank):
    gathered = [torch.empty_like(x) for _ in range(world)]
    dist.all_gather(gathered, x)
    local_heads = x.shape[2] // world
    return torch.cat(
        [
            part[:, :, rank * local_heads : (rank + 1) * local_heads, :]
            for part in gathered
        ],
        dim=1,
    ).contiguous()


def measure(call, warmup, trials, iters):
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    samples = []
    for _ in range(trials):
        dist.barrier()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            call()
        end.record()
        end.synchronize()
        elapsed = torch.tensor(start.elapsed_time(end) / iters, device="cuda")
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        samples.append(elapsed.item())
    return {
        "ms": statistics.median(samples),
        "trial_ms": samples,
        "sd_pct": statistics.pstdev(samples) / statistics.mean(samples) * 100,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-dir", type=Path)
    parser.add_argument("--label", default="current")
    parser.add_argument("--lengths", default="4096,16384,65536")
    parser.add_argument("--head-dims", default="34,66")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()
    if args.include_dir:
        jit_env.FLASHINFER_INCLUDE_DIR = args.include_dir.resolve()
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=10))
    world = dist.get_world_size()
    if world != 8:
        raise ValueError("This benchmark expects one 8-GPU NVLink node")
    rows = []
    for length in map(int, args.lengths.split(",")):
        for head_dim in map(int, args.head_dims.split(",")):
            shape = (1, length, 48, head_dim)
            torch.manual_seed(41 + rank)
            x = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
            expected = reference_scatter(x, world, rank)
            for backend in ("nccl", "nvlink"):
                communicator = UlyssesCommunicator(
                    group=dist.group.WORLD,
                    max_elems=x.numel(),
                    dtype=x.dtype,
                    device=rank,
                    backend=backend,
                )
                try:

                    def call():
                        return communicator.scatter_heads(x)

                    actual = call()
                    torch.cuda.synchronize()
                    correct = torch.tensor(
                        int(torch.equal(actual, expected)), device="cuda"
                    )
                    dist.all_reduce(correct, op=dist.ReduceOp.MIN)
                    if not correct.item():
                        raise AssertionError(f"{backend} mismatch for {shape}")
                    stats = measure(call, args.warmup, args.trials, args.iters)
                    row = {
                        "label": args.label,
                        "shape": shape,
                        "world_size": world,
                        "backend": backend,
                        "kernel": communicator.nvlink_kernel,
                        "correct": True,
                        **stats,
                    }
                    rows.append(row)
                    if rank == 0:
                        args.output.parent.mkdir(parents=True, exist_ok=True)
                        args.output.write_text(
                            json.dumps(
                                {
                                    "environment": {
                                        "gpu": torch.cuda.get_device_name(),
                                        "torch": torch.__version__,
                                        "cuda": torch.version.cuda,
                                        "world_size": world,
                                        "args": {
                                            key: str(value)
                                            for key, value in vars(args).items()
                                        },
                                    },
                                    "results": rows,
                                },
                                indent=2,
                            )
                        )
                        print(json.dumps(row), flush=True)
                finally:
                    communicator.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
