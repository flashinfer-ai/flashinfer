#!/usr/bin/env python3
"""Benchmark TP2 b12x plain all-reduce with graph qualification.

The benchmark records the per-sample slowest-rank latency because that is the
latency observed by a distributed caller. Inputs are reset before every replay,
and b12x must produce the exact two-rank sum before timings are emitted.
Run it with ``torchrun --standalone --nproc-per-node=2``.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import subprocess
import sys
from pathlib import Path

import torch
import torch.distributed as dist


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument(
        "--rows",
        default="1,2,4,6,8,16,24,32,48,64,96,128",
        help="Comma-separated row counts.",
    )
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16"),
        default="bfloat16",
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--samples", type=int, default=60)
    parser.add_argument("--base-revision", required=True)
    parser.add_argument(
        "--topology-description",
        required=True,
        help=(
            "Semantic description of the physical path between the selected "
            "GPUs, for example 'separate CPU root ports without a PCIe switch'"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _run_text(command: list[str], *, cwd: Path | None = None) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return completed.stdout.strip()


def _source_identity() -> dict[str, object]:
    root = Path(__file__).resolve().parents[3]
    return {
        "worktree": str(root),
        "revision": _run_text(["git", "rev-parse", "HEAD"], cwd=root),
        "tree": _run_text(["git", "rev-parse", "HEAD^{tree}"], cwd=root),
        "dirty_paths": _run_text(["git", "status", "--short"], cwd=root).splitlines(),
    }


def _topology() -> str:
    return _run_text(["nvidia-smi", "topo", "-m"])


def _nvidia_smi_inventory() -> str:
    return _run_text(
        [
            "nvidia-smi",
            (
                "--query-gpu=index,uuid,pci.bus_id,name,driver_version,"
                "persistence_mode,compute_mode,clocks.sm,clocks.max.sm,power.limit"
            ),
            "--format=csv,noheader",
        ]
    )


def _group_max_samples(samples_us: list[float], device: torch.device) -> list[float]:
    values = torch.tensor(samples_us, dtype=torch.float64, device=device)
    dist.all_reduce(values, op=dist.ReduceOp.MAX)
    return [float(value) for value in values.cpu().tolist()]


def _measure_graph(graph, reset, *, warmup, samples, device):
    from b12x.preparation import PreparedCall
    from b12x.testing.benchmark import measure_calls

    call = PreparedCall(
        run=graph._b12x_benchmark_call, reset=reset, produce=lambda: None
    )
    dist.barrier()
    result = measure_calls(
        {"allreduce": call},
        warmup=warmup,
        samples=samples,
        device=device,
        collective=True,
    )
    return _group_max_samples(result.raw_samples("allreduce"), device)


def _distribution(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    return {
        "minimum_us": ordered[0],
        "median_us": ordered[(len(ordered) - 1) // 2],
        "p95_us": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
        "maximum_us": ordered[-1],
    }


def _correct(output: torch.Tensor, expected: float, device: torch.device) -> bool:
    local = torch.tensor(
        int(bool(torch.all(output == expected).item())),
        dtype=torch.int32,
        device=device,
    )
    dist.all_reduce(local, op=dist.ReduceOp.MIN)
    return bool(local.item())


def _benchmark_b12x(
    pool: object,
    rows: int,
    hidden_size: int,
    dtype: torch.dtype,
    rank: int,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, object]:
    from b12x.comm.pcie import plan, query_from_runtime
    from b12x.comm.pcie._oneshot_preparation import _prepare_plain_call
    from b12x.preparation import CollectiveRequirement, PreparationSession

    value = float(rank + 1)
    inp = torch.full((rows, hidden_size), value, dtype=dtype, device=device)
    out = torch.empty_like(inp)
    channel = pool.for_stream()
    query = query_from_runtime(
        channel, surface="OneshotAllReduce.all_reduce", call={"inp": inp}
    )
    declaration = plan(query, runtime=channel)
    collective = CollectiveRequirement(
        key="plain_allreduce", ranks=tuple(range(dist.get_world_size()))
    )
    request = declaration.request(
        name="plain_allreduce",
        collective=collective,
        prepare_call=lambda state: _prepare_plain_call(state, inp=inp, out=out),
    )
    with (
        PreparationSession(device=device, autotune=False) as session,
        session.prepare(
            (request,),
            coordinator=lambda progress: (
                collective.key if progress.ready_collectives else None
            ),
        ),
    ):

        def invoke():
            channel.all_reduce(inp, out=out, plan=declaration)

        invoke()
        torch.cuda.synchronize(device)
        graph = torch.cuda.CUDAGraph()
        try:
            with session.capture(), torch.cuda.graph(graph):
                invoke()
            graph._b12x_benchmark_call = invoke
            out.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize(device)
            if not _correct(out, 3.0, device):
                raise RuntimeError("b12x all-reduce graph correctness failed")
            samples = _measure_graph(
                graph,
                lambda: inp.fill_(value),
                warmup=args.warmup,
                samples=args.samples,
                device=device,
            )
            correct = _correct(out, 3.0, device)
        finally:
            graph.reset()
    return {
        "backend": "b12x",
        "correct": correct,
        "launch_plan": {
            name: query.call[name] for name in ("transport", "threads", "blocks")
        },
        "samples_slowest_rank_us": samples,
        **_distribution(samples),
    }


def main() -> None:
    args = _parse_args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 2:
        raise ValueError(f"TP2 benchmark requires world_size=2, got {world_size}")

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    dtype = getattr(torch, args.dtype)
    row_counts = [int(value) for value in args.rows.split(",")]

    from b12x.comm.pcie import OneshotAllReducePool

    maximum_bytes = max(row_counts) * args.hidden_size * dtype.itemsize
    pool = OneshotAllReducePool(
        rank=rank,
        world_size=world_size,
        device=device,
        exchange_group=dist.group.WORLD,
        eager_buffer_bytes=maximum_bytes,
        max_size=maximum_bytes,
        rank_data_bytes=maximum_bytes,
        single_channel=True,
    )

    rank_device = {
        "rank": rank,
        "logical_device": local_rank,
        "name": torch.cuda.get_device_name(device),
        "capability": list(torch.cuda.get_device_capability(device)),
    }
    devices: list[dict[str, object] | None] = [None] * world_size
    dist.all_gather_object(devices, rank_device)

    results = []
    try:
        for rows in row_counts:
            b12x = _benchmark_b12x(
                pool,
                rows,
                args.hidden_size,
                dtype,
                rank,
                args,
                device,
            )
            if not b12x["correct"]:
                raise RuntimeError(f"all-reduce oracle failed for rows={rows}")
            results.append(
                {
                    "rows": rows,
                    "hidden_size": args.hidden_size,
                    "bytes": rows * args.hidden_size * dtype.itemsize,
                    "b12x": b12x,
                }
            )
    finally:
        pool.close()

    if rank == 0:
        report = {
            "contract": "TP2 stream-gated prepared plain all-reduce with graph qualification",
            "metric_direction": "latency_us; lower is better",
            "base_revision": args.base_revision,
            "source": _source_identity(),
            "host": socket.gethostname(),
            "topology_description": args.topology_description,
            "worker_command": [sys.executable, *sys.argv],
            "distributed_environment": {
                name: os.environ.get(name)
                for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE")
            },
            "python_version": platform.python_version(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "collective_environment": {
                name: os.environ.get(name)
                for name in (
                    "B12X_PCIE_ONESHOT_BLOCK_LIMIT",
                    "B12X_PCIE_ONESHOT_THREADS",
                    "B12X_PCIE_TP2_PLAIN_REMOTE_PUSH",
                    "B12X_PCIE_TP2_REMOTE_PUSH",
                    "NCCL_ALLOC_P2P_NET_LL_BUFFERS",
                    "NCCL_BUFFSIZE",
                    "NCCL_DMABUF_ENABLE",
                    "NCCL_IB_DISABLE",
                    "NCCL_IGNORE_CPU_AFFINITY",
                    "NCCL_MIN_NCHANNELS",
                    "NCCL_P2P_LEVEL",
                    "NCCL_PROTO",
                )
            },
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "nccl_version": torch.cuda.nccl.version(),
            "dtype": args.dtype,
            "warmup_replays_per_group": args.warmup,
            "measurement_order": ["b12x"],
            "timed_replays": args.samples,
            "devices": devices,
            "nvidia_smi_inventory": _nvidia_smi_inventory(),
            "nvidia_smi_topology": _topology(),
            "results": results,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote {args.output}")
        for result in results:
            print(f"M={result['rows']:>3}: B12X={result['b12x']['median_us']:.2f} us ")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    from b12x.testing.memory import absorb_small_page_fragments

    absorb_small_page_fragments()
    main()
