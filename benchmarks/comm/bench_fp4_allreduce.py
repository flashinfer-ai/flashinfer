"""Benchmark the NVFP4-transport all-reduce against NCCL on two or four ranks.

Launch with ``torchrun --nproc-per-node {2,4} benchmarks/comm/bench_fp4_allreduce.py``.

Two arms, interleaved in the order of ``--order`` (default ``fp4,nccl,nccl,fp4`` so
same-session drift is visible):

* ``fp4``: ``fp4_allreduce`` (``fp4_allreduce_nvfp4`` with ``--input-format nvfp4``)
  as a CUDA graph of ``--graph-calls`` back-to-back calls (default 10: the
  three-slot rotation wraps around);
* ``nccl``: ``torch.distributed.all_reduce`` of the same bfloat16 tensor, the
  same number of calls per graph (captured when the NCCL build supports graph
  capture, eager otherwise; the mode is recorded).

Every timed iteration replays one graph.  CUPTI records the kernel spans of the
iteration on every rank; the per-iteration time is the maximum over ranks of
``last kernel end - first kernel start`` and the row value is the median over
iterations divided by ``--graph-calls``.  Every leg is bracketed by a
correctness check: the ``fp4`` arm must match the float32 codec reference
bitwise, the ``nccl`` arm must be close to the exact float32 sum.  Published
numbers of another implementation can be passed as ``--published`` (JSON
``{"tp2/bf16/4096": microseconds_per_call, ...}``) and are printed next to the
measured rows; none are embedded here.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Callable

import torch
import torch.distributed as dist

from flashinfer.comm.fp4_allreduce import (
    Fp4AllReduceWorkspace,
    fp4_allreduce,
    fp4_allreduce_nvfp4,
)
from flashinfer.jit.fp4_allreduce import HIDDEN
from flashinfer.testing.utils import bench_gpu_time
from tests.comm import fp4_allreduce_reference as codec

_BOOL = {"false": False, "true": True}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--rows", default="1024,2048,4096,8192,16384", help="comma-separated row counts"
    )
    parser.add_argument("--input-format", choices=("bf16", "nvfp4"), default="bf16")
    parser.add_argument("--graph-calls", type=int, default=10)
    parser.add_argument("--dry-run-iters", type=int, default=20)
    parser.add_argument("--repeat-iters", type=int, default=200)
    parser.add_argument(
        "--order", default="fp4,nccl,nccl,fp4", help="comma-separated legs"
    )
    parser.add_argument("--launch-with-pdl", choices=tuple(_BOOL), default="true")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument(
        "--published",
        type=Path,
        default=None,
        help="JSON of published per-call microseconds",
    )
    parser.add_argument(
        "--output", type=Path, default=None, help="write the JSON report here (rank 0)"
    )
    return parser.parse_args()


def _gather(tensor: torch.Tensor, group) -> list[torch.Tensor]:
    peers = [torch.empty_like(tensor) for _ in range(dist.get_world_size(group))]
    dist.all_gather(peers, tensor.contiguous(), group=group)
    return peers


def _exact_sum(partials: list[torch.Tensor]) -> torch.Tensor:
    total = torch.zeros_like(partials[0], dtype=torch.float32)
    for partial in partials:
        total = codec.f32_add_rn(total, partial.to(torch.float32))
    return total


def _time(
    calls: Callable[[], None],
    *,
    graph: bool,
    dry_run_iters: int,
    repeat_iters: int,
    require_graph: bool,
):
    """Per-iteration spans (ms), already reduced to the maximum over ranks, and the execution mode."""
    runner = calls
    mode = "eager"
    if graph:
        try:
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                calls()
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
            captured = torch.cuda.CUDAGraph()
            with torch.cuda.graph(captured, stream=side):
                calls()
            runner = captured.replay
            mode = "graph"
        except Exception as exc:  # noqa: BLE001 - NCCL builds without graph capture fall back to eager
            if require_graph:
                raise
            torch.cuda.synchronize()
            mode = f"eager (capture failed: {type(exc).__name__})"
    samples = bench_gpu_time(
        runner,
        enable_cupti=True,
        use_cuda_graph=False,
        cold_l2_cache=False,
        dry_run_iters=dry_run_iters,
        repeat_iters=repeat_iters,
        aggregate_op=max,
    )
    return [float(sample) for sample in samples], mode


def _summary(samples_ms: list[float], graph_calls: int) -> dict:
    per_call_us = sorted(1000.0 * sample / graph_calls for sample in samples_ms)
    n = len(per_call_us)
    return {
        "median_us_per_call": statistics.median(per_call_us),
        "p10_us_per_call": per_call_us[int(0.10 * (n - 1))],
        "p90_us_per_call": per_call_us[int(0.90 * (n - 1))],
        "iterations": n,
    }


def main() -> None:
    args = _parse_args()
    dist.init_process_group(backend="nccl")
    group = dist.group.WORLD
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    device = torch.device("cuda", rank % torch.cuda.device_count())
    torch.cuda.set_device(device)
    launch_with_pdl = _BOOL[args.launch_with_pdl]
    legs = [leg.strip() for leg in args.order.split(",") if leg.strip()]
    if any(leg not in ("fp4", "nccl") for leg in legs):
        raise SystemExit(f"--order may only name fp4 and nccl legs, got {legs}")
    published = (
        json.loads(args.published.read_text()) if args.published is not None else {}
    )
    started = time.monotonic()
    workspace = Fp4AllReduceWorkspace(group, device=device)
    rows_list = [int(value) for value in args.rows.split(",") if value.strip()]
    report_rows = []
    try:
        for rows in rows_list:
            x = codec.generate_rank_input(rows, rank, args.seed + 17 * rows, device)
            partials = _gather(x, group)
            exact = _exact_sum(partials)
            out = torch.empty((rows, HIDDEN), dtype=torch.bfloat16, device=device)
            nccl_buffer = torch.empty_like(x)
            if args.input_format == "bf16":
                expected = codec.reference_allreduce(partials)

                def fp4_call(x=x, out=out):
                    fp4_allreduce(
                        x, workspace, out=out, launch_with_pdl=launch_with_pdl
                    )

            else:
                scale = codec.tensor_global_scale(x)
                records = codec.encode_nvfp4_records(x, scale)
                scales = [
                    float(value.item())
                    for value in _gather(
                        torch.tensor([scale], dtype=torch.float32, device=device), group
                    )
                ]
                scale_tensor = torch.tensor(scales, dtype=torch.float32, device=device)
                expected = codec.reference_allreduce_nvfp4(
                    _gather(records, group), scales
                )

                def fp4_call(records=records, scale_tensor=scale_tensor, out=out):
                    fp4_allreduce_nvfp4(
                        records,
                        scale_tensor,
                        workspace,
                        out=out,
                        launch_with_pdl=launch_with_pdl,
                    )

            def nccl_call(x=x, nccl_buffer=nccl_buffer):
                nccl_buffer.copy_(x)
                dist.all_reduce(nccl_buffer, group=group)

            def fp4_calls():
                for _ in range(args.graph_calls):
                    fp4_call()

            def nccl_calls():
                for _ in range(args.graph_calls):
                    nccl_call()

            def check(leg: str, phase: str) -> None:
                torch.cuda.synchronize(device)
                if leg == "fp4":
                    out.zero_()
                    fp4_call()
                    torch.cuda.synchronize(device)
                    if not torch.equal(out, expected):
                        differing = int((out != expected).sum().item())
                        raise AssertionError(
                            f"rows={rows} {leg}/{phase}: {differing} elements differ from the codec reference"
                        )
                else:
                    nccl_call()
                    torch.cuda.synchronize(device)
                    if not torch.allclose(
                        nccl_buffer.float(), exact, rtol=1 / 64, atol=0.5
                    ):
                        raise AssertionError(
                            f"rows={rows} {leg}/{phase}: NCCL result is not close to the exact sum"
                        )

            row = {
                "world_size": world_size,
                "rows": rows,
                "input_format": args.input_format,
                "launch_with_pdl": launch_with_pdl,
                "kernel": workspace.kernel_name(args.input_format, rows),
                "variant": workspace.variant(rows),
                "legs": [],
            }
            for leg in legs:
                dist.barrier(group=group)
                check(leg, "pre")
                dist.barrier(group=group)
                samples, mode = _time(
                    fp4_calls if leg == "fp4" else nccl_calls,
                    graph=True,
                    dry_run_iters=args.dry_run_iters,
                    repeat_iters=args.repeat_iters,
                    require_graph=leg == "fp4",
                )
                check(leg, "post")
                row["legs"].append(
                    {"arm": leg, "mode": mode, **_summary(samples, args.graph_calls)}
                )
            medians = {}
            for arm in ("fp4", "nccl"):
                values = [
                    leg["median_us_per_call"]
                    for leg in row["legs"]
                    if leg["arm"] == arm
                ]
                if values:
                    medians[arm] = statistics.fmean(values)
            row["median_us_per_call"] = medians
            if "fp4" in medians and "nccl" in medians:
                row["nccl_over_fp4"] = medians["nccl"] / medians["fp4"]
            key = f"tp{world_size}/{args.input_format}/{rows}"
            if key in published:
                row["published_us_per_call"] = float(published[key])
                row["published_over_fp4"] = float(published[key]) / medians["fp4"]
            report_rows.append(row)
            if rank == 0:
                print(json.dumps(row, sort_keys=True), flush=True)
    finally:
        dist.barrier(group=group)
        workspace.destroy()
    if rank == 0:
        report = {
            "world_size": world_size,
            "gpu": torch.cuda.get_device_name(device),
            "arch": workspace.arch,
            "grid": workspace.grid,
            "torch": torch.__version__,
            "timing": {
                "method": "bench_gpu_time(enable_cupti=True) over graph replays",
                "graph_calls": args.graph_calls,
                "dry_run_iters": args.dry_run_iters,
                "repeat_iters": args.repeat_iters,
                "order": legs,
                "rollup": "per-iteration span, maximum over ranks, median over iterations, divided by graph_calls",
            },
            "rows": report_rows,
            "physical_runtime_seconds": time.monotonic() - started,
        }
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
