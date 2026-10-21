"""Calibrate plain B12X all-reduce routing against NCCL CUDA graphs.

Run this module under ``torchrun`` before starting the model server. It emits a
versioned JSON policy for ``B12X_PCIE_PLAIN_ALLREDUCE_POLICY``. Every measured
row receives an explicit B12X or NCCL decision; unmeasured tensor contracts
retain the runtime's static compatibility policy.
"""

from __future__ import annotations

import argparse
import json
import os
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, ContextManager

import torch
import torch.distributed as dist

from .pcie_oneshot import PCIeOneshotAllReducePool, PlainAllReduceRoutePolicy


@dataclass(frozen=True)
class CalibrationPoint:
    """Slowest-rank CUDA-graph latency for one all-reduce input shape."""

    rows: int
    custom_median_us: float
    custom_p95_us: float
    nccl_median_us: float
    nccl_p95_us: float
    correct: bool = True
    custom_trials_us: tuple[float, ...] = ()
    nccl_trials_us: tuple[float, ...] = ()


def select_plain_allreduce_policy(
    points: list[CalibrationPoint],
    *,
    world_size: int,
    hidden_size: int,
    dtype: torch.dtype,
    min_absolute_margin_us: float,
    min_relative_margin: float,
) -> PlainAllReduceRoutePolicy:
    """Select B12X only when its median win exceeds the noise guard."""

    if min_absolute_margin_us < 0:
        raise ValueError("min_absolute_margin_us must be non-negative")
    if min_relative_margin < 0:
        raise ValueError("min_relative_margin must be non-negative")
    measured_rows = frozenset(point.rows for point in points)
    if len(measured_rows) != len(points):
        raise ValueError("calibration rows must be unique")
    if any(not point.correct for point in points):
        raise ValueError("calibration cannot select a numerically invalid result")

    def custom_wins(custom_us: float, nccl_us: float) -> bool:
        return (
            custom_us
            + max(
                min_absolute_margin_us,
                nccl_us * min_relative_margin,
            )
            <= nccl_us
        )

    custom_rows = set()
    for point in points:
        aggregate_win = custom_wins(point.custom_median_us, point.nccl_median_us)
        trial_win = True
        if point.custom_trials_us or point.nccl_trials_us:
            if len(point.custom_trials_us) != len(point.nccl_trials_us):
                raise ValueError("custom and NCCL trial counts must match")
            wins = sum(
                custom_wins(custom_us, nccl_us)
                for custom_us, nccl_us in zip(
                    point.custom_trials_us,
                    point.nccl_trials_us,
                    strict=True,
                )
            )
            trial_win = wins > len(point.custom_trials_us) // 2
        if aggregate_win and trial_win:
            custom_rows.add(point.rows)
    return PlainAllReduceRoutePolicy(
        world_size=world_size,
        hidden_size=hidden_size,
        dtype=dtype,
        measured_rows=measured_rows,
        custom_rows=frozenset(custom_rows),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden-size", type=int, required=True)
    parser.add_argument(
        "--rows",
        default="1,2,4,8,16,24,32",
        help="Comma-separated row counts that the serving scheduler can emit.",
    )
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--repeat", type=int, default=41)
    parser.add_argument("--ops-per-sample", type=int, default=8)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument(
        "--stabilization-replays",
        type=int,
        default=2048,
        help=(
            "Balanced B12X+NCCL graph replays on the largest shape before "
            "measurement. This raises GPU and PCIe links out of idle states."
        ),
    )
    parser.add_argument("--min-absolute-margin-us", type=float, default=0.25)
    parser.add_argument("--min-relative-margin", type=float, default=0.01)
    parser.add_argument("--json", type=Path)
    return parser.parse_args()


def _slowest_rank_distribution(
    samples_us: list[float],
    device: torch.device,
) -> tuple[float, float]:
    values = torch.tensor(samples_us, dtype=torch.float64, device=device)
    dist.all_reduce(values, op=dist.ReduceOp.MAX)
    values = values.sort().values
    median = float(values[(len(values) - 1) // 2].item())
    p95_index = min(len(values) - 1, int(0.95 * len(values)))
    return median, float(values[p95_index].item())


def _all_ranks(value: bool, device: torch.device) -> bool:
    flag = torch.tensor(int(value), dtype=torch.int32, device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def _measure_graph(
    op: Callable[[], None],
    *,
    capture_context: Callable[[torch.cuda.Stream], ContextManager[object]],
    warmup: int,
    repeat: int,
    ops_per_sample: int,
    device: torch.device,
) -> tuple[float, float]:
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    graph = torch.cuda.CUDAGraph()
    with capture_context(stream), torch.cuda.graph(graph, stream=stream):
        for _ in range(ops_per_sample):
            op()
    torch.cuda.current_stream(device).wait_stream(stream)
    torch.cuda.synchronize(device)

    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize(device)
    dist.barrier()

    samples_us: list[float] = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples_us.append(start.elapsed_time(end) * 1e3 / ops_per_sample)
    result = _slowest_rank_distribution(samples_us, device)
    graph.reset()
    torch.cuda.synchronize(device)
    return result


def _measure_graph_trials(
    op: Callable[[], None],
    *,
    capture_context: Callable[[torch.cuda.Stream], ContextManager[object]],
    warmup: int,
    repeat: int,
    ops_per_sample: int,
    trials: int,
    device: torch.device,
) -> tuple[float, float, tuple[float, ...]]:
    runs = tuple(
        _measure_graph(
            op,
            capture_context=capture_context,
            warmup=warmup,
            repeat=repeat,
            ops_per_sample=ops_per_sample,
            device=device,
        )
        for _ in range(trials)
    )
    medians = tuple(run[0] for run in runs)
    p95s = tuple(run[1] for run in runs)
    return (
        sorted(medians)[len(medians) // 2],
        sorted(p95s)[len(p95s) // 2],
        medians,
    )


def _stabilize_transports(
    pool: PCIeOneshotAllReducePool,
    custom_input: torch.Tensor,
    custom_output: torch.Tensor,
    nccl_input: torch.Tensor,
    *,
    replays: int,
    ops_per_replay: int,
    device: torch.device,
) -> None:
    """Raise both transports to an active link and clock state symmetrically."""

    # CUDA graph capture must not trigger JIT compilation or communicator
    # initialization. Prime both transports once, then replay an equal number
    # of B12X and NCCL operations from one graph.
    pool.prepare_graph_all_reduce(custom_input)
    pool.all_reduce(custom_input, out=custom_output)
    dist.all_reduce(nccl_input)
    nccl_input.zero_()
    torch.cuda.synchronize(device)
    dist.barrier()

    def combined_op() -> None:
        pool.all_reduce(custom_input, out=custom_output)
        dist.all_reduce(nccl_input)

    _measure_graph(
        combined_op,
        capture_context=lambda stream: pool.capture(stream),
        warmup=replays,
        repeat=1,
        ops_per_sample=ops_per_replay,
        device=device,
    )


def _measure_row(
    pool: PCIeOneshotAllReducePool,
    *,
    rows: int,
    hidden_size: int,
    dtype: torch.dtype,
    rank: int,
    world_size: int,
    warmup: int,
    repeat: int,
    ops_per_sample: int,
    trials: int,
    device: torch.device,
) -> CalibrationPoint:
    expected = float(world_size * (world_size + 1) // 2)
    custom_input = torch.full((rows, hidden_size), rank + 1, dtype=dtype, device=device)
    custom_output = torch.empty_like(custom_input)

    pool.all_reduce(custom_input, out=custom_output)
    torch.cuda.synchronize(device)
    custom_correct = _all_ranks(
        bool(torch.all(custom_output == expected).item()), device
    )

    def custom_op() -> None:
        pool.all_reduce(custom_input, out=custom_output)

    pool.prepare_graph_all_reduce(custom_input)
    custom_median, custom_p95, custom_trials = _measure_graph_trials(
        custom_op,
        capture_context=lambda stream: pool.capture(stream),
        warmup=warmup,
        repeat=repeat,
        ops_per_sample=ops_per_sample,
        trials=trials,
        device=device,
    )

    nccl_input = torch.full((rows, hidden_size), rank + 1, dtype=dtype, device=device)
    dist.all_reduce(nccl_input)
    torch.cuda.synchronize(device)
    nccl_correct = _all_ranks(bool(torch.all(nccl_input == expected).item()), device)
    nccl_input.fill_(rank + 1)

    def nccl_op() -> None:
        dist.all_reduce(nccl_input)

    nccl_median, nccl_p95, nccl_trials = _measure_graph_trials(
        nccl_op,
        capture_context=lambda _stream: nullcontext(),
        warmup=warmup,
        repeat=repeat,
        ops_per_sample=ops_per_sample,
        trials=trials,
        device=device,
    )
    return CalibrationPoint(
        rows=rows,
        custom_median_us=custom_median,
        custom_p95_us=custom_p95,
        nccl_median_us=nccl_median,
        nccl_p95_us=nccl_p95,
        correct=custom_correct and nccl_correct,
        custom_trials_us=custom_trials,
        nccl_trials_us=nccl_trials,
    )


def main() -> None:
    args = _parse_args()
    rows = tuple(int(value) for value in args.rows.split(","))
    if not rows or any(row <= 0 for row in rows) or len(set(rows)) != len(rows):
        raise ValueError("--rows must contain unique positive integers")
    if args.hidden_size <= 0:
        raise ValueError("--hidden-size must be positive")
    if (
        args.warmup < 0
        or args.repeat <= 0
        or args.ops_per_sample <= 0
        or args.trials <= 0
        or args.trials % 2 == 0
        or args.stabilization_replays <= 0
    ):
        raise ValueError(
            "warmup must be non-negative; repeat, odd trials, stabilization "
            "replays, and operations per sample must be positive"
        )

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dtype = getattr(torch, args.dtype)
    dist.init_process_group("nccl", device_id=device)

    max_bytes = (
        max(rows) * args.hidden_size * torch.empty((), dtype=dtype).element_size()
    )
    pool = PCIeOneshotAllReducePool(
        rank=rank,
        world_size=world_size,
        device=device,
        exchange_group=dist.group.WORLD,
        eager_buffer_bytes=max_bytes,
        max_size=max_bytes,
        rank_data_bytes=max_bytes,
        single_channel=True,
    )
    try:
        stabilization_input = torch.full(
            (max(rows), args.hidden_size),
            rank + 1,
            dtype=dtype,
            device=device,
        )
        _stabilize_transports(
            pool,
            stabilization_input,
            torch.empty_like(stabilization_input),
            torch.zeros_like(stabilization_input),
            replays=args.stabilization_replays,
            ops_per_replay=args.ops_per_sample,
            device=device,
        )
        points = [
            _measure_row(
                pool,
                rows=row,
                hidden_size=args.hidden_size,
                dtype=dtype,
                rank=rank,
                world_size=world_size,
                warmup=args.warmup,
                repeat=args.repeat,
                ops_per_sample=args.ops_per_sample,
                trials=args.trials,
                device=device,
            )
            for row in rows
        ]
        policy = select_plain_allreduce_policy(
            points,
            world_size=world_size,
            hidden_size=args.hidden_size,
            dtype=dtype,
            min_absolute_margin_us=args.min_absolute_margin_us,
            min_relative_margin=args.min_relative_margin,
        )
        if rank == 0:
            payload = {
                "policy": json.loads(policy.to_json()),
                "points": [asdict(point) for point in points],
                "selection": {
                    "min_absolute_margin_us": args.min_absolute_margin_us,
                    "min_relative_margin": args.min_relative_margin,
                },
            }
            if args.json is not None:
                args.json.parent.mkdir(parents=True, exist_ok=True)
                args.json.write_text(json.dumps(payload, indent=2) + "\n")
            print(f"B12X_PCIE_PLAIN_ALLREDUCE_POLICY={policy.to_json()}")
            for point in points:
                winner = "B12X" if point.rows in policy.custom_rows else "NCCL"
                print(
                    f"rows={point.rows:4d} B12X={point.custom_median_us:8.3f}us "
                    f"NCCL={point.nccl_median_us:8.3f}us route={winner}"
                )
    finally:
        pool.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
