# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only helpers for the AllReduce communication benchmark."""

import json
import math
import statistics
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


def add_allreduce_control_args(parser):
    """Register AllReduce benchmark controls on an argparse parser."""
    parser.add_argument(
        "--strategy",
        type=str,
        default="both",
        choices=["oneshot", "twoshot", "both", "auto"],
        help=(
            "AllReduce strategy. The default 'both' reports forced oneshot and "
            "twoshot separately; 'auto' uses the API heuristic."
        ),
    )
    completion_group = parser.add_mutually_exclusive_group()
    completion_group.add_argument(
        "--trigger_completion_at_end",
        dest="trigger_completion_at_end",
        action="store_true",
        help="Signal PDL completion after the TRT-LLM kernel finishes (default).",
    )
    completion_group.add_argument(
        "--no_trigger_completion_at_end",
        dest="trigger_completion_at_end",
        action="store_false",
        help=(
            "Signal PDL completion early. Only use before another PDL-aware "
            "kernel that synchronizes grid dependencies."
        ),
    )
    parser.set_defaults(trigger_completion_at_end=True)
    parser.add_argument(
        "--fp32_acc",
        action="store_true",
        default=False,
        help="Use FP32 accumulation with the TRT-LLM backend.",
    )
    parser.add_argument(
        "--l2_cache",
        type=str,
        default="cold",
        choices=["cold", "warm"],
        help="Benchmark with cold (default) or warm L2 cache state.",
    )
    parser.add_argument(
        "--rank_aggregation",
        type=str,
        default="max",
        choices=["max", "rank0", "mean"],
        help="How to aggregate raw per-rank timings for each measured iteration.",
    )
    parser.add_argument(
        "--raw_jsonl_path",
        type=str,
        default=None,
        help=(
            "Optional rank-0 JSONL output containing raw per-rank and aggregated "
            "per-iteration timings."
        ),
    )


def strategies_for_mode(strategy: str) -> list[Optional[bool]]:
    """Expand a CLI strategy into ``allreduce_fusion`` control values."""
    strategies = {
        "oneshot": [True],
        "twoshot": [False],
        "both": [True, False],
        "auto": [None],
    }
    try:
        return strategies[strategy]
    except KeyError as err:
        raise ValueError(f"Unsupported AllReduce strategy: {strategy}") from err


def strategy_request_name(use_oneshot: Optional[bool]) -> str:
    """Return the request label for an API ``use_oneshot`` value."""
    if use_oneshot is True:
        return "oneshot"
    if use_oneshot is False:
        return "twoshot"
    return "auto"


def timing_mode_request(enable_cupti: bool, use_cuda_graph: bool) -> str:
    """Return the requested timer path, before any CUPTI fallback."""
    if enable_cupti:
        return "cupti"
    if use_cuda_graph:
        return "cuda_graph"
    return "cuda_events"


def select_rank_value(values: Sequence[float], rank: int) -> float:
    """Select one rank's value from the timer's internal rank gather.

    ``bench_gpu_time`` invokes its aggregation callback after gathering values
    through ``torch.distributed``. Selecting the caller's rank here keeps each
    process's raw timing instead of applying an early cross-rank reduction.
    """
    if rank < 0 or rank >= len(values):
        raise ValueError(
            f"rank {rank} is outside a gathered world size of {len(values)}"
        )
    return float(values[rank])


def aggregate_rank_times(
    per_rank_times: Sequence[Sequence[float]], policy: str
) -> list[float]:
    """Aggregate equally sized per-rank timing vectors once, per iteration."""
    rows = [[float(value) for value in row] for row in per_rank_times]
    if not rows:
        raise ValueError("per_rank_times must contain at least one rank")

    num_iters = len(rows[0])
    if num_iters == 0:
        raise ValueError("per_rank_times must contain at least one iteration")
    if any(len(row) != num_iters for row in rows):
        raise ValueError("all ranks must provide the same number of timing samples")

    if policy == "rank0":
        return rows[0].copy()

    columns = zip(*rows, strict=True)
    if policy == "max":
        return [max(column) for column in columns]
    if policy == "mean":
        world_size = len(rows)
        return [sum(column) / world_size for column in columns]
    raise ValueError(f"Unsupported rank aggregation policy: {policy}")


def summarize_times(times: Sequence[float]) -> dict[str, float]:
    """Return population stddev, median, and linear-interpolated p90."""
    values = [float(value) for value in times]
    if not values:
        raise ValueError("times must contain at least one sample")

    ordered = sorted(values)
    position = 0.9 * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    weight = position - lower
    p90_time = ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "median_time": float(statistics.median(values)),
        "p90_time": float(p90_time),
        "std_time": float(statistics.pstdev(values)),
    }


def build_allreduce_control_kwargs(
    enable_pdl: bool,
    trigger_completion_at_end: bool,
    fp32_acc: bool,
) -> dict[str, bool]:
    """Build the API controls shared by validation and measured launches."""
    return {
        "launch_with_pdl": enable_pdl,
        "trigger_completion_at_end": trigger_completion_at_end,
        "fp32_acc": fp32_acc,
    }


def append_jsonl(path: str, records: Sequence[Mapping[str, Any]]) -> None:
    """Append JSON records to ``path``, creating parent directories as needed."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")


def raise_if_rank0_error(comm, rank0_error: Optional[str]) -> None:
    """Broadcast a rank-0 error and raise it consistently on every rank."""
    shared_error = comm.bcast(rank0_error, root=0)
    if shared_error is not None:
        raise RuntimeError(shared_error)


def validate_initialized_process_group(
    expected_rank: int,
    expected_world_size: int,
    actual_rank: int,
    actual_world_size: int,
) -> None:
    """Reject a pre-existing torch process group that disagrees with MPI."""
    if actual_rank != expected_rank or actual_world_size != expected_world_size:
        raise RuntimeError(
            "Existing torch.distributed process group does not match MPI: "
            f"expected rank/world_size={expected_rank}/{expected_world_size}, "
            f"got {actual_rank}/{actual_world_size}"
        )


def gather_process_group_initialization(
    comm,
    local_error: Optional[str],
    local_created: bool,
    local_initialized: bool,
) -> dict[str, Any]:
    """Make process-group initialization a collective MPI decision.

    Every rank must either observe the same usable process group or stop before
    workspace creation. ``created_by_benchmark`` is only true when all ranks
    created the group in this benchmark invocation; caller-owned groups must
    never be destroyed by benchmark cleanup.
    """
    states = comm.allgather(
        {
            "error": local_error,
            "created": bool(local_created),
            "initialized": bool(local_initialized),
        }
    )
    errors = [
        f"rank {rank}: {state['error']}"
        for rank, state in enumerate(states)
        if state["error"] is not None
    ]

    if not errors and not all(state["initialized"] for state in states):
        inactive = [
            str(rank) for rank, state in enumerate(states) if not state["initialized"]
        ]
        errors.append(
            "torch.distributed is not initialized on rank(s) " + ", ".join(inactive)
        )

    ownership = {state["created"] for state in states}
    if not errors and len(ownership) != 1:
        errors.append(
            "torch.distributed process-group ownership differs across MPI ranks"
        )

    return {
        "ok": not errors,
        "created_by_benchmark": bool(states[0]["created"]) if not errors else False,
        "error": "; ".join(errors) if errors else None,
    }


def gather_process_group_presence(comm, local_initialized: bool) -> dict[str, Any]:
    """Check process-group presence before any rank enters NCCL initialization."""
    initialized = [bool(value) for value in comm.allgather(local_initialized)]
    if all(initialized):
        return {"ok": True, "all_initialized": True, "error": None}
    if not any(initialized):
        return {"ok": True, "all_initialized": False, "error": None}

    present = [str(rank) for rank, value in enumerate(initialized) if value]
    absent = [str(rank) for rank, value in enumerate(initialized) if not value]
    return {
        "ok": False,
        "all_initialized": False,
        "error": (
            "torch.distributed is pre-initialized only on MPI rank(s) "
            + ", ".join(present)
            + "; missing on rank(s) "
            + ", ".join(absent)
        ),
    }


def gather_rank_errors(
    comm, component: str, local_error: Optional[str]
) -> Optional[str]:
    """Collect rank-local initialization errors into one deterministic message."""
    errors = comm.allgather(local_error)
    failures = [
        f"rank {rank}: {error}"
        for rank, error in enumerate(errors)
        if error is not None
    ]
    if not failures:
        return None
    return f"{component} failed on " + "; ".join(failures)
