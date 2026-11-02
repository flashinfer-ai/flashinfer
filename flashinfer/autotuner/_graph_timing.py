# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Individual-invocation GPU timing for latency-oriented graph workloads."""

from statistics import median
from typing import Callable, Optional

import torch


def profile_individual_graph_calls(
    run_once: Callable[[int], object],
    *,
    repeat: int,
    replays: int = 1,
    record_global_timer: Optional[Callable[[torch.Tensor], None]] = None,
) -> float:
    """Return median GPU milliseconds per call, excluding host submission.

    The caller must warm up every compiled variant before entering this helper.
    Each invocation has timestamp nodes *inside* the captured graph: external
    CUDA events, or a precompiled globaltimer kernel when requested.
    This keeps adjacent independent invocations out of one amortized interval,
    without charging Python/graph submission to a small GPU workload. One
    untimed replay absorbs graph initialization. Inputs and outputs retain their
    normal graph ownership; the helper never caches their pointers.
    """
    if repeat < 1 or replays < 1:
        raise ValueError("repeat and replays must be positive")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Individual graph profiling must run before capture")
    if record_global_timer is not None:
        timestamps = torch.empty((repeat, 2), dtype=torch.int64, device="cuda")
        starts = [timestamps[i, :1] for i in range(repeat)]
        ends = [timestamps[i, 1:] for i in range(repeat)]

        def record_start(index):
            record_global_timer(starts[index])

        def record_end(index):
            record_global_timer(ends[index])

        def synchronize():
            torch.cuda.current_stream().synchronize()

        def elapsed_times():
            return [(end - start) / 1e6 for start, end in timestamps.cpu().tolist()]
    else:
        intervals = [
            (
                torch.cuda.Event(enable_timing=True, external=True),
                torch.cuda.Event(enable_timing=True, external=True),
            )
            for _ in range(repeat)
        ]

        def record_start(index):
            intervals[index][0].record()

        def record_end(index):
            intervals[index][1].record()

        def synchronize():
            intervals[-1][1].synchronize()

        def elapsed_times():
            return [start.elapsed_time(end) for start, end in intervals]

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for index in range(repeat):
            record_start(index)
            run_once(index)
            record_end(index)
    graph.replay()
    synchronize()
    samples: list[float] = []
    for _ in range(replays):
        graph.replay()
        synchronize()
        samples.extend(elapsed_times())
    return median(samples)
