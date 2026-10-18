# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Individual-invocation GPU timing for latency-oriented graph workloads."""

from statistics import median
from typing import Callable

import torch


def profile_individual_graph_calls(
    run_once: Callable[[int], object], *, repeat: int, replays: int = 1
) -> float:
    """Return median GPU milliseconds per call, excluding host submission.

    The caller must warm up every compiled variant before entering this helper.
    Each invocation has external CUDA event nodes *inside* the captured graph.
    This keeps adjacent independent invocations out of one amortized interval,
    without charging Python/graph submission to a small GPU workload. One
    untimed replay absorbs graph initialization. Inputs and outputs retain their
    normal graph ownership; the helper never caches their pointers.
    """
    if repeat < 1 or replays < 1:
        raise ValueError("repeat and replays must be positive")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Individual graph profiling must run before capture")
    intervals = [
        (
            torch.cuda.Event(enable_timing=True, external=True),
            torch.cuda.Event(enable_timing=True, external=True),
        )
        for _ in range(repeat)
    ]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for index, (start, end) in enumerate(intervals):
            start.record()
            run_once(index)
            end.record()
    graph.replay()
    intervals[-1][1].synchronize()
    samples: list[float] = []
    for _ in range(replays):
        graph.replay()
        intervals[-1][1].synchronize()
        samples.extend(start.elapsed_time(end) for start, end in intervals)
    return median(samples)
