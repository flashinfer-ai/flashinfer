# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Exercise the production profiler with Gloo and simulated CUDA/CUPTI calls."""

import contextlib
import datetime
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist

from flashinfer.autotune_cache import MeasurementPolicy
from flashinfer.autotuner import AutoTuner, TunableRunner, TuningConfig
import flashinfer.autotuner.autotuner as module


def _profile_case(rank, case, use_graph):
    tuner = AutoTuner()
    tuner.warmup = 1
    tuner.repeat = 2
    tuner.stream_delay_micro_secs = 0
    tuner._CUDA_GRAPH_DELAY_MICRO_SECS = 0
    tuner._use_global_timer = False
    tuner._cupti_disabled = case == "disabled" and rank == 1
    trace = SimpleNamespace(
        initialized=False, enabled=[], records=[], clock=0, callback=None, attempts=0
    )
    calls = []
    captured = []
    event_calls = []

    class Cupti:
        ActivityKind = SimpleNamespace(RUNTIME=1, DRIVER=2, CONCURRENT_KERNEL=3)

        def activity_enable(self, kind):
            trace.attempts += 1
            if case == "init" and rank == 1 and kind == self.ActivityKind.DRIVER:
                raise RuntimeError("simulated CUPTI initialization failure")
            trace.enabled.append(kind)

        def activity_disable(self, kind):
            trace.enabled.remove(kind)
            trace.initialized = False

        def activity_register_callbacks(self, requested, completed):
            trace.callback = completed
            trace.initialized = True

        def activity_flush_all(self, flags):
            records = trace.records
            if case == "records" and rank == 1:
                records = [
                    r for r in records if r.kind != self.ActivityKind.CONCURRENT_KERNEL
                ]
            if trace.callback is not None:
                trace.callback(records)
            trace.records = []

        def get_timestamp(self):
            trace.clock += 10_000_000
            return trace.clock

    cupti = Cupti()

    class Runner(TunableRunner):
        def get_valid_tactics(self, inputs, profile):
            return [0]

        def forward(self, inputs, tactic=-1, **kwargs):
            # Warmup and graph capture must wait for the readiness decision.
            assert trace.initialized or tuner._cupti_disabled or case == "events"
            if captured:
                captured[-1].operations.append(self.launch)
            else:
                self.launch()

        def launch(self):
            calls.append("cupti" if trace.initialized else "events")
            if case == "kernel" and rank == 1 and len(calls) == tuner.warmup + 1:
                raise RuntimeError("simulated candidate failure")
            if trace.initialized:
                correlation = len(calls)
                trace.records.extend(
                    [
                        SimpleNamespace(
                            kind=cupti.ActivityKind.RUNTIME,
                            start=trace.clock + 1,
                            correlation_id=correlation,
                        ),
                        SimpleNamespace(
                            kind=cupti.ActivityKind.CONCURRENT_KERNEL,
                            start=trace.clock + 2,
                            end=trace.clock + 2 + (rank + 1) * 1_000_000,
                            correlation_id=correlation,
                        ),
                    ]
                )

    class Graph:
        def __init__(self):
            self.operations = []

        def replay(self):
            for operation in self.operations:
                operation()

    @contextlib.contextmanager
    def capture(graph):
        assert trace.initialized or tuner._cupti_disabled or case == "events"
        captured.append(graph)
        try:
            yield
        finally:
            captured.pop()

    class Event:
        def __init__(self, **kwargs):
            pass

        def record(self):
            event_calls.append(True)

        def elapsed_time(self, other):
            return 80.0

    stream = SimpleNamespace(synchronize=lambda: None)
    policy = MeasurementPolicy(_timer="events" if case == "events" else "cupti")
    error = None
    result = None
    with contextlib.ExitStack() as stack:
        for name, value in {
            "is_current_stream_capturing": lambda: False,
            "current_stream": lambda: stream,
            "synchronize": lambda: None,
            "stream": lambda stream: contextlib.nullcontext(),
            "CUDAGraph": Graph,
            "graph": capture,
            "Event": Event,
        }.items():
            stack.enter_context(patch.object(torch.cuda, name, value))
        stack.enter_context(
            patch.object(
                module,
                "_load_cupti",
                lambda: None if case == "unavailable" and rank == 1 else cupti,
            )
        )
        stack.enter_context(
            patch.object(
                AutoTuner, "_effective_measure_policy", property(lambda self: policy)
            )
        )
        stack.enter_context(
            patch.object(tuner, "_get_profiling_repeat", return_value=2)
        )
        try:
            result = tuner._profile_single_kernel(
                Runner(),
                [],
                0,
                TuningConfig(use_cuda_graph=use_graph, use_cold_l2_cache=False),
                input_tensor_batches=[[]],
            )
        except RuntimeError as exc:
            error = str(exc)

    assert not trace.enabled
    if case == "kernel":
        if rank == 1:
            assert error == "simulated candidate failure"
        else:
            assert error is None and result == float("inf")
        assert not event_calls
        assert not tuner._cupti_disabled
    else:
        assert error is None, error
        assert result == (2.5 if case == "success" else 40.0)
        assert bool(event_calls) == (case != "success")
        if case in ("unavailable", "disabled", "init"):
            assert calls and set(calls) == {"events"}
        if case in ("unavailable", "disabled"):
            assert trace.attempts == 0
        assert tuner._cupti_disabled == (case not in ("success", "events"))


def _worker(rank, store, use_graph):
    dist.init_process_group(
        "gloo",
        init_method="file://" + store,
        rank=rank,
        world_size=3,
        timeout=datetime.timedelta(seconds=30),
    )
    try:
        group = dist.new_group([1, 2], backend="gloo")
        if rank in (1, 2):
            module.set_autotune_process_group(group)
            for case in (
                "success",
                "unavailable",
                "disabled",
                "init",
                "records",
                "kernel",
                "events",
            ):
                _profile_case(rank, case, use_graph)
        dist.barrier()
    finally:
        module.set_autotune_process_group(None)
        dist.destroy_process_group()


@pytest.mark.parametrize("use_graph", [False, True], ids=["eager", "graph"])
def test_cupti_fallback_is_collective(tmp_path, use_graph):
    """A rank-local tracing failure changes timers for the whole tuning subgroup."""
    processes = []
    deadline = time.monotonic() + 120
    try:
        for rank in range(3):
            processes.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        str(rank),
                        str(tmp_path / "store"),
                        str(int(use_graph)),
                    ],
                    env={**os.environ, "TORCH_DISTRIBUTED_DEBUG": "DETAIL"},
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            )
        outputs = []
        for rank, process in enumerate(processes):
            output, _ = process.communicate(timeout=max(0, deadline - time.monotonic()))
            outputs.append(f"rank {rank}:\n{output}")
        assert all(process.returncode == 0 for process in processes), "\n".join(outputs)
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
        for process in processes:
            process.communicate()


if __name__ == "__main__":
    _worker(int(sys.argv[1]), sys.argv[2], bool(int(sys.argv[3])))
