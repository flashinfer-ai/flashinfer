# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Timing-policy regressions with a deterministic asynchronous stream model.

The real profiler runs unchanged against CPU CUDA-API stand-ins. These tests
check event windows and cache identity; the simulated durations are not GPU
performance measurements.
"""

from contextlib import contextmanager, nullcontext
import importlib
from types import SimpleNamespace
import pytest
import torch
import flashinfer.autotuner as at_package
from flashinfer import MeasurementPolicy, autotune_v2
from flashinfer.autotuner import TuningConfig
from .utils import DummyRunner, reset_autotuner

at_module = importlib.import_module("flashinfer.autotuner.autotuner")


class StreamModel:
    def __init__(self, flush_ms):
        self.host = 0.0
        self.gpu = 0.0
        self.flush_ms = flush_ms
        self.events = []
        self.capture = None
        self.stream = SimpleNamespace(
            synchronize=self.synchronize, device=SimpleNamespace(index=0)
        )
        model = self

        class Tensor:
            is_cuda = True
            device = SimpleNamespace(index=0)

            def size(self):
                return (32,)

            def zero_(self):
                model.events.append("flush")
                model.enqueue(model.flush_ms)

        class Event:
            def __init__(self, enable_timing):
                assert enable_timing
                self.timestamp = None

            def record(self, stream=None):
                model.gpu = max(model.gpu, model.host)
                self.timestamp = model.gpu
                model.events.append("event")

            def elapsed_time(self, other):
                return other.timestamp - self.timestamp

        class Graph:
            def __init__(self):
                self.ops = []

            def replay(self):
                model.events.append("replay")
                for duration in self.ops:
                    model.enqueue(duration)

        @contextmanager
        def capture(graph):
            assert model.capture is None
            model.capture = graph
            try:
                yield
            finally:
                model.capture = None

        self.Tensor = Tensor
        self.torch = SimpleNamespace(
            Tensor=Tensor,
            int8="int8",
            uint8="uint8",
            empty=lambda *a, **kw: Tensor(),
            cuda=SimpleNamespace(
                Stream=object,
                Event=Event,
                CUDAGraph=Graph,
                current_stream=lambda: self.stream,
                is_current_stream_capturing=lambda: self.capture is not None,
                stream=lambda s: nullcontext(),
                graph=capture,
            ),
        )

    def enqueue(self, duration):
        if self.capture is not None:
            self.capture.ops.append(duration)
        else:
            self.gpu = max(self.gpu, self.host) + duration

    def synchronize(self):
        self.events.append("sync")
        self.host = max(self.host, self.gpu)

    def delay(self, microseconds):
        self.events.append("delay")
        self.enqueue(microseconds / 1000.0)

    def run(self, inputs, tactic, **kwargs):
        assert tactic == 1 and len(inputs) == 1
        self.events.append("runner")
        self.host += 0.3
        self.enqueue(0.03)


@pytest.fixture
def tuner(monkeypatch):
    result = reset_autotuner()
    result._managed_cache = None
    result._managed_stores.clear()
    result._managed_decoded.clear()
    monkeypatch.setattr(result, "warmup", 1)
    monkeypatch.setattr(result, "repeat", 3)
    monkeypatch.setattr(result, "_use_global_timer", False)
    monkeypatch.setattr(result, "_get_l2_cache_size_in_bytes", lambda *a: 32768)
    monkeypatch.setattr(
        at_package,
        "_collect_metadata",
        lambda: dict(
            flashinfer_version="cpu-cold-timer",
            gpu_name="synthetic-cpu-only",
            cuda_version="synthetic",
        ),
    )
    yield result
    reset_autotuner()
    result._managed_cache = None
    result._managed_stores.clear()
    result._managed_decoded.clear()


@pytest.mark.parametrize("flush_ms", [1.0, 10.0])
@pytest.mark.parametrize(
    "mode,timer,graph,include_host",
    [
        ("eager", "auto", False, True),
        ("eager", "events", False, False),
        ("eager", "events_no_delay", False, True),
        ("cuda_graph", "auto", True, False),
        ("cuda_graph", "events_no_delay", True, False),
        ("auto", "events_no_delay", False, True),
        ("auto", "auto", False, False),
    ],
)
def test_cold_event_window_respects_host_cost(
    tuner, monkeypatch, flush_ms, mode, timer, graph, include_host
):
    model = StreamModel(flush_ms)
    inputs = [model.Tensor()]
    policy = MeasurementPolicy(execution_mode=mode, cold_l2=True, _timer=timer)
    config = TuningConfig(
        use_cuda_graph=graph, use_cold_l2_cache=True, cuda_graph_profile_replays=2
    )
    with (
        autotune_v2(mode="tune", persistent_cache=False, measurement_policy=policy),
        monkeypatch.context() as m,
    ):
        m.setattr(at_module, "torch", model.torch)
        m.setattr(at_module, "delay_kernel", model.delay)
        latency = tuner._profile_single_kernel(
            model.run, inputs, 1, config, input_tensor_batches=[inputs]
        )
    assert latency == pytest.approx(0.33 if include_host else 0.03)
    assert ("delay" in model.events) == (policy.timer != "events_no_delay")
    # The L2 flush is outside the event window for every timing policy.
    assert model.events.count("flush") == 3 * (2 if graph else 1)
    assert not torch.cuda.is_initialized()


def test_disabling_delay_alone_does_not_allow_flush_to_hide_host(tuner, monkeypatch):
    model = StreamModel(10.0)
    inputs = [model.Tensor()]
    monkeypatch.setattr(tuner, "stream_delay_micro_secs", 0)
    policy = MeasurementPolicy(execution_mode="eager", cold_l2=True)
    with (
        autotune_v2(mode="tune", persistent_cache=False, measurement_policy=policy),
        monkeypatch.context() as m,
    ):
        m.setattr(at_module, "torch", model.torch)
        m.setattr(at_module, "delay_kernel", model.delay)
        latency = tuner._profile_single_kernel(
            model.run,
            inputs,
            1,
            TuningConfig(use_cuda_graph=False, use_cold_l2_cache=True),
            input_tensor_batches=[inputs],
        )
    assert "delay" not in model.events
    assert latency == pytest.approx(0.33)


@pytest.mark.parametrize("cold", [True, None, False])
def test_corrected_cold_policy_does_not_hydrate_old_timing_records(
    tuner, monkeypatch, tmp_path, cold
):
    policy = MeasurementPolicy(execution_mode="eager", cold_l2=cold)
    runner = DummyRunner(valid_tactics=(1, 2))
    inputs = [torch.zeros(4, 8)]
    cfg = TuningConfig(use_cuda_graph=False, use_cold_l2_cache=True)
    original = MeasurementPolicy.manifest_fields

    def old_manifest(self):
        fields = original(self)
        fields.pop("measure_cold_events_no_delay_revision", None)
        return fields

    calls = []

    def profile(*args, **kwargs):
        calls.append(args[2])
        return {1: 1.0, 2: 2.0, -1: 3.0}[args[2]]

    monkeypatch.setattr(tuner, "_profile_single_kernel", profile)
    with monkeypatch.context() as m:
        m.setattr(MeasurementPolicy, "manifest_fields", old_manifest)
        with autotune_v2(mode="tune", cache_root=tmp_path, measurement_policy=policy):
            assert (
                tuner.choose_one("cold_timing_identity", [runner], cfg, inputs)[1] == 1
            )
    measured = len(calls)
    with autotune_v2(mode="replay", cache_root=tmp_path, measurement_policy=policy):
        actual = tuner.choose_one("cold_timing_identity", [runner], cfg, inputs)[1]
    assert actual == (1 if cold is False else -1)
    assert len(calls) == measured
    fields = policy.manifest_fields()
    assert ("measure_cold_events_no_delay_revision" in fields) == (cold is not False)


def test_default_and_host_excluded_cache_identities_unchanged():
    assert MeasurementPolicy().manifest_fields() == {}
    assert MeasurementPolicy(
        execution_mode="cuda_graph", cold_l2=True
    ).manifest_fields() == dict(
        measure_execution_mode="cuda_graph", measure_cold_l2="True"
    )
    assert MeasurementPolicy(
        execution_mode="eager", cold_l2=True, _timer="events"
    ).manifest_fields() == dict(
        measure_execution_mode="eager", measure_cold_l2="True", measure_timer="events"
    )
