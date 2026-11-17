# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Cold-L2 MegaMoE trials evict L2 with the shared read flush: once to warm up, then once per sample."""

import contextlib
from types import SimpleNamespace

import pytest
import torch

import flashinfer.utils
from flashinfer.autotuner import AutoTuner
from flashinfer.moe_ep.core import tuning_measurement


class _Trial:
    def __init__(self, device):
        self.device = device
        self.config = SimpleNamespace(world_size=1)
        self.launches = 0

    def launch(self):
        self.launches += 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("cold_l2", [True, False])
def test_measure_samples_cold_l2_uses_the_read_flush(monkeypatch, cold_l2):
    device = torch.device("cuda:0")
    tuner = AutoTuner.get()
    made = []
    flushes = []
    original = flashinfer.utils.make_l2_flush

    def tracked_make_l2_flush(num_bytes, flush_device):
        made.append((num_bytes, flush_device))
        flush = original(num_bytes, flush_device)

        def tracked_flush():
            flushes.append(1)
            flush()

        return tracked_flush

    monkeypatch.setattr(flashinfer.utils, "make_l2_flush", tracked_make_l2_flush)
    monkeypatch.setattr(tuner, "_get_l2_cache_size_in_bytes", lambda device_id: 4096)
    monkeypatch.setattr(tuner, "stream_delay_micro_secs", 0)
    trial = _Trial(device)
    settings = {
        "cuda_graph_profile_replays": None,
        "cold_l2": cold_l2,
        "timer": "events",
    }

    samples = tuning_measurement.measure_samples(trial, (), 3, settings, None)

    assert len(samples) == 3
    assert trial.launches == 3
    if cold_l2:
        # One helper for the trial, sized at twice the L2. One warm-up flush, then
        # one before each of the 3 timed launches.
        assert made == [(2 * 4096, device)]
        assert len(flushes) == 4
    else:
        assert made == []
        assert flushes == []


_COLD = {"cuda_graph_profile_replays": None, "cold_l2": True, "timer": "events"}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("failing_step", ["make", "flush"])
def test_measure_samples_raises_when_the_flush_fails(monkeypatch, failing_step):
    tuner = AutoTuner.get()

    def failing_make_l2_flush(num_bytes, flush_device):
        if failing_step == "make":
            raise RuntimeError("CUDA out of memory")

        def flush():
            if failing_step == "flush":
                raise RuntimeError("CUDA error: invalid argument")

        return flush

    monkeypatch.setattr(flashinfer.utils, "make_l2_flush", failing_make_l2_flush)
    monkeypatch.setattr(tuner, "_get_l2_cache_size_in_bytes", lambda device_id: 4096)
    trial = _Trial(torch.device("cuda:0"))
    with pytest.raises(RuntimeError, match="Cold-L2 flush failed on EP ranks"):
        tuning_measurement.measure_samples(trial, (), 3, _COLD, None)
    assert trial.launches == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_measure_samples_resets_the_graph_when_the_flush_fails(monkeypatch):
    tuner = AutoTuner.get()
    graphs = []

    class _Graph:
        def __init__(self):
            self.resets = 0
            graphs.append(self)

        def replay(self):
            pass

        def reset(self):
            self.resets += 1

    def failing_make_l2_flush(num_bytes, flush_device):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(torch.cuda, "CUDAGraph", _Graph)
    monkeypatch.setattr(torch.cuda, "graph", lambda graph: contextlib.nullcontext())
    monkeypatch.setattr(flashinfer.utils, "make_l2_flush", failing_make_l2_flush)
    monkeypatch.setattr(tuner, "_get_l2_cache_size_in_bytes", lambda device_id: 4096)
    trial = _Trial(torch.device("cuda:0"))
    with pytest.raises(RuntimeError, match="Cold-L2 flush failed on EP ranks"):
        tuning_measurement.measure_samples(
            trial, (), 3, {**_COLD, "cuda_graph_profile_replays": 1}, None
        )
    # The captured graph is reset before the error is raised.
    assert [g.resets for g in graphs] == [1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_measure_samples_raises_when_the_flush_fails_on_another_rank(monkeypatch):
    # Two ranks, simulated: this rank's flush works, the other rank's fails. This
    # rank raises too, before any barrier.
    tuner = AutoTuner.get()

    def all_gather_object(values, value, group=None):
        values[0] = value
        values[1] = "RuntimeError: CUDA out of memory"

    def barrier(group=None):
        raise AssertionError("no barrier may run after a failed flush")

    monkeypatch.setattr(torch.distributed, "all_gather_object", all_gather_object)
    monkeypatch.setattr(torch.distributed, "barrier", barrier)
    monkeypatch.setattr(tuner, "_get_l2_cache_size_in_bytes", lambda device_id: 4096)
    trial = _Trial(torch.device("cuda:0"))
    trial.config.world_size = 2
    with pytest.raises(RuntimeError, match="failed on EP ranks: .*CUDA out of memory"):
        tuning_measurement.measure_samples(trial, (), 3, _COLD, None)
    assert trial.launches == 0
