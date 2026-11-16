# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Cold-L2 MegaMoE trials evict L2 with the shared read flush, once per sample."""

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
        # one helper for the trial, sized at twice the L2, and one flush before each timed launch
        assert made == [(2 * 4096, device)]
        assert len(flushes) == 3
    else:
        assert made == []
        assert flushes == []
