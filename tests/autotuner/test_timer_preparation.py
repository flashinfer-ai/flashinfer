# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""First-use timer setup must precede candidate warmup."""

import importlib

import pytest
import torch

from flashinfer.autotune_cache import MeasurementPolicy, autotune_v2
from flashinfer.autotuner import AutoTuner, TunableRunner, TuningConfig


@pytest.mark.parametrize(
    "graph,cold,individual,timer,delay,expected",
    [
        (False, False, False, "events", 100, True),
        (True, False, False, "events", 100, True),
        (False, True, False, "events", 100, True),
        (True, True, True, "events", 100, True),
        (False, False, True, "events", 100, True),
        (True, False, True, "events", 100, False),
        (False, False, False, "events_no_delay", 100, False),
        (False, True, False, "events_no_delay", 100, True),
        (True, False, False, "events", 0, False),
        (False, False, False, "events", 0, False),
    ],
)
def test_delay_helper_prepared_before_warmup(
    monkeypatch, graph, cold, individual, timer, delay, expected
):
    implementation = importlib.import_module("flashinfer.autotuner.autotuner")
    tuner = AutoTuner(warmup=1, stream_delay_micro_secs=delay)
    monkeypatch.setattr(AutoTuner, "_instance", tuner)
    monkeypatch.setattr(tuner, "_CUDA_GRAPH_DELAY_MICRO_SECS", delay)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: None)
    prepared = []
    monkeypatch.setattr(
        implementation, "get_trtllm_utils_module", lambda: prepared.append(True)
    )

    class ReachedWarmup(Exception):
        pass

    class Runner(TunableRunner):
        def get_valid_tactics(self, inputs, profile):
            return [0]

        def forward(self, inputs, tactic=-1, **kwargs):
            assert prepared == ([True] if expected else [])
            raise ReachedWarmup

    config = TuningConfig(
        use_cuda_graph=graph,
        use_cold_l2_cache=cold,
        use_cuda_graph_individual_timing=individual,
    )
    inputs = [torch.ones(1)]
    with (
        autotune_v2(
            persistent_cache=False, measurement_policy=MeasurementPolicy(_timer=timer)
        ),
        pytest.raises(ReachedWarmup),
    ):
        tuner._profile_single_kernel(
            Runner(), inputs, 0, config, input_tensor_batches=[inputs]
        )


def test_cupti_does_not_prepare_delay_helper(monkeypatch):
    implementation = importlib.import_module("flashinfer.autotuner.autotuner")
    tuner = AutoTuner()
    monkeypatch.setattr(AutoTuner, "_instance", tuner)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: None)
    monkeypatch.setattr(implementation, "_load_cupti", lambda: object())
    monkeypatch.setattr(tuner, "_profile_single_kernel_cupti", lambda *a, **kw: 1.0)
    prepared = []
    monkeypatch.setattr(
        implementation, "get_trtllm_utils_module", lambda: prepared.append(True)
    )
    with autotune_v2(
        persistent_cache=False, measurement_policy=MeasurementPolicy(_timer="cupti")
    ):
        assert (
            tuner._profile_single_kernel(object(), [torch.ones(1)], 0, TuningConfig())
            == 1.0
        )
    assert not prepared


def test_delay_helper_failure_still_reaches_group_reduction(monkeypatch):
    import math
    import torch.distributed as dist

    implementation = importlib.import_module("flashinfer.autotuner.autotuner")
    tuner = AutoTuner()
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: None)
    monkeypatch.setattr(implementation, "_tune_process_group", object())
    monkeypatch.setattr(dist, "get_backend", lambda group: "gloo")
    monkeypatch.setattr(dist, "get_world_size", lambda group: 2)
    reduced = []
    monkeypatch.setattr(
        dist, "all_reduce", lambda value, **kw: reduced.append(value.item())
    )

    def fail_setup():
        raise RuntimeError("timer module setup failed")

    monkeypatch.setattr(implementation, "get_trtllm_utils_module", fail_setup)
    inputs = [torch.ones(1)]
    with pytest.raises(RuntimeError, match="timer module setup failed"):
        tuner._profile_single_kernel(
            object(), inputs, 0, TuningConfig(), input_tensor_batches=[inputs]
        )
    assert len(reduced) == 1 and math.isinf(reduced[0])
