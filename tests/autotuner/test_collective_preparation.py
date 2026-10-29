# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Preparation and storage contracts for collective tunable runners."""

import datetime
import multiprocessing

import pytest
import torch

from flashinfer.autotuner import (
    AutoTuner,
    TuningConfig,
    TunableRunner,
    autotune,
    set_autotune_process_group,
)


class PreparedRunner(TunableRunner):
    def __init__(self, fail=False):
        self.fail = fail
        self.prepared = []

    def get_valid_tactics(self, inputs, profile):
        return [0, 1]

    def prepare_tactic(self, inputs, tactic, **kwargs):
        if self.fail:
            raise ValueError("compile failed")
        self.prepared.append(tactic)

    def forward(self, inputs, tactic=-1, **kwargs):
        assert self.prepared[-1] == tactic
        return inputs[0].add_(1)


def test_collective_config_survives_context_overrides(monkeypatch):
    tuner = AutoTuner()
    monkeypatch.setattr(AutoTuner, "_instance", tuner)
    config = TuningConfig(
        preserve_input_storage=True,
        profiling_warmup=0,
        profiling_repeat=2,
        rank_reduction="max",
    )
    with autotune(cuda_graph_profile_replays=5):
        overridden = tuner.effective_config(config)
    assert overridden.preserve_input_storage
    assert overridden.profiling_warmup == 0
    assert overridden.profiling_repeat == 2
    assert overridden.rank_reduction == "max"
    assert overridden.cuda_graph_profile_replays == 5


def test_preserve_storage_even_with_cold_l2():
    inputs = [torch.zeros(3, 4)]
    config = TuningConfig(preserve_input_storage=True, use_cold_l2_cache=True)
    batches = AutoTuner()._prepare_input_tensors_with_batches(inputs, config)
    assert len(batches) == 1 and batches[0] is inputs


def test_rank_reduction_has_distinct_persisted_identity():
    runner = PreparedRunner()
    mean = AutoTuner._get_cache_key("test", runner, ((3, 4),), TuningConfig())
    maximum = AutoTuner._get_cache_key(
        "test", runner, ((3, 4),), TuningConfig(rank_reduction="max")
    )
    assert mean.file_key != maximum.file_key
    assert mean.extras == ()


def _preparation_rank(rank, path):
    import torch.distributed as dist

    dist.init_process_group(
        "gloo",
        init_method=f"file://{path}",
        rank=rank,
        world_size=2,
        timeout=datetime.timedelta(seconds=30),
    )
    set_autotune_process_group(dist.group.WORLD)
    try:
        tuner = AutoTuner()
        runner = PreparedRunner(fail=rank == 1)
        try:
            tuner.prepare_selected_tactic(runner, [torch.zeros(1)], 0)
        except (ValueError, RuntimeError):
            pass
        else:
            raise AssertionError("both ranks must reject before forward")
        runner.fail = False
        tuner.prepare_selected_tactic(runner, [torch.zeros(1)], 1)
        assert runner.prepared[-1] == 1
    finally:
        set_autotune_process_group(None)
        dist.destroy_process_group()


def test_one_rank_compile_failure_rejects_before_launch(tmp_path):
    ctx = multiprocessing.get_context("spawn")
    processes = [
        ctx.Process(target=_preparation_rank, args=(rank, str(tmp_path / "rdzv")))
        for rank in range(2)
    ]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=50)
        assert [process.exitcode for process in processes] == [0, 0]
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA profiling requires a GPU"
)
@pytest.mark.parametrize("graph", [False, True])
def test_preparation_before_measurement_with_zero_warmup(graph):
    runner = PreparedRunner()
    tuner = AutoTuner(stream_delay_micro_secs=0)
    inputs = [torch.zeros(1, device="cuda")]
    config = TuningConfig(use_cuda_graph=graph, profiling_warmup=0, profiling_repeat=2)
    for tactic in runner.get_valid_tactics(inputs, None):
        assert tuner._profile_single_kernel(runner, inputs, tactic, config) >= 0
    assert runner.prepared == [0, 1]
    assert inputs[0].item() >= 4


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA profiling requires a GPU"
)
@pytest.mark.parametrize("reduction,expected", [("mean", 5.0), ("max", 10.0)])
def test_profile_uses_requested_rank_reduction(monkeypatch, reduction, expected):
    import torch.distributed as dist

    runner = PreparedRunner()
    tuner = AutoTuner(stream_delay_micro_secs=0)
    ops = []

    def reduce(value, op, group):
        if value.dtype == torch.float64:
            ops.append(op)
            value.fill_(10.0)

    monkeypatch.setattr(dist, "get_backend", lambda group: "gloo")
    monkeypatch.setattr(dist, "get_world_size", lambda group: 2)
    monkeypatch.setattr(dist, "all_reduce", reduce)
    set_autotune_process_group(object())
    try:
        result = tuner._profile_single_kernel(
            runner,
            [torch.zeros(1, device="cuda")],
            0,
            TuningConfig(
                rank_reduction=reduction, profiling_warmup=0, profiling_repeat=2
            ),
        )
        assert result == expected
        assert ops == [dist.ReduceOp.MAX if reduction == "max" else dist.ReduceOp.SUM]
    finally:
        set_autotune_process_group(None)


@pytest.mark.parametrize(
    "indices",
    [
        {"profile_arena_input_indices": (0,)},
        {"value_aware_input_indices": (0,)},
        {"profile_arena_input_indices": (0,), "value_aware_input_indices": (0,)},
    ],
)
def test_preserve_storage_rejects_rotating_inputs(indices):
    with pytest.raises(ValueError, match="cannot be combined"):
        TuningConfig(preserve_input_storage=True, **indices)


def test_public_preflight_tracks_context_and_named_cache_state(monkeypatch):
    from dataclasses import replace
    from flashinfer.autotuner import get_autotune_process_group

    tuner = AutoTuner()
    monkeypatch.setattr(AutoTuner, "_instance", tuner)
    runner = PreparedRunner()
    inputs = [torch.zeros(3, 4)]
    config = TuningConfig(preserve_input_storage=True, profiling_warmup=0)
    with tuner.collective_session(None):
        initial = tuner.selection_preflight(
            "prepared", runner, config, inputs, tactics=(0, 1)
        )
        assert not initial.cache_hit and not initial.is_tuning and not initial.skipped
        assert initial.allowed_tactics == (0, 1)
        assert initial.config == config
        with autotune(skip_ops={"prepared"}, cuda_graph_profile_replays=3):
            active = tuner.selection_preflight(
                "prepared", runner, config, inputs, tactics=(0, 1)
            )
            assert active == replace(
                initial,
                is_tuning=True,
                skipped=True,
                config=replace(config, cuda_graph_profile_replays=3),
            )
    original = get_autotune_process_group()
    group = object()
    with pytest.raises(ValueError, match="unwind"), tuner.collective_session(group):
        assert get_autotune_process_group() is group
        raise ValueError("unwind")
    assert get_autotune_process_group() is original
