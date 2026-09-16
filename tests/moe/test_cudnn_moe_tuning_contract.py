# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Preparation and stable-identity contracts for joint BF16 MoE tuning."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from flashinfer.fused_moe import CudnnMoeConfig
from flashinfer.fused_moe import cudnn_backend


FIRST = (20400, ((26, 32), (1000, 0)))
SECOND = (20400, ((26, 32), (1000, 1)))


def test_joint_domains_are_immutable_canonical_and_part_of_config_identity():
    reordered = (20400, tuple(reversed(FIRST[1])))
    config = CudnnMoeConfig(
        fc1_tactics=(reordered, FIRST, SECOND),
        fc2_tactics=(SECOND, FIRST),
        fc1_fusion=False,
    )
    assert config.fc1_tactics == (FIRST, SECOND)
    assert config.fc2_tactics == (SECOND, FIRST)
    assert hash(config) == hash(
        CudnnMoeConfig(
            fc1_tactics=(FIRST, SECOND), fc2_tactics=(SECOND, FIRST), fc1_fusion=False
        )
    )
    assert config != CudnnMoeConfig(
        fc1_tactics=(FIRST, SECOND), fc2_tactics=(SECOND, FIRST), fc1_fusion=True
    )


@pytest.mark.parametrize(
    "options",
    [
        {"fc1_tactic": FIRST, "fc1_tactics": (FIRST,)},
        {"fc2_tactic": SECOND, "fc2_tactics": (SECOND,)},
        {"fc1_tactics": [FIRST]},
        {"fc2_tactics": ((True, ()),)},
        {"fc1_tactics": ((20400, ((26, 32), (26, 64))),)},
        {"fc1_fusion": 1},
        {"fc1_fusion": "auto"},
    ],
)
def test_invalid_domains_fail_before_gpu_preparation(options):
    with pytest.raises(ValueError):
        CudnnMoeConfig(**options)


@pytest.mark.parametrize("decline_first", [False, True])
def test_stage_builds_each_plan_once_and_sizes_workspace_for_all(
    monkeypatch, decline_first
):
    import cudnn

    graph = Mock()
    graph.get_execution_plan_count.return_value = 2
    graph.get_engine_and_knobs_at_index.side_effect = lambda index: (
        20400,
        dict((FIRST, SECOND)[index][1]),
    )
    graph.get_workspace_size.return_value = 8
    graph.get_workspace_size_plan_at_index.side_effect = lambda index: (8, 72)[index]
    if decline_first:

        def build(index):
            if index == 0:
                raise NotImplementedError("unsupported automatic proposal")

        graph.build_plan_at_index.side_effect = build
    monkeypatch.setattr(cudnn, "create_handle", lambda: object())
    monkeypatch.setattr(cudnn, "pygraph", lambda **kwargs: graph)
    monkeypatch.setattr(
        cudnn_backend, "_check_cudnn_plan_build_not_capturing", lambda _: None
    )
    a = torch.empty((3, 16), dtype=torch.bfloat16)
    weights = torch.empty((2, 16, 16), dtype=torch.bfloat16)
    offsets = torch.tensor([0, 1, 3], dtype=torch.int32)
    stage = cudnn_backend._Stage(
        a, [weights], offsets, a, tactics=() if decline_first else (FIRST, SECOND)
    )
    assert stage.tactics == ((SECOND,) if decline_first else (FIRST, SECOND))
    assert stage.workspace.numel() == 72
    assert graph.create_execution_plan.call_count == (0 if decline_first else 2)
    assert [call.args for call in graph.build_plan_at_index.call_args_list] == [
        (0,),
        (1,),
    ]
    if decline_first:
        with pytest.raises(ValueError, match="unavailable"):
            stage.plan_index(FIRST)


def test_joint_tactics_cover_both_stages_and_reject_stale_fc2_before_execution():
    def stage(records):
        graph = Mock()
        graph.get_execution_plan_count.return_value = len(records)
        graph.get_engine_and_knobs_at_index.side_effect = lambda index: (
            records[index][0],
            dict(records[index][1]),
        )
        prepared = object.__new__(cudnn_backend._Stage)
        prepared.graph = graph
        prepared.tactics = records
        prepared.tactic_indices = {
            record: index for index, record in enumerate(records)
        }
        return prepared

    state = {"fc1": stage((FIRST, SECOND)), "fc2": stage((SECOND, FIRST))}
    runner = object.__new__(cudnn_backend.CudnnMoeRunner)
    runner._resources = lambda inputs: state
    assert runner.get_valid_tactics([], None) == [
        (FIRST, SECOND),
        (FIRST, FIRST),
        (SECOND, SECOND),
        (SECOND, FIRST),
    ]
    assert runner._stage_indices(state, (SECOND, FIRST)) == (1, 1)
    assert runner._stage_indices(state, FIRST) == (0, -1)
    assert runner._stage_indices(state, -1) == (-1, -1)
    with pytest.raises(ValueError, match="unavailable"):
        runner._stage_indices(state, (FIRST, (999999, ())))


@pytest.mark.parametrize("fusion", [True, False])
def test_explicit_fusion_selects_only_the_requested_preparation(monkeypatch, fusion):
    calls = []

    def stage(*args, **kwargs):
        calls.append(kwargs)
        return object()

    monkeypatch.setattr(cudnn_backend, "_Stage", stage)
    monkeypatch.setattr(cudnn_backend, "_Activation", lambda *args: object())
    runner = object.__new__(cudnn_backend.CudnnMoeRunner)
    runner.backend_config = CudnnMoeConfig(
        fc1_tactics=(FIRST, SECOND), fc2_tactics=(SECOND,), fc1_fusion=fusion
    )
    runner.config = SimpleNamespace(
        routing=SimpleNamespace(top_k=2), activation=object()
    )
    a, weight = torch.empty((3, 16)), torch.empty((2, 16, 16))
    state = {
        name: object() for name in ("routed", "offsets", "intermediate", "projected")
    }
    runner._prepare_stages(
        state,
        [a, None, None, weight, weight, weight, weight],
        lambda shape, dtype: torch.empty(shape, dtype=dtype),
    )
    assert len(calls) == 2 and calls[0].get("fused", False) is fusion
    assert calls[0]["tactics"] == (FIRST, SECOND) and calls[1]["tactics"] == (SECOND,)
    assert state["fused"] is fusion
    assert ("fc1_output" in state) is not fusion


def test_required_fusion_decline_is_not_silently_replaced(monkeypatch):
    calls = []

    def stage(*args, **kwargs):
        calls.append(kwargs)
        raise NotImplementedError("unsupported fusion")

    monkeypatch.setattr(cudnn_backend, "_Stage", stage)
    runner = object.__new__(cudnn_backend.CudnnMoeRunner)
    runner.backend_config = CudnnMoeConfig(fc1_fusion=True)
    runner.config = SimpleNamespace(
        routing=SimpleNamespace(top_k=2), activation=object()
    )
    a, weight = torch.empty((3, 16)), torch.empty((2, 16, 16))
    state = {
        name: object() for name in ("routed", "offsets", "intermediate", "projected")
    }
    with pytest.raises(NotImplementedError, match="unsupported fusion"):
        runner._prepare_stages(
            state,
            [a, None, None, weight, weight, weight, weight],
            Mock(side_effect=AssertionError("fallback allocated storage")),
        )
    assert len(calls) == 1 and calls[0]["fused"]
