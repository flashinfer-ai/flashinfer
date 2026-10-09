"""CPU proof of the SM90 collective autotune scoring definition."""

from __future__ import annotations

from itertools import cycle
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
import torch.distributed as dist

from flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel.shim import (
    autotune as autotune_module,
    comm,
    hopper_fp8,
)


@pytest.fixture
def mock_collective(monkeypatch):
    group = object()
    barriers = mock.Mock()
    tensor = torch.tensor
    clock = cycle((0.0, 1.0))
    monkeypatch.setattr(comm, "ensure_not_capturing", lambda _: None)
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda group=None: 2)
    monkeypatch.setattr(dist, "get_rank", lambda group=None: 0)
    monkeypatch.setattr(dist, "barrier", barriers)
    monkeypatch.setattr(dist, "all_reduce", mock.Mock())
    monkeypatch.setattr(
        torch,
        "tensor",
        lambda values, **kwargs: tensor(values, **dict(kwargs, device="cpu")),
    )
    monkeypatch.setattr(autotune_module.time, "perf_counter", lambda: next(clock))
    return group, barriers


def test_score_is_max_across_rank_local_iteration_medians(monkeypatch, mock_collective):
    candidates = [{"id": "a"}, {"id": "b"}]
    frontend = SimpleNamespace(apply_knobs=mock.Mock())
    callback = mock.Mock()
    ep_group, barriers = mock_collective
    observed_local = []
    # Local medians are 3 and 5 despite A's outlier. Remote medians make B win.
    clock = iter([0, 1, 0, 100, 0, 3, 0, 4, 0, 5, 0, 6])
    monkeypatch.setattr(autotune_module.time, "perf_counter", lambda: next(clock))

    def all_reduce(scores, op, group=None):
        assert group is ep_group
        if op == dist.ReduceOp.MIN:
            assert scores.tolist() == [1]
        else:
            assert op == dist.ReduceOp.MAX
            observed_local.append(scores.tolist())
            scores[:] = torch.tensor([7.0, 5.5])

    monkeypatch.setattr(dist, "all_reduce", all_reduce)
    winner = autotune_module.autotune_knobs(
        frontend,
        lambda: None,
        candidates,
        label="score-contract",
        warmup_iters=0,
        timed_iters=3,
        process_group=ep_group,
        expected_world_size=2,
        on_winner=callback,
    )
    assert observed_local == [[3.0, 5.0]]
    assert winner == candidates[1]
    assert frontend.apply_knobs.call_args_list == [
        mock.call(candidates[0]),
        mock.call(candidates[1]),
        mock.call(candidates[1]),
    ]
    callback.assert_called_once_with(candidates[1], pytest.approx(5.5))
    assert barriers.called
    assert all(call == mock.call(group=ep_group) for call in barriers.call_args_list)


@pytest.mark.parametrize("failed_phase", ["apply", "prepare"])
def test_remote_candidate_failure_is_discarded_before_launch(
    monkeypatch, mock_collective, failed_phase
):
    candidates = [{"id": "bad"}, {"id": "good"}]
    ep_group, _ = mock_collective
    current = phase = None
    launches, prepared, discarded = [], [], []

    def apply(knobs):
        nonlocal current, phase
        current, phase = knobs["id"], "apply"

    def prepare():
        nonlocal phase
        prepared.append(current)
        phase = "prepare"

    def all_reduce(scores, op, group=None):
        nonlocal phase
        assert group is ep_group
        if op == dist.ReduceOp.MIN:
            if current == "bad" and phase == failed_phase:
                scores[0] = 0
            phase = None
        else:
            assert op == dist.ReduceOp.MAX
            assert scores.tolist() == [float("inf"), 1.0]

    monkeypatch.setattr(dist, "all_reduce", all_reduce)
    with pytest.warns(RuntimeWarning, match="failed on another EP rank"):
        winner = autotune_module.autotune_knobs(
            SimpleNamespace(apply_knobs=apply),
            lambda: launches.append(current),
            candidates,
            label="failure-contract",
            warmup_iters=0,
            timed_iters=1,
            process_group=ep_group,
            expected_world_size=2,
            prepare_candidate=prepare,
            discard_candidate=lambda: discarded.append(current),
        )
    assert winner == candidates[1]
    assert launches == ["good"]
    assert prepared == (["bad"] if failed_phase == "prepare" else []) + ["good", "good"]
    assert discarded == ["bad", "good"]


def test_winner_prepare_failure_discards_and_prevents_record(
    monkeypatch, mock_collective
):
    discard, record = mock.Mock(), mock.Mock()
    prepare = mock.Mock(side_effect=[None, ValueError("winner compile rejected")])
    monkeypatch.setattr(dist, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="winner preparation failed"):
        autotune_module.autotune_knobs(
            SimpleNamespace(apply_knobs=mock.Mock()),
            lambda: None,
            [{"id": "winner"}],
            label="winner-prepare",
            warmup_iters=0,
            timed_iters=1,
            prepare_candidate=prepare,
            discard_candidate=discard,
            on_winner=record,
        )
    assert discard.call_count == 2
    record.assert_not_called()


def test_prepare_launch_makes_first_run_use_hot_path():
    frontend = object.__new__(hopper_fp8.MegaMoEHopperFp8Frontend)
    frontend._config = SimpleNamespace(
        gate_up_clamp=None,
        in_kernel_fc2_reduce=False,
    )
    frontend._gate_up_clamp = None
    inputs = SimpleNamespace()
    launch_inputs = SimpleNamespace(output_activation=object())
    compiled = mock.Mock()
    mega = SimpleNamespace(
        compiled=compiled,
        launch_key=None,
        launch_kwargs=None,
        launch_output=None,
    )
    frontend._mega = mega
    frontend._mega_key = ("compiled",)
    frontend._resolve_num_tokens = mock.Mock(return_value=8)
    frontend._prepare_launch_inputs = mock.Mock(return_value=launch_inputs)
    frontend._launch_cache_key = mock.Mock(return_value=("launch",))
    frontend._ensure_mega_compiled = mock.Mock(return_value=mega)
    frontend._build_mega_runtime_kwargs = mock.Mock(return_value={"arg": 1})

    frontend.prepare_launch(inputs, num_tokens=None)
    assert (
        frontend.run(inputs, num_tokens=None, sync=False)
        is launch_inputs.output_activation
    )

    frontend._ensure_mega_compiled.assert_called_once_with(inputs)
    frontend._build_mega_runtime_kwargs.assert_called_once_with(launch_inputs, mega)
    compiled.assert_called_once_with(arg=1)


def test_partial_fused_workspace_owner_is_releasable(monkeypatch):
    frontend = object.__new__(hopper_fp8.MegaMoEHopperFp8Frontend)
    shared_workspace = object()
    frontend._mega = SimpleNamespace(shared_workspace=shared_workspace)
    frontend._mega_key = None
    free = mock.Mock()
    monkeypatch.setattr(hopper_fp8, "free_sym_tensor", free)
    monkeypatch.setattr(hopper_fp8, "ensure_not_capturing", mock.Mock())

    frontend.release()

    free.assert_called_once_with(shared_workspace)
    assert frontend._mega is None


def test_expected_ep_world_size_rejects_wrong_process_group(monkeypatch):
    ep_group = object()
    frontend = SimpleNamespace(apply_knobs=mock.Mock())

    monkeypatch.setattr(comm, "ensure_not_capturing", mock.Mock())
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda group=None: 4)

    with pytest.raises(RuntimeError, match="expected EP world size 2"):
        autotune_module.autotune_knobs(
            frontend,
            mock.Mock(),
            [{"id": "unused"}],
            label="wrong-group",
            process_group=ep_group,
            expected_world_size=2,
        )
    frontend.apply_knobs.assert_not_called()


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"prepare_candidate": mock.Mock()}, "provided together"),
        ({"discard_candidate": mock.Mock()}, "provided together"),
    ],
)
def test_autotune_callback_groups_are_all_or_none(monkeypatch, kwargs, match):
    monkeypatch.setattr(comm, "ensure_not_capturing", mock.Mock())
    with pytest.raises(ValueError, match=match):
        autotune_module.autotune_knobs(
            SimpleNamespace(apply_knobs=mock.Mock()),
            mock.Mock(),
            [{"id": "unused"}],
            label="callback-contract",
            **kwargs,
        )
