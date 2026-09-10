"""CPU proof of the SM90 collective autotune scoring definition."""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest
import torch.distributed as dist

from flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel.shim import (
    autotune as autotune_module,
    comm,
    hopper_fp8,
)


class _Scalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value

    def __float__(self):
        return float(self.value)


class _Scores:
    def __init__(self, values):
        self.values = list(values)

    def __getitem__(self, index):
        return _Scalar(self.values[index])

    def tolist(self):
        return list(self.values)


def test_score_is_max_across_rank_local_iteration_medians(monkeypatch):
    """Official score is MAX_rank(MEDIAN_iteration), not median of rank maxes."""

    candidates = [{"id": "a"}, {"id": "b"}]
    frontend = SimpleNamespace(apply_knobs=mock.Mock())
    callback = mock.Mock()
    barriers = mock.Mock()
    ep_group = object()
    observed_local = []
    status_calls = []

    monkeypatch.setattr(comm, "ensure_not_capturing", mock.Mock())
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda group=None: 2)
    monkeypatch.setattr(dist, "get_rank", lambda group=None: 0)
    monkeypatch.setattr(dist, "barrier", barriers)

    # Candidate A local times have a 100s outlier: local median is 3s.
    # Candidate B local median is 5s. A remote rank then raises A's collective
    # score to 7s and B's to 5.5s, so B must win.
    clock = iter(
        [
            0.0,
            1.0,
            0.0,
            100.0,
            0.0,
            3.0,
            0.0,
            4.0,
            0.0,
            5.0,
            0.0,
            6.0,
        ]
    )
    monkeypatch.setattr(autotune_module.time, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(
        autotune_module.torch,
        "tensor",
        lambda values, **kwargs: _Scores(values),
    )
    monkeypatch.setattr(
        autotune_module.torch,
        "argmin",
        lambda scores: SimpleNamespace(
            item=lambda: min(
                range(len(scores.values)),
                key=scores.values.__getitem__,
            )
        ),
    )

    def all_reduce(scores, op, group=None):
        assert group is ep_group
        if op == dist.ReduceOp.MIN:
            status_calls.append(scores.tolist())
            assert scores.tolist() == [1]
            return
        observed_local.append(scores.tolist())
        assert op == dist.ReduceOp.MAX
        scores.values[:] = [7.0, 5.5]

    monkeypatch.setattr(dist, "all_reduce", all_reduce)

    winner = autotune_module.autotune_knobs(
        frontend,
        lambda: None,
        candidates,
        label="score-contract",
        warmup_iters=0,
        process_group=ep_group,
        expected_world_size=2,
        timed_iters=3,
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
    # Three candidate phases plus winner apply and winner commit are aligned.
    expected_barriers = 3 * len(candidates) + 2
    assert barriers.call_args_list == [mock.call(group=ep_group)] * expected_barriers
    assert status_calls == [[1]] * expected_barriers


def test_remote_candidate_failure_is_collectively_skipped(monkeypatch):
    """A rank-local failure makes every EP rank reject the same candidate."""

    candidates = [{"id": "remote-failure"}, {"id": "good"}]
    frontend = SimpleNamespace(apply_knobs=mock.Mock())
    launch = mock.Mock()
    barriers = mock.Mock()
    ep_group = object()
    status_round = 0

    monkeypatch.setattr(comm, "ensure_not_capturing", mock.Mock())
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda group=None: 2)
    monkeypatch.setattr(dist, "get_rank", lambda group=None: 0)
    monkeypatch.setattr(dist, "barrier", barriers)
    clock = iter([0.0, 1.0])
    monkeypatch.setattr(autotune_module.time, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(
        autotune_module.torch,
        "tensor",
        lambda values, **kwargs: _Scores(values),
    )
    monkeypatch.setattr(
        autotune_module.torch,
        "argmin",
        lambda scores: SimpleNamespace(
            item=lambda: min(
                range(len(scores.values)),
                key=scores.values.__getitem__,
            )
        ),
    )

    def all_reduce(scores, op, group=None):
        nonlocal status_round
        assert group is ep_group
        if op == dist.ReduceOp.MIN:
            status_round += 1
            # Simulate the other EP rank failing the first candidate's apply.
            if status_round == 1:
                scores.values[0] = 0
            return
        assert op == dist.ReduceOp.MAX
        assert scores.tolist() == [float("inf"), 1.0]

    monkeypatch.setattr(dist, "all_reduce", all_reduce)

    with pytest.warns(RuntimeWarning, match="failed on another EP rank"):
        winner = autotune_module.autotune_knobs(
            frontend,
            launch,
            candidates,
            label="failure-contract",
            warmup_iters=0,
            timed_iters=1,
            process_group=ep_group,
            expected_world_size=2,
        )

    assert winner == candidates[1]
    assert frontend.apply_knobs.call_args_list == [
        mock.call(candidates[0]),
        mock.call(candidates[1]),
        mock.call(candidates[1]),
    ]
    # The rejected candidate never launches; the next candidate remains usable.
    launch.assert_called_once_with()
    assert status_round == 6
    assert barriers.call_args_list == [mock.call(group=ep_group)] * 6


def test_remote_prepare_failure_is_discarded_before_any_launch(monkeypatch):
    candidates = [{"id": "compile-fails-remotely"}, {"id": "good"}]
    current = None
    launches = []
    prepare_calls = []
    discard_calls = []
    status_round = 0
    ep_group = object()

    class Frontend:
        def apply_knobs(self, knobs):
            nonlocal current
            current = knobs["id"]

    monkeypatch.setattr(comm, "ensure_not_capturing", mock.Mock())
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda group=None: 2)
    monkeypatch.setattr(dist, "get_rank", lambda group=None: 0)
    monkeypatch.setattr(dist, "barrier", mock.Mock())
    clock = iter([0.0, 1.0])
    monkeypatch.setattr(autotune_module.time, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(
        autotune_module.torch,
        "tensor",
        lambda values, **kwargs: _Scores(values),
    )
    monkeypatch.setattr(
        autotune_module.torch,
        "argmin",
        lambda scores: SimpleNamespace(item=lambda: 1),
    )

    def all_reduce(scores, op, group=None):
        nonlocal status_round
        if op == dist.ReduceOp.MIN:
            status_round += 1
            # apply(1), prepare(2): only the remote rank fails preparation.
            if status_round == 2:
                scores.values[0] = 0
            return
        assert scores.tolist() == [float("inf"), 1.0]

    monkeypatch.setattr(dist, "all_reduce", all_reduce)

    with pytest.warns(RuntimeWarning, match="compile-only prepare"):
        winner = autotune_module.autotune_knobs(
            Frontend(),
            lambda: launches.append(current),
            candidates,
            label="prepare-gate",
            warmup_iters=0,
            timed_iters=1,
            process_group=ep_group,
            expected_world_size=2,
            prepare_candidate=lambda: prepare_calls.append(current),
            discard_candidate=lambda: discard_calls.append(current),
        )

    assert winner == candidates[1]
    assert launches == ["good"]
    assert prepare_calls == ["compile-fails-remotely", "good", "good"]
    assert discard_calls == ["compile-fails-remotely", "good"]


def test_winner_prepare_failure_discards_and_prevents_record(monkeypatch):
    candidate = {"id": "winner"}
    prepare_calls = 0
    discard = mock.Mock()
    record = mock.Mock()

    monkeypatch.setattr(comm, "ensure_not_capturing", mock.Mock())
    monkeypatch.setattr(dist, "is_available", lambda: False)
    clock = iter([0.0, 1.0])
    monkeypatch.setattr(autotune_module.time, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(
        autotune_module.torch,
        "tensor",
        lambda values, **kwargs: _Scores(values),
    )
    monkeypatch.setattr(
        autotune_module.torch,
        "argmin",
        lambda scores: SimpleNamespace(item=lambda: 0),
    )

    def prepare():
        nonlocal prepare_calls
        prepare_calls += 1
        if prepare_calls == 2:
            raise ValueError("winner compile rejected")

    with pytest.raises(RuntimeError, match="winner preparation failed"):
        autotune_module.autotune_knobs(
            SimpleNamespace(apply_knobs=mock.Mock()),
            lambda: None,
            [candidate],
            label="winner-prepare",
            warmup_iters=0,
            timed_iters=1,
            prepare_candidate=prepare,
            discard_candidate=discard,
            on_winner=record,
        )

    assert discard.call_args_list == [mock.call(), mock.call()]
    record.assert_not_called()


def test_failed_winner_commit_rolls_back_before_record(monkeypatch):
    candidate = {"id": "winner"}
    rollback = mock.Mock()
    finalize = mock.Mock()
    record = mock.Mock()

    monkeypatch.setattr(comm, "ensure_not_capturing", mock.Mock())
    monkeypatch.setattr(dist, "is_available", lambda: False)
    clock = iter([0.0, 1.0])
    monkeypatch.setattr(autotune_module.time, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(
        autotune_module.torch,
        "tensor",
        lambda values, **kwargs: _Scores(values),
    )
    monkeypatch.setattr(
        autotune_module.torch,
        "argmin",
        lambda scores: SimpleNamespace(item=lambda: 0),
    )

    def fail_commit():
        raise ValueError("commit rejected")

    with pytest.raises(RuntimeError, match="winner commit failed"):
        autotune_module.autotune_knobs(
            SimpleNamespace(apply_knobs=mock.Mock()),
            lambda: None,
            [candidate],
            label="winner-commit",
            warmup_iters=0,
            timed_iters=1,
            commit_winner=fail_commit,
            rollback_winner=rollback,
            finalize_winner=finalize,
            on_winner=record,
        )

    rollback.assert_called_once_with()
    finalize.assert_not_called()
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
        (
            {"commit_winner": mock.Mock(), "rollback_winner": mock.Mock()},
            "must be provided together",
        ),
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


def test_winner_is_materialized_before_record(monkeypatch):
    candidate = {"id": "winner"}
    events = []
    frontend = SimpleNamespace(
        apply_knobs=lambda knobs: events.append(("apply", knobs)),
    )

    monkeypatch.setattr(comm, "ensure_not_capturing", mock.Mock())
    monkeypatch.setattr(dist, "is_available", lambda: False)
    clock = iter([0.0, 1.0])
    monkeypatch.setattr(autotune_module.time, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(
        autotune_module.torch,
        "tensor",
        lambda values, **kwargs: _Scores(values),
    )
    monkeypatch.setattr(
        autotune_module.torch,
        "argmin",
        lambda scores: SimpleNamespace(item=lambda: 0),
    )

    winner = autotune_module.autotune_knobs(
        frontend,
        lambda: events.append(("launch", None)),
        [candidate],
        label="materialize-order",
        warmup_iters=0,
        timed_iters=1,
        materialize_winner=lambda: events.append(("materialize", None)),
        on_winner=lambda knobs, score: events.append(("record", knobs, score)),
    )

    assert winner == candidate
    assert events == [
        ("apply", candidate),
        ("launch", None),
        ("apply", candidate),
        ("materialize", None),
        ("record", candidate, 1.0),
    ]


def test_winner_materialization_failure_prevents_record(monkeypatch):
    candidate = {"id": "winner"}
    frontend = SimpleNamespace(apply_knobs=mock.Mock())
    callback = mock.Mock()

    monkeypatch.setattr(comm, "ensure_not_capturing", mock.Mock())
    monkeypatch.setattr(dist, "is_available", lambda: False)
    clock = iter([0.0, 1.0])
    monkeypatch.setattr(autotune_module.time, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(
        autotune_module.torch,
        "tensor",
        lambda values, **kwargs: _Scores(values),
    )
    monkeypatch.setattr(
        autotune_module.torch,
        "argmin",
        lambda scores: SimpleNamespace(item=lambda: 0),
    )

    def fail_materialize():
        raise ValueError("compile rejected")

    with pytest.raises(RuntimeError, match="winner materialization failed"):
        autotune_module.autotune_knobs(
            frontend,
            lambda: None,
            [candidate],
            label="materialize-failure",
            warmup_iters=0,
            timed_iters=1,
            materialize_winner=fail_materialize,
            on_winner=callback,
        )

    callback.assert_not_called()
