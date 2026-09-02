"""CPU proof of the SM90 collective autotune scoring definition."""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest
import torch.distributed as dist

from flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel.shim import (
    autotune as autotune_module,
    comm,
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
