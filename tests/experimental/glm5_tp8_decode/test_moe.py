# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Run with torchrun --nproc-per-node=8 -m pytest on eight B300 GPUs."""

from dataclasses import replace
import weakref

import pytest
import torch

from . import _api, data
from . import moe_reference as ref
from ._moe_case import MoECase
from ._moe_checks import check_moe

pytestmark = pytest.mark.node_exclusive


@pytest.mark.parametrize(
    "weight_seed,cancellation",
    [
        (531017, False),
        (531018, False),
        (531019, False),
        (847223, False),
        (974189, True),
    ],
)
def test_moe_eager_graph_changed_inputs_and_q1_q4_alternation(
    collective, collective_watchdog, weight_seed, cancellation
):
    case = MoECase(collective, weight_seed, cancellation=cancellation)
    try:
        case.run(4, 1)
        case.run(1, 7)
        case.capture()
        for q_len, seed in ((4, 7), (1, 23), (4, 23), (1, 1), (4, 1), (1, 7)):
            expected = case.run(q_len, seed, graph=True)
            repeated = case.run(q_len, seed, graph=True, validate=False)
            assert ref.bitwise(repeated, expected), (
                "same-input graph replay changed the output"
            )
    finally:
        case.close()


def test_moe_independent_workspaces_destroy_and_recreate(
    collective, collective_watchdog
):
    first = MoECase(collective)
    second = None
    rebuilt = None
    try:
        first.run(4, 1)
        first.run(1, 7)
        first.capture()
        second = MoECase(collective, layer=first.layer, packed=first.packed)
        second.run(1, 23)
        second.run(4, 31)
        first.run(1, 41, graph=True)
        first.run(4, 43, graph=True)
        old_plan = second.plans[4]
        second.close()
        second = None
        with pytest.raises(RuntimeError, match="destroy|closed|invalid"):
            old_plan()
        rebuilt = MoECase(collective, layer=first.layer, packed=first.packed)
        rebuilt.run(4, 53)
        rebuilt.run(1, 59)
        first.run(4, 61, graph=True)
    finally:
        if rebuilt is not None:
            rebuilt.close()
        if second is not None:
            second.close()
        first.close()


def test_moe_cross_stream_workspace_ordering(collective, collective_watchdog):
    case = MoECase(collective)
    try:
        x_a = data.make_input(4, 101, collective.device)
        x_b = data.make_input(4, 103, collective.device)
        out_a, out_b = torch.empty_like(x_a), torch.empty_like(x_b)
        a = _api.prepare_moe(x_a, case.packed, out_a, case.ws)
        b = _api.prepare_moe(x_b, case.packed, out_b, case.ws)
        collective.sync()
        a()
        collective.sync()
        expected_a = check_moe(collective, a, case.weights, x_a, out_a)
        b()
        collective.sync()
        expected_b = check_moe(collective, b, case.weights, x_b, out_b)
        stream_a = torch.cuda.Stream(device=collective.device)
        stream_b = torch.cuda.Stream(device=collective.device)
        out_a.fill_(17)
        out_b.fill_(19)
        collective.sync()
        with torch.cuda.stream(stream_a):
            a()
        with torch.cuda.stream(stream_b):
            b()
        collective.sync()
        assert ref.bitwise(out_a.cpu(), expected_a)
        assert ref.bitwise(out_b.cpu(), expected_b)
    finally:
        case.close()


def test_moe_invalid_prepare_does_not_poison_workspace(collective, collective_watchdog):
    case = MoECase(collective)
    try:
        before = {
            q: int(plan._diagnostics["generation"]()) for q, plan in case.plans.items()
        }
        bad_x = torch.zeros((2, 6144), dtype=torch.bfloat16, device=collective.device)
        bad_out = torch.empty_like(bad_x)
        with pytest.raises((ValueError, RuntimeError), match="q_len|query|1.*4|shape"):
            _api.prepare_moe(bad_x, case.packed, bad_out, case.ws)
        collective.sync()
        assert {
            q: int(plan._diagnostics["generation"]()) for q, plan in case.plans.items()
        } == before
        case.run(4, 1)
        case.run(1, 7)
    finally:
        case.close()


def test_moe_temporary_plan_keeps_side_stream_inputs_alive(
    collective, collective_watchdog
):
    case = MoECase(collective)
    churn = []
    try:
        expected = case.run(4, 101)
        original_stream = torch.cuda.current_stream(collective.device)
        side_stream = torch.cuda.Stream(device=collective.device)
        # Loading the delay kernel must precede any peer-waiting launch.
        torch.cuda._sleep(1)
        collective.sync()
        temporary_x = data.make_input(4, 101, collective.device)
        temporary_weights = replace(
            case.packed,
            **{name: tensor.clone() for name, tensor in vars(case.packed).items()},
        )
        result = torch.empty_like(temporary_x)
        temporary_plan = _api.prepare_moe(
            temporary_x, temporary_weights, result, case.ws
        )
        allocations = [(tuple(temporary_x.shape), temporary_x.dtype)]
        allocations += [
            (tuple(tensor.shape), tensor.dtype)
            for tensor in vars(temporary_weights).values()
        ]
        x_owner, plan_owner = weakref.ref(temporary_x), weakref.ref(temporary_plan)
        side_stream.wait_stream(original_stream)
        with torch.cuda.stream(side_stream):
            torch.cuda._sleep(200_000_000)
            temporary_plan()
        # The output and workspace remain owned, but nothing in this scope keeps
        # the launched plan's input/weight tensors alive. The caching allocator
        # must defer reuse until the side-stream work has consumed them.
        del temporary_plan, temporary_x, temporary_weights
        assert x_owner() is None and plan_owner() is None
        with torch.cuda.stream(original_stream):
            for shape, dtype in allocations:
                allocation = torch.empty(shape, dtype=dtype, device=collective.device)
                allocation.zero_()
                churn.append(allocation)
        collective.sync()
        collective.checked(lambda: _assert_lifetime_result(result, expected))
    finally:
        case.close()


def _assert_lifetime_result(actual, expected):
    assert ref.bitwise(actual.cpu(), expected), (
        "temporary plan/input storage was reused before side-stream completion"
    )
