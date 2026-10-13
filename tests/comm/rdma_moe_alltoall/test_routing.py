# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

"""Unequal sources, contiguous expert ranges, empty rounds, and Graph reuse."""

import pytest
import torch
import torch.distributed as dist

from .reference import (
    EXPERTS,
    WORLD,
    _same_peer_routes,
    allocate_dispatch_out,
    check_dispatch,
    captured_graph,
    make_case,
    set_source_lengths,
    prepare_expert_outputs,
)
from .test_combine import (
    arbitrary_weights,
    check_partial,
    combine_planned,
    dispatch_planned,
    planned_buffer,
    planned_pair_runner,
    snapshot_case,
)


def _mixed_routes(case):
    # Four distinct experts on owner0, two on owner1, then one each on owners2
    # and3. EP4/8 both exercise self/remote, shared packets and unequal fanout.
    local = EXPERTS // WORLD
    ids = (0, 1, 2, 3, local, local + 1, 2 * local, 3 * local)
    case["routes"].copy_(
        torch.tensor(ids, dtype=torch.int32, device=case["input"].device).expand_as(
            case["routes"]
        )
    )


def _case(capacity, lengths, device, hot=False, seed=20261210):
    case = make_case(capacity, 4096, 8, dist.get_rank(), device, "bf16", seed=seed)
    arbitrary_weights(case)
    if hot:
        # Every source contributes C rows to each of the same eight experts;
        # pair reservations must be disjoint and fill exactly WORLD*C rows.
        case["routes"].copy_(
            torch.arange(8, dtype=torch.int32, device=device).expand_as(case["routes"])
        )
    else:
        _mixed_routes(case)
    set_source_lengths(case, lengths)
    return case


def _check_round(buffer, case, hooks):
    saved = snapshot_case(case)
    result = dispatch_planned(buffer, case, hooks=hooks, bf16_input=True)
    # This checks every per-source contiguous range, exact source-ID set and
    # every expanded payload, independently of the combine oracle.
    x, _ = prepare_expert_outputs(result, saved)

    case["routes"].fill_(-1)
    case["weights"].neg_().add_(3.0)
    actual = combine_planned(buffer, result, x, hooks=hooks)
    return check_partial(actual, saved)


@pytest.mark.parametrize("hooks", (False, True))
@pytest.mark.parametrize(
    "scenario", ("unequal", "single_full", "hot_full", "tail_packets")
)
def test_fanout_source_lengths_and_saturated_ranges(
    distributed_device, hooks, scenario, record_property
):
    capacity = 128 if scenario == "tail_packets" else 9
    if scenario == "unequal":
        lengths = [0, 1, 8, 9] + [2] * (WORLD - 4)
    elif scenario == "single_full":
        lengths = [capacity if source == 2 else 0 for source in range(WORLD)]
    elif scenario == "hot_full":
        lengths = [capacity] * WORLD
    else:
        lengths = [0, 1, 33, 65] + [7] * (WORLD - 4)
    with planned_buffer(capacity=capacity, dispatch_mode="coalesced") as buffer:
        case = _case(capacity, lengths, distributed_device, hot=scenario == "hot_full")
        metrics = _check_round(buffer, case, hooks)
        record_property(f"common_fp32_error_rank{dist.get_rank()}", metrics)


@pytest.mark.parametrize("hooks", (False, True))
def test_fanout_actual_empty_round_then_recover(distributed_device, hooks):
    with planned_buffer(capacity=9, dispatch_mode="coalesced") as buffer:
        for phase, lengths in enumerate(([9] * WORLD, [0] * WORLD, [9] * WORLD)):
            case = _case(
                9, lengths, distributed_device, hot=True, seed=20261220 + phase
            )
            _check_round(buffer, case, hooks)


@pytest.mark.parametrize("hooks", (False, True))
def test_fanout_two_handle_graph_zero_counts_then_recover(distributed_device, hooks):
    with planned_buffer(capacity=9, dispatch_mode="coalesced") as buffer:
        lengths = [0, 1, 8, 9] + [2] * (WORLD - 4)
        cases = [
            _case(
                9, lengths, distributed_device, hot=bool(index), seed=20261230 + index
            )
            for index in range(2)
        ]
        routes = [case["routes"].clone() for case in cases]
        weights = [case["weights"].clone() for case in cases]
        run_pair = planned_pair_runner(buffer, cases, hooks, distributed_device)

        with captured_graph(run_pair, distributed_device) as (
            graph,
            (captured, actual),
            _,
        ):
            # Fixed shapes: the middle replay has no packets from any
            # source. The final replay must rebuild both sets of cursors.
            for phase in range(3):
                for index, case in enumerate(cases):
                    case["packed"].neg_().add_(0.125 * (phase + 1))
                    case["routes"].copy_(routes[index])
                    if phase == 1:
                        case["routes"].fill_(-1)
                    case["weights"].copy_(weights[index]).mul_((-1.0) ** phase)
                dist.barrier()
                graph.replay()
                torch.cuda.synchronize()
                for case, result, output in zip(cases, captured, actual, strict=True):
                    check_dispatch(result, case)

                    check_partial(output, case)


@pytest.mark.parametrize("tokens", (1, 9))
@pytest.mark.parametrize("topk", range(2, 8))
@pytest.mark.parametrize("hooks", (False, True))
def test_planned_rank_major_boundary(
    distributed_device, tokens, topk, hooks, record_property
):
    with planned_buffer(capacity=9, dispatch_mode="coalesced") as buffer:
        case = make_case(
            tokens, 4096, topk, dist.get_rank(), distributed_device, "bf16"
        )
        arbitrary_weights(case)
        original_weights = case["weights"].clone()
        out = allocate_dispatch_out(case, buffer)
        combined = torch.empty(
            (tokens, 4096), dtype=torch.bfloat16, device=distributed_device
        )
        for phase in range(2):
            # Every source targets distinct experts on one peer: per-expert
            # counts reach capacity at T=9, with multiple expert routes per destination.
            # Phase1 moves the owner and makes route0 invalid. K2 then has
            # only one valid contribution; K>=3 retains multiple expert routes to that destination.
            _same_peer_routes(case, phase)
            case["packed"].neg_().add_(0.125 * (phase + 1))
            case["weights"].copy_(original_weights).neg_().add_(0.125 * (phase + 1))
            saved = snapshot_case(case)
            result = dispatch_planned(
                buffer, case, hooks=hooks, out=out, bf16_input=bool(phase)
            )
            check_dispatch(result, saved)

            expert_x, _ = prepare_expert_outputs(result, saved)
            case["routes"].fill_(-1)
            case["weights"].fill_(17.0)
            # Explicit output reuse consumes only the current handle and its
            # expert outputs; the independent frozen plan supplies C weights.
            actual = combine_planned(
                buffer, result, expert_x, hooks=hooks, out=combined
            )
            metrics = check_partial(actual, saved)
            record_property(
                f"common_fp32_error_rank{dist.get_rank()}_phase{phase}", metrics
            )
