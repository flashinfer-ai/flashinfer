# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

"""Dispatch routing, capacity boundaries, and NVFP4 rank packets."""

import pytest
import torch
import torch.distributed as dist

from .reference import (
    EXPERTS,
    WORLD,
    _same_peer_routes,
    allocate_dispatch_out,
    check_combine,
    check_dispatch,
    captured_graph,
    combine,
    dispatch,
    make_case,
    managed_buffer,
    set_source_lengths,
    prepare_expert_outputs,
)


def _require_legacy(request):
    if request.config.getoption("--rdma-moe-mode") != "legacy":
        pytest.skip("Use --rdma-moe-mode legacy to exercise reference-header dispatch")


@pytest.mark.parametrize("tokens", (1, 9))
@pytest.mark.parametrize("topk", range(2, 8))
@pytest.mark.parametrize("hooks", (False, True))
def test_bf16_rank_dedup_capacity_boundary(
    transport_factory, distributed_device, request, tokens, topk, hooks
):
    _require_legacy(request)
    with managed_buffer(transport_factory("bf16", 4096, capacity=9)) as buffer:
        case = make_case(
            tokens, 4096, topk, dist.get_rank(), distributed_device, "bf16"
        )
        out = allocate_dispatch_out(case, buffer)
        combined = torch.empty(
            (tokens, 4096), dtype=torch.bfloat16, device=distributed_device
        )
        for phase in range(2):
            _same_peer_routes(case, phase)
            case["packed"].neg_().add_(0.125 * (phase + 1))
            result = dispatch(buffer, case, hooks, out)
            expert_x, scales = prepare_expert_outputs(result, case)
            check_combine(
                combine(buffer, case, result, expert_x, scales, hooks, combined),
                buffer,
                case,
            )
        # K=2 first exercises a real follower, then a leading negative and
        # one valid route. K>=3 still has followers after the leading negative.


@pytest.mark.parametrize("hidden", (6144, 7168))
@pytest.mark.parametrize("hooks", (False, True))
def test_bf16_rank_dedup_wide_graph(
    transport_factory, distributed_device, request, hidden, hooks
):
    _require_legacy(request)
    with managed_buffer(transport_factory("bf16", hidden, capacity=128)) as buffer:
        case = make_case(128, hidden, 7, dist.get_rank(), distributed_device, "bf16")
        _same_peer_routes(case, 0)
        out = allocate_dispatch_out(case, buffer)
        with captured_graph(
            lambda: dispatch(buffer, case, hooks, out),
            distributed_device,
            capture_calls=2,
        ) as (graph, captured, _):
            for phase in range(2):
                _same_peer_routes(case, phase)
                case["packed"].neg_().add_(0.125 * (phase + 1))
                dist.barrier()
                graph.replay()
                torch.cuda.synchronize(distributed_device)
                check_dispatch(captured, case)


def _routes(case, phase):
    k = case["routes"].shape[-1]
    for source in range(WORLD):
        owner = (source + phase + 1) % WORLD
        routes = torch.arange(k, dtype=torch.int32, device=case["routes"].device)
        case["routes"][source].copy_(
            (routes + owner * (EXPERTS // WORLD)).expand_as(case["routes"][source])
        )
    if phase == 1:
        case["routes"].fill_(-1)
    elif phase == 2:
        case["routes"][..., 0] = -1
    elif phase == 4:
        # Mixed self/remote owners and different first-owner K positions.
        tokens = case["routes"].shape[1]
        token = torch.arange(tokens, dtype=torch.int32, device=case["routes"].device)[
            :, None
        ]
        route = torch.arange(k, dtype=torch.int32, device=case["routes"].device)[
            None, :
        ]
        for source in range(WORLD):
            case["routes"][source].copy_(
                ((token + route + source) % WORLD) * (EXPERTS // WORLD) + route
            )
    elif phase == 5:
        # Saturate the same experts across all sources, including self traffic.
        case["routes"].copy_(
            torch.arange(k, dtype=torch.int32, device=case["routes"].device).expand_as(
                case["routes"]
            )
        )


def _mutate(case):
    case["packed"].bitwise_xor_(0x88)
    case["scales"].copy_(case["scales"].roll(1, dims=-1))


def _check_rank_reuse(buffer, capacity):
    info = buffer.backend_info()
    packet = capacity > 32 and WORLD in (4, 8)
    assert info["dispatch_nvfp4_rank_packet"] == packet
    assert info["dispatch_nvfp4_rank_dedup"] == (capacity > 32 and not packet)


@pytest.mark.parametrize("capacity", (32, 33))
@pytest.mark.parametrize("full", (False, True))
@pytest.mark.parametrize("topk", (2, 8))
@pytest.mark.parametrize("hooks", (False, True))
def test_nvfp4_rank_dedup_threshold(
    transport_factory, distributed_device, request, capacity, full, topk, hooks
):
    _require_legacy(request)
    tokens = capacity if full else 1
    with managed_buffer(transport_factory("nvfp4", 4096, capacity=capacity)) as buffer:
        _check_rank_reuse(buffer, capacity)
        case = make_case(
            tokens, 4096, topk, dist.get_rank(), distributed_device, "nvfp4"
        )
        out = allocate_dispatch_out(case, buffer)
        combined = torch.empty(
            (tokens, 4096), dtype=torch.bfloat16, device=distributed_device
        )
        for phase in range(6):
            _routes(case, phase)
            _mutate(case)
            result = dispatch(buffer, case, hooks, out)
            expert_x, scales = prepare_expert_outputs(result, case)
            check_combine(
                combine(buffer, case, result, expert_x, scales, hooks, combined),
                buffer,
                case,
            )


@pytest.mark.parametrize("hooks", (False, True))
def test_nvfp4_rank_dedup_zero_tokens(
    transport_factory, distributed_device, request, hooks
):
    _require_legacy(request)
    with managed_buffer(transport_factory("nvfp4", 4096, capacity=33)) as buffer:
        _check_rank_reuse(buffer, 33)
        for tokens in (1, 0, 1):
            case = make_case(
                tokens, 4096, 8, dist.get_rank(), distributed_device, "nvfp4"
            )
            _routes(case, 0)
            result = dispatch(buffer, case, hooks)
            expert_x, scales = prepare_expert_outputs(result, case)
            check_combine(
                combine(buffer, case, result, expert_x, scales, hooks), buffer, case
            )


@pytest.mark.parametrize("hidden", (6144, 7168))
@pytest.mark.parametrize("hooks", (False, True))
def test_nvfp4_rank_dedup_wide_graph(
    transport_factory, distributed_device, request, hidden, hooks
):
    _require_legacy(request)
    with managed_buffer(transport_factory("nvfp4", hidden, capacity=128)) as buffer:
        _check_rank_reuse(buffer, 128)
        case = make_case(128, hidden, 7, dist.get_rank(), distributed_device, "nvfp4")
        _routes(case, 0)
        out = allocate_dispatch_out(case, buffer)
        with captured_graph(
            lambda: dispatch(buffer, case, hooks, out),
            distributed_device,
            capture_calls=2,
        ) as (graph, captured, _):
            for phase in range(6):
                _routes(case, phase)
                _mutate(case)
                dist.barrier()
                graph.replay()
                torch.cuda.synchronize(distributed_device)
                check_dispatch(captured, case)


@pytest.mark.parametrize("hooks", (False, True))
@pytest.mark.parametrize("scenario", ("unequal", "hot_full"))
def test_nvfp4_rank_packet_source_ranges(
    transport_factory, distributed_device, request, hooks, scenario
):
    _require_legacy(request)
    capacity = 33
    with managed_buffer(transport_factory("nvfp4", 4096, capacity=capacity)) as buffer:
        _check_rank_reuse(buffer, capacity)
        for empty in (False, True, False):
            case = make_case(
                capacity, 4096, 8, dist.get_rank(), distributed_device, "nvfp4"
            )
            _routes(case, 5 if scenario == "hot_full" else 4)
            lengths = (
                [capacity] * WORLD
                if scenario == "hot_full"
                else [0, 1, 17, capacity] + [2] * (WORLD - 4)
            )
            if empty:
                lengths = [0] * WORLD
            set_source_lengths(case, lengths)
            result = dispatch(buffer, case, hooks)
            expert_x, scales = prepare_expert_outputs(result, case)
            check_combine(
                combine(buffer, case, result, expert_x, scales, hooks), buffer, case
            )
