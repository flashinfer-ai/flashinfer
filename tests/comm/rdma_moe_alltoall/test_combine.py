# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

"""Dynamic and dispatch-planned BF16 reduction with independent numerical references."""

from contextlib import contextmanager

import pytest
import torch
import torch.distributed as dist

from .reference import (
    EXPERTS,
    WORLD,
    allocate_dispatch_out,
    assert_all,
    check_combine,
    check_dispatch,
    captured_graph,
    combine,
    dispatch,
    expert_output,
    make_case,
    managed_buffer,
    prepare_expert_outputs,
)


@contextmanager
def partial_buffer(hidden=4096, capacity=128):
    from flashinfer.comm import RdmaMoeAlltoAll, RdmaMoeAlltoAllConfig

    buffer = RdmaMoeAlltoAll(
        dist.group.WORLD,
        capacity,
        hidden,
        EXPERTS,
        config=RdmaMoeAlltoAllConfig(
            communication_dtype="bf16",
            communication_mode="legacy",
            combine_reduction="bf16_local_partial",
        ),
    )
    with managed_buffer(buffer):
        yield buffer


def two_level_reference(case):
    """Independent tensor math with explicit casts and original route order."""
    source = case["packed"][case["rank"], : case["tokens"]].float()
    ids, weights = case["topk"], case["topk_weights"]
    common = torch.zeros_like(source)
    leaves = []
    for k in range(ids.shape[1]):
        x = expert_output(source, ids[:, k].clamp_min(0)).float()
        product = torch.mul(x, weights[:, k, None])
        leaves.append(product.to(torch.bfloat16).float())
        common = torch.where((ids[:, k] >= 0)[:, None], common + product, common)
    partials = []
    for owner in range(WORLD):
        local = torch.zeros_like(source)
        for k, leaf in enumerate(leaves):
            selected = (ids[:, k] >= 0) & (ids[:, k] // (EXPERTS // WORLD) == owner)
            local = torch.where(selected[:, None], local + leaf, local)
        partials.append(local.to(torch.bfloat16).float())
    final = torch.zeros_like(source)
    for k in range(ids.shape[1]):
        owner = ids[:, k].clamp_min(0) // (EXPERTS // WORLD)
        first = ids[:, k] >= 0
        for prior in range(k):
            first &= (ids[:, prior] < 0) | (
                ids[:, prior] // (EXPERTS // WORLD) != owner
            )
        for candidate, partial in enumerate(partials):
            selected = first & (owner == candidate)
            final = torch.where(selected[:, None], final + partial, final)
    return final.to(torch.bfloat16), common


def check_partial(actual, case):
    expected, common = two_level_reference(case)
    assert_all(
        actual.shape == expected.shape
        and torch.equal(actual.view(torch.int16), expected.view(torch.int16)),
        "Local-partial output differs bitwise from its explicit BF16 hierarchy",
        actual.device,
    )
    # Report, rather than conceal, error against the common full-FP32 task.
    error = actual.float() - common
    metrics = {
        "max_abs": error.abs().max().item() if error.numel() else 0.0,
        "relative_rmse": (
            (error.square().sum() / common.square().sum().clamp_min(1e-30))
            .sqrt()
            .item()
            if error.numel()
            else 0.0
        ),
    }
    assert_all(
        bool(torch.isfinite(actual).all()),
        "Nonfinite local partial output",
        actual.device,
    )
    return metrics


def arbitrary_weights(case):
    case["weights"].mul_(4).sub_(0.5)
    case["weights"][..., ::3] = 0.0
    case["weights"][..., 1::4] *= -1


@pytest.mark.parametrize("hooks", (False, True))
@pytest.mark.parametrize(
    "tokens,pattern",
    (
        (0, "uniform"),
        (1, "uniform"),
        (65, "uniform"),
        (128, "skew"),
        (5, "empty_rank"),
        (5, "invalid"),
    ),
)
def test_partial_routes_weights_and_hooks(
    distributed_device, hooks, tokens, pattern, record_property
):
    with partial_buffer() as buffer:
        case = make_case(
            tokens, 4096, 8, dist.get_rank(), distributed_device, "bf16", pattern
        )
        arbitrary_weights(case)
        for repeat in range(2):
            result = dispatch(buffer, case, hooks=hooks)
            x, scales = prepare_expert_outputs(result, case)
            if repeat:
                case["routes"].copy_(case["routes"].roll(3, dims=-1))
                case["weights"].copy_(case["weights"].roll(3, dims=-1))
            metrics = check_partial(
                combine(buffer, case, result, x, scales, hooks=hooks), case
            )
            record_property(
                f"common_fp32_error_rank{dist.get_rank()}_repeat{repeat}", metrics
            )
            case["packed"].neg_()


@pytest.mark.parametrize("hidden", (6144, 7168))
def test_partial_hidden_and_subset(distributed_device, hidden):
    with partial_buffer(hidden=hidden) as buffer:
        case = make_case(9, hidden, 7, dist.get_rank(), distributed_device, "bf16")
        arbitrary_weights(case)
        result = dispatch(buffer, case, hooks=True)
        x, scales = prepare_expert_outputs(result, case)
        subset = dict(case)
        subset["tokens"] = 5
        subset["topk"] = case["topk"][:5, [4, 1, 3]].contiguous()
        subset["topk_weights"] = case["topk_weights"][:5, [4, 1, 3]].contiguous()
        subset["topk"][::2, 1] = -1
        check_partial(combine(buffer, subset, result, x, scales, hooks=True), subset)


@pytest.mark.parametrize("hooks", (False, True))
def test_partial_dynamic_graph(distributed_device, hooks):
    with partial_buffer() as buffer:
        rank = dist.get_rank()
        case = make_case(65, 4096, 8, rank, distributed_device, "bf16")
        arbitrary_weights(case)
        original_routes = case["routes"].clone()
        out = allocate_dispatch_out(case, buffer)
        output = torch.empty(
            (65, 4096), dtype=torch.bfloat16, device=distributed_device
        )
        local_experts = torch.arange(
            rank * (EXPERTS // WORLD),
            (rank + 1) * (EXPERTS // WORLD),
            device=distributed_device,
        )[:, None]

        def run():
            result = dispatch(buffer, case, hooks=hooks, out=out)
            x = expert_output(result[0], local_experts)
            actual = combine(buffer, case, result, x, None, hooks=hooks, out=output)
            return result, actual

        with captured_graph(run, distributed_device) as (graph, (captured, actual), _):
            for repeat in range(3):
                case["packed"].neg_().add_(0.125)
                case["routes"].copy_(
                    (original_routes + repeat * (EXPERTS // WORLD)) % EXPERTS
                )
                if repeat == 2:
                    case["routes"][..., 1::2] = -1
                case["weights"].copy_(case["weights"].roll(1, dims=-1))
                case["weights"].neg_()
                dist.barrier()
                graph.replay()
                torch.cuda.synchronize()
                check_dispatch(captured, case)
                check_partial(actual, case)


@contextmanager
def planned_buffer(
    capacity=128,
    reduction="bf16_local_partial_planned",
    dispatch_mode=None,
    hidden=4096,
):
    from flashinfer.comm import RdmaMoeAlltoAll, RdmaMoeAlltoAllConfig

    buffer = RdmaMoeAlltoAll(
        dist.group.WORLD,
        capacity,
        hidden,
        EXPERTS,
        config=RdmaMoeAlltoAllConfig(
            communication_dtype="bf16",
            communication_mode="legacy",
            combine_reduction=reduction,
            dispatch_mode=dispatch_mode,
            combine_mode="legacy",
        ),
    )
    with managed_buffer(buffer):
        yield buffer


def snapshot_case(case):
    """Keep the oracle independent of later mutations of caller input storage."""
    saved = dict(case)
    for key in ("packed", "routes", "weights"):
        saved[key] = case[key].clone()
    rank, tokens = case["rank"], case["tokens"]
    saved["input"] = saved["packed"][rank, :tokens]
    saved["topk"] = saved["routes"][rank, :tokens]
    saved["topk_weights"] = saved["weights"][rank, :tokens]
    return saved


def dispatch_planned(buffer, case, hooks=False, out=None, bf16_input=False):
    result = buffer.dispatch(
        case["input"],
        case["topk"],
        topk_weights=case["topk_weights"],
        return_recv_hook=hooks,
        out=out,
        input_dtype="bf16" if bf16_input else "prequantized",
    )
    if hooks:
        result.recv_hook()
    return result


def combine_planned(buffer, result, x, hooks=False, out=None):
    combined = buffer.combine(x, result[3], return_recv_hook=hooks, out=out)
    if hooks:
        combined[-1]()
    return combined[0]


@pytest.mark.parametrize("hooks", (False, True))
@pytest.mark.parametrize(
    "tokens,topk,pattern",
    (
        (0, 8, "uniform"),
        (1, 1, "uniform"),
        (65, 8, "uniform"),
        (128, 8, "skew"),
        (5, 7, "empty_rank"),
        (5, 3, "invalid"),
    ),
)
def test_planned_snapshot_weights_routes_and_hooks(
    planned_mode, distributed_device, hooks, tokens, topk, pattern, record_property
):
    with planned_buffer(dispatch_mode=planned_mode) as buffer:
        case = make_case(
            tokens, 4096, topk, dist.get_rank(), distributed_device, "bf16", pattern
        )
        arbitrary_weights(case)
        if topk == 1:
            case["weights"].fill_(-0.75)
        saved = snapshot_case(case)
        result = dispatch_planned(
            buffer, case, hooks=hooks, bf16_input=tokens in (1, 65)
        )
        x, _ = prepare_expert_outputs(result, saved)

        # Dispatch has completed. Neither caller mutation nor two arena slots
        # may change a retained handle's immutable routing/weight snapshot.
        case["routes"].fill_(-1)
        case["weights"].fill_(11.0)
        case["packed"].neg_()
        for repeat in range(2):
            metrics = check_partial(combine_planned(buffer, result, x, hooks), saved)
            record_property(
                f"common_fp32_error_rank{dist.get_rank()}_repeat{repeat}", metrics
            )


@pytest.mark.parametrize("hooks", (False, True))
def test_planned_three_retained_handles(planned_mode, distributed_device, hooks):
    with planned_buffer(dispatch_mode=planned_mode) as buffer:
        retained = []
        for index, (tokens, pattern) in enumerate(
            ((65, "uniform"), (128, "skew"), (3, "invalid"))
        ):
            case = make_case(
                tokens,
                4096,
                8,
                dist.get_rank(),
                distributed_device,
                "bf16",
                pattern,
                seed=20261130 + index,
            )
            arbitrary_weights(case)
            saved = snapshot_case(case)
            result = dispatch_planned(buffer, case, hooks=hooks)
            x, _ = prepare_expert_outputs(result, saved)
            case["routes"].fill_(-1)
            case["weights"].zero_()
            retained.append((saved, result, x))

        # More than two dispatches wrap the transport arena; repeat the first
        # combine after consuming both later handles as an eager reuse check.
        for index in (0, 2, 1, 0):
            case, result, x = retained[index]

            check_partial(combine_planned(buffer, result, x, hooks), case)


@pytest.mark.parametrize("hidden", (6144, 7168))
def test_planned_hidden_templates(planned_mode, distributed_device, hidden):
    with planned_buffer(hidden=hidden, dispatch_mode=planned_mode) as buffer:
        case = make_case(9, hidden, 7, dist.get_rank(), distributed_device, "bf16")
        arbitrary_weights(case)
        saved = snapshot_case(case)
        result = dispatch_planned(buffer, case, hooks=True, bf16_input=True)
        x, _ = prepare_expert_outputs(result, saved)
        case["routes"].fill_(-1)
        case["weights"].zero_()

        check_partial(combine_planned(buffer, result, x, hooks=True), saved)


def planned_pair_runner(buffer, cases, hooks, device):
    rank = dist.get_rank()
    outs = [allocate_dispatch_out(case, buffer) for case in cases]
    outputs = [
        torch.empty((case["tokens"], 4096), dtype=torch.bfloat16, device=device)
        for case in cases
    ]
    local_experts = torch.arange(
        rank * (EXPERTS // WORLD),
        (rank + 1) * (EXPERTS // WORLD),
        device=device,
    )[:, None]

    def run_pair():
        results = [
            dispatch_planned(buffer, case, hooks=hooks, out=out)
            for case, out in zip(cases, outs, strict=True)
        ]
        xs = [expert_output(result[0], local_experts) for result in results]
        actual = [
            combine_planned(buffer, result, x, hooks=hooks, out=output)
            for result, x, output in zip(results, xs, outputs, strict=True)
        ]
        return results, actual

    return run_pair


@pytest.mark.parametrize("hooks", (False, True))
def test_planned_two_handles_dynamic_graph(planned_mode, distributed_device, hooks):
    # Exercise the other dispatch implementation and its receive-hook finalizer.
    with planned_buffer(capacity=8, dispatch_mode=planned_mode) as buffer:
        rank = dist.get_rank()
        cases = [
            make_case(
                tokens, 4096, 8, rank, distributed_device, "bf16", seed=20261200 + i
            )
            for i, tokens in enumerate((5, 3))
        ]
        for case in cases:
            arbitrary_weights(case)
        original_routes = [case["routes"].clone() for case in cases]
        run_pair = planned_pair_runner(buffer, cases, hooks, distributed_device)

        with captured_graph(run_pair, distributed_device) as (
            graph,
            (captured, actual),
            _,
        ):
            for replay in range(3):
                for index, case in enumerate(cases):
                    case["packed"].neg_().add_(0.125)
                    case["routes"].copy_(
                        (original_routes[index] + replay * (EXPERTS // WORLD)) % EXPERTS
                    )
                    if replay == 1:
                        case["routes"][..., 1::2] = -1
                    elif replay == 2 and index == 1:
                        case["routes"].fill_(-1)
                    case["weights"].copy_(case["weights"].roll(1, dims=-1)).neg_()
                dist.barrier()
                graph.replay()
                torch.cuda.synchronize()
                for case, result, output in zip(cases, captured, actual, strict=True):
                    check_dispatch(result, case)

                    check_partial(output, case)


def test_planned_out_reuse_and_pending_hook(distributed_device, planned_mode):
    with planned_buffer(capacity=8, dispatch_mode=planned_mode) as buffer:
        case = make_case(5, 4096, 8, dist.get_rank(), distributed_device, "bf16")
        arbitrary_weights(case)
        out = allocate_dispatch_out(case, buffer)
        first = buffer.dispatch(
            case["input"],
            case["topk"],
            topk_weights=case["topk_weights"],
            out=out,
            return_recv_hook=True,
        )
        with pytest.raises(RuntimeError, match="outstanding receive hook"):
            dispatch_planned(buffer, case)
        first.recv_hook()
        with pytest.raises(RuntimeError, match="only be invoked once"):
            first.recv_hook()
        case["routes"].copy_((case["routes"] + EXPERTS // WORLD) % EXPERTS)
        case["weights"].neg_()
        second = dispatch_planned(buffer, case, out=out)
        # Explicit out reuse overwrites old expert output/src/ranges. Consume
        # only the newest pairing; retained-handle tests use independent outputs.
        x, _ = prepare_expert_outputs(second, case)
        check_partial(combine_planned(buffer, second, x, hooks=True), case)


def test_planned_api_guards_and_default_fp32(distributed_device, planned_mode):
    case = make_case(5, 4096, 8, dist.get_rank(), distributed_device, "bf16")
    arbitrary_weights(case)
    with planned_buffer(capacity=8, dispatch_mode=planned_mode) as buffer:
        for weights in (
            None,
            case["topk_weights"].to(torch.bfloat16),
            case["topk_weights"][:, :3].contiguous(),
        ):
            with pytest.raises(ValueError, match="topk_weights"):
                buffer.dispatch(case["input"], case["topk"], topk_weights=weights)
        result = dispatch_planned(buffer, case)
        x, _ = prepare_expert_outputs(result, case)
        with pytest.raises(ValueError, match="planned|snapshot|dispatch"):
            combine(buffer, case, result, x, None)
        with pytest.raises(ValueError, match="handle"):
            buffer.combine(x, tuple(result.handle))
        check_partial(combine_planned(buffer, result, x), case)
    with planned_buffer(
        capacity=8, reduction="fp32", dispatch_mode=planned_mode
    ) as buffer:
        with pytest.raises(ValueError, match="topk_weights"):
            dispatch_planned(buffer, case)
        result = dispatch(buffer, case)
        x, scales = prepare_expert_outputs(result, case)
        with pytest.raises(ValueError, match="topk|routes|weights"):
            buffer.combine(x, result.handle)
        case["weights"].neg_()
        check_combine(combine(buffer, case, result, x, scales), buffer, case)
