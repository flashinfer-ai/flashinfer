# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

"""Independent handles, CUDA Graph scheduling, and single-use receive hooks."""

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
    global_scales,
    make_case,
    mutate_payload,
    prepare_expert_outputs,
)


def test_one_rank_invalid_config_fails_collectively(distributed_device):
    from flashinfer.comm import RdmaMoeAlltoAll, RdmaMoeAlltoAllConfig

    with pytest.raises(RuntimeError, match="dtype|config|precision"):
        RdmaMoeAlltoAll(
            dist.group.WORLD,
            8,
            4096,
            288,
            config=RdmaMoeAlltoAllConfig(
                communication_dtype="invalid" if dist.get_rank() == 0 else "bf16"
            ),
        )
    dist.barrier()


def expect_rejection(call, message, device):
    error = None
    try:
        call()
    except (ValueError, RuntimeError, NotImplementedError) as exc:
        error = exc
    assert_all(
        error is not None and message in str(error),
        f"Expected rejection containing {message!r}; got {error!r}",
        device,
    )


@pytest.mark.parametrize("topk", (1, 8))
def test_retained_dispatch_handles(
    transport_buffer, transport_config, distributed_device, topk
):
    dtype, hidden = transport_config
    rank = dist.get_rank()
    first = make_case(5, hidden, topk, rank, distributed_device, dtype)
    second = make_case(
        3, hidden, topk, rank, distributed_device, dtype, "skew", seed=20260918
    )
    first_result = dispatch(transport_buffer, first)
    first_x, first_scales = prepare_expert_outputs(first_result, first)
    second_result = dispatch(transport_buffer, second)
    second_x, second_scales = prepare_expert_outputs(second_result, second)
    for case, result, x, scales in (
        (first, first_result, first_x, first_scales),
        (second, second_result, second_x, second_scales),
        (first, first_result, first_x, first_scales),
    ):
        check_combine(
            combine(transport_buffer, case, result, x, scales), transport_buffer, case
        )


def test_serial_dispatch_combine_graph(
    transport_buffer, transport_config, distributed_device
):
    dtype, hidden = transport_config
    rank = dist.get_rank()
    case = make_case(5, hidden, 8, rank, distributed_device, dtype)
    out = allocate_dispatch_out(case, transport_buffer)
    columns = ((torch.arange(hidden, device=distributed_device) % 31).float() - 15) / 16
    ids = torch.arange(EXPERTS, device=distributed_device)
    profiles = expert_output(columns[None].expand(EXPERTS, -1), ids)
    local = profiles[rank * (EXPERTS // WORLD) : (rank + 1) * (EXPERTS // WORLD)]
    x = (
        local[:, None, :]
        .expand(-1, transport_buffer.num_max_dispatch_tokens_per_rank * WORLD, -1)
        .contiguous()
    )
    scales = global_scales(x).contiguous() if dtype == "nvfp4" else None
    output = torch.empty(
        (case["tokens"], hidden), dtype=torch.bfloat16, device=distributed_device
    )

    def run():
        captured = dispatch(transport_buffer, case, out=out)
        actual = combine(transport_buffer, case, captured, x, scales, out=output)
        return captured, actual

    with captured_graph(run, distributed_device) as (graph, (captured, actual), _):
        original_routes = case["routes"].clone()
        for replay in range(3):
            case["routes"].copy_(
                (original_routes + replay * (EXPERTS // WORLD)) % EXPERTS
            )
            if replay == 1:
                case["routes"][..., 1::2] = -1
            mutate_payload(case)
            case["weights"].copy_(case["weights"].roll(1, dims=-1))
            dist.barrier()
            graph.replay()
            torch.cuda.synchronize()
            check_dispatch(captured, case)
            check_combine(actual, transport_buffer, case, profiles)


def test_gin_serial_even_dispatch_graph(
    transport_buffer, transport_config, distributed_device
):
    dtype, hidden = transport_config
    case = make_case(5, hidden, 8, dist.get_rank(), distributed_device, dtype)
    out = allocate_dispatch_out(case, transport_buffer)
    with captured_graph(
        lambda: dispatch(transport_buffer, case, out=out),
        distributed_device,
        capture_calls=2,
    ) as (graph, captured, stream):
        for _ in range(3):
            mutate_payload(case)
            dist.barrier()
            graph.replay()
            torch.cuda.synchronize()
            check_dispatch(captured, case)
    # Retire the graph before advancing its host slots with an eager operation.
    with torch.cuda.stream(stream):
        x, scales = prepare_expert_outputs(captured, case)
        check_combine(
            combine(transport_buffer, case, captured, x, scales),
            transport_buffer,
            case,
        )


def test_ordered_cross_stream_operations(
    transport_buffer, transport_config, distributed_device
):
    dtype, hidden = transport_config
    case = make_case(2, hidden, 8, dist.get_rank(), distributed_device, dtype)
    first_result = dispatch(transport_buffer, case)
    x, scales = prepare_expert_outputs(first_result, case)
    original_stream = torch.cuda.current_stream()
    other_stream = torch.cuda.Stream(device=distributed_device)
    other_stream.wait_stream(original_stream)
    with torch.cuda.stream(other_stream):
        other_result = dispatch(transport_buffer, case)
        check_dispatch(other_result, case)
    original_stream.wait_stream(other_stream)
    check_combine(
        combine(transport_buffer, case, first_result, x, scales),
        transport_buffer,
        case,
    )


def test_receive_hook_lifecycle(transport_buffer, transport_config, distributed_device):
    dtype, hidden = transport_config
    case = make_case(2, hidden, 8, dist.get_rank(), distributed_device, dtype)

    def send_dispatch():
        return transport_buffer.dispatch(
            case["input"],
            case["topk"],
            return_recv_hook=True,
            input_dtype="prequantized",
            x_scales=None if dtype == "bf16" else case["input_scales"],
        )

    pending = send_dispatch()
    expect_rejection(send_dispatch, "outstanding receive hook", distributed_device)
    expect_rejection(transport_buffer.close, "close", distributed_device)
    with torch.cuda.stream(torch.cuda.Stream(device=distributed_device)):
        expect_rejection(pending[-1], "stream", distributed_device)
    pending[-1]()
    expect_rejection(pending[-1], "only be invoked once", distributed_device)
    x, scales = prepare_expert_outputs(pending, case)

    def send_combine():
        return transport_buffer.combine(
            x,
            pending[3],
            return_recv_hook=True,
            topk_idx=case["topk"],
            topk_weights=case["topk_weights"],
            global_scale=scales,
        )

    combined = send_combine()
    expect_rejection(send_dispatch, "outstanding receive hook", distributed_device)
    expect_rejection(send_combine, "outstanding receive hook", distributed_device)
    expect_rejection(transport_buffer.close, "close", distributed_device)
    combined[-1]()
    expect_rejection(combined[-1], "only be invoked once", distributed_device)
    check_combine(combined[0], transport_buffer, case)
    torch.cuda.synchronize()
    transport_buffer.close()
    transport_buffer.close()
    expect_rejection(send_dispatch, "closed", distributed_device)
