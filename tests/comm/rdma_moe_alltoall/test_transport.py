# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

"""RDMA MoE dispatch/combine and BF16-input transport correctness."""

import pytest
import torch
import torch.distributed as dist

from .reference import (
    DISPATCH_GLOBAL_SCALE,
    EXPERTS,
    allocate_dispatch_out,
    assert_all,
    check_combine,
    check_dispatch,
    captured_graph,
    combine,
    dispatch,
    make_case,
    mutate_payload,
    prepare_expert_outputs,
)


@pytest.mark.parametrize("tokens", (0, 1, 8))
@pytest.mark.parametrize("topk", (1, 8))
@pytest.mark.parametrize("pattern", ("uniform", "skew", "empty_rank", "invalid"))
@pytest.mark.parametrize("hooks", (False, True))
def test_dispatch_combine(
    transport_buffer, transport_config, distributed_device, tokens, topk, pattern, hooks
):
    dtype, hidden = transport_config
    case = make_case(
        tokens, hidden, topk, dist.get_rank(), distributed_device, dtype, pattern
    )
    out = allocate_dispatch_out(case, transport_buffer)
    combine_out = torch.empty(
        (case["tokens"], hidden), dtype=torch.bfloat16, device=distributed_device
    )
    for _ in range(2):
        result = dispatch(transport_buffer, case, hooks, out)
        x, scales = prepare_expert_outputs(result, case)
        actual = combine(transport_buffer, case, result, x, scales, hooks, combine_out)
        check_combine(actual, transport_buffer, case)
        assert_all(
            actual.data_ptr() == combine_out.data_ptr(),
            "combine did not reuse the provided output",
            distributed_device,
        )
        mutate_payload(case)


def test_dispatch_handle_provenance(
    transport_buffer, transport_factory, transport_config, distributed_device
):
    dtype, hidden = transport_config
    case = make_case(2, hidden, 8, dist.get_rank(), distributed_device, dtype)
    result = dispatch(transport_buffer, case)
    x, scales = prepare_expert_outputs(result, case)
    with pytest.raises(ValueError, match="dispatch handle produced by this buffer"):
        transport_buffer.combine(
            x,
            tuple(result[3]),
            topk_idx=case["topk"],
            topk_weights=case["topk_weights"],
            global_scale=scales,
        )
    other = transport_factory(dtype, hidden)
    try:
        with pytest.raises(ValueError, match="dispatch handle produced by this buffer"):
            other.combine(
                x,
                result[3],
                topk_idx=case["topk"],
                topk_weights=case["topk_weights"],
                global_scale=scales,
            )
    finally:
        other.close()
    check_combine(
        combine(transport_buffer, case, result, x, scales), transport_buffer, case
    )


def test_metadata_validation(transport_buffer, transport_config, distributed_device):
    dtype, hidden = transport_config
    case = make_case(2, hidden, 8, dist.get_rank(), distributed_device, dtype)
    with pytest.raises(ValueError, match="dtype|precision"):
        transport_buffer.dispatch(case["input"], case["topk"], input_dtype="invalid")
    with pytest.raises((ValueError, RuntimeError), match="capacity|Capacity|tokens"):
        oversized = make_case(9, hidden, 8, dist.get_rank(), distributed_device, dtype)
        dispatch(transport_buffer, oversized)
    with pytest.raises((ValueError, RuntimeError), match="topk|shape|token|row|size"):
        transport_buffer.dispatch(
            case["input"],
            case["topk"][:1],
            input_dtype="prequantized",
            x_scales=None if dtype == "bf16" else case["input_scales"],
        )
    result = dispatch(transport_buffer, case)
    x, scales = prepare_expert_outputs(result, case)
    check_combine(
        combine(transport_buffer, case, result, x, scales), transport_buffer, case
    )


def refresh_expected(case, buffer, source):
    """An independent quantizer supplies exact encoded transport expectations."""
    dtype = buffer.communication_dtype
    if dtype == "bf16":
        packed = source
        scales = torch.empty(
            (*source.shape[:-1], 0), dtype=torch.uint8, device=source.device
        )
    elif dtype == "fp8":
        packed, scales = buffer.quantize_bf16_to_fp8(source)
    else:
        global_scale = torch.full(
            (*source.shape[:-1], 1),
            DISPATCH_GLOBAL_SCALE,
            dtype=torch.float32,
            device=source.device,
        )
        packed, scales = buffer.quantize_bf16_to_nvfp4(source, global_scale)
    case.update(dtype=dtype, packed=packed, scales=scales)
    rank, tokens = case["rank"], case["tokens"]
    case["input"] = packed[rank, :tokens]
    case["input_scales"] = scales[rank, :tokens]


def make_bf16_case(buffer, tokens, hidden, device, pattern="uniform"):
    case = make_case(tokens, hidden, 8, dist.get_rank(), device, "bf16", pattern)
    source = case["packed"]
    refresh_expected(case, buffer, source)
    return case, source


@pytest.mark.parametrize("hooks", (False, True))
@pytest.mark.parametrize("pattern", ("uniform", "skew", "empty_rank", "invalid"))
def test_bf16_dispatch_and_combine(
    transport_buffer, transport_config, distributed_device, pattern, hooks
):
    _, hidden = transport_config
    case, source = make_bf16_case(
        transport_buffer, 5, hidden, distributed_device, pattern
    )
    out = allocate_dispatch_out(case, transport_buffer)
    for _ in range(2):
        result = transport_buffer.dispatch(
            source[case["rank"], : case["tokens"]],
            case["topk"],
            input_dtype="bf16",
            return_recv_hook=hooks,
            out=out,
            x_scales=None,
        )
        if hooks:
            result[-1]()
        check_dispatch(result, case)
        assert_all(
            result[0].data_ptr() == out[0].data_ptr(),
            "BF16-input dispatch did not reuse the supplied receive tensor",
            distributed_device,
        )
        expert_x, scales = prepare_expert_outputs(result, case)
        check_combine(
            combine(transport_buffer, case, result, expert_x, scales),
            transport_buffer,
            case,
        )
        source.neg_().add_(0.125)
        refresh_expected(case, transport_buffer, source)


def test_bf16_zero_token_dispatch(
    transport_buffer, transport_config, distributed_device
):
    _, hidden = transport_config
    case, source = make_bf16_case(transport_buffer, 0, hidden, distributed_device)
    result = transport_buffer.dispatch(source[case["rank"]], case["topk"])
    check_dispatch(result, case)


def test_bf16_dispatch_graph_requantizes_changed_source(
    transport_buffer, transport_config, distributed_device
):
    _, hidden = transport_config
    case, source = make_bf16_case(transport_buffer, 5, hidden, distributed_device)
    out = allocate_dispatch_out(case, transport_buffer)
    original_routes = case["routes"].clone()

    def run():
        result = transport_buffer.dispatch(
            source[case["rank"], : case["tokens"]],
            case["topk"],
            return_recv_hook=True,
            out=out,
        )
        result[-1]()
        return result

    with captured_graph(run, distributed_device, capture_calls=2) as (
        graph,
        captured,
        _,
    ):
        for replay in range(3):
            source.neg_().add_(0.125 * (replay + 1))
            case["routes"].copy_(
                (original_routes + (replay + 1) * (EXPERTS // dist.get_world_size()))
                % EXPERTS
            )
            if replay == 2:
                case["routes"].fill_(-1)
            refresh_expected(case, transport_buffer, source)
            dist.barrier()
            graph.replay()
            torch.cuda.synchronize()
            check_dispatch(captured, case)


def test_bf16_dispatch_rejects_metadata_before_quantizing(
    transport_buffer, transport_config, distributed_device
):
    _, hidden = transport_config
    case, source = make_bf16_case(transport_buffer, 2, hidden, distributed_device)
    x = source[case["rank"]]

    with pytest.raises(ValueError, match="scales"):
        transport_buffer.dispatch(
            x,
            case["topk"],
            input_dtype="bf16",
            x_scales=torch.empty((2, 0), dtype=torch.uint8, device=distributed_device),
        )
    with pytest.raises(ValueError, match="topk|dtype"):
        transport_buffer.dispatch(x, case["topk"].long())
    out = list(allocate_dispatch_out(case, transport_buffer))
    storage_bytes = max(
        x.numel() * x.element_size(), out[0].numel() * out[0].element_size()
    )
    storage = torch.empty(storage_bytes, dtype=torch.uint8, device=distributed_device)
    alias_x = storage[: x.numel() * x.element_size()].view(torch.bfloat16)
    alias_x = alias_x.view_as(x)
    out[0] = storage.view(out[0].dtype).view_as(out[0])
    with pytest.raises(ValueError, match="overlap"):
        transport_buffer.dispatch(alias_x, case["topk"], out=out)
    if transport_buffer.communication_dtype == "nvfp4":
        for value in (0.0, -1.0, float("nan"), float("inf")):
            with pytest.raises(ValueError, match="scale"):
                transport_buffer.dispatch(x, case["topk"], nvfp4_global_scale=value)
    check_dispatch(transport_buffer.dispatch(x, case["topk"]), case)
