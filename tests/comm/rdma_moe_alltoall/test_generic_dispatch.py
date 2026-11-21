# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0

"""Opaque row transport and independent dispatch/combine codecs."""

import math

import pytest
import torch
import torch.distributed as dist

from .reference import (
    EXPERTS,
    WORLD,
    captured_graph,
    check_dispatch,
    decode,
    managed_buffer,
)


def routes(device, tokens=3, topk=2):
    # Distinct destinations with both same-rank fanout and cross-rank traffic.
    idx = torch.arange(WORLD * tokens * topk, device=device).reshape(
        WORLD, tokens, topk
    )
    idx = (idx * 37 + 1).remainder(EXPERTS).to(torch.int32)
    idx[0, -1].fill_(-1)
    return idx


@pytest.mark.parametrize("capacity,recv_hook", [(4, True), (33, True), (33, False)])
@pytest.mark.parametrize(
    "shape,dtype,scale_shape,scale_dtype,combine_dtype",
    [
        ((1,), torch.uint8, (0,), torch.uint8, "bf16"),
        ((13,), torch.uint8, (3,), torch.uint8, "bf16"),
        ((2, 7), torch.bfloat16, (3,), torch.int32, "nvfp4"),
        ((5120,), torch.uint8, (160,), torch.uint8, "bf16"),
        ((2560,), torch.uint8, (160,), torch.uint8, "nvfp4"),
        ((), torch.int32, (0,), torch.uint8, "bf16"),
    ],
)
def test_raw_rows_graph_and_independent_combine(
    distributed_device,
    shape,
    dtype,
    scale_shape,
    scale_dtype,
    combine_dtype,
    capacity,
    recv_hook,
):
    from flashinfer.comm import (
        RdmaDispatchLayout,
        RdmaMoeAlltoAll,
        RdmaMoeAlltoAllConfig,
    )

    device, rank, hidden = distributed_device, dist.get_rank(), 5120
    # Existing output metadata requires WORLD * capacity to be divisible by 4.
    alignment = 4 // math.gcd(WORLD, 4)
    capacity = (capacity + alignment - 1) // alignment * alignment
    tokens, topk_size = capacity - 1, 8 if capacity > 32 else 2
    data = torch.arange(WORLD * tokens * max(1, math.prod(shape)) + 1, device=device)
    data = data.remainder(97).to(dtype)[1:].reshape(WORLD, tokens, *shape)
    if dtype == torch.bfloat16:
        byte_view = data.view(torch.uint8)
        byte_view.copy_(
            torch.arange(byte_view.numel(), device=device)
            .to(torch.uint8)
            .reshape(byte_view.shape)
        )
    scales = torch.arange(WORLD * tokens * math.prod(scale_shape) + 1, device=device)
    scales = (
        scales.remainder(113).to(scale_dtype)[1:].reshape(WORLD, tokens, *scale_shape)
    )
    topk = routes(device, tokens, topk_size)
    empty_source = not shape
    if empty_source:
        topk[0].fill_(-1)
    local_tokens = 0 if empty_source and rank == 0 else tokens
    config = RdmaMoeAlltoAllConfig(
        communication_dtype=combine_dtype,
        dispatch_layout=RdmaDispatchLayout(
            data[0, 0].numel() * data.element_size(),
            scales[0, 0].numel() * scales.element_size(),
        ),
    )
    with managed_buffer(
        RdmaMoeAlltoAll(dist.group.WORLD, capacity, hidden, EXPERTS, config=config)
    ) as buf:
        info = buf.backend_info()
        assert info["dispatch_rank_packet"] == (capacity > 32)
        assert info["dispatch_nvfp4_rank_packet"] is False
        expert_values = torch.ones(
            (EXPERTS // WORLD, WORLD * capacity, hidden),
            dtype=torch.bfloat16,
            device=device,
        )
        gains = (torch.arange(EXPERTS, device=device).remainder(3) + 1).float() / 2
        expert_values.mul_(
            gains[
                rank * (EXPERTS // WORLD) : (rank + 1) * (EXPERTS // WORLD), None, None
            ]
        )
        global_scale = (
            torch.full((*expert_values.shape[:2], 1), 192.0, device=device)
            if combine_dtype == "nvfp4"
            else None
        )
        weights = torch.full((local_tokens, topk_size), 1 / topk_size, device=device)
        data_bytes = config.dispatch_layout.data_bytes
        scale_bytes = config.dispatch_layout.scale_bytes
        case = dict(
            rank=rank,
            packed=data.view(torch.uint8).reshape(WORLD, tokens, data_bytes),
            scales=scales.view(torch.uint8).reshape(WORLD, tokens, scale_bytes),
            routes=topk,
        )
        output = None

        def run():
            result = buf.dispatch(
                data[rank, :local_tokens],
                topk[rank, :local_tokens],
                input_dtype="raw",
                x_scales=scales[rank, :local_tokens] if scales.numel() else None,
                return_recv_hook=recv_hook,
                out=output,
            )
            if recv_hook:
                result[-1]()
            combined = buf.combine(
                expert_values,
                result[3],
                topk_idx=topk[rank, :local_tokens],
                topk_weights=weights,
                global_scale=global_scale,
                return_recv_hook=recv_hook,
            )
            if recv_hook:
                combined[-1]()
            return result, combined[0]

        def verify(result, combined):
            # Bitwise oracle also checks NaNs and signed zero in opaque floats.
            byte_result = (
                result[0]
                .view(torch.uint8)
                .reshape(EXPERTS // WORLD, WORLD * capacity, data_bytes),
                result[1]
                .view(torch.uint8)
                .reshape(EXPERTS // WORLD, WORLD * capacity, scale_bytes),
                *result[2:],
            )
            check_dispatch(byte_result, case)
            assert result[0].dtype == dtype and result[0].shape[2:] == shape
            assert result[1].dtype == scale_dtype and result[1].shape[2:] == scale_shape
            ids = topk[rank, :local_tokens]
            expected = (gains[ids.clamp_min(0).long()] * (ids >= 0)).sum(
                -1, keepdim=True
            ) / topk_size
            torch.testing.assert_close(
                combined.float(), expected.expand_as(combined), rtol=0, atol=0
            )

        eager = run()
        verify(*eager)

        # Reuse caller-owned outputs with offset bases as well as odd row strides.
        def offset_output(tensor):
            return torch.empty(tensor.numel() + 1, dtype=tensor.dtype, device=device)[
                1:
            ].reshape(tensor.shape)

        output = (
            offset_output(eager[0][0]),
            offset_output(eager[0][1]),
            eager[0][2],
            *eager[0][3][:2],
        )
        with captured_graph(run, device) as (graph, result, _):
            for phase in range(2):
                data.view(torch.uint8).bitwise_xor_(0x81)
                scales.view(torch.uint8).bitwise_xor_(0x81)
                updated_routes = routes(device, tokens, topk_size)
                # Restore the masked row without introducing duplicate routes.
                updated_routes[0, -1] = torch.arange(topk_size, device=device)
                topk.copy_((updated_routes + 11 * (phase + 1)).remainder(EXPERTS))
                if phase == 1 or empty_source:
                    topk[0].fill_(-1)
                graph.replay()
                verify(*result)


@pytest.mark.parametrize(
    "dispatch_dtype,combine_dtype",
    [("nvfp4", "bf16"), ("fp8", "nvfp4"), ("bf16", "nvfp4")],
)
def test_automatic_bf16_receive_with_mixed_codecs(
    distributed_device, dispatch_dtype, combine_dtype
):
    from flashinfer.comm import RdmaMoeAlltoAll, RdmaMoeAlltoAllConfig

    device, rank, hidden = distributed_device, dist.get_rank(), 5120
    tokens, capacity = (9, 12) if dispatch_dtype == "bf16" else (3, 4)
    source = (
        torch.arange(WORLD * tokens * hidden, device=device)
        .remainder(29)
        .sub_(14)
        .to(torch.bfloat16)
        .reshape(WORLD, tokens, hidden)
        / 8
    )
    topk = routes(device, tokens)
    if dispatch_dtype == "bf16":
        # Same-rank followers exercise the reference tail with capacity > 8.
        topk[..., 0] = torch.arange(WORLD * tokens, device=device).reshape(
            WORLD, tokens
        ).remainder(WORLD) * (EXPERTS // WORLD)
        topk[..., 1] = topk[..., 0] + 1
    scale = 128.0
    config = RdmaMoeAlltoAllConfig(
        communication_dtype=combine_dtype, dispatch_dtype=dispatch_dtype
    )
    with managed_buffer(
        RdmaMoeAlltoAll(dist.group.WORLD, capacity, hidden, EXPERTS, config=config)
    ) as buf:
        expert_values = torch.ones(
            (EXPERTS // WORLD, WORLD * capacity, hidden),
            dtype=torch.bfloat16,
            device=device,
        )
        gains = (torch.arange(EXPERTS, device=device).remainder(3) + 1).float() / 2
        expert_values.mul_(
            gains[
                rank * (EXPERTS // WORLD) : (rank + 1) * (EXPERTS // WORLD), None, None
            ]
        )
        combine_scale = (
            torch.full((*expert_values.shape[:2], 1), 192.0, device=device)
            if combine_dtype == "nvfp4"
            else None
        )
        weights = torch.full((tokens, 2), 0.5, device=device)

        def reference():
            if dispatch_dtype == "bf16":
                return source
            if dispatch_dtype == "nvfp4":
                packed, scales = buf.quantize_bf16_to_nvfp4(
                    source, torch.full((*source.shape[:2], 1), scale, device=device)
                )
            else:
                packed, scales = buf.quantize_bf16_to_fp8(source)
            return decode(
                packed, scales, dispatch_dtype, scale, kernel_arithmetic=True
            ).to(torch.bfloat16)

        def run():
            result = buf.dispatch(
                source[rank],
                topk[rank],
                output_dtype="bf16",
                nvfp4_global_scale=scale,
                return_recv_hook=True,
            )
            result[-1]()
            combined = buf.combine(
                expert_values,
                result[3],
                topk_idx=topk[rank],
                topk_weights=weights,
                global_scale=combine_scale,
            )
            return result, combined[0]

        def verify(result, combined):
            check_dispatch(
                result,
                dict(
                    rank=rank,
                    routes=topk,
                    packed=reference(),
                    scales=torch.empty(
                        (WORLD, tokens, 0), dtype=torch.uint8, device=device
                    ),
                ),
            )
            assert result[3].precision == combine_dtype
            ids = topk[rank]
            expected = (gains[ids.clamp_min(0).long()] * (ids >= 0)).sum(
                -1, keepdim=True
            ) * 0.5
            torch.testing.assert_close(
                combined.float(), expected.expand_as(combined), rtol=0, atol=0
            )
            if dispatch_dtype != "bf16":
                valid = (
                    torch.arange(WORLD * capacity, device=device)[None]
                    < result[2][:, None]
                )
                assert not result[0][~valid].count_nonzero().item()

        verify(*run())
        with captured_graph(run, device) as (graph, result, _):
            source.neg_()
            topk.copy_(torch.where(topk >= 0, (topk + 19).remainder(EXPERTS), topk))
            graph.replay()
            verify(*result)
