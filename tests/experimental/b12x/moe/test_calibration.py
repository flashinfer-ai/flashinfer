from __future__ import annotations

import pytest
import torch

from b12x.moe.calibration import (
    MidBuffers,
    RouteInputBuffers,
    collect_mid,
    collect_paired_token_rows,
    collect_route_input,
    unscale_route_rows_,
)


def _zeros(*shape: int, dtype: torch.dtype) -> torch.Tensor:
    return torch.zeros(shape, dtype=dtype, device="cuda")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_calibration_sidecars_match_torch_reference() -> None:
    x = torch.tensor(
        [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [2, 3, 5, 7, 11]],
        dtype=torch.bfloat16,
        device="cuda",
    )
    topk_ids = torch.tensor([[0, 2], [1, 3], [2, 3]], dtype=torch.int32, device="cuda")
    topk_weights = torch.tensor(
        [[0.75, 0.25], [0.6, 0.4], [0.8, 0.2]],
        dtype=torch.float32,
        device="cuda",
    )
    is_padding = torch.tensor([False, True, False], device="cuda")
    input_samples = 8
    route_buffers = RouteInputBuffers(
        enabled=torch.ones(1, dtype=torch.int32, device="cuda"),
        epoch_counter=_zeros(1, dtype=torch.int64),
        epoch=_zeros(1, dtype=torch.int64),
        tokens_routed=_zeros(4, dtype=torch.int64),
        gate_sum=_zeros(4, dtype=torch.float64),
        gate_sq_sum=_zeros(4, dtype=torch.float64),
        input_sq_sum=_zeros(2, 5, dtype=torch.float32),
        input_weight_sum=_zeros(2, dtype=torch.float64),
        input_count=_zeros(2, dtype=torch.int64),
        sample_cursor=_zeros(1, dtype=torch.int64),
        sample_dropped=_zeros(1, dtype=torch.int64),
        sample_slots=torch.full((3,), -1, dtype=torch.int32, device="cuda"),
        sample_values=_zeros(input_samples, 5, dtype=torch.bfloat16),
        sample_weight=_zeros(input_samples, dtype=torch.float32),
        sample_observation=_zeros(input_samples, dtype=torch.int64),
        sample_experts=_zeros(input_samples, 2, dtype=torch.int32),
        sample_gates=_zeros(input_samples, 2, dtype=torch.float32),
        sample_split=_zeros(input_samples, dtype=torch.int8),
    )
    collect_route_input(
        x,
        topk_weights,
        topk_ids,
        is_padding,
        route_buffers,
        num_experts=4,
        expert_begin=0,
        expert_end=2,
        moment_sample_rate=1,
        hessian_sample_rate=1,
        validation_modulus=2,
        collect_routing=True,
    )
    torch.cuda.synchronize()

    assert route_buffers.epoch_counter.item() == 1
    assert route_buffers.tokens_routed.cpu().tolist() == [1, 0, 2, 1]
    torch.testing.assert_close(
        route_buffers.gate_sum.cpu(),
        torch.tensor([0.75, 0.0, 1.05, 0.2], dtype=torch.float64),
    )
    torch.testing.assert_close(
        route_buffers.gate_sq_sum.cpu(),
        torch.tensor([0.75**2, 0.0, 0.25**2 + 0.8**2, 0.2**2], dtype=torch.float64),
    )
    expected_input = x[0].float().square() * 0.75**2
    torch.testing.assert_close(
        route_buffers.input_sq_sum[0].cpu(), expected_input.cpu()
    )
    assert torch.count_nonzero(route_buffers.input_sq_sum[1]).item() == 0
    assert route_buffers.input_weight_sum.cpu().tolist() == [0.75**2, 0.0]
    assert route_buffers.input_count.cpu().tolist() == [1, 0]

    input_count = int(route_buffers.sample_cursor.item())
    assert input_count == 2
    sampled_input = route_buffers.sample_values[:input_count].cpu()
    sampled_observation = route_buffers.sample_observation[:input_count].cpu()
    order = torch.argsort(sampled_observation)
    torch.testing.assert_close(sampled_input[order], x[[0, 2]].cpu())
    torch.testing.assert_close(
        route_buffers.sample_weight[:input_count].cpu()[order],
        torch.tensor([0.75**2 + 0.25**2, 0.8**2 + 0.2**2]),
    )
    torch.testing.assert_close(
        route_buffers.sample_experts[:input_count].cpu()[order],
        topk_ids[[0, 2]].cpu(),
    )
    torch.testing.assert_close(
        route_buffers.sample_gates[:input_count].cpu()[order],
        topk_weights[[0, 2]].cpu(),
    )

    paired = x + 32
    paired_samples = _zeros(input_samples, 5, dtype=torch.bfloat16)
    paired_ready = _zeros(input_samples, dtype=torch.int8)
    collect_paired_token_rows(
        paired,
        route_buffers.sample_slots,
        paired_samples,
        paired_ready,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(
        paired_samples[:input_count].cpu()[order], paired[[0, 2]].cpu()
    )
    assert paired_ready[:input_count].cpu().tolist() == [1, 1]

    mid = torch.arange(1, 19, dtype=torch.bfloat16, device="cuda").reshape(6, 3)
    mid_samples = 16
    mid_buffers = MidBuffers(
        enabled=route_buffers.enabled,
        epoch=route_buffers.epoch,
        mid_sq_sum=_zeros(4, 3, dtype=torch.float32),
        mid_weight_sum=_zeros(4, dtype=torch.float64),
        mid_count=_zeros(4, dtype=torch.int64),
        sample_cursor=_zeros(1, dtype=torch.int64),
        sample_dropped=_zeros(1, dtype=torch.int64),
        sample_slots=torch.full((6,), -1, dtype=torch.int32, device="cuda"),
        sample_values=_zeros(mid_samples, 3, dtype=torch.bfloat16),
        sample_weight=_zeros(mid_samples, dtype=torch.float32),
        sample_observation=_zeros(mid_samples, dtype=torch.int64),
        sample_expert=_zeros(mid_samples, dtype=torch.int32),
        sample_split=_zeros(mid_samples, dtype=torch.int8),
    )
    collect_mid(
        mid,
        topk_weights,
        topk_ids,
        None,
        is_padding,
        mid_buffers,
        num_experts=4,
        width=3,
        source_stride=3,
        moment_sample_rate=1,
        hessian_sample_rate=1,
        validation_modulus=2,
    )
    torch.cuda.synchronize()

    live_routes = [0, 1, 4, 5]
    expected_mid = torch.zeros(4, 3)
    expected_weight = torch.zeros(4, dtype=torch.float64)
    expected_count = torch.zeros(4, dtype=torch.int64)
    flat_ids = topk_ids.flatten().cpu()
    flat_weights = topk_weights.flatten().cpu()
    for route in live_routes:
        expert = int(flat_ids[route])
        weight_sq = float(flat_weights[route]) ** 2
        expected_mid[expert] += mid[route].float().cpu().square() * weight_sq
        expected_weight[expert] += weight_sq
        expected_count[expert] += 1
    torch.testing.assert_close(mid_buffers.mid_sq_sum.cpu(), expected_mid)
    torch.testing.assert_close(mid_buffers.mid_weight_sum.cpu(), expected_weight)
    torch.testing.assert_close(mid_buffers.mid_count.cpu(), expected_count)

    mid_count = int(mid_buffers.sample_cursor.item())
    assert mid_count == len(live_routes)
    observations = mid_buffers.sample_observation[:mid_count].cpu()
    order = torch.argsort(observations)
    torch.testing.assert_close(
        mid_buffers.sample_values[:mid_count].cpu()[order],
        mid[live_routes].cpu(),
    )
    assert mid_buffers.sample_expert[:mid_count].cpu()[order].tolist() == [0, 2, 2, 3]
    input_splits = route_buffers.sample_split[:input_count].cpu()[
        torch.argsort(sampled_observation)
    ]
    assert mid_buffers.sample_split[:mid_count].cpu()[order].tolist() == [
        int(input_splits[0]),
        int(input_splits[0]),
        int(input_splits[1]),
        int(input_splits[1]),
    ]

    logical = torch.arange(1, 19, dtype=torch.float16, device="cuda").reshape(6, 3)
    expert_map = torch.tensor([1, -1, 0, -1], dtype=torch.int32, device="cuda")
    scale_table = torch.tensor(
        [[2.0, 4.0, 8.0, -1.0], [3.0, 5.0, 7.0, -1.0]],
        dtype=torch.float16,
        device="cuda",
    )
    scaled = logical.clone()
    flat_ids = topk_ids.flatten()
    for route in (0, 1, 4, 5):
        local = int(expert_map[flat_ids[route]])
        if local >= 0:
            scaled[route] *= scale_table[local, :3]
    unscale_route_rows_(
        scaled,
        topk_ids,
        expert_map,
        scale_table,
        num_experts=4,
        scale_stride=4,
    )
    torch.cuda.synchronize()
    for route in (0, 1, 4, 5):
        if int(expert_map[flat_ids[route]]) >= 0:
            torch.testing.assert_close(scaled[route].cpu(), logical[route].cpu())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_route_collector_is_cuda_graph_replay_safe() -> None:
    x = torch.arange(8, dtype=torch.bfloat16, device="cuda").reshape(2, 4)
    topk_ids = torch.tensor([[0, 1], [1, 2]], dtype=torch.int32, device="cuda")
    topk_weights = torch.full((2, 2), 0.5, dtype=torch.float32, device="cuda")
    padding = torch.zeros(2, dtype=torch.bool, device="cuda")
    buffers = RouteInputBuffers(
        enabled=torch.ones(1, dtype=torch.int32, device="cuda"),
        epoch_counter=_zeros(1, dtype=torch.int64),
        epoch=_zeros(1, dtype=torch.int64),
        tokens_routed=_zeros(3, dtype=torch.int64),
        gate_sum=_zeros(3, dtype=torch.float64),
        gate_sq_sum=_zeros(3, dtype=torch.float64),
        input_sq_sum=_zeros(3, 4, dtype=torch.float32),
        input_weight_sum=_zeros(3, dtype=torch.float64),
        input_count=_zeros(3, dtype=torch.int64),
        sample_cursor=_zeros(1, dtype=torch.int64),
        sample_dropped=_zeros(1, dtype=torch.int64),
        sample_slots=torch.full((2,), -1, dtype=torch.int32, device="cuda"),
        sample_values=_zeros(8, 4, dtype=torch.bfloat16),
        sample_weight=_zeros(8, dtype=torch.float32),
        sample_observation=_zeros(8, dtype=torch.int64),
        sample_experts=_zeros(8, 2, dtype=torch.int32),
        sample_gates=_zeros(8, 2, dtype=torch.float32),
        sample_split=_zeros(8, dtype=torch.int8),
    )
    paired_values = _zeros(8, 4, dtype=torch.bfloat16)
    paired_ready = _zeros(8, dtype=torch.int8)
    paired_source = x + 64

    def launch() -> None:
        collect_route_input(
            x,
            topk_weights,
            topk_ids,
            padding,
            buffers,
            num_experts=3,
            expert_begin=0,
            expert_end=3,
            moment_sample_rate=1,
            hessian_sample_rate=1,
            validation_modulus=16,
            collect_routing=True,
        )
        collect_paired_token_rows(
            paired_source,
            buffers.sample_slots,
            paired_values,
            paired_ready,
        )

    # Compile before capture, then restore every persistent output buffer.
    launch()
    torch.cuda.synchronize()
    for tensor in (
        buffers.epoch_counter,
        buffers.epoch,
        buffers.tokens_routed,
        buffers.gate_sum,
        buffers.gate_sq_sum,
        buffers.input_sq_sum,
        buffers.input_weight_sum,
        buffers.input_count,
        buffers.sample_cursor,
        buffers.sample_dropped,
        buffers.sample_values,
        buffers.sample_weight,
        buffers.sample_observation,
        buffers.sample_experts,
        buffers.sample_gates,
        buffers.sample_split,
        paired_values,
        paired_ready,
    ):
        tensor.zero_()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    torch.cuda.synchronize()
    graph.replay()
    graph.replay()
    torch.cuda.synchronize()

    # This PyTorch/CUDA build does not execute work while instantiating a
    # graph. Two replays must accumulate twice into the same stable buffers.
    assert buffers.epoch_counter.item() == 2
    assert buffers.tokens_routed.cpu().tolist() == [2, 4, 2]
    assert buffers.sample_cursor.item() == 4
    assert paired_ready[:4].cpu().tolist() == [1, 1, 1, 1]
