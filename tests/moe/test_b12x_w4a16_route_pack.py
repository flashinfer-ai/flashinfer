"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Route-packing tests for the SM12x W4A16 fused MoE path.
"""

from __future__ import annotations

import math

import pytest
import torch

from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_w4a16_kernel import (
    pack_topk_routes_by_expert,
)


def _expected_route_pack(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    raw_ids = topk_ids.detach().cpu().reshape(-1).to(torch.int64)
    valid = (raw_ids >= 0) & (raw_ids < num_experts)
    block_expert_ids = raw_ids.clone()
    if expert_map is not None:
        host_map = expert_map.detach().cpu().to(torch.int64)
        safe_raw = raw_ids.clamp(0, num_experts - 1)
        block_expert_ids = host_map[safe_raw]
        valid &= (block_expert_ids >= 0) & (block_expert_ids < num_experts)

    counts = torch.bincount(block_expert_ids[valid], minlength=num_experts)
    padded_counts = torch.tensor(
        [math.ceil(int(count.item()) / block_size) * block_size for count in counts],
        dtype=torch.int64,
    )
    expected_packed_route_count = padded_counts.sum().to(torch.int32).reshape(1)
    block_experts = [
        expert
        for expert, count in enumerate(counts.tolist())
        for _ in range(math.ceil(count / block_size))
    ]
    expected_block_experts = torch.tensor(block_experts, dtype=torch.int32)
    return block_expert_ids, valid, expected_packed_route_count, expected_block_experts


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("shape", [(1, 6), (23, 3), (128, 6), (160, 6), (512, 6)])
@pytest.mark.parametrize("block_size", [8, 16, 32, 48, 64])
def test_pack_topk_routes_by_expert_groups_and_pads_routes(
    dtype: torch.dtype,
    shape: tuple[int, int],
    block_size: int,
) -> None:
    torch.manual_seed(20260514 + shape[0] + shape[1] + block_size)
    num_experts = 16 if shape[1] == 3 else 128
    topk_ids = torch.randint(
        0,
        num_experts,
        shape,
        dtype=dtype,
        device="cuda",
    )

    local_packed_routes, local_block_experts, local_packed_route_count = (
        pack_topk_routes_by_expert(
            topk_ids,
            block_size,
            num_experts,
        )
    )
    (
        expected_ids,
        expected_valid,
        expected_packed_route_count,
        expected_block_experts,
    ) = _expected_route_pack(topk_ids, block_size, num_experts)

    sentinel = int(topk_ids.numel())
    valid = int(expected_packed_route_count.item())
    valid_blocks = valid // block_size
    assert torch.equal(local_packed_route_count.cpu(), expected_packed_route_count)
    assert torch.equal(local_block_experts[:valid_blocks].cpu(), expected_block_experts)
    assert bool(torch.all(local_packed_routes[valid:] == sentinel).item())

    host_packed_routes = local_packed_routes[:valid].detach().cpu().to(torch.int64)
    host_route_payload = host_packed_routes[host_packed_routes < sentinel]
    assert host_route_payload.numel() == int(expected_valid.sum().item())
    for expert in range(num_experts):
        actual = (
            host_route_payload[expected_ids[host_route_payload] == expert].sort().values
        )
        expected = torch.nonzero(
            expected_valid & (expected_ids == expert), as_tuple=False
        )
        expected = expected.flatten().sort().values
        assert torch.equal(actual, expected), expert

    host_block_experts = (
        local_block_experts[:valid_blocks].detach().cpu().to(torch.int64)
    )
    for block, expert in enumerate(host_block_experts.tolist()):
        block_routes = host_packed_routes[block * block_size : (block + 1) * block_size]
        payload = block_routes[block_routes < sentinel]
        if payload.numel() > 0:
            assert bool(torch.all(expected_ids[payload] == expert).item())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_pack_topk_routes_by_expert_handles_large_prefill_plan_shape() -> None:
    tokens = 8192
    topk = 6
    num_experts = 256
    block_size = 64
    topk_ids = (
        torch.arange(tokens * topk, dtype=torch.int32, device="cuda")
        .reshape(tokens, topk)
        .remainder_(num_experts)
    )

    local_packed_routes, local_block_experts, local_packed_route_count = (
        pack_topk_routes_by_expert(
            topk_ids,
            block_size,
            num_experts,
        )
    )
    (
        expected_ids,
        expected_valid,
        expected_packed_route_count,
        expected_block_experts,
    ) = _expected_route_pack(topk_ids, block_size, num_experts)

    sentinel = int(topk_ids.numel())
    valid = int(expected_packed_route_count.item())
    valid_blocks = valid // block_size
    assert torch.equal(local_packed_route_count.cpu(), expected_packed_route_count)
    assert torch.equal(local_block_experts[:valid_blocks].cpu(), expected_block_experts)
    assert bool(torch.all(local_packed_routes[valid:] == sentinel).item())

    host_packed_routes = local_packed_routes[:valid].detach().cpu().to(torch.int64)
    host_route_payload = host_packed_routes[host_packed_routes < sentinel]
    assert host_route_payload.numel() == int(expected_valid.sum().item())
    for expert in range(num_experts):
        actual = (
            host_route_payload[expected_ids[host_route_payload] == expert].sort().values
        )
        expected = torch.nonzero(
            expected_valid & (expected_ids == expert), as_tuple=False
        )
        expected = expected.flatten().sort().values
        assert torch.equal(actual, expected), expert


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("block_size", [8, 16, 32, 48, 64])
def test_pack_topk_routes_by_expert_applies_expert_map(
    dtype: torch.dtype,
    block_size: int,
) -> None:
    torch.manual_seed(20260516 + block_size)
    global_experts = 16
    local_experts = 8
    shape = (65, 4)
    topk_ids = torch.randint(
        0,
        global_experts,
        shape,
        dtype=dtype,
        device="cuda",
    )
    expert_map = torch.full((global_experts,), -1, dtype=torch.int32, device="cuda")
    expert_map[::2] = torch.arange(local_experts, dtype=torch.int32, device="cuda")

    local_packed_routes, local_block_experts, local_packed_route_count = (
        pack_topk_routes_by_expert(
            topk_ids,
            block_size,
            global_experts,
            expert_map=expert_map,
        )
    )
    (
        expected_ids,
        expected_valid,
        expected_packed_route_count,
        expected_block_experts,
    ) = _expected_route_pack(topk_ids, block_size, global_experts, expert_map)

    sentinel = int(topk_ids.numel())
    valid = int(expected_packed_route_count.item())
    valid_blocks = valid // block_size
    assert torch.equal(local_packed_route_count.cpu(), expected_packed_route_count)
    assert torch.equal(local_block_experts[:valid_blocks].cpu(), expected_block_experts)
    assert bool(torch.all(local_packed_routes[valid:] == sentinel).item())

    host_packed_routes = local_packed_routes[:valid].detach().cpu().to(torch.int64)
    host_route_payload = host_packed_routes[host_packed_routes < sentinel]
    assert host_route_payload.numel() == int(expected_valid.sum().item())
    for local_expert in range(local_experts):
        actual = (
            host_route_payload[expected_ids[host_route_payload] == local_expert]
            .sort()
            .values
        )
        expected = torch.nonzero(
            expected_valid & (expected_ids == local_expert),
            as_tuple=False,
        )
        expected = expected.flatten().sort().values
        assert torch.equal(actual, expected), local_expert


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_pack_topk_routes_by_expert_ignores_invalid_ids() -> None:
    topk_ids = torch.tensor(
        [[0, -1, 4], [7, 99, 4], [3, 3, -5]],
        dtype=torch.int32,
        device="cuda",
    )
    block_size = 4
    num_experts = 8

    local_packed_routes, local_block_experts, local_packed_route_count = (
        pack_topk_routes_by_expert(
            topk_ids,
            block_size,
            num_experts,
        )
    )
    (
        expected_ids,
        expected_valid,
        expected_packed_route_count,
        expected_block_experts,
    ) = _expected_route_pack(topk_ids, block_size, num_experts)

    sentinel = int(topk_ids.numel())
    valid = int(expected_packed_route_count.item())
    valid_blocks = valid // block_size
    assert torch.equal(local_packed_route_count.cpu(), expected_packed_route_count)
    assert torch.equal(local_block_experts[:valid_blocks].cpu(), expected_block_experts)
    payload = local_packed_routes[:valid].detach().cpu().to(torch.int64)
    payload = payload[payload < sentinel]
    assert payload.numel() == int(expected_valid.sum().item())
    assert torch.equal(
        payload[expected_ids[payload] == 4].sort().values, torch.tensor([2, 5])
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_pack_topk_routes_single_exact_capacity_buffer_degrades() -> None:
    """A preallocated buffer sized for the exact-numel capacity degrades the
    packing capacity instead of raising, even when it is the only buffer
    provided."""
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_w4a16_host import (
        max_packed_route_slots,
    )

    tokens, topk, num_experts, block_size = 48, 8, 32, 8
    topk_ids = torch.randint(
        0, num_experts, (tokens, topk), dtype=torch.int32, device="cuda"
    )
    exact_slots = max_packed_route_slots(tokens * topk, block_size, num_experts)
    preallocated = torch.empty((exact_slots,), dtype=torch.int32, device="cuda")

    packed, _, count = pack_topk_routes_by_expert(
        topk_ids,
        block_size,
        num_experts,
        packed_route_indices=preallocated,
    )
    reference, _, reference_count = pack_topk_routes_by_expert(
        topk_ids, block_size, num_experts
    )
    valid = int(reference_count.item())
    assert int(count.item()) == valid
    # Route order within an expert's blocks depends on the capacity
    # specialization, so compare the payload as a set.
    sentinel = tokens * topk
    payload = packed[:valid][packed[:valid] < sentinel]
    reference_payload = reference[:valid][reference[:valid] < sentinel]
    assert torch.equal(payload.sort().values, reference_payload.sort().values)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_w4a16_workspace_route_buffers_cover_route_pack_capacity(monkeypatch):
    """The dispatch workspace must size route buffers for the power-of-2
    capacity route packing specializes on, so packing never degrades to
    per-token-count capacities."""
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_w4a16_host import (
        _W4A16_ALLOWED_ROUTED_SIZES,
        max_packed_route_slots,
        route_pack_numel_capacity,
    )

    tokens, topk, num_experts = 48, 8, 32  # 48 tokens round up to 64
    routed_rows = tokens * topk
    monkeypatch.setattr(moe_dispatch, "get_num_sm", lambda device: 120)
    slots, blocks, _, _, _ = moe_dispatch._w4a16_workspace_geometry(
        routed_rows=routed_rows,
        route_num_experts=num_experts,
        num_topk=topk,
        k=4096,
        n=4096,
        is_gated=True,
        device=torch.device("cuda"),
    )
    numel_capacity = route_pack_numel_capacity(routed_rows, topk=topk)
    assert numel_capacity > routed_rows
    for block_size in _W4A16_ALLOWED_ROUTED_SIZES:
        required = max_packed_route_slots(numel_capacity, block_size, num_experts)
        assert slots >= required
        assert blocks >= (required + block_size - 1) // block_size
