from __future__ import annotations

import math

import pytest
import torch

from b12x._lib.compile_plan import load_programs
from b12x._lib.runtime_control import kernel_resolution_guard
import b12x.moe._shared.kernels.w4a16.route_pack as route_pack_module
from b12x.moe._shared.kernels.w4a16.kernel import pack_topk_routes_by_expert
from b12x.moe._shared.kernels.w4a16.host import (
    route_block_sizes_for_capacity,
    route_pack_capacity,
    select_route_block_size_m,
)


@pytest.mark.parametrize(
    ("max_tokens", "topk", "num_experts"),
    [
        (1, 8, 160),
        (64, 8, 160),
        (144, 8, 160),
        (1024, 8, 256),
        (32, 64, 8),
    ],
)
def test_capacity_block_sizes_cover_every_live_selection(
    max_tokens: int,
    topk: int,
    num_experts: int,
) -> None:
    planned = route_block_sizes_for_capacity(max_tokens, topk, num_experts)
    selected = {
        select_route_block_size_m(m, topk, num_experts)
        for m in range(1, max_tokens + 1)
    }

    assert selected.issubset(planned)
    assert planned[0] == select_route_block_size_m(1, topk, num_experts)
    assert planned[-1] == select_route_block_size_m(
        max_tokens,
        topk,
        num_experts,
    )


def test_small_capacity_needs_only_block_8() -> None:
    assert route_block_sizes_for_capacity(64, 8, 160) == (8,)


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
def test_route_pack_reuses_provided_fixed_capacity_for_prefill_tail() -> None:
    """A fixed-capacity serving arena must not specialize every tail length.

    With a 1536-token serving capacity, top-k 16, and 896 experts, a
    1177-token prefill tail belongs to the 2048-token compile bucket. That bucket
    the fixed arena even though the arena safely covers the live tail. Reuse
    the caller's full capacity so startup warmup and runtime select the same
    Triton constexprs instead of compiling an exact tail specialization.
    """
    max_tokens = 1536
    tail_tokens = 1177
    topk = 16
    block_size = 8
    num_experts = 896
    _, max_routes, max_blocks = route_pack_capacity(
        max_tokens * topk,
        block_size,
        num_experts,
        topk=topk,
        bucket_tokens=False,
    )
    device = torch.device("cuda")
    topk_ids = (
        torch.arange(tail_tokens * topk, dtype=torch.int32, device=device)
        .reshape(tail_tokens, topk)
        .remainder_(num_experts)
    )
    packed_routes = torch.empty(max_routes, dtype=torch.int32, device=device)
    block_experts = torch.empty(max_blocks, dtype=torch.int32, device=device)
    packed_route_count = torch.empty(1, dtype=torch.int32, device=device)

    returned_routes, returned_blocks, returned_count = pack_topk_routes_by_expert(
        topk_ids,
        block_size,
        num_experts,
        packed_route_indices=packed_routes,
        block_expert_ids=block_experts,
        packed_route_count=packed_route_count,
        expert_offsets=torch.empty(num_experts + 1, dtype=torch.int32, device=device),
        expert_counts=torch.empty(num_experts, dtype=torch.int32, device=device),
    )
    (
        expected_ids,
        expected_valid,
        expected_count,
        expected_blocks,
    ) = _expected_route_pack(topk_ids, block_size, num_experts)
    sentinel = int(topk_ids.numel())
    valid_routes = int(expected_count.item())
    valid_blocks = valid_routes // block_size

    assert returned_routes.data_ptr() == packed_routes.data_ptr()
    assert returned_blocks.data_ptr() == block_experts.data_ptr()
    assert returned_count.data_ptr() == packed_route_count.data_ptr()
    assert returned_routes.numel() == max_routes
    assert returned_blocks.numel() == max_blocks
    assert torch.equal(returned_count.cpu(), expected_count)
    assert torch.equal(returned_blocks[:valid_blocks].cpu(), expected_blocks)
    assert bool(torch.all(returned_blocks[valid_blocks:] == -1).item())
    assert bool(torch.all(returned_routes[valid_routes:] == sentinel).item())

    host_routes = returned_routes[:valid_routes].cpu().to(torch.int64)
    payload = host_routes[host_routes < sentinel]
    assert payload.numel() == int(expected_valid.sum().item())
    assert torch.equal(payload.sort().values, torch.arange(sentinel))
    for block, expert in enumerate(expected_blocks.tolist()):
        block_routes = host_routes[block * block_size : (block + 1) * block_size]
        block_payload = block_routes[block_routes < sentinel]
        if block_payload.numel() > 0:
            assert bool(torch.all(expected_ids[block_payload] == expert).item())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("capacity, bucket_tokens", ((128, True), (129, False), (8192, True)))
@pytest.mark.parametrize("ids_dtype", (torch.int32, torch.int64))
@pytest.mark.parametrize("mapped", (False, True))
def test_prepared_route_pack_uses_live_bounds_with_fixed_geometry(capacity, bucket_tokens, ids_dtype, mapped):
    launches = route_pack_module.compile_w4a16_route_pack_launches(
        tokens=capacity, topk=2, block_size=8, num_experts=32,
        ordinal=torch.cuda.current_device(),
        bucket_tokens=bucket_tokens,
    )
    if not bucket_tokens:
        assert launches.numel_capacity == capacity * 2
    buffers = dict(
        packed_route_indices=torch.empty(launches.max_packed_routes, dtype=torch.int32, device="cuda"),
        block_expert_ids=torch.empty(launches.max_route_blocks, dtype=torch.int32, device="cuda"),
        packed_route_count=torch.empty(1, dtype=torch.int32, device="cuda"),
        expert_offsets=torch.empty(33, dtype=torch.int32, device="cuda"),
        expert_counts=torch.empty(32, dtype=torch.int32, device="cuda"),
    )
    ids = torch.arange(capacity * 2, dtype=ids_dtype, device="cuda").remainder_(32).reshape(capacity, 2)
    expert_map = None
    if mapped:
        expert_map = torch.arange(31, -1, -1, dtype=torch.int32, device="cuda")
        expert_map[::3] = -1
    load_programs(launches.carriers())

    def run(rows):
        return route_pack_module.pack_topk_routes_by_expert(
            ids[:rows], 8, 32, expert_map=expert_map, launches=launches, **buffers,
        )

    def check(rows, output):
        routes, blocks, count = output
        expected_ids, valid, expected_count, expected_blocks = _expected_route_pack(
            ids[:rows], 8, 32, expert_map,
        )
        torch.testing.assert_close(count.cpu(), expected_count, rtol=0, atol=0)
        torch.testing.assert_close(blocks[:len(expected_blocks)].cpu(), expected_blocks, rtol=0, atol=0)
        assert (blocks[len(expected_blocks):] == -1).all()
        assert (routes[int(expected_count.item()):] == rows * 2).all()
        host_routes = routes.cpu().to(torch.int64)
        payload = host_routes[host_routes < rows * 2]
        expected_payload = torch.nonzero(valid).flatten()
        torch.testing.assert_close(payload.sort().values, expected_payload, rtol=0, atol=0)
        for block, expert in enumerate(expected_blocks.tolist()):
            block_routes = host_routes[block * 8:(block + 1) * 8]
            block_payload = block_routes[block_routes < rows * 2]
            assert (expected_ids[block_payload] == expert).all()
        for tensor, name in zip(
            output, ("packed_route_indices", "block_expert_ids", "packed_route_count"), strict=True,
        ):
            assert tensor.data_ptr() == buffers[name].data_ptr()
        assert routes.numel() == launches.max_packed_routes

    run(capacity)
    torch.cuda.synchronize()
    with kernel_resolution_guard("prepared route packing reuses fixed geometry"):
        for rows in (1, 3, 17, capacity):
            check(rows, run(rows))
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = run(rows)
            for tensor in buffers.values():
                tensor.fill_(-999)
            ids.add_(5).remainder_(32)
            graph.replay()
            check(rows, output)


def test_small_prefix_reuses_fixed_arena_numel_capacity(monkeypatch) -> None:
    """Fixed small-prefix arenas must not specialize on each live tail."""

    class LaunchRecorder:
        def __init__(self) -> None:
            self.kwargs: list[dict[str, object]] = []

        def __getitem__(self, _grid):
            def launch(*_args, **kwargs) -> None:
                self.kwargs.append(kwargs)

            return launch

    small_prefix = LaunchRecorder()
    sort = LaunchRecorder()
    monkeypatch.setattr(
        route_pack_module,
        "_pack_topk_routes_small_prefix_kernel",
        small_prefix,
    )
    monkeypatch.setattr(route_pack_module, "_pack_topk_routes_sort_kernel", sort)

    max_tokens = 24
    topk = 1
    block_size = 8
    num_experts = 32
    planned_numel, max_routes, max_blocks = route_pack_capacity(
        max_tokens * topk,
        block_size,
        num_experts,
        topk=topk,
        bucket_tokens=False,
    )

    observed = []
    for tail_tokens in (17, 23):
        route_pack_module.pack_topk_routes_by_expert(
            torch.arange(tail_tokens * topk, dtype=torch.int32).reshape(
                tail_tokens, topk
            )
            % num_experts,
            block_size,
            num_experts,
            packed_route_indices=torch.empty(max_routes, dtype=torch.int32),
            block_expert_ids=torch.empty(max_blocks, dtype=torch.int32),
            packed_route_count=torch.empty(1, dtype=torch.int32),
            expert_offsets=torch.empty(num_experts + 1, dtype=torch.int32),
            expert_counts=torch.empty(num_experts, dtype=torch.int32),
        )
        observed.append(small_prefix.kwargs[-1]["NUMEL_CAPACITY"])

    assert observed == [planned_numel, planned_numel]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_route_pack_capacity_error_reports_live_contract() -> None:
    topk_ids = torch.zeros((17, 8), dtype=torch.int32, device="cuda")

    with pytest.raises(
        ValueError,
        match=(
            r"topk_shape=\(17, 8\).*live_routes=136.*block_size=8.*"
            r"num_experts=32.*packed_route_indices=1/.*block_expert_ids=1/"
        ),
    ):
        pack_topk_routes_by_expert(
            topk_ids,
            8,
            32,
            packed_route_indices=torch.empty(1, dtype=torch.int32, device="cuda"),
            block_expert_ids=torch.empty(1, dtype=torch.int32, device="cuda"),
            packed_route_count=torch.empty(1, dtype=torch.int32, device="cuda"),
            expert_offsets=torch.empty(33, dtype=torch.int32, device="cuda"),
        )


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
@pytest.mark.parametrize("num_experts", [896, 1024])
@pytest.mark.parametrize("tokens", [1, 8, 64])
def test_pack_topk_routes_by_expert_large_expert_decode_capture(
    num_experts: int, tokens: int
) -> None:
    topk = 16
    block_size = 8
    torch.manual_seed(20260811 + num_experts + tokens)
    topk_ids = (
        torch.stack(
            [
                torch.randperm(num_experts, device="cuda")[:topk]
                for _ in range(tokens)
            ]
        )
        .to(torch.int32)
        .contiguous()
    )
    numel = topk_ids.numel()
    _, cap_routes, cap_blocks = route_pack_capacity(
        numel, block_size, num_experts, topk=topk
    )
    workspaces = {
        "packed_route_indices": torch.empty(
            cap_routes, dtype=torch.int32, device="cuda"
        ),
        "block_expert_ids": torch.empty(
            cap_blocks, dtype=torch.int32, device="cuda"
        ),
        "packed_route_count": torch.empty(1, dtype=torch.int32, device="cuda"),
        "expert_offsets": torch.empty(
            num_experts + 1, dtype=torch.int32, device="cuda"
        ),
        "expert_counts": torch.empty(
            num_experts, dtype=torch.int32, device="cuda"
        ),
    }

    def run() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return pack_topk_routes_by_expert(
            topk_ids, block_size, num_experts, **workspaces
        )

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        local_packed_routes, local_block_experts, local_packed_route_count = run()
    graph.replay()
    torch.cuda.synchronize()

    (
        expected_ids,
        expected_valid,
        expected_packed_route_count,
        expected_block_experts,
    ) = _expected_route_pack(topk_ids, block_size, num_experts)

    sentinel = int(numel)
    valid = int(expected_packed_route_count.item())
    valid_blocks = valid // block_size
    assert torch.equal(local_packed_route_count.cpu(), expected_packed_route_count)
    assert torch.equal(local_block_experts[:valid_blocks].cpu(), expected_block_experts)
    assert bool(torch.all(local_block_experts[valid_blocks:] == -1).item())
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
