from __future__ import annotations

import inspect
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

import b12x.comm.pcie.pcie_allreduce as pcie_allreduce
import b12x.comm.pcie.pcie_island_rs as pcie_island_rs
from b12x.comm.pcie import _hierarchical_cute
from b12x.comm.pcie.pcie_allreduce import (
    ISLAND_RS_MAX_BYTES,
    PCIeAllReduce,
    _algorithm_for_world_size,
    recommended_max_bytes,
)
from b12x.comm.pcie.pcie_hierarchical import (
    PCIeHierarchicalAllReduce,
    _buffer_modes_from_env,
    _make_layout,
    _pick_blocks,
    _selected_peers,
    _threads_from_env,
    _vectorized_bf16x2_from_env,
    _vectorized_bf16x2_max_elements_from_env,
    _wait_nanosleep_cycles_from_env,
)
from b12x.comm.pcie.pcie_island_rs import PCIeIslandRSAllReduce


def _prepared_dispatch(manager, inp, **call):
    """Exercise plan-time dispatch without compiling a GPU program in CPU tests."""
    from b12x.comm.pcie import _preparation

    with (
        patch.object(_preparation, "query_from_runtime", side_effect=lambda runtime, **_: runtime),
        patch.object(_preparation, "plan", side_effect=lambda query, **_: SimpleNamespace(runtime=query)),
        patch.object(pcie_allreduce, "require_prepared", side_effect=lambda plan, *_: plan),
    ):
        plan = manager.plan(inp, **call)
        manager.all_reduce(inp, plan=plan, **call)
    return plan


@pytest.mark.parametrize("strided", [False, True])
def test_hierarchical_query_preserves_tensor_metadata(strided: bool) -> None:
    from b12x.comm.pcie import _owner_preparation
    from b12x.preparation import FrozenMapping

    runtime = SimpleNamespace(
        rank=0, world_size=16, threads=256, wait_nanosleep_cycles=0,
        double_buffered=True, deferred_consumption=False,
        vectorized_bf16x2=True, vectorized_bf16x2_max_elements=7168,
        max_elements=8192, blocks=None, slab_bytes=65536,
        mapped_peers=(1, 2, 3, 4, 8, 12),
    )
    inp = torch.empty((2, 64), dtype=torch.bfloat16)
    if strided:
        inp = inp[:, ::2]
    out = torch.empty_like(inp)
    query = _owner_preparation.query_from_runtime(
        runtime, surface="PCIeHierarchicalAllReduce.all_reduce",
        call={"inp": inp, "out": out},
    )
    encoded = query.call["inp"]
    assert isinstance(encoded, FrozenMapping)
    assert encoded["shape"] == tuple(inp.shape)
    assert encoded["stride"] == tuple(inp.stride())
    assert encoded["dtype"] == str(inp.dtype)
    assert encoded["alignment"] >= 4
    assert query.call["output"]["shape"] == tuple(out.shape)
    assert hash(query.call) == hash(FrozenMapping(query.call.to_dict()))


@pytest.mark.parametrize("world_size", [2, 4, 6, 8])
def test_allreduce_uses_direct_path_for_peer_safe_worlds(world_size: int) -> None:
    assert _algorithm_for_world_size(world_size) == "oneshot"


@pytest.mark.parametrize("world_size", [9, 10, 12, 16])
def test_allreduce_uses_bounded_degree_path_for_large_worlds(
    world_size: int,
) -> None:
    assert _algorithm_for_world_size(world_size) == "hierarchical"


def test_allreduce_factory_keeps_tp16_equal_quarter_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("B12X_PCIE_ALLREDUCE_ALGORITHM", raising=False)
    runtime = SimpleNamespace(
        rank=0,
        world_size=16,
        device=torch.device("cuda:0"),
    )
    hierarchy_factory = MagicMock(return_value=runtime)
    island_runtime = MagicMock()
    island_factory = MagicMock(return_value=island_runtime)
    exchange_group = object()
    monkeypatch.setattr(pcie_allreduce.dist, "get_world_size", lambda group: 16)
    monkeypatch.setattr(
        pcie_allreduce,
        "PCIeHierarchicalAllReduce",
        hierarchy_factory,
    )
    monkeypatch.setattr(
        pcie_allreduce,
        "PCIeIslandRSAllReduce",
        island_factory,
    )

    allreduce = PCIeAllReduce.from_exchange_group(
        exchange_group=exchange_group,  # type: ignore[arg-type]
        device="cuda:0",
        max_size=16 * 1024,
    )

    assert allreduce.algorithm == "hierarchical"
    hierarchy_factory.assert_called_once_with(
        exchange_group=exchange_group,
        device="cuda:0",
        max_elements=8 * 1024,
        ext_module=None,
    )
    island_factory.assert_not_called()
    assert allreduce._island_rs is None

    monkeypatch.setenv("B12X_PCIE_ALLREDUCE_ALGORITHM", "island_rs")
    allreduce = PCIeAllReduce.from_exchange_group(
        exchange_group=exchange_group,  # type: ignore[arg-type]
        device="cuda:0",
        max_size=16 * 1024,
    )
    island_factory.assert_called_once_with(
        exchange_group=exchange_group,
        device="cuda:0",
        max_elements=ISLAND_RS_MAX_BYTES // torch.bfloat16.itemsize,
    )
    assert allreduce._island_rs is island_runtime


@pytest.mark.parametrize(
    ("world_size", "algorithm", "expected"),
    [
        (2, "auto", 512 * 1024),
        (8, "auto", 84 * 1024),
        (9, "auto", 32 * 1024),
        (10, "auto", 32 * 1024),
        (10, "island_rs", 32 * 1024),
        (12, "auto", 84 * 1024),
        (16, "auto", 84 * 1024),
        (16, "island_rs", ISLAND_RS_MAX_BYTES),
        (16, "hierarchical", 84 * 1024),
    ],
)
def test_recommended_max_bytes_matches_available_tp_runtime(
    monkeypatch: pytest.MonkeyPatch,
    world_size: int,
    algorithm: str,
    expected: int,
) -> None:
    monkeypatch.setenv("B12X_PCIE_TP2_PLAIN_REMOTE_PUSH", "1")
    monkeypatch.setenv("B12X_PCIE_ALLREDUCE_ALGORITHM", algorithm)
    assert recommended_max_bytes(world_size, default=84 * 1024) == expected


def test_recommended_max_bytes_rejects_unknown_algorithm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("B12X_PCIE_ALLREDUCE_ALGORITHM", "unknown")
    with pytest.raises(ValueError, match="must be auto"):
        recommended_max_bytes(16, default=84 * 1024)


@pytest.mark.parametrize("world_size", [9, 10])
def test_partial_island_recommendation_does_not_raise_smaller_limit(world_size):
    assert recommended_max_bytes(world_size, default=16 * 1024) == 16 * 1024


def test_tp16_auto_routes_by_message_size() -> None:
    hierarchy = MagicMock()
    hierarchy.rank = 0
    hierarchy.world_size = 16
    hierarchy.device = torch.device("cpu")
    hierarchy.should_allreduce.return_value = True
    island = MagicMock()
    island.should_allreduce.return_value = True
    allreduce = PCIeAllReduce(hierarchy, "hierarchical", island)
    small = torch.empty(pcie_allreduce.ISLAND_RS_CROSSOVER_ELEMENTS)
    large = torch.empty(pcie_allreduce.ISLAND_RS_CROSSOVER_ELEMENTS * 2)
    small_out = torch.empty_like(small)
    large_out = torch.empty_like(large)

    small_plan = _prepared_dispatch(allreduce, small, out=small_out)
    large_plan = _prepared_dispatch(allreduce, large, out=large_out)

    hierarchy.all_reduce.assert_called_once_with(
        small,
        plan=small_plan,
        out=small_out,
        blocks=None,
        stream=None,
        channel_id=None,
    )
    island.all_reduce.assert_called_once_with(
        large,
        plan=large_plan,
        out=large_out,
        blocks=None,
        stream=None,
        channel_id=None,
    )


def test_island_capture_allocates_implicit_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = PCIeIslandRSAllReduce.__new__(PCIeIslandRSAllReduce)
    runtime._closed = False
    runtime.device = torch.device("cpu")
    runtime.max_elements = 8
    runtime.blocks = None
    runtime._launcher = MagicMock()
    runtime._slab_ptrs = ()
    runtime.stage_offset = 0
    runtime.part_offset = 0
    runtime.final_offset = 0
    runtime.quarter_capacity = 2
    inp = torch.empty(8, dtype=torch.bfloat16)
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    state = MagicMock()
    state.query.call = {"inp": {"shape": (8,)}, "blocks": 1}
    state.launcher.return_value = runtime._launcher
    monkeypatch.setattr(pcie_island_rs, "require_prepared", lambda *_: state)

    with runtime.capture() as captured:
        out = captured.all_reduce(inp, plan=object())

    assert captured is runtime
    assert out.shape == inp.shape
    assert out.dtype == inp.dtype
    assert out.device == inp.device
    runtime._launcher.assert_called_once()
    state.require_runtime.assert_called_once_with(runtime)


def test_island_close_coordinates_ipc_teardown_and_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = PCIeIslandRSAllReduce.__new__(PCIeIslandRSAllReduce)
    runtime._closed = False
    runtime.device = torch.device("cuda:0")
    runtime.group = object()
    runtime._slab_ptrs = (101, 102, 303)
    runtime._remote_ptrs = [101, 102]
    runtime._local_ptr = 303
    runtime._ipc = MagicMock()
    events = []

    def close_remote(ptr: int) -> None:
        assert runtime._slab_ptrs == ()
        events.append(("close", ptr))

    def free_local(ptr: int) -> None:
        assert runtime._remote_ptrs == []
        events.append(("free", ptr))

    monkeypatch.setattr(
        torch.cuda,
        "synchronize",
        lambda device: events.append(("synchronize", device)),
    )
    monkeypatch.setattr(
        pcie_island_rs.dist,
        "barrier",
        lambda *, group: events.append(("barrier", group)),
    )
    runtime._ipc.cudaIpcCloseMemHandle.side_effect = close_remote
    runtime._ipc.cudaFree.side_effect = free_local

    runtime.close()
    runtime.close()

    assert events == [
        ("synchronize", runtime.device),
        ("barrier", runtime.group),
        ("close", 101),
        ("close", 102),
        ("barrier", runtime.group),
        ("free", 303),
        ("barrier", runtime.group),
    ]
    assert runtime._closed
    assert runtime._slab_ptrs == ()
    assert runtime._remote_ptrs == []
    assert runtime._local_ptr == 0


def test_allreduce_close_propagates_island_teardown_error() -> None:
    hierarchy = MagicMock()
    hierarchy.rank = 0
    hierarchy.world_size = 16
    hierarchy.device = torch.device("cpu")
    island = MagicMock()
    island.close.side_effect = RuntimeError("island teardown failed")
    allreduce = PCIeAllReduce(
        hierarchy,
        "hierarchical",
        island,
        algorithm_override="island_rs",
    )

    with pytest.raises(RuntimeError, match="island teardown failed"):
        allreduce.close()

    hierarchy.close.assert_not_called()
    assert allreduce._island_rs is island


def test_tp16_auto_implicit_output_uses_island_for_large_aligned_input() -> None:
    hierarchy = MagicMock()
    hierarchy.rank = 0
    hierarchy.world_size = 16
    hierarchy.device = torch.device("cpu")
    hierarchy.should_allreduce.return_value = True
    island = MagicMock()
    island.should_allreduce.return_value = True
    allreduce = PCIeAllReduce(hierarchy, "hierarchical", island)
    inp = torch.empty(
        pcie_allreduce.ISLAND_RS_CROSSOVER_ELEMENTS * 2,
        dtype=torch.bfloat16,
    )

    plan = _prepared_dispatch(allreduce, inp)

    island.all_reduce.assert_called_once_with(
        inp,
        plan=plan,
        out=None,
        blocks=None,
        stream=None,
        channel_id=None,
    )
    hierarchy.all_reduce.assert_not_called()


def test_tp16_capture_returns_dispatcher() -> None:
    hierarchy = MagicMock()
    hierarchy.rank = 0
    hierarchy.world_size = 16
    hierarchy.device = torch.device("cpu")
    hierarchy.capture.return_value = nullcontext(hierarchy)
    island = MagicMock()
    island.capture.return_value = nullcontext(island)
    allreduce = PCIeAllReduce(hierarchy, "hierarchical", island)

    with allreduce.capture(stream="17", channel_id="target") as captured:
        assert captured is allreduce

    hierarchy.capture.assert_called_once_with(stream="17", channel_id="target")
    island.capture.assert_called_once_with(stream="17", channel_id="target")


def test_tp16_island_opt_in_keeps_small_input_on_hierarchy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("B12X_PCIE_ALLREDUCE_ALGORITHM", "island_rs")
    hierarchy = MagicMock()
    hierarchy.rank = 0
    hierarchy.world_size = 16
    hierarchy.device = torch.device("cpu")
    hierarchy.should_allreduce.return_value = True
    island = MagicMock()
    island.should_allreduce.return_value = True
    allreduce = PCIeAllReduce(hierarchy, "hierarchical", island)
    inp = torch.empty(2)

    plan = _prepared_dispatch(allreduce, inp)

    hierarchy.all_reduce.assert_called_once_with(
        inp,
        plan=plan,
        out=None,
        blocks=None,
        stream=None,
        channel_id=None,
    )
    island.all_reduce.assert_not_called()


def test_tp16_auto_uses_island_when_hierarchy_rejects_below_crossover() -> None:
    hierarchy = MagicMock()
    hierarchy.rank = 0
    hierarchy.world_size = 16
    hierarchy.device = torch.device("cpu")
    hierarchy.should_allreduce.return_value = False
    island = MagicMock()
    island.should_allreduce.return_value = True
    allreduce = PCIeAllReduce(hierarchy, "hierarchical", island)
    inp = torch.empty(pcie_allreduce.ISLAND_RS_CROSSOVER_ELEMENTS - 2)

    assert allreduce.should_allreduce(inp)
    plan = _prepared_dispatch(allreduce, inp, stream="17", channel_id="target")

    island.all_reduce.assert_called_once_with(
        inp,
        plan=plan,
        out=None,
        blocks=None,
        stream="17",
        channel_id="target",
    )
    hierarchy.all_reduce.assert_not_called()


def test_tp16_auto_keeps_unaligned_large_input_on_hierarchy() -> None:
    hierarchy = MagicMock()
    hierarchy.rank = 0
    hierarchy.world_size = 16
    hierarchy.device = torch.device("cpu")
    hierarchy.should_allreduce.return_value = True
    island = MagicMock()
    island.should_allreduce.return_value = True
    allreduce = PCIeAllReduce(hierarchy, "hierarchical", island)
    inp = torch.empty(pcie_allreduce.ISLAND_RS_CROSSOVER_ELEMENTS * 2 + 2)

    plan = _prepared_dispatch(allreduce, inp)

    hierarchy.all_reduce.assert_called_once_with(
        inp,
        plan=plan,
        out=None,
        blocks=None,
        stream=None,
        channel_id=None,
    )
    island.all_reduce.assert_not_called()


def test_tp16_forced_hierarchy_does_not_advertise_island_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("B12X_PCIE_ALLREDUCE_ALGORITHM", "hierarchical")
    hierarchy = MagicMock()
    hierarchy.rank = 0
    hierarchy.world_size = 16
    hierarchy.device = torch.device("cpu")
    hierarchy.should_allreduce.return_value = False
    island = MagicMock()
    island.should_allreduce.return_value = True
    allreduce = PCIeAllReduce(hierarchy, "hierarchical", island)

    assert not allreduce.should_allreduce(torch.empty(16_000))


def test_tp16_auto_falls_back_when_island_rejects_shape() -> None:
    hierarchy = MagicMock()
    hierarchy.rank = 0
    hierarchy.world_size = 16
    hierarchy.device = torch.device("cpu")
    island = MagicMock()
    island.should_allreduce.return_value = False
    allreduce = PCIeAllReduce(hierarchy, "hierarchical", island)
    inp = torch.empty(pcie_allreduce.ISLAND_RS_CROSSOVER_ELEMENTS * 2)

    plan = _prepared_dispatch(allreduce, inp)

    hierarchy.all_reduce.assert_called_once_with(
        inp,
        plan=plan,
        out=None,
        blocks=None,
        stream=None,
        channel_id=None,
    )
    island.all_reduce.assert_not_called()


def test_allreduce_factory_forwards_oneshot_channel_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = SimpleNamespace(
        rank=0,
        world_size=8,
        device=torch.device("cuda:0"),
    )
    oneshot_factory = MagicMock(return_value=runtime)
    exchange_group = object()
    monkeypatch.setattr(pcie_allreduce.dist, "get_world_size", lambda group: 8)
    monkeypatch.setattr(
        pcie_allreduce.PCIeOneshotAllReducePool,
        "from_exchange_group",
        oneshot_factory,
    )

    allreduce = PCIeAllReduce.from_exchange_group(
        exchange_group=exchange_group,  # type: ignore[arg-type]
        device="cuda:0",
        max_concurrent_channels=3,
    )

    assert allreduce.algorithm == "oneshot"
    oneshot_factory.assert_called_once_with(
        exchange_group=exchange_group,
        device="cuda:0",
        eager_buffer_bytes=8388608,
        max_size=8388608,
        rank_data_bytes=8388608,
        ext_module=None,
        single_channel=False,
        max_concurrent_channels=3,
    )


@pytest.mark.parametrize("max_concurrent_channels", [0, 2])
def test_allreduce_factory_rejects_hierarchical_concurrency(
    monkeypatch: pytest.MonkeyPatch,
    max_concurrent_channels: int,
) -> None:
    hierarchy_factory = MagicMock()
    exchange_group = object()
    monkeypatch.setattr(pcie_allreduce.dist, "get_world_size", lambda group: 16)
    monkeypatch.setattr(
        pcie_allreduce,
        "PCIeHierarchicalAllReduce",
        hierarchy_factory,
    )

    with pytest.raises(ValueError, match="exactly one concurrent channel"):
        PCIeAllReduce.from_exchange_group(
            exchange_group=exchange_group,  # type: ignore[arg-type]
            device="cuda:0",
            max_concurrent_channels=max_concurrent_channels,
        )

    hierarchy_factory.assert_not_called()


def test_allreduce_oneshot_forwards_named_channel_contract() -> None:
    runtime = MagicMock()
    runtime.rank = 0
    runtime.world_size = 8
    runtime.device = torch.device("cuda:0")
    allreduce = PCIeAllReduce(runtime, "oneshot")
    stream = object()
    inp = torch.randn(2, 4)
    out = torch.empty_like(inp)

    allreduce.prepare_channels(("target", "draft"))
    allreduce.for_stream(stream, channel_id="target")
    plan = _prepared_dispatch(allreduce, inp, out=out, stream=stream, channel_id="target")
    with allreduce.capture(stream, channel_id="target"):
        pass

    runtime.prepare_channels.assert_called_once_with(("target", "draft"))
    runtime.for_stream.assert_called_once_with(stream, channel_id="target")
    runtime.all_reduce.assert_called_once_with(
        inp,
        plan=plan,
        out=out,
        peer_input_ptrs=None,
        stream=stream,
        channel_id="target",
    )
    runtime.capture.assert_called_once_with(stream=stream, channel_id="target")


def test_allreduce_hierarchy_accepts_named_channel_contract() -> None:
    runtime = MagicMock(
        spec=[
            "rank",
            "world_size",
            "device",
            "prepare_channels",
            "for_stream",
            "all_reduce",
            "capture",
        ]
    )
    runtime.rank = 0
    runtime.world_size = 16
    runtime.device = torch.device("cuda:0")
    allreduce = PCIeAllReduce(runtime, "hierarchical")
    stream = object()
    inp = torch.randn(2, 4)
    out = torch.empty_like(inp)

    allreduce.prepare_channels(("target", "draft"))
    allreduce.for_stream(stream, channel_id="target")
    plan = _prepared_dispatch(allreduce, inp, out=out, stream=stream, channel_id="target")
    with allreduce.capture(stream, channel_id="target"):
        pass

    runtime.prepare_channels.assert_called_once_with(("target", "draft"))
    runtime.for_stream.assert_called_once_with(stream, channel_id="target")
    runtime.all_reduce.assert_called_once_with(
        inp,
        plan=plan,
        out=out,
        blocks=None,
        stream=stream,
        channel_id="target",
    )
    runtime.capture.assert_called_once_with(stream=stream, channel_id="target")


def test_hierarchical_named_channel_surface_remains_serial() -> None:
    runtime = object.__new__(PCIeHierarchicalAllReduce)
    stream = object()

    runtime.prepare_channels(("target", "draft"))
    assert runtime.for_stream(stream, channel_id="target") is runtime
    with runtime.capture(stream, channel_id="draft") as captured:
        assert captured is runtime


@pytest.mark.parametrize("world_size", [1, 3, 11, 14, 20])
def test_allreduce_rejects_unsupported_or_peer_unsafe_worlds(
    world_size: int,
) -> None:
    with pytest.raises(ValueError, match="unsupported"):
        _algorithm_for_world_size(world_size)


@pytest.mark.parametrize(
    ("world_size", "rank", "expected"),
    [
        (9, 0, (1, 2, 3, 4, 8)),
        (9, 8, (0, 4)),
        (10, 0, (1, 2, 3, 4, 8)),
        (10, 8, (0, 4, 9)),
        (10, 9, (8,)),
        (12, 0, (1, 2, 3, 4, 8)),
        (12, 1, (0,)),
        (12, 3, (0,)),
        (12, 4, (0, 5, 6, 7, 8)),
        (12, 7, (4,)),
        (12, 8, (0, 4, 9, 10, 11)),
        (12, 11, (8,)),
        (16, 0, (1, 2, 3, 4, 8, 12)),
        (16, 5, (4,)),
        (16, 12, (0, 4, 8, 13, 14, 15)),
        (16, 15, (12,)),
    ],
)
def test_selected_peers_respect_four_gpu_islands(
    world_size: int,
    rank: int,
    expected: tuple[int, ...],
) -> None:
    assert _selected_peers(rank, world_size) == expected


def test_selected_peers_never_exceed_cuda_peer_limit() -> None:
    expected_maximums = {9: 5, 10: 5, 12: 5, 16: 6}
    for world_size, expected in expected_maximums.items():
        mapped_peers = [
            len(_selected_peers(rank, world_size)) for rank in range(world_size)
        ]
        assert max(mapped_peers) == expected
        assert max(mapped_peers) <= 8


@pytest.mark.parametrize("world_size", [9, 10, 12, 16])
def test_selected_peer_graph_is_reciprocal_and_connected(world_size: int) -> None:
    peer_sets = {
        rank: set(_selected_peers(rank, world_size)) for rank in range(world_size)
    }
    for rank, peers in peer_sets.items():
        for peer in peers:
            assert rank in peer_sets[peer]

    reached = {0}
    frontier = [0]
    while frontier:
        rank = frontier.pop()
        for peer in peer_sets[rank] - reached:
            reached.add(peer)
            frontier.append(peer)
    assert reached == set(range(world_size))


@pytest.mark.parametrize("world_size", [9, 10, 12, 16])
def test_hierarchical_protocol_simulation_matches_full_sum(world_size: int) -> None:
    generator = torch.Generator().manual_seed(1234 + world_size)
    inputs = torch.randn(world_size, 257, generator=generator)
    peer_sets = {
        rank: set(_selected_peers(rank, world_size)) for rank in range(world_size)
    }
    leaders = tuple(range(0, world_size, 4))

    partials = {}
    for leader in leaders:
        local_ranks = set(range(leader, min(leader + 4, world_size)))
        assert local_ranks - {leader} <= peer_sets[leader]
        partials[leader] = inputs[leader : leader + 4].sum(dim=0)

    outputs = []
    for rank in range(world_size):
        leader = rank // 4 * 4
        if rank != leader:
            assert leader in peer_sets[rank]
        assert set(leaders) - {leader} <= peer_sets[leader]
        outputs.append(torch.stack(tuple(partials.values())).sum(dim=0))

    expected = inputs.sum(dim=0).expand(world_size, -1)
    torch.testing.assert_close(torch.stack(outputs), expected)


@pytest.mark.parametrize(
    ("rank", "world_size"),
    [(-1, 12), (12, 12), (0, 8), (0, 20)],
)
def test_selected_peers_rejects_invalid_topologies(
    rank: int,
    world_size: int,
) -> None:
    with pytest.raises(ValueError):
        _selected_peers(rank, world_size)


@pytest.mark.parametrize(
    ("elements", "expected"),
    [
        (1, 16),
        (3584, 16),
        (4096, 16),
        (4097, 32),
        (7168, 32),
    ],
)
def test_pick_blocks_for_k3_decode(elements: int, expected: int) -> None:
    assert _pick_blocks(elements) == expected


def test_pick_blocks_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="positive"):
        _pick_blocks(0)


@pytest.mark.parametrize(
    ("double_buffered", "deferred_consumption", "expected"),
    [
        ("0", "0", (False, False)),
        ("1", "0", (True, False)),
        ("0", "1", (False, True)),
    ],
)
def test_hierarchical_buffer_modes_are_opt_in(
    monkeypatch: pytest.MonkeyPatch,
    double_buffered: str,
    deferred_consumption: str,
    expected: tuple[bool, bool],
) -> None:
    monkeypatch.setenv(
        "B12X_PCIE_HIERARCHICAL_DOUBLE_BUFFER",
        double_buffered,
    )
    monkeypatch.setenv(
        "B12X_PCIE_HIERARCHICAL_DEFERRED_CONSUMPTION",
        deferred_consumption,
    )
    assert _buffer_modes_from_env() == expected


def test_hierarchical_buffer_modes_reject_ambiguous_protocol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("B12X_PCIE_HIERARCHICAL_DOUBLE_BUFFER", "1")
    monkeypatch.setenv(
        "B12X_PCIE_HIERARCHICAL_DEFERRED_CONSUMPTION",
        "1",
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        _buffer_modes_from_env()


@pytest.mark.parametrize("value", ["0", "16", "32", "64", "1024"])
def test_hierarchical_wait_nanosleep_cycles(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    monkeypatch.setenv(
        "B12X_PCIE_HIERARCHICAL_NANOSLEEP_CYCLES",
        value,
    )
    assert _wait_nanosleep_cycles_from_env() == int(value)


@pytest.mark.parametrize("value", ["-1", "1025", "not-an-int"])
def test_hierarchical_wait_nanosleep_cycles_rejects_invalid_values(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    monkeypatch.setenv(
        "B12X_PCIE_HIERARCHICAL_NANOSLEEP_CYCLES",
        value,
    )
    with pytest.raises(ValueError):
        _wait_nanosleep_cycles_from_env()


@pytest.mark.parametrize("value", ["32", "128", "224", "256", "1024"])
def test_hierarchical_threads(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    monkeypatch.setenv("B12X_PCIE_HIERARCHICAL_THREADS", value)
    assert _threads_from_env() == int(value)


@pytest.mark.parametrize("value", ["0", "31", "225", "1056", "bad"])
def test_hierarchical_threads_rejects_invalid_values(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    monkeypatch.setenv("B12X_PCIE_HIERARCHICAL_THREADS", value)
    with pytest.raises(ValueError):
        _threads_from_env()


@pytest.mark.parametrize(("value", "expected"), [("0", False), ("1", True)])
def test_hierarchical_vectorized_bf16x2(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
    expected: bool,
) -> None:
    monkeypatch.setenv("B12X_PCIE_HIERARCHICAL_BF16X2", value)
    assert _vectorized_bf16x2_from_env() is expected


def test_hierarchical_vectorized_bf16x2_rejects_invalid_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("B12X_PCIE_HIERARCHICAL_BF16X2", "true")
    with pytest.raises(ValueError):
        _vectorized_bf16x2_from_env()


@pytest.mark.parametrize("value", ["0", "7168", "57344", str(1 << 30)])
def test_hierarchical_vectorized_bf16x2_max_elements(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    monkeypatch.setenv(
        "B12X_PCIE_HIERARCHICAL_BF16X2_MAX_ELEMENTS",
        value,
    )
    assert _vectorized_bf16x2_max_elements_from_env() == int(value)


@pytest.mark.parametrize("value", ["-1", str((1 << 30) + 1), "bad"])
def test_hierarchical_vectorized_bf16x2_max_elements_rejects_invalid_value(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    monkeypatch.setenv(
        "B12X_PCIE_HIERARCHICAL_BF16X2_MAX_ELEMENTS",
        value,
    )
    with pytest.raises(ValueError):
        _vectorized_bf16x2_max_elements_from_env()


@pytest.mark.parametrize("max_elements", [1, 3584, 4096, 7168])
def test_hierarchical_slab_layout_matches_native_alignment(
    max_elements: int,
) -> None:
    layout = _make_layout(max_elements)
    assert layout.stage[0] == 69_888
    for slot in range(2):
        assert (
            layout.partial[slot]
            == ((layout.stage[slot] + max_elements * 2 + 255) // 256) * 256
        )
        assert (
            layout.final[slot]
            == ((layout.partial[slot] + max_elements * 4 + 255) // 256) * 256
        )
        if slot == 0:
            assert (
                layout.stage[1]
                == ((layout.final[0] + max_elements * 2 + 255) // 256) * 256
            )
    assert layout.bytes == ((layout.final[1] + max_elements * 2 + 255) // 256) * 256
    assert all(
        value % 256 == 0
        for offsets in (layout.stage, layout.partial, layout.final)
        for value in offsets
    )
    assert layout.bytes % 256 == 0


def test_hierarchical_protocol_intrinsics_pin_native_ptx_modifiers() -> None:
    assert "ld.acquire.sys.global.u32" in inspect.getsource(
        _hierarchical_cute._load_acquire_sys_u32
    )
    assert "st.release.sys.global.u32" in inspect.getsource(
        _hierarchical_cute._store_release_sys_u32
    )
    assert "fence.sc.sys" in inspect.getsource(_hierarchical_cute._fence_sc_sys)
    assert "nanosleep.u32 $0" in inspect.getsource(_hierarchical_cute._nanosleep)


@pytest.mark.parametrize(
    ("world_size", "rank", "island", "local_rank", "leader_rank"),
    [
        (9, 8, 2, 0, 8),
        (10, 8, 2, 0, 8),
        (10, 9, 2, 1, 8),
        (12, 0, 0, 0, 0),
        (12, 5, 1, 1, 4),
        (12, 11, 2, 3, 8),
        (16, 15, 3, 3, 12),
    ],
)
def test_hierarchical_launch_specializes_rank_topology(
    world_size: int,
    rank: int,
    island: int,
    local_rank: int,
    leader_rank: int,
) -> None:
    launch = _hierarchical_cute._HierarchicalLaunch(world_size, rank)

    assert launch._rank == rank
    assert launch._island == island
    assert launch._local_rank == local_rank
    assert launch._leader_rank == leader_rank
    assert launch._num_islands == (world_size + 3) // 4
    assert launch._local_size == min(4, world_size - leader_rank)


def test_hierarchical_launch_uses_direct_slab_pointer_parameters() -> None:
    parameters = inspect.signature(
        _hierarchical_cute._HierarchicalLaunch.kernel
    ).parameters

    assert tuple(parameters)[1:17] == tuple(f"slab{rank}" for rank in range(16))
    assert "rank" not in parameters
    assert "slabs" not in parameters


@pytest.mark.parametrize(("world_size", "rank"), [(8, 0), (12, -1), (12, 12)])
def test_hierarchical_launch_rejects_invalid_specialization(
    world_size: int, rank: int
) -> None:
    with pytest.raises(ValueError):
        _hierarchical_cute._HierarchicalLaunch(world_size, rank)
