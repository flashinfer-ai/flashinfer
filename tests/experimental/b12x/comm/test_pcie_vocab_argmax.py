from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from b12x.comm.pcie import pcie_vocab_argmax as vocab_argmax_module
from b12x.comm.pcie.pcie_hierarchical import (
    _selected_peers as _allreduce_peers,
)
from b12x.comm.pcie.pcie_vocab_argmax import (
    PCIeVocabParallelArgmax,
    _exchange_ipc_handles,
    _require_uniform_geometry,
    _selected_peers,
    _wait_nanosleep_cycles_from_env,
)


@pytest.mark.parametrize(
    ("world_size", "rank", "expected"),
    [
        (8, 0, (1, 2, 3, 4)),
        (12, 6, (2, 4, 5, 7, 10)),
        (16, 0, (1, 2, 3, 4, 8, 12)),
        (16, 15, (3, 7, 11, 12, 13, 14)),
    ],
)
def test_vocab_argmax_uses_bounded_lane_topology(
    world_size: int,
    rank: int,
    expected: tuple[int, ...],
) -> None:
    assert _selected_peers(rank, world_size) == expected


@pytest.mark.parametrize("world_size", (12, 16))
def test_vocab_argmax_and_hierarchical_allreduce_union_stays_bounded(
    world_size: int,
) -> None:
    expected_bound = world_size // 4 + 2
    for rank in range(world_size):
        peers = set(_selected_peers(rank, world_size)) | set(
            _allreduce_peers(rank, world_size)
        )
        assert len(peers) <= expected_bound


@pytest.mark.parametrize("world_size", (8, 12, 16))
def test_vocab_argmax_peer_graph_is_reciprocal_and_connected(
    world_size: int,
) -> None:
    peer_sets = {
        rank: set(_selected_peers(rank, world_size))
        for rank in range(world_size)
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


def test_vocab_argmax_exchanges_metadata_over_gloo(monkeypatch) -> None:
    group = MagicMock()
    monkeypatch.setattr(torch.distributed, "get_backend", lambda group: "gloo")
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 2)
    exchanges = []

    def fake_all_gather(objects, local_handle, group):
        exchanges.append((objects, local_handle, group))
        objects[:] = [b"rank-0", b"rank-1"]

    monkeypatch.setattr(torch.distributed, "all_gather_object", fake_all_gather)

    assert _exchange_ipc_handles(b"rank-0", group) == [b"rank-0", b"rank-1"]
    assert exchanges == [([b"rank-0", b"rank-1"], b"rank-0", group)]


def test_vocab_argmax_rejects_missing_gloo_handle(monkeypatch) -> None:
    group = MagicMock()
    monkeypatch.setattr(torch.distributed, "get_backend", lambda group: "gloo")
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 2)

    def fake_all_gather(objects, local_handle, group):
        objects[:] = [local_handle, None]

    monkeypatch.setattr(torch.distributed, "all_gather_object", fake_all_gather)

    with pytest.raises(RuntimeError, match="empty handle"):
        _exchange_ipc_handles(b"rank-0", group)


def test_vocab_argmax_rejects_mismatched_rank_geometry(monkeypatch) -> None:
    group = MagicMock()
    monkeypatch.setattr(
        vocab_argmax_module,
        "_exchange_ipc_handles",
        lambda geometry, group: [geometry] * 15 + [(2048, 4)],
    )

    with pytest.raises(RuntimeError, match="geometry differs across ranks"):
        _require_uniform_geometry(1024, 4, group)


def test_vocab_argmax_validates_uniform_geometry_over_gloo(monkeypatch) -> None:
    group = MagicMock()
    monkeypatch.setattr(torch.distributed, "get_backend", lambda group: "gloo")
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 2)

    def fake_all_gather(objects, geometry, group):
        objects[:] = [geometry, geometry]

    monkeypatch.setattr(torch.distributed, "all_gather_object", fake_all_gather)

    _require_uniform_geometry(1024, 4, group)


@pytest.mark.parametrize(("rank", "world_size"), [(-1, 8), (8, 8), (16, 16)])
def test_vocab_argmax_rejects_invalid_rank(rank: int, world_size: int) -> None:
    with pytest.raises(ValueError, match="invalid rank"):
        _selected_peers(rank, world_size)


@pytest.mark.parametrize("world_size", [1, 2, 4, 20, 32])
def test_vocab_argmax_rejects_unsupported_world(world_size: int) -> None:
    with pytest.raises(ValueError, match="requires"):
        _selected_peers(0, world_size)


@pytest.mark.parametrize("value", ["0", "24", "1024"])
def test_vocab_argmax_wait_cycles(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    monkeypatch.setenv("B12X_PCIE_VOCAB_ARGMAX_NANOSLEEP_CYCLES", value)
    assert _wait_nanosleep_cycles_from_env() == int(value)


@pytest.mark.parametrize("value", ["-1", "1025", "bad"])
def test_vocab_argmax_rejects_invalid_wait_cycles(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    monkeypatch.setenv("B12X_PCIE_VOCAB_ARGMAX_NANOSLEEP_CYCLES", value)
    with pytest.raises(ValueError):
        _wait_nanosleep_cycles_from_env()


def _fake_runtime() -> PCIeVocabParallelArgmax:
    runtime = PCIeVocabParallelArgmax.__new__(PCIeVocabParallelArgmax)
    runtime.device = torch.device("cpu")
    runtime.local_vocab_size = 16
    runtime.max_batch_size = 8
    runtime._closed = False
    runtime._launcher = MagicMock()
    runtime._slab_ptrs = tuple(range(16))
    runtime._ipc = MagicMock()
    runtime._remote_ptrs = [456]
    runtime._local_ptr = 789
    return runtime


def test_vocab_argmax_resolves_implicit_cuda_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    group = MagicMock()
    launcher = MagicMock()
    ipc = MagicMock()
    shared = SimpleNamespace(
        local_ptr=1000,
        peer_ptrs=tuple(range(16)),
        remote_ptrs=tuple(range(2000, 2006)),
    )
    allocate_shared_buffer = MagicMock(return_value=shared)
    setup_owners = []

    def run_collective_setup(*, owner, exchange_group, setup):
        assert exchange_group is group
        setup_owners.append(owner)
        return setup()

    monkeypatch.setattr(torch.distributed, "get_rank", lambda group: 0)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 16)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 7)
    monkeypatch.setattr(vocab_argmax_module, "CudaRTLibrary", lambda: ipc)
    monkeypatch.setattr(
        vocab_argmax_module,
        "get_vocab_argmax_launcher",
        lambda *args, **kwargs: launcher,
    )
    monkeypatch.setattr(
        vocab_argmax_module,
        "_require_uniform_geometry",
        lambda *args: None,
    )
    monkeypatch.setattr(
        vocab_argmax_module,
        "_run_collective_preallocation_setup",
        run_collective_setup,
    )
    monkeypatch.setattr(
        vocab_argmax_module.PCIeOneshotAllReduce,
        "_allocate_shared_buffer",
        allocate_shared_buffer,
    )

    runtime = PCIeVocabParallelArgmax(
        exchange_group=group,
        device="cuda",
        local_vocab_size=16,
    )

    assert runtime.device == torch.device("cuda:7")
    assert setup_owners == [
        "PCIe vocabulary argmax argument validation",
        "PCIe vocabulary argmax runtime preparation",
    ]
    allocate_shared_buffer.assert_called_once_with(
        group,
        vocab_argmax_module.SLAB_BYTES,
        zero_fill=True,
        ipc=ipc,
        peer_ranks=(1, 2, 3, 4, 8, 12),
    )
    assert ipc.cudaSetDevice.call_count == 2
    assert runtime._slab_ptrs == tuple(range(16))
    assert runtime._remote_ptrs == list(range(2000, 2006))
    runtime._closed = True


def test_vocab_argmax_destructor_never_enters_collective_teardown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _fake_runtime()
    runtime.group = MagicMock()
    barrier = MagicMock()
    monkeypatch.setattr(torch.distributed, "barrier", barrier)

    with pytest.warns(ResourceWarning, match="without close"):
        runtime.__del__()

    barrier.assert_not_called()
    runtime._ipc.cudaIpcCloseMemHandle.assert_called_once_with(456)
    runtime._ipc.cudaFree.assert_called_once_with(789)
    assert runtime._closed
    assert runtime._launcher is None
    assert runtime._slab_ptrs == ()
    assert runtime._remote_ptrs == []
    assert runtime._local_ptr == 0


def test_vocab_argmax_close_reaches_all_barriers_after_ipc_close_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _fake_runtime()
    runtime.group = MagicMock()
    runtime._ipc.cudaIpcCloseMemHandle.side_effect = RuntimeError(
        "IPC close failed"
    )
    barrier = MagicMock()
    monkeypatch.setattr(torch.cuda, "synchronize", MagicMock())
    monkeypatch.setattr(torch.distributed, "barrier", barrier)

    with pytest.raises(RuntimeError, match="IPC close failed"):
        runtime.close()

    assert barrier.call_count == 3
    runtime._ipc.cudaFree.assert_called_once_with(789)
    assert runtime._launcher is None
    assert runtime._slab_ptrs == ()
    assert runtime._remote_ptrs == []
    assert runtime._local_ptr == 0


def test_vocab_argmax_allocates_int64_output_and_dispatches() -> None:
    runtime = _fake_runtime()
    base = torch.randn(4, 16, dtype=torch.bfloat16)
    bias = torch.randn_like(base)

    output = runtime.fused_add_argmax(base, bias)

    assert output.shape == (4,)
    assert output.dtype == torch.int64
    runtime._launcher.assert_called_once_with(
        tuple(range(16)),
        base.data_ptr(),
        bias.data_ptr(),
        output.data_ptr(),
        16,
        base.stride(0),
        bias.stride(0),
        4,
    )


def test_vocab_argmax_accepts_row_strided_inputs() -> None:
    runtime = _fake_runtime()
    base_storage = torch.randn(4, 3, 16, dtype=torch.bfloat16)
    bias_storage = torch.randn(4, 5, 16, dtype=torch.bfloat16)
    base = base_storage[:, 1]
    bias = bias_storage[:, 3]

    assert not base.is_contiguous()
    assert not bias.is_contiguous()
    output = runtime.fused_add_argmax(base, bias)

    runtime._launcher.assert_called_once_with(
        tuple(range(16)),
        base.data_ptr(),
        bias.data_ptr(),
        output.data_ptr(),
        16,
        base.stride(0),
        bias.stride(0),
        4,
    )


def test_vocab_argmax_rejects_noncontiguous_last_dimension() -> None:
    base = torch.zeros(1, 32, dtype=torch.bfloat16)[:, ::2]
    bias = torch.zeros_like(base)

    with pytest.raises(ValueError, match="last dimensions"):
        _fake_runtime().fused_add_argmax(base, bias)


@pytest.mark.parametrize(
    ("base", "bias", "error"),
    [
        (
            torch.zeros(1, 16, dtype=torch.float32),
            torch.zeros(1, 16, dtype=torch.float32),
            "BF16",
        ),
        (
            torch.zeros(1, 15, dtype=torch.bfloat16),
            torch.zeros(1, 15, dtype=torch.bfloat16),
            "local vocabulary",
        ),
        (
            torch.zeros(9, 16, dtype=torch.bfloat16),
            torch.zeros(9, 16, dtype=torch.bfloat16),
            "capacity",
        ),
        (
            torch.zeros(1, 16, dtype=torch.bfloat16),
            torch.zeros(2, 16, dtype=torch.bfloat16),
            "matching",
        ),
    ],
)
def test_vocab_argmax_validates_inputs(
    base: torch.Tensor,
    bias: torch.Tensor,
    error: str,
) -> None:
    with pytest.raises(ValueError, match=error):
        _fake_runtime().fused_add_argmax(base, bias)
