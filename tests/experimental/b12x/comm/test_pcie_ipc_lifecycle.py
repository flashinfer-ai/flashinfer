from __future__ import annotations

import gc
import weakref

import pytest
import torch

from b12x.comm.pcie.pcie_dcp_a2a import PCIeDCPA2A, PCIeDCPA2APool
from b12x.comm.pcie.pcie_oneshot import (
    _ABANDONED_PCIE_RUNTIME_QUARANTINE,
    PCIeOneshotAllReduce,
    PCIeOneshotAllReducePool,
)
from b12x.comm.pcie.pcie_twoshot import PCIeTwoShotSP


class _FakeChannel:
    def __init__(self, name: str, events: list[str]) -> None:
        self.name = name
        self.events = events

    def _close_ipc_imports(self) -> None:
        self.events.append(f"imports:{self.name}")

    def _free_ipc_exports(self) -> None:
        self.events.append(f"exports:{self.name}")


def test_pool_explicit_close_coordinates_imports_before_exports(monkeypatch) -> None:
    events = []
    group = object()
    channel_a = _FakeChannel("a", events)
    channel_b = _FakeChannel("b", events)
    pool = PCIeOneshotAllReducePool(
        rank=0,
        world_size=2,
        device=torch.device("cuda:0"),
        exchange_group=group,
        channel_factory=lambda stream_key: channel_a,
    )
    pool._all_channels = [channel_a, channel_b]
    pool._channels = {1: channel_a, 2: channel_b}
    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_oneshot.dist.barrier",
        lambda *, group, device_ids: events.append(("barrier", device_ids)),
    )
    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_oneshot.torch.cuda.synchronize",
        lambda device: events.append("synchronize"),
    )
    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_oneshot._broadcast_gather_object",
        lambda local_status, group: [local_status, ()],
    )

    pool.close()
    pool.close()

    assert events == [
        "synchronize",
        ("barrier", [0]),
        "imports:a",
        "imports:b",
        ("barrier", [0]),
        "exports:a",
        "exports:b",
        ("barrier", [0]),
    ]
    assert pool._closed
    assert pool._all_channels == []
    assert pool._channels == {}


def test_pool_destructor_is_nonblocking_and_does_not_touch_cuda_ownership(
    monkeypatch,
) -> None:
    events = []
    channel = _FakeChannel("a", events)
    pool = PCIeOneshotAllReducePool(
        rank=0,
        world_size=2,
        device=torch.device("cpu"),
        exchange_group=object(),
        channel_factory=lambda stream_key: pytest.fail("factory must not run"),
    )
    pool._all_channels = [channel]
    pool._channels = {1: channel}
    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_oneshot.dist.barrier",
        lambda *, group: events.append("barrier"),
    )

    pool.__del__()

    assert events == []


class _FakeExt:
    def __init__(self, events: list[object], *, dispose_failures: int = 0) -> None:
        self.events = events
        self.dispose_failures = dispose_failures

    def dispose(self, ptr: int) -> None:
        self.events.append(("dispose", ptr))
        if self.dispose_failures:
            self.dispose_failures -= 1
            raise RuntimeError("dispose failed")

    def dispose_best_effort(self, ptr: int) -> None:
        self.events.append(("dispose_best_effort", ptr))


class _FakeIPC:
    def __init__(
        self,
        events: list[object],
        *,
        close_failures: dict[int, int] | None = None,
        free_failures: dict[int, int] | None = None,
    ) -> None:
        self.events = events
        self.close_failures = dict(close_failures or {})
        self.free_failures = dict(free_failures or {})

    def cudaIpcCloseMemHandle(self, ptr: int) -> None:
        self.events.append(("close", ptr))
        if self.close_failures.get(ptr, 0):
            self.close_failures[ptr] -= 1
            raise RuntimeError(f"close {ptr} failed")

    def cudaFree(self, ptr: int) -> None:
        self.events.append(("free", ptr))
        if self.free_failures.get(ptr, 0):
            self.free_failures[ptr] -= 1
            raise RuntimeError(f"free {ptr} failed")


class _Shared:
    def __init__(self, local_ptr: int, remote_ptrs: tuple[int, ...]) -> None:
        self.local_ptr = local_ptr
        self.remote_ptrs = remote_ptrs


def _fake_twoshot(events: list[object]) -> PCIeTwoShotSP:
    runtime = object.__new__(PCIeTwoShotSP)
    runtime.rank = 0
    runtime.world_size = 2
    runtime.device = torch.device("cpu")
    runtime.exchange_group = object()
    runtime._owned_buffers = [_Shared(1000, (2000, 3000))]
    runtime._ipc = _FakeIPC(events)
    runtime.max_rows = 8
    runtime.row_elems = 16
    runtime._closed = False
    runtime._ipc_imports_closed = False
    runtime._ipc_exports_freed = False
    runtime._coordinated_close_complete = False
    runtime._closed_ipc_import_indices = set()
    return runtime


def _fake_oneshot(events: list[object]) -> PCIeOneshotAllReduce:
    runtime = object.__new__(PCIeOneshotAllReduce)
    runtime.rank = 0
    runtime.world_size = 2
    runtime.device = torch.device("cpu")
    runtime.exchange_group = None
    runtime._ext = _FakeExt(events)
    runtime._ptr = 123
    runtime._owned_buffers = [_Shared(1000, (2000, 3000))]
    runtime._ipc = _FakeIPC(events)
    runtime._registered_input_ptrs = {10: (20,)}
    runtime._closed = False
    runtime._ipc_imports_closed = False
    runtime._ipc_exports_freed = False
    return runtime


def _fake_dcp(events: list[object]) -> PCIeDCPA2A:
    runtime = object.__new__(PCIeDCPA2A)
    runtime.rank = 0
    runtime.world_size = 2
    runtime.device = torch.device("cpu")
    runtime.exchange_group = None
    runtime._legacy_ext_module = _FakeExt(events)
    runtime._ptr = 123
    runtime._owned_buffers = [_Shared(1000, (2000, 3000))]
    runtime._ipc = _FakeIPC(events)
    runtime._closed = False
    runtime._ipc_imports_closed = False
    runtime._ipc_exports_freed = False
    runtime._coordinated_close_complete = False
    runtime._closed_ipc_import_indices = set()
    return runtime


def test_dcp_explicit_close_retries_failed_unmap_before_export_free() -> None:
    events: list[object] = []
    runtime = _fake_dcp(events)
    runtime._ipc = _FakeIPC(events, close_failures={2000: 1})

    with pytest.raises(RuntimeError, match="failed during IPC unmap"):
        runtime.close()

    assert events == [
        ("dispose", 123),
        ("close", 2000),
        ("close", 3000),
    ]
    assert runtime._ptr == 0
    assert not runtime._ipc_imports_closed
    assert not runtime._ipc_exports_freed

    runtime.close()
    assert events[-2:] == [("close", 2000), ("free", 1000)]
    assert runtime._ipc_imports_closed
    assert runtime._ipc_exports_freed


def test_dcp_pool_failed_rollback_retains_transient_channel() -> None:
    retained = _fake_dcp([])
    transient_events: list[object] = []
    transient = _fake_dcp(transient_events)
    transient._ipc = _FakeIPC(transient_events, close_failures={2000: 1})
    pool = PCIeDCPA2APool(
        rank=0,
        world_size=2,
        device=torch.device("cpu"),
        max_batch_size=4,
        total_heads=32,
        head_dim=64,
        channel_factory=lambda stream_key: retained,
    )
    pool._all_channels = [retained, transient]
    pool._channels = {1: retained, 2: transient}

    with pytest.raises(RuntimeError, match="failed during IPC unmap"):
        pool.rollback_channels((1, {1: retained}))

    assert pool._all_channels == [retained, transient]
    assert pool._channels == {1: retained, 2: transient}
    assert ("free", 1000) not in transient_events


def test_twoshot_explicit_close_coordinates_unmap_then_free(monkeypatch) -> None:
    events: list[object] = []
    runtime = _fake_twoshot(events)
    runtime.device = torch.device("cuda:0")
    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_oneshot.dist.barrier",
        lambda *, group, device_ids: events.append(("barrier", device_ids)),
    )
    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_oneshot.torch.cuda.synchronize",
        lambda device: events.append("synchronize"),
    )
    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_oneshot._broadcast_gather_object",
        lambda local_status, group: [local_status, ()],
    )

    runtime.close()
    runtime.close()

    assert events == [
        "synchronize",
        ("barrier", [0]),
        ("close", 2000),
        ("close", 3000),
        ("barrier", [0]),
        ("free", 1000),
        ("barrier", [0]),
    ]
    assert runtime._owned_buffers == []
    assert runtime._ipc_imports_closed
    assert runtime._ipc_exports_freed


def test_twoshot_destructor_retains_imports_and_exports_without_cuda_calls(
    monkeypatch,
) -> None:
    events: list[object] = []
    runtime = _fake_twoshot(events)
    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_oneshot.dist.barrier",
        lambda *, group: events.append("barrier"),
    )

    runtime.__del__()

    assert events == []
    assert runtime._owned_buffers != []
    assert not runtime._ipc_exports_freed


@pytest.mark.parametrize("runtime_factory", (_fake_dcp, _fake_twoshot))
def test_real_gc_quarantines_dcp_and_twoshot_ownership(runtime_factory) -> None:
    events: list[object] = []
    runtime = runtime_factory(events)
    runtime_id = id(runtime)
    runtime_ref = weakref.ref(runtime)

    del runtime
    gc.collect()

    retained = _ABANDONED_PCIE_RUNTIME_QUARANTINE.pop(runtime_id)
    assert runtime_ref() is retained
    assert events == []
    if isinstance(retained, PCIeDCPA2A):
        assert retained._ptr == 123
    assert retained._owned_buffers
    assert not retained._ipc_exports_freed

    # Production deliberately keeps this ownership until process teardown.
    # The fake can be released after marking the coordinated path complete.
    retained._coordinated_close_complete = True
    del retained
    gc.collect()
    assert runtime_ref() is None


@pytest.mark.parametrize("runtime_factory", (_fake_dcp, _fake_twoshot))
def test_explicitly_closed_dcp_and_twoshot_are_not_quarantined(runtime_factory) -> None:
    events: list[object] = []
    runtime = runtime_factory(events)
    runtime.exchange_group = None
    runtime.close()
    runtime_id = id(runtime)
    runtime_ref = weakref.ref(runtime)

    del runtime
    gc.collect()

    assert runtime_ref() is None
    assert runtime_id not in _ABANDONED_PCIE_RUNTIME_QUARANTINE


def test_real_gc_quarantines_dcp_pool_and_its_channel() -> None:
    channel = _fake_dcp([])
    pool = PCIeDCPA2APool(
        rank=0,
        world_size=2,
        device=torch.device("cpu"),
        max_batch_size=4,
        total_heads=32,
        head_dim=64,
        channel_factory=lambda stream_key: pytest.fail("factory must not run"),
    )
    pool._all_channels = [channel]
    pool._channels = {0: channel}
    pool_id = id(pool)
    pool_ref = weakref.ref(pool)

    del pool
    gc.collect()

    retained = _ABANDONED_PCIE_RUNTIME_QUARANTINE.pop(pool_id)
    assert pool_ref() is retained
    assert retained._all_channels == [channel]
    assert retained._channels

    retained._closed = True
    channel._coordinated_close_complete = True
    del retained
    del channel
    gc.collect()
    assert pool_ref() is None


@pytest.mark.parametrize("runtime_factory", (_fake_oneshot, _fake_twoshot))
def test_strict_close_reports_unmap_failure_retains_exports_and_retries_only_failure(
    runtime_factory,
) -> None:
    events: list[object] = []
    runtime = runtime_factory(events)
    runtime.exchange_group = None
    runtime._ipc = _FakeIPC(events, close_failures={2000: 1})

    with pytest.raises(
        RuntimeError,
        match=r"coordinated PCIe close failed during IPC unmap.*not freed",
    ):
        runtime.close()

    expected_events = (
        [] if isinstance(runtime, PCIeTwoShotSP) else [("dispose", 123)]
    )
    assert events == expected_events + [
        ("close", 2000),
        ("close", 3000),
    ]
    if isinstance(runtime, PCIeOneshotAllReduce):
        assert runtime._ptr == 0
    assert runtime._closed
    assert not runtime._ipc_imports_closed
    assert not runtime._ipc_exports_freed
    assert [shared.local_ptr for shared in runtime._owned_buffers] == [1000]

    runtime.close()

    assert events == expected_events + [
        ("close", 2000),
        ("close", 3000),
        ("close", 2000),
        ("free", 1000),
    ]
    assert runtime._ipc_imports_closed
    assert runtime._ipc_exports_freed
    assert runtime._owned_buffers == []


@pytest.mark.parametrize("runtime_factory", (_fake_oneshot,))
def test_strict_close_reports_native_dispose_failure_without_losing_pointer(
    runtime_factory,
) -> None:
    events: list[object] = []
    runtime = runtime_factory(events)
    runtime.exchange_group = None
    runtime._ext = _FakeExt(events, dispose_failures=1)

    with pytest.raises(RuntimeError, match="failed during IPC unmap"):
        runtime.close()

    assert runtime._ptr == 123
    assert events == [
        ("dispose", 123),
        ("close", 2000),
        ("close", 3000),
    ]
    assert not runtime._ipc_imports_closed
    assert not runtime._ipc_exports_freed

    runtime.close()

    assert runtime._ptr == 0
    assert events == [
        ("dispose", 123),
        ("close", 2000),
        ("close", 3000),
        ("dispose", 123),
        ("free", 1000),
    ]
    assert runtime._ipc_imports_closed
    assert runtime._ipc_exports_freed


@pytest.mark.parametrize("runtime_factory", (_fake_oneshot, _fake_twoshot))
def test_strict_close_reports_free_failure_and_retains_only_failed_export(
    runtime_factory,
) -> None:
    events: list[object] = []
    runtime = runtime_factory(events)
    runtime.exchange_group = None
    runtime._owned_buffers.append(_Shared(4000, (5000,)))
    runtime._ipc = _FakeIPC(events, free_failures={1000: 1})

    with pytest.raises(RuntimeError, match="failed during IPC export free"):
        runtime.close()

    expected_events = (
        [] if isinstance(runtime, PCIeTwoShotSP) else [("dispose", 123)]
    )
    assert events == expected_events + [
        ("close", 2000),
        ("close", 3000),
        ("close", 5000),
        ("free", 1000),
        ("free", 4000),
    ]
    assert runtime._ipc_imports_closed
    assert not runtime._ipc_exports_freed
    assert [shared.local_ptr for shared in runtime._owned_buffers] == [1000]

    runtime.close()

    assert events[-1] == ("free", 1000)
    assert runtime._ipc_exports_freed
    assert runtime._owned_buffers == []


def test_peer_unmap_failure_is_exchanged_before_any_local_export_free(
    monkeypatch,
) -> None:
    events: list[object] = []
    runtime = _fake_oneshot(events)
    runtime.device = torch.device("cuda:0")
    runtime.exchange_group = object()
    peer_statuses = iter(
        [
            lambda local: [local, ("peer close failed",)],
            lambda local: [local, ()],
            lambda local: [local, ()],
        ]
    )

    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_oneshot.dist.barrier",
        lambda *, group, device_ids: events.append(("barrier", device_ids)),
    )
    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_oneshot.torch.cuda.synchronize",
        lambda device: events.append("synchronize"),
    )
    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_oneshot._broadcast_gather_object",
        lambda local_status, group: next(peer_statuses)(local_status),
    )

    with pytest.raises(RuntimeError, match="group rank 1: peer close failed"):
        runtime.close()

    assert ("free", 1000) not in events
    assert runtime._ipc_imports_closed
    assert not runtime._ipc_exports_freed

    runtime.close()

    expected_events = [
        "synchronize",
        ("barrier", [0]),
    ]
    if isinstance(runtime, PCIeOneshotAllReduce):
        expected_events.append(("dispose", 123))
    expected_events += [
        ("close", 2000),
        ("close", 3000),
        "synchronize",
        ("barrier", [0]),
        ("barrier", [0]),
        ("free", 1000),
        ("barrier", [0]),
    ]
    assert events == expected_events
    assert runtime._ipc_exports_freed


def test_status_exchange_failure_retains_local_exports(monkeypatch) -> None:
    events: list[object] = []
    runtime = _fake_oneshot(events)
    runtime.exchange_group = object()

    def fail_status_exchange(local_status, group):
        raise RuntimeError("status exchange failed")

    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_oneshot.dist.barrier",
        lambda *, group: events.append("barrier"),
    )
    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_oneshot._broadcast_gather_object",
        fail_status_exchange,
    )

    with pytest.raises(RuntimeError, match="failed to exchange peer IPC unmap"):
        runtime.close()

    assert events == [
        "barrier",
        ("dispose", 123),
        ("close", 2000),
        ("close", 3000),
    ]
    assert runtime._ipc_imports_closed
    assert not runtime._ipc_exports_freed
    assert runtime._owned_buffers


@pytest.mark.parametrize("runtime_factory", (_fake_oneshot, _fake_twoshot))
def test_rank_that_freed_locally_retries_until_peer_free_succeeds(
    monkeypatch,
    runtime_factory,
) -> None:
    events: list[object] = []
    runtime = runtime_factory(events)
    runtime.device = torch.device("cuda:0")
    runtime.exchange_group = object()
    peer_statuses = iter(
        [
            lambda local: [local, ()],
            lambda local: [local, ("peer export free failed",)],
            lambda local: [local, ()],
            lambda local: [local, ()],
        ]
    )

    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_oneshot.dist.barrier",
        lambda *, group, device_ids: events.append(("barrier", device_ids)),
    )
    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_oneshot.torch.cuda.synchronize",
        lambda device: events.append("synchronize"),
    )
    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_oneshot._broadcast_gather_object",
        lambda local_status, group: next(peer_statuses)(local_status),
    )

    with pytest.raises(RuntimeError, match="group rank 1: peer export free failed"):
        runtime.close()

    assert runtime._ipc_exports_freed
    assert not getattr(runtime, "_coordinated_close_complete", False)

    runtime.close()
    events_after_success = list(events)
    runtime.close()

    assert events == events_after_success
    expected_events = [
        "synchronize",
        ("barrier", [0]),
    ]
    if isinstance(runtime, PCIeOneshotAllReduce):
        expected_events.append(("dispose", 123))
    expected_events += [
        ("close", 2000),
        ("close", 3000),
        ("barrier", [0]),
        ("free", 1000),
        "synchronize",
        ("barrier", [0]),
        ("barrier", [0]),
        ("barrier", [0]),
    ]
    assert events == expected_events
    assert runtime._coordinated_close_complete


@pytest.mark.parametrize("runtime_factory", (_fake_oneshot, _fake_twoshot))
def test_gc_never_attempts_native_dispose_or_ipc_unmap(
    runtime_factory,
) -> None:
    events: list[object] = []
    runtime = runtime_factory(events)
    runtime._ipc = _FakeIPC(events, close_failures={2000: 1})

    runtime.__del__()

    assert events == []
    if isinstance(runtime, PCIeOneshotAllReduce):
        assert runtime._ptr == 123
    assert not runtime._ipc_imports_closed
    assert not runtime._ipc_exports_freed
    assert runtime._owned_buffers


def test_gc_retention_preserves_explicit_close_retry_path() -> None:
    events: list[object] = []
    runtime = _fake_oneshot(events)

    runtime.__del__()

    assert events == []
    assert runtime._ptr == 123
    assert not runtime._ipc_imports_closed
    runtime.close()
    assert events == [
        ("dispose", 123),
        ("close", 2000),
        ("close", 3000),
        ("free", 1000),
    ]
    assert runtime._ipc_exports_freed


def test_failed_rollback_keeps_transient_channel_owned_for_retry() -> None:
    retained_events: list[object] = []
    transient_events: list[object] = []
    retained = _fake_oneshot(retained_events)
    transient = _fake_oneshot(transient_events)
    transient._ipc = _FakeIPC(transient_events, close_failures={2000: 1})
    pool = PCIeOneshotAllReducePool(
        rank=0,
        world_size=2,
        device=torch.device("cpu"),
        channel_factory=lambda stream_key: retained,
    )
    pool._all_channels = [retained, transient]
    pool._channels = {1: retained, 2: transient}
    checkpoint = (1, {1: retained})

    with pytest.raises(RuntimeError, match="failed during IPC unmap"):
        pool.rollback_channels(checkpoint)

    assert pool._all_channels == [retained, transient]
    assert pool._channels == {1: retained, 2: transient}
    assert not pool._closed
    assert ("free", 1000) not in transient_events
