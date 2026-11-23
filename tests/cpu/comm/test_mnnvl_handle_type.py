# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Handle-type choice and allocation fallback of MNNVL memory, on one process.

``_GroupComm`` stands in for a communicator of several ranks: ``allgather``
returns this rank's value at its own index and the other ranks' values from a
script, so collective decisions can be checked without launching ranks.
"""

import logging
import threading
from types import SimpleNamespace

import pytest

from flashinfer.comm import mnnvl

FABRIC = mnnvl._FABRIC_HANDLE
POSIX = mnnvl._POSIX_FD_HANDLE
CLUSTER = bytes(range(1, 17))
OTHER_CLUSTER = bytes(range(2, 18))


class _GroupComm:
    def __init__(self, rank, peers):
        # peers(local_value) -> the values of every rank, in rank order.
        self.rank = rank
        self.peers = peers
        self.gathered = []

    def Get_rank(self):
        return self.rank

    def Get_size(self):
        return len(self.peers(None))

    def allgather(self, value):
        values = list(self.peers(value))
        values[self.rank] = value
        self.gathered.append(values)
        return values


def _status_comm(statuses):
    """A rank-0 communicator whose peers report ``statuses`` (host, reason, uuid)."""
    return _GroupComm(0, lambda value: list(statuses))


@pytest.fixture(autouse=True)
def _one_host(monkeypatch):
    monkeypatch.setattr(mnnvl.socket, "gethostname", lambda: "node0")
    monkeypatch.setattr(mnnvl, "_fabric_fallback_warned", False)


def _local_status(monkeypatch, reason=None, uuid=CLUSTER):
    monkeypatch.setattr(mnnvl, "_local_fabric_status", lambda dev_id: (reason, uuid))


@pytest.mark.cpu_only
def test_single_host_prefers_fabric(monkeypatch):
    _local_status(monkeypatch)
    comm = _status_comm([("node0", None, b""), ("node0", None, b"")])
    assert mnnvl.choose_mnnvl_handle_type(comm, 0) == (FABRIC, True)


@pytest.mark.cpu_only
def test_single_host_uses_posix_when_any_rank_lacks_fabric(monkeypatch):
    _local_status(monkeypatch)
    comm = _status_comm(
        [("node0", None, b""), ("node0", "the driver does not support FABRIC", b"")]
    )
    assert mnnvl.choose_mnnvl_handle_type(comm, 0) == (POSIX, True)


@pytest.mark.cpu_only
def test_multi_host_needs_one_fabric_cluster(monkeypatch):
    _local_status(monkeypatch)
    comm = _status_comm([("node0", None, CLUSTER), ("node1", None, CLUSTER)])
    assert mnnvl.choose_mnnvl_handle_type(comm, 0) == (FABRIC, False)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("peer", "message"),
    [
        (("node1", "the driver does not support FABRIC handles", b""), "rank 1"),
        (("node1", None, bytes(16)), "clusterUuid unset"),
        (("node1", None, OTHER_CLUSTER), "different NVLink domains"),
    ],
)
def test_multi_host_rejects_ranks_outside_the_cluster(monkeypatch, peer, message):
    _local_status(monkeypatch)
    comm = _status_comm([("node0", None, CLUSTER), peer])
    with pytest.raises(RuntimeError, match=message) as error:
        mnnvl.choose_mnnvl_handle_type(comm, 0)
    assert "IMEX" in str(error.value)


def _attempts(monkeypatch, results):
    """Script _create_and_import_handles: one (handles, errors) per attempt."""
    attempts = []

    def attempt(comm, prop, size, local_error=None):
        attempts.append(prop)
        return results[len(attempts) - 1]

    monkeypatch.setattr(mnnvl, "_create_and_import_handles", attempt)
    return attempts


def _choose(monkeypatch, handle_type, single_host):
    monkeypatch.setattr(
        mnnvl,
        "choose_mnnvl_handle_type",
        lambda comm, dev_id: (handle_type, single_host),
    )


_FABRIC_FAILS = (
    None,
    [None, "creating or exporting the allocation failed: NOT_PERMITTED"],
)
_SUCCEEDS = (["h0", "h1"], [None, None])


def _allocate():
    return mnnvl.allocate_mnnvl_handles(None, 0, lambda handle_type: (handle_type, 64))


@pytest.mark.cpu_only
def test_allocation_uses_the_chosen_type(monkeypatch):
    _choose(monkeypatch, FABRIC, single_host=True)
    attempts = _attempts(monkeypatch, [_SUCCEEDS])
    assert _allocate() == (["h0", "h1"], FABRIC)
    assert attempts == [FABRIC]


@pytest.mark.cpu_only
def test_single_host_falls_back_to_posix_on_every_allocation(monkeypatch, caplog):
    # Nothing is remembered between allocations, so ranks that allocate over
    # different groups always decide alike; each allocation tries FABRIC again.
    caplog.set_level(logging.DEBUG, logger=mnnvl.logger.name)
    _choose(monkeypatch, FABRIC, single_host=True)
    attempts = _attempts(monkeypatch, [_FABRIC_FAILS, _SUCCEEDS] * 2)
    assert _allocate() == (["h0", "h1"], POSIX)
    assert _allocate() == (["h0", "h1"], POSIX)
    assert attempts == [FABRIC, POSIX, FABRIC, POSIX]
    levels = [r.levelno for r in caplog.records if "rank 1: creating" in r.message]
    assert levels == [logging.WARNING, logging.DEBUG]


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("handle_type", "single_host"), [(FABRIC, False), (POSIX, True)]
)
def test_multi_host_or_posix_allocations_do_not_fall_back(
    monkeypatch, handle_type, single_host
):
    _choose(monkeypatch, handle_type, single_host)
    attempts = _attempts(monkeypatch, [(None, ["importing failed", None])])
    with pytest.raises(RuntimeError, match="rank 0: importing failed") as error:
        _allocate()
    assert attempts == [handle_type]
    assert ("IMEX" in str(error.value)) == (handle_type == FABRIC)


class _FakeDriver:
    """The cuda-python calls _create_and_import_handles makes, with failures on demand."""

    def __init__(self, fail_create=False, fail_import=False):
        self.fail_create = fail_create
        self.fail_import = fail_import
        self.released = []
        self.next_handle = 100

    @staticmethod
    def _ok(value):
        return (SimpleNamespace(value=0), value)

    @staticmethod
    def _error():
        return (SimpleNamespace(value=1),)

    def cuMemCreate(self, size, prop, flags):
        if self.fail_create:
            return self._error()
        self.next_handle += 1
        return self._ok(self.next_handle)

    def cuMemExportToShareableHandle(self, handle, handle_type, flags):
        if handle_type == POSIX:
            return self._ok(handle + 1000)
        return self._ok(SimpleNamespace(data=f"export-{handle}".encode()))

    def cuMemImportFromShareableHandle(self, data, handle_type):
        if self.fail_import:
            return self._error()
        self.next_handle += 1
        return self._ok(self.next_handle)

    def cuMemRelease(self, handle):
        self.released.append(handle)
        return (SimpleNamespace(value=0),)


@pytest.fixture
def fake_driver(monkeypatch):
    def install(**failures):
        driver = _FakeDriver(**failures)
        monkeypatch.setattr(mnnvl, "cuda", driver)
        monkeypatch.setattr(mnnvl, "checkCudaErrors", _check)
        return driver

    return install


def _check(result):
    if result[0].value:
        raise RuntimeError("CUDA error")
    return result[1] if len(result) == 2 else None


def _two_rank_comm(peer_error=None):
    # The peer exports its own data and reports peer_error in every vote.
    def peers(value):
        if isinstance(value, bytes):
            return [value, b"export-peer"]
        return [value, peer_error]

    return _GroupComm(0, peers)


@pytest.mark.cpu_only
def test_attempt_imports_every_peer(fake_driver):
    driver = fake_driver()
    prop = SimpleNamespace(requestedHandleTypes=FABRIC)
    handles, errors = mnnvl._create_and_import_handles(_two_rank_comm(), prop, 64)
    assert handles == [101, 102] and errors == [None, None]
    assert driver.released == []


@pytest.mark.cpu_only
def test_attempt_releases_its_allocation_when_a_peer_fails(fake_driver):
    driver = fake_driver()
    prop = SimpleNamespace(requestedHandleTypes=FABRIC)
    handles, errors = mnnvl._create_and_import_handles(
        _two_rank_comm(peer_error="creating failed"), prop, 64
    )
    assert handles is None and errors == [None, "creating failed"]
    assert driver.released == [101]


@pytest.mark.cpu_only
def test_attempt_votes_on_a_local_import_failure(fake_driver):
    driver = fake_driver(fail_import=True)
    prop = SimpleNamespace(requestedHandleTypes=FABRIC)
    handles, errors = mnnvl._create_and_import_handles(_two_rank_comm(), prop, 64)
    assert handles is None and "importing the allocation of rank 1" in errors[0]
    assert driver.released == [101]


@pytest.mark.cpu_only
def test_attempt_releases_its_allocation_when_the_fd_exchange_fails(
    fake_driver, monkeypatch
):
    driver = fake_driver()
    closed = []
    monkeypatch.setattr(mnnvl, "os", SimpleNamespace(close=closed.append))

    def exchange_fds(comm, fd):
        raise ConnectionError("peer went away")

    monkeypatch.setattr(mnnvl, "exchange_fds", exchange_fds)
    prop = SimpleNamespace(requestedHandleTypes=POSIX)
    with pytest.raises(ConnectionError):
        mnnvl._create_and_import_handles(_two_rank_comm(), prop, 64)
    assert driver.released == [101] and closed == [1101]


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("granularities", "expected"),
    [
        ({FABRIC: 1 << 21, POSIX: 1 << 21}, 1 << 21),
        ({FABRIC: 1 << 22, POSIX: 1 << 21}, 1 << 22),
        ({POSIX: 1 << 21}, 1 << 21),
    ],
)
def test_granularity_fits_both_handle_types(monkeypatch, granularities, expected):
    # A type missing from ``granularities`` is one the driver rejects.
    def granularity(prop, option):
        if prop.requestedHandleTypes in granularities:
            return (SimpleNamespace(value=0), granularities[prop.requestedHandleTypes])
        return (SimpleNamespace(value=1),)

    monkeypatch.setattr(mnnvl, "checkCudaErrors", _check)
    monkeypatch.setattr(mnnvl.cuda, "cuMemGetAllocationGranularity", granularity)
    monkeypatch.setattr(mnnvl.MnnvlMemory, "allocation_granularity", 0)
    assert mnnvl.MnnvlMemory.get_allocation_granularity(0) == expected


# ---------------------------------------------------------------------------
# A make_prop failure on one rank, with two ranks on threads.
# ---------------------------------------------------------------------------


class _ThreadGroup:
    """An in-process allgather over threads. A rank that makes fewer or more
    collective calls than its peer breaks the barrier instead of hanging."""

    def __init__(self, size=2, timeout=10.0):
        self.barrier = threading.Barrier(size, timeout=timeout)
        self.slots = [None] * size
        self.calls = [0] * size

    def comm(self, rank):
        group = self

        class _Comm:
            def Get_rank(self):
                return rank

            def Get_size(self):
                return len(group.slots)

            def allgather(self, value):
                group.calls[rank] += 1
                group.slots[rank] = value
                group.barrier.wait()
                values = list(group.slots)
                group.barrier.wait()
                return values

        return _Comm()


def _allocate_on_two_ranks(monkeypatch, fake_driver, single_host, failing):
    """Run allocate_mnnvl_handles on two ranks; ``failing`` maps a rank to the
    handle types for which its make_prop raises. Returns each rank's outcome
    and the collective calls per rank."""
    fake_driver()
    closed = []
    monkeypatch.setattr(mnnvl, "os", SimpleNamespace(close=closed.append))

    def choose(comm, dev_id):
        comm.allgather(None)
        return FABRIC, single_host

    monkeypatch.setattr(mnnvl, "choose_mnnvl_handle_type", choose)

    def exchange_fds(comm, fd):
        return comm.allgather(fd)

    monkeypatch.setattr(mnnvl, "exchange_fds", exchange_fds)
    group = _ThreadGroup()
    outcomes = [None, None]

    def rank_main(rank):
        def make_prop(handle_type):
            if handle_type in failing.get(rank, ()):
                raise RuntimeError(f"injected granularity failure ({handle_type})")
            return SimpleNamespace(requestedHandleTypes=handle_type), 64

        try:
            handles, handle_type = mnnvl.allocate_mnnvl_handles(
                group.comm(rank), 0, make_prop
            )
            outcomes[rank] = ("ok", handle_type, len(handles))
        except BaseException as exc:  # noqa: BLE001 - compared below
            outcomes[rank] = ("raised", type(exc), str(exc))

    threads = [threading.Thread(target=rank_main, args=(r,)) for r in range(2)]
    for th in threads:
        th.start()
    for th in threads:
        th.join(30)
    assert not any(th.is_alive() for th in threads), "a rank hung"
    return outcomes, group.calls


@pytest.mark.cpu_only
@pytest.mark.parametrize("failing_rank", [0, 1])
def test_make_prop_failure_on_fabric_retries_posix_on_every_rank(
    monkeypatch, fake_driver, failing_rank
):
    outcomes, calls = _allocate_on_two_ranks(
        monkeypatch, fake_driver, single_host=True, failing={failing_rank: {FABRIC}}
    )
    assert outcomes == [("ok", POSIX, 2)] * 2
    assert calls[0] == calls[1]


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "failing", [{1: {POSIX}}, {0: {FABRIC}, 1: {POSIX}}, {0: {FABRIC, POSIX}}]
)
def test_make_prop_failure_on_the_posix_retry_raises_on_every_rank(
    monkeypatch, fake_driver, failing
):
    # A FABRIC failure (make_prop on one rank, or the allocation) starts the
    # retry; a make_prop failure on the retry then raises on both ranks.
    if not any(FABRIC in types for types in failing.values()):
        monkeypatch.setattr(_FakeDriver, "cuMemCreate", _fail_fabric_create)
    outcomes, calls = _allocate_on_two_ranks(
        monkeypatch, fake_driver, single_host=True, failing=failing
    )
    for kind, error_type, message in outcomes:
        assert kind == "raised" and error_type is RuntimeError, outcomes
        assert "POSIX file descriptor handles failed" in message
        assert "querying the allocation properties failed" in message
    assert calls[0] == calls[1]


def _fail_fabric_create(self, size, prop, flags):
    if prop.requestedHandleTypes == FABRIC:
        return self._error()
    self.next_handle += 1
    return self._ok(self.next_handle)


@pytest.mark.cpu_only
@pytest.mark.parametrize("failing_rank", [0, 1])
def test_make_prop_failure_across_hosts_raises_on_every_rank(
    monkeypatch, fake_driver, failing_rank
):
    outcomes, calls = _allocate_on_two_ranks(
        monkeypatch, fake_driver, single_host=False, failing={failing_rank: {FABRIC}}
    )
    for kind, error_type, message in outcomes:
        assert kind == "raised" and error_type is RuntimeError, outcomes
        assert f"rank {failing_rank}: querying the allocation properties" in message
        assert "IMEX" in message
    assert calls[0] == calls[1]
