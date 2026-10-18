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
def _fresh_choices(monkeypatch):
    monkeypatch.setattr(mnnvl, "_HANDLE_TYPE_CHOICES", {})
    monkeypatch.setattr(mnnvl.socket, "gethostname", lambda: "node0")


def _local_status(monkeypatch, reason=None, uuid=CLUSTER):
    monkeypatch.setattr(mnnvl, "_local_fabric_status", lambda dev_id: (reason, uuid))


def test_single_host_prefers_fabric_tentatively(monkeypatch):
    _local_status(monkeypatch)
    comm = _status_comm([("node0", None, b""), ("node0", None, b"")])
    choice = mnnvl.choose_mnnvl_handle_type(comm, 0)
    assert (choice.handle_type, choice.final, choice.single_host) == (
        FABRIC,
        False,
        True,
    )
    # One choice per set of hosts.
    assert mnnvl.choose_mnnvl_handle_type(comm, 0) is choice


def test_single_host_uses_posix_when_any_rank_lacks_fabric(monkeypatch):
    _local_status(monkeypatch)
    comm = _status_comm(
        [("node0", None, b""), ("node0", "the driver does not support FABRIC", b"")]
    )
    choice = mnnvl.choose_mnnvl_handle_type(comm, 0)
    assert (choice.handle_type, choice.final) == (POSIX, True)


def test_multi_host_needs_one_fabric_cluster(monkeypatch):
    _local_status(monkeypatch)
    comm = _status_comm([("node0", None, CLUSTER), ("node1", None, CLUSTER)])
    choice = mnnvl.choose_mnnvl_handle_type(comm, 0)
    assert (choice.handle_type, choice.final, choice.single_host) == (
        FABRIC,
        False,
        False,
    )


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
    assert not mnnvl._HANDLE_TYPE_CHOICES


def _attempts(monkeypatch, results):
    """Script _create_and_import_handles: one (handles, errors) per attempt."""
    attempts = []

    def attempt(comm, prop, size):
        attempts.append(prop)
        return results[len(attempts) - 1]

    monkeypatch.setattr(mnnvl, "_create_and_import_handles", attempt)
    return attempts


def _choose(monkeypatch, handle_type, final, single_host):
    choice = mnnvl._HandleTypeChoice(handle_type, final, single_host)
    monkeypatch.setattr(mnnvl, "choose_mnnvl_handle_type", lambda comm, dev_id: choice)
    return choice


def test_allocation_confirms_fabric(monkeypatch):
    choice = _choose(monkeypatch, FABRIC, final=False, single_host=True)
    attempts = _attempts(monkeypatch, [(["h0", "h1"], [None, None])])
    handles, handle_type = mnnvl.allocate_mnnvl_handles(
        None, 0, lambda handle_type: (handle_type, 64)
    )
    assert (handles, handle_type, attempts) == (["h0", "h1"], FABRIC, [FABRIC])
    assert choice.final


def test_single_host_falls_back_to_posix_for_good(monkeypatch, caplog):
    choice = _choose(monkeypatch, FABRIC, final=False, single_host=True)
    attempts = _attempts(
        monkeypatch,
        [
            (
                None,
                [None, "creating or exporting the allocation failed: NOT_PERMITTED"],
            ),
            (["h0", "h1"], [None, None]),
        ],
    )
    handles, handle_type = mnnvl.allocate_mnnvl_handles(
        None, 0, lambda handle_type: (handle_type, 64)
    )
    assert (handles, handle_type, attempts) == (["h0", "h1"], POSIX, [FABRIC, POSIX])
    assert (choice.handle_type, choice.final) == (POSIX, True)
    assert "rank 1: creating or exporting" in caplog.text


@pytest.mark.parametrize(
    ("handle_type", "final", "single_host"),
    [(FABRIC, True, True), (FABRIC, False, False), (POSIX, True, True)],
)
def test_settled_or_multi_host_choices_do_not_fall_back(
    monkeypatch, handle_type, final, single_host
):
    _choose(monkeypatch, handle_type, final, single_host)
    attempts = _attempts(monkeypatch, [(None, ["importing failed", None])])
    with pytest.raises(RuntimeError, match="rank 0: importing failed") as error:
        mnnvl.allocate_mnnvl_handles(None, 0, lambda handle_type: (handle_type, 64))
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


def test_attempt_imports_every_peer(fake_driver):
    driver = fake_driver()
    prop = SimpleNamespace(requestedHandleTypes=FABRIC)
    handles, errors = mnnvl._create_and_import_handles(_two_rank_comm(), prop, 64)
    assert handles == [101, 102] and errors == [None, None]
    assert driver.released == []


def test_attempt_releases_its_allocation_when_a_peer_fails(fake_driver):
    driver = fake_driver()
    prop = SimpleNamespace(requestedHandleTypes=FABRIC)
    handles, errors = mnnvl._create_and_import_handles(
        _two_rank_comm(peer_error="creating failed"), prop, 64
    )
    assert handles is None and errors == [None, "creating failed"]
    assert driver.released == [101]


def test_attempt_votes_on_a_local_import_failure(fake_driver):
    driver = fake_driver(fail_import=True)
    prop = SimpleNamespace(requestedHandleTypes=FABRIC)
    handles, errors = mnnvl._create_and_import_handles(_two_rank_comm(), prop, 64)
    assert handles is None and "importing the allocation of rank 1" in errors[0]
    assert driver.released == [101]
