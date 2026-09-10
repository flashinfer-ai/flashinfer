# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for the CommBackend adapters (MPIBackend, TorchDistBackend).

TorchDistBackend is exercised for real over the gloo backend across processes
(CPU only, no GPU). MPIBackend needs an MPI runtime for real use, so its adapter
logic is unit-tested against an injected fake mpi4py communicator.
"""

import importlib.util
import multiprocessing as mp
import os
import shutil
import tempfile
from typing import Any

import pytest
import torch.distributed as dist

from flashinfer.comm.comm_backend import (
    MPIBackend,
    TorchDistBackend,
    _split_partition,  # pyright: ignore[reportPrivateUsage]
)

_TIMEOUT_S = 120.0


def _backend_checks(rank: int) -> dict[str, Any]:
    """Run every CommBackend method once and return the observed results.

    All ranks issue the same sequence of collectives, so the calls stay matched.
    The world size is read from the process group, not passed in.
    """
    b = TorchDistBackend()
    res: dict[str, Any] = {}
    res["rank"] = b.Get_rank()
    res["size"] = b.Get_size()

    gathered = b.allgather(rank)
    res["allgather"] = gathered
    res["allgather_type"] = type(gathered).__name__

    # Only the root supplies a value; every rank must receive it.
    res["bcast"] = b.bcast(12345 if rank == 0 else None, root=0)

    b.barrier()

    # Split by parity, ordered within each sub-group by key=rank.
    sub = b.Split(rank % 2, key=rank)
    res["sub_type"] = type(sub).__name__
    res["sub_rank"] = sub.Get_rank()
    res["sub_size"] = sub.Get_size()
    return res


def _dist_entry(rank: int, world_size: int, store_path: str, q: Any) -> None:
    # Pin gloo to loopback; otherwise it may pick a real NIC and fail to connect
    # the full mesh between local processes.
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{store_path}",
        rank=rank,
        world_size=world_size,
    )
    try:
        q.put((rank, _backend_checks(rank)))
    finally:
        dist.destroy_process_group()


def _run_dist(world_size: int) -> dict[int, dict[str, Any]]:
    # spawn (not fork/forkserver): starts fresh interpreters, so it stays safe
    # even if an MPI test in the same session has already called MPI_Init.
    ctx = mp.get_context("spawn")
    tmpdir = tempfile.mkdtemp(prefix="tdb_store_")
    store_path = os.path.join(tmpdir, "store")
    q = ctx.Queue()
    procs = [
        ctx.Process(target=_dist_entry, args=(r, world_size, store_path, q))
        for r in range(world_size)
    ]
    try:
        for p in procs:
            p.start()
        for p in procs:
            p.join(timeout=_TIMEOUT_S)
        for p in procs:
            assert p.exitcode == 0, f"worker pid={p.pid} exited with {p.exitcode}"
        results: dict[int, dict[str, Any]] = {}
        while not q.empty():
            rank, res = q.get()
            results[rank] = res
        assert len(results) == world_size, (
            f"expected {world_size} results, got {sorted(results)}"
        )
        return results
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


_gloo_unavailable = not dist.is_available() or not dist.is_gloo_available()


@pytest.mark.skipif(_gloo_unavailable, reason="torch.distributed gloo backend required")
@pytest.mark.parametrize("world_size", [2, 4])
def test_torch_dist_backend_collectives(world_size: int) -> None:
    results = _run_dist(world_size)

    for rank in range(world_size):
        res = results[rank]
        assert res["rank"] == rank
        assert res["size"] == world_size

        # allgather returns a per-rank tuple, indexed by rank.
        assert res["allgather"] == tuple(range(world_size))
        assert res["allgather_type"] == "tuple"

        # every rank received the root's value.
        assert res["bcast"] == 12345

        # Split by parity: same-parity ranks form a sub-group, ordered by rank.
        peers = [r for r in range(world_size) if r % 2 == rank % 2]
        assert res["sub_type"] == "TorchDistBackend"
        assert res["sub_size"] == len(peers)
        assert res["sub_rank"] == peers.index(rank)


class _FakeGroup:
    def __init__(self, ranks: tuple[int, ...]) -> None:
        self.ranks = ranks


class _FakeDist:
    """Single-process stand-in for torch.distributed, driving one logical rank."""

    def __init__(
        self, rank: int, world_size: int, all_info: list[tuple[int, int, int]]
    ) -> None:
        self._rank = rank
        self._world = world_size
        self._all_info = all_info
        self.created: list[tuple[int, ...]] = []  # sub-groups created, in order

    def get_rank(self, group: Any = None) -> int:
        return self._rank if group is None else group.ranks.index(self._rank)

    def get_world_size(self, group: Any = None) -> int:
        return self._world if group is None else len(group.ranks)

    def all_gather_object(
        self, out_list: list[Any], data: Any, group: Any = None
    ) -> None:
        # Simulate the collective: every rank observes all ranks' contributions,
        # and its own slot must match what it put in.
        assert data == self._all_info[self._rank]
        for i, info in enumerate(self._all_info):
            out_list[i] = info

    def new_subgroups_by_enumeration(
        self, ranks_per_subgroup_list: list[list[int]]
    ) -> tuple[Any, list[Any]]:
        subs = [_FakeGroup(tuple(r)) for r in ranks_per_subgroup_list]
        self.created.extend(g.ranks for g in subs)
        cur = next((g for g in subs if self._rank in g.ranks), None)
        return cur, subs

    def new_group(self, ranks: list[int]) -> Any:
        # Unused by the fixed Split, but recorded so a regression to the buggy
        # "create only my own color" implementation is caught by the same
        # created-sequence assertion below.
        g = _FakeGroup(tuple(ranks))
        self.created.append(g.ranks)
        return g


def _drive_split(
    colors_keys: list[tuple[int, int]],
) -> list[tuple[_FakeDist, TorchDistBackend]]:
    """Run Split once per logical rank against a fake torch.distributed.

    Caller must stub ``dist.is_initialized`` -- the child TorchDistBackend that
    Split constructs goes through the real ``__init__`` guard.
    """
    world_size = len(colors_keys)
    all_info = [(c, k, r) for r, (c, k) in enumerate(colors_keys)]
    out: list[tuple[_FakeDist, TorchDistBackend]] = []
    for rank, (color, key) in enumerate(colors_keys):
        fake = _FakeDist(rank, world_size, all_info)
        b = object.__new__(TorchDistBackend)
        b._group = None  # pyright: ignore[reportPrivateUsage]
        b._dist = fake  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue]
        out.append((fake, b.Split(color, key)))
    return out


def test_torch_dist_backend_split_creates_every_subgroup_on_every_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 4 ranks, 2 colors, 2 ranks each
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    driven = _drive_split([(r % 2, r) for r in range(4)])

    # Full color partition, ranks ordered by (key, rank) within each color.
    expected = [(0, 2), (1, 3)]
    for rank, (fake, sub) in enumerate(driven):
        # Every rank creates the SAME sequence of sub-groups (all colors, in
        # order) -- the invariant new_group requires; "own color only" breaks it.
        assert fake.created == expected, f"rank {rank}: {fake.created}"
        # ...and gets back a backend scoped to its own color's sub-group.
        assert sub._group.ranks == expected[rank % 2]  # pyright: ignore[reportPrivateUsage, reportOptionalMemberAccess]


def test_split_partition_covers_all_colors_ordered_by_key() -> None:
    # The rank-independent core of Split, tested directly (no dist needed).
    # allgathered (color, key, global_rank) from 4 ranks, 2 colors.
    all_info = [(0, 0, 0), (1, 1, 1), (0, 2, 2), (1, 3, 3)]
    # Full partition, colors ascending; within a color, ranks by (key, rank).
    assert _split_partition(all_info) == [[0, 2], [1, 3]]

    # Single color, keys reverse the natural order: lower key -> lower rank,
    # ties broken by global rank -- matching MPI_Comm_split.
    reversed_keys = [(0, 10, 0), (0, 5, 1), (0, 20, 2), (0, 0, 3)]
    assert _split_partition(reversed_keys) == [[3, 1, 0, 2]]


class _FakeMpiComm:
    """Records calls and returns canned values, standing in for mpi4py's comm."""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []

    def Get_rank(self) -> int:
        self.calls.append(("Get_rank",))
        return 3

    def Get_size(self) -> int:
        self.calls.append(("Get_size",))
        return 8

    def allgather(self, data: Any) -> list[Any]:
        self.calls.append(("allgather", data))
        return [data, data]  # a list -- the adapter must freeze it to a tuple

    def bcast(self, data: Any, root: int) -> Any:
        self.calls.append(("bcast", data, root))
        return data

    def Barrier(self) -> None:
        self.calls.append(("Barrier",))

    def Split(self, color: int, key: int) -> "_FakeMpiComm":
        self.calls.append(("Split", color, key))
        return _FakeMpiComm()


def _fake_backend() -> tuple[MPIBackend, _FakeMpiComm]:
    b = MPIBackend()  # MpiComm() is created but mpi4py is only imported on use
    fake = _FakeMpiComm()
    b._mpicomm = fake  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue]
    return b, fake


def test_mpi_backend_rank_and_size_delegate() -> None:
    b, fake = _fake_backend()
    assert b.Get_rank() == 3
    assert b.Get_size() == 8
    assert ("Get_rank",) in fake.calls and ("Get_size",) in fake.calls


def test_mpi_backend_allgather_returns_tuple() -> None:
    b, fake = _fake_backend()
    result = b.allgather(7)
    assert result == (7, 7)
    assert isinstance(result, tuple)  # adapter freezes mpi4py's list
    assert ("allgather", 7) in fake.calls


def test_mpi_backend_bcast_forwards_data_and_root() -> None:
    b, fake = _fake_backend()
    assert b.bcast("payload", root=2) == "payload"
    assert ("bcast", "payload", 2) in fake.calls


def test_mpi_backend_barrier_calls_uppercase_Barrier() -> None:
    b, fake = _fake_backend()
    b.barrier()
    assert ("Barrier",) in fake.calls


def test_mpi_backend_split_returns_subgroup_without_mutating_self() -> None:
    b, fake = _fake_backend()
    b2 = b.Split(1, 5)
    assert ("Split", 1, 5) in fake.calls
    assert isinstance(b2, MPIBackend) and b2 is not b
    # self must be left unchanged (still delegating to the original comm),
    # mirroring TorchDistBackend.Split. A regression to "mutate self into the
    # sub-group" would route this call to the split child instead.
    fake.calls.clear()
    b.Get_rank()
    assert ("Get_rank",) in fake.calls


@pytest.mark.skipif(
    importlib.util.find_spec("mpi4py") is None, reason="mpi4py required"
)
def test_mpi_backend_real_collectives() -> None:
    """Real mpi4py collectives over COMM_WORLD.

    Runs at world_size 1 under plain pytest (real delegation + tuple wrapping)
    and at world_size N under ``mpirun -np N pytest <this file>``.
    """
    b = MPIBackend()
    world_size = b.Get_size()
    rank = b.Get_rank()
    assert 0 <= rank < world_size

    gathered = b.allgather(rank)
    assert isinstance(gathered, tuple)
    assert gathered == tuple(range(world_size))

    assert b.bcast(12345 if rank == 0 else None, root=0) == 12345

    b.barrier()

    # Split by parity, ordered within each sub-group by key=rank.
    sub = b.Split(rank % 2, key=rank)
    assert isinstance(sub, MPIBackend)
    peers = [r for r in range(world_size) if r % 2 == rank % 2]
    assert sub.Get_size() == len(peers)
    assert sub.Get_rank() == peers.index(rank)
    # Split must not mutate the parent communicator.
    assert b.Get_size() == world_size
