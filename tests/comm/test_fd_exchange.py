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
"""Correctness tests for flashinfer.comm.fd_exchange."""

import array
import multiprocessing as mp
import os
import socket
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from flashinfer.comm.abstractions import CommBackend
from flashinfer.comm.fd_exchange import (
    _extract_fd,  # pyright: ignore[reportPrivateUsage]
    _fd_ancillary,  # pyright: ignore[reportPrivateUsage]
    broadcast_fd,
    exchange_fds,
    recv_fd_dgram,
    send_fd_dgram,
)

pytestmark = pytest.mark.skipif(
    not hasattr(socket, "AF_UNIX") or not hasattr(socket, "CMSG_SPACE"),
    reason="fd_exchange requires AF_UNIX sockets with SCM_RIGHTS (POSIX only)",
)

_TIMEOUT_S = 60.0


class _MPComm:
    """A minimal CommBackend test double over multiprocessing sync primitives.

    ``barrier``/``shared`` are multiprocessing proxies (dynamically typed), hence
    ``Any``. ``bcast``/``Split`` are unused by the fd helpers and only exist to
    satisfy the CommBackend protocol.
    """

    def __init__(self, rank: int, size: int, barrier: Any, shared: Any):
        self.rank, self.size = rank, size
        self._barrier, self._shared = barrier, shared

    def Get_rank(self) -> int:
        return self.rank

    def Get_size(self) -> int:
        return self.size

    def allgather(self, data: Any) -> tuple[Any, ...]:
        self._shared[self.rank] = data
        self._barrier.wait()
        result = tuple(self._shared)
        self._barrier.wait()
        return result

    def bcast(self, data: Any, root: int) -> Any:
        raise NotImplementedError

    def barrier(self) -> None:
        self._barrier.wait()

    def Split(self, color: int, key: int) -> CommBackend:
        raise NotImplementedError


def _make_file(rank: int) -> tuple[int, str]:
    """A fresh temp file whose contents uniquely identify ``rank``."""
    fd, path = tempfile.mkstemp(prefix=f"fdx_{rank}_")
    os.write(fd, f"rank-{rank}".encode())
    return fd, path


def _read_fd(fd: int) -> str:
    return os.pread(fd, 128, 0).decode()


def _worker_exchange(rank: int, size: int, barrier: Any, shared: Any, q: Any) -> None:
    comm = _MPComm(rank, size, barrier, shared)
    fd, path = _make_file(rank)
    try:
        fds = exchange_fds(comm, fd)
        # Own slot must be the original fd (not a dup), per the documented contract.
        own_slot_is_original = fds[rank] == fd
        contents = [_read_fd(f) for f in fds]
        # Caller owns every returned fd and closes each exactly once.
        for f in fds:
            os.close(f)
        q.put((rank, (contents, own_slot_is_original)))
    finally:
        os.unlink(path)


def _worker_broadcast(
    rank: int, size: int, barrier: Any, shared: Any, q: Any, root: int
) -> None:
    comm = _MPComm(rank, size, barrier, shared)
    fd, path = _make_file(rank)
    try:
        rfd = broadcast_fd(comm, fd if rank == root else None, root)
        content = _read_fd(rfd)
        os.close(rfd)
        if rank != root:
            os.close(fd)  # our own file fd is not handed back to us
        q.put((rank, content))
    finally:
        os.unlink(path)


def _run_workers(target: Callable[..., None], size: int, *extra: Any) -> dict[int, Any]:
    # forkserver forks workers from a clean single-threaded server, avoiding the
    # "fork() in a multi-threaded process" hazard (torch/cuda leave threads live).
    ctx = mp.get_context("forkserver")
    barrier = ctx.Barrier(size)
    shared = ctx.Manager().list([None] * size)
    q = ctx.Queue()
    procs = [
        ctx.Process(target=target, args=(r, size, barrier, shared, q, *extra))
        for r in range(size)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=_TIMEOUT_S)
    for p in procs:
        assert p.exitcode == 0, f"worker pid={p.pid} exited with {p.exitcode}"
    results: dict[int, Any] = {}
    while not q.empty():
        rank, payload = q.get()
        results[rank] = payload
    assert len(results) == size, f"expected {size} results, got {sorted(results)}"
    return results


@pytest.mark.parametrize("size", [1, 2, 3, 4])
def test_exchange_fds_all_to_all(size: int) -> None:
    results = _run_workers(_worker_exchange, size)
    expected = [f"rank-{i}" for i in range(size)]
    for rank in range(size):
        contents, own_slot_is_original = results[rank]
        # Every rank can read every rank's file through the exchanged fds.
        assert contents == expected, f"rank {rank} read {contents}, expected {expected}"
        assert own_slot_is_original, f"rank {rank} own slot was not its original fd"


@pytest.mark.parametrize(
    "size,root",
    [(2, 0), (2, 1), (3, 0), (3, 2), (4, 0), (4, 3)],
)
def test_broadcast_fd(size: int, root: int) -> None:
    results = _run_workers(_worker_broadcast, size, root)
    expected = f"rank-{root}"
    for rank in range(size):
        assert results[rank] == expected, (
            f"rank {rank} read {results[rank]}, expected root's content {expected}"
        )


def test_broadcast_fd_single_rank_returns_local() -> None:
    comm = _MPComm(0, 1, None, None)  # size==1 short-circuits before any sync
    fd, path = _make_file(0)
    try:
        assert broadcast_fd(comm, fd, root=0) == fd
    finally:
        os.close(fd)
        os.unlink(path)


def test_dgram_send_recv_roundtrip(tmp_path: Path) -> None:
    recv_path = str(tmp_path / "recv.sock")
    send_path = str(tmp_path / "send.sock")
    recv_sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    send_sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    recv_sock.bind(recv_path)
    send_sock.bind(send_path)
    recv_sock.settimeout(_TIMEOUT_S)
    fd, path = _make_file(0)
    try:
        send_fd_dgram(send_sock, fd, recv_path)
        got_fd = recv_fd_dgram(recv_sock)
        try:
            assert got_fd != fd  # a distinct descriptor referring to the same file
            assert _read_fd(got_fd) == "rank-0"
        finally:
            os.close(got_fd)
    finally:
        os.close(fd)
        os.unlink(path)
        recv_sock.close()
        send_sock.close()


def test_recv_fd_dgram_without_fd_raises(tmp_path: Path) -> None:
    recv_path = str(tmp_path / "recv.sock")
    send_path = str(tmp_path / "send.sock")
    recv_sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    send_sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    recv_sock.bind(recv_path)
    send_sock.bind(send_path)
    recv_sock.settimeout(_TIMEOUT_S)
    try:
        send_sock.sendto(b"\x00", recv_path)  # payload but no ancillary fd
        with pytest.raises(RuntimeError, match="no file descriptor"):
            recv_fd_dgram(recv_sock)
    finally:
        recv_sock.close()
        send_sock.close()


def test_fd_ancillary_extract_roundtrip() -> None:
    anc = _fd_ancillary(7)
    assert isinstance(anc, tuple) and len(anc) == 1
    level, cmsg_type, buf = anc[0]
    assert level == socket.SOL_SOCKET and cmsg_type == socket.SCM_RIGHTS
    assert list(buf) == [7]
    # _extract_fd decodes the received-side form, where cmsg_data is bytes.
    assert _extract_fd([(socket.SOL_SOCKET, socket.SCM_RIGHTS, buf.tobytes())]) == 7


def test_extract_fd_ignores_unrelated_control_messages() -> None:
    assert _extract_fd([]) is None
    wrong = array.array("i", [123]).tobytes()
    assert _extract_fd([(socket.IPPROTO_IP, 0, wrong)]) is None
