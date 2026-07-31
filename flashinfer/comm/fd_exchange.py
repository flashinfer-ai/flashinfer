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
"""File-descriptor exchange across ranks over AF_UNIX sockets (SCM_RIGHTS).

Pure OS-level fd passing — no CUDA. The kernel installs a fresh entry in the
receiver's descriptor table, so no ``CAP_SYS_PTRACE`` (as ``pidfd_getfd`` would
need) is required. Two collective helpers are built on a ``CommBackend``:

* :func:`exchange_fds` — all-to-all (every rank ends up with every rank's fd).
* :func:`broadcast_fd` — one-to-all (root's fd to every rank).

plus the low-level SCM_RIGHTS send/recv primitives shared by connectionless
(SOCK_DGRAM) callers.
"""

import array
import contextlib
import logging
import os
import socket
import tempfile
import threading
import time
from collections.abc import Iterable

from .abstractions import CommBackend

logger = logging.getLogger(__name__)

_FD_ITEMSIZE = array.array("i").itemsize
_TIMEOUT_S = 30.0

# One control message as delivered by socket.recvmsg():
# (cmsg_level, cmsg_type, cmsg_data), where cmsg_data is always ``bytes``.
CmsgTriple = tuple[int, int, bytes]


# ============================================================================
# Low-level SCM_RIGHTS primitives
# ============================================================================


def _fd_ancillary(fd: int) -> tuple[tuple[int, int, array.array[int]]]:
    """Ancillary-data payload carrying a single fd via SCM_RIGHTS."""
    return ((socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", [fd])),)


def _extract_fd(ancdata: Iterable[CmsgTriple]) -> int | None:
    """Return the first fd found in received ancillary data, or None."""
    fds = array.array("i")
    for cmsg_level, cmsg_type, cmsg_data in ancdata:
        if cmsg_level == socket.SOL_SOCKET and cmsg_type == socket.SCM_RIGHTS:
            trim = len(cmsg_data) - (len(cmsg_data) % fds.itemsize)
            fds.frombytes(cmsg_data[:trim])
    return fds[0] if fds else None


def send_fd_dgram(sock: socket.socket, fd: int, dest_path: str) -> None:
    """Send one fd over a bound AF_UNIX/SOCK_DGRAM socket to ``dest_path``."""
    sock.sendmsg([b"\x00"], _fd_ancillary(fd), 0, dest_path)


def recv_fd_dgram(sock: socket.socket) -> int:
    """Receive one fd from a bound AF_UNIX/SOCK_DGRAM socket."""
    _, ancdata, _, _ = sock.recvmsg(1, socket.CMSG_SPACE(_FD_ITEMSIZE))
    fd = _extract_fd(ancdata)
    if fd is None:
        raise RuntimeError("[fd_exchange] no file descriptor in received message")
    return fd


def _send_fd_stream(sock: socket.socket, fd: int) -> None:
    sock.sendmsg([b"\x00"], _fd_ancillary(fd))


def _recv_fd_stream(conn: socket.socket) -> int | None:
    _, ancdata, _, _ = conn.recvmsg(1, socket.CMSG_SPACE(_FD_ITEMSIZE))
    return _extract_fd(ancdata)


def _connect_with_retry(path: str, timeout: float) -> socket.socket:
    """Connect to a listening AF_UNIX socket, retrying until it is bound."""
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    deadline = time.monotonic() + timeout
    while True:
        try:
            sock.connect(path)
            return sock
        except (ConnectionRefusedError, FileNotFoundError):
            if time.monotonic() > deadline:
                sock.close()
                raise RuntimeError(
                    f"[fd_exchange] timed out connecting to {path}"
                ) from None
            time.sleep(0.01)


# ============================================================================
# Collective fd exchange over a CommBackend
# ============================================================================


def exchange_fds(comm: CommBackend, local_fd: int) -> list[int]:
    """All-to-all fd exchange over AF_UNIX/SOCK_STREAM sockets.

    Returns a list of length ``comm.Get_size()`` where entry ``[rank]`` is the
    fd received from that rank. Entry ``[own_rank]`` is ``local_fd`` itself
    (not a dup). The caller owns every returned fd and must close each exactly
    once; closing ``entry[own_rank]`` closes ``local_fd``. ``local_fd`` is never
    closed by this function, so on error the caller still owns it.
    """
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    # Private temp dir per rank prevents path collisions between concurrent groups.
    with tempfile.TemporaryDirectory(
        prefix=f"cuda_fd_xchg_{os.getpid()}_{comm_rank}_",
        ignore_cleanup_errors=True,
    ) as xchg_dir:
        sock_path = os.path.join(xchg_dir, "fd.sock")
        all_sock_paths = comm.allgather(sock_path)

        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            server.bind(sock_path)
            server.listen(comm_size)
            server.settimeout(_TIMEOUT_S)

            received_fds: list[int | None] = [None] * comm_size
            received_fds[comm_rank] = local_fd
            accept_errors: list[Exception] = []

            def accept_loop():
                for _ in range(comm_size - 1):
                    try:
                        conn, _ = server.accept()
                        try:
                            rank_bytes = b""
                            while len(rank_bytes) < 4:
                                chunk = conn.recv(4 - len(rank_bytes))
                                if not chunk:
                                    break
                                rank_bytes += chunk
                            sender_rank = int.from_bytes(rank_bytes, "little")
                            fd = _recv_fd_stream(conn)
                            if fd is not None:
                                received_fds[sender_rank] = fd
                        finally:
                            conn.close()
                    except Exception as exc:
                        accept_errors.append(exc)

            t = threading.Thread(target=accept_loop, daemon=True)
            t.start()

            for target_rank in range(comm_size):
                if target_rank == comm_rank:
                    continue
                sock = _connect_with_retry(all_sock_paths[target_rank], _TIMEOUT_S)
                try:
                    sock.sendall(comm_rank.to_bytes(4, "little"))
                    _send_fd_stream(sock, local_fd)
                finally:
                    sock.close()

            t.join(timeout=_TIMEOUT_S)

            if accept_errors:
                _close_fds(
                    fd for fd in received_fds if fd is not None and fd != local_fd
                )
                raise RuntimeError(
                    f"[fd_exchange] SCM_RIGHTS accept errors: {accept_errors}"
                )
            missing = [i for i, fd in enumerate(received_fds) if fd is None]
            if missing:
                _close_fds(
                    fd for fd in received_fds if fd is not None and fd != local_fd
                )
                raise RuntimeError(
                    f"[fd_exchange] did not receive file descriptors from ranks: {missing}"
                )
            return received_fds  # type: ignore[return-value]
        finally:
            server.close()


def broadcast_fd(comm: CommBackend, local_fd: int | None, root: int) -> int:
    """Broadcast one fd from ``root`` to all ranks over AF_UNIX/SOCK_STREAM.

    ``root`` passes its fd as ``local_fd`` and gets the same fd back; every
    other rank passes ``None`` and receives root's fd (a fresh local dup). The
    caller closes the returned fd exactly once.
    """
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()
    if comm_size <= 1:
        assert local_fd is not None
        return local_fd

    with tempfile.TemporaryDirectory(
        prefix=f"cuda_fd_bcast_{os.getpid()}_{comm_rank}_",
        ignore_cleanup_errors=True,
    ) as xchg_dir:
        sock_path = os.path.join(xchg_dir, "fd.sock")
        all_sock_paths = comm.allgather(sock_path)

        if comm_rank == root:
            assert local_fd is not None, "root must supply the fd to broadcast"
            for target_rank in range(comm_size):
                if target_rank == root:
                    continue
                sock = _connect_with_retry(all_sock_paths[target_rank], _TIMEOUT_S)
                try:
                    _send_fd_stream(sock, local_fd)
                finally:
                    sock.close()
            return local_fd

        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            server.bind(sock_path)
            server.listen(1)
            server.settimeout(_TIMEOUT_S)
            conn, _ = server.accept()
            try:
                fd = _recv_fd_stream(conn)
            finally:
                conn.close()
            if fd is None:
                raise RuntimeError(
                    f"[fd_exchange] rank {comm_rank} received no fd from root {root}"
                )
            return fd
        finally:
            server.close()


def _close_fds(fds: Iterable[int]) -> None:
    for fd in fds:
        with contextlib.suppress(OSError):
            os.close(fd)
