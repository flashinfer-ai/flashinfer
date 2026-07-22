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
"""Concrete ``CommBackend`` adapters.

``MPIBackend`` and ``TorchDistBackend`` are adapters that satisfy the
``CommBackend`` structural interface (see ``abstractions``) without inheritance,
so any object exposing the same methods (rank/size/allgather/bcast/barrier/
Split) is a valid backend. Nothing here depends on CUDA.
"""

from collections.abc import Sequence
from typing import Any


def lazy_import_mpi() -> Any:
    """Lazy import for mpi4py."""
    try:
        from mpi4py import MPI

        return MPI
    except ImportError as err:
        raise ImportError("mpi4py is not installed") from err  # type: ignore[no-redef]


class MpiComm:  # type: ignore[no-redef]
    _comm: Any = None
    _MPI: Any = None

    @classmethod
    def _get_mpi(cls):
        if cls._MPI is None:
            cls._MPI = lazy_import_mpi()
            cls._comm = cls._MPI.COMM_WORLD
        return cls._MPI

    @classmethod
    def set_mpi_comm(cls, new_comm: Any):
        cls._get_mpi()
        # Optional: add type checking here
        cls._comm = new_comm

    def __getattr__(self, name: str) -> Any:
        if self._comm is None:
            self._get_mpi()
        return getattr(self._comm, name)


class MPIBackend:
    """``CommBackend`` adapter over mpi4py."""

    def __init__(self):
        self._mpicomm = MpiComm()

    def Get_rank(self) -> int:
        return self._mpicomm.Get_rank()

    def Get_size(self) -> int:
        return self._mpicomm.Get_size()

    def allgather(self, data: Any) -> tuple[Any, ...]:
        return tuple(self._mpicomm.allgather(data))

    def bcast(self, data: Any, root: int) -> Any:
        return self._mpicomm.bcast(data, root)

    def barrier(self) -> None:
        self._mpicomm.Barrier()

    def Split(self, color: int, key: int) -> "MPIBackend":
        # Return the sub-group as a new adapter and leave self unchanged,
        # consistent with TorchDistBackend.Split.
        split = MPIBackend()
        split._mpicomm = self._mpicomm.Split(color, key)
        return split


def _split_partition(all_info: Sequence[tuple[int, int, int]]) -> list[list[int]]:
    """Full color partition for an MPI-style ``Split``.

    ``all_info`` is the allgathered ``(color, key, global_rank)`` from every rank.
    Returns one rank-list per color (colors sorted ascending); within each color,
    ranks are ordered by ``(key, global_rank)`` -- lower key first, ties broken by
    rank, matching ``MPI_Comm_split``. The result is independent of the calling
    rank: every rank must build the *same* partition, because
    ``torch.distributed.new_group`` names each sub-group by a global counter (not
    by its member ranks), so ranks that create only their own sub-group get
    colliding rendezvous keys and deadlock.
    """
    colors = sorted({c for c, _, _ in all_info})
    return [
        [r for _, r in sorted((k, r) for c, k, r in all_info if c == col)]
        for col in colors
    ]


class TorchDistBackend:
    """``CommBackend`` adapter over torch.distributed."""

    def __init__(self, group: Any | None = None):
        """
        Initialize TorchDistBackend.

        Args:
            group: Optional process group. If None, uses the default process group.
        """
        import torch.distributed as dist

        if not dist.is_initialized():
            raise RuntimeError(
                "torch.distributed is not initialized. "
                "Please call torch.distributed.init_process_group() first."
            )
        self._group = group
        self._dist = dist

    def Get_rank(self) -> int:
        return self._dist.get_rank(self._group)

    def Get_size(self) -> int:
        return self._dist.get_world_size(self._group)

    def allgather(self, data: Any) -> tuple[Any, ...]:
        """All-gather arbitrary Python objects across all ranks."""
        output_list = [None] * self.Get_size()
        self._dist.all_gather_object(output_list, data, group=self._group)  # pyright: ignore[reportUnknownMemberType]
        return tuple(output_list)

    def bcast(self, data: Any, root: int) -> Any:
        """Broadcast a Python object from root to all ranks.

        Args:
            data: object to broadcast (only used on the root rank).
            root: group-local rank of the sender (consistent with MPI).
        """
        object_list = [data]
        global_root = (
            self._dist.get_global_rank(self._group, root)
            if self._group is not None
            else root
        )
        self._dist.broadcast_object_list(
            object_list, src=global_root, group=self._group
        )
        return object_list[0]

    def barrier(self) -> None:
        self._dist.barrier(group=self._group)  # pyright: ignore[reportUnknownMemberType]

    def Split(self, color: int, key: int) -> "TorchDistBackend":
        """
        Split the communicator into sub-groups based on color.

        All processes with the same color will be in the same new group.
        The key determines the rank ordering within the new group.

        Args:
            color: Processes with the same color are placed in the same group
            key: Determines rank ordering within the new group (lower key = lower rank)

        Returns:
            New TorchDistBackend with the split process group
        """
        # Gather (color, key, global_rank) from every rank, then create the full
        # color partition on *every* rank (see _split_partition) and let
        # new_subgroups_by_enumeration hand back the sub-group this rank is in.
        all_info = self.allgather((color, key, self.Get_rank()))
        cur_group, _ = self._dist.new_subgroups_by_enumeration(
            _split_partition(all_info)
        )  # pyright: ignore
        return TorchDistBackend(group=cur_group)  # pyright: ignore
