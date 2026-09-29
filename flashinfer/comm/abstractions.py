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
"""Lightweight interface for the comm package for convenient implementation-agnostic references."""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class CommBackend(Protocol):
    """Structural interface for a collective communication backend."""

    def Get_rank(self) -> int:
        """Return the caller's rank within the group."""
        ...

    def Get_size(self) -> int:
        """Return the number of ranks in the group."""
        ...

    def allgather(self, data: Any) -> tuple[Any, ...]:
        """Gather ``data`` from every rank into a per-rank tuple (indexed by rank)."""
        ...

    def bcast(self, data: Any, root: int) -> Any:
        """Broadcast ``data`` from ``root`` to every rank and return it."""
        ...

    def barrier(self) -> None:
        """Block until every rank has reached this barrier."""
        ...

    def Split(self, color: int, key: int) -> "CommBackend":
        """Partition the group into sub-groups by ``color``, ranked by ``key``."""
        ...
