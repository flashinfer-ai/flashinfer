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

"""Python owner of one rank's native transport session (windows, LSA peers, GIN).

Create and close are collective over the EP process group.  The windows are
``ncclMemAlloc`` storage registered as symmetric NCCL windows; every window is
also resolved to a load/store peer pointer for every LSA-reachable peer, and,
when the communicator offers GIN, one ``ncclDevComm`` serves the GIN puts.
"""

from __future__ import annotations

from typing import Any

import torch

_PROPERTY_SLOTS = 8


class TransportSession:
    def __init__(
        self,
        module: Any,
        *,
        group: Any,
        comm_pointer: int,
        rank: int,
        world_size: int,
        device: torch.device,
        windows: list[tuple[str, int]],
        gin_context_count: int,
        gin_signal_count: int,
        gin_world_barrier_count: int,
        gin_strong_signals_required: bool,
    ) -> None:
        import torch.distributed as dist

        names = [name for name, _ in windows]
        if len(names) != len(set(names)):
            raise ValueError("transport window names must be unique")
        contract = (
            tuple(windows),
            gin_context_count,
            gin_signal_count,
            gin_world_barrier_count,
            gin_strong_signals_required,
        )
        gathered: list[Any] = [None] * world_size
        dist.all_gather_object(gathered, contract, group=group)
        if any(c != contract for c in gathered):
            raise RuntimeError(
                "transport window/requirement contract differs across EP ranks"
            )
        dist.barrier(group=group)
        self.module = module
        self.group = group
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self._index = {name: i for i, name in enumerate(names)}
        self._bytes = dict(windows)
        self.handle = int(
            module.transport_session_create(
                comm_pointer,
                rank,
                world_size,
                device.index,
                torch.tensor([b for _, b in windows], dtype=torch.int64),
                gin_context_count,
                gin_signal_count,
                gin_world_barrier_count,
                gin_strong_signals_required,
            )
        )
        dist.barrier(group=group)
        props = torch.zeros(_PROPERTY_SLOTS, dtype=torch.int64)
        module.transport_session_properties(self.handle, props)
        self.gin_available = bool(props[1])
        self.gin_type = int(props[2])
        # GIN can be switched off at runtime (NCCL_GIN_ENABLE=0), in which case the native
        # session falls back to LSA only; every rank must reach the same answer.
        gathered_gin: list[Any] = [None] * world_size
        dist.all_gather_object(gathered_gin, self.gin_available, group=group)
        if any(bool(v) != self.gin_available for v in gathered_gin):
            module.transport_session_close(self.handle)
            raise RuntimeError(
                f"ranks disagree about GIN availability: {[bool(v) for v in gathered_gin]}"
            )
        self.lsa_team_count = int(props[3])
        peers = torch.zeros(len(names) * world_size, dtype=torch.int64)
        module.transport_session_peer_pointers(self.handle, peers)
        self._peer_pointers = [int(v) for v in peers.tolist()]
        reach = torch.zeros(world_size, dtype=torch.int64)
        module.transport_session_lsa_reachable(self.handle, reach)
        self.lsa_reachable = [bool(v) for v in reach.tolist()]
        self.dev_comm = int(module.transport_session_dev_comm_pointer(self.handle))
        self.closed = False

    @property
    def window_names(self) -> list[str]:
        return list(self._index)

    def window_pointer(self, name: str) -> int:
        return int(
            self.module.transport_session_window_pointer(self.handle, self._index[name])
        )

    def window_handle(self, name: str) -> int:
        """``ncclWindow_t`` of the window, or 0 when GIN is unavailable (never dereferenced)."""
        if not self.gin_available:
            return 0
        return int(
            self.module.transport_session_window_handle(self.handle, self._index[name])
        )

    def window_bytes(self, name: str) -> int:
        return self._bytes[name]

    def peer_pointers(self, name: str) -> list[int]:
        """Load/store pointers of window ``name`` on every peer (0 = not LSA reachable)."""
        base = self._index[name] * self.world_size
        return self._peer_pointers[base : base + self.world_size]

    def all_peer_pointers(self) -> list[int]:
        """Window-major ``[window][peer]`` pointer table, the kernel's ``lsa_peer_ptrs``."""
        return list(self._peer_pointers)

    def zero_window(self, name: str) -> None:
        stream = torch.cuda.current_stream(self.device)
        self.module.transport_session_window_zero(
            self.handle, self._index[name], self.device.index, stream.cuda_stream
        )
        stream.synchronize()

    def close(self) -> None:
        import torch.distributed as dist

        if self.closed:
            return
        dist.barrier(group=self.group)
        self.module.transport_session_close(self.handle)
        self.closed = True
        dist.barrier(group=self.group)


__all__ = ["TransportSession"]
