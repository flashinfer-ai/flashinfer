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
"""Per-rank MNNVL inboxes for the tile-ready combine ship path.

Each rank owns ``inbox[src, local_expert, slot, hidden]``. The consumer on
source rank ``S`` stores a finished GEMM2 row into dest rank ``D`` at
``inbox_D[S, expert, slot]``. Dest-side reduce lives in :mod:`.combine`.

Multi-rank buffers are fabric-mapped via :class:`MnnvlMemory` (GB200 MNNVL).
Classic ``cudaIpcOpenMemHandle`` is same-node only and fails on Blyris 2-node
with ``invalid resource handle``. World-size 1 keeps a local ``cudaMalloc``.
"""

from __future__ import annotations

import os
from typing import List

import torch
import torch.distributed as dist

from flashinfer.comm.dlpack_utils import pack_strided_memory


def _gpus_per_node() -> int:
    raw = os.environ.get("LOCAL_WORLD_SIZE")
    if raw is not None:
        return max(1, int(raw))
    n = torch.cuda.device_count()
    return max(1, int(n) if n else 1)


def _wrap_bf16(
    ptr: int, nbytes: int, shape: tuple[int, ...], dev_id: int
) -> torch.Tensor:
    wrapped = pack_strided_memory(ptr, nbytes, nbytes, 1, torch.bfloat16, dev_id)
    flat = wrapped.view(torch.bfloat16)
    inbox = flat.reshape(shape)
    inbox._capsule_wrapper = getattr(wrapped, "_capsule_wrapper", None)
    return inbox


def _alloc_mnnvl(nbytes: int) -> tuple[object, List[int]]:
    from flashinfer.comm.comm_backend import TorchDistBackend
    from flashinfer.comm.mapping import Mapping
    from flashinfer.comm.mnnvl import MnnvlConfig, MnnvlMemory

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    group = dist.group.WORLD
    MnnvlMemory.initialize()
    mapping = Mapping(
        world_size=world_size,
        rank=rank,
        gpus_per_node=_gpus_per_node(),
        tp_size=world_size,
    )
    backend = TorchDistBackend(group)
    MnnvlMemory.set_comm_from_config(mapping, MnnvlConfig(comm_backend=backend))
    mem = MnnvlMemory(mapping, nbytes)
    pointers = [int(mem.ptr + i * mem.rank_stride) for i in range(world_size)]
    return mem, pointers


class CombineInboxWorkspace:
    """MNNVL-mapped ``[world, nle, tokens_per_rank, hidden]`` bf16 inbox."""

    def __init__(
        self,
        world_size: int,
        num_local_experts: int,
        tokens_per_rank: int,
        hidden: int,
        device: torch.device,
    ) -> None:
        if world_size <= 0 or num_local_experts <= 0:
            raise ValueError("world_size and num_local_experts must be positive")
        if tokens_per_rank <= 0 or hidden <= 0:
            raise ValueError("tokens_per_rank and hidden must be positive")
        if hidden % 2 != 0:
            raise ValueError(f"hidden must be even for 4-byte copies, got {hidden}")

        self.world_size = int(world_size)
        self.num_local_experts = int(num_local_experts)
        self.tokens_per_rank = int(tokens_per_rank)
        self.hidden = int(hidden)
        self.device = device

        shape = (
            self.world_size,
            self.num_local_experts,
            self.tokens_per_rank,
            self.hidden,
        )
        nbytes = int(
            self.world_size
            * self.num_local_experts
            * self.tokens_per_rank
            * self.hidden
            * 2
        )
        self._group = dist.group.WORLD if dist.is_initialized() else None
        self._mnnvl = None
        self._local_malloc_ptr = None
        self._pointers: List[int] = []
        self._keepalive: list[object] = []

        if dist.is_initialized() and self.world_size > 1:
            mem, pointers = _alloc_mnnvl(nbytes)
            self._mnnvl = mem
            self._pointers = pointers
            self._keepalive.append(mem)
            rank = dist.get_rank()
            local_ptr = pointers[rank]
            dev_id = int(
                device.index
                if device.index is not None
                else torch.cuda.current_device()
            )
            local = _wrap_bf16(local_ptr, nbytes, shape, dev_id)
            self._keepalive.append(local)
            self.inbox = local
        else:
            from flashinfer.comm.cuda_ipc import cudart

            local_ptr = cudart.cudaMalloc(nbytes)
            self._local_malloc_ptr = local_ptr
            ptr = int(local_ptr.value)
            self._pointers = [ptr]
            dev_id = int(
                device.index
                if device.index is not None
                else torch.cuda.current_device()
            )
            local = _wrap_bf16(ptr, nbytes, shape, dev_id)
            self._keepalive.append(local)
            self.inbox = local

        self.inbox.zero_()
        self.peer_ptrs = torch.tensor(self._pointers, device=device, dtype=torch.int64)
        self._quorum = torch.ones(1, device=device, dtype=torch.int32)

    def zero(self) -> None:
        self.inbox.zero_()

    def wait_peers(self) -> None:
        """Device-ordered quorum so dest sees every source's sys stores."""
        if not dist.is_initialized() or dist.get_world_size() <= 1:
            return
        self._quorum.fill_(1)
        dist.all_reduce(self._quorum, group=self._group)

    def destroy(self) -> None:
        self.inbox = None  # type: ignore[assignment]
        self.peer_ptrs = None  # type: ignore[assignment]
        self._quorum = None
        self._keepalive = []
        self._pointers = []
        mem = self._mnnvl
        self._mnnvl = None
        if mem is not None:
            from flashinfer.comm.mnnvl import MnnvlMemory

            ptr = getattr(mem, "ptr", None)
            if ptr is not None:
                MnnvlMemory.close_mnnvl_memory(ptr)
        if self._local_malloc_ptr is not None:
            import ctypes

            from flashinfer.comm.cuda_ipc import cudart

            cudart.cudaFree(ctypes.c_void_p(int(self._local_malloc_ptr.value)))
            self._local_malloc_ptr = None
