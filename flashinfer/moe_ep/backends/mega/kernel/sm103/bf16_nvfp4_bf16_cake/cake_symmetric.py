"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# Symmetric buffers and peer-address tables of the Cake W4A16 MegaMoE route.
#
# The programs address their peers through by-value tables of base addresses
# (slot ``r`` = rank ``r``), so any allocator that yields peer-mapped memory
# serves them. With more than one rank this module allocates through the
# NVSHMEM runtime that ``bootstrap_moe_ep_runtime`` initializes for the EP
# group (requirement ``NVSHMEM``, the recipe of the CuTe DSL MegaMoE
# backends). A world of one needs no peer mapping: its buffers are plain
# device tensors and the table is the local address. A table crosses the
# binding as a contiguous CPU int64 tensor of exactly the table's capacity,
# entry ``r`` rank ``r``'s address, the entries at or above the world size
# zero; the binding copies it into the kernel's by-value parameter at every
# launch and checks that the local pointer bound beside it is its own entry.
# ``nvshmem`` is imported inside the allocation functions only.

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from .cake_plan import PEER_TABLE_CAPACITY, PG_FLAG_WORDS


def as_int64_image(raw: int) -> int:
    """Two's-complement int64 image of an unsigned 64-bit address."""

    if not 0 <= raw < 2**64:
        raise ValueError(f"address {raw!r} is not an unsigned 64-bit value")
    return raw - 2**64 if raw >= 2**63 else raw


def peer_table(pointers: Sequence[int], *, world_size: int) -> tuple[int, ...]:
    """Validate exactly ``world_size`` non-null peer base addresses."""

    if isinstance(pointers, (str, bytes)) or not isinstance(pointers, (list, tuple)):
        raise ValueError("peer pointers must be a list or tuple of ints")
    if len(pointers) != world_size:
        raise ValueError(
            f"peer table needs exactly {world_size} addresses, got {len(pointers)}"
        )
    if not 1 <= world_size <= PEER_TABLE_CAPACITY:
        raise ValueError(
            f"peer table capacity is {PEER_TABLE_CAPACITY} slots, got world size "
            f"{world_size}"
        )
    for index, pointer in enumerate(pointers):
        if isinstance(pointer, bool) or not isinstance(pointer, int):
            raise ValueError(f"peer address {index} must be an int, got {pointer!r}")
        if not 0 < pointer < 2**64:
            raise ValueError(f"peer address {index} is out of range: {pointer!r}")
    return tuple(int(p) for p in pointers)


def peer_table_images(table: Sequence[int], *, rank: int) -> tuple[int, ...]:
    """Signed int64 images of a peer table, padded to the capacity with the
    self entry (slot ``rank``): a pad slot is never a null pointer."""

    images = [as_int64_image(int(p)) for p in table]
    if len(images) > PEER_TABLE_CAPACITY:
        raise ValueError(
            f"peer table has {len(images)} entries; the capacity is {PEER_TABLE_CAPACITY}"
        )
    if (
        isinstance(rank, bool)
        or not isinstance(rank, int)
        or not 0 <= rank < len(images)
    ):
        raise ValueError(
            f"rank {rank!r} is outside a peer table of {len(images)} entries"
        )
    return tuple(images + [images[rank]] * (PEER_TABLE_CAPACITY - len(images)))


def peer_table_tensor(table: Sequence[int], *, rank: int) -> torch.Tensor:
    """The binding's argument: a contiguous CPU int64 ``[PEER_TABLE_CAPACITY]``
    tensor of the table's addresses (as images); slots at or past the world
    size repeat this rank's own address."""

    return torch.tensor(
        peer_table_images(table, rank=rank), dtype=torch.int64, device="cpu"
    )


@dataclass
class SymmetricBuffer:
    """One peer-mapped tensor with its peer base-address table."""

    tensor: torch.Tensor
    peer_pointers: tuple[int, ...]  # unsigned addresses; slot r = rank r
    peers: torch.Tensor  # CPU int64 [PEER_TABLE_CAPACITY]; the binding's table
    nvshmem: bool
    freed: bool = False

    @property
    def local(self) -> torch.Tensor:
        return self.tensor

    def free(self) -> None:
        if self.freed:
            return
        self.freed = True
        if self.nvshmem:
            _free_symmetric_tensor(self.tensor)


def check_self_entry(buffer: SymmetricBuffer, *, rank: int, name: str) -> None:
    """Slot ``rank`` of the table must be the local base the program reads
    through: the programs poll or read a buffer through the local pointer
    while the peers write it through the table, so a table whose own entry is
    another address is a silent hang, not a fault. Asserted once at
    allocation (the binding re-checks it per launch); ``rank`` is the
    process-group rank, so the assert also pins the NVSHMEM PE numbering to
    the process-group numbering the tables assume."""

    table = buffer.peer_pointers
    if not 0 <= rank < len(table):
        raise RuntimeError(
            f"symmetric buffer '{name}': peer table has no entry for rank {rank}"
        )
    local = int(buffer.local.data_ptr())
    if int(table[rank]) != local:
        raise RuntimeError(
            f"symmetric buffer '{name}': peer table entry {rank} is "
            f"{int(table[rank]):#x}, not this rank's local pointer {local:#x} "
            "(symmetric self entry)"
        )
    if int(buffer.peers[rank].item()) != as_int64_image(local):
        raise RuntimeError(
            f"symmetric buffer '{name}': the binding's table disagrees with the "
            "peer pointers"
        )


def _peer_bases(sym_tensor: torch.Tensor, world_size: int) -> list[int]:
    import nvshmem.core
    from nvshmem.core.interop.torch import tensor_get_buffer

    local_base = int(sym_tensor.data_ptr())
    my_pe = int(nvshmem.core.my_pe())
    buf, _size, _dtype = tensor_get_buffer(sym_tensor)
    bases = []
    for peer in range(world_size):
        if peer == my_pe:
            # The self peer maps to the local base; resolving it through the
            # peer-buffer tracker would defer the explicit free to GC.
            bases.append(local_base)
            continue
        peer_buf = nvshmem.core.get_peer_buffer(buf, peer)
        bases.append(int(torch.utils.dlpack.from_dlpack(peer_buf).data_ptr()))
    return bases


def _free_symmetric_tensor(tensor: torch.Tensor) -> None:
    import nvshmem.core

    try:
        nvshmem.core.free_tensor(tensor)
    except (RuntimeError, ValueError, TypeError) as exc:
        message = str(exc).lower()
        if any(token in message for token in ("already", "freed", "invalid")):
            return
        raise


def allocate_symmetric(
    shape: Sequence[int],
    dtype: torch.dtype,
    *,
    device: torch.device,
    world_size: int,
    rank: int,
) -> SymmetricBuffer:
    """Zero-initialised peer-mapped tensor plus its peer table.

    COLLECTIVE over the NVSHMEM PEs when ``world_size > 1`` (every rank
    allocates the same shape); a world of one allocates a plain device tensor.
    """

    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "symmetric allocation cannot run during CUDA graph capture; warm up first"
        )
    if world_size == 1:
        tensor = torch.zeros(tuple(int(s) for s in shape), dtype=dtype, device=device)
        table = peer_table([int(tensor.data_ptr())], world_size=1)
        nvshmem_backed = False
    else:
        import nvshmem.core

        if int(nvshmem.core.n_pes()) != world_size:
            raise RuntimeError(
                f"NVSHMEM reports {int(nvshmem.core.n_pes())} PEs; this EP group has "
                f"{world_size} ranks"
            )
        tensor = nvshmem.core.tensor(tuple(int(s) for s in shape), dtype=dtype)
        if tensor.device != device:
            raise RuntimeError(
                f"symmetric tensor landed on {tensor.device}, expected {device}"
            )
        tensor.zero_()
        table = peer_table(_peer_bases(tensor, world_size), world_size=world_size)
        nvshmem_backed = True
    return SymmetricBuffer(
        tensor=tensor,
        peer_pointers=table,
        peers=peer_table_tensor(table, rank=rank),
        nvshmem=nvshmem_backed,
    )


def allocate_flag_pad(
    *,
    device: torch.device,
    group: Any,
    world_size: int,
    rank: int,
) -> SymmetricBuffer:
    """The process-group flag pad (``PG_FLAG_WORDS`` u32 per rank), zeroed
    once and never re-zeroed. Every rank zeroes its own pad, synchronizes its
    device and joins a process-group barrier so no peer observes a stale word.
    """

    pad = allocate_symmetric(
        (PG_FLAG_WORDS,),
        torch.uint32,
        device=device,
        world_size=world_size,
        rank=rank,
    )
    torch.cuda.synchronize(device)
    if world_size > 1:
        import torch.distributed as dist

        dist.barrier(group=group)
    return pad


__all__ = [
    "SymmetricBuffer",
    "allocate_flag_pad",
    "allocate_symmetric",
    "as_int64_image",
    "check_self_entry",
    "peer_table",
    "peer_table_images",
    "peer_table_tensor",
]
