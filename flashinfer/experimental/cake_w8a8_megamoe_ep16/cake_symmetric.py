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

# NVSHMEM symmetric buffers and peer pointer tables for the Cake W8A8 MXFP8
# MegaMoE EP16 route.
#
# The device programs use plain CUDA over peer base pointers (bulk copies into
# the peers' inboxes, release/acquire flag stores, generic stores of the route
# terms and system-scope atomic adds into the completion counters); any
# allocator that yields peer-mapped base addresses works. This module uses the NVSHMEM runtime that
# ``bootstrap_moe_ep_runtime`` initializes for the EP group (requirement
# ``NVSHMEM``), the same recipe as the SM100 CuTe DSL MegaMoE backends. A peer
# table is exactly the 16 peer base addresses (slot ``r`` = rank ``r``, all
# non-null), kept as Python ints, as their signed int64 images packed once
# into a ``tvm_ffi.Shape`` (the bindings' ``Shape`` argument, so no launch
# rebuilds it) and as an int64 device tensor. The bindings copy the 16 entries
# into slots 0..15 of the programs' 32-slot device table and zero the rest; no
# padding crosses the FFI. ``nvshmem`` and ``tvm_ffi`` are imported inside the
# allocation functions only (this module is imported at core import time).

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from .cake_plan import PEER_TABLE_CAPACITY, PG_FLAG_WORDS, WORLD_SIZE


def as_int64_image(raw: int) -> int:
    """Two's-complement int64 image of an unsigned 64-bit address."""

    if not 0 <= raw < 2**64:
        raise ValueError(f"address {raw!r} is not an unsigned 64-bit value")
    return raw - 2**64 if raw >= 2**63 else raw


def peer_table(
    pointers: Sequence[int], *, world_size: int = WORLD_SIZE
) -> tuple[int, ...]:
    """Validate exactly ``world_size`` non-null peer base addresses (slot ``r`` = rank ``r``)."""

    if isinstance(pointers, (str, bytes)) or not isinstance(pointers, (list, tuple)):
        raise ValueError("peer pointers must be a list or tuple of ints")
    if len(pointers) != world_size:
        raise ValueError(
            f"peer table needs exactly {world_size} addresses, got {len(pointers)}"
        )
    if world_size > PEER_TABLE_CAPACITY:
        raise ValueError(
            f"peer table capacity is {PEER_TABLE_CAPACITY} slots, got world size {world_size}"
        )
    for index, pointer in enumerate(pointers):
        if isinstance(pointer, bool) or not isinstance(pointer, int):
            raise ValueError(f"peer address {index} must be an int, got {pointer!r}")
        if not 0 < pointer < 2**64:
            raise ValueError(f"peer address {index} is out of range: {pointer!r}")
    return tuple(int(p) for p in pointers)


def peer_table_images(
    table: Sequence[int], *, world_size: int = WORLD_SIZE
) -> tuple[int, ...]:
    """Signed int64 images of a peer table: the bindings' ``Shape`` argument.

    Exactly ``world_size`` entries; an address at or above ``2**63`` travels
    as its two's-complement image and the binding casts it back to ``uint64``.
    """

    if len(table) != world_size:
        raise ValueError(
            f"peer table must have exactly {world_size} entries, got {len(table)}"
        )
    return tuple(as_int64_image(int(p)) for p in table)


def peer_table_tensor(
    table: Sequence[int], *, device: torch.device | str
) -> torch.Tensor:
    """int64 ``[world_size]`` device tensor of a peer table (addresses as images)."""

    return torch.tensor(peer_table_images(table), dtype=torch.int64, device=device)


def peer_table_shape(
    table: Sequence[int], *, world_size: int = WORLD_SIZE
) -> tuple[int, ...]:
    """``tvm_ffi.Shape`` of :func:`peer_table_images` (a ``tuple`` subclass).

    Built once per symmetric buffer: the bindings take the table as
    ``tvm::ffi::Shape``, and a pre-built ``Shape`` crosses the FFI without the
    per-call tuple -> Array -> Shape conversion a bare tuple would pay.
    """

    import tvm_ffi

    return tvm_ffi.Shape(peer_table_images(table, world_size=world_size))


@dataclass
class SymmetricBuffer:
    """One NVSHMEM symmetric tensor with its peer base-address table."""

    tensor: torch.Tensor
    peer_pointers: tuple[int, ...]  # 16 unsigned addresses; slot r = rank r
    # The same 16 entries as signed int64 in a ``tvm_ffi.Shape`` (host arg).
    peer_images: tuple[int, ...]
    peers: torch.Tensor  # int64 [16] device tensor of the images
    freed: bool = False

    @property
    def local(self) -> torch.Tensor:
        return self.tensor

    def free(self) -> None:
        if self.freed:
            return
        self.freed = True
        _free_symmetric_tensor(self.tensor)


def check_self_entry(buffer: SymmetricBuffer, *, rank: int, name: str) -> None:
    """Symmetric self entry: slot ``rank`` of the peer table is the local base.

    Manifest ``route_contract.push_dispatch.peer_table_rule`` and
    ``route_contract.fc2_completion.counter_layout``: on every rank the
    table's own entry must be the local pointer bound beside it
    (``push_inbox_peers[rank] == push_inbox``, ``push_flags_peers[rank] ==
    push_flags``, ``remote_terms_peers[rank] == remote_terms``,
    ``fc2_completion_peers[rank] == fc2_completion``). The programs poll or
    read the buffer through the local pointer while the peers write it
    through the table, so a table whose own entry is another address is a
    silent hang, not a fault. The workspace asserts it once at allocation
    (the bindings re-check it per launch); ``rank`` is the process-group rank
    the bindings receive, so the assert also pins the NVSHMEM PE numbering to
    the process-group numbering the tables assume (slot ``r`` = rank ``r``).
    """

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
    world_size: int = WORLD_SIZE,
) -> SymmetricBuffer:
    """Zero-initialised symmetric tensor plus its ``world_size``-entry peer table.

    COLLECTIVE over the NVSHMEM PEs (every rank allocates the same shape).
    """

    import nvshmem.core

    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "symmetric allocation cannot run during CUDA graph capture; warm up first"
        )
    if int(nvshmem.core.n_pes()) != world_size:
        raise RuntimeError(
            f"NVSHMEM reports {int(nvshmem.core.n_pes())} PEs; this route needs {world_size}"
        )
    tensor = nvshmem.core.tensor(tuple(int(s) for s in shape), dtype=dtype)
    if tensor.device != device:
        raise RuntimeError(
            f"symmetric tensor landed on {tensor.device}, expected {device}"
        )
    tensor.zero_()
    table = peer_table(_peer_bases(tensor, world_size), world_size=world_size)
    return SymmetricBuffer(
        tensor=tensor,
        peer_pointers=table,
        peer_images=peer_table_shape(table, world_size=world_size),
        peers=peer_table_tensor(table, device=device),
    )


def allocate_flag_pad(
    *,
    device: torch.device,
    group: Any,
    world_size: int = WORLD_SIZE,
) -> SymmetricBuffer:
    """The process-group flag pad: ``PG_FLAG_WORDS`` u32, zeroed once.

    The pad is the process group's flag words and is passed to every program
    that names the process group (``publish_push``, ``local_pack``, ``fc2``,
    ``combine``); it persists across forwards and is never re-zeroed. The
    route has no device-side barrier program: ``fc2`` signals its completion
    into the symmetric ``fc2_completion`` counter and ``combine`` waits on it.
    Every rank zeroes its own pad, synchronizes its device and joins a
    ``torch.distributed`` barrier so no peer observes a stale word.
    """

    import torch.distributed as dist

    pad = allocate_symmetric(
        (PG_FLAG_WORDS,), torch.uint32, device=device, world_size=world_size
    )
    torch.cuda.synchronize(device)
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
    "peer_table_shape",
    "peer_table_tensor",
]
