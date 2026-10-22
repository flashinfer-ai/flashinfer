"""NCCL GIN arena for the internode SM90 FP8 MegaMoE path.

One nccl4py-allocated buffer is registered as a ``COLL_SYMMETRIC`` window on a
GIN-capable communicator.  Every peer-visible buffer of a session (inputs,
output, shared workspace with the Stage 1 landing / return planes) is carved
out of it, so:

* the device can RDMA into any of them with ``gin.put`` (window + offset);
* node-local NVLink peers address them through one per-peer offset
  (``window.get_peer_device_pointer``), which keeps the kernel's single
  offset-table invariant (``SymBufferHost``) intact -- the table covers the
  ``ranks_per_node`` peers of this NVLink domain only.

Requirements: nccl4py with the CuTe GIN device API (0.6.0 was validated) and a
matching NCCL (``nvidia-nccl-cu13`` >= 2.32.3 for nccl4py 0.6.0; point
``NCCL_HOME`` at it so the device bitcode matches the runtime library), RDMA
NICs with GPUDirect (``nvidia_peermem``), and torch.distributed initialised for
the unique-id broadcast.
"""

from __future__ import annotations

from typing import Tuple

import torch


def _broadcast_unique_id(rank: int, world_size: int) -> bytes:
    from nccl.core import UniqueId, get_unique_id

    if world_size == 1:
        return bytes(get_unique_id())
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        raise RuntimeError(
            "the NCCL GIN arena needs torch.distributed for the unique-id broadcast."
        )
    payload = [bytes(get_unique_id()) if rank == 0 else None]
    torch.distributed.broadcast_object_list(payload, src=0)
    if payload[0] is None:
        raise RuntimeError("rank 0 did not broadcast an NCCL unique id.")
    UniqueId.from_bytes(payload[0])
    return payload[0]


class GinArena:
    """Registered symmetric arena + GIN device communicator of one session."""

    _align = 256

    def __init__(
        self,
        *,
        rank: int,
        world_size: int,
        ranks_per_node: int,
        capacity_bytes: int,
        signal_count: int,
        gin_force_enable: bool = False,
    ) -> None:
        from nccl.core import (
            Communicator,
            NCCLDevCommRequirements,
            NcclGinConnectionType,
            UniqueId,
            WindowFlag,
        )
        from nccl.core.interop.torch import empty as nccl_empty

        if world_size % ranks_per_node != 0:
            raise ValueError(
                f"ranks_per_node={ranks_per_node} must divide world_size={world_size}."
            )
        if signal_count <= 0:
            raise ValueError(f"signal_count must be positive, got {signal_count}.")
        self.rank = rank
        self.world_size = world_size
        self.ranks_per_node = ranks_per_node
        self.capacity = (
            int(capacity_bytes + self._align - 1) // self._align * self._align
        )

        uid = UniqueId.from_bytes(_broadcast_unique_id(rank, world_size))
        self.comm = Communicator.init(world_size, rank, uid)
        req = dict(
            gin_signal_count=int(signal_count),
            gin_connection_type=NcclGinConnectionType.RAIL,
        )
        if gin_force_enable:
            req["gin_force_enable"] = True
        try:
            self.dev_comm = self.comm.create_dev_comm(NCCLDevCommRequirements(**req))
        except Exception:
            self.comm.destroy()
            raise
        self.buffer = nccl_empty(
            (self.capacity,), dtype=torch.uint8, device=torch.cuda.current_device()
        )
        self.buffer.zero_()
        self.window = self.comm.register_window(self.buffer, WindowFlag.COLL_SYMMETRIC)
        if self.window is None:
            raise RuntimeError("NCCL returned a null registered-window handle.")
        self.base = int(self.buffer.data_ptr())
        node_base = (rank // ranks_per_node) * ranks_per_node
        offsets = []
        for peer in range(node_base, node_base + ranks_per_node):
            ptr = self.window.get_peer_device_pointer(peer)
            if ptr is None:
                raise RuntimeError(
                    f"NCCL LSA peer {peer} is not addressable from rank {rank}; the "
                    "ranks of one logical node must share an NVLink domain."
                )
            offsets.append(int(ptr) - self.base)
        self.peer_offsets: Tuple[int, ...] = tuple(offsets)
        self._cursor = 0
        torch.cuda.synchronize()

    @property
    def local_rank_in_node(self) -> int:
        return self.rank % self.ranks_per_node

    def alloc(self, shape: Tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        """Carve a zeroed, 256-byte aligned tensor out of the arena (bump allocator)."""
        numel = 1
        for d in shape:
            numel *= int(d)
        nbytes = numel * torch.empty((), dtype=dtype).element_size()
        start = (self._cursor + self._align - 1) // self._align * self._align
        end = start + nbytes
        if end > self.capacity:
            raise RuntimeError(
                f"GIN arena exhausted: need {end} bytes, capacity {self.capacity}."
            )
        self._cursor = end
        return self.buffer[start:end].view(dtype).view(shape)

    @staticmethod
    def padded_bytes(shape: Tuple[int, ...], dtype: torch.dtype) -> int:
        numel = 1
        for d in shape:
            numel *= int(d)
        nbytes = numel * torch.empty((), dtype=dtype).element_size()
        return (nbytes + GinArena._align - 1) // GinArena._align * GinArena._align

    def kernel_handles(self):
        """``(DevComm, Window)`` CuTe kernel-argument wrappers."""
        from nccl.core.device import cute as nccl_cute

        return nccl_cute.DevComm(self.dev_comm), nccl_cute.Window(self.window)

    def close(self) -> None:
        """Release in reverse dependency order (window, dev comm, comm)."""
        torch.cuda.synchronize()
        for name in ("window", "dev_comm"):
            obj = getattr(self, name, None)
            if obj is None:
                continue
            for meth in ("close", "destroy"):
                fn = getattr(obj, meth, None)
                if fn is not None:
                    fn()
                    break
            setattr(self, name, None)
        comm = getattr(self, "comm", None)
        if comm is not None:
            comm.destroy()
            self.comm = None
        self.buffer = None
