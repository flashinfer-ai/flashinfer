"""Pull-based island reduce-scatter TP16 all-reduce runtime."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Optional, Sequence

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from b12x.preparation.types import Plan, require_prepared

from ._island_rs_cute import (
    HEADER_BYTES,
    MAX_BLOCKS,
    island_rs_peers,
)
from ._cuda_ipc import CudaRTLibrary
from .pcie_oneshot import (
    PCIeOneshotAllReduce,
    _finish_collective_runtime_setup,
    _normalize_device,
)


SUPPORTED_WORLD_SIZES = (16,)
SUPPORTED_BLOCKS = (1, 2, 4, 8, 16, 32)
_ISLAND_SIZE = 4
_ALIGNMENT = 256


def _align_up(value: int, alignment: int = _ALIGNMENT) -> int:
    return (int(value) + alignment - 1) // alignment * alignment


def _threads_from_env() -> int:
    # Default CUDA thread count for the TP16 equal-quarter launch.
    raw = os.getenv("B12X_PCIE_ISLAND_RS_THREADS", "512")
    threads = int(raw)
    if not 32 <= threads <= 1024 or threads % 32:
        raise ValueError(
            "B12X_PCIE_ISLAND_RS_THREADS must be a multiple of 32 in [32, 1024]"
        )
    return threads


def _wait_nanosleep_cycles_from_env() -> int:
    cycles = int(os.getenv("B12X_PCIE_ISLAND_RS_NANOSLEEP_CYCLES", "24"))
    if not 0 <= cycles <= 1024:
        raise ValueError("B12X_PCIE_ISLAND_RS_NANOSLEEP_CYCLES must be in [0, 1024]")
    return cycles


def _pick_blocks(elements: int) -> int:
    """Return the default launch geometry for a BF16 element count."""

    if elements <= 4096:
        return 8
    return 16


# Auto dispatch keeps a single 7,168-element row on the shorter hierarchical
# protocol and routes larger vectors through equal-quarter ownership when each
# rank-owned quarter is aligned to a complete 32-word transfer group.
CROSSOVER_ELEMENTS = 7_168
PREFERRED_ALIGNMENT_ELEMENTS = 128


class PCIeIslandRSAllReduce:
    """Equal-quarter BF16 TP16 all-reduce with no leader hotspot.

    Every rank owns one quarter of the vector and reaches six peers: the three
    other lanes of its four-GPU island, and the same lane in the three other
    islands. Peer degree is bounded at six, and no rank carries the whole
    vector on behalf of the others.
    """

    def __init__(
        self,
        *,
        exchange_group: ProcessGroup,
        device: torch.device | int | str,
        max_elements: int,
        blocks: Optional[int] = None,
    ) -> None:
        self.group = exchange_group
        self.rank = dist.get_rank(group=exchange_group)
        self.world_size = dist.get_world_size(group=exchange_group)
        self.device = _normalize_device(device)
        if self.world_size not in SUPPORTED_WORLD_SIZES:
            raise ValueError(
                f"island reduce-scatter requires TP16, got TP{self.world_size}"
            )
        if self.device.type != "cuda":
            raise ValueError("island reduce-scatter requires a CUDA device")
        if blocks is not None and blocks not in SUPPORTED_BLOCKS:
            raise ValueError(f"blocks must be one of {SUPPORTED_BLOCKS}")
        if max_elements <= 0 or max_elements % 2:
            raise ValueError("max_elements must be a positive even count")

        self.max_elements = int(max_elements)
        self.blocks = None if blocks is None else int(blocks)
        self.threads = _threads_from_env()
        self.wait_nanosleep_cycles = _wait_nanosleep_cycles_from_env()

        max_pairs = self.max_elements // 2
        # Quarter stride in BF16x2 words. Every transient region has two slots
        # selected by the device generation so a rank cannot overwrite data a
        # slower peer is still consuming from the preceding invocation.
        self.quarter_capacity = _align_up(
            (max_pairs + _ISLAND_SIZE - 1) // _ISLAND_SIZE, 8
        )
        quarter_bytes = self.quarter_capacity * 4
        vector_bytes = quarter_bytes * _ISLAND_SIZE
        self.stage_offset = _align_up(HEADER_BYTES)
        self.part_offset = _align_up(self.stage_offset + 2 * vector_bytes)
        self.final_offset = _align_up(self.part_offset + 2 * quarter_bytes)
        self.slab_bytes = _align_up(self.final_offset + 2 * vector_bytes)

        self._ipc = CudaRTLibrary()
        self._ipc.cudaSetDevice(self.device.index or 0)
        self._slab_ptrs: tuple[int, ...] = ()
        self._local_ptr = 0
        self._remote_ptrs: list[int] = []
        self._closed = False
        self._mapped_peers = island_rs_peers(self.rank, self.world_size)

        shared = PCIeOneshotAllReduce._allocate_shared_buffer(
            exchange_group,
            self.slab_bytes,
            zero_fill=True,
            ipc=self._ipc,
            peer_ranks=self._mapped_peers,
        )
        self._local_ptr = shared.local_ptr
        self._remote_ptrs = list(shared.remote_ptrs)
        self._slab_ptrs = shared.peer_ptrs

        # The native launcher is resolved and retained by the prepared plan;
        # construction allocates only the actual communicator-owned slabs.
        init_error: BaseException | None = None

        def detach_shared_ownership() -> None:
            self._slab_ptrs = ()
            self._remote_ptrs.clear()
            self._local_ptr = 0

        _finish_collective_runtime_setup(
            owner="PCIe island reduce-scatter",
            exchange_group=exchange_group,
            ipc=self._ipc,
            shared=shared,
            local_error=init_error,
            detach_shared_ownership=detach_shared_ownership,
        )

    @property
    def mapped_peer_count(self) -> int:
        return len(self._remote_ptrs)

    @property
    def mapped_peers(self) -> tuple[int, ...]:
        """Group ranks whose CUDA IPC allocations are mapped by this rank."""

        return self._mapped_peers

    def should_allreduce(self, inp: torch.Tensor) -> bool:
        return (
            not self._closed
            and inp.device == self.device
            and inp.dtype == torch.bfloat16
            and inp.is_contiguous()
            and 0 < inp.numel() <= self.max_elements
            and inp.numel() % 2 == 0
            and inp.data_ptr() % 4 == 0
        )

    def all_reduce(
        self,
        inp: torch.Tensor,
        *,
        plan: Plan,
        out: Optional[torch.Tensor] = None,
        blocks: Optional[int] = None,
        stream: object = None,
        channel_id: Optional[str] = None,
    ) -> torch.Tensor:
        del channel_id
        state = require_prepared(plan, "comm.pcie", self.device)
        state.require_runtime(self)
        if not self.should_allreduce(inp):
            raise ValueError(
                "input does not satisfy island reduce-scatter requirements "
                f"(shape={tuple(inp.shape)}, dtype={inp.dtype})"
            )
        if tuple(inp.shape) != tuple(state.query.call["inp"]["shape"]):
            raise ValueError("input shape differs from the prepared island plan")
        if out is None:
            out = torch.empty_like(inp)
        if (
            out.dtype != inp.dtype
            or out.device != inp.device
            or out.shape != inp.shape
            or not out.is_contiguous()
            or out.data_ptr() % 4 != 0
        ):
            raise ValueError("output must match input and be 4-byte aligned")
        selected = int(state.query.call["blocks"])
        if blocks is not None and int(blocks) != selected:
            raise ValueError("blocks differs from the prepared island plan")
        if selected not in SUPPORTED_BLOCKS or selected > MAX_BLOCKS:
            raise ValueError(f"blocks must be one of {SUPPORTED_BLOCKS}")
        return self._run_prepared(state.launcher(), inp, out, selected, stream)

    def _run_prepared(self, launcher, inp, out, blocks, stream):
        """Invoke an already-prepared native launcher after caller validation."""
        with torch.cuda.device(self.device):
            launcher(
                self._slab_ptrs,
                inp.data_ptr(),
                out.data_ptr(),
                self.stage_offset,
                self.part_offset,
                self.final_offset,
                self.quarter_capacity,
                inp.numel(),
                blocks,
                stream,
            )
        return out

    def prepare_channels(self, channel_ids: Sequence[str]) -> None:
        """Accept owner names for the single ordered collective channel."""

        del channel_ids

    def for_stream(
        self,
        stream: object = None,
        *,
        channel_id: Optional[str] = None,
    ) -> "PCIeIslandRSAllReduce":
        # Binding is validated at launch through the explicit stream argument.
        # Owner names are accepted because this runtime has one serialized
        # channel rather than per-owner storage.
        del channel_id
        if stream is not None and not hasattr(stream, "cuda_stream"):
            int(stream)
        return self

    @contextmanager
    def capture(
        self,
        stream: object = None,
        *,
        channel_id: Optional[str] = None,
    ):
        self.for_stream(stream, channel_id=channel_id)
        if stream is None:
            yield self
        else:
            torch_stream = (
                stream
                if hasattr(stream, "cuda_stream")
                else torch.cuda.ExternalStream(int(stream), device=self.device)
            )
            with torch.cuda.stream(torch_stream):
                yield self

    def register_graph_buffers(self) -> None:
        return None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        dist.barrier(group=self.group)
        self._slab_ptrs = ()
        for ptr in self._remote_ptrs:
            self._ipc.cudaIpcCloseMemHandle(ptr)
        self._remote_ptrs.clear()
        dist.barrier(group=self.group)
        if self._local_ptr:
            self._ipc.cudaFree(self._local_ptr)
            self._local_ptr = 0
        dist.barrier(group=self.group)


__all__ = [
    "CROSSOVER_ELEMENTS",
    "PCIeIslandRSAllReduce",
    "PREFERRED_ALIGNMENT_ELEMENTS",
    "SUPPORTED_WORLD_SIZES",
]
