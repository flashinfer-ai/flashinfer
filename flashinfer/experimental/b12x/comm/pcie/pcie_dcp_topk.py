"""Exact owner-sharded PCIe transport for DCP sparse top-k."""

from __future__ import annotations

from contextlib import contextmanager, suppress
from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from b12x.preparation.types import Plan, require_prepared

from ._cuda_ipc import CudaRTLibrary
from ._dcp_cute_common import signal_bytes
from .pcie_oneshot import (
    IPC_SLAB_ALIGNMENT,
    PCIeOneshotAllReduce,
    _align_up,
    _coordinated_close_channels,
    _current_stream_key,
    _device_guard,
    _is_current_stream_capturing,
    _normalize_device,
    _OwnedSharedBuffer,
)


SUPPORTED_WORLD_SIZES = (2, 3, 4, 6, 8)
_MAX_BLOCKS = 128
_SIGNAL_BYTES = signal_bytes(_MAX_BLOCKS)


def _validate_launch_config(*, threads: int, block_limit: int, world_size: int) -> None:
    if threads < world_size or threads > 512 or threads % 32 != 0:
        raise ValueError("threads must be a multiple of 32 in [32, 512]")
    if block_limit <= 0 or block_limit > 128:
        raise ValueError("block_limit must be in [1, 128]")


@dataclass(frozen=True)
class _DoubleBufferLayout:
    signal_bytes: int
    staging0_offset: int
    staging1_offset: int
    slot_bytes: int
    slab_bytes: int
    plane_bytes: int = 0


def _candidate_staging_layout(
    *,
    signal_bytes: int,
    max_rows: int,
    topk: int,
    world_size: int,
) -> _DoubleBufferLayout:
    if signal_bytes <= 0:
        raise ValueError("signal_bytes must be positive")
    if world_size not in SUPPORTED_WORLD_SIZES:
        raise ValueError(f"unsupported world size {world_size}")
    if max_rows <= 0 or max_rows % world_size != 0:
        raise ValueError("max_rows must be positive and divisible by world_size")
    if topk <= 0 or topk % 4 != 0:
        raise ValueError("topk must be a positive multiple of 4")

    max_owner_rows = max_rows // world_size
    plane_bytes = max_owner_rows * world_size * topk * 4
    slot_bytes = _align_up(plane_bytes * 2, IPC_SLAB_ALIGNMENT)
    staging0_offset = _align_up(signal_bytes, IPC_SLAB_ALIGNMENT)
    staging1_offset = staging0_offset + slot_bytes
    return _DoubleBufferLayout(
        signal_bytes=signal_bytes,
        staging0_offset=staging0_offset,
        staging1_offset=staging1_offset,
        slot_bytes=slot_bytes,
        slab_bytes=staging1_offset + slot_bytes,
        plane_bytes=plane_bytes,
    )


def _tensor_from_cuda_pointer(
    pointer: int,
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Create a non-owning tensor view over channel-owned CUDA storage."""
    numel = 1
    for extent in shape:
        numel *= int(extent)
    nbytes = numel * torch.empty((), dtype=dtype).element_size()
    storage = torch._C._construct_storage_from_data_pointer(
        int(pointer),
        device,
        int(nbytes),
    )
    stride: list[int] = []
    running = 1
    for extent in reversed(shape):
        stride.append(running)
        running *= int(extent)
    return torch._C._construct_CUDA_Tensor_From_Storage_And_Metadata(
        {
            "dtype": dtype,
            "size": tuple(int(value) for value in shape),
            "stride": tuple(reversed(stride)),
            "storage_offset": 0,
        },
        storage,
    )


def owner_stage_reference(
    rank_indices: torch.Tensor,
    rank_scores: torch.Tensor,
    owner_rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build one DCP owner's exact row-major candidate planes."""
    if rank_indices.shape != rank_scores.shape or rank_indices.ndim != 3:
        raise ValueError(
            "rank candidates must have matching [world, rows, topk] shapes"
        )
    world_size, rows, _ = rank_indices.shape
    if rows % world_size != 0:
        raise ValueError("rows must be divisible by world size")
    if not 0 <= owner_rank < world_size:
        raise ValueError("invalid owner rank")
    owner_rows = rows // world_size
    row_slice = slice(owner_rank * owner_rows, (owner_rank + 1) * owner_rows)
    return (
        rank_indices[:, row_slice].transpose(0, 1).flatten(1).contiguous(),
        rank_scores[:, row_slice].transpose(0, 1).flatten(1).contiguous(),
    )


class _IPCChannel:
    def _init_channel(
        self,
        *,
        device: torch.device | int | str,
        exchange_group: Optional[ProcessGroup],
        ipc: Optional[CudaRTLibrary],
        owned_buffers: Optional[Sequence[_OwnedSharedBuffer]],
        stream_affine: bool,
    ) -> None:
        self.device = _normalize_device(device)
        self.exchange_group = exchange_group
        self._ipc = ipc
        self._owned_buffers = list(owned_buffers or ())
        self._stream_affine = bool(stream_affine)
        self._owner_stream_key: Optional[int] = None
        self._closed = False
        self._ipc_imports_closed = False
        self._ipc_exports_freed = False

    def _bind_stream(self) -> None:
        if not self._stream_affine or self.device.type != "cuda":
            return
        # CUDA graph capture substitutes a torch-owned capture stream handle
        # for the logical stream.  Keep affinity bound to the caller's eager
        # stream so the same channel can be warmed, captured, and reused.
        if _is_current_stream_capturing(self.device):
            return
        stream_key = _current_stream_key(self.device)
        if stream_key is None:
            return
        if self._owner_stream_key is None:
            self._owner_stream_key = int(stream_key)
        elif self._owner_stream_key != int(stream_key):
            raise RuntimeError(
                f"{type(self).__name__} is stream-affine; create one instance "
                "per CUDA stream"
            )

    def _close_ipc_imports(self) -> None:
        if self._ipc_imports_closed:
            return
        self._closed = True
        if self._ipc is not None:
            for shared in self._owned_buffers:
                for ptr in shared.remote_ptrs:
                    with suppress(Exception):
                        self._ipc.cudaIpcCloseMemHandle(ptr)
        self._ipc_imports_closed = True

    def _free_ipc_exports(self) -> None:
        if self._ipc_exports_freed:
            return
        self._close_ipc_imports()
        self._candidate_views = ()
        if self._ipc is not None:
            for shared in self._owned_buffers:
                with suppress(Exception):
                    self._ipc.cudaFree(shared.local_ptr)
        self._owned_buffers.clear()
        self._ipc_exports_freed = True

    def close(self) -> None:
        _coordinated_close_channels(
            (self,),
            exchange_group=self.exchange_group,
            device=self.device,
        )

    def close_coordinated(self) -> None:
        """Collectively close peer mappings before freeing exported storage."""
        _coordinated_close_channels(
            (self,),
            exchange_group=self.exchange_group,
            device=self.device,
        )

    def __del__(self) -> None:
        return None


class PCIeDCPTopKOwnerExchange(_IPCChannel):
    """Write exact DCP candidates directly into each row owner's IPC slab.

    CUDA graphs captured from one instance share a device epoch and must be
    replayed serially. Use a separate instance for independently replayable
    graphs. Returned candidate views alias channel-owned staging and their
    consumers must be enqueued on the capture stream before the next replay.
    """

    def __init__(
        self,
        *,
        rank: int,
        world_size: int,
        device: torch.device | int | str,
        signal_ptrs: Sequence[int],
        staging0_ptrs: Sequence[int],
        staging1_ptrs: Sequence[int],
        max_rows: int,
        topk: int,
        exchange_group: Optional[ProcessGroup] = None,
        ipc: Optional[CudaRTLibrary] = None,
        owned_buffers: Optional[Sequence[_OwnedSharedBuffer]] = None,
        ext_module=None,
        stream_affine: bool = True,
    ) -> None:
        # Compatibility-only: native extension injection was removed.
        del ext_module
        if world_size not in SUPPORTED_WORLD_SIZES:
            raise ValueError(f"unsupported world size {world_size}")
        if not 0 <= rank < world_size:
            raise ValueError(f"invalid rank {rank} for world size {world_size}")
        if max_rows <= 0 or max_rows % world_size != 0:
            raise ValueError("max_rows must be positive and divisible by world_size")
        if topk <= 0 or topk % 4 != 0:
            raise ValueError("topk must be a positive multiple of 4")
        if not (
            len(signal_ptrs) == len(staging0_ptrs) == len(staging1_ptrs) == world_size
        ):
            raise ValueError("signal and staging pointers must match world size")

        self._init_channel(
            device=device,
            exchange_group=exchange_group,
            ipc=ipc,
            owned_buffers=owned_buffers,
            stream_affine=stream_affine,
        )
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.max_rows = int(max_rows)
        self.max_owner_rows = self.max_rows // self.world_size
        self.topk = int(topk)
        self._signal_ptrs = tuple(int(ptr) for ptr in signal_ptrs)
        self._staging_ptrs = (
            tuple(int(ptr) for ptr in staging0_ptrs),
            tuple(int(ptr) for ptr in staging1_ptrs),
        )
        self._candidate_plane_elems = (
            self.max_owner_rows * self.world_size * self.topk
        )
        self._next_slot = 0
        self._graph_slot: Optional[int] = None
        self._capture_context_depth = 0
        candidate_shape = (self.max_owner_rows, self.world_size * self.topk)
        if self.device.type == "cuda":
            self._candidate_views = tuple(
                (
                    _tensor_from_cuda_pointer(
                        slot_ptrs[self.rank],
                        candidate_shape,
                        dtype=torch.int32,
                        device=self.device,
                    ),
                    _tensor_from_cuda_pointer(
                        slot_ptrs[self.rank] + self._candidate_plane_elems * 4,
                        candidate_shape,
                        dtype=torch.float32,
                        device=self.device,
                    ),
                )
                for slot_ptrs in self._staging_ptrs
            )
        else:
            # CPU construction remains useful for validation-only unit tests;
            # no launch is possible until tests install synthetic views.
            self._candidate_views = ()

    @classmethod
    def from_exchange_group(
        cls,
        *,
        exchange_group: ProcessGroup,
        device: torch.device | int | str,
        max_rows: int,
        topk: int,
        ext_module=None,
        stream_affine: bool = True,
    ) -> "PCIeDCPTopKOwnerExchange":
        rank = dist.get_rank(group=exchange_group)
        world_size = dist.get_world_size(group=exchange_group)
        device_obj = _normalize_device(device)
        if device_obj.type != "cuda":
            raise ValueError("DCP top-k owner exchange requires a CUDA device")
        ipc = CudaRTLibrary()
        ipc.cudaSetDevice(device_obj.index or 0)
        layout = _candidate_staging_layout(
            signal_bytes=_SIGNAL_BYTES,
            max_rows=max_rows,
            topk=topk,
            world_size=world_size,
        )
        owned: list[_OwnedSharedBuffer] = []
        try:
            slab = PCIeOneshotAllReduce._allocate_shared_buffer(
                exchange_group,
                layout.slab_bytes,
                zero_fill=True,
                ipc=ipc,
            )
            owned.append(slab)
            return cls(
                rank=rank,
                world_size=world_size,
                device=device_obj,
                signal_ptrs=slab.peer_ptrs,
                staging0_ptrs=tuple(
                    ptr + layout.staging0_offset for ptr in slab.peer_ptrs
                ),
                staging1_ptrs=tuple(
                    ptr + layout.staging1_offset for ptr in slab.peer_ptrs
                ),
                max_rows=max_rows,
                topk=topk,
                exchange_group=exchange_group,
                ipc=ipc,
                owned_buffers=owned,
                ext_module=ext_module,
                stream_affine=stream_affine,
            )
        except Exception:
            _release_failed_allocations(owned, ipc)
            raise

    @classmethod
    def from_process_group(
        cls,
        *,
        process_group: ProcessGroup,
        device: torch.device | int | str,
        max_rows: int,
        topk: int,
        ext_module=None,
        stream_affine: bool = True,
    ) -> "PCIeDCPTopKOwnerExchange":
        return cls.from_exchange_group(
            exchange_group=process_group,
            device=device,
            max_rows=max_rows,
            topk=topk,
            ext_module=ext_module,
            stream_affine=stream_affine,
        )

    def stage_candidates(
        self,
        local_indices: torch.Tensor,
        local_scores: torch.Tensor,
        *,
        plan: Plan,
        threads: int | None = None,
        block_limit: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Stage candidates through the session-prepared native launcher."""
        state = require_prepared(plan, "comm.pcie", self.device)
        state.require_runtime(self)
        call = state.query.call
        prepared_threads = int(call["threads"])
        prepared_limit = int(call["block_limit"])
        if int(local_indices.shape[0]) != int(call["rows"]):
            raise ValueError("rows differs from the prepared DCP top-k plan")
        if threads is not None and int(threads) != prepared_threads:
            raise ValueError("threads differs from the prepared DCP top-k plan")
        if block_limit is not None and int(block_limit) != prepared_limit:
            raise ValueError("block_limit differs from the prepared DCP top-k plan")
        with _device_guard(self.device):
            return self._stage_candidates_on_device(
                local_indices,
                local_scores,
                launcher=state.launcher(),
                threads=prepared_threads,
                block_limit=prepared_limit,
            )

    @contextmanager
    def capture(self, *, plan: Plan):
        """Serialize capture for an already-prepared owner exchange."""
        state = require_prepared(plan, "comm.pcie", self.device)
        state.require_runtime(self)
        if self._capture_context_depth:
            raise RuntimeError("overlapping DCP top-k capture contexts are not allowed")
        self._capture_context_depth = 1
        try:
            yield self
        finally:
            self._capture_context_depth = 0

    def _stage_candidates_on_device(
        self,
        local_indices: torch.Tensor,
        local_scores: torch.Tensor,
        *,
        launcher,
        threads: int,
        block_limit: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._closed:
            raise RuntimeError("PCIeDCPTopKOwnerExchange is closed")
        if local_indices.device != self.device or local_scores.device != self.device:
            raise ValueError("inputs must be on the runtime device")
        if local_indices.dtype != torch.int32:
            raise ValueError("local_indices must be int32")
        if local_scores.dtype != torch.float32:
            raise ValueError("local_scores must be float32")
        if local_indices.shape != local_scores.shape or local_indices.ndim != 2:
            raise ValueError("inputs must have matching [rows, topk] shapes")
        rows, topk = (int(value) for value in local_indices.shape)
        if rows <= 0 or rows > self.max_rows or rows % self.world_size != 0:
            raise ValueError(
                f"rows must be in (0, {self.max_rows}] and divisible by world_size"
            )
        if topk != self.topk:
            raise ValueError(f"topk {topk} does not match configured topk {self.topk}")
        if not local_indices.is_contiguous() or not local_scores.is_contiguous():
            raise ValueError("inputs must be contiguous")
        _validate_launch_config(
            threads=int(threads),
            block_limit=int(block_limit),
            world_size=self.world_size,
        )
        self._bind_stream()
        owner_packs = (rows // self.world_size) * (self.topk // 4)
        blocks = max(1, min(int(block_limit), (owner_packs + threads - 1) // threads))
        capturing = _is_current_stream_capturing(self.device)
        if capturing:
            if self._capture_context_depth <= 0:
                raise RuntimeError(
                    "DCP top-k CUDA graph capture requires an active "
                    "owner.capture() context"
                )
        if capturing and self._graph_slot is None:
            self._graph_slot = self._next_slot
        if self._graph_slot is not None:
            slot = self._graph_slot
            wait_for_prior_consumer = True
        else:
            slot = self._next_slot
            self._next_slot ^= 1
            wait_for_prior_consumer = False
        self._launch_stage(
            launcher,
            local_indices,
            local_scores,
            slot=slot,
            rows=rows,
            threads=int(threads),
            blocks=blocks,
            wait_for_prior_consumer=wait_for_prior_consumer,
        )
        candidate_views = self._candidate_views[slot] if self._candidate_views else ()
        if not candidate_views:
            raise RuntimeError("DCP top-k candidate views require a CUDA device")
        owner_rows = rows // self.world_size
        candidate_indices = candidate_views[0][:owner_rows]
        candidate_scores = candidate_views[1][:owner_rows]
        expected = (rows // self.world_size, self.world_size * self.topk)
        _validate_candidate_view(
            candidate_indices,
            expected=expected,
            dtype=torch.int32,
            device=self.device,
            label="index",
        )
        _validate_candidate_view(
            candidate_scores,
            expected=expected,
            dtype=torch.float32,
            device=self.device,
            label="score",
        )
        return candidate_indices, candidate_scores

    def _launch_stage(
        self,
        launcher,
        local_indices: torch.Tensor,
        local_scores: torch.Tensor,
        *,
        slot: int,
        rows: int,
        threads: int,
        blocks: int,
        wait_for_prior_consumer: bool,
    ) -> None:
        with torch.cuda.device(self.device):
            launcher(
                local_indices.data_ptr(),
                local_scores.data_ptr(),
                self._staging_ptrs[slot],
                self._signal_ptrs,
                rows,
                self._candidate_plane_elems,
                blocks,
                wait_for_prior_consumer,
            )


def _validate_candidate_view(
    tensor: torch.Tensor,
    *,
    expected: tuple[int, int],
    dtype: torch.dtype,
    device: torch.device,
    label: str,
) -> None:
    if (
        tuple(tensor.shape) != expected
        or tensor.dtype != dtype
        or tensor.device != device
        or not tensor.is_contiguous()
    ):
        raise RuntimeError(f"channel returned an invalid candidate {label} view")


def _release_failed_allocations(
    owned: Sequence[_OwnedSharedBuffer],
    ipc: CudaRTLibrary,
) -> None:
    for shared in owned:
        for ptr in shared.remote_ptrs:
            with suppress(Exception):
                ipc.cudaIpcCloseMemHandle(ptr)
        with suppress(Exception):
            ipc.cudaFree(shared.local_ptr)
