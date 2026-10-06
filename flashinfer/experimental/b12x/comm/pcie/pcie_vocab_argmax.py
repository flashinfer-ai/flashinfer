"""Bounded-degree vocabulary-parallel argmax runtime."""

from __future__ import annotations

import os
from contextlib import contextmanager, suppress
from typing import Iterator, Optional
from warnings import warn

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from b12x.preparation.types import Plan, require_prepared

from ._cuda_ipc import CudaRTLibrary
from ._vocab_argmax_cute import SLAB_BYTES
from .pcie_oneshot import (
    PCIeOneshotAllReduce,
    _broadcast_gather_object,
    _run_collective_preallocation_setup,
)


ISLAND_SIZE = 4
MAX_BATCH_SIZE = 8
SUPPORTED_WORLD_SIZES = (8, 12, 16)


def _exchange_ipc_handles(
    local_handle: object,
    group: ProcessGroup,
) -> list[object]:
    """Exchange CUDA IPC handles over NCCL or a metadata-only Gloo group."""

    backend = str(dist.get_backend(group=group)).lower()
    if "nccl" in backend:
        return _broadcast_gather_object(local_handle, group)
    if "gloo" not in backend:
        raise ValueError(
            "vocabulary argmax IPC exchange requires an NCCL or Gloo group, "
            f"got {backend}"
        )

    world_size = dist.get_world_size(group=group)
    handles: list[object | None] = [None] * world_size
    dist.all_gather_object(handles, local_handle, group=group)
    if any(handle is None for handle in handles):
        raise RuntimeError("vocabulary argmax IPC exchange returned an empty handle")
    return list(handles)


def _require_uniform_geometry(
    local_vocab_size: int,
    max_batch_size: int,
    group: ProcessGroup,
) -> None:
    """Reject rank-local geometry before allocating collective resources."""
    geometry = (int(local_vocab_size), int(max_batch_size))
    # vLLM supplies its metadata-only Gloo TP group here so initialization does
    # not allocate an NCCL object-collective workspace.  Geometry validation
    # must preserve the same NCCL-or-Gloo contract as CUDA IPC handle exchange.
    peer_geometry = _exchange_ipc_handles(geometry, group)
    if any(candidate != geometry for candidate in peer_geometry):
        raise RuntimeError(
            f"vocabulary argmax geometry differs across ranks: {peer_geometry}"
        )


def _wait_nanosleep_cycles_from_env() -> int:
    raw = os.getenv("B12X_PCIE_VOCAB_ARGMAX_NANOSLEEP_CYCLES", "24")
    try:
        cycles = int(raw)
    except ValueError as exc:
        raise ValueError(
            "B12X_PCIE_VOCAB_ARGMAX_NANOSLEEP_CYCLES must be an integer"
        ) from exc
    if not 0 <= cycles <= 1024:
        raise ValueError("B12X_PCIE_VOCAB_ARGMAX_NANOSLEEP_CYCLES must be in [0, 1024]")
    return cycles


def _selected_peers(rank: int, world_size: int = 16) -> tuple[int, ...]:
    """Return three island peers plus the same lane in other islands."""

    if world_size not in SUPPORTED_WORLD_SIZES:
        raise ValueError(
            "vocabulary argmax requires "
            f"{SUPPORTED_WORLD_SIZES}, got TP{world_size}"
        )
    if not 0 <= rank < world_size:
        raise ValueError(f"invalid rank {rank} for TP{world_size}")
    island = rank // ISLAND_SIZE
    lane = rank % ISLAND_SIZE
    local = set(range(island * ISLAND_SIZE, (island + 1) * ISLAND_SIZE))
    cross_island = set(range(lane, world_size, ISLAND_SIZE))
    return tuple(sorted((local | cross_island) - {rank}))


class PCIeVocabParallelArgmax:
    """Fuse BF16 add and exact global greedy sampling across TP shards.

    Every TP rank must invoke ``fused_add_argmax`` once per logical step with
    the same batch size. Tensor-parallel schedulers satisfy this collective
    ordering invariant; callers that cannot guarantee it must not use this
    runtime. Construction verifies that local vocabulary and capacity geometry
    match on all ranks.
    """

    def __init__(
        self,
        *,
        exchange_group: ProcessGroup,
        device: torch.device | int | str,
        local_vocab_size: int,
        max_batch_size: int = MAX_BATCH_SIZE,
    ) -> None:
        self.group = exchange_group
        self.rank = dist.get_rank(group=exchange_group)
        self.world_size = dist.get_world_size(group=exchange_group)

        def normalize_and_validate():
            device_obj = (
                device
                if isinstance(device, torch.device)
                else torch.device(
                    f"cuda:{device}" if isinstance(device, int) else device
                )
            )
            if self.world_size not in SUPPORTED_WORLD_SIZES:
                raise ValueError(
                    "vocabulary argmax requires "
                    f"{SUPPORTED_WORLD_SIZES}, got TP{self.world_size}"
                )
            if device_obj.type != "cuda":
                raise ValueError("vocabulary argmax requires a CUDA device")
            if device_obj.index is None:
                device_obj = torch.device("cuda", torch.cuda.current_device())
            normalized_local_vocab_size = int(local_vocab_size)
            normalized_max_batch_size = int(max_batch_size)
            if (
                normalized_local_vocab_size <= 0
                or normalized_local_vocab_size * self.world_size >= 1 << 31
            ):
                raise ValueError("global vocabulary must fit a positive int32 index")
            if not 0 < normalized_max_batch_size <= MAX_BATCH_SIZE:
                raise ValueError(
                    f"max_batch_size must be in [1, {MAX_BATCH_SIZE}]"
                )
            return (
                device_obj,
                normalized_local_vocab_size,
                normalized_max_batch_size,
                _wait_nanosleep_cycles_from_env(),
            )

        (
            self.device,
            self.local_vocab_size,
            self.max_batch_size,
            wait_nanosleep_cycles,
        ) = _run_collective_preallocation_setup(
            owner="PCIe vocabulary argmax argument validation",
            exchange_group=exchange_group,
            setup=normalize_and_validate,
        )
        _require_uniform_geometry(
            self.local_vocab_size,
            self.max_batch_size,
            exchange_group,
        )
        self._slab_ptrs: tuple[int, ...] = ()
        self._local_ptr = 0
        self._remote_ptrs: list[int] = []
        self._closed = True

        def prepare_runtime():
            return CudaRTLibrary()

        self._ipc = _run_collective_preallocation_setup(
            owner="PCIe vocabulary argmax runtime preparation",
            exchange_group=exchange_group,
            setup=prepare_runtime,
        )
        self.wait_nanosleep_cycles = wait_nanosleep_cycles
        with self._cuda_runtime_device():
            shared = PCIeOneshotAllReduce._allocate_shared_buffer(
                exchange_group,
                SLAB_BYTES,
                zero_fill=True,
                ipc=self._ipc,
                peer_ranks=_selected_peers(self.rank, self.world_size),
            )
        self._local_ptr = shared.local_ptr
        self._slab_ptrs = shared.peer_ptrs
        self._remote_ptrs = list(shared.remote_ptrs)
        self._closed = False

    @classmethod
    def from_exchange_group(
        cls,
        *,
        exchange_group: ProcessGroup,
        device: torch.device | int | str,
        local_vocab_size: int,
        max_batch_size: int = MAX_BATCH_SIZE,
    ) -> "PCIeVocabParallelArgmax":
        return cls(
            exchange_group=exchange_group,
            device=device,
            local_vocab_size=local_vocab_size,
            max_batch_size=max_batch_size,
        )

    @classmethod
    def from_process_group(
        cls,
        *,
        process_group: ProcessGroup,
        device: torch.device | int | str,
        local_vocab_size: int,
        max_batch_size: int = MAX_BATCH_SIZE,
    ) -> "PCIeVocabParallelArgmax":
        return cls.from_exchange_group(
            exchange_group=process_group,
            device=device,
            local_vocab_size=local_vocab_size,
            max_batch_size=max_batch_size,
        )

    @property
    def mapped_peer_count(self) -> int:
        return len(self._remote_ptrs)

    @contextmanager
    def _cuda_runtime_device(self) -> Iterator[None]:
        """Select this runtime's CUDA device and restore the caller's device."""
        if self.device.type != "cuda" or self.device.index is None:
            yield
            return
        previous_device: int | None
        try:
            previous_device = torch.cuda.current_device()
        except Exception:
            previous_device = None
        self._ipc.cudaSetDevice(self.device.index)
        try:
            yield
        finally:
            if previous_device is not None and previous_device != self.device.index:
                self._ipc.cudaSetDevice(previous_device)

    def fused_add_argmax(
        self,
        base: torch.Tensor,
        bias: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        *,
        plan: Plan,
    ) -> torch.Tensor:
        """Return exact global argmax of the BF16-rounded local sum."""

        state = require_prepared(plan, "comm.pcie", self.device)
        state.require_runtime(self)
        if self._closed:
            raise RuntimeError("vocabulary argmax runtime is closed")
        if base.device != self.device or bias.device != self.device:
            raise ValueError("inputs must be on the runtime device")
        if base.dtype != torch.bfloat16 or bias.dtype != torch.bfloat16:
            raise ValueError("inputs must be BF16")
        if base.ndim != 2 or base.shape != bias.shape:
            raise ValueError("inputs must have matching [batch, local_vocab] shapes")
        batch, local_vocab = (int(value) for value in base.shape)
        if batch != int(state.query.call["batch"]):
            raise ValueError("batch differs from the prepared vocabulary plan")
        if not 0 < batch <= self.max_batch_size:
            raise ValueError(
                f"batch size {batch} exceeds capacity {self.max_batch_size}"
            )
        if local_vocab != self.local_vocab_size:
            raise ValueError(
                f"local vocabulary must be {self.local_vocab_size}, got {local_vocab}"
            )
        if base.stride(1) != 1 or bias.stride(1) != 1:
            raise ValueError("input last dimensions must be contiguous")
        if base.stride(0) <= 0 or bias.stride(0) <= 0:
            raise ValueError("input row strides must be positive")
        if out is None:
            out = torch.empty(batch, dtype=torch.int64, device=self.device)
        if (
            out.device != self.device
            or out.dtype != torch.int64
            or out.shape != (batch,)
            or not out.is_contiguous()
        ):
            raise ValueError("output must be contiguous int64 [batch] on the device")
        return self._run_prepared(
            state.launcher(),
            base,
            bias,
            out,
        )
        return out

    def _run_prepared(self, launcher, base, bias, out):
        """Invoke an already-prepared native launcher after caller validation."""
        with self._cuda_runtime_device():
            launcher(
                self._slab_ptrs,
                base.data_ptr(),
                bias.data_ptr(),
                out.data_ptr(),
                self.local_vocab_size,
                base.stride(0),
                bias.stride(0),
                int(base.shape[0]),
            )
        return out

    def for_stream(self, stream: object = None) -> "PCIeVocabParallelArgmax":
        del stream
        return self

    @contextmanager
    def capture(self, stream: object = None):
        del stream
        yield self

    def register_graph_buffers(self) -> None:
        return None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        with self._cuda_runtime_device():
            torch.cuda.synchronize(self.device)
            dist.barrier(group=self.group)
            try:
                self._slab_ptrs = ()
                for ptr in self._remote_ptrs:
                    self._ipc.cudaIpcCloseMemHandle(ptr)
            finally:
                self._remote_ptrs.clear()
                dist.barrier(group=self.group)
                try:
                    if self._local_ptr:
                        self._ipc.cudaFree(self._local_ptr)
                finally:
                    self._local_ptr = 0
                    dist.barrier(group=self.group)

    def __enter__(self) -> "PCIeVocabParallelArgmax":
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    def __del__(self) -> None:
        if getattr(self, "_closed", True):
            return

        # Python garbage collection is not ordered across distributed ranks.
        # Calling close() here could deadlock on its collective barriers, so
        # finalization releases only resources owned by this rank. Callers
        # must use close() or the context manager for coordinated teardown.
        self._closed = True
        with suppress(Exception):
            warn(
                "PCIeVocabParallelArgmax was garbage-collected without close(); "
                "releasing rank-local CUDA resources without collective teardown",
                ResourceWarning,
                stacklevel=2,
            )

        with suppress(Exception), self._cuda_runtime_device():
            self._slab_ptrs = ()

            remote_ptrs = list(getattr(self, "_remote_ptrs", ()))
            self._remote_ptrs = []
            for ptr in remote_ptrs:
                with suppress(Exception):
                    self._ipc.cudaIpcCloseMemHandle(ptr)

            local_ptr = getattr(self, "_local_ptr", 0)
            self._local_ptr = 0
            if local_ptr:
                with suppress(Exception):
                    self._ipc.cudaFree(local_ptr)


__all__ = ["PCIeVocabParallelArgmax"]
