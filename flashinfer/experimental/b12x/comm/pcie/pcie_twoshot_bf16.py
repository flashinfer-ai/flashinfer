"""Single-rounding BF16 PCIe two-shot collective runtime.

Host-side twin of :mod:`pcie_twoshot` without the fp8 wire codec: payloads
travel as bf16 packs, are accumulated in fp32 in a fixed rank order and
rounded once.  Intended for TP decode all-reduces above the one-shot
ceiling (tens of KB) and below the DMA ring floor (MB), where NCCL ring is
the incumbent.  Graph capture follows the two-shot contract: enter
``runtime.capture()`` around ``torch.cuda.graph``. All eager launches and graph
replays from one instance must be serialized. CUDA graph capture requires a
caller-owned preallocated output for every operation. Callers must stop
submitting work before closing that instance.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from b12x.preparation.types import Plan, require_prepared

from ._cuda_ipc import CudaRTLibrary
from .pcie_oneshot import (
    _ABANDONED_PCIE_RUNTIME_QUARANTINE,
    IPC_SLAB_ALIGNMENT,
    PCIeOneshotAllReduce,
    _finish_collective_runtime_setup,
    _raise_local_cleanup_errors,
    _align_up,
    _coordinated_close_channels,
    _cuda_device_index,
    _device_guard,
    _is_current_stream_capturing,
    _normalize_device,
    _OwnedSharedBuffer,
    _require_collective_contract,
    _require_full_grid_residency,
    _run_collective_preallocation_setup,
)
from .pcie_twoshot import (
    TWOSHOT_REQUIRED_SMS,
    _MAX_BLOCKS,
    _SIGNAL_BYTES,
    _pad_scalar_peer_ptrs,
)

_PACK_ELEMS = 8
SUPPORTED_WORLD_SIZES = (4,)


@dataclass(frozen=True)
class _TwoShotBf16Layout:
    signal_bytes: int
    pack_stride: int
    reduced_offset: int
    slot_bytes: int
    slab_bytes: int


def _make_layout(max_rows: int, row_elems: int, world_size: int) -> _TwoShotBf16Layout:
    if world_size not in SUPPORTED_WORLD_SIZES:
        raise ValueError(
            f"PCIeTwoShotBF16 supports only world size 4, got {world_size}"
        )
    if max_rows <= 0 or max_rows % world_size != 0:
        raise ValueError("max_rows must be positive and divisible by world size")
    if row_elems <= 0 or row_elems % _PACK_ELEMS != 0:
        raise ValueError("row_elems must be a positive multiple of 8")
    max_rows_per_rank = max_rows // world_size
    packs_per_row = row_elems // _PACK_ELEMS
    pack_stride = _align_up(max_rows_per_rank * packs_per_row, 16)
    payload_bytes = world_size * pack_stride * 16
    # The pull all-reduce keeps one reduced shard (pack_stride packs) per slot
    # after the full staged payload.
    reduced_offset = _align_up(payload_bytes, IPC_SLAB_ALIGNMENT)
    slot_bytes = _align_up(reduced_offset + pack_stride * 16, IPC_SLAB_ALIGNMENT)
    signal_bytes = _align_up(_SIGNAL_BYTES, IPC_SLAB_ALIGNMENT)
    return _TwoShotBf16Layout(
        signal_bytes=signal_bytes,
        pack_stride=pack_stride,
        reduced_offset=reduced_offset,
        slot_bytes=slot_bytes,
        slab_bytes=signal_bytes + 2 * slot_bytes,
    )


def _contiguous_storage_interval(tensor: torch.Tensor) -> tuple[int, int]:
    """Return the occupied byte interval of a validated contiguous tensor."""
    start = int(tensor.data_ptr())
    return start, start + int(tensor.numel()) * int(tensor.element_size())


def _require_disjoint(
    output: torch.Tensor,
    source: torch.Tensor,
    *,
    source_name: str,
) -> None:
    """Reject aliases that violate the non-coherent pull-kernel contract."""
    if output.device != source.device:
        return
    output_start, output_end = _contiguous_storage_interval(output)
    source_start, source_end = _contiguous_storage_interval(source)
    if max(output_start, source_start) < min(output_end, source_end):
        raise ValueError(f"output must not overlap {source_name}")


class PCIeTwoShotBF16:
    """Serialized single-rounding BF16 collective runtime for world size 4."""

    def __init__(self, *args, **kwargs) -> None:
        raise RuntimeError("use PCIeTwoShotBF16.from_exchange_group()")

    @classmethod
    def _from_prepared_factory(
        cls,
        *,
        rank: int,
        world_size: int,
        device: torch.device,
        signal_ptrs: Sequence[int],
        staging_ptrs: Sequence[Sequence[int]],
        owned_buffers: Sequence[_OwnedSharedBuffer],
        ipc: CudaRTLibrary,
        exchange_group: ProcessGroup,
        max_rows: int,
        row_elems: int,
        pack_stride: int,
        reduced_offset: int,
        slot_bytes: int,
    ) -> "PCIeTwoShotBF16":
        self = object.__new__(cls)
        self.rank = rank
        self.world_size = world_size
        self.device = _normalize_device(device)
        self.exchange_group = exchange_group
        self._signal_ptrs = tuple(int(pointer) for pointer in signal_ptrs)
        self._staging_ptrs = tuple(
            tuple(int(pointer) for pointer in slot) for slot in staging_ptrs
        )
        if len(self._signal_ptrs) != 8 or (
            len(self._staging_ptrs) != 2
            or any(len(slot) != 8 for slot in self._staging_ptrs)
        ):
            raise ValueError("two-shot scalar pointer ABI requires 8 peers and 2 slots")
        self._owned_buffers = list(owned_buffers)
        self._ipc = ipc
        self.max_rows = max_rows
        self.row_elems = row_elems
        self._pack_stride = int(pack_stride)
        self._reduced_offset = int(reduced_offset)
        self._slot_bytes = int(slot_bytes)
        self._slot = 0
        self._device_slot_selection = False
        self._device_slot_bias = 0
        self._capture_context_depth = 0
        self._closed = False
        self._ipc_imports_closed = False
        self._ipc_exports_freed = False
        self._coordinated_close_complete = False
        self._closed_ipc_import_indices: set[tuple[int, int]] = set()
        return self

    @classmethod
    def from_exchange_group(
        cls,
        *,
        exchange_group: ProcessGroup,
        device: torch.device | int | str,
        max_rows: int,
        row_elems: int,
    ) -> "PCIeTwoShotBF16":
        rank = dist.get_rank(group=exchange_group)
        world_size = dist.get_world_size(group=exchange_group)

        def validate_factory_arguments():
            device_obj = _normalize_device(device)
            normalized_max_rows = int(max_rows)
            normalized_row_elems = int(row_elems)
            if world_size not in SUPPORTED_WORLD_SIZES:
                raise ValueError(
                    f"PCIeTwoShotBF16 supports only world size 4, got {world_size}"
                )
            if device_obj.type != "cuda":
                raise ValueError("PCIe twoshot requires a CUDA device")
            if normalized_max_rows <= 0:
                raise ValueError("max_rows must be positive")
            if normalized_row_elems <= 0 or normalized_row_elems % _PACK_ELEMS != 0:
                raise ValueError("row_elems must be a positive multiple of 8")
            if normalized_max_rows % world_size != 0:
                raise ValueError("max_rows must be divisible by world size")
            return device_obj, normalized_max_rows, normalized_row_elems

        device_obj, max_rows, row_elems = _run_collective_preallocation_setup(
            owner="PCIe twoshot-bf16 argument validation",
            exchange_group=exchange_group,
            setup=validate_factory_arguments,
        )
        _require_full_grid_residency(
            owner="PCIe twoshot-bf16",
            required_sms=TWOSHOT_REQUIRED_SMS,
            device=device_obj,
            exchange_group=exchange_group,
        )

        def prepare():
            prepared_ipc = CudaRTLibrary()
            prepared_ipc.cudaSetDevice(_cuda_device_index(device_obj))
            return prepared_ipc, _make_layout(max_rows, row_elems, world_size)

        ipc, layout = _run_collective_preallocation_setup(
            owner="PCIe twoshot-bf16",
            exchange_group=exchange_group,
            setup=prepare,
        )
        _require_collective_contract(
            owner="PCIe twoshot-bf16 channel layout",
            exchange_group=exchange_group,
            contract=(int(max_rows), int(row_elems), layout),
        )
        shared = PCIeOneshotAllReduce._allocate_shared_buffer(
            exchange_group,
            layout.slab_bytes,
            zero_fill=True,
            ipc=ipc,
        )
        peer_ptrs = list(shared.peer_ptrs)
        signal_ptrs = _pad_scalar_peer_ptrs(peer_ptrs, rank=rank, world_size=world_size)
        staging_ptrs = (
            _pad_scalar_peer_ptrs(
                [p + layout.signal_bytes for p in peer_ptrs],
                rank=rank,
                world_size=world_size,
            ),
            _pad_scalar_peer_ptrs(
                [p + layout.signal_bytes + layout.slot_bytes for p in peer_ptrs],
                rank=rank,
                world_size=world_size,
            ),
        )
        runtime: Optional[PCIeTwoShotBF16] = None
        init_error: BaseException | None = None
        try:
            runtime = cls._from_prepared_factory(
                rank=rank,
                world_size=world_size,
                device=device_obj,
                signal_ptrs=signal_ptrs,
                staging_ptrs=staging_ptrs,
                owned_buffers=[shared],
                ipc=ipc,
                exchange_group=exchange_group,
                max_rows=max_rows,
                row_elems=row_elems,
                pack_stride=layout.pack_stride,
                reduced_offset=layout.reduced_offset,
                slot_bytes=layout.slot_bytes,
            )
        except Exception as exc:
            init_error = exc

        def detach_shared_ownership() -> None:
            if runtime is not None:
                runtime._owned_buffers.clear()

        _finish_collective_runtime_setup(
            owner="PCIe twoshot-bf16",
            exchange_group=exchange_group,
            ipc=ipc,
            shared=shared,
            local_error=init_error,
            detach_shared_ownership=detach_shared_ownership,
        )
        assert runtime is not None
        return runtime

    # ---- checks ---------------------------------------------------------

    def _check_tensor(
        self,
        tensor: torch.Tensor,
        *,
        shape: tuple[int, ...],
        name: str,
    ) -> None:
        if tensor.shape != shape:
            raise ValueError(f"{name} shape {tuple(tensor.shape)} != {shape}")
        if tensor.device != self.device:
            raise ValueError(f"{name} must be on the runtime CUDA device")
        if tensor.dtype != torch.bfloat16:
            raise TypeError(f"{name} must be bfloat16")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
        if tensor.data_ptr() % 16 != 0:
            raise ValueError(f"{name} must be 16-byte aligned")

    def _check(self, payload: torch.Tensor, rows: int) -> None:
        if self._closed:
            raise RuntimeError("PCIeTwoShotBF16 is closed")
        if rows > self.max_rows:
            raise ValueError("pcie_twoshot_bf16 staging capacity exceeded")
        self._check_tensor(
            payload,
            shape=(rows, self.row_elems),
            name="payload",
        )

    def _device_index(self) -> int:
        return (
            self.device.index
            if self.device.index is not None
            else torch.cuda.current_device()
        )

    def accepts(self, inp: torch.Tensor) -> bool:
        """True when ``all_reduce`` can serve this tensor."""
        if (
            self._closed
            or inp.dtype != torch.bfloat16
            or not inp.is_contiguous()
            or inp.data_ptr() % 16 != 0
        ):
            return False
        if inp.device != self.device:
            return False
        numel = inp.numel()
        if numel == 0 or numel % (self.row_elems * self.world_size) != 0:
            return False
        return numel // self.row_elems <= self.max_rows

    # ---- graph plumbing ---------------------------------------------------

    @contextmanager
    def capture(self, *, plan: Plan):
        state = require_prepared(plan, "comm.pcie", self.device)
        state.require_runtime(self)
        if self._capture_context_depth:
            raise RuntimeError("overlapping PCIe twoshot-bf16 capture contexts are not allowed")
        self._capture_context_depth = 1
        try:
            yield self
        finally:
            self._capture_context_depth = 0

    def _launch_prepared(self, payload, scale, out, *, state, threads, block_limit):
        del scale
        operation = state.query.call["operation"]
        # All-reduce accepts any contiguous shape; transport rows have the
        # fixed runtime width, independently of the producer's tensor shape.
        rows = payload.numel() // self.row_elems
        rows_per_rank = (
            rows // self.world_size
            if operation in ("reduce_scatter", "all_reduce") else rows
        )
        if int(threads) != state.query.call["threads"] or int(block_limit) != state.query.call["block_limit"]:
            raise ValueError("two-shot launch controls differ from the prepared plan")
        shard_packs = rows_per_rank * (self.row_elems // _PACK_ELEMS)
        if shard_packs > self._pack_stride:
            raise ValueError("pcie_twoshot_bf16 staging capacity exceeded")
        blocks = max(1, min(int(block_limit), (shard_packs + threads - 1) // threads))
        capturing = _is_current_stream_capturing(self.device)
        if capturing and self._capture_context_depth <= 0:
            raise RuntimeError("PCIe twoshot-bf16 capture requires runtime.capture(plan=...)")
        if capturing and not self._device_slot_selection:
            self._device_slot_bias = self._slot & 1
            self._device_slot_selection = True
        if self._device_slot_selection:
            slot = 0
        else:
            slot = self._slot % 2
            self._slot += 1
        launcher = state.launcher(self._device_slot_selection, self._device_slot_bias)
        if operation == "all_reduce":
            launcher(payload.data_ptr(), self._staging_ptrs[slot], self._signal_ptrs,
                     out.data_ptr(), self.rank, self._reduced_offset, self._slot_bytes,
                     rows_per_rank, blocks)
        else:
            launcher(payload.data_ptr(), self._staging_ptrs[slot], self._signal_ptrs,
                     out.data_ptr(), self.rank, self._pack_stride, self._slot_bytes,
                     rows_per_rank, blocks)

    # ---- public collectives ---------------------------------------------

    def reduce_scatter(
        self, payload: torch.Tensor, out: Optional[torch.Tensor] = None, *,
        plan: Plan, threads: int = 512, block_limit: int = 64,
    ) -> torch.Tensor:
        state = require_prepared(plan, "comm.pcie", self.device)
        state.require_runtime(self)
        if state.query.surface != "PCIeTwoShotBF16.reduce_scatter":
            raise ValueError("plan does not prepare BF16 reduce-scatter")
        with _device_guard(self.device):
            rows = payload.shape[0]
            self._check(payload, rows)
            if rows % self.world_size:
                raise ValueError("rows must be divisible by world size")
            if out is None:
                if _is_current_stream_capturing(self.device):
                    raise RuntimeError("PCIeTwoShotBF16.reduce_scatter capture requires caller-owned output")
                out = torch.empty(rows // self.world_size, self.row_elems, dtype=torch.bfloat16, device=self.device)
            self._check_tensor(out, shape=(rows // self.world_size, self.row_elems), name="output")
            _require_disjoint(out, payload, source_name="payload")
            self._launch_prepared(payload, None, out, state=state, threads=threads, block_limit=block_limit)
            return out

    def all_gather(
        self, payload: torch.Tensor, out: Optional[torch.Tensor] = None, *,
        plan: Plan, threads: int = 512, block_limit: int = 64,
    ) -> torch.Tensor:
        state = require_prepared(plan, "comm.pcie", self.device)
        state.require_runtime(self)
        if state.query.surface != "PCIeTwoShotBF16.all_gather":
            raise ValueError("plan does not prepare BF16 all-gather")
        with _device_guard(self.device):
            rows = payload.shape[0]
            self._check(payload, rows)
            if out is None:
                if _is_current_stream_capturing(self.device):
                    raise RuntimeError("PCIeTwoShotBF16.all_gather capture requires caller-owned output")
                out = torch.empty(rows * self.world_size, self.row_elems, dtype=torch.bfloat16, device=self.device)
            self._check_tensor(out, shape=(rows * self.world_size, self.row_elems), name="output")
            _require_disjoint(out, payload, source_name="payload")
            self._launch_prepared(payload, None, out, state=state, threads=threads, block_limit=block_limit)
            return out

    def all_reduce(
        self, inp: torch.Tensor, out: Optional[torch.Tensor] = None, *,
        plan: Plan, threads: int = 512, block_limit: int = 64,
    ) -> torch.Tensor:
        """FP32-accumulating BF16 all-reduce with one BF16 rounding."""
        state = require_prepared(plan, "comm.pcie", self.device)
        state.require_runtime(self)
        if state.query.surface != "PCIeTwoShotBF16.all_reduce":
            raise ValueError("plan does not prepare BF16 all-reduce")
        if not self.accepts(inp):
            raise ValueError("input not accepted by PCIeTwoShotBF16.all_reduce")
        rows = inp.numel() // self.row_elems
        with _device_guard(self.device):
            if out is None:
                if _is_current_stream_capturing(self.device):
                    raise RuntimeError("PCIeTwoShotBF16.all_reduce capture requires caller-owned output")
                out = torch.empty_like(inp)
            self._check_tensor(out, shape=tuple(inp.shape), name="output")
            _require_disjoint(out, inp, source_name="input")
            self._launch_prepared(inp.view(rows, self.row_elems), None, out.view(rows, self.row_elems),
                                  state=state, threads=threads, block_limit=block_limit)
        return out

    # ---- teardown (mirrors pcie_twoshot) -----------------------------------

    def _closed_import_indices(self) -> set[tuple[int, int]]:
        closed = getattr(self, "_closed_ipc_import_indices", None)
        if closed is None:
            closed = set()
            self._closed_ipc_import_indices = closed
        return closed

    def _all_python_ipc_imports_closed(self, closed: set[tuple[int, int]]) -> bool:
        return all(
            (buffer_index, remote_index) in closed
            for buffer_index, shared in enumerate(self._owned_buffers)
            for remote_index, _ in enumerate(shared.remote_ptrs)
        )

    def _close_ipc_imports_strict(self) -> None:
        if self._ipc_imports_closed:
            return
        self._closed = True
        failures: list[tuple[str, Exception]] = []
        closed = self._closed_import_indices()
        for buffer_index, shared in enumerate(self._owned_buffers):
            for remote_index, ptr in enumerate(shared.remote_ptrs):
                key = (buffer_index, remote_index)
                if key in closed:
                    continue
                try:
                    self._ipc.cudaIpcCloseMemHandle(ptr)
                except Exception as exc:
                    failures.append((f"CUDA IPC import {ptr}", exc))
                else:
                    closed.add(key)
        if not failures and self._all_python_ipc_imports_closed(closed):
            self._ipc_imports_closed = True
        if failures:
            _raise_local_cleanup_errors(
                "PCIe twoshot-bf16", "IPC import close", failures
            )

    def _free_ipc_exports_strict(self) -> None:
        if self._ipc_exports_freed:
            return
        self._close_ipc_imports_strict()
        failures: list[tuple[str, Exception]] = []
        remaining = []
        for shared in self._owned_buffers:
            try:
                self._ipc.cudaFree(shared.local_ptr)
            except Exception as exc:
                remaining.append(shared)
                failures.append((f"CUDA IPC export {shared.local_ptr}", exc))
        self._owned_buffers = remaining
        if not remaining:
            self._ipc_exports_freed = True
        if failures:
            _raise_local_cleanup_errors(
                "PCIe twoshot-bf16", "IPC export free", failures
            )

    def close(self) -> None:
        """Synchronize submitted work and release channels on every rank.

        The caller must prevent any eager launch or graph replay from being
        submitted concurrently with or after this collective close.
        """
        if getattr(self, "_coordinated_close_complete", False):
            return
        _coordinated_close_channels(
            (self,),
            exchange_group=self.exchange_group,
            device=self.device,
        )

    def __enter__(self) -> "PCIeTwoShotBF16":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def __del__(
        self,
        _quarantine: dict[int, object] = _ABANDONED_PCIE_RUNTIME_QUARANTINE,
    ) -> None:
        if getattr(self, "_coordinated_close_complete", False):
            return
        if getattr(self, "_owned_buffers", ()):
            _quarantine[id(self)] = self


__all__ = ["PCIeTwoShotBF16"]
