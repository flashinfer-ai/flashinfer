"""PCIe two-shot sequence-parallel collectives with fp8 transport.

Push-based reduce_scatter / all_gather over IPC-mapped peer slabs for
TP sequence parallelism: values are quantized exactly once at the
source (per-token e4m3 scales), moved as fp8, and dequantized fused
with the fp32 reduction (RS) or the bf16 store (AG). Staging goes
through alternating eager slots so both ops are CUDA-graph-capturable;
a runtime instance must not be shared concurrently across CUDA streams.
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

SUPPORTED_WORLD_SIZES = (2, 4, 8)
FP8_MAX = 448.0
TWOSHOT_REQUIRED_SMS = 64
_MAX_BLOCKS = 64
_MAX_RANKS = 16
_FLAG_STRIDE = 32
_SIGNAL_BYTES = (
    _MAX_BLOCKS * _MAX_RANKS * 4
    + 2 * _MAX_BLOCKS * (_MAX_RANKS * _FLAG_STRIDE) * 4
)


@dataclass(frozen=True)
class _TwoShotLayout:
    signal_bytes: int
    pack_stride: int
    scale_offset: int
    scale_stride: int
    slot_bytes: int
    slab_bytes: int


def _make_layout(max_rows: int, row_elems: int, world_size: int) -> _TwoShotLayout:
    """Return the exact native signal/payload/scale double-slot layout."""

    if world_size not in SUPPORTED_WORLD_SIZES:
        raise ValueError(f"unsupported world size {world_size}")
    if max_rows <= 0 or max_rows % world_size != 0:
        raise ValueError("max_rows must be positive and divisible by world size")
    if row_elems <= 0 or row_elems % 16 != 0:
        raise ValueError("row_elems must be a positive multiple of 16")
    max_rows_per_rank = max_rows // world_size
    packs_per_row = row_elems // 16
    pack_stride = _align_up(max_rows_per_rank * packs_per_row, 16)
    payload_bytes = world_size * pack_stride * 16
    scale_offset = _align_up(payload_bytes, IPC_SLAB_ALIGNMENT)
    scale_stride = _align_up(max_rows_per_rank, 64)
    slot_bytes = _align_up(
        scale_offset + world_size * scale_stride * 4, IPC_SLAB_ALIGNMENT
    )
    signal_bytes = _align_up(_SIGNAL_BYTES, IPC_SLAB_ALIGNMENT)
    return _TwoShotLayout(
        signal_bytes=signal_bytes,
        pack_stride=pack_stride,
        scale_offset=scale_offset,
        scale_stride=scale_stride,
        slot_bytes=slot_bytes,
        slab_bytes=signal_bytes + 2 * slot_bytes,
    )


def quantize_per_row(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference per-row e4m3 quantization (tests / non-fused callers)."""
    assert x.dim() == 2
    amax = x.abs().amax(dim=-1, keepdim=True).float().clamp_(min=1e-12)
    scale = amax / FP8_MAX
    payload = (x.float() / scale).clamp_(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return payload, scale.squeeze(-1).contiguous()


def _pad_scalar_peer_ptrs(
    pointers: Sequence[int], *, rank: int, world_size: int
) -> tuple[int, ...]:
    """Pad a live peer set to the fixed eight-scalar kernel ABI."""

    live = tuple(int(pointer) for pointer in pointers)
    if len(live) != world_size or rank < 0 or rank >= world_size:
        raise ValueError("invalid two-shot scalar peer pointer set")
    return live + (live[rank],) * (8 - world_size)


class PCIeTwoShotSP:
    """Two-shot fp8-transport reduce_scatter / all_gather runtime.

    Graphs captured from one instance share a device epoch and must replay
    serially. Use a separate instance for independently replayable graphs.
    """

    def __init__(self, *args, **kwargs) -> None:
        raise RuntimeError("use PCIeTwoShotSP.from_exchange_group()")

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
        scale_offset: int,
        scale_stride: int,
    ) -> "PCIeTwoShotSP":
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
        self.exchange_group = exchange_group
        self.max_rows = max_rows
        self.row_elems = row_elems
        self._pack_stride = int(pack_stride)
        self._scale_offset = int(scale_offset)
        self._scale_stride = int(scale_stride)
        self._slot_bytes = _align_up(
            self._scale_offset + self.world_size * self._scale_stride * 4,
            IPC_SLAB_ALIGNMENT,
        )
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
        ext_module=None,
    ) -> "PCIeTwoShotSP":
        del ext_module  # native-extension injection is obsolete
        rank = dist.get_rank(group=exchange_group)
        world_size = dist.get_world_size(group=exchange_group)

        def validate_factory_arguments():
            device_obj = _normalize_device(device)
            normalized_max_rows = int(max_rows)
            normalized_row_elems = int(row_elems)
            if world_size not in SUPPORTED_WORLD_SIZES:
                raise ValueError(f"unsupported world size {world_size}")
            if device_obj.type != "cuda":
                raise ValueError("PCIe twoshot requires a CUDA device")
            if normalized_max_rows <= 0:
                raise ValueError("max_rows must be positive")
            if normalized_row_elems <= 0 or normalized_row_elems % 16 != 0:
                raise ValueError("row_elems must be a positive multiple of 16")
            if normalized_max_rows % world_size != 0:
                raise ValueError("max_rows must be divisible by world size")
            return device_obj, normalized_max_rows, normalized_row_elems

        device_obj, max_rows, row_elems = _run_collective_preallocation_setup(
            owner="PCIe twoshot argument validation",
            exchange_group=exchange_group,
            setup=validate_factory_arguments,
        )

        _require_full_grid_residency(
            owner="PCIe twoshot",
            required_sms=TWOSHOT_REQUIRED_SMS,
            device=device_obj,
            exchange_group=exchange_group,
        )

        def prepare():
            prepared_ipc = CudaRTLibrary()
            prepared_ipc.cudaSetDevice(_cuda_device_index(device_obj))
            return prepared_ipc, _make_layout(max_rows, row_elems, world_size)

        ipc, layout = _run_collective_preallocation_setup(
            owner="PCIe twoshot",
            exchange_group=exchange_group,
            setup=prepare,
        )
        _require_collective_contract(
            owner="PCIe twoshot channel layout",
            exchange_group=exchange_group,
            contract=(
                int(max_rows),
                int(row_elems),
                layout,
            ),
        )

        shared = PCIeOneshotAllReduce._allocate_shared_buffer(
            exchange_group,
            layout.slab_bytes,
            zero_fill=True,
            ipc=ipc,
        )
        peer_ptrs = list(shared.peer_ptrs)
        signal_ptrs = _pad_scalar_peer_ptrs(
            peer_ptrs, rank=rank, world_size=world_size
        )
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

        runtime: Optional[PCIeTwoShotSP] = None
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
                scale_offset=layout.scale_offset,
                scale_stride=layout.scale_stride,
            )
        except Exception as exc:
            init_error = exc

        def detach_shared_ownership() -> None:
            if runtime is not None:
                runtime._owned_buffers.clear()

        _finish_collective_runtime_setup(
            owner="PCIe twoshot",
            exchange_group=exchange_group,
            ipc=ipc,
            shared=shared,
            local_error=init_error,
            detach_shared_ownership=detach_shared_ownership,
        )
        assert runtime is not None
        return runtime

    def _check(self, payload: torch.Tensor, scale: torch.Tensor, rows: int) -> None:
        if self._closed:
            raise RuntimeError("PCIeTwoShotSP is closed")
        if payload.shape != (rows, self.row_elems):
            raise ValueError(
                f"payload shape {tuple(payload.shape)} != ({rows}, {self.row_elems})"
            )
        if scale.numel() != rows:
            raise ValueError(f"scale numel {scale.numel()} != {rows}")
        if rows > self.max_rows:
            raise ValueError("pcie_twoshot staging capacity exceeded")
        if payload.device != self.device or scale.device != self.device:
            raise ValueError("payload and scale must be on the runtime CUDA device")
        if payload.dtype not in (torch.float8_e4m3fn, torch.uint8):
            raise TypeError("payload must contain E4M3 bytes")
        if scale.dtype != torch.float32:
            raise TypeError("scale must be float32")
        if not payload.is_contiguous() or not scale.is_contiguous():
            raise ValueError("payload and scale must be contiguous")

    def _device_index(self) -> int:
        return (
            self.device.index
            if self.device.index is not None
            else torch.cuda.current_device()
        )

    @contextmanager
    def capture(self, *, plan: Plan):
        """Serialize capture; launchers are already resolved by preparation."""
        state = require_prepared(plan, "comm.pcie", self.device)
        state.require_runtime(self)
        if self._capture_context_depth:
            raise RuntimeError("overlapping PCIe twoshot capture contexts are not allowed")
        self._capture_context_depth = 1
        try:
            yield self
        finally:
            self._capture_context_depth = 0

    def _launch_prepared(
        self, payload: torch.Tensor, scale: torch.Tensor, out: torch.Tensor, *,
        state, threads: int, block_limit: int,
    ) -> None:
        operation = state.query.call["operation"]
        rows_per_rank = (
            payload.shape[0] // self.world_size
            if operation == "reduce_scatter" else payload.shape[0]
        )
        if int(threads) != state.query.call["threads"] or int(block_limit) != state.query.call["block_limit"]:
            raise ValueError("two-shot launch controls differ from the prepared plan")
        shard_packs = rows_per_rank * (self.row_elems // 16)
        if shard_packs > self._pack_stride or rows_per_rank > self._scale_stride:
            raise ValueError("pcie_twoshot staging capacity exceeded")
        blocks = max(1, min(int(block_limit), (shard_packs + threads - 1) // threads))
        capturing = _is_current_stream_capturing(self.device)
        if capturing and self._capture_context_depth <= 0:
            raise RuntimeError("PCIe twoshot CUDA graph capture requires runtime.capture(plan=...)")
        if capturing and not self._device_slot_selection:
            self._device_slot_bias = self._slot & 1
            self._device_slot_selection = True
        if self._device_slot_selection:
            slot = 0
        else:
            slot = self._slot % 2
            self._slot += 1
        launcher = state.launcher(self._device_slot_selection, self._device_slot_bias)
        launcher(
            payload.data_ptr(), scale.data_ptr(), self._staging_ptrs[slot],
            self._signal_ptrs, out.data_ptr(), self.rank, self._pack_stride,
            self._scale_offset, self._scale_stride, self._slot_bytes,
            rows_per_rank, self.row_elems, blocks,
        )

    def reduce_scatter_fp8(
        self,
        payload: torch.Tensor,
        scale: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        *,
        plan: Plan,
        threads: int = 512,
        block_limit: int = 64,
    ) -> torch.Tensor:
        """Sum per-token-quantized partials with a prepared native launcher."""
        state = require_prepared(plan, "comm.pcie", self.device)
        state.require_runtime(self)
        if state.query.surface != "TwoShotReduceScatter.reduce_scatter_fp8":
            raise ValueError("plan does not prepare FP8 reduce-scatter")
        with _device_guard(self.device):
            return self._reduce_scatter_fp8_on_device(
                payload, scale, out, state=state, threads=threads, block_limit=block_limit,
            )

    def _reduce_scatter_fp8_on_device(
        self,
        payload: torch.Tensor,
        scale: torch.Tensor,
        out: Optional[torch.Tensor],
        *,
        state,
        threads: int,
        block_limit: int,
    ) -> torch.Tensor:
        rows = payload.shape[0]
        self._check(payload, scale, rows)
        if rows % self.world_size != 0:
            raise ValueError("rows must be divisible by world size")
        if out is None:
            out = torch.empty(
                rows // self.world_size,
                self.row_elems,
                dtype=torch.bfloat16,
                device=self.device,
            )
        if (
            out.shape != (rows // self.world_size, self.row_elems)
            or out.dtype != torch.bfloat16
            or out.device != self.device
            or not out.is_contiguous()
        ):
            raise ValueError("output must be contiguous BF16 with the local shard shape")
        self._launch_prepared(
            payload, scale, out, state=state, threads=threads, block_limit=block_limit,
        )
        return out

    def all_gather_fp8(
        self,
        payload: torch.Tensor,
        scale: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        *,
        plan: Plan,
        threads: int = 512,
        block_limit: int = 64,
    ) -> torch.Tensor:
        """Gather per-token-quantized shards with a prepared native launcher."""
        state = require_prepared(plan, "comm.pcie", self.device)
        state.require_runtime(self)
        if state.query.surface != "TwoShotReduceScatter.all_gather_fp8":
            raise ValueError("plan does not prepare FP8 all-gather")
        with _device_guard(self.device):
            return self._all_gather_fp8_on_device(
                payload, scale, out, state=state, threads=threads, block_limit=block_limit,
            )

    def _all_gather_fp8_on_device(
        self,
        payload: torch.Tensor,
        scale: torch.Tensor,
        out: Optional[torch.Tensor],
        *,
        state,
        threads: int,
        block_limit: int,
    ) -> torch.Tensor:
        rows = payload.shape[0]
        self._check(payload, scale, rows)
        if out is None:
            out = torch.empty(
                rows * self.world_size,
                self.row_elems,
                dtype=torch.bfloat16,
                device=self.device,
            )
        if (
            out.shape != (rows * self.world_size, self.row_elems)
            or out.dtype != torch.bfloat16
            or out.device != self.device
            or not out.is_contiguous()
        ):
            raise ValueError("output must be contiguous BF16 with the gathered shape")
        self._launch_prepared(
            payload, scale, out, state=state, threads=threads, block_limit=block_limit,
        )
        return out

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
            _raise_local_cleanup_errors("PCIe twoshot", "IPC import close", failures)

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
            _raise_local_cleanup_errors("PCIe twoshot", "IPC export free", failures)

    def close(self) -> None:
        if getattr(self, "_coordinated_close_complete", False):
            return
        _coordinated_close_channels(
            (self,),
            exchange_group=self.exchange_group,
            device=self.device,
        )

    def __enter__(self) -> "PCIeTwoShotSP":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def __del__(
        self,
        _quarantine: dict[int, object] = _ABANDONED_PCIE_RUNTIME_QUARANTINE,
    ) -> None:
        # GC cannot prove queued CUDA work is complete.  Explicit close() is the
        # only path allowed to synchronize, unmap imports, and free exports.
        if getattr(self, "_coordinated_close_complete", False):
            return
        if getattr(self, "_owned_buffers", ()):
            _quarantine[id(self)] = self
