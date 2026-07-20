# SPDX-FileCopyrightText: 2026 FlashInfer team
# SPDX-License-Identifier: Apache-2.0
# Ported from b12x b12x/distributed/pcie_oneshot.py @ 00695ee8 (2026-07-19) -- one-time curated port.
# Upstream b12x is a research sandbox; this tree is the canonical home.
"""PCIe oneshot allreduce runtime with optional crossover autotuning."""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Optional, Sequence

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup
from torch.utils.cpp_extension import load

from ._cuda_ipc import CudaRTLibrary


logger = logging.getLogger(__name__)

SUPPORTED_WORLD_SIZES = (2, 4, 6, 8, 10)
SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
DEFAULT_MAX_SIZE = 8 * 1024 * 1024
DEFAULT_RANK_DATA_BYTES = 8 * 1024 * 1024
AUTOTUNE_CEILING = 1 * 1024 * 1024
AUTOTUNE_FINE_STEP = 8 * 1024
IPC_SLAB_ALIGNMENT = 256


def parse_pcie_oneshot_max_size(value: str | int | None) -> Optional[int]:
    """Parse a byte-size string, or return ``None`` for ``auto``."""

    if value is None:
        return None
    if isinstance(value, int):
        return value
    if value.lower() == "auto":
        return None
    normalized = value.upper().strip()
    suffixes = {
        "KB": 1024,
        "K": 1024,
        "MB": 1024 * 1024,
        "M": 1024 * 1024,
    }
    for suffix, multiplier in sorted(suffixes.items(), key=lambda item: -len(item[0])):
        if normalized.endswith(suffix):
            return int(normalized[: -len(suffix)]) * multiplier
    return int(value)


def _normalize_device(device: torch.device | int | str) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if isinstance(device, int):
        return torch.device(f"cuda:{device}")
    return torch.device(device)


def _current_stream_key(
    device: torch.device | int | str, stream: object = None
) -> Optional[int]:
    device_obj = _normalize_device(device)
    if device_obj.type != "cuda":
        return None
    if stream is None:
        stream = torch.cuda.current_stream(device_obj)
    if hasattr(stream, "cuda_stream"):
        return int(stream.cuda_stream)
    return int(stream)


def _push_mode_enabled() -> bool:
    """Match the extension's FLASHINFER_EXP_SM12X_PCIE_ONESHOT_PUSH transport toggle.

    Push transport writes each rank's input into a per-source shard of every
    peer's eager slot, so each slot must hold world_size * max_size bytes.
    """

    return os.getenv("FLASHINFER_EXP_SM12X_PCIE_ONESHOT_PUSH", "0") not in ("", "0")


def _is_weak_contiguous(inp: torch.Tensor) -> bool:
    if inp.is_contiguous():
        return True
    storage = inp.untyped_storage()
    return (
        storage.nbytes() - inp.storage_offset() * inp.element_size()
        == inp.numel() * inp.element_size()
    )


def _align_up(value: int, alignment: int) -> int:
    return ((int(value) + int(alignment) - 1) // int(alignment)) * int(alignment)


def _resolve_exchange_group(
    exchange_group: Optional[ProcessGroup],
    process_group: Optional[ProcessGroup],
) -> Optional[ProcessGroup]:
    if (
        exchange_group is not None
        and process_group is not None
        and exchange_group is not process_group
    ):
        raise ValueError("pass only one of exchange_group or process_group")
    return exchange_group if exchange_group is not None else process_group


def _group_ranks(group: ProcessGroup) -> list[int]:
    world_size = dist.get_world_size(group=group)
    if hasattr(dist, "get_process_group_ranks"):
        ranks = sorted(dist.get_process_group_ranks(group=group))
        if len(ranks) != world_size:
            raise RuntimeError("process-group rank list does not match world size")
        return ranks
    return list(range(world_size))


def _object_broadcast_device(group: ProcessGroup) -> torch.device | str:
    try:
        try:
            backend = dist.get_backend(group=group)
        except TypeError:
            backend = dist.get_backend(group)
    except Exception as exc:
        raise RuntimeError(
            "PCIe oneshot IPC exchange requires a CUDA/NCCL process group"
        ) from exc
    backend_name = str(backend).lower()
    if "nccl" not in backend_name:
        raise RuntimeError(
            f"PCIe oneshot IPC exchange requires an NCCL process group, got {backend}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("PCIe oneshot IPC exchange requires CUDA")
    return torch.device("cuda", torch.cuda.current_device())


def _broadcast_gather_object(local_object: object, group: ProcessGroup) -> list[object]:
    # `broadcast_object_list` is more robust than `all_gather_object` for
    # the host-side exchange that happens around IPC setup and graph capture.
    world_size = dist.get_world_size(group=group)
    rank = dist.get_rank(group=group)
    all_objects: list[list[object | None]] = [[None] for _ in range(world_size)]
    all_objects[rank][0] = local_object
    device = _object_broadcast_device(group)
    for index, src_rank in enumerate(_group_ranks(group)):
        dist.broadcast_object_list(
            all_objects[index], src=src_rank, group=group, device=device
        )
    return [entry[0] for entry in all_objects]


@dataclass(frozen=True)
class _OwnedSharedBuffer:
    local_ptr: int
    peer_ptrs: tuple[int, ...]
    remote_ptrs: tuple[int, ...]


@dataclass(frozen=True)
class _ChannelSharedBuffers:
    owned_buffer: _OwnedSharedBuffer
    signal_ptrs: tuple[int, ...]
    eager0_ptrs: tuple[int, ...]
    eager1_ptrs: tuple[int, ...]


@dataclass(frozen=True)
class _BenchmarkResult:
    size_bytes: int
    custom_us: float
    nccl_us: float
    winner: str


@lru_cache(maxsize=1)
def _load_extension():
    source = Path(__file__).with_name("pcie_oneshot.cu")
    verbose = os.getenv("FLASHINFER_EXP_SM12X_PCIE_ONESHOT_VERBOSE_BUILD", "0") == "1"
    return load(
        name="sm12x_pcie_oneshot_ext",
        sources=[str(source)],
        extra_cuda_cflags=["-O2", "--expt-relaxed-constexpr"],
        extra_ldflags=["-lcuda"],
        verbose=verbose,
    )


def _compute_crossover_size(
    benchmark: Callable[[int], tuple[float, float]],
    *,
    ceiling_bytes: int = AUTOTUNE_CEILING,
    fine_step_bytes: int = AUTOTUNE_FINE_STEP,
) -> tuple[int, list[_BenchmarkResult]]:
    coarse_sizes = []
    current = 1024
    while current <= ceiling_bytes:
        coarse_sizes.append(current)
        current *= 2

    results: list[_BenchmarkResult] = []
    seen_sizes: set[int] = set()
    first_nccl_win: Optional[int] = None
    last_custom_win = 0

    def record(size_bytes: int) -> None:
        nonlocal first_nccl_win, last_custom_win
        custom_us, nccl_us = benchmark(size_bytes)
        winner = "custom" if custom_us < nccl_us else "NCCL"
        results.append(_BenchmarkResult(size_bytes, custom_us, nccl_us, winner))
        seen_sizes.add(size_bytes)
        if winner == "custom":
            last_custom_win = max(last_custom_win, size_bytes)
        elif first_nccl_win is None:
            first_nccl_win = size_bytes

    for size_bytes in coarse_sizes:
        record(size_bytes)

    if last_custom_win > 0 and first_nccl_win is not None:
        fine_start = last_custom_win
        fine_end = min(first_nccl_win, last_custom_win * 4)
        fine_size = fine_start + fine_step_bytes
        while fine_size < fine_end:
            aligned = (fine_size // 16) * 16
            if aligned not in seen_sizes:
                record(aligned)
            fine_size += fine_step_bytes

    results.sort(key=lambda item: item.size_bytes)
    crossover = 1024
    for result in results:
        if result.winner == "custom":
            crossover = result.size_bytes
    return crossover, results


class PCIeOneshotAllReduce:
    """Standalone unfused PCIe oneshot allreduce runtime."""

    def __init__(
        self,
        *,
        rank: int,
        world_size: int,
        device: torch.device | int | str,
        signal_ptrs: Sequence[int],
        eager_buffer_ptrs0: Optional[Sequence[int]] = None,
        eager_buffer_ptrs1: Optional[Sequence[int]] = None,
        exchange_group: Optional[ProcessGroup] = None,
        process_group: Optional[ProcessGroup] = None,
        ipc: Optional[CudaRTLibrary] = None,
        owned_buffers: Optional[Sequence[_OwnedSharedBuffer]] = None,
        max_size: int = DEFAULT_MAX_SIZE,
        rank_data_bytes: int = DEFAULT_RANK_DATA_BYTES,
        ext_module=None,
        stream_affine: bool = True,
    ):
        if world_size not in SUPPORTED_WORLD_SIZES:
            raise ValueError(f"unsupported world size {world_size}")
        if rank < 0 or rank >= world_size:
            raise ValueError(f"invalid rank {rank} for world size {world_size}")
        if len(signal_ptrs) != world_size:
            raise ValueError("signal_ptrs must match world size")
        if (eager_buffer_ptrs0 is None) != (eager_buffer_ptrs1 is None):
            raise ValueError("eager buffers must be provided as a pair")
        if eager_buffer_ptrs0 is not None and len(eager_buffer_ptrs0) != world_size:
            raise ValueError("eager_buffer_ptrs0 must match world size")
        if eager_buffer_ptrs1 is not None and len(eager_buffer_ptrs1) != world_size:
            raise ValueError("eager_buffer_ptrs1 must match world size")

        self.rank = int(rank)
        self.world_size = int(world_size)
        self.device = _normalize_device(device)
        self.exchange_group = _resolve_exchange_group(exchange_group, process_group)
        # Compatibility alias for older callers that still refer to this as
        # `process_group`.
        self.process_group = self.exchange_group
        self.max_size = int(max_size)
        self._ipc = ipc
        if self._ipc is None and (ext_module is None or owned_buffers):
            self._ipc = CudaRTLibrary()
        self._signal_ptrs = tuple(int(ptr) for ptr in signal_ptrs)
        self._owned_buffers = list(owned_buffers or [])
        self._registered_input_ptrs: dict[int, tuple[int, ...]] = {}
        self._stream_affine = bool(stream_affine)
        self._owner_stream_key: Optional[int] = None
        self._closed = False
        self._ext = ext_module or _load_extension()

        if ext_module is None and self.device.type != "cuda":
            raise ValueError("PCIe oneshot allreduce requires a CUDA device")

        self.rank_data = torch.empty(
            rank_data_bytes, dtype=torch.uint8, device=self.device
        )
        self._ptr = self._ext.init_custom_ar(
            list(self._signal_ptrs), self.rank_data, self.rank
        )

        self._eager_ptrs: Optional[tuple[tuple[int, ...], tuple[int, ...]]] = None
        if eager_buffer_ptrs0 is not None and eager_buffer_ptrs1 is not None:
            self._eager_ptrs = (
                tuple(int(ptr) for ptr in eager_buffer_ptrs0),
                tuple(int(ptr) for ptr in eager_buffer_ptrs1),
            )
            self._ext.register_pcie_buffers(
                self._ptr,
                list(self._eager_ptrs[0]),
                list(self._eager_ptrs[1]),
            )

    @classmethod
    def from_ipc(
        cls,
        *,
        rank: int,
        world_size: int,
        device: torch.device | int | str,
        signal_ptrs: Sequence[int],
        eager_buffer_ptrs0: Optional[Sequence[int]] = None,
        eager_buffer_ptrs1: Optional[Sequence[int]] = None,
        exchange_group: Optional[ProcessGroup] = None,
        process_group: Optional[ProcessGroup] = None,
        max_size: int = DEFAULT_MAX_SIZE,
        rank_data_bytes: int = DEFAULT_RANK_DATA_BYTES,
        ext_module=None,
        stream_affine: bool = True,
    ) -> "PCIeOneshotAllReduce":
        return cls(
            rank=rank,
            world_size=world_size,
            device=device,
            signal_ptrs=signal_ptrs,
            eager_buffer_ptrs0=eager_buffer_ptrs0,
            eager_buffer_ptrs1=eager_buffer_ptrs1,
            exchange_group=exchange_group,
            process_group=process_group,
            max_size=max_size,
            rank_data_bytes=rank_data_bytes,
            ext_module=ext_module,
            stream_affine=stream_affine,
        )

    @classmethod
    def from_exchange_group(
        cls,
        *,
        exchange_group: ProcessGroup,
        device: torch.device | int | str,
        eager_buffer_bytes: int = DEFAULT_MAX_SIZE,
        max_size: int = DEFAULT_MAX_SIZE,
        rank_data_bytes: int = DEFAULT_RANK_DATA_BYTES,
        ext_module=None,
        stream_affine: bool = True,
    ) -> "PCIeOneshotAllReduce":
        rank = dist.get_rank(group=exchange_group)
        world_size = dist.get_world_size(group=exchange_group)
        if world_size not in SUPPORTED_WORLD_SIZES:
            raise ValueError(f"unsupported world size {world_size}")

        device_obj = _normalize_device(device)
        if device_obj.type != "cuda":
            raise ValueError("PCIe oneshot requires a CUDA device")

        ipc = CudaRTLibrary()
        ipc.cudaSetDevice(device_obj.index or 0)
        ext = ext_module or _load_extension()

        owned_buffers: list[_OwnedSharedBuffer] = []
        try:
            channel_buffers = cls._allocate_eager_channel_buffers(
                exchange_group,
                signal_bytes=ext.meta_size(),
                eager_buffer_bytes=eager_buffer_bytes,
                ipc=ipc,
            )
            owned_buffers.append(channel_buffers.owned_buffer)

            return cls(
                rank=rank,
                world_size=world_size,
                device=device_obj,
                signal_ptrs=channel_buffers.signal_ptrs,
                eager_buffer_ptrs0=channel_buffers.eager0_ptrs,
                eager_buffer_ptrs1=channel_buffers.eager1_ptrs,
                exchange_group=exchange_group,
                ipc=ipc,
                owned_buffers=owned_buffers,
                max_size=max_size,
                rank_data_bytes=rank_data_bytes,
                ext_module=ext,
                stream_affine=stream_affine,
            )
        except Exception:
            for shared in owned_buffers:
                for ptr in shared.remote_ptrs:
                    with suppress(Exception):
                        ipc.cudaIpcCloseMemHandle(ptr)
                with suppress(Exception):
                    ipc.cudaFree(shared.local_ptr)
            raise

    @classmethod
    def from_process_group(
        cls,
        *,
        process_group: ProcessGroup,
        device: torch.device | int | str,
        max_input_bytes: int = DEFAULT_MAX_SIZE,
        eager_buffer_bytes: Optional[int] = None,
        max_size: int = DEFAULT_MAX_SIZE,
        rank_data_bytes: int = DEFAULT_RANK_DATA_BYTES,
        ext_module=None,
        stream_affine: bool = True,
    ) -> "PCIeOneshotAllReduce":
        return cls.from_exchange_group(
            exchange_group=process_group,
            device=device,
            eager_buffer_bytes=max_input_bytes
            if eager_buffer_bytes is None
            else eager_buffer_bytes,
            max_size=max_size,
            rank_data_bytes=rank_data_bytes,
            ext_module=ext_module,
            stream_affine=stream_affine,
        )

    @staticmethod
    def _allocate_shared_buffer(
        exchange_group: ProcessGroup,
        size_in_bytes: int,
        *,
        zero_fill: bool,
        ipc: CudaRTLibrary,
    ) -> _OwnedSharedBuffer:
        local_ptr = ipc.cudaMalloc(size_in_bytes)
        peer_ptrs: list[int] = []
        remote_ptrs: list[int] = []
        try:
            if zero_fill:
                ipc.cudaMemset(local_ptr, 0, size_in_bytes)
            local_handle = ipc.cudaIpcGetMemHandleBytes(local_ptr)
            world_size = dist.get_world_size(group=exchange_group)
            rank = dist.get_rank(group=exchange_group)
            handles = _broadcast_gather_object(local_handle, exchange_group)
            for idx, handle in enumerate(handles):
                if idx == rank:
                    peer_ptrs.append(local_ptr)
                else:
                    try:
                        remote_ptr = ipc.cudaIpcOpenMemHandleBytes(handle)
                    except Exception as exc:
                        raise RuntimeError(
                            f"failed to open CUDA IPC handle for peer rank {idx}"
                        ) from exc
                    peer_ptrs.append(remote_ptr)
                    remote_ptrs.append(remote_ptr)
            if len(peer_ptrs) != world_size:
                raise RuntimeError("failed to gather IPC handles for all ranks")
            return _OwnedSharedBuffer(
                local_ptr=local_ptr,
                peer_ptrs=tuple(peer_ptrs),
                remote_ptrs=tuple(remote_ptrs),
            )
        except Exception:
            for ptr in remote_ptrs:
                with suppress(Exception):
                    ipc.cudaIpcCloseMemHandle(ptr)
            with suppress(Exception):
                ipc.cudaFree(local_ptr)
            raise

    @classmethod
    def _allocate_eager_channel_buffers(
        cls,
        exchange_group: ProcessGroup,
        *,
        signal_bytes: int,
        eager_buffer_bytes: int,
        ipc: CudaRTLibrary,
    ) -> _ChannelSharedBuffers:
        signal_bytes = int(signal_bytes)
        eager_buffer_bytes = int(eager_buffer_bytes)
        if signal_bytes <= 0:
            raise ValueError("signal_bytes must be positive")
        if eager_buffer_bytes <= 0:
            raise ValueError("eager_buffer_bytes must be positive")
        if _push_mode_enabled():
            eager_buffer_bytes *= dist.get_world_size(group=exchange_group)

        signal_offset = 0
        eager0_offset = _align_up(signal_bytes, IPC_SLAB_ALIGNMENT)
        eager1_offset = eager0_offset + _align_up(
            eager_buffer_bytes, IPC_SLAB_ALIGNMENT
        )
        slab_bytes = eager1_offset + eager_buffer_bytes
        slab = cls._allocate_shared_buffer(
            exchange_group,
            slab_bytes,
            # Clear signals before publishing the IPC handle so a peer cannot
            # post an arrival that a later local memset would erase.
            zero_fill=True,
            ipc=ipc,
        )
        return _ChannelSharedBuffers(
            owned_buffer=slab,
            signal_ptrs=tuple(ptr + signal_offset for ptr in slab.peer_ptrs),
            eager0_ptrs=tuple(ptr + eager0_offset for ptr in slab.peer_ptrs),
            eager1_ptrs=tuple(ptr + eager1_offset for ptr in slab.peer_ptrs),
        )

    @property
    def signal_ptrs(self) -> tuple[int, ...]:
        return self._signal_ptrs

    def _bind_stream_key(self, stream_key: Optional[int]) -> None:
        if not self._stream_affine:
            return
        if stream_key is None:
            return
        if self._owner_stream_key is None:
            self._owner_stream_key = int(stream_key)
            return
        if self._owner_stream_key != int(stream_key):
            raise RuntimeError(
                "PCIe oneshot allreduce channels are stream-affine; "
                "create or use a separate channel for each CUDA stream"
            )

    def _check_stream(self, stream: object = None) -> None:
        if stream is None and _is_current_stream_capturing(self.device):
            # During CUDA graph capture the current stream is a torch-owned,
            # ephemeral capture stream (true for piecewise/inductor graphs used
            # by MTP and spec-decode). Stream affinity does not apply: the
            # captured kernel replays on the caller's stream, not this one, so
            # skip the affinity guard instead of rejecting the capture stream.
            return
        self._bind_stream_key(_current_stream_key(self.device, stream))

    def should_allreduce(self, inp: torch.Tensor) -> bool:
        if self._closed:
            return False
        if inp.device != self.device:
            return False
        if inp.dtype not in SUPPORTED_DTYPES:
            return False
        inp_bytes = inp.numel() * inp.element_size()
        if inp_bytes > self.max_size:
            return False
        if inp_bytes % 16 != 0:
            return False
        return _is_weak_contiguous(inp)

    def register_buffer(self, peer_input_ptrs: Sequence[int]) -> None:
        if self._closed:
            raise RuntimeError("runtime is closed")
        if len(peer_input_ptrs) != self.world_size:
            raise ValueError("peer_input_ptrs must match world size")
        ptrs = tuple(int(ptr) for ptr in peer_input_ptrs)
        local_ptr = ptrs[self.rank]
        existing = self._registered_input_ptrs.get(local_ptr)
        if existing is not None:
            if existing != ptrs:
                raise ValueError(
                    "input pointer is already registered with different peer_input_ptrs"
                )
            return
        self._ext.register_buffer(self._ptr, list(ptrs))
        self._registered_input_ptrs[local_ptr] = ptrs

    def _prepare_input(
        self,
        inp: torch.Tensor,
        peer_input_ptrs: Optional[Sequence[int]],
    ) -> None:
        local_ptr = int(inp.data_ptr())
        if peer_input_ptrs is not None:
            if len(peer_input_ptrs) != self.world_size:
                raise ValueError("peer_input_ptrs must match world size")
            ptrs = tuple(int(ptr) for ptr in peer_input_ptrs)
            if ptrs[self.rank] != local_ptr:
                raise ValueError("peer_input_ptrs[self.rank] must match inp.data_ptr()")
            self.register_buffer(ptrs)
        elif self._eager_ptrs is None and local_ptr not in self._registered_input_ptrs:
            raise ValueError(
                "peer_input_ptrs are required unless eager IPC buffers are configured "
                "or this input was already registered"
            )

    def get_graph_buffer_ipc_meta(self) -> tuple[list[int], list[int]]:
        if self._closed:
            raise RuntimeError("runtime is closed")
        handle, offsets = self._ext.get_graph_buffer_ipc_meta(self._ptr)
        return list(handle), list(offsets)

    def register_graph_buffers_from_ranks(
        self,
        handles: Sequence[Sequence[int]],
        offsets: Sequence[Sequence[int]],
    ) -> None:
        if self._closed:
            raise RuntimeError("runtime is closed")
        if len(handles) != self.world_size:
            raise ValueError("handles must match world size")
        if len(offsets) != self.world_size:
            raise ValueError("offsets must match world size")
        self._ext.register_graph_buffers(
            self._ptr,
            [list(map(int, handle)) for handle in handles],
            [list(map(int, rank_offsets)) for rank_offsets in offsets],
        )

    def all_reduce(
        self,
        inp: torch.Tensor,
        *,
        out: Optional[torch.Tensor] = None,
        peer_input_ptrs: Optional[Sequence[int]] = None,
    ) -> torch.Tensor:
        if self._closed:
            raise RuntimeError("runtime is closed")
        if inp.device != self.device:
            raise ValueError(
                f"input device {inp.device} does not match runtime device {self.device}"
            )
        self._check_stream()
        if not self.should_allreduce(inp):
            raise ValueError(
                "input does not satisfy device/dtype/size/alignment/contiguity requirements "
                f"(shape={tuple(inp.shape)}, dtype={inp.dtype})"
            )

        if out is None:
            out = torch.empty_like(inp)
        if out.device != inp.device:
            raise ValueError("output tensor must be on the same device as the input")
        if out.shape != inp.shape or out.dtype != inp.dtype:
            raise ValueError("output tensor must match input shape and dtype")
        if not _is_weak_contiguous(out):
            raise ValueError("output tensor must be weak-contiguous")

        self._prepare_input(inp, peer_input_ptrs)

        self._ext.all_reduce(self._ptr, inp, out, 0, 0)
        return out

    def all_reduce_fused_add_rms_norm(
        self,
        inp: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        epsilon: float,
        *,
        out: Optional[torch.Tensor] = None,
        residual_out: Optional[torch.Tensor] = None,
        peer_input_ptrs: Optional[Sequence[int]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """All-reduce ``inp``, add ``residual``, and apply RMSNorm."""

        if self._closed:
            raise RuntimeError("runtime is closed")
        if inp.device != self.device:
            raise ValueError(
                f"input device {inp.device} does not match runtime device {self.device}"
            )
        self._check_stream()
        if not self.should_allreduce(inp):
            raise ValueError(
                "input does not satisfy device/dtype/size/alignment/contiguity "
                f"requirements (shape={tuple(inp.shape)}, dtype={inp.dtype})"
            )
        if inp.ndim == 0:
            raise ValueError("input must have at least one dimension")
        hidden_size = inp.shape[-1]
        if hidden_size * inp.element_size() % 16 != 0:
            raise ValueError(
                "the last input dimension must occupy a multiple of 16 bytes"
            )
        if residual.device != inp.device:
            raise ValueError("residual tensor must be on the same device as the input")
        if residual.shape != inp.shape or residual.dtype != inp.dtype:
            raise ValueError("residual tensor must match input shape and dtype")
        if not _is_weak_contiguous(residual):
            raise ValueError("residual tensor must be weak-contiguous")
        if weight.device != inp.device:
            raise ValueError("weight tensor must be on the same device as the input")
        if weight.shape != (hidden_size,) or weight.dtype != inp.dtype:
            raise ValueError(
                "weight tensor must match the input dtype and last dimension"
            )
        if not weight.is_contiguous():
            raise ValueError("weight tensor must be contiguous")
        if epsilon < 0:
            raise ValueError("epsilon must be non-negative")

        if out is None:
            out = torch.empty_like(inp)
        if residual_out is None:
            residual_out = torch.empty_like(residual)
        for name, tensor in (("output", out), ("residual output", residual_out)):
            if tensor.device != inp.device:
                raise ValueError(
                    f"{name} tensor must be on the same device as the input"
                )
            if tensor.shape != inp.shape or tensor.dtype != inp.dtype:
                raise ValueError(f"{name} tensor must match input shape and dtype")
            if not _is_weak_contiguous(tensor):
                raise ValueError(f"{name} tensor must be weak-contiguous")
        if out.data_ptr() == residual_out.data_ptr():
            raise ValueError("output and residual output must not alias")

        self._prepare_input(inp, peer_input_ptrs)
        self._ext.all_reduce_fused_add_rms_norm(
            self._ptr,
            inp,
            residual,
            weight,
            out,
            residual_out,
            float(epsilon),
            0,
            0,
        )
        return out, residual_out

    @contextmanager
    def capture(self, stream: object = None):
        if self.exchange_group is None and self._eager_ptrs is None:
            raise ValueError(
                "exchange_group is required for CUDA graph capture registration"
            )
        self._check_stream(stream)
        try:
            yield
        finally:
            if self.exchange_group is not None and self._eager_ptrs is None:
                self.register_graph_buffers()

    def register_graph_buffers(self) -> None:
        if self.exchange_group is None:
            raise ValueError("exchange_group is required to register graph buffers")
        local_meta = self.get_graph_buffer_ipc_meta()
        all_meta = _broadcast_gather_object(local_meta, self.exchange_group)
        num_buffers = [len(entry[1]) for entry in all_meta]
        if any(count != num_buffers[0] for count in num_buffers):
            raise RuntimeError(
                "graph capture registered a different number of buffers across ranks"
            )
        if num_buffers[0] == 0:
            return
        self.register_graph_buffers_from_ranks(
            [entry[0] for entry in all_meta],
            [entry[1] for entry in all_meta],
        )

    def _bench_graph_latency(
        self,
        size_bytes: int,
        nccl_group: ProcessGroup,
        stream: torch.cuda.Stream,
        warmup: int,
        iters: int,
    ) -> tuple[float, float]:
        if self.exchange_group is None:
            raise ValueError("exchange_group is required for graph-based autotuning")
        self._check_stream(stream)

        numel = size_bytes // torch.tensor([], dtype=torch.bfloat16).element_size()
        device = self.device

        def run_custom() -> float:
            with torch.cuda.stream(stream):
                graph_inp = torch.ones(numel, dtype=torch.bfloat16, device=device)
                graph_out = torch.zeros_like(graph_inp)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                self._ext.all_reduce(self._ptr, graph_inp, graph_out, 0, 0)
            self.register_graph_buffers()
            dist.barrier(group=nccl_group)
            with torch.cuda.stream(stream):
                for _ in range(warmup):
                    graph.replay()
            stream.synchronize()
            start = time.perf_counter()
            with torch.cuda.stream(stream):
                for _ in range(iters):
                    graph.replay()
            stream.synchronize()
            return (time.perf_counter() - start) / iters * 1e6

        def run_nccl() -> float:
            with torch.cuda.stream(stream):
                graph_inp = torch.ones(numel, dtype=torch.bfloat16, device=device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                dist.all_reduce(graph_inp, group=nccl_group)
            with torch.cuda.stream(stream):
                for _ in range(warmup):
                    graph.replay()
            stream.synchronize()
            start = time.perf_counter()
            with torch.cuda.stream(stream):
                for _ in range(iters):
                    graph.replay()
            stream.synchronize()
            return (time.perf_counter() - start) / iters * 1e6

        custom_runs = sorted(run_custom() for _ in range(3))
        nccl_runs = sorted(run_nccl() for _ in range(3))
        # Reduce timings across ranks so every rank reaches the same
        # crossover verdicts; divergent local verdicts would desynchronize
        # the sweep's collective sequence and deadlock.
        stats = torch.tensor(
            [custom_runs[1], nccl_runs[1]], dtype=torch.float64, device=device
        )
        dist.all_reduce(stats, op=dist.ReduceOp.MAX, group=nccl_group)
        return float(stats[0].item()), float(stats[1].item())

    def find_crossover_size(
        self,
        nccl_group: ProcessGroup,
        *,
        ceiling_bytes: int = AUTOTUNE_CEILING,
        fine_step_bytes: int = AUTOTUNE_FINE_STEP,
        warmup: int = 100,
        iters: int = 1000,
    ) -> int:
        if self.device.type != "cuda":
            raise ValueError("autotune requires a CUDA device")
        bench_stream = torch.cuda.Stream(device=self.device)
        crossover, results = _compute_crossover_size(
            lambda size_bytes: self._bench_graph_latency(
                size_bytes,
                nccl_group,
                bench_stream,
                warmup,
                iters,
            ),
            ceiling_bytes=ceiling_bytes,
            fine_step_bytes=fine_step_bytes,
        )
        self.max_size = crossover

        if self.rank == 0:

            def fmt_size(size_bytes: int) -> str:
                if size_bytes >= 1024 * 1024:
                    return f"{size_bytes // (1024 * 1024)}MB"
                if size_bytes >= 1024:
                    return f"{size_bytes // 1024}KB"
                return f"{size_bytes}B"

            lines = [
                f"[PCIe oneshot allreduce] Crossover benchmark ({self.world_size} GPUs, bf16):"
            ]
            for result in results:
                lines.append(
                    f"  {fmt_size(result.size_bytes):>6s}:  custom {result.custom_us:6.1f} us  "
                    f"vs  NCCL {result.nccl_us:6.1f} us  -> {result.winner} wins"
                )
            lines.append(
                f"  Setting max_size = {fmt_size(crossover)} (last size where custom AR wins)"
            )
            logger.info("\n".join(lines))
        return crossover

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if getattr(self, "_ptr", 0):
            self._ext.dispose(self._ptr)
            self._ptr = 0
        for shared in self._owned_buffers:
            for ptr in shared.remote_ptrs:
                if self._ipc is not None:
                    self._ipc.cudaIpcCloseMemHandle(ptr)
            if self._ipc is not None:
                self._ipc.cudaFree(shared.local_ptr)
        self._owned_buffers.clear()
        self._registered_input_ptrs.clear()

    def __del__(self) -> None:
        with suppress(Exception):
            self.close()


def _is_current_stream_capturing(device: torch.device) -> bool:
    if device.type != "cuda":
        return False
    is_capturing = getattr(torch.cuda, "is_current_stream_capturing", None)
    if is_capturing is None:
        return False
    return bool(is_capturing())


class PCIeOneshotAllReducePool:
    """Stream-affine PCIe oneshot wrapper.

    A ``PCIeOneshotAllReduce`` instance is a single ordered channel with one
    signal buffer and one double-buffered staging pair. The pool creates a
    separate channel for each CUDA stream key so multi-stream callers never
    reuse those buffers concurrently.
    """

    def __init__(
        self,
        *,
        rank: int,
        world_size: int,
        device: torch.device | int | str,
        exchange_group: Optional[ProcessGroup] = None,
        process_group: Optional[ProcessGroup] = None,
        eager_buffer_bytes: int = DEFAULT_MAX_SIZE,
        max_size: int = DEFAULT_MAX_SIZE,
        rank_data_bytes: int = DEFAULT_RANK_DATA_BYTES,
        ext_module=None,
        ipc: Optional[CudaRTLibrary] = None,
        single_channel: bool = False,
        channel_factory: Optional[
            Callable[[Optional[int]], PCIeOneshotAllReduce]
        ] = None,
    ):
        if world_size not in SUPPORTED_WORLD_SIZES:
            raise ValueError(f"unsupported world size {world_size}")
        if rank < 0 or rank >= world_size:
            raise ValueError(f"invalid rank {rank} for world size {world_size}")

        self.rank = int(rank)
        self.world_size = int(world_size)
        self.device = _normalize_device(device)
        self.exchange_group = _resolve_exchange_group(exchange_group, process_group)
        self.process_group = self.exchange_group
        self.eager_buffer_bytes = int(eager_buffer_bytes)
        self.max_size = int(max_size)
        self.rank_data_bytes = int(rank_data_bytes)
        self.single_channel = bool(single_channel)
        self._channel_factory = channel_factory
        self._channels: dict[int, PCIeOneshotAllReduce] = {}
        self._capture_channel_stack: list[PCIeOneshotAllReduce] = []
        self._closed = False

        self._ipc = ipc
        self._ext = ext_module
        if self._channel_factory is None:
            if self.exchange_group is None:
                raise ValueError(
                    "exchange_group is required unless channel_factory is provided"
                )
            if self.device.type != "cuda":
                raise ValueError("PCIe oneshot pool requires a CUDA device")
            self._ipc = self._ipc or CudaRTLibrary()
            self._ipc.cudaSetDevice(self.device.index or 0)
            self._ext = self._ext or _load_extension()

    @classmethod
    def from_exchange_group(
        cls,
        *,
        exchange_group: ProcessGroup,
        device: torch.device | int | str,
        eager_buffer_bytes: int = DEFAULT_MAX_SIZE,
        max_size: int = DEFAULT_MAX_SIZE,
        rank_data_bytes: int = DEFAULT_RANK_DATA_BYTES,
        ext_module=None,
        single_channel: bool = False,
    ) -> "PCIeOneshotAllReducePool":
        return cls(
            rank=dist.get_rank(group=exchange_group),
            world_size=dist.get_world_size(group=exchange_group),
            device=device,
            exchange_group=exchange_group,
            eager_buffer_bytes=eager_buffer_bytes,
            max_size=max_size,
            rank_data_bytes=rank_data_bytes,
            ext_module=ext_module,
            single_channel=single_channel,
        )

    @classmethod
    def from_process_group(
        cls,
        *,
        process_group: ProcessGroup,
        device: torch.device | int | str,
        max_input_bytes: int = DEFAULT_MAX_SIZE,
        eager_buffer_bytes: Optional[int] = None,
        max_size: int = DEFAULT_MAX_SIZE,
        rank_data_bytes: int = DEFAULT_RANK_DATA_BYTES,
        ext_module=None,
        single_channel: bool = False,
    ) -> "PCIeOneshotAllReducePool":
        return cls.from_exchange_group(
            exchange_group=process_group,
            device=device,
            eager_buffer_bytes=max_input_bytes
            if eager_buffer_bytes is None
            else eager_buffer_bytes,
            max_size=max_size,
            rank_data_bytes=rank_data_bytes,
            ext_module=ext_module,
            single_channel=single_channel,
        )

    def _new_channel(self, stream_key: Optional[int]) -> PCIeOneshotAllReduce:
        if self._channel_factory is not None:
            channel = self._channel_factory(stream_key)
            if self.single_channel:
                channel._stream_affine = False
            channel._bind_stream_key(stream_key)
            return channel

        if self.exchange_group is None or self._ipc is None or self._ext is None:
            raise RuntimeError("pool is not configured to allocate channels")

        owned_buffers: list[_OwnedSharedBuffer] = []
        try:
            channel_buffers = PCIeOneshotAllReduce._allocate_eager_channel_buffers(
                self.exchange_group,
                signal_bytes=self._ext.meta_size(),
                eager_buffer_bytes=self.eager_buffer_bytes,
                ipc=self._ipc,
            )
            owned_buffers.append(channel_buffers.owned_buffer)

            channel = PCIeOneshotAllReduce(
                rank=self.rank,
                world_size=self.world_size,
                device=self.device,
                signal_ptrs=channel_buffers.signal_ptrs,
                eager_buffer_ptrs0=channel_buffers.eager0_ptrs,
                eager_buffer_ptrs1=channel_buffers.eager1_ptrs,
                exchange_group=self.exchange_group,
                ipc=self._ipc,
                owned_buffers=owned_buffers,
                max_size=self.max_size,
                rank_data_bytes=self.rank_data_bytes,
                ext_module=self._ext,
                stream_affine=not self.single_channel,
            )
        except Exception:
            for shared in owned_buffers:
                for ptr in shared.remote_ptrs:
                    with suppress(Exception):
                        self._ipc.cudaIpcCloseMemHandle(ptr)
                with suppress(Exception):
                    self._ipc.cudaFree(shared.local_ptr)
            raise
        channel._bind_stream_key(stream_key)
        return channel

    def for_stream(self, stream: object = None) -> PCIeOneshotAllReduce:
        if self._closed:
            raise RuntimeError("pool is closed")
        if self.single_channel:
            channel = self._channels.get(0)
            if channel is not None:
                return channel
            if _is_current_stream_capturing(self.device):
                raise RuntimeError(
                    "PCIe oneshot pool has no channel to reuse during CUDA graph "
                    "capture; perform an eager all-reduce (or call for_stream) "
                    "before capture starts"
                )
            channel = self._new_channel(None)
            self._channels[0] = channel
            return channel

        stream_key = _current_stream_key(self.device, stream)
        channel_key = 0 if stream_key is None else int(stream_key)
        channel = self._channels.get(channel_key)
        if channel is not None:
            return channel
        if _is_current_stream_capturing(self.device):
            # Piecewise / inductor CUDA graphs (MTP, spec-decode) capture on a
            # torch-owned stream that we cannot pre-register before capture
            # starts. Reuse the channel selected by the enclosing vLLM graph
            # capture, not an arbitrary channel from another graph manager.
            # Target and draft graph managers can replay independently; if
            # their nested captures share signal/staging buffers, they race.
            if self._capture_channel_stack:
                channel = self._capture_channel_stack[-1]
                self._channels[channel_key] = channel
                return channel
            # Preserve compatibility for callers that enter CUDA capture
            # without the pool.capture() context. They must already have a
            # channel because allocating CUDA IPC storage mid-capture is
            # illegal.
            if self._channels:
                channel = next(iter(self._channels.values()))
                self._channels[channel_key] = channel
                return channel
            raise RuntimeError(
                "PCIe oneshot pool has no channel to reuse during CUDA graph "
                "capture; perform an eager all-reduce (or call for_stream) "
                "before capture starts"
            )
        channel = self._new_channel(stream_key)
        self._channels[channel_key] = channel
        return channel

    def all_reduce(
        self,
        inp: torch.Tensor,
        *,
        out: Optional[torch.Tensor] = None,
        peer_input_ptrs: Optional[Sequence[int]] = None,
        stream: object = None,
    ) -> torch.Tensor:
        channel = self.for_stream(stream)
        if stream is not None and self.device.type == "cuda":
            with torch.cuda.stream(stream):
                return channel.all_reduce(inp, out=out, peer_input_ptrs=peer_input_ptrs)
        return channel.all_reduce(inp, out=out, peer_input_ptrs=peer_input_ptrs)

    def all_reduce_fused_add_rms_norm(
        self,
        inp: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        epsilon: float,
        *,
        out: Optional[torch.Tensor] = None,
        residual_out: Optional[torch.Tensor] = None,
        peer_input_ptrs: Optional[Sequence[int]] = None,
        stream: object = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        channel = self.for_stream(stream)

        def run() -> tuple[torch.Tensor, torch.Tensor]:
            return channel.all_reduce_fused_add_rms_norm(
                inp,
                residual,
                weight,
                epsilon,
                out=out,
                residual_out=residual_out,
                peer_input_ptrs=peer_input_ptrs,
            )

        if stream is not None and self.device.type == "cuda":
            with torch.cuda.stream(stream):
                return run()
        return run()

    @contextmanager
    def capture(self, stream: object = None):
        channel = self.for_stream(stream)
        with channel.capture(stream=stream):
            self._capture_channel_stack.append(channel)
            try:
                yield channel
            finally:
                popped = self._capture_channel_stack.pop()
                if popped is not channel:
                    raise RuntimeError("PCIe oneshot capture channel stack corrupted")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for channel in self._channels.values():
            channel.close()
        self._channels.clear()

    def __del__(self) -> None:
        with suppress(Exception):
            self.close()


__all__ = [
    "PCIeOneshotAllReduce",
    "PCIeOneshotAllReducePool",
    "SUPPORTED_WORLD_SIZES",
    "parse_pcie_oneshot_max_size",
]
