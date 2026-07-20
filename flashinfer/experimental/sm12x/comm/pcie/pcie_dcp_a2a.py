# SPDX-FileCopyrightText: 2026 FlashInfer team
# SPDX-License-Identifier: Apache-2.0
# Ported from b12x b12x/distributed/pcie_dcp_a2a.py @ 00695ee8 (2026-07-19) -- one-time curated port.
# Upstream b12x is a research sandbox; this tree is the canonical home.
"""PCIe one-shot DCP attention exchange with fused LSE reduction."""

from __future__ import annotations

import os
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
from .pcie_oneshot import (
    IPC_SLAB_ALIGNMENT,
    PCIeOneshotAllReduce,
    _align_up,
    _current_stream_key,
    _is_current_stream_capturing,
    _normalize_device,
    _OwnedSharedBuffer,
)


SUPPORTED_WORLD_SIZES = (2, 4, 8)
SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)


@dataclass(frozen=True)
class _StagingLayout:
    signal_bytes: int
    staging0_offset: int
    staging1_offset: int
    output_capacity_elems: int
    lse_offset: int
    lse_capacity: int
    slot_bytes: int
    slab_bytes: int


def _staging_layout(
    *,
    signal_bytes: int,
    world_size: int,
    max_batch_size: int,
    total_heads: int,
    head_dim: int,
    query_head_dim: Optional[int] = None,
) -> _StagingLayout:
    if signal_bytes <= 0:
        raise ValueError("signal_bytes must be positive")
    if world_size not in SUPPORTED_WORLD_SIZES:
        raise ValueError(f"unsupported world size {world_size}")
    if max_batch_size <= 0:
        raise ValueError("max_batch_size must be positive")
    if total_heads <= 0 or total_heads % world_size != 0:
        raise ValueError("total_heads must be positive and divisible by world_size")
    if head_dim <= 0 or head_dim % 8 != 0:
        raise ValueError("head_dim must be a positive multiple of 8")
    if query_head_dim is None:
        query_head_dim = head_dim
    if query_head_dim <= 0 or query_head_dim % 8 != 0:
        raise ValueError("query_head_dim must be a positive multiple of 8")

    output_elems = max_batch_size * total_heads * max(head_dim, query_head_dim)
    output_bytes = _align_up(output_elems * 2, IPC_SLAB_ALIGNMENT)
    output_capacity_elems = output_bytes // 2
    lse_offset = output_bytes
    lse_elems = max_batch_size * total_heads
    lse_capacity = _align_up(lse_elems * 4, IPC_SLAB_ALIGNMENT) // 4
    slot_bytes = _align_up(
        lse_offset + lse_capacity * 4,
        IPC_SLAB_ALIGNMENT,
    )
    staging0_offset = _align_up(signal_bytes, IPC_SLAB_ALIGNMENT)
    staging1_offset = staging0_offset + slot_bytes
    return _StagingLayout(
        signal_bytes=signal_bytes,
        staging0_offset=staging0_offset,
        staging1_offset=staging1_offset,
        output_capacity_elems=output_capacity_elems,
        lse_offset=lse_offset,
        lse_capacity=lse_capacity,
        slot_bytes=slot_bytes,
        slab_bytes=staging1_offset + slot_bytes,
    )


@lru_cache(maxsize=1)
def _load_extension():
    source = Path(__file__).with_name("pcie_dcp_a2a.cu")
    verbose = os.getenv("FLASHINFER_EXP_SM12X_PCIE_DCP_A2A_VERBOSE_BUILD", "0") == "1"
    return load(
        name="sm12x_pcie_dcp_a2a_ext",
        sources=[str(source)],
        extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr"],
        extra_ldflags=["-lcuda"],
        verbose=verbose,
    )


def lse_reduce_scatter_reference(
    partial_outputs: torch.Tensor,
    partial_lses: torch.Tensor,
    rank: int,
    *,
    is_lse_base_on_e: bool = True,
) -> torch.Tensor:
    """Reference LSE-weighted reduction for stacked rank contributions.

    Args:
        partial_outputs: Tensor shaped ``[world, batch, heads, head_dim]``.
        partial_lses: FP32 tensor shaped ``[world, batch, heads]``.
        rank: Destination rank whose contiguous head shard is returned.
        is_lse_base_on_e: Whether LSE values use natural logarithms.

    Returns:
        The reduced output shaped ``[batch, heads // world, head_dim]``.
    """
    if partial_outputs.ndim != 4:
        raise ValueError("partial_outputs must have rank 4")
    if partial_lses.shape != partial_outputs.shape[:-1]:
        raise ValueError("partial_lses shape must match partial_outputs[:-1]")
    world_size, _, total_heads, _ = partial_outputs.shape
    if not 0 <= rank < world_size:
        raise ValueError(f"invalid rank {rank} for world size {world_size}")
    if total_heads % world_size != 0:
        raise ValueError("total heads must be divisible by world size")

    heads_per_rank = total_heads // world_size
    head_slice = slice(rank * heads_per_rank, (rank + 1) * heads_per_rank)
    outputs = partial_outputs[:, :, head_slice, :].float()
    lses = partial_lses[:, :, head_slice].float()
    valid = torch.isfinite(lses)
    sanitized = torch.where(valid, lses, torch.full_like(lses, -torch.inf))
    max_lse = sanitized.amax(dim=0)
    max_lse = torch.where(torch.isfinite(max_lse), max_lse, 0.0)
    if is_lse_base_on_e:
        weights = torch.exp(sanitized - max_lse.unsqueeze(0))
    else:
        weights = torch.exp2(sanitized - max_lse.unsqueeze(0))
    weights = torch.where(valid, weights, torch.zeros_like(weights))
    weights /= weights.sum(dim=0, keepdim=True).clamp_min_(1e-10)
    return (outputs * weights.unsqueeze(-1)).sum(dim=0).to(partial_outputs.dtype)


class PCIeDCPA2A:
    """One ordered IPC channel for DCP attention collectives."""

    def __init__(
        self,
        *,
        rank: int,
        world_size: int,
        device: torch.device | int | str,
        signal_ptrs: Sequence[int],
        staging0_ptrs: Sequence[int],
        staging1_ptrs: Sequence[int],
        max_batch_size: int,
        total_heads: int,
        head_dim: int,
        output_capacity_elems: int,
        lse_offset: int,
        lse_capacity: int,
        query_head_dim: Optional[int] = None,
        exchange_group: Optional[ProcessGroup] = None,
        ipc: Optional[CudaRTLibrary] = None,
        owned_buffers: Optional[Sequence[_OwnedSharedBuffer]] = None,
        ext_module=None,
        stream_affine: bool = True,
    ) -> None:
        if world_size not in SUPPORTED_WORLD_SIZES:
            raise ValueError(f"unsupported world size {world_size}")
        if not 0 <= rank < world_size:
            raise ValueError(f"invalid rank {rank} for world size {world_size}")
        if (
            len(signal_ptrs) != world_size
            or len(staging0_ptrs) != world_size
            or len(staging1_ptrs) != world_size
        ):
            raise ValueError("signal and staging pointers must match world size")
        if total_heads <= 0 or total_heads % world_size != 0:
            raise ValueError("total_heads must be divisible by world_size")
        if head_dim <= 0 or head_dim % 8 != 0:
            raise ValueError("head_dim must be a positive multiple of 8")

        self.rank = int(rank)
        self.world_size = int(world_size)
        self.device = _normalize_device(device)
        self.exchange_group = exchange_group
        self.max_batch_size = int(max_batch_size)
        self.total_heads = int(total_heads)
        self.head_dim = int(head_dim)
        self.query_head_dim = int(query_head_dim or head_dim)
        if self.query_head_dim <= 0 or self.query_head_dim % 8 != 0:
            raise ValueError("query_head_dim must be a positive multiple of 8")
        self.heads_per_rank = self.total_heads // self.world_size
        self._ipc = ipc
        self._owned_buffers = list(owned_buffers or ())
        self._ext = ext_module or _load_extension()
        self._stream_affine = bool(stream_affine)
        self._owner_stream_key: Optional[int] = None
        self._closed = False
        self._ptr = self._ext.init_dcp_a2a(
            list(signal_ptrs),
            list(staging0_ptrs),
            list(staging1_ptrs),
            int(output_capacity_elems),
            int(lse_offset),
            int(lse_capacity),
            self.rank,
        )

    @classmethod
    def from_exchange_group(
        cls,
        *,
        exchange_group: ProcessGroup,
        device: torch.device | int | str,
        max_batch_size: int,
        total_heads: int,
        head_dim: int,
        query_head_dim: Optional[int] = None,
        ext_module=None,
        stream_affine: bool = True,
    ) -> "PCIeDCPA2A":
        rank = dist.get_rank(group=exchange_group)
        world_size = dist.get_world_size(group=exchange_group)
        device_obj = _normalize_device(device)
        if device_obj.type != "cuda":
            raise ValueError("PCIe DCP A2A requires a CUDA device")

        ipc = CudaRTLibrary()
        ipc.cudaSetDevice(device_obj.index or 0)
        ext = ext_module or _load_extension()
        layout = _staging_layout(
            signal_bytes=int(ext.meta_size()),
            world_size=world_size,
            max_batch_size=max_batch_size,
            total_heads=total_heads,
            head_dim=head_dim,
            query_head_dim=query_head_dim,
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
                max_batch_size=max_batch_size,
                total_heads=total_heads,
                head_dim=head_dim,
                output_capacity_elems=layout.output_capacity_elems,
                lse_offset=layout.lse_offset,
                lse_capacity=layout.lse_capacity,
                query_head_dim=query_head_dim,
                exchange_group=exchange_group,
                ipc=ipc,
                owned_buffers=owned,
                ext_module=ext,
                stream_affine=stream_affine,
            )
        except Exception:
            for shared in owned:
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
        max_batch_size: int,
        total_heads: int,
        head_dim: int,
        query_head_dim: Optional[int] = None,
        ext_module=None,
        stream_affine: bool = True,
    ) -> "PCIeDCPA2A":
        return cls.from_exchange_group(
            exchange_group=process_group,
            device=device,
            max_batch_size=max_batch_size,
            total_heads=total_heads,
            head_dim=head_dim,
            query_head_dim=query_head_dim,
            ext_module=ext_module,
            stream_affine=stream_affine,
        )

    def _bind_stream_key(self, stream_key: Optional[int]) -> None:
        if not self._stream_affine or stream_key is None:
            return
        if self._owner_stream_key is None:
            self._owner_stream_key = int(stream_key)
            return
        if self._owner_stream_key != int(stream_key):
            raise RuntimeError(
                "PCIe DCP A2A channels are stream-affine; use a separate "
                "channel for each CUDA stream"
            )

    def _check_stream(self, stream: object = None) -> None:
        if self.device.type != "cuda":
            return
        if stream is None and _is_current_stream_capturing(self.device):
            return
        self._bind_stream_key(_current_stream_key(self.device, stream))

    def _validate(
        self,
        partial_output: torch.Tensor,
        partial_lse: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        if self._closed:
            raise RuntimeError("PCIeDCPA2A is closed")
        if partial_output.device != self.device or partial_lse.device != self.device:
            raise ValueError("inputs must be on the runtime device")
        if out.device != self.device:
            raise ValueError("output must be on the runtime device")
        if partial_output.dtype not in SUPPORTED_DTYPES:
            raise ValueError(f"unsupported output dtype {partial_output.dtype}")
        if partial_lse.dtype != torch.float32:
            raise ValueError("partial_lse must be float32")
        if out.dtype != partial_output.dtype:
            raise ValueError("output dtype must match partial_output")
        if partial_output.ndim != 3:
            raise ValueError("partial_output must have shape [batch, heads, head_dim]")
        batch, heads, head_dim = partial_output.shape
        if batch <= 0 or batch > self.max_batch_size:
            raise ValueError(
                f"batch size {batch} exceeds configured capacity {self.max_batch_size}"
            )
        if heads != self.total_heads or head_dim != self.head_dim:
            raise ValueError(
                "partial_output shape does not match configured heads/head_dim: "
                f"{tuple(partial_output.shape)}"
            )
        if partial_lse.shape != (batch, heads):
            raise ValueError("partial_lse must have shape [batch, heads]")
        expected_out = (batch, self.heads_per_rank, self.head_dim)
        if out.shape != expected_out:
            raise ValueError(
                f"output shape must be {expected_out}, got {tuple(out.shape)}"
            )
        if not partial_output.is_contiguous():
            raise ValueError("partial_output must be contiguous")
        if not partial_lse.is_contiguous():
            raise ValueError("partial_lse must be contiguous")
        if not out.is_contiguous():
            raise ValueError("output must be contiguous")

    def lse_reduce_scatter(
        self,
        partial_output: torch.Tensor,
        partial_lse: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        *,
        is_lse_base_on_e: bool = True,
        threads: int = 256,
        block_limit: int = 16,
    ) -> torch.Tensor:
        """Exchange rank contributions and return this rank's reduced heads."""
        self._check_stream()
        if out is None:
            out = torch.empty(
                partial_output.shape[0],
                self.heads_per_rank,
                self.head_dim,
                device=partial_output.device,
                dtype=partial_output.dtype,
            )
        self._validate(partial_output, partial_lse, out)
        self._ext.lse_reduce_scatter(
            self._ptr,
            partial_output,
            partial_lse,
            out,
            bool(is_lse_base_on_e),
            int(threads),
            int(block_limit),
        )
        return out

    def all_gather_heads(
        self,
        local_input: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        *,
        threads: int = 256,
        block_limit: int = 16,
    ) -> torch.Tensor:
        """Gather rank-local heads into a rank-major head dimension."""
        self._check_stream()
        if self._closed:
            raise RuntimeError("PCIeDCPA2A is closed")
        if local_input.device != self.device:
            raise ValueError("input must be on the runtime device")
        if local_input.dtype not in SUPPORTED_DTYPES:
            raise ValueError(f"unsupported input dtype {local_input.dtype}")
        if local_input.ndim != 3:
            raise ValueError("input must have shape [batch, local_heads, head_dim]")
        batch, local_heads, head_dim = local_input.shape
        if batch <= 0 or batch > self.max_batch_size:
            raise ValueError(
                f"batch size {batch} exceeds configured capacity {self.max_batch_size}"
            )
        if local_heads != self.heads_per_rank or head_dim != self.query_head_dim:
            raise ValueError(
                "input shape does not match configured local heads/head_dim: "
                f"{tuple(local_input.shape)}"
            )
        if not local_input.is_contiguous():
            raise ValueError("input must be contiguous")
        expected_out = (batch, self.total_heads, self.query_head_dim)
        if out is None:
            out = torch.empty(
                expected_out,
                device=local_input.device,
                dtype=local_input.dtype,
            )
        if out.device != self.device or out.dtype != local_input.dtype:
            raise ValueError("output device and dtype must match input")
        if out.shape != expected_out:
            raise ValueError(
                f"output shape must be {expected_out}, got {tuple(out.shape)}"
            )
        if not out.is_contiguous():
            raise ValueError("output must be contiguous")
        self._ext.all_gather_heads(
            self._ptr,
            local_input,
            out,
            int(threads),
            int(block_limit),
        )
        return out

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        with suppress(Exception):
            self._ext.dispose(self._ptr)
        if self._ipc is not None:
            for shared in self._owned_buffers:
                for ptr in shared.remote_ptrs:
                    with suppress(Exception):
                        self._ipc.cudaIpcCloseMemHandle(ptr)
                with suppress(Exception):
                    self._ipc.cudaFree(shared.local_ptr)
        self._owned_buffers.clear()

    def __del__(self) -> None:
        with suppress(Exception):
            self.close()


class PCIeDCPA2APool:
    """Create an independent DCP collective channel for each CUDA stream."""

    def __init__(
        self,
        *,
        rank: int,
        world_size: int,
        device: torch.device | int | str,
        max_batch_size: int,
        total_heads: int,
        head_dim: int,
        query_head_dim: Optional[int] = None,
        exchange_group: Optional[ProcessGroup] = None,
        ext_module=None,
        single_channel: bool = False,
        channel_factory: Optional[Callable[[Optional[int]], PCIeDCPA2A]] = None,
    ) -> None:
        if world_size not in SUPPORTED_WORLD_SIZES:
            raise ValueError(f"unsupported world size {world_size}")
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.device = _normalize_device(device)
        self.max_batch_size = int(max_batch_size)
        self.total_heads = int(total_heads)
        self.head_dim = int(head_dim)
        self.query_head_dim = int(query_head_dim or head_dim)
        self.exchange_group = exchange_group
        self._ext = ext_module
        self.single_channel = bool(single_channel)
        self._channel_factory = channel_factory
        self._channels: dict[int, PCIeDCPA2A] = {}
        self._capture_channel_stack: list[PCIeDCPA2A] = []
        self._closed = False
        if channel_factory is None and exchange_group is None:
            raise ValueError("exchange_group is required unless channel_factory is set")

    @classmethod
    def from_exchange_group(
        cls,
        *,
        exchange_group: ProcessGroup,
        device: torch.device | int | str,
        max_batch_size: int,
        total_heads: int,
        head_dim: int,
        query_head_dim: Optional[int] = None,
        ext_module=None,
        single_channel: bool = False,
    ) -> "PCIeDCPA2APool":
        return cls(
            rank=dist.get_rank(group=exchange_group),
            world_size=dist.get_world_size(group=exchange_group),
            device=device,
            max_batch_size=max_batch_size,
            total_heads=total_heads,
            head_dim=head_dim,
            query_head_dim=query_head_dim,
            exchange_group=exchange_group,
            ext_module=ext_module,
            single_channel=single_channel,
        )

    @classmethod
    def from_process_group(
        cls,
        *,
        process_group: ProcessGroup,
        device: torch.device | int | str,
        max_batch_size: int,
        total_heads: int,
        head_dim: int,
        query_head_dim: Optional[int] = None,
        ext_module=None,
        single_channel: bool = False,
    ) -> "PCIeDCPA2APool":
        return cls.from_exchange_group(
            exchange_group=process_group,
            device=device,
            max_batch_size=max_batch_size,
            total_heads=total_heads,
            head_dim=head_dim,
            query_head_dim=query_head_dim,
            ext_module=ext_module,
            single_channel=single_channel,
        )

    def _new_channel(self, stream_key: Optional[int]) -> PCIeDCPA2A:
        if self._channel_factory is not None:
            channel = self._channel_factory(stream_key)
        else:
            assert self.exchange_group is not None
            channel = PCIeDCPA2A.from_exchange_group(
                exchange_group=self.exchange_group,
                device=self.device,
                max_batch_size=self.max_batch_size,
                total_heads=self.total_heads,
                head_dim=self.head_dim,
                query_head_dim=self.query_head_dim,
                ext_module=self._ext,
                stream_affine=not self.single_channel,
            )
        channel._bind_stream_key(stream_key)
        return channel

    def for_stream(self, stream: object = None) -> PCIeDCPA2A:
        if self._closed:
            raise RuntimeError("PCIeDCPA2APool is closed")
        if self.single_channel:
            key = 0
            stream_key = None
        else:
            stream_key = _current_stream_key(self.device, stream)
            key = 0 if stream_key is None else int(stream_key)
        channel = self._channels.get(key)
        if channel is not None:
            return channel
        if _is_current_stream_capturing(self.device):
            # Nested piecewise captures use a torch-owned stream key that is
            # unavailable before capture begins. Reuse the channel selected by
            # the enclosing graph manager, never one owned by another graph.
            if self._capture_channel_stack:
                channel = self._capture_channel_stack[-1]
                self._channels[key] = channel
                return channel
            if self._channels:
                channel = next(iter(self._channels.values()))
                self._channels[key] = channel
                return channel
            raise RuntimeError(
                "PCIe DCP A2A pool has no channel during CUDA graph capture; "
                "create or warm a channel before capture"
            )
        channel = self._new_channel(stream_key)
        self._channels[key] = channel
        return channel

    def lse_reduce_scatter(
        self,
        partial_output: torch.Tensor,
        partial_lse: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        *,
        is_lse_base_on_e: bool = True,
        threads: int = 256,
        block_limit: int = 16,
        stream: object = None,
    ) -> torch.Tensor:
        channel = self.for_stream(stream)
        if stream is not None and self.device.type == "cuda":
            with torch.cuda.stream(stream):
                return channel.lse_reduce_scatter(
                    partial_output,
                    partial_lse,
                    out,
                    is_lse_base_on_e=is_lse_base_on_e,
                    threads=threads,
                    block_limit=block_limit,
                )
        return channel.lse_reduce_scatter(
            partial_output,
            partial_lse,
            out,
            is_lse_base_on_e=is_lse_base_on_e,
            threads=threads,
            block_limit=block_limit,
        )

    def all_gather_heads(
        self,
        local_input: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        *,
        threads: int = 256,
        block_limit: int = 16,
        stream: object = None,
    ) -> torch.Tensor:
        channel = self.for_stream(stream)
        if stream is not None and self.device.type == "cuda":
            with torch.cuda.stream(stream):
                return channel.all_gather_heads(
                    local_input,
                    out,
                    threads=threads,
                    block_limit=block_limit,
                )
        return channel.all_gather_heads(
            local_input,
            out,
            threads=threads,
            block_limit=block_limit,
        )

    @contextmanager
    def capture(self, stream: object = None):
        """Bind nested CUDA captures to the enclosing stream's channel."""
        channel = self.for_stream(stream)
        self._capture_channel_stack.append(channel)
        try:
            yield channel
        finally:
            popped = self._capture_channel_stack.pop()
            if popped is not channel:
                raise RuntimeError("PCIe DCP A2A capture channel stack corrupted")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        seen: set[int] = set()
        for channel in self._channels.values():
            if id(channel) not in seen:
                seen.add(id(channel))
                channel.close()
        self._channels.clear()

    def __del__(self) -> None:
        with suppress(Exception):
            self.close()


__all__ = [
    "PCIeDCPA2A",
    "PCIeDCPA2APool",
    "SUPPORTED_WORLD_SIZES",
    "lse_reduce_scatter_reference",
]
