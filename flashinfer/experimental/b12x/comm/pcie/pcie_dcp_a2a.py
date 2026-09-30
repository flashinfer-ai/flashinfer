"""PCIe one-shot DCP attention exchange with fused LSE reduction."""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from b12x.preparation.types import Plan, require_prepared

from ._cuda_ipc import CudaRTLibrary
from .pcie_oneshot import (
    _ABANDONED_PCIE_RUNTIME_QUARANTINE,
    _SINGLE_CHANNEL_ID,
    IPC_SLAB_ALIGNMENT,
    PCIeOneshotAllReduce,
    _finish_collective_runtime_setup,
    _raise_local_cleanup_errors,
    _align_up,
    _broadcast_gather_object,
    _collective_capture_needs_preparation,
    _coordinated_close_channels,
    _current_stream_key,
    _cuda_device_index,
    _device_guard,
    _is_current_stream_capturing,
    _normalize_device,
    _normalize_logical_channel_id,
    _OwnedSharedBuffer,
    _finish_collective_unowned_runtime_setup,
    _require_collective_contract,
    _require_full_grid_residency,
    _run_collective_preallocation_setup,
)


SUPPORTED_WORLD_SIZES = (2, 4, 8, 9, 10, 12, 16)
SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)
SUPPORTED_GATHER_DTYPES = (*SUPPORTED_DTYPES, torch.float8_e4m3fn)
SUPPORTED_PAIR_DTYPES = (*SUPPORTED_GATHER_DTYPES, torch.float32)
DCP_A2A_REQUIRED_SMS = 64
_MAX_BLOCKS = 64
_MAX_RANKS = 16
_FLAG_STRIDE = 32
_SIGNAL_BYTES = (
    _MAX_BLOCKS * _MAX_RANKS
    + 2 * _MAX_BLOCKS * _MAX_RANKS * _FLAG_STRIDE
) * 4


def _env_int(name: str, fallback: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return int(fallback)
    try:
        return int(raw)
    except ValueError:
        return int(fallback)


def _is_supported_bhd_layout(tensor: torch.Tensor) -> bool:
    """Accept packed token-major or capacity-strided head-major BHD views."""
    if tensor.ndim != 3 or int(tensor.stride(2)) != 1:
        return False
    batch, heads, head_dim = (int(value) for value in tensor.shape)
    stride_batch, stride_head, _ = (int(value) for value in tensor.stride())
    packed_token_major = stride_batch == heads * head_dim and stride_head == head_dim
    capacity_strided_head_major = (
        stride_batch == head_dim
        and stride_head >= batch * head_dim
        and stride_head % 8 == 0
    )
    return packed_token_major or capacity_strided_head_major




def kimi_topk16(
    router_logits: torch.Tensor,
    correction_bias: torch.Tensor,
    output_weights: Optional[torch.Tensor] = None,
    output_ids: Optional[torch.Tensor] = None,
    *,
    plan: Plan,
    threads: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select Kimi-K3's 16 routed experts with prepared native code.

    The operation accepts one to eight assembled FP32 router rows. It launches
    on the current CUDA stream and is CUDA-graph safe with caller-owned outputs.
    """

    if router_logits.ndim != 2:
        raise ValueError("router_logits must be a contiguous rank-2 tensor")
    rows = int(router_logits.shape[0])
    if rows < 1 or rows > 8:
        raise ValueError(f"Kimi top-16 rows {rows} must be between 1 and 8")
    device = router_logits.device
    if device.type != "cuda":
        raise ValueError("Kimi top-16 requires CUDA tensors")
    expected = (
        (router_logits, (rows, 896), torch.float32, "router_logits"),
        (correction_bias, (896,), torch.float32, "correction_bias"),
    )
    for value, shape, dtype, name in expected:
        if (
            value.device != device
            or value.shape != shape
            or value.dtype != dtype
            or not value.is_contiguous()
        ):
            raise ValueError(f"{name} must be contiguous {shape} {dtype} on {device}")

    capturing = _is_current_stream_capturing(device)
    if capturing and (output_weights is None or output_ids is None):
        raise RuntimeError(
            "Kimi top-16 CUDA graph capture requires caller-owned "
            "output_weights and output_ids"
        )
    if output_weights is None:
        output_weights = torch.empty((rows, 16), device=device, dtype=torch.float32)
    if output_ids is None:
        output_ids = torch.empty((rows, 16), device=device, dtype=torch.int32)
    outputs = (
        (output_weights, torch.float32, "output_weights"),
        (output_ids, torch.int32, "output_ids"),
    )
    for value, dtype, name in outputs:
        if (
            value.device != device
            or value.shape != (rows, 16)
            or value.dtype != dtype
            or not value.is_contiguous()
        ):
            raise ValueError(
                f"{name} must be contiguous {(rows, 16)} {dtype} on {device}"
            )
    state = require_prepared(plan, "comm.pcie", device)
    if not hasattr(state, "launcher"):
        raise TypeError("plan does not contain prepared Kimi top-k code")
    with torch.cuda.device(device):
        from ._dcp_a2a_cute import kimi_topk16 as launch_kimi_topk16
        launch_kimi_topk16(
            router_logits_ptr=router_logits.data_ptr(),
            correction_bias_ptr=correction_bias.data_ptr(),
            output_weights_ptr=output_weights.data_ptr(),
            output_ids_ptr=output_ids.data_ptr(),
            rows=rows, threads=threads, launcher=state.launcher(False),
        )
    return output_weights, output_ids


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
    outputs = torch.where(valid.unsqueeze(-1), outputs, torch.zeros_like(outputs))
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
        def normalize_and_validate():
            device_obj = _normalize_device(device)
            normalized_rank = int(rank)
            normalized_world_size = int(world_size)
            normalized_signals = tuple(int(ptr) for ptr in signal_ptrs)
            normalized_staging0 = tuple(int(ptr) for ptr in staging0_ptrs)
            normalized_staging1 = tuple(int(ptr) for ptr in staging1_ptrs)
            normalized_max_batch = int(max_batch_size)
            normalized_total_heads = int(total_heads)
            normalized_head_dim = int(head_dim)
            normalized_query_dim = int(
                head_dim if query_head_dim is None else query_head_dim
            )
            normalized_output_capacity = int(output_capacity_elems)
            normalized_lse_offset = int(lse_offset)
            normalized_lse_capacity = int(lse_capacity)
            if normalized_world_size not in SUPPORTED_WORLD_SIZES:
                raise ValueError(f"unsupported world size {normalized_world_size}")
            if not 0 <= normalized_rank < normalized_world_size:
                raise ValueError(
                    f"invalid rank {normalized_rank} for world size "
                    f"{normalized_world_size}"
                )
            if (
                len(normalized_signals) != normalized_world_size
                or len(normalized_staging0) != normalized_world_size
                or len(normalized_staging1) != normalized_world_size
            ):
                raise ValueError("signal and staging pointers must match world size")
            if normalized_max_batch <= 0:
                raise ValueError("max_batch_size must be positive")
            if (
                normalized_total_heads <= 0
                or normalized_total_heads % normalized_world_size != 0
            ):
                raise ValueError("total_heads must be divisible by world_size")
            if normalized_head_dim <= 0 or normalized_head_dim % 8 != 0:
                raise ValueError("head_dim must be a positive multiple of 8")
            if normalized_query_dim <= 0 or normalized_query_dim % 8 != 0:
                raise ValueError("query_head_dim must be a positive multiple of 8")
            required_output = (
                normalized_max_batch
                * normalized_total_heads
                * max(normalized_head_dim, normalized_query_dim)
            )
            if normalized_output_capacity < required_output:
                raise ValueError("output capacity is smaller than configured shape")
            if normalized_lse_offset < 0:
                raise ValueError("lse_offset must be non-negative")
            if normalized_lse_capacity < normalized_max_batch * normalized_total_heads:
                raise ValueError("LSE capacity is smaller than configured shape")
            if ext_module is None and device_obj.type != "cuda":
                raise ValueError("PCIe DCP A2A requires a CUDA device")
            if device_obj.type == "cuda" and exchange_group is None:
                raise ValueError(
                    "exchange_group is required for a CUDA PCIe DCP A2A runtime; "
                    "use from_exchange_group()"
                )
            if exchange_group is not None:
                if device_obj.type != "cuda":
                    raise ValueError("distributed PCIe DCP A2A requires a CUDA device")
                group_rank = dist.get_rank(group=exchange_group)
                group_world_size = dist.get_world_size(group=exchange_group)
                if normalized_rank != group_rank:
                    raise ValueError(
                        f"supplied rank {normalized_rank} does not match process "
                        f"group rank {group_rank}"
                    )
                if normalized_world_size != group_world_size:
                    raise ValueError(
                        f"supplied world size {normalized_world_size} does not "
                        f"match process group size {group_world_size}"
                    )
            return (
                device_obj,
                normalized_rank,
                normalized_world_size,
                normalized_signals,
                normalized_staging0,
                normalized_staging1,
                normalized_max_batch,
                normalized_total_heads,
                normalized_head_dim,
                normalized_query_dim,
                normalized_output_capacity,
                normalized_lse_offset,
                normalized_lse_capacity,
            )

        if exchange_group is not None:
            normalized = _run_collective_preallocation_setup(
                owner="PCIe DCP A2A direct constructor argument validation",
                exchange_group=exchange_group,
                setup=normalize_and_validate,
            )
        else:
            normalized = normalize_and_validate()

        (
            prepared_device,
            prepared_rank,
            prepared_world_size,
            prepared_signals,
            prepared_staging0,
            prepared_staging1,
            prepared_max_batch,
            prepared_total_heads,
            prepared_head_dim,
            prepared_query_dim,
            prepared_output_capacity,
            prepared_lse_offset,
            prepared_lse_capacity,
        ) = normalized
        self._initialize_prepared_state(
            rank=prepared_rank,
            world_size=prepared_world_size,
            device=prepared_device,
            signal_ptrs=prepared_signals,
            staging0_ptrs=prepared_staging0,
            staging1_ptrs=prepared_staging1,
            max_batch_size=prepared_max_batch,
            total_heads=prepared_total_heads,
            head_dim=prepared_head_dim,
            query_head_dim=prepared_query_dim,
            output_capacity_elems=prepared_output_capacity,
            lse_offset=prepared_lse_offset,
            lse_capacity=prepared_lse_capacity,
            exchange_group=exchange_group,
            ipc=ipc,
            owned_buffers=owned_buffers,
            ext_module=ext_module,
            stream_affine=stream_affine,
        )

        if self.device.type == "cuda":
            _require_full_grid_residency(
                owner="PCIe DCP A2A direct constructor",
                required_sms=DCP_A2A_REQUIRED_SMS,
                device=self.device,
                exchange_group=self.exchange_group,
            )

        if self.device.type == "cuda" and self.exchange_group is not None:
            _require_collective_contract(
                owner="PCIe DCP A2A direct constructor",
                exchange_group=self.exchange_group,
                contract=(
                    self.world_size,
                    self.max_batch_size,
                    self.total_heads,
                    self.head_dim,
                    self.query_head_dim,
                    self.output_capacity_elems,
                    self.lse_offset,
                    self.lse_capacity,
                ),
            )

        init_error = self._initialize_backend_runtime()
        if self.device.type == "cuda" and self.exchange_group is not None:

            def abort_native_runtime() -> None:
                if self._ptr:
                    assert self._legacy_ext_module is not None
                    self._legacy_ext_module.dispose(self._ptr)
                    self._ptr = 0

            _finish_collective_unowned_runtime_setup(
                owner="PCIe DCP A2A direct constructor",
                exchange_group=self.exchange_group,
                local_error=init_error,
                local_cleanup=abort_native_runtime,
            )
        elif init_error is not None:
            raise init_error

    def _initialize_prepared_state(
        self,
        *,
        rank: int,
        world_size: int,
        device: torch.device,
        signal_ptrs: Sequence[int],
        staging0_ptrs: Sequence[int],
        staging1_ptrs: Sequence[int],
        max_batch_size: int,
        total_heads: int,
        head_dim: int,
        query_head_dim: int,
        output_capacity_elems: int,
        lse_offset: int,
        lse_capacity: int,
        exchange_group: Optional[ProcessGroup],
        ipc: Optional[CudaRTLibrary],
        owned_buffers: Optional[Sequence[_OwnedSharedBuffer]],
        ext_module,
        stream_affine: bool,
    ) -> None:
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.device = device
        self.exchange_group = exchange_group
        self.max_batch_size = int(max_batch_size)
        self.total_heads = int(total_heads)
        self.head_dim = int(head_dim)
        self.query_head_dim = int(query_head_dim)
        self.heads_per_rank = self.total_heads // self.world_size
        self.output_capacity_elems = int(output_capacity_elems)
        self.lse_offset = int(lse_offset)
        self.lse_capacity = int(lse_capacity)
        self._signal_ptrs = tuple(int(ptr) for ptr in signal_ptrs)
        self._staging0_ptrs = tuple(int(ptr) for ptr in staging0_ptrs)
        self._staging1_ptrs = tuple(int(ptr) for ptr in staging1_ptrs)
        self._ipc = ipc
        self._owned_buffers = list(owned_buffers or ())
        self._legacy_ext_module = ext_module
        self._staging_ptrs = (
            tuple(int(ptr) for ptr in staging0_ptrs),
            tuple(int(ptr) for ptr in staging1_ptrs),
        )
        slot_deltas = {
            int(slot1) - int(slot0)
            for slot0, slot1 in zip(staging0_ptrs, staging1_ptrs, strict=True)
        }
        if len(slot_deltas) != 1 or next(iter(slot_deltas)) <= 0:
            raise ValueError(
                "staging slot pointers must have one shared positive stride"
            )
        self._slot_bytes = next(iter(slot_deltas))
        self._output_capacity_elems = int(output_capacity_elems)
        self._lse_offset = int(lse_offset)
        self._lse_capacity = int(lse_capacity)
        self._next_slot = 0
        self._device_slot_selection = False
        self._graph_base_slot = 0
        self._threads_override = _env_int("B12X_PCIE_DCP_THREADS", 0)
        self._block_limit_override = _env_int(
            "B12X_PCIE_DCP_BLOCK_LIMIT", 0
        )
        self._stream_affine = bool(stream_affine)
        self._owner_stream_key: Optional[int] = None
        self._closed = False
        self._ipc_imports_closed = False
        self._ipc_exports_freed = False
        self._coordinated_close_complete = False
        self._closed_ipc_import_indices: set[tuple[int, int]] = set()
        self._ptr = 0

    def _initialize_backend_runtime(self) -> BaseException | None:
        if self._legacy_ext_module is None:
            return None
        init_error: BaseException | None = None
        try:
            self._ptr = self._legacy_ext_module.init_dcp_a2a(
                list(self._signal_ptrs),
                list(self._staging0_ptrs),
                list(self._staging1_ptrs),
                self.output_capacity_elems,
                self.lse_offset,
                self.lse_capacity,
                self.rank,
            )
        except Exception as exc:
            init_error = exc
        return init_error

    @classmethod
    def _from_prepared_factory(
        cls, **kwargs
    ) -> tuple["PCIeDCPA2A", BaseException | None]:
        runtime = object.__new__(cls)
        runtime._initialize_prepared_state(**kwargs)
        return runtime, runtime._initialize_backend_runtime()

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
        def validate_factory_arguments():
            device_obj = _normalize_device(device)
            normalized_max_batch = int(max_batch_size)
            normalized_total_heads = int(total_heads)
            normalized_head_dim = int(head_dim)
            normalized_query_dim = int(
                head_dim if query_head_dim is None else query_head_dim
            )
            if world_size not in SUPPORTED_WORLD_SIZES:
                raise ValueError(f"unsupported world size {world_size}")
            if device_obj.type != "cuda":
                raise ValueError("PCIe DCP A2A requires a CUDA device")
            if normalized_max_batch <= 0:
                raise ValueError("max_batch_size must be positive")
            if normalized_total_heads <= 0 or normalized_total_heads % world_size != 0:
                raise ValueError("total_heads must be divisible by world_size")
            if normalized_head_dim <= 0 or normalized_head_dim % 8 != 0:
                raise ValueError("head_dim must be a positive multiple of 8")
            if normalized_query_dim <= 0 or normalized_query_dim % 8 != 0:
                raise ValueError("query_head_dim must be a positive multiple of 8")
            return (
                device_obj,
                normalized_max_batch,
                normalized_total_heads,
                normalized_head_dim,
                normalized_query_dim,
            )

        (
            device_obj,
            max_batch_size,
            total_heads,
            head_dim,
            query_head_dim,
        ) = _run_collective_preallocation_setup(
            owner="PCIe DCP A2A argument validation",
            exchange_group=exchange_group,
            setup=validate_factory_arguments,
        )

        _require_full_grid_residency(
            owner="PCIe DCP A2A",
            required_sms=DCP_A2A_REQUIRED_SMS,
            device=device_obj,
            exchange_group=exchange_group,
        )

        def prepare():
            prepared_ipc = CudaRTLibrary()
            prepared_ipc.cudaSetDevice(_cuda_device_index(device_obj))
            layout = _staging_layout(
                signal_bytes=(
                    int(ext_module.meta_size())
                    if ext_module is not None
                    else _SIGNAL_BYTES
                ),
                world_size=world_size,
                max_batch_size=max_batch_size,
                total_heads=total_heads,
                head_dim=head_dim,
                query_head_dim=query_head_dim,
            )
            return prepared_ipc, layout

        ipc, layout = _run_collective_preallocation_setup(
            owner="PCIe DCP A2A",
            exchange_group=exchange_group,
            setup=prepare,
        )
        _require_collective_contract(
            owner="PCIe DCP A2A channel layout",
            exchange_group=exchange_group,
            contract=(
                int(max_batch_size),
                int(total_heads),
                int(head_dim),
                int(query_head_dim),
                layout,
            ),
        )
        slab = PCIeOneshotAllReduce._allocate_shared_buffer(
            exchange_group,
            layout.slab_bytes,
            zero_fill=True,
            ipc=ipc,
        )
        runtime, init_error = cls._from_prepared_factory(
            rank=rank,
            world_size=world_size,
            device=device_obj,
            signal_ptrs=slab.peer_ptrs,
            staging0_ptrs=tuple(ptr + layout.staging0_offset for ptr in slab.peer_ptrs),
            staging1_ptrs=tuple(ptr + layout.staging1_offset for ptr in slab.peer_ptrs),
            max_batch_size=max_batch_size,
            total_heads=total_heads,
            head_dim=head_dim,
            output_capacity_elems=layout.output_capacity_elems,
            lse_offset=layout.lse_offset,
            lse_capacity=layout.lse_capacity,
            query_head_dim=query_head_dim,
            exchange_group=exchange_group,
            ipc=ipc,
            owned_buffers=[slab],
            ext_module=ext_module,
            stream_affine=stream_affine,
        )

        def abort_native_runtime() -> None:
            pointer = getattr(runtime, "_ptr", 0)
            if pointer:
                assert runtime._legacy_ext_module is not None
                runtime._legacy_ext_module.dispose(pointer)
                runtime._ptr = 0

        def detach_shared_ownership() -> None:
            runtime._owned_buffers.clear()

        _finish_collective_runtime_setup(
            owner="PCIe DCP A2A",
            exchange_group=exchange_group,
            ipc=ipc,
            shared=slab,
            local_error=init_error,
            local_cleanup=abort_native_runtime,
            detach_shared_ownership=detach_shared_ownership,
        )
        return runtime

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

    def _resolve_launch_config(
        self,
        *,
        threads: int,
        block_limit: int,
    ) -> tuple[int, int]:
        threads = int(threads)
        block_limit = int(block_limit)
        if self._threads_override > 0:
            threads = min(512, max(64, (self._threads_override // 32) * 32))
        if self._block_limit_override > 0:
            block_limit = min(self._block_limit_override, _MAX_BLOCKS)
        if (
            threads < self.world_size
            or threads > 512
            or threads % 32 != 0
        ):
            raise ValueError("threads must be a multiple of 32 in [32, 512]")
        if block_limit <= 0 or block_limit > _MAX_BLOCKS:
            raise ValueError(f"block_limit must be in [1, {_MAX_BLOCKS}]")
        return threads, block_limit

    def _prepared_state(self, plan: Plan):
        state = require_prepared(
            plan, "comm.pcie",
            self.device if self.device.type == "cuda" else None,
        )
        if not hasattr(state, "require_runtime") or not hasattr(state, "launcher"):
            raise TypeError("plan does not contain a prepared DCP channel")
        state.require_runtime(self)
        return state


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
        if batch * heads * head_dim > self._output_capacity_elems:
            raise ValueError("PCIe DCP A2A staging capacity exceeded")
        if batch * heads > self._lse_capacity:
            raise ValueError("PCIe DCP A2A LSE staging capacity exceeded")
        expected_out = (batch, self.heads_per_rank, self.head_dim)
        if out.shape != expected_out:
            raise ValueError(
                f"output shape must be {expected_out}, got {tuple(out.shape)}"
            )
        if not _is_supported_bhd_layout(partial_output):
            raise ValueError("partial_output must be packed token-major or head-major")
        if not partial_lse.is_contiguous():
            raise ValueError("partial_lse must be contiguous")
        if not _is_supported_bhd_layout(out):
            raise ValueError("output must be packed token-major or head-major")

    def lse_reduce_scatter(
        self, partial_output: torch.Tensor, partial_lse: torch.Tensor,
        out: Optional[torch.Tensor] = None, *, plan: Plan,
        is_lse_base_on_e: bool = True, threads: int = 256,
        block_limit: int = 16,
    ) -> torch.Tensor:
        with _device_guard(self.device):
            state = self._prepared_state(plan)
            return self._lse_reduce_scatter_on_device(
                partial_output, partial_lse, out, state=state,
                is_lse_base_on_e=is_lse_base_on_e, threads=threads,
                block_limit=block_limit,
            )

    def _lse_reduce_scatter_on_device(
        self, partial_output: torch.Tensor, partial_lse: torch.Tensor,
        out: Optional[torch.Tensor], *, state, is_lse_base_on_e: bool,
        threads: int, block_limit: int,
    ) -> torch.Tensor:
        self._check_stream()
        if out is None:
            out = torch.empty(partial_output.shape[0], self.heads_per_rank,
                              self.head_dim, device=partial_output.device,
                              dtype=partial_output.dtype)
        self._validate(partial_output, partial_lse, out)
        threads, block_limit = self._resolve_launch_config(
            threads=threads, block_limit=block_limit)
        rows = int(partial_output.shape[0]) * self.heads_per_rank
        blocks = max(1, min(block_limit, (rows + threads // 32 - 1) // (threads // 32)))
        capturing = _is_current_stream_capturing(self.device)
        if capturing and not self._device_slot_selection:
            self._graph_base_slot = self._next_slot & 1
            self._device_slot_selection = True
        slot = self._graph_base_slot if self._device_slot_selection else self._next_slot
        if not self._device_slot_selection:
            self._next_slot ^= 1
        self._launch_lse_reduce_scatter(
            partial_output, partial_lse, out, slot=slot,
            natural_log=bool(is_lse_base_on_e), threads=threads, blocks=blocks,
            device_slot_selection=self._device_slot_selection,
            launcher=state.launcher(self._device_slot_selection),
        )
        return out

    def _launch_lse_reduce_scatter(
        self, partial_output: torch.Tensor, partial_lse: torch.Tensor,
        out: torch.Tensor, *, slot: int, natural_log: bool, threads: int,
        blocks: int, device_slot_selection: bool, launcher,
    ) -> None:
        from ._dcp_a2a_cute import lse_reduce_scatter
        dtype_name = "fp16" if partial_output.dtype == torch.float16 else "bf16"
        with torch.cuda.device(self.device):
            lse_reduce_scatter(
                world_size=self.world_size, rank=self.rank, dtype_name=dtype_name,
                threads=threads, local_output_ptr=partial_output.data_ptr(),
                local_lse_ptr=partial_lse.data_ptr(), output_ptr=out.data_ptr(),
                staging_ptrs=self._staging_ptrs[slot], signal_ptrs=self._signal_ptrs,
                lse_offset=self._lse_offset, batch=int(partial_output.shape[0]),
                total_heads=self.total_heads, head_dim=self.head_dim,
                input_stride_batch=int(partial_output.stride(0)) // 8,
                input_stride_head=int(partial_output.stride(1)) // 8,
                output_stride_batch=int(out.stride(0)) // 8,
                output_stride_head=int(out.stride(1)) // 8, natural_log=natural_log,
                device_slot_selection=device_slot_selection,
                slot_delta_bytes=self._slot_bytes if slot == 0 else -self._slot_bytes,
                blocks=blocks, launcher=launcher,
            )

    def all_gather_heads(
        self,
        local_input: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        *,
        plan: Plan,
        threads: int = 256,
        block_limit: int = 16,
    ) -> torch.Tensor:
        """Gather rank-local heads into a rank-major head dimension."""
        with _device_guard(self.device):
            return self._all_gather_heads_on_device(
                local_input, out, state=self._prepared_state(plan),
                threads=threads, block_limit=block_limit,
            )

    def _all_gather_heads_on_device(
        self,
        local_input: torch.Tensor,
        out: Optional[torch.Tensor],
        *,
        state,
        threads: int,
        block_limit: int,
    ) -> torch.Tensor:
        self._check_stream()
        if self._closed:
            raise RuntimeError("PCIeDCPA2A is closed")
        if local_input.device != self.device:
            raise ValueError("input must be on the runtime device")
        if local_input.dtype not in SUPPORTED_GATHER_DTYPES:
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
        if batch * self.total_heads * self.query_head_dim > self._output_capacity_elems:
            raise ValueError("PCIe DCP all-gather staging capacity exceeded")
        threads, block_limit = self._resolve_launch_config(
            threads=threads,
            block_limit=block_limit,
        )
        rows = int(batch) * self.total_heads
        warps_per_block = threads // 32
        blocks = max(
            1,
            min(block_limit, (rows + warps_per_block - 1) // warps_per_block),
        )
        capturing = _is_current_stream_capturing(self.device)
        if capturing:
            from ._dcp_a2a_cute import is_all_gather_heads_prepared

            if not is_all_gather_heads_prepared(
                self.world_size,
                self.rank,
                threads,
                True,
            ):
                raise RuntimeError(
                    "cold PCIe DCP gather CUDA graph capture is not allowed; "
                    "call prepare_graph_all_gather_heads() before capture"
                )
        if capturing and not self._device_slot_selection:
            self._graph_base_slot = self._next_slot & 1
            self._device_slot_selection = True
        if self._device_slot_selection:
            slot = self._graph_base_slot
        else:
            slot = self._next_slot
            self._next_slot ^= 1
        self._launch_all_gather_heads(
            local_input, out, slot=slot, threads=threads, blocks=blocks,
            device_slot_selection=self._device_slot_selection,
            launcher=state.launcher(self._device_slot_selection),
        )
        return out

    def _launch_all_gather_heads(
        self,
        local_input: torch.Tensor,
        out: torch.Tensor,
        *,
        slot: int,
        threads: int,
        blocks: int,
        device_slot_selection: bool,
        launcher,
    ) -> None:
        from ._dcp_a2a_cute import all_gather_heads

        with torch.cuda.device(self.device):
            all_gather_heads(
                world_size=self.world_size,
                rank=self.rank,
                threads=threads,
                local_input_ptr=local_input.data_ptr(),
                output_ptr=out.data_ptr(),
                staging_ptrs=self._staging_ptrs[slot],
                signal_ptrs=self._signal_ptrs,
                batch=int(local_input.shape[0]),
                local_heads=self.heads_per_rank,
                head_dim=self.query_head_dim,
                element_size=local_input.element_size(),
                device_slot_selection=device_slot_selection,
                slot_delta_bytes=(
                    self._slot_bytes if slot == 0 else -self._slot_bytes
                ),
                blocks=blocks,
                launcher=launcher,
            )

    def all_gather_pair(
        self,
        local_first: torch.Tensor,
        local_second: torch.Tensor,
        out_first: Optional[torch.Tensor] = None,
        out_second: Optional[torch.Tensor] = None,
        *,
        plan: Plan,
        threads: int = 512,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with _device_guard(self.device):
            return self._all_gather_pair_on_device(
                local_first, local_second, out_first, out_second,
                state=self._prepared_state(plan), threads=threads,
            )

    def _all_gather_pair_on_device(
        self,
        local_first: torch.Tensor,
        local_second: torch.Tensor,
        out_first: Optional[torch.Tensor] = None,
        out_second: Optional[torch.Tensor] = None,
        *,
        state,
        threads: int = 512,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather two raw projection rows behind one IPC barrier.

        The two tensors may use different floating dtypes. Their bytes are
        copied independently into rank-major outputs without conversion.
        """
        self._check_stream()
        if self._closed:
            raise RuntimeError("PCIeDCPA2A is closed")
        inputs = (local_first, local_second)
        if any(value.device != self.device for value in inputs):
            raise ValueError("paired inputs must be on the runtime device")
        if any(value.dtype not in SUPPORTED_PAIR_DTYPES for value in inputs):
            raise ValueError(
                "paired inputs must be float16, bfloat16, float32, or "
                "float8_e4m3fn"
            )
        if any(value.ndim != 2 for value in inputs):
            raise ValueError("paired inputs must have shape [batch, width]")
        if any(not value.is_contiguous() for value in inputs):
            raise ValueError("paired inputs must be contiguous")
        batch = int(local_first.shape[0])
        if batch <= 0 or batch > self.max_batch_size:
            raise ValueError(
                f"batch size {batch} exceeds configured capacity {self.max_batch_size}"
            )
        if int(local_second.shape[0]) != batch:
            raise ValueError("paired inputs must have the same batch size")
        row_bytes = tuple(
            int(value.shape[1]) * value.element_size() for value in inputs
        )
        if any(value % 16 for value in row_bytes):
            raise ValueError("paired row widths must be multiples of 16 bytes")
        combined_row_bytes = sum(row_bytes)
        if combined_row_bytes != self.query_head_dim:
            raise ValueError(
                "paired row bytes must match the runtime query dimension: "
                f"got {combined_row_bytes}, expected {self.query_head_dim}"
            )

        expected_first = (batch, int(local_first.shape[1]) * self.world_size)
        expected_second = (batch, int(local_second.shape[1]) * self.world_size)
        if out_first is None:
            out_first = torch.empty(
                expected_first,
                device=local_first.device,
                dtype=local_first.dtype,
            )
        if out_second is None:
            out_second = torch.empty(
                expected_second,
                device=local_second.device,
                dtype=local_second.dtype,
            )
        for name, output, expected, source in (
            ("first", out_first, expected_first, local_first),
            ("second", out_second, expected_second, local_second),
        ):
            if output.device != self.device or output.dtype != source.dtype:
                raise ValueError(f"{name} output device and dtype must match input")
            if output.shape != expected:
                raise ValueError(
                    f"{name} output shape must be {expected}, got {tuple(output.shape)}"
                )
            if not output.is_contiguous():
                raise ValueError(f"{name} output must be contiguous")
        threads, _ = self._resolve_launch_config(threads=threads, block_limit=1)
        capturing = _is_current_stream_capturing(self.device)
        if capturing:
            from ._dcp_a2a_cute import is_all_gather_pair_prepared

            if not is_all_gather_pair_prepared(
                self.world_size,
                self.rank,
                threads,
                True,
                False,
            ):
                raise RuntimeError(
                    "cold PCIe DCP paired gather CUDA graph capture is not "
                    "allowed; call prepare_graph_all_gather_pair() before capture"
                )
        if capturing and not self._device_slot_selection:
            self._graph_base_slot = self._next_slot & 1
            self._device_slot_selection = True
        if self._device_slot_selection:
            slot = self._graph_base_slot
        else:
            slot = self._next_slot
            self._next_slot ^= 1
        self._launch_all_gather_pair(
            local_first, local_second, out_first, out_second, slot=slot,
            threads=threads, device_slot_selection=self._device_slot_selection,
            launcher=state.launcher(self._device_slot_selection),
        )
        return out_first, out_second

    def _launch_all_gather_pair(
        self,
        local_first: torch.Tensor,
        local_second: torch.Tensor,
        out_first: torch.Tensor,
        out_second: torch.Tensor,
        *,
        slot: int,
        threads: int,
        device_slot_selection: bool,
        launcher,
    ) -> None:
        from ._dcp_a2a_cute import all_gather_pair

        with torch.cuda.device(self.device):
            all_gather_pair(
                world_size=self.world_size,
                rank=self.rank,
                threads=threads,
                local_first_ptr=local_first.data_ptr(),
                local_second_ptr=local_second.data_ptr(),
                output_first_ptr=out_first.data_ptr(),
                output_second_ptr=out_second.data_ptr(),
                staging_ptrs=self._staging_ptrs[slot],
                signal_ptrs=self._signal_ptrs,
                batch=int(local_first.shape[0]),
                first_row_bytes=int(local_first.shape[1])
                * local_first.element_size(),
                second_row_bytes=int(local_second.shape[1])
                * local_second.element_size(),
                device_slot_selection=device_slot_selection,
                slot_delta_bytes=(
                    self._slot_bytes if slot == 0 else -self._slot_bytes
                ),
                launcher=launcher,
            )

    def all_gather_pair_kimi_topk(
        self, local_down: torch.Tensor, local_router: torch.Tensor,
        correction_bias: torch.Tensor, out_down: Optional[torch.Tensor] = None,
        topk_weights: Optional[torch.Tensor] = None,
        topk_ids: Optional[torch.Tensor] = None, *, plan: Plan,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with _device_guard(self.device):
            return self._all_gather_pair_kimi_topk_on_device(
                local_down, local_router, correction_bias, out_down, topk_weights,
                topk_ids, state=self._prepared_state(plan))

    def _all_gather_pair_kimi_topk_on_device(
        self, local_down, local_router, correction_bias, out_down, topk_weights,
        topk_ids, *, state,
    ):
        self._check_stream()
        if self._closed:
            raise RuntimeError("PCIeDCPA2A is closed")
        down_width, router_width = 3584 // self.world_size, 896 // self.world_size
        expected = ((local_down, (1, down_width), torch.bfloat16, "local_down"),
                    (local_router, (1, router_width), torch.float32, "local_router"),
                    (correction_bias, (896,), torch.float32, "correction_bias"))
        for value, shape, dtype, name in expected:
            if value.device != self.device or value.shape != shape or value.dtype != dtype or not value.is_contiguous():
                raise ValueError(f"{name} must be contiguous {shape} {dtype} on {self.device}")
        out_down = torch.empty((1, 3584), device=self.device, dtype=torch.bfloat16) if out_down is None else out_down
        topk_weights = torch.empty((1, 16), device=self.device, dtype=torch.float32) if topk_weights is None else topk_weights
        topk_ids = torch.empty((1, 16), device=self.device, dtype=torch.int32) if topk_ids is None else topk_ids
        for value, shape, dtype, name in ((out_down, (1, 3584), torch.bfloat16, "out_down"), (topk_weights, (1, 16), torch.float32, "topk_weights"), (topk_ids, (1, 16), torch.int32, "topk_ids")):
            if value.device != self.device or value.shape != shape or value.dtype != dtype or not value.is_contiguous():
                raise ValueError(f"{name} must be contiguous {shape} {dtype} on {self.device}")
        if _is_current_stream_capturing(self.device) and not self._device_slot_selection:
            self._graph_base_slot = self._next_slot & 1
            self._device_slot_selection = True
        slot = self._graph_base_slot if self._device_slot_selection else self._next_slot
        if not self._device_slot_selection:
            self._next_slot ^= 1
        self._launch_all_gather_pair_kimi_topk(
            local_down, local_router, correction_bias, out_down, topk_weights, topk_ids,
            slot=slot, device_slot_selection=self._device_slot_selection,
            launcher=state.launcher(self._device_slot_selection))
        return out_down, topk_weights, topk_ids

    def _launch_all_gather_pair_kimi_topk(
        self, local_down, local_router, correction_bias, out_down, topk_weights,
        topk_ids, *, slot: int, device_slot_selection: bool, launcher,
    ) -> None:
        from ._dcp_a2a_cute import all_gather_pair_kimi_topk
        with torch.cuda.device(self.device):
            all_gather_pair_kimi_topk(
                world_size=self.world_size, rank=self.rank,
                local_down_ptr=local_down.data_ptr(), local_router_ptr=local_router.data_ptr(),
                correction_bias_ptr=correction_bias.data_ptr(), output_down_ptr=out_down.data_ptr(),
                topk_weights_ptr=topk_weights.data_ptr(), topk_ids_ptr=topk_ids.data_ptr(),
                staging_ptrs=self._staging_ptrs[slot], signal_ptrs=self._signal_ptrs,
                device_slot_selection=device_slot_selection,
                slot_delta_bytes=self._slot_bytes if slot == 0 else -self._slot_bytes,
                launcher=launcher)
    def kimi_topk16(
        self, router_logits: torch.Tensor, correction_bias: torch.Tensor,
        output_weights: Optional[torch.Tensor] = None,
        output_ids: Optional[torch.Tensor] = None, *, plan: Plan,
        threads: int = 256,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with _device_guard(self.device):
            return self._kimi_topk16_on_device(
                router_logits, correction_bias, output_weights, output_ids,
                state=self._prepared_state(plan), threads=threads)

    def _kimi_topk16_on_device(
        self, router_logits, correction_bias, output_weights, output_ids, *,
        state, threads,
    ):
        self._check_stream()
        rows = int(router_logits.shape[0]) if router_logits.ndim == 2 else 0
        if self._closed or rows <= 0 or rows > min(self.max_batch_size, 8):
            raise ValueError("Kimi top-16 rows exceed prepared channel capacity")
        for value, shape, dtype, name in ((router_logits, (rows, 896), torch.float32, "router_logits"), (correction_bias, (896,), torch.float32, "correction_bias")):
            if value.device != self.device or value.shape != shape or value.dtype != dtype or not value.is_contiguous():
                raise ValueError(f"{name} must be contiguous {shape} {dtype} on {self.device}")
        if _is_current_stream_capturing(self.device) and (output_weights is None or output_ids is None):
            raise RuntimeError("Kimi top-16 CUDA graph capture requires caller-owned outputs")
        output_weights = torch.empty((rows, 16), device=self.device, dtype=torch.float32) if output_weights is None else output_weights
        output_ids = torch.empty((rows, 16), device=self.device, dtype=torch.int32) if output_ids is None else output_ids
        for value, dtype, name in ((output_weights, torch.float32, "output_weights"), (output_ids, torch.int32, "output_ids")):
            if value.device != self.device or value.shape != (rows, 16) or value.dtype != dtype or not value.is_contiguous():
                raise ValueError(f"{name} must be contiguous {(rows, 16)} {dtype} on {self.device}")
        self._launch_kimi_topk16(router_logits, correction_bias, output_weights, output_ids, threads=threads, launcher=state.launcher(False))
        return output_weights, output_ids

    def _launch_kimi_topk16(
        self, router_logits, correction_bias, output_weights, output_ids, *,
        threads: int, launcher,
    ) -> None:
        from ._dcp_a2a_cute import kimi_topk16
        with torch.cuda.device(self.device):
            kimi_topk16(
                router_logits_ptr=router_logits.data_ptr(),
                correction_bias_ptr=correction_bias.data_ptr(),
                output_weights_ptr=output_weights.data_ptr(),
                output_ids_ptr=output_ids.data_ptr(), rows=int(router_logits.shape[0]),
                threads=threads, launcher=launcher)

    def _closed_import_indices(self) -> set[tuple[int, int]]:
        closed = getattr(self, "_closed_ipc_import_indices", None)
        if closed is None:
            closed = set()
            self._closed_ipc_import_indices = closed
        return closed

    def _all_python_ipc_imports_closed(self, closed: set[tuple[int, int]]) -> bool:
        if self._ipc is None:
            return not any(shared.remote_ptrs for shared in self._owned_buffers)
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
        if getattr(self, "_ptr", 0):
            try:
                assert self._legacy_ext_module is not None
                self._legacy_ext_module.dispose(self._ptr)
            except Exception as exc:
                failures.append(("native runtime", exc))
            else:
                self._ptr = 0
        closed = self._closed_import_indices()
        if self._ipc is not None:
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
        elif any(shared.remote_ptrs for shared in self._owned_buffers):
            failures.append(
                (
                    "CUDA IPC imports",
                    RuntimeError("CUDA runtime is unavailable for IPC unmap"),
                )
            )

        if (
            not failures
            and not getattr(self, "_ptr", 0)
            and self._all_python_ipc_imports_closed(closed)
        ):
            self._ipc_imports_closed = True
        if failures:
            _raise_local_cleanup_errors("PCIe DCP A2A", "IPC import close", failures)

    def _free_ipc_exports_strict(self) -> None:
        if self._ipc_exports_freed:
            return
        self._close_ipc_imports_strict()
        failures: list[tuple[str, Exception]] = []
        remaining = []
        if self._ipc is not None:
            for shared in self._owned_buffers:
                try:
                    self._ipc.cudaFree(shared.local_ptr)
                except Exception as exc:
                    remaining.append(shared)
                    failures.append((f"CUDA IPC export {shared.local_ptr}", exc))
        elif self._owned_buffers:
            remaining = list(self._owned_buffers)
            failures.append(
                (
                    "CUDA IPC exports",
                    RuntimeError("CUDA runtime is unavailable for export free"),
                )
            )
        self._owned_buffers = remaining
        if not remaining:
            self._ipc_exports_freed = True
        if failures:
            _raise_local_cleanup_errors("PCIe DCP A2A", "IPC export free", failures)

    def close(self) -> None:
        if getattr(self, "_coordinated_close_complete", False):
            return
        _coordinated_close_channels(
            (self,),
            exchange_group=self.exchange_group,
            device=self.device,
        )

    def __del__(
        self,
        _quarantine: dict[int, object] = _ABANDONED_PCIE_RUNTIME_QUARANTINE,
    ) -> None:
        # Never unmap potentially in-flight CUDA work from GC. Retaining the
        # complete object preserves both native ownership and Python IPC maps;
        # explicit close() is the only synchronized teardown path.
        if getattr(self, "_coordinated_close_complete", False):
            return
        if getattr(self, "_ptr", 0) or getattr(self, "_owned_buffers", ()):
            _quarantine[id(self)] = self


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
        max_concurrent_channels: int = 1,
        channel_factory: Optional[Callable[[Optional[int]], PCIeDCPA2A]] = None,
    ) -> None:
        def normalize_and_validate():
            normalized_rank = int(rank)
            normalized_world_size = int(world_size)
            device_obj = _normalize_device(device)
            normalized_max_batch = int(max_batch_size)
            normalized_total_heads = int(total_heads)
            normalized_head_dim = int(head_dim)
            normalized_query_dim = int(
                head_dim if query_head_dim is None else query_head_dim
            )
            normalized_single_channel = bool(single_channel)
            normalized_max_concurrent_channels = int(max_concurrent_channels)
            if normalized_world_size not in SUPPORTED_WORLD_SIZES:
                raise ValueError(f"unsupported world size {normalized_world_size}")
            if not 0 <= normalized_rank < normalized_world_size:
                raise ValueError(
                    f"invalid rank {normalized_rank} for world size "
                    f"{normalized_world_size}"
                )
            if normalized_max_batch <= 0:
                raise ValueError("max_batch_size must be positive")
            if (
                normalized_total_heads <= 0
                or normalized_total_heads % normalized_world_size != 0
            ):
                raise ValueError("total_heads must be divisible by world_size")
            if normalized_head_dim <= 0 or normalized_head_dim % 8 != 0:
                raise ValueError("head_dim must be a positive multiple of 8")
            if normalized_query_dim <= 0 or normalized_query_dim % 8 != 0:
                raise ValueError("query_head_dim must be a positive multiple of 8")
            if normalized_max_concurrent_channels <= 0:
                raise ValueError("max_concurrent_channels must be positive")
            if channel_factory is None:
                if exchange_group is None:
                    raise ValueError(
                        "exchange_group is required unless channel_factory is set"
                    )
                if device_obj.type != "cuda":
                    raise ValueError("PCIe DCP A2A pool requires a CUDA device")
                group_rank = dist.get_rank(group=exchange_group)
                group_world_size = dist.get_world_size(group=exchange_group)
                if normalized_rank != group_rank:
                    raise ValueError(
                        f"supplied rank {normalized_rank} does not match process "
                        f"group rank {group_rank}"
                    )
                if normalized_world_size != group_world_size:
                    raise ValueError(
                        f"supplied world size {normalized_world_size} does not "
                        f"match process group size {group_world_size}"
                    )
            return (
                normalized_rank,
                normalized_world_size,
                device_obj,
                normalized_max_batch,
                normalized_total_heads,
                normalized_head_dim,
                normalized_query_dim,
                normalized_single_channel,
                normalized_max_concurrent_channels,
            )

        if channel_factory is None and exchange_group is not None:
            normalized = _run_collective_preallocation_setup(
                owner="PCIe DCP A2A pool argument validation",
                exchange_group=exchange_group,
                setup=normalize_and_validate,
            )
        else:
            normalized = normalize_and_validate()
        (
            self.rank,
            self.world_size,
            self.device,
            self.max_batch_size,
            self.total_heads,
            self.head_dim,
            self.query_head_dim,
            self.single_channel,
            self.max_concurrent_channels,
        ) = normalized
        self.exchange_group = exchange_group
        self._legacy_ext_module = ext_module
        self.single_channel = bool(single_channel)
        self._channel_factory = channel_factory
        self._channels: dict[int, PCIeDCPA2A] = {}
        self._logical_channels: dict[str, PCIeDCPA2A] = {}
        self._captured_channel_ids: set[str] = set()
        self._all_channels: list[PCIeDCPA2A] = []
        self._capture_channel_stack: list[PCIeDCPA2A] = []
        self._closed = False
        if channel_factory is None:
            assert self.exchange_group is not None
            _require_collective_contract(
                owner="PCIe DCP A2A pool overlap contract",
                exchange_group=self.exchange_group,
                contract=self.max_concurrent_channels,
            )
            _require_full_grid_residency(
                owner="PCIe DCP A2A pool",
                required_sms=(DCP_A2A_REQUIRED_SMS * self.max_concurrent_channels),
                device=self.device,
                exchange_group=self.exchange_group,
            )
        if channel_factory is None and self.single_channel:
            self.prepare_channels((_SINGLE_CHANNEL_ID,))

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
        max_concurrent_channels: int = 1,
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
            max_concurrent_channels=max_concurrent_channels,
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
        max_concurrent_channels: int = 1,
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
            max_concurrent_channels=max_concurrent_channels,
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
                ext_module=self._legacy_ext_module,
                stream_affine=not self.single_channel,
            )
        channel._bind_stream_key(stream_key)
        self._all_channels.append(channel)
        return channel

    def prepare_channels(self, channel_ids: Sequence[str]) -> None:
        """Collectively allocate globally named channels in canonical order."""

        if self._channel_factory is None:
            assert self.exchange_group is not None

            def normalize_and_validate() -> tuple[str, ...]:
                if self._closed:
                    raise RuntimeError("PCIeDCPA2APool is closed")
                return tuple(
                    sorted(
                        {_normalize_logical_channel_id(value) for value in channel_ids}
                    )
                )

            normalized = _run_collective_preallocation_setup(
                owner="PCIe DCP A2A logical channel validation",
                exchange_group=self.exchange_group,
                setup=normalize_and_validate,
            )
            local_state = (normalized, tuple(sorted(self._logical_channels)))
            gathered = _broadcast_gather_object(local_state, self.exchange_group)
            if any(state != local_state for state in gathered):
                raise RuntimeError(
                    "PCIe DCP A2A logical channel preparation differs across "
                    f"ranks: {gathered}"
                )
        else:
            if self._closed:
                raise RuntimeError("PCIeDCPA2APool is closed")
            normalized = tuple(
                sorted({_normalize_logical_channel_id(value) for value in channel_ids})
            )

        if not normalized:
            return

        for channel_id in normalized:
            if channel_id in self._logical_channels:
                continue
            self._logical_channels[channel_id] = self._new_channel(None)

    def checkpoint_channels(
        self,
    ) -> tuple[
        int,
        dict[int, PCIeDCPA2A],
        dict[str, PCIeDCPA2A],
        set[str],
    ]:
        """Snapshot channel ownership before a throwaway graph capture."""
        if self._closed:
            raise RuntimeError("PCIeDCPA2APool is closed")
        if self._capture_channel_stack:
            raise RuntimeError("cannot checkpoint channels during capture")
        return (
            len(self._all_channels),
            dict(self._channels),
            dict(self._logical_channels),
            set(self._captured_channel_ids),
        )

    def rollback_channels(
        self,
        checkpoint: tuple,
    ) -> None:
        """Close channels created after ``checkpoint`` and restore mappings.

        Callers must destroy and synchronize any graphs that reference the
        transient channels before rolling back.
        """
        if self._closed:
            raise RuntimeError("PCIeDCPA2APool is closed")
        if self._capture_channel_stack:
            raise RuntimeError("cannot roll back channels during capture")
        if len(checkpoint) == 2:
            all_channels_len, channels = checkpoint
            logical_channels = dict(self._logical_channels)
            captured_channel_ids = set(self._captured_channel_ids)
        elif len(checkpoint) == 4:
            (
                all_channels_len,
                channels,
                logical_channels,
                captured_channel_ids,
            ) = checkpoint
        else:
            raise ValueError("invalid channel checkpoint")
        if not 0 <= all_channels_len <= len(self._all_channels):
            raise ValueError("channel checkpoint does not belong to this pool")

        retained = self._all_channels[:all_channels_len]
        retained_ids = {id(channel) for channel in retained}
        transient = self._all_channels[all_channels_len:]

        channels_to_close = tuple(
            dict.fromkeys(
                channel for channel in transient if id(channel) not in retained_ids
            )
        )
        _coordinated_close_channels(
            channels_to_close,
            exchange_group=self.exchange_group,
            device=self.device,
        )
        # Ownership changes only after coordinated teardown succeeds.  A
        # failed unmap/free remains reachable for an explicit retry.
        self._all_channels = retained
        self._channels = dict(channels)
        self._logical_channels = dict(logical_channels)
        self._captured_channel_ids = set(captured_channel_ids)

    def for_stream(
        self,
        stream: object = None,
        *,
        channel_id: Optional[str] = None,
    ) -> PCIeDCPA2A:
        if self._closed:
            raise RuntimeError("PCIeDCPA2APool is closed")
        capturing = _is_current_stream_capturing(self.device)
        if capturing:
            if not self._capture_channel_stack:
                raise RuntimeError(
                    "PCIe DCP A2A CUDA graph capture requires an active "
                    "pool.capture() graph-owned channel context"
                )
            channel = self._capture_channel_stack[-1]
            if not self.single_channel:
                stream_key = _current_stream_key(self.device, stream)
                key = 0 if stream_key is None else int(stream_key)
                self._channels[key] = channel
            return channel
        if self.single_channel:
            key = 0
            stream_key = None
            if self._channel_factory is None and channel_id is None:
                channel_id = _SINGLE_CHANNEL_ID
        else:
            stream_key = _current_stream_key(self.device, stream)
            key = 0 if stream_key is None else int(stream_key)
        if not self.single_channel and self._capture_channel_stack:
            # The semantic capture scope also owns vLLM's eager pre-capture
            # warmup. During actual CUDA capture torch may use an ephemeral
            # nested stream key, which must remain only a temporary alias;
            # outside CUDA capture retain the normal stream-affinity check.
            channel = self._capture_channel_stack[-1]
            if not _is_current_stream_capturing(self.device):
                channel._bind_stream_key(stream_key)
            self._channels[key] = channel
            return channel
        if self._channel_factory is None:
            if channel_id is None:
                raise RuntimeError(
                    "distributed PCIe DCP A2A eager use requires an explicit "
                    "semantic channel_id shared by every rank"
                )
            logical_id = _normalize_logical_channel_id(channel_id)
            channel = self._logical_channels.get(logical_id)
            if channel is None:
                raise RuntimeError(
                    f"logical channel {logical_id!r} is not prepared; call "
                    "prepare_channels() collectively before use"
                )
            mapped = self._channels.get(key)
            if mapped is not None and mapped is not channel:
                raise RuntimeError(
                    f"CUDA stream key {key} is already bound to another logical "
                    "PCIe DCP A2A channel"
                )
            channel._bind_stream_key(stream_key)
            self._channels[key] = channel
            return channel
        channel = self._channels.get(key)
        if channel is not None:
            return channel
        channel = self._new_channel(stream_key)
        self._channels[key] = channel
        return channel


    def lse_reduce_scatter(self, partial_output, partial_lse, out=None, *, plan: Plan,
                           is_lse_base_on_e=True, threads=256, block_limit=16,
                           stream=None, channel_id=None):
        channel = self.for_stream(stream, channel_id=channel_id)
        channel._prepared_state(plan)
        with _device_guard(self.device):
            if stream is not None and self.device.type == "cuda":
                with torch.cuda.stream(stream):
                    return channel.lse_reduce_scatter(partial_output, partial_lse, out, plan=plan, is_lse_base_on_e=is_lse_base_on_e, threads=threads, block_limit=block_limit)
            return channel.lse_reduce_scatter(partial_output, partial_lse, out, plan=plan, is_lse_base_on_e=is_lse_base_on_e, threads=threads, block_limit=block_limit)

    def all_gather_heads(self, local_input, out=None, *, plan: Plan, threads=256,
                         block_limit=16, stream=None, channel_id=None):
        channel = self.for_stream(stream, channel_id=channel_id)
        channel._prepared_state(plan)
        with _device_guard(self.device):
            if stream is not None and self.device.type == "cuda":
                with torch.cuda.stream(stream):
                    return channel.all_gather_heads(local_input, out, plan=plan, threads=threads, block_limit=block_limit)
            return channel.all_gather_heads(local_input, out, plan=plan, threads=threads, block_limit=block_limit)

    def all_gather_pair(self, local_first, local_second, out_first=None, out_second=None,
                        *, plan: Plan, threads=512, stream=None, channel_id=None):
        channel = self.for_stream(stream, channel_id=channel_id)
        channel._prepared_state(plan)
        with _device_guard(self.device):
            if stream is not None and self.device.type == "cuda":
                with torch.cuda.stream(stream):
                    return channel.all_gather_pair(local_first, local_second, out_first, out_second, plan=plan, threads=threads)
            return channel.all_gather_pair(local_first, local_second, out_first, out_second, plan=plan, threads=threads)

    def all_gather_pair_kimi_topk(self, local_down, local_router, correction_bias,
                                  out_down=None, topk_weights=None, topk_ids=None,
                                  *, plan: Plan, stream=None, channel_id=None):
        channel = self.for_stream(stream, channel_id=channel_id)
        channel._prepared_state(plan)
        with _device_guard(self.device):
            if stream is not None and self.device.type == "cuda":
                with torch.cuda.stream(stream):
                    return channel.all_gather_pair_kimi_topk(local_down, local_router, correction_bias, out_down, topk_weights, topk_ids, plan=plan)
            return channel.all_gather_pair_kimi_topk(local_down, local_router, correction_bias, out_down, topk_weights, topk_ids, plan=plan)

    def kimi_topk16(self, router_logits, correction_bias, output_weights=None,
                    output_ids=None, *, plan: Plan, threads=256, stream=None,
                    channel_id=None):
        channel = self.for_stream(stream, channel_id=channel_id)
        channel._prepared_state(plan)
        with _device_guard(self.device):
            if stream is not None and self.device.type == "cuda":
                with torch.cuda.stream(stream):
                    return channel.kimi_topk16(router_logits, correction_bias, output_weights, output_ids, plan=plan, threads=threads)
            return channel.kimi_topk16(router_logits, correction_bias, output_weights, output_ids, plan=plan, threads=threads)

    @contextmanager
    def capture(self, stream: object = None, *, channel_id: Optional[str] = None):
        """Bind capture to a globally named channel.

        Explicit unknown ids are prepared collectively and fail closed when
        rank ids differ. Production callers should pre-prepare the full set of
        independently replayable graph ids before entering any capture; after
        that collective preparation, ranks may capture members of the agreed
        catalog in different orders. Each graph must still replay with its
        same-id peers using collectively compatible kernel sequences and
        shapes.

        Distributed pools require this semantic id. Local ordinals and stream
        handles cannot identify target versus draft graphs across ranks.
        """
        previous_channels: Optional[dict[int, PCIeDCPA2A]] = None
        if not self.single_channel and _is_current_stream_capturing(self.device):
            raise RuntimeError(
                "PCIe DCP A2A capture context must be entered before CUDA graph "
                "capture starts"
            )
        if self.single_channel:
            channel = self.for_stream(
                stream,
                channel_id=_SINGLE_CHANNEL_ID if channel_id is None else channel_id,
            )
        elif self._channel_factory is None:
            assert self.exchange_group is not None

            def validate_capture_id() -> str:
                if channel_id is None:
                    raise RuntimeError(
                        "distributed PCIe DCP A2A capture requires a stable "
                        "semantic channel_id shared by every rank"
                    )
                logical_id = _normalize_logical_channel_id(channel_id)
                if logical_id in self._captured_channel_ids:
                    raise RuntimeError(
                        f"logical channel {logical_id!r} was already captured; "
                        "each independently replayable graph requires a unique id"
                    )
                return logical_id

            logical_id = _run_collective_preallocation_setup(
                owner="PCIe DCP A2A capture channel validation",
                exchange_group=self.exchange_group,
                setup=validate_capture_id,
            )
            needs_preparation = _collective_capture_needs_preparation(
                owner="PCIe DCP A2A",
                logical_id=logical_id,
                prepared_channel_ids=self._logical_channels,
                exchange_group=self.exchange_group,
            )
            if needs_preparation:
                self.prepare_channels((logical_id,))
            previous_channels = dict(self._channels)
            stream_key = _current_stream_key(self.device, stream)
            key = 0 if stream_key is None else int(stream_key)
            channel = self._logical_channels[logical_id]
            channel._bind_stream_key(stream_key)
            self._channels[key] = channel
            self._captured_channel_ids.add(logical_id)
        else:
            previous_channels = dict(self._channels)
            stream_key = _current_stream_key(self.device, stream)
            key = 0 if stream_key is None else int(stream_key)
            # Keep a graph-owned channel even when CUDA recycles the enclosing
            # stream handle used by a previously captured graph manager.
            channel = self._new_channel(stream_key)
            self._channels[key] = channel
        self._capture_channel_stack.append(channel)
        try:
            yield channel
        finally:
            popped = self._capture_channel_stack.pop()
            if popped is not channel:
                raise RuntimeError("PCIe DCP A2A capture channel stack corrupted")
            if previous_channels is not None:
                # Captured graph nodes retain the channel through
                # ``_all_channels``. Restore eager mappings and discard every
                # nested capture-stream alias so a recycled stream key cannot
                # hand this graph-owned channel to a later unwrapped capture.
                for key, mapped in tuple(self._channels.items()):
                    if mapped is not channel:
                        continue
                    previous = previous_channels.get(key)
                    if previous is None:
                        del self._channels[key]
                    else:
                        self._channels[key] = previous

    def close(self) -> None:
        if self._closed:
            return
        seen: set[int] = set()
        channels = []
        for channel in (*self._all_channels, *self._channels.values()):
            if id(channel) not in seen:
                seen.add(id(channel))
                channels.append(channel)
        _coordinated_close_channels(
            channels,
            exchange_group=self.exchange_group,
            device=self.device,
        )
        self._closed = True
        self._all_channels.clear()
        self._channels.clear()
        self._logical_channels.clear()
        self._captured_channel_ids.clear()

    def __del__(
        self,
        _quarantine: dict[int, object] = _ABANDONED_PCIE_RUNTIME_QUARANTINE,
    ) -> None:
        # Graphs may retain these channels after Python ownership disappears.
        # Explicit close() is required for synchronization and peer teardown.
        if not getattr(self, "_closed", True):
            _quarantine[id(self)] = self


__all__ = [
    "PCIeDCPA2A",
    "PCIeDCPA2APool",
    "SUPPORTED_WORLD_SIZES",
    "kimi_topk16",
    "lse_reduce_scatter_reference",
]
