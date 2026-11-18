"""Persistent PLE table storage allocation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import torch
from .._shared.disk_table import MappedHostAllocation

if TYPE_CHECKING:
    from ._contracts import TableLayout


class MappedHostRegion(Protocol):
    """Caller-provided CUDA-mapped host bytes for one table plane.

    ``host_view`` is a CPU tensor and ``device_view`` the CUDA tensor over the
    same mapped bytes (equal pointers under unified addressing). ``close``
    releases them; the storage that receives a region owns it from then on.
    """

    host_view: torch.Tensor
    device_view: torch.Tensor
    nbytes: int

    def close(self) -> None: ...


HostAllocator = Callable[[str, tuple[int, ...], torch.dtype], MappedHostRegion]


@dataclass(kw_only=True)
class TableStorage:
    """Owning persistent table tensors and their checkpoint loading views.

    Kernel-visible tensors are always CUDA tensors. For mapped-host table
    storage, a loading view is a CPU tensor over the same page-locked bytes.
    The owner must outlive every binding that references its tensors.
    """

    weight: torch.Tensor
    weight_scale: torch.Tensor | None
    weight_scale_2: torch.Tensor | None
    weight_load_view: torch.Tensor
    weight_scale_load_view: torch.Tensor | None
    weight_scale_2_load_view: torch.Tensor | None
    mapped_host_nbytes: int
    _mapped_allocations: tuple[MappedHostRegion, ...]

    def close(self) -> None:
        """Synchronize and release any mapped-host allocations."""
        allocations, self._mapped_allocations = self._mapped_allocations, ()
        error = None
        for allocation in reversed(allocations):
            try:
                allocation.close()
            except BaseException as failure:
                if error is None:
                    error = failure
                else:
                    error.add_note(f"additional region cleanup failed: {failure!r}")
        if error is not None:
            raise error


def _device_tensor(
    shape: tuple[int, ...], dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    return torch.empty(shape, dtype=dtype, device=device)


def _require_host_region(
    name: str,
    region: MappedHostRegion,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    from ._contracts import _require_mapped_host_tensor

    host_view, device_view = region.host_view, region.device_view
    for view, view_device in ((host_view, torch.device("cpu")), (device_view, device)):
        if tuple(view.shape) != shape or view.dtype != dtype:
            raise ValueError(
                f"host_allocator returned {name} as {tuple(view.shape)} {view.dtype}, "
                f"expected {shape} {dtype}"
            )
        if view.device != view_device or not view.is_contiguous():
            raise ValueError(
                f"host_allocator returned a {name} view on {view.device}; expected "
                f"a contiguous view on {view_device}"
            )
    if host_view.data_ptr() != device_view.data_ptr():
        raise ValueError(f"host_allocator returned {name} views over different bytes")
    _require_mapped_host_tensor(name, device_view, device=device)


def allocate_storage(
    layout: TableLayout, *, host_allocator: HostAllocator | None = None
) -> TableStorage:
    """Allocate checkpoint-owned tensors from a host-only storage layout.

    ``host_allocator(name, shape, dtype)`` supplies the bytes of each
    mapped-host plane (``"weight"`` and, for NVFP4, ``"weight_scale"``) instead
    of a private ``cudaHostAlloc``, for example a file mapping shared with other
    processes. It requires ``table_memory="mapped_host"``.
    """
    from ._contracts import TableLayout

    if not isinstance(layout, TableLayout):
        raise TypeError(f"layout must be TableLayout, got {type(layout)!r}")
    caps = layout.caps
    if caps.table_memory == "io_uring":
        raise ValueError(
            "disk tables must be loaded with DiskTable(layout, shard_rows)"
        )
    if host_allocator is not None and caps.table_memory != "mapped_host":
        raise ValueError(
            "host_allocator requires table_memory='mapped_host', "
            f"got {caps.table_memory!r}"
        )

    allocations: list[MappedHostRegion] = []

    def table_tensor(
        name: str, shape: tuple[int, ...], dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if caps.table_memory == "device":
            tensor = _device_tensor(shape, dtype, caps.device)
            return tensor, tensor
        if host_allocator is None:
            allocation: MappedHostRegion = MappedHostAllocation(
                shape, dtype, caps.device
            )
        else:
            allocation = host_allocator(name, shape, dtype)
        allocations.append(allocation)
        if host_allocator is not None:
            _require_host_region(name, allocation, shape, dtype, caps.device)
        return allocation.device_view, allocation.host_view

    try:
        return _allocate_tables(layout, table_tensor, allocations)
    except BaseException as error:
        for allocation in reversed(allocations):
            try:
                allocation.close()
            except BaseException as cleanup:
                error.add_note(f"region cleanup failed: {cleanup!r}")
        raise


def _allocate_tables(
    layout: TableLayout,
    table_tensor: Callable[
        [str, tuple[int, ...], torch.dtype], tuple[torch.Tensor, torch.Tensor]
    ],
    allocations: list[MappedHostRegion],
) -> TableStorage:
    caps = layout.caps
    weight, weight_load_view = table_tensor(
        "weight", layout.weight_shape, layout.weight_dtype
    )

    weight_scale: torch.Tensor | None = None
    weight_scale_load_view: torch.Tensor | None = None
    if layout.weight_scale_shape is not None:
        assert layout.weight_scale_dtype is not None
        if caps.quant_mode == "nvfp4_group16":
            weight_scale, weight_scale_load_view = table_tensor(
                "weight_scale", layout.weight_scale_shape, layout.weight_scale_dtype
            )
        else:
            weight_scale = _device_tensor(
                layout.weight_scale_shape, layout.weight_scale_dtype, caps.device
            )
            weight_scale_load_view = weight_scale

    weight_scale_2: torch.Tensor | None = None
    weight_scale_2_load_view: torch.Tensor | None = None
    if layout.weight_scale_2_shape is not None:
        assert layout.weight_scale_2_dtype is not None
        weight_scale_2 = _device_tensor(
            layout.weight_scale_2_shape, layout.weight_scale_2_dtype, caps.device
        )
        weight_scale_2_load_view = weight_scale_2

    return TableStorage(
        weight=weight,
        weight_scale=weight_scale,
        weight_scale_2=weight_scale_2,
        weight_load_view=weight_load_view,
        weight_scale_load_view=weight_scale_load_view,
        weight_scale_2_load_view=weight_scale_2_load_view,
        mapped_host_nbytes=sum(allocation.nbytes for allocation in allocations),
        _mapped_allocations=tuple(allocations),
    )


__all__ = ["HostAllocator", "MappedHostRegion", "TableStorage", "allocate_storage"]
