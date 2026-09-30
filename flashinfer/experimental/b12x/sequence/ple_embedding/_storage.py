"""Persistent PLE table storage allocation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from .._shared.disk_table import MappedHostAllocation

if TYPE_CHECKING:
    from ._contracts import TableLayout


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
    _mapped_allocations: tuple[MappedHostAllocation, ...]

    def close(self) -> None:
        """Synchronize and release any mapped-host allocations."""
        for allocation in reversed(self._mapped_allocations):
            allocation.close()


def _device_tensor(
    shape: tuple[int, ...], dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    return torch.empty(shape, dtype=dtype, device=device)


def allocate_storage(layout: TableLayout) -> TableStorage:
    """Allocate checkpoint-owned tensors from a host-only storage layout."""
    from ._contracts import TableLayout

    if not isinstance(layout, TableLayout):
        raise TypeError(f"layout must be TableLayout, got {type(layout)!r}")
    caps = layout.caps
    if caps.table_memory == "io_uring":
        raise ValueError("disk tables must be loaded with DiskTable(layout, shard_rows)")

    allocations: list[MappedHostAllocation] = []

    def table_tensor(
        shape: tuple[int, ...], dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if caps.table_memory == "device":
            tensor = _device_tensor(shape, dtype, caps.device)
            return tensor, tensor
        allocation = MappedHostAllocation(shape, dtype, caps.device)
        allocations.append(allocation)
        return allocation.device_view, allocation.host_view

    weight, weight_load_view = table_tensor(layout.weight_shape, layout.weight_dtype)

    weight_scale: torch.Tensor | None = None
    weight_scale_load_view: torch.Tensor | None = None
    if layout.weight_scale_shape is not None:
        assert layout.weight_scale_dtype is not None
        if caps.quant_mode == "nvfp4_group16":
            weight_scale, weight_scale_load_view = table_tensor(
                layout.weight_scale_shape, layout.weight_scale_dtype
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


__all__ = ["TableStorage", "allocate_storage"]
