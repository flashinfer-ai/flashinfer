"""Input layout accepted by the kernels, and the view they run on.

The kernels read rows with 16-byte vector loads, so the layout they run on has every row
starting on a 16-byte boundary: the row stride is a multiple of 16 bytes and the base is
16-byte aligned.  The row length itself may be anything (lengths are arbitrary already; the
row's tail past its last whole vector is read by the scalar tail paths).  The stride may exceed
the row length (a paged logits arena, a column-sliced view): the kernels compile the row
length ``n`` and a column offset separately from the tensor's second dimension, which carries
the stride, so a strided input is passed as the compact ``(rows, stride)`` view over its
storage and each row is read from ``offset`` for ``n`` columns.  Stride and offset are part of
the compiled specialization, like ``n``.

Inputs whose rows are not 16-byte aligned (an odd row length in a contiguous tensor, a slice
at an odd column, a non-unit inner stride) are copied once into a padded arena whose rows are.
That copy is one extra pass over the batch; it keeps the contract total without a scalar-load
variant of every kernel for a layout no model shape produces.
"""

from __future__ import annotations

import torch

__all__ = ["check_layout", "arena_view", "arena_bytes"]

ALIGN = 16


def check_layout(x: torch.Tensor) -> None:
    """Raise unless ``x`` is a 2-D CUDA tensor (any layout the kernels can take or copy)."""
    if x.dim() != 2 or not x.is_cuda:
        raise ValueError("expected a 2-D CUDA tensor")


def _aligned(x: torch.Tensor) -> bool:
    """Whether the kernels can read ``x`` in place: unit inner stride, 16-byte-aligned base and
    row stride, rows not overlapping."""
    esize = x.element_size()
    if x.stride(1) != 1 or x.data_ptr() % ALIGN:
        return False
    if x.shape[0] > 1 and ((x.stride(0) * esize) % ALIGN or x.stride(0) < x.shape[1]):
        return False
    return True


def _storage_view(x: torch.Tensor) -> tuple[torch.Tensor, int] | None:
    """The compact ``(rows, stride)`` view over ``x``'s storage and ``x``'s column offset in it,
    when the storage holds ``rows x stride`` elements from the row-aligned base; else None."""
    rows = x.shape[0]
    stride = x.stride(0)
    base = x.storage_offset()
    row_base = (
        base - base % stride
    )  # the storage element where x's first row's stride period begins
    storage_numel = x.untyped_storage().nbytes() // x.element_size()
    if row_base + rows * stride > storage_numel:
        return None
    view = torch.empty(0, dtype=x.dtype, device=x.device).set_(
        x.untyped_storage(), row_base, (rows, stride), (stride, 1)
    )
    return view, base - row_base


def _padded_stride(x: torch.Tensor) -> int:
    """The row length rounded up to whole 16-byte vectors: the arena's row stride."""
    per_align = ALIGN // x.element_size()
    return -(-x.shape[1] // per_align) * per_align


def arena_bytes(x: torch.Tensor) -> int:
    """Bytes of the padded arena :func:`arena_view` copies ``x`` into; 0 when ``x`` is read in
    place (aligned rows over a storage that holds the compact view)."""
    if _aligned(x) and (x.is_contiguous() or _storage_view(x) is not None):
        return 0
    return x.shape[0] * _padded_stride(x) * x.element_size()


def arena_view(
    x: torch.Tensor, buffer: torch.Tensor | None = None
) -> tuple[torch.Tensor, int]:
    """``(view, offset)``: the compact ``(rows, stride)`` tensor the kernels take and the column
    at which each row starts in it.  ``x`` itself with offset 0 when contiguous and aligned.
    A view over the storage is used when the storage holds ``rows x stride`` elements from the
    view's row-aligned base; otherwise, and whenever the rows are not 16-byte aligned, the rows
    are copied into a padded arena (correct, one extra pass): a fresh allocation, or the first
    :func:`arena_bytes` bytes of ``buffer`` (a uint8 tensor, 16-byte-aligned) when given."""
    rows, n = x.shape
    if _aligned(x):
        if x.is_contiguous():
            return x, 0
        found = _storage_view(x)
        if found is not None:
            return found
    stride = _padded_stride(x)
    if buffer is None:
        arena = torch.empty(rows, stride, dtype=x.dtype, device=x.device)
    else:
        nbytes = rows * stride * x.element_size()
        if (
            buffer.dtype != torch.uint8
            or buffer.numel() < nbytes
            or buffer.data_ptr() % ALIGN
        ):
            raise ValueError(
                f"arena buffer must be an aligned uint8 tensor of at least {nbytes} bytes"
            )
        arena = buffer[:nbytes].view(x.dtype).view(rows, stride)
    arena[:, :n].copy_(x)
    return arena, 0
