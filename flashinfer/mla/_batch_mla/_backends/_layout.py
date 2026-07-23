"""Private layout helpers shared by batch MLA backends with combined inputs."""

from __future__ import annotations

import torch


def _view_fits_storage(
    shape: tuple[int, ...],
    stride: tuple[int, ...],
    storage_offset: int,
    storage_numel: int,
) -> bool:
    """Return whether a nonnegative-stride view stays inside its storage."""
    if storage_offset < 0 or storage_numel < 0 or len(shape) != len(stride):
        return False
    if any(size < 0 for size in shape) or any(step < 0 for step in stride):
        return False
    if any(size == 0 for size in shape):
        return storage_offset <= storage_numel
    last_offset = storage_offset
    for size, step in zip(shape, stride, strict=True):
        last_offset += (size - 1) * step
    return last_offset < storage_numel


def _concat_adjacent_views_or_cat(
    left: torch.Tensor, right: torch.Tensor
) -> torch.Tensor:
    """Join compatible adjacent last-dimension views without copying.

    The fast path is metadata-only and deliberately narrow: it accepts two
    views of the same storage only when their common strides and offsets form
    one in-bounds contiguous combined tensor. All other inputs retain the
    allocating ``torch.cat`` behavior.
    """
    if (
        left.ndim == 0
        or left.ndim != right.ndim
        or left.shape[:-1] != right.shape[:-1]
        or left.dtype != right.dtype
        or left.device != right.device
        or left.stride() != right.stride()
        or left.shape[-1] == 0
        or right.shape[-1] == 0
    ):
        return torch.cat((left, right), dim=-1)

    left_storage = left.untyped_storage()
    right_storage = right.untyped_storage()
    if left_storage._cdata != right_storage._cdata:
        return torch.cat((left, right), dim=-1)

    stride = left.stride()
    storage_offset = left.storage_offset()
    if right.storage_offset() != storage_offset + left.shape[-1] * stride[-1]:
        return torch.cat((left, right), dim=-1)

    shape = left.shape[:-1] + (left.shape[-1] + right.shape[-1],)
    storage_nbytes = left_storage.nbytes()
    element_size = left.element_size()
    if storage_nbytes % element_size != 0 or not _view_fits_storage(
        shape, stride, storage_offset, storage_nbytes // element_size
    ):
        return torch.cat((left, right), dim=-1)

    combined = left.as_strided(shape, stride, storage_offset)
    if not combined.is_contiguous():
        return torch.cat((left, right), dim=-1)
    return combined
