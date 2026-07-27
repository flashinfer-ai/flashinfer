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


def _same_tensor_view(expected: torch.Tensor, actual: torch.Tensor) -> bool:
    """Return whether two tensors describe the same view without reading data."""
    return (
        expected.shape == actual.shape
        and expected.dtype == actual.dtype
        and expected.device == actual.device
        and expected.stride() == actual.stride()
        and expected.storage_offset() == actual.storage_offset()
        and expected.untyped_storage()._cdata == actual.untyped_storage()._cdata
    )


def _split_combined_last_dim(
    tensor: torch.Tensor,
    *,
    first_width: int,
    second_width: int,
    name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a combined last dimension into two zero-copy views."""
    if tensor.ndim == 0:
        raise ValueError(f"{name} must have at least one dimension.")
    if first_width <= 0 or second_width <= 0:
        raise ValueError("split widths must be positive.")
    if tensor.shape[-1] != first_width + second_width:
        raise ValueError(
            f"{name} last dimension must equal the two split widths, got "
            f"{tensor.shape[-1]} for widths {first_width} and {second_width}."
        )
    return tensor[..., :first_width], tensor[..., first_width:]


def _split_combined_kv_cache(
    kv_cache: torch.Tensor, *, ckv_width: int, kpe_width: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a combined KV cache into zero-copy CKV and KPE views."""
    return _split_combined_last_dim(
        kv_cache,
        first_width=ckv_width,
        second_width=kpe_width,
        name="kv_cache",
    )


def _split_combined_query(
    q: torch.Tensor, *, q_nope_width: int, q_pe_width: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a combined query into zero-copy NoPE and PE views."""
    return _split_combined_last_dim(
        q,
        first_width=q_nope_width,
        second_width=q_pe_width,
        name="q",
    )


def _split_views_match_combined(
    kv_cache: torch.Tensor,
    ckv_cache: torch.Tensor,
    kpe_cache: torch.Tensor,
    *,
    ckv_width: int,
    kpe_width: int,
) -> bool:
    """Return whether split tensors are exactly the canonical combined-cache views."""
    expected_ckv, expected_kpe = _split_combined_kv_cache(
        kv_cache, ckv_width=ckv_width, kpe_width=kpe_width
    )
    return _same_tensor_view(expected_ckv, ckv_cache) and _same_tensor_view(
        expected_kpe, kpe_cache
    )


def _split_query_views_match_combined(
    q: torch.Tensor,
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    *,
    q_nope_width: int,
    q_pe_width: int,
) -> bool:
    """Return whether split query tensors are exact canonical combined-Q views."""
    expected_q_nope, expected_q_pe = _split_combined_query(
        q, q_nope_width=q_nope_width, q_pe_width=q_pe_width
    )
    return _same_tensor_view(expected_q_nope, q_nope) and _same_tensor_view(
        expected_q_pe, q_pe
    )


def _adjacent_last_dim_view(
    left: torch.Tensor, right: torch.Tensor
) -> torch.Tensor | None:
    """Return a combined contiguous view when compatible inputs are adjacent."""
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
        return None

    left_storage = left.untyped_storage()
    right_storage = right.untyped_storage()
    if left_storage._cdata != right_storage._cdata:
        return None

    stride = left.stride()
    storage_offset = left.storage_offset()
    if right.storage_offset() != storage_offset + left.shape[-1] * stride[-1]:
        return None

    shape = left.shape[:-1] + (left.shape[-1] + right.shape[-1],)
    storage_nbytes = left_storage.nbytes()
    element_size = left.element_size()
    if storage_nbytes % element_size != 0 or not _view_fits_storage(
        shape, stride, storage_offset, storage_nbytes // element_size
    ):
        return None

    combined = left.as_strided(shape, stride, storage_offset)
    return combined if combined.is_contiguous() else None


def _concat_adjacent_views_or_cat(
    left: torch.Tensor, right: torch.Tensor
) -> torch.Tensor:
    """Join compatible adjacent last-dimension views without copying.

    The fast path is metadata-only and deliberately narrow: it accepts two
    views of the same storage only when their common strides and offsets form
    one in-bounds contiguous combined tensor. All other inputs retain the
    allocating ``torch.cat`` behavior.
    """
    combined = _adjacent_last_dim_view(left, right)
    if combined is None:
        return torch.cat((left, right), dim=-1)
    return combined
