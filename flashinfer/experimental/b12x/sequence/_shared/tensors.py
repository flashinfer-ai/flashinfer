"""Host-side tensor contract checks shared by the sequence ops."""

from __future__ import annotations

import torch


def canonical_device(device: torch.device | str) -> torch.device:
    resolved = torch.device(device)
    if resolved.type == "cuda" and resolved.index is None:
        resolved = torch.device("cuda", torch.cuda.current_device())
    return resolved


def positive(name: str, value: int) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def require_tensor(
    name: str,
    tensor: torch.Tensor,
    *,
    shape: tuple[int, ...],
    device: torch.device,
    dtypes: tuple[torch.dtype, ...],
    contiguous: bool = True,
) -> None:
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
    if tensor.dtype not in dtypes:
        expected = " or ".join(str(dtype) for dtype in dtypes)
        raise TypeError(f"{name} must have dtype {expected}, got {tensor.dtype}")
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    if contiguous and not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def require_row_contiguous(
    name: str,
    tensor: torch.Tensor,
    *,
    shape: tuple[int, ...],
    device: torch.device,
    dtypes: tuple[torch.dtype, ...],
) -> None:
    """Require contiguous rows whose outer stride may exceed the row size."""
    require_tensor(name, tensor, shape=shape, device=device, dtypes=dtypes, contiguous=False)
    expected_inner_strides = []
    stride = 1
    for size in reversed(shape[1:]):
        expected_inner_strides.append(stride)
        stride *= size
    expected_inner = tuple(reversed(expected_inner_strides))
    if tuple(tensor.stride()[1:]) != expected_inner:
        raise ValueError(
            f"{name} must be contiguous within each token row; expected inner "
            f"strides {expected_inner}, got {tuple(tensor.stride()[1:])}"
        )
    if tensor.stride(0) < stride:
        raise ValueError(
            f"{name} token rows must not overlap; expected outer stride at "
            f"least {stride}, got {tensor.stride(0)}"
        )


def require_paged_recurrent_state(
    tensor: torch.Tensor,
    *,
    shape: tuple[int, int, int, int],
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """Require ``[slot, head, value_dim, key_dim]`` slots that may be padded."""
    require_tensor(
        "recurrent_state", tensor, shape=shape, device=device, dtypes=(dtype,), contiguous=False
    )
    _, heads, value_dim, key_dim = shape
    slot_elements = heads * value_dim * key_dim
    expected_inner_strides = (value_dim * key_dim, key_dim, 1)
    if tuple(tensor.stride()[1:]) != expected_inner_strides:
        raise ValueError(
            "recurrent_state must be contiguous within each state slot; "
            f"expected inner strides {expected_inner_strides}, got "
            f"{tuple(tensor.stride()[1:])}"
        )
    if tensor.stride(0) < slot_elements:
        raise ValueError(
            "recurrent_state slots must not overlap; expected outer stride at "
            f"least {slot_elements}, got {tensor.stride(0)}"
        )


def byte_interval(tensor: torch.Tensor) -> tuple[int, int]:
    element_size = int(tensor.element_size())
    min_element = max_element = int(tensor.storage_offset())
    for size, stride in zip(tensor.shape, tensor.stride(), strict=True):
        if size == 0:
            storage = int(tensor.untyped_storage().data_ptr())
            start = storage + min_element * element_size
            return start, start
        extent = (int(size) - 1) * int(stride)
        min_element += min(0, extent)
        max_element += max(0, extent)
    storage = int(tensor.untyped_storage().data_ptr())
    return (
        storage + min_element * element_size,
        storage + (max_element + 1) * element_size,
    )


def overlaps(left: torch.Tensor, right: torch.Tensor) -> bool:
    left_start, left_end = byte_interval(left)
    right_start, right_end = byte_interval(right)
    return left_start < right_end and right_start < left_end


__all__ = [
    "byte_interval",
    "canonical_device",
    "overlaps",
    "positive",
    "require_paged_recurrent_state",
    "require_row_contiguous",
    "require_tensor",
]
