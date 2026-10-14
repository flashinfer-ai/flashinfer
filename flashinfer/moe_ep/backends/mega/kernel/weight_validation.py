"""Shared helpers for kernel-ready mega weight validation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

from ....core.validation.common import MoEEpConfigError


def _dtype_itemsize(dtype: "torch.dtype") -> int:
    import torch

    return torch.empty((), dtype=dtype).element_size()


def _scale_dtype_matches(actual: "torch.dtype", expected: "torch.dtype") -> bool:
    """True when scale dtypes match or are 1-byte layout aliases (uint8 vs float8_*)."""
    if actual == expected:
        return True
    return _dtype_itemsize(actual) == 1 and _dtype_itemsize(expected) == 1


def check_transformed_mega_weights_structure(transformed: Any) -> None:
    """``((w, sf), (w, sf))`` top-level layout for all mega kernels."""
    if not isinstance(transformed, tuple) or len(transformed) != 2:
        raise MoEEpConfigError(
            f"transformed_weights must be a 2-tuple (fc1, fc2), got {type(transformed).__name__}"
        )
    for idx, layer in enumerate(transformed):
        label = "fc1" if idx == 0 else "fc2"
        if not isinstance(layer, tuple) or len(layer) != 2:
            raise MoEEpConfigError(
                f"transformed_weights {label} must be (weight, scale_factor), "
                f"got {type(layer).__name__}"
            )


def check_transformed_weight_pair(
    pair: tuple[Any, Any],
    *,
    label: str,
    num_local_experts: int,
    weight_dtype: "torch.dtype",
    expected_weight_shape: tuple[int, ...],
    scale_dtype: "torch.dtype",
    expected_scale_shape: tuple[int, ...],
) -> None:
    import torch

    weight, scale = pair
    if not isinstance(weight, torch.Tensor):
        raise MoEEpConfigError(
            f"transformed_weights {label} weight must be a torch.Tensor, "
            f"got {type(weight).__name__}"
        )
    if not isinstance(scale, torch.Tensor):
        raise MoEEpConfigError(
            f"transformed_weights {label} scale must be a torch.Tensor, "
            f"got {type(scale).__name__}"
        )
    if weight.shape != expected_weight_shape:
        raise MoEEpConfigError(
            f"transformed_weights {label} weight must have shape "
            f"{expected_weight_shape}, got {tuple(weight.shape)}"
        )
    if weight.dtype != weight_dtype:
        raise MoEEpConfigError(
            f"transformed_weights {label} weight must have dtype {weight_dtype}, "
            f"got {weight.dtype}"
        )
    if weight.shape[0] != num_local_experts:
        raise MoEEpConfigError(
            f"transformed_weights {label} leading dim ({weight.shape[0]}) must match "
            f"num_experts // world_size ({num_local_experts})"
        )
    if scale.shape != expected_scale_shape:
        raise MoEEpConfigError(
            f"transformed_weights {label} scale must have shape "
            f"{expected_scale_shape}, got {tuple(scale.shape)}"
        )
    if not _scale_dtype_matches(scale.dtype, scale_dtype):
        raise MoEEpConfigError(
            f"transformed_weights {label} scale must have dtype {scale_dtype}, "
            f"got {scale.dtype}"
        )


__all__ = ["check_transformed_mega_weights_structure", "check_transformed_weight_pair"]
