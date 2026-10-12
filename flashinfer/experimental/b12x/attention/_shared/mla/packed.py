"""Packed sparse-MLA cache view helpers shared by active runtime code."""

from __future__ import annotations

import torch

from .reference import _MLA_NOPE_DIM, _MLA_SCALE_BYTES


def view_last_dim_as_u32(tensor: torch.Tensor) -> torch.Tensor:
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    byte_width = tensor.shape[-1] * tensor.element_size()
    if byte_width % 4 != 0:
        raise ValueError(
            f"last dimension byte-width must be divisible by 4, got {byte_width}"
        )
    byte_view = tensor.view(torch.uint8).reshape(*tensor.shape[:-1], byte_width)
    return byte_view.view(torch.uint32).reshape(*tensor.shape[:-1], byte_width // 4)


def extract_packed_kv_runtime_views(
    kv_cache: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    kv_rows_bytes = kv_cache[:, 0, :].view(torch.uint8)
    kv_rows_u32 = view_last_dim_as_u32(kv_rows_bytes)
    kv_scales = kv_rows_bytes[:, _MLA_NOPE_DIM : _MLA_NOPE_DIM + _MLA_SCALE_BYTES].view(
        torch.float32
    )
    return kv_rows_u32, kv_scales


# Backward-compatible private spelling for internal callers that only need the
# layout helper, not the retired kernel module.
_extract_packed_kv_runtime_views = extract_packed_kv_runtime_views


__all__ = [
    "_extract_packed_kv_runtime_views",
    "extract_packed_kv_runtime_views",
    "view_last_dim_as_u32",
]
