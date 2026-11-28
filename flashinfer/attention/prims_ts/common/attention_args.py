# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Validation of attention arguments shared by the PrimTS host frontends."""

from typing import Literal, Optional, cast

import torch

from flashinfer.utils import _check_workspace_buffer_alignment, check_shape_dtype_device

SUPPORTED_PAGE_SIZES = (4, 8, 16, 32, 64, 128)


def prepare_lse(
    lse: torch.Tensor | None,
    *,
    q: torch.Tensor,
    mode: str,
    return_lse: bool,
    validate: bool,
) -> torch.Tensor | None:
    """Bind an independent caller buffer, or allocate the enabled output."""
    if mode == "none":
        if return_lse or lse is not None:
            raise ValueError("plan() must enable lse_mode='base2'")
        return None
    if lse is None:
        return torch.empty(q.shape[:-1], dtype=torch.float32, device=q.device)
    if validate:
        check_shape_dtype_device(lse, q.shape[:-1], torch.float32, q.device, "lse")
        if not lse.is_contiguous():
            raise ValueError("lse must be contiguous")
        _check_workspace_buffer_alignment(lse, "lse", alignment=4)
    return lse


def resolve_kv_dtypes(
    q_dtype: torch.dtype,
    k_dtype: Optional[torch.dtype],
    v_dtype: Optional[torch.dtype],
    kv_dtype: Optional[torch.dtype],
) -> tuple[torch.dtype, torch.dtype]:
    """Resolve the common-KV compatibility alias without hiding disagreement."""
    if kv_dtype is not None:
        if (k_dtype is not None and k_dtype != kv_dtype) or (
            v_dtype is not None and v_dtype != kv_dtype
        ):
            raise ValueError("explicit K/V dtypes must agree with the common KV dtype")
        k_dtype = v_dtype = kv_dtype
    if k_dtype is None:
        k_dtype = q_dtype
    if v_dtype is None:
        v_dtype = k_dtype
    return k_dtype, v_dtype


def resolve_max_sequence_length(
    value: int | None, actual: int, name: str, capacity: int
) -> int:
    """Infer an omitted launch bound, or check coverage within plan capacity."""

    if value is not None:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if value < actual:
            raise ValueError(f"{name} must cover the maximum input length ({actual})")
        if value > capacity:
            raise ValueError(
                f"{name} must not exceed the planned capacity ({capacity})"
            )
        return value
    return actual


def validate_attention_sinks(attention_sinks, *, q, enabled, validate):
    """Bind sink logits; callers guarantee finite values or negative infinity."""
    if enabled != (attention_sinks is not None):
        raise ValueError(
            "attention_sinks must be supplied exactly when plan(use_attention_sinks=True)"
        )
    if attention_sinks is not None and validate:
        check_shape_dtype_device(
            attention_sinks, (q.shape[-2],), torch.float32, q.device, "attention_sinks"
        )
        if not attention_sinks.is_contiguous():
            raise ValueError("attention_sinks must be contiguous")
        _check_workspace_buffer_alignment(
            attention_sinks, "attention_sinks", alignment=4
        )
    return attention_sinks


def validate_layout(kv_layout: str) -> None:
    if not isinstance(kv_layout, str):
        raise TypeError("kv_layout must be a string")
    if kv_layout not in ("HND", "NHD"):
        raise ValueError(f"kv_layout must be 'HND' or 'NHD', got {kv_layout!r}")


def validate_lse_mode(mode: str) -> Literal["none", "base2"]:
    if mode not in ("none", "base2"):
        raise ValueError("lse_mode must be 'none' or 'base2'")
    return cast(Literal["none", "base2"], mode)


def validate_mask(mask_type: str) -> None:
    if not isinstance(mask_type, str):
        raise TypeError("mask_type must be a string")
    if mask_type not in ("dense", "causal"):
        raise ValueError(
            f"mask_type must be exactly 'dense' or 'causal', got {mask_type!r}"
        )


def validate_page_size(page_size: int) -> int:
    page_size = validate_positive_int(page_size, "page_size")
    if page_size not in SUPPORTED_PAGE_SIZES:
        raise ValueError(
            "attention-ts decode requires page_size in "
            f"{SUPPORTED_PAGE_SIZES}, got {page_size}"
        )
    return page_size


def validate_positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a positive integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value
