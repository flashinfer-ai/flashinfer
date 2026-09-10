# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Small helpers for constructing deliberate paged-KV stride families."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch


def make_padded_view(
    source: torch.Tensor,
    dimension: int,
    *,
    padding: int = 2,
) -> torch.Tensor:
    """Copy ``source`` into an equal-shaped view with padding in one dimension."""
    if padding <= 0:
        raise ValueError(f"padding must be positive, got {padding}")
    dimension %= source.ndim
    parent_shape = list(source.shape)
    parent_shape[dimension] += padding
    parent = torch.empty(
        parent_shape,
        dtype=source.dtype,
        device=source.device,
    )
    view = parent.narrow(dimension, 0, source.shape[dimension])
    view.copy_(source)
    if view.stride() == source.stride():
        raise AssertionError(
            f"padding dimension {dimension} did not change strides for {source.shape}"
        )
    return view


def make_padded_paged_kv_view(
    source: torch.Tensor,
    kv_layout: str,
    *,
    padding_heads: int = 2,
) -> torch.Tensor:
    """Return an equal-shaped paged-KV view with a padded head dimension."""
    if source.ndim != 4:
        raise ValueError(f"expected rank-4 paged KV, got shape {source.shape}")
    if kv_layout == "NHD":
        head_dimension = 2
    elif kv_layout == "HND":
        head_dimension = 1
    else:
        raise ValueError(f"kv_layout must be 'NHD' or 'HND', got {kv_layout!r}")
    return make_padded_view(source, head_dimension, padding=padding_heads)


def get_closure_value(function: Callable[..., Any], name: str) -> Any:
    """Return a named closure value from a routed test-facing callable."""
    closure = function.__closure__ or ()
    values = dict(zip(function.__code__.co_freevars, closure, strict=True))
    return values[name].cell_contents
