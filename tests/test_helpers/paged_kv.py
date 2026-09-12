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

import torch


def make_padded_paged_kv_view(
    source: torch.Tensor,
    kv_layout: str,
    *,
    padding_heads: int = 2,
) -> torch.Tensor:
    """Copy paged KV into an equal-shaped view with different outer strides."""
    head_dimension = {"NHD": 2, "HND": 1}[kv_layout]
    shape = list(source.shape)
    shape[head_dimension] += padding_heads
    view = torch.empty(shape, dtype=source.dtype, device=source.device).narrow(
        head_dimension, 0, source.shape[head_dimension]
    )
    view.copy_(source)
    assert view.stride() != source.stride()
    return view
