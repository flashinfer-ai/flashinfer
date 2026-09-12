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

"""Index-width selection shared by cuTile MoE kernels."""

from __future__ import annotations

import torch


INT32_INDEX_LIMIT = 1 << 31


def needs_int64_indexing(*tensors: torch.Tensor) -> bool:
    """Return whether any array cannot be addressed by signed int32 offsets."""
    return any(tensor.numel() >= INT32_INDEX_LIMIT for tensor in tensors)


__all__ = ["INT32_INDEX_LIMIT", "needs_int64_indexing"]
