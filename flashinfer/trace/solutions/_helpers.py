# Copyright (c) 2025 by FlashInfer team.
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

"""Small helpers shared by explicit trace solution modules."""

from __future__ import annotations

import torch

_WORKSPACE_SIZE_BYTES = 128 * 1024 * 1024


def default_paged_metadata(batch_size: int, num_pages: int, device):
    pages_per_seq = max(1, num_pages // max(1, batch_size))
    indptr = (
        torch.arange(batch_size + 1, dtype=torch.int32, device=device) * pages_per_seq
    )
    indices = torch.arange(int(indptr[-1].item()), dtype=torch.int32, device=device)
    return indptr, indices


def full_last_page_len(kv_indptr, page_size: int):
    return torch.full(
        (kv_indptr.numel() - 1,),
        page_size,
        dtype=torch.int32,
        device=kv_indptr.device,
    )


def workspace(device):
    return torch.empty(_WORKSPACE_SIZE_BYTES, dtype=torch.uint8, device=device)
