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
"""Tests for prefill graph compact non-split metadata packing kernel."""

from __future__ import annotations

import pytest
import torch
import triton

from flashinfer.experimental.b12x.attention.paged.graph_replay import (
    _PREFILL_BLOCK_WORK_ITEMS,
    update_prefill_graph_compact_nonsplit_work_metadata_triton,
)


@pytest.mark.parametrize("batch", [1, 4, 16, 64, 128, 256, 512, 1024])
def test_prefill_graph_compact_metadata_dynamic_batch(batch: int) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA device not available")

    dev = torch.device("cuda")
    cta_tile_q = 128
    gqa_group_size = 2
    page_size = 128
    window_left = -1
    cap = max(512, batch * 4)

    cu_seqlens_q = torch.arange(batch + 1, dtype=torch.int32, device=dev) * 32
    cache_seqlens = torch.full((batch,), 128, dtype=torch.int32, device=dev)
    request_indices = torch.zeros(cap, dtype=torch.int32, device=dev)
    qo_tile_indices = torch.zeros(cap, dtype=torch.int32, device=dev)
    kv_tile_indices = torch.zeros(cap, dtype=torch.int32, device=dev)
    block_valid_mask = torch.zeros(cap, dtype=torch.int32, device=dev)
    kv_window_start_tokens = torch.zeros(batch, dtype=torch.int32, device=dev)

    grid = (triton.cdiv(cap, _PREFILL_BLOCK_WORK_ITEMS),)
    update_prefill_graph_compact_nonsplit_work_metadata_triton[grid](
        cache_seqlens,
        cu_seqlens_q,
        request_indices,
        qo_tile_indices,
        kv_tile_indices,
        block_valid_mask,
        kv_window_start_tokens,
        cap,
        cap,
        BATCH=batch,
        CTA_TILE_Q=cta_tile_q,
        GQA_GROUP_SIZE=gqa_group_size,
        PAGE_SIZE=page_size,
        WINDOW_LEFT=window_left,
        BLOCK_WORK_ITEMS=_PREFILL_BLOCK_WORK_ITEMS,
    )
    torch.cuda.synchronize(dev)

    assert (block_valid_mask[:batch] == 1).all()
    assert torch.equal(
        request_indices[:batch],
        torch.arange(batch, dtype=torch.int32, device=dev),
    )
    assert (kv_window_start_tokens >= 0).all()
