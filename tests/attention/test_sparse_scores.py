"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import pytest
import torch

import flashinfer

DEV = "cuda:0"

requires_cuda_sm80 = pytest.mark.skipif(
    torch.cuda.get_device_capability()[0] < 8,
    reason="the FA2 large-head path starts at sm_80",
)


def _scores_case(
    head_dim=128, num_heads=4, rows=8, pages=6, page_size=8, dtype=torch.bfloat16
):
    """An aligned scorer case, plus what it takes to re-run it from a view."""
    torch.manual_seed(0)
    q = torch.randn(rows, num_heads, head_dim, dtype=dtype, device=DEV)
    k_cache = torch.randn(pages, page_size, head_dim, dtype=dtype, device=DEV)
    page_table = torch.arange(pages, dtype=torch.int32, device=DEV).reshape(1, pages)
    token_to_req = torch.zeros(rows, dtype=torch.int32, device=DEV)
    positions = torch.arange(rows, dtype=torch.int32, device=DEV) + pages * page_size
    seq_lens = torch.full((1,), pages * page_size, dtype=torch.int32, device=DEV)
    return q, k_cache, page_table, token_to_req, positions, seq_lens


def _run_scores(q, k_cache, page_table, token_to_req, positions, seq_lens):
    return flashinfer.sparse_paged_scores(
        q, k_cache, page_table, token_to_req, positions, seq_lens, 1, q.shape[2] ** 0.5
    )


@requires_cuda_sm80
def test_scores_accept_a_query_view_with_a_storage_offset():
    """A view onto the middle of a tensor keeps its strides and loses its
    alignment, which the staged 128-bit loads cannot assume away."""
    q, k_cache, table, t2r, pos, lens = _scores_case()
    want, want_visible = _run_scores(q, k_cache, table, t2r, pos, lens)

    head_dim = q.shape[2]
    # Pad by a whole vector so every stride stays a multiple of it, and take the
    # slice one element in. Padding by one instead would misalign the strides
    # too, and the staging already refuses those -- the base would never be
    # reached.
    backing = torch.empty(
        q.shape[0], q.shape[1], head_dim + 8, dtype=q.dtype, device=DEV
    )
    backing[..., 1 : 1 + head_dim] = q
    offset_q = backing[..., 1 : 1 + head_dim]
    assert offset_q.storage_offset() % 8 != 0
    assert all(stride % 8 == 0 for stride in offset_q.stride()[:-1])
    got, got_visible = _run_scores(offset_q, k_cache, table, t2r, pos, lens)
    torch.testing.assert_close(got, want)
    torch.testing.assert_close(got_visible, want_visible)


@requires_cuda_sm80
def test_scores_accept_a_cache_view_with_a_storage_offset():
    """The same for the cache, which is staged through cp_async."""
    q, k_cache, table, t2r, pos, lens = _scores_case()
    want, want_visible = _run_scores(q, k_cache, table, t2r, pos, lens)

    pages, page_size, head_dim = k_cache.shape
    # Same shape of view: aligned page and entry strides, misaligned base.
    backing = torch.empty(
        pages, page_size, head_dim + 8, dtype=k_cache.dtype, device=DEV
    )
    backing[..., 1 : 1 + head_dim] = k_cache
    offset_cache = backing[..., 1 : 1 + head_dim]
    assert offset_cache.storage_offset() % 8 != 0
    assert all(stride % 8 == 0 for stride in offset_cache.stride()[:-1])
    got, got_visible = _run_scores(q, offset_cache, table, t2r, pos, lens)
    torch.testing.assert_close(got, want)
    torch.testing.assert_close(got_visible, want_visible)
