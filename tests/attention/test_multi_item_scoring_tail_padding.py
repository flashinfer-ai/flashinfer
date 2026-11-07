# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0

"""Unused V slots must stay masked when MIS prefetch skips to a partial tile."""

import math

import pytest
import torch

import flashinfer


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout", ["NHD", "HND"])
@pytest.mark.parametrize("group", [1, 3])
@pytest.mark.parametrize("page_size", [16, 32])
@pytest.mark.parametrize("tail_fill", [float("nan"), float("inf"), 7.0])
def test_mis_prefetch_partial_v_tail(dtype, layout, group, page_size, tail_fill):
    """Keep unused V padding invisible when MIS jumps to a partial last page."""
    prefix, item_len, num_items, kv_heads, dim = 17, 15, 32, 4, 128
    kv_len = prefix + (item_len + 1) * num_items
    pages = math.ceil(kv_len / page_size)
    last_len = (kv_len - 1) % page_size + 1
    q = torch.zeros(1, kv_heads * group, dim, device="cuda", dtype=dtype)
    kv = torch.zeros(pages, 2, page_size, kv_heads, dim, device="cuda", dtype=dtype)
    kv[:, 1].fill_(1)
    kv[-1, 1, last_len:].fill_(tail_fill)
    if layout == "HND":
        kv = kv.permute(0, 1, 3, 2, 4).contiguous()
    positions = torch.tensor(
        list(range(item_len + 1)) * num_items, device="cuda", dtype=torch.uint16
    ).unsqueeze(0)
    workspace = torch.empty(128 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace, layout, backend="fa2"
    )
    wrapper.plan(
        qo_indptr=torch.tensor([0, 1], device="cuda", dtype=torch.int32),
        paged_kv_indptr=torch.tensor([0, pages], device="cuda", dtype=torch.int32),
        paged_kv_indices=torch.arange(pages, device="cuda", dtype=torch.int32),
        paged_kv_last_page_len=torch.tensor(
            [last_len], device="cuda", dtype=torch.int32
        ),
        num_qo_heads=kv_heads * group,
        num_kv_heads=kv_heads,
        head_dim_qk=dim,
        page_size=page_size,
        causal=True,
        q_data_type=dtype,
        kv_data_type=dtype,
        prefix_len_ptr=torch.tensor([prefix], device="cuda", dtype=torch.uint32),
        token_pos_in_items_ptr=positions,
        token_pos_in_items_len=positions.shape[1],
        max_item_len_ptr=torch.tensor([item_len], device="cuda", dtype=torch.uint16),
        disable_split_kv=True,
    )
    out, lse = wrapper.run(q, kv, return_lse=True)
    # The final query sees 17 prefix keys and 15 item-content keys. Q=0 makes
    # all 32 scores zero; valid V=1 gives output 1 and base-2 LSE=5 exactly.
    torch.testing.assert_close(out, torch.ones_like(out), rtol=0, atol=0)
    torch.testing.assert_close(lse, torch.full_like(lse, 5), rtol=0, atol=0)
