"""Regression coverage for count-aware compact MoE routing."""

from __future__ import annotations

import pytest
import torch

from flashinfer.fused_moe.cute_dsl.moe_utils import (
    allocate_moe_sort_buffers,
    moe_sort,
)
from flashinfer.utils import is_sm100a_supported


@pytest.mark.skipif(
    not (torch.cuda.is_available() and is_sm100a_supported(torch.device("cuda"))),
    reason="requires SM100 CuTe-DSL routing",
)
def test_expert_counts_are_populated_by_routing():
    num_tokens, num_experts, top_k = 256, 512, 8
    ids = torch.zeros((num_tokens, top_k), dtype=torch.int32, device="cuda")
    weights = torch.ones((num_tokens, top_k), dtype=torch.bfloat16, device="cuda")
    buffers = allocate_moe_sort_buffers(
        num_tokens,
        num_experts,
        top_k,
        include_expert_counts=True,
    )
    assert buffers["out_expert_counts"] is not None
    assert buffers["out_expert_counts"].numel() == 2 * num_experts
    (
        tile_idx_to_expert_idx,
        tile_idx_to_mn_limit,
        expanded_idx_to_permuted_idx,
        permuted_idx_to_expanded_idx,
        total_num_padded_tokens,
        num_non_exiting_tiles,
        local_expert_counts,
    ) = moe_sort(
        ids,
        weights,
        num_experts,
        top_k,
        **buffers,
    )
    assert local_expert_counts is not None
    torch.cuda.synchronize()
    expected = torch.zeros(num_experts, dtype=torch.int32, device="cuda")
    expected[0] = num_tokens * top_k
    torch.testing.assert_close(local_expert_counts, expected)


def test_expert_counts_use_global_indices_for_nonzero_local_offset():
    num_tokens, num_experts, top_k = 1_000_000, 512, 8
    local_expert_offset, num_local_experts = 5, 2
    ids = torch.full(
        (num_tokens, top_k),
        local_expert_offset + 1,
        dtype=torch.int32,
        device="cuda",
    )
    weights = torch.ones((num_tokens, top_k), dtype=torch.bfloat16, device="cuda")
    buffers = allocate_moe_sort_buffers(
        num_tokens,
        num_experts,
        top_k,
        num_local_experts,
        include_expert_counts=True,
    )
    (
        tile_idx_to_expert_idx,
        tile_idx_to_mn_limit,
        expanded_idx_to_permuted_idx,
        permuted_idx_to_expanded_idx,
        total_num_padded_tokens,
        num_non_exiting_tiles,
        local_expert_counts,
    ) = moe_sort(
        ids,
        weights,
        num_experts,
        top_k,
        local_expert_offset=local_expert_offset,
        num_local_experts=num_local_experts,
        **buffers,
    )
    assert local_expert_counts is not None
    torch.cuda.synchronize()
    expected = torch.tensor([0, num_tokens * top_k], dtype=torch.int32, device="cuda")
    torch.testing.assert_close(local_expert_counts, expected)
