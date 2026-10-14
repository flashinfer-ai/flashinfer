from __future__ import annotations

import pytest
import torch

from b12x.attention.paged.reference import paged_attention_reference


def test_query_blocked_paged_reference_preserves_global_row_indexing() -> None:
    generator = torch.Generator().manual_seed(712)
    q = torch.randn(5, 4, 4, generator=generator)
    k_cache = torch.randn(4, 2, 2, 4, generator=generator)
    v_cache = torch.randn(4, 2, 2, 3, generator=generator)
    page_table = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32)
    cache_seqlens = torch.tensor([7], dtype=torch.int32)
    cu_seqlens_q = torch.tensor([0, 5], dtype=torch.int32)
    sink = torch.linspace(-0.25, 0.25, 4)
    relative_bias = torch.randn(5, 4, 8, generator=generator) * 0.1

    expected_out, expected_lse = paged_attention_reference(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        causal=True,
        window_left=3,
        attention_sink_bias=sink,
        relative_attention_bias=relative_bias,
        query_block_size=None,
    )
    blocked_out, blocked_lse = paged_attention_reference(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        causal=True,
        window_left=3,
        attention_sink_bias=sink,
        relative_attention_bias=relative_bias,
        query_block_size=2,
    )

    torch.testing.assert_close(blocked_out, expected_out, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(blocked_lse, expected_lse, atol=1e-6, rtol=1e-6)


def test_query_blocked_paged_reference_rejects_nonpositive_blocks() -> None:
    q = torch.zeros(1, 1, 1)
    cache = torch.zeros(1, 1, 1, 1)
    page_table = torch.zeros((1, 1), dtype=torch.int32)
    seqlens = torch.ones(1, dtype=torch.int32)
    cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32)

    with pytest.raises(ValueError, match="query_block_size"):
        paged_attention_reference(
            q,
            cache,
            cache,
            page_table,
            seqlens,
            cu_seqlens_q,
            query_block_size=0,
        )
