"""Bidirectional dense MLA with right-aligned left-window bounds."""

import pytest
import torch

from b12x.attention import dense_mla

from .test_dense_mla import _bind, _prepared_scope
from ..conftest import require_b12x


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("window", [None, 129])
@torch.inference_mode()
def test_ragged_bidirectional_graph_replay(causal, window):
    device = require_b12x()
    lengths = [8, 65, 258]
    query_lengths = [8, 4, 8]
    query = torch.randn(20, 4, 576, device=device, dtype=torch.bfloat16) * 0.1
    cache = torch.randn(15, 64, 576, device=device, dtype=torch.bfloat16) * 0.1
    pages = torch.arange(15, device=device, dtype=torch.int32).reshape(3, 5).flip(1)
    cache_lengths = torch.tensor(lengths, device=device, dtype=torch.int32)
    offsets = torch.tensor([0, 8, 12, 20], device=device, dtype=torch.int32)
    output = torch.empty(20, 4, 512, device=device, dtype=torch.bfloat16)
    plan = dense_mla.plan(
        dense_mla.Caps(
            device=device,
            mode="decode",
            kv_dtype=torch.bfloat16,
            num_q_heads=4,
            page_size=64,
            max_total_q=20,
            max_batch=3,
            max_cache_tokens=320,
            max_page_table_width=5,
            num_cache_pages=15,
            window_size=window,
            causal=causal,
            use_cuda_graph=True,
            partial_dtype=torch.float32,
        )
    )
    with _prepared_scope():
        binding = _bind(
            plan,
            q=query,
            kv_cache=cache,
            output=output,
            page_table=pages,
            cache_seqlens=cache_lengths,
            cu_seqlens_q=offsets,
            sm_scale=0.07,
        )
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual, actual_lse = dense_mla.run(binding=binding)
        for first in [8] if causal else [0, 3, 8]:
            cache_lengths[0] = first
            expected, expected_lse = dense_mla.reference(
                query,
                cache,
                pages,
                cache_lengths,
                offsets,
                sm_scale=0.07,
                window_size=window,
                causal=causal,
            )
            allocated, reserved = (
                torch.cuda.memory_allocated(),
                torch.cuda.memory_reserved(),
            )
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_allocated() == allocated
            assert torch.cuda.memory_reserved() == reserved
            torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.001)
            torch.testing.assert_close(actual_lse, expected_lse, rtol=2e-5, atol=2e-5)
