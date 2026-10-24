"""Signed weighted index scores over paged FP8 region summaries."""

from types import SimpleNamespace

import pytest
import torch

from b12x.attention.qsa._score_cute import launch_score_representatives


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_weighted_representative_scores(weighted, dtype):
    torch.manual_seed(621)
    rows, heads, dim, groups = 9, 4, 256, 1024
    query = torch.randn(rows, heads, dim, device="cuda").to(dtype)
    weights = torch.randn(rows, heads, device="cuda") if weighted else None
    cache = torch.randn(64, 16, dim, device="cuda").to(dtype)
    table = (
        torch.arange(64, dtype=torch.int32, device="cuda").expand(rows, -1).contiguous()
    )
    positions = torch.tensor(
        [0, 7, 8, 15, 16, 4095, 4103, 8191, -1], device="cuda", dtype=torch.int64
    )
    requests = torch.arange(rows, device="cuda", dtype=torch.int32)
    requests[-1] = -1
    lengths = (positions + 1).int()
    scores = torch.empty(rows, groups, device="cuda")
    counts = torch.empty(rows, device="cuda", dtype=torch.int32)
    merges = torch.empty_like(counts)
    caps = SimpleNamespace(
        index_heads=heads,
        index_head_dim=dim,
        compress_ratio=8,
        compressed_page_size=16,
        max_groups=groups,
        group_budget=512,
        dcp_size=1,
        dcp_rank=0,
        cp_kv_cache_interleave_size=8,
    )
    kwargs = dict(
        prepared_query=query,
        query_positions=positions,
        request_ids=requests,
        sequence_lengths=lengths,
        compressed_cache=cache,
        compressed_block_table=table,
        scores=scores,
        eligible_counts=counts,
        merge_lengths=merges,
        group_offset=0,
        group_count=groups,
        caps=caps,
        weights=weights,
        position_offset=0,
    )

    def run():
        launch_score_representatives(**kwargs)

    run()
    expected = torch.einsum(
        "rhd,kd->rhk", query.float(), cache.flatten(0, 1).float()
    ).relu()
    expected = (
        (expected * weights[..., None]).sum(1)
        if weighted
        else expected.sum(1) * dim**-0.5
    )
    visible = positions.clamp_min(0) // 8
    expected.masked_fill_(
        torch.arange(groups, device="cuda")[None] >= visible[:, None], -float("inf")
    )
    torch.testing.assert_close(scores, expected, rtol=2e-5, atol=4e-5)
    torch.testing.assert_close(counts.long(), visible, rtol=0, atol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    scores.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(scores, expected, rtol=2e-5, atol=4e-5)
