"""Exact winner construction across ties, score chunks, and graph replay."""

import pytest
import torch

from b12x._lib.runtime_control import kernel_resolution_guard
from b12x.attention.qsa._kernels import launch_stabilize_topk
from b12x.attention.qsa._stable_select_cute import _CACHE
from ..conftest import require_b12x


@pytest.mark.parametrize('budget', [512, 2048])
@pytest.mark.parametrize('group_offset', [0, 4096])
@pytest.mark.parametrize('pattern', ['zero', 'ties', 'random'])
def test_stable_selection_exact_replay(budget, group_offset, pattern):
    device = require_b12x()
    rows, width = 9, budget + 4096
    generator = torch.Generator(device=device).manual_seed(931)
    scores = torch.empty((rows, width), dtype=torch.float32, device=device)
    lengths = torch.tensor([0, 1, 31, 32, budget-1, budget, budget+1, width-1, width], dtype=torch.int32, device=device)
    eligible = lengths.clone()
    if group_offset:
        eligible.copy_(torch.where(lengths <= budget, lengths, lengths - budget + group_offset))
    prior = torch.arange(budget, dtype=torch.int32, device=device).expand(rows, -1).contiguous()
    values = torch.empty((rows, budget), dtype=torch.float32, device=device)
    ids = torch.empty((rows, budget), dtype=torch.int32, device=device)
    stable_values = torch.empty_like(values)
    stable_ids = torch.empty_like(ids)
    blocks = (width + 511) // 512
    counts = torch.empty((rows, blocks), dtype=torch.int32, device=device)
    other_counts = torch.empty_like(counts)
    thresholds = torch.empty((rows,), dtype=torch.float32, device=device)
    totals = torch.empty_like(lengths)

    def produce():
        if pattern == 'zero':
            scores.zero_()
        elif pattern == 'ties':
            scores.copy_(torch.randint(0, 7, scores.shape, generator=generator, device=device).float())
        else:
            scores.uniform_(generator=generator)
        if group_offset:
            scores[:, :budget].copy_(scores[:, :budget].sort(descending=True).values)
        masked = scores.masked_fill(torch.arange(width, device=device)[None, :] >= lengths[:, None], -float('inf'))
        values.copy_(masked.topk(budget, sorted=False).values)
        expected_values = torch.full_like(values, -float('inf'))
        expected_ids = torch.full_like(ids, -1)
        for row, length in enumerate(lengths.tolist()):
            carry = min(int(eligible[row]), group_offset, budget)
            global_ids = torch.arange(length, device=device, dtype=torch.int32) + group_offset - carry
            global_ids[:carry] = prior[row, :carry]
            # Input ties are in ascending global-ID order within each chunk.
            order = torch.argsort(scores[row, :length], descending=True, stable=True)[:budget]
            count = len(order)
            expected_values[row, :count] = scores[row, order]
            expected_ids[row, :count] = global_ids[order]
        return expected_values, expected_ids

    def run(live_rows=rows):
        launch_stabilize_topk(
            scores=scores[:live_rows], merge_lengths=lengths[:live_rows],
            prior_ids=prior[:live_rows], eligible_counts=eligible[:live_rows],
            topk_values=values[:live_rows], topk_group_ids=ids[:live_rows],
            tie_counts=counts[:live_rows], greater_counts=other_counts[:live_rows],
            stable_values=stable_values[:live_rows], stable_ids=stable_ids[:live_rows],
            thresholds=thresholds[:live_rows], greater_totals=totals[:live_rows],
            group_offset=group_offset, group_budget=budget)

    expected = produce()
    run()
    torch.testing.assert_close(values, expected[0], rtol=0, atol=0)
    assert torch.equal(ids, expected[1])
    resolved = dict(_CACHE)
    graphs = {}
    with kernel_resolution_guard('Stable selection reuses prepared programs across live row counts'):
        for live_rows in (1, 4, rows):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run(live_rows)
            graphs[live_rows] = graph
    for _ in range(3):
        lengths.copy_(lengths.roll(1))
        eligible.copy_(lengths if not group_offset else torch.where(lengths <= budget, lengths, lengths-budget+group_offset))
        expected = produce()
        for live_rows, graph in graphs.items():
            # Restore the radix threshold inputs before each in-place selection.
            masked = scores.masked_fill(torch.arange(width, device=device)[None, :] >= lengths[:, None], -float('inf'))
            values.copy_(masked.topk(budget, sorted=False).values)
            ids.fill_(-98765)
            allocation = torch.cuda.memory_allocated()
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_allocated() == allocation
            torch.testing.assert_close(values[:live_rows], expected[0][:live_rows], rtol=0, atol=0)
            assert torch.equal(ids[:live_rows], expected[1][:live_rows])
            assert torch.all(ids[live_rows:] == -98765)
        assert _CACHE.keys() == resolved.keys()
        assert all(_CACHE[key] is value for key, value in resolved.items())
