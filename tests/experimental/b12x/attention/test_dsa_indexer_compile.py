"""Prepared DSA selection retains inputs, outputs and reused scratch through compile."""

import pytest
import torch

from b12x.attention import dsa_indexer
from b12x.attention.dsa_indexer.reference import pack_index_k_cache_reference
from b12x.preparation import PreparationSession, PreparedCall
from b12x.preparation.types import require_prepared

from ..conftest import require_b12x


@pytest.mark.parametrize(
    "mode,rows", [("decode", 2), ("decode", 32), ("prefill", 65), ("prefill", 4096)]
)
@pytest.mark.parametrize("with_scores", [False, True])
@pytest.mark.parametrize("dynamic_width", [False, True])
@torch.no_grad()
def test_prepared_indexer_compiled_graph(mode, rows, with_scores, dynamic_width):
    torch.compiler.reset()
    device = require_b12x()
    heads, pages, topk = 32, 32, 512
    source = torch.randn((rows, heads, 128), device=device).to(torch.float8_e4m3fn)
    weights = torch.randn((rows, heads), device=device)
    cache_source = pack_index_k_cache_reference(
        torch.randn((pages * 64, 128), device=device)
    )
    query, query_weights, cache = source.clone(), weights.clone(), cache_source.clone()
    table = torch.randperm(pages, dtype=torch.int32, device=device).unsqueeze(0)
    table = table.expand(rows, -1)
    if mode == "decode":
        table = table.contiguous()
    lengths = (
        torch.arange(rows, dtype=torch.int32, device=device) // 4 + 600
    ).clamp_max(pages * 64)
    width = (
        torch.tensor([pages * 64], dtype=torch.int32, device=device)
        if dynamic_width
        else None
    )
    indices = torch.empty((rows, topk), dtype=torch.int32, device=device)
    scores = (
        torch.empty((rows, topk), dtype=torch.float32, device=device)
        if with_scores
        else None
    )
    observed = torch.empty_like(indices)
    observed_scores = torch.empty_like(scores) if scores is not None else None
    caps = dsa_indexer.Caps(
        device=device,
        num_q_heads=heads,
        max_q_rows=rows,
        max_page_table_width=pages,
        topk=topk,
        mode=mode,
    )
    inputs = dict(
        q_fp8=query,
        query_weights=query_weights,
        index_k_cache=cache,
        page_table=table,
        cache_lengths=lengths,
        active_width=width,
        output_indices=indices,
        output_scores=scores,
    )
    plan = dsa_indexer.plan(
        caps, invocation=dsa_indexer.invocation_from_tensors(caps, **inputs)
    )
    scratch = None

    def bind(state):
        return state.bind(
            scratch=scratch,
            real_page_table=table,
            cache_seqlens_int32=lengths,
            active_width=width,
            expected_num_q_heads=heads,
            shared_page_table=mode == "prefill",
        )

    def run(state, runtime):
        state.run(
            runtime,
            q_fp8=query,
            query_weights=query_weights,
            index_k_cache=cache,
            output_indices=indices,
            output_scores=scores,
        )

    with PreparationSession(
        device=device, autotune=False, compile_workers=0
    ) as session:

        def prepare(state):
            nonlocal scratch
            (spec,) = state.layout.scratch_specs()
            scratch = torch.empty(spec.shape, dtype=spec.dtype, device=device)
            runtime = bind(state)
            return PreparedCall(
                run=lambda: run(state, runtime), owners=(scratch, runtime)
            )

        session.prepare((plan.request(name="indexer", prepare_call=prepare),))
        binding = dsa_indexer.bind(plan, scratch=scratch, **inputs)
        other_width = (
            None
            if dynamic_width
            else torch.full((1,), pages * 64, dtype=torch.int32, device=device)
        )
        with pytest.raises(ValueError, match="presence must match"):
            dsa_indexer.bind(
                plan, scratch=scratch, **{**inputs, "active_width": other_width}
            )

        def forward():
            query.copy_(source)
            query_weights.copy_(weights)
            cache.copy_(cache_source)
            observed.copy_(dsa_indexer.run(binding))
            if scores is not None:
                observed_scores.copy_(scores)
            scratch.fill_(255)
            query.zero_()
            query_weights.zero_()
            cache.zero_()

        compiled = torch.compile(forward, fullgraph=True, dynamic=False)
        compiled()
        torch.cuda.synchronize(device)
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            compiled()
        session.freeze()
        state = require_prepared(plan, "attention.dsa_indexer", query.device)
        for count in (0, 256, 1024, pages * 64):
            source.copy_(torch.randn(source.shape, device=device).to(source.dtype))
            weights.normal_()
            cache_source.copy_(
                pack_index_k_cache_reference(
                    torch.randn((pages * 64, 128), device=device)
                )
            )
            lengths.copy_(
                (
                    torch.arange(rows, dtype=torch.int32, device=device) // 4 + count
                ).clamp_max(pages * 64)
            )
            lengths[0] = 0
            if count == 0:
                lengths.zero_()
            if width is not None:
                width.copy_(lengths.amax().reshape(1))
            query.copy_(source)
            query_weights.copy_(weights)
            cache.copy_(cache_source)
            runtime = bind(state)
            before = torch.cuda.memory_stats(device)
            run(state, runtime)
            after = torch.cuda.memory_stats(device)
            for field in ("allocation.all.allocated", "allocation.all.freed"):
                assert before[field] == after[field]
            expected = indices.clone()
            expected_scores = scores.clone() if scores is not None else None
            expected_logits = None
            if mode == "prefill":
                block_k = runtime.prefill_block_k
                tiles = runtime.supertile_k // block_k
                expected_logits = (
                    runtime.scratch.get_indexer_contiguous_tile_logits()
                    .reshape(-1, tiles, 32, block_k)
                    .permute(0, 2, 1, 3)
                    .reshape(-1, tiles * block_k)[:rows]
                    .clone()
                )
            observed.fill_(-42)
            scratch.fill_(255)
            torch.cuda.synchronize(device)
            before = torch.cuda.memory_stats(device)
            graph.replay()
            torch.cuda.synchronize(device)
            after = torch.cuda.memory_stats(device)
            for field in (
                "allocation.all.allocated",
                "allocation.all.freed",
                "segment.all.allocated",
                "segment.all.freed",
            ):
                assert before[field] == after[field]
            actual_ids, actual_order = observed.sort()
            expected_ids, expected_order = expected.sort()
            differing = (actual_ids != expected_ids).any(1)
            if differing.any():
                assert expected_logits is not None
                for row in differing.nonzero().flatten().tolist():
                    actual_row = observed[row].tolist()
                    expected_row = expected[row].tolist()
                    valid_actual = [index for index in actual_row if index >= 0]
                    valid_expected = [index for index in expected_row if index >= 0]
                    assert (
                        len(valid_actual)
                        == len(valid_expected)
                        == len(set(valid_actual))
                    )
                    assert actual_row.count(-1) == expected_row.count(-1)
                    assert all(0 <= index < lengths[row] for index in valid_actual)
                    changed = sorted(set(valid_actual) ^ set(valid_expected))
                    cutoff = expected_logits[row, valid_expected].min()
                    # Equal-score cutoff ties have no stable winner across CTA arrivals.
                    assert torch.all(expected_logits[row, changed] == cutoff)
            if scores is not None:
                if expected_logits is not None:
                    reference_scores = expected_logits.gather(
                        1, observed.clamp_min(0).long()
                    )
                    reference_scores.masked_fill_(observed < 0, float("-inf"))
                    torch.testing.assert_close(
                        observed_scores, reference_scores, rtol=0, atol=0
                    )
                else:
                    torch.testing.assert_close(
                        observed_scores.gather(1, actual_order),
                        expected_scores.gather(1, expected_order),
                        rtol=0,
                        atol=0,
                    )
        graph.reset()
