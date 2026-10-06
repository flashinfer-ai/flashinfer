"""Prepared DSA selection across live page-table widths and output contracts."""

import pytest
import torch

from b12x.attention import dsa_indexer as dsa
from b12x.attention.dsa_indexer.reference import (
    pack_index_k_cache_reference,
    unpack_index_k_cache_reference,
)
from b12x.preparation import PreparationSession, PreparedCall
from b12x.testing.reference.helpers import require_b12x


@pytest.mark.parametrize("physical", [False, True])
@pytest.mark.parametrize("with_scores", [False, True])
@pytest.mark.parametrize("mode", ["decode", "prefill"])
@torch.inference_mode()
def test_prepared_dsa_single_and_multiple_supertiles(physical, with_scores, mode, tmp_path):
    device = require_b12x()
    torch.manual_seed(161)
    rows, heads, topk, max_pages = 3, 16, 512, 48
    packed = pack_index_k_cache_reference(
        torch.randn(max_pages * 64, 128, device=device)
    )
    decoded = unpack_index_k_cache_reference(packed, num_tokens=max_pages * 64).float()
    first_page = 2**31 // packed.stride(0) + 1
    cache = torch.empty(
        (first_page + max_pages, packed.shape[1]), dtype=torch.uint8, device=device
    )
    cache[first_page:].copy_(packed)
    q = torch.randn(rows, heads, 128, device=device).to(torch.float8_e4m3fn)
    weights = torch.rand(rows, heads, device=device)
    pages = torch.arange(
        first_page, first_page + max_pages, dtype=torch.int32, device=device
    )[None].repeat(rows, 1)
    lengths = torch.full((rows,), max_pages * 64, dtype=torch.int32, device=device)
    active = torch.full((1,), max_pages * 64, dtype=torch.int32, device=device)
    output = torch.empty(rows, topk, dtype=torch.int32, device=device)
    scores = (
        torch.empty(rows, topk, dtype=torch.float32, device=device)
        if with_scores
        else None
    )
    caps = dsa.Caps(
        device=device,
        num_q_heads=heads,
        max_q_rows=rows,
        max_page_table_width=max_pages,
        topk=topk,
        mode=mode,
        route="packed_contiguous" if mode == "prefill" else "paged_tiled",
        supertile_k=1024,
        output_index_space="physical" if physical else "logical",
    )
    operands = dict(
        q_fp8=q,
        query_weights=weights,
        index_k_cache=cache,
        page_table=pages,
        cache_lengths=lengths,
        active_width=active,
        output_indices=output,
        output_scores=scores,
    )
    plan = dsa.plan(caps, invocation=dsa.invocation_from_tensors(caps, **operands))

    def prepare(state):
        (spec,) = state.layout.scratch_specs()
        scratch = torch.empty(spec.shape, dtype=spec.dtype, device=device)
        binding = state.bind(
            scratch=scratch,
            real_page_table=pages,
            cache_seqlens_int32=lengths,
            active_width=active,
            expected_num_q_heads=heads,
            shared_page_table=mode == "prefill",
            output_physical_slots=physical,
        )
        return PreparedCall(
            run=lambda: state.run(
                binding,
                q_fp8=q,
                query_weights=weights,
                index_k_cache=cache,
                output_indices=output,
                output_scores=scores,
            )
        )

    def check():
        logits = torch.einsum("rhd,kd->rhk", q.float(), decoded)
        reference = (logits.relu_() * weights[:, :, None]).sum(dim=1)
        reference.masked_fill_(
            torch.arange(max_pages * 64, device=device)[None] >= lengths[:, None],
            -float("inf"),
        )
        values, indices = reference.topk(topk, dim=1)
        if physical:
            indices += first_page * 64
        torch.testing.assert_close(
            output.sort(dim=1).values,
            indices.to(torch.int32).sort(dim=1).values,
            rtol=0,
            atol=0,
        )
        if scores is not None:
            torch.testing.assert_close(
                scores.sort(dim=1).values,
                values.sort(dim=1).values,
                rtol=2e-4,
                atol=2e-3,
            )

    with PreparationSession(device=device, autotune=True, cache_dir=tmp_path) as session:
        session.prepare(
            [
                plan.request(
                    name="dsa-live-page-width",
                    prepare_call=prepare,
                    benchmark_call=prepare,
                )
            ]
        )
        (spec,) = plan.scratch_specs()
        scratch = torch.empty(spec.shape, dtype=spec.dtype, device=device)
        session.freeze()
        for width in (16, 32, 48):
            lengths.fill_(width * 64)
            active.fill_(width * 64)
            binding = dsa.bind(
                plan,
                scratch=scratch,
                **{**operands, "page_table": pages[:, :width].contiguous()},
            )
            dsa.run(binding)
            check()
            graph = torch.cuda.CUDAGraph()
            try:
                with session.capture(), torch.cuda.graph(graph):
                    dsa.run(binding)
                q.copy_((-q.float()).to(q.dtype))
                lengths[0] -= 128
                output.fill_(-1)
                allocated = torch.cuda.memory_allocated(device)
                graph.replay()
                torch.cuda.synchronize(device)
                assert torch.cuda.memory_allocated(device) == allocated
                check()
            finally:
                graph.reset()
