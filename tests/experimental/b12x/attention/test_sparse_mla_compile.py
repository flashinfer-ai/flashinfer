"""Prepared latent writes and sparse attention preserve mutations through compile."""

import pytest
import torch

from b12x.attention import sparse_mla
from b12x.attention.sparse_mla._preparation import state, writer_state
from b12x.preparation import PreparationSession, PreparedCall

from ..conftest import require_b12x

_ALLOCATOR_COUNTERS = (
    "allocation.all.allocated",
    "allocation.all.freed",
    "segment.all.allocated",
    "segment.all.freed",
)


@pytest.mark.parametrize("heads", [16, 32])
@pytest.mark.parametrize("rows", [1, 12])
def test_glm_decode_compiler_uses_alignment_without_fake_pointers(
    monkeypatch, heads, rows
):
    from torch._subclasses.fake_tensor import FakeTensor
    from b12x.attention.sparse_mla._preparation import compile_sparse_mla

    device = require_b12x()

    def reject_pointer(self):
        raise AssertionError("FakeTensor has no runtime data pointer")

    monkeypatch.setattr(FakeTensor, "data_ptr", reject_pointer)
    caps = sparse_mla.Caps(
        device=device,
        num_q_heads=heads,
        max_q_rows=rows,
        max_width=2051,
        softmax_scale=256**-0.5,
        max_batch=rows,
        max_kv_rows=256,
        kv_dtype=torch.uint8,
        head_dim=512,
        v_head_dim=512,
        page_size=64,
        model_type=sparse_mla.ModelType.GLM_NEXT,
        cache_record_bytes=528,
        mode="decode",
    )
    launch = compile_sparse_mla(caps, torch.cuda.current_device())
    assert launch.vector_q
    assert launch.grid_programs and launch.merge_program is not None


@pytest.mark.parametrize(
    "mode,rows,heads",
    [("decode", 2, 32), ("decode", 2, 16), ("extend", 65, 32), ("extend", 4096, 16)],
)
@pytest.mark.parametrize("return_lse", [False, True])
@torch.no_grad()
def test_glm_latent_write_and_sparse_attention_compiled_graph(
    mode, rows, heads, return_lse
):
    device = require_b12x()
    page_size, width = 64, 2051
    records = ((rows + 192 + page_size - 1) // page_size) * page_size
    visible = min(rows, width)
    returns_pair = mode != "decode" or return_lse
    generator = torch.Generator(device=device).manual_seed(530)
    latent = (
        torch.randn(
            (rows, 512), generator=generator, device=device, dtype=torch.bfloat16
        )
        * 0.1
    )
    queries = (
        torch.randn(
            (rows, heads, 512), generator=generator, device=device, dtype=torch.bfloat16
        )
        * 0.1
    )
    source, q = torch.empty_like(latent), torch.empty_like(queries)
    cache = torch.zeros(
        (records // page_size, page_size, 528), device=device, dtype=torch.uint8
    )
    slots = torch.arange(rows, device=device, dtype=torch.int64)
    selected = torch.full((rows, width), -1, device=device, dtype=torch.int32)
    selected[:, :visible] = torch.arange(visible, device=device, dtype=torch.int32)
    lengths = torch.arange(1, rows + 1, device=device, dtype=torch.int32).clamp_max(
        width
    )
    source.copy_(latent)
    q.copy_(queries)
    plan = sparse_mla.plan(
        sparse_mla.Caps(
            device=device,
            num_q_heads=heads,
            max_q_rows=rows,
            max_width=width,
            softmax_scale=256**-0.5,
            max_batch=rows,
            max_kv_rows=records,
            kv_dtype=torch.uint8,
            head_dim=512,
            v_head_dim=512,
            page_size=page_size,
            model_type=sparse_mla.ModelType.GLM_NEXT,
            cache_record_bytes=528,
            mode=mode,
            return_lse=return_lse,
        )
    )
    writer = sparse_mla.plan_cache_writer(source, cache, slots)
    (spec,) = plan.scratch_specs()
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=device)
    output = torch.empty_like(q)
    lse = torch.empty((rows, heads), dtype=torch.float32, device=device)
    with PreparationSession(
        device=device, autotune=False, compile_workers=0
    ) as session:

        def prepare(state):
            runtime = state.bind(
                scratch=scratch,
                q=q,
                selected_indices=selected,
                cache_seqlens_int32=lengths,
                nsa_cache_seqlens_int32=lengths,
                kv_cache=cache,
            )
            return PreparedCall(
                run=lambda: state.run(runtime, kv_cache=cache), owners=(runtime,)
            )

        session.prepare(
            (
                writer.request(
                    name="latent-cache",
                    prepare_call=lambda state: PreparedCall(
                        run=lambda: state.run(source, cache, slots)
                    ),
                ),
                plan.request(name="sparse-attention", prepare_call=prepare),
            )
        )
        binding = sparse_mla.bind(
            plan,
            scratch=scratch,
            q=q,
            kv_cache=cache,
            selected_indices=selected,
            cache_lengths=lengths,
            selected_lengths=lengths,
        )

        def forward():
            source.copy_(latent)
            q.copy_(queries)
            sparse_mla.concat_and_cache_glm_next_mla(source, cache, slots, plan=writer)
            result = sparse_mla.run(binding)
            output.copy_(result[0] if returns_pair else result)
            if returns_pair:
                lse.copy_(result[1])
            scratch.fill_(255)
            q.zero_()
            source.zero_()

        compiled = torch.compile(forward, fullgraph=True, dynamic=False)
        compiled()
        torch.cuda.synchronize(device)
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            compiled()
        session.freeze()
        for offset in (0, 64, 192):
            latent.normal_(std=0.1)
            queries.normal_(std=0.1)
            slots.copy_(torch.arange(rows, device=device, dtype=torch.int64) + offset)
            selected[:, :visible] = slots[:visible].to(torch.int32)
            cache.zero_()
            source.copy_(latent)
            q.copy_(queries)
            before = torch.cuda.memory_stats(device)
            writer_state(writer, device=cache.device).run(source, cache, slots)
            reference = state(plan, device=q.device).run(
                binding.runtime, kv_cache=cache
            )
            after = torch.cuda.memory_stats(device)
            for key in _ALLOCATOR_COUNTERS:
                assert before[key] == after[key]
            expected = (
                (reference[0] if returns_pair else reference).clone(),
                cache.clone(),
                reference[1].clone() if returns_pair else None,
            )
            cache.zero_()
            output.fill_(42)
            lse.fill_(42)
            torch.cuda.synchronize(device)
            before = torch.cuda.memory_stats(device)
            graph.replay()
            torch.cuda.synchronize(device)
            after = torch.cuda.memory_stats(device)
            for key in _ALLOCATOR_COUNTERS:
                assert before[key] == after[key]
            torch.testing.assert_close(output, expected[0], rtol=0, atol=0)
            torch.testing.assert_close(cache, expected[1], rtol=0, atol=0)
            if returns_pair:
                torch.testing.assert_close(lse, expected[2], rtol=0, atol=0)
        graph.reset()
