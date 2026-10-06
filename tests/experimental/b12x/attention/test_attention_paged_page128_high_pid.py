from __future__ import annotations

import math

import pytest
import torch

from b12x.attention import paged
from b12x.attention.paged.reference import paged_attention_reference
from b12x.preparation import PreparationSession, PreparedCall

from b12x.testing.reference.helpers import require_b12x


def _require_exact_laguna_device() -> torch.device:
    device = require_b12x()
    if torch.cuda.get_device_capability(device) != (12, 0):
        pytest.skip("the exact Laguna KV128 specialization requires SM120")
    return device
@torch.inference_mode()
def test_laguna_gqa6_extend_prepared_graph_replay_high_page_ids_and_tails() -> None:
    """A prepared graph must retain high-page addressing and live replay metadata."""
    device = _require_exact_laguna_device()
    torch.manual_seed(20260726)

    page_size, head_dim, q_heads, kv_heads, q_rows = 128, 128, 24, 4, 64
    max_cache_seqlen = 129
    live_page_count = math.ceil(max_cache_seqlen / page_size)
    element_size = torch.empty((), dtype=torch.float8_e4m3fn).element_size()
    page_stride_bytes = 2 * page_size * kv_heads * head_dim * element_size
    high_page_id = torch.iinfo(torch.int32).max // page_stride_bytes + 2
    num_cache_pages = high_page_id + live_page_count
    combined_kv_cache = torch.empty(
        (num_cache_pages, 2, page_size, kv_heads, head_dim),
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    k_cache, v_cache = combined_kv_cache[:, 0], combined_kv_cache[:, 1]
    assert high_page_id * k_cache.stride(0) * element_size > torch.iinfo(torch.int32).max

    # Low capacity pages are intentionally unlike the live pages: truncating a
    # high page id must not accidentally reproduce the reference result.
    combined_kv_cache[:live_page_count].fill_(3)
    combined_kv_cache[high_page_id:].copy_(
        (torch.randn(
            (live_page_count, 2, page_size, kv_heads, head_dim),
            dtype=torch.bfloat16,
            device=device,
        ) / 4).to(torch.float8_e4m3fn)
    )
    page_table = torch.arange(
        high_page_id, num_cache_pages, dtype=torch.int32, device=device
    ).unsqueeze(0)
    q = torch.randn((q_rows, q_heads, head_dim), dtype=torch.bfloat16, device=device) / 4
    cache_seqlens = torch.tensor([max_cache_seqlen], dtype=torch.int32, device=device)
    cu_seqlens_q = torch.tensor([0, q_rows], dtype=torch.int32, device=device)
    k_descale = torch.tensor([[0.5, 0.75, 1.25, 1.5]], dtype=torch.float32, device=device)
    v_descale = torch.tensor([[1.5, 1.25, 0.75, 0.5]], dtype=torch.float32, device=device)
    output = torch.empty((q_rows, q_heads, head_dim), dtype=q.dtype, device=device)

    caps = paged.Caps(
        device=device, mode="extend", dtype=q.dtype, kv_dtype=k_cache.dtype,
        num_q_heads=q_heads, num_kv_heads=kv_heads, head_dim_qk=head_dim,
        head_dim_vo=head_dim, page_size=page_size, max_total_q=q_rows, max_batch=1,
        max_page_table_width=live_page_count, max_work_items=1024,
        max_partial_rows=1024, num_cache_pages=num_cache_pages, use_cuda_graph=True,
    )
    declaration = paged.plan(
        caps,
        invocation=paged.invocation_from_tensors(
            caps, q=q, k_cache=k_cache, v_cache=v_cache, output=output,
            page_table=page_table, cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q, k_descale=k_descale, v_descale=v_descale,
        ),
    )
    metadata: dict[str, object] = {}

    def prepare_call(state: object) -> PreparedCall:
        # The callback reads only session-materialized scratch metadata; all
        # serving tensors remain caller-owned and are bound after preparation.
        spec = state.scratch_plan.scratch_specs()[0]
        metadata["spec"] = spec
        scratch = torch.empty(spec.shape, dtype=spec.dtype, device=device)
        prime_output = torch.empty_like(output)
        binding = state.bind(
            scratch=scratch, q=q, k_cache=k_cache, v_cache=v_cache,
            output=prime_output, page_table=page_table, cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q, active_total_q=q_rows,
            k_descale=k_descale, v_descale=v_descale,
        )
        return PreparedCall(run=lambda: state.run(binding), output=prime_output, owners=(scratch, binding))

    request = declaration.request(
        name="laguna-high-page",
        prepare_call=prepare_call,
    )
    with PreparationSession(device=device, autotune=False) as session:
        result = session.prepare((request,))
        try:
            plan = result.plans["laguna-high-page"]
            spec = metadata["spec"]
            scratch = torch.empty(spec.shape, dtype=spec.dtype, device=device)
            binding = paged.bind(
                plan, scratch=scratch, q=q, k_cache=k_cache, v_cache=v_cache,
                output=output, page_table=page_table, cache_seqlens=cache_seqlens,
                cu_seqlens_q=cu_seqlens_q, active_total_q=q_rows,
                k_descale=k_descale, v_descale=v_descale,
            )
            graph = torch.cuda.CUDAGraph()
            with session.capture(), torch.cuda.graph(graph):
                captured_output, _ = paged.run(binding=binding, plan=plan)
            assert captured_output.data_ptr() == output.data_ptr()
            addresses = (output.data_ptr(), scratch.data_ptr(), page_table.data_ptr(), cache_seqlens.data_ptr())
            for cache_seqlen in (127, 128, 129):
                cache_seqlens.fill_(cache_seqlen)
                output.fill_(torch.nan)
                graph.replay()
                torch.cuda.synchronize(device)
                reference, _ = paged_attention_reference(
                    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q,
                    causal=True, k_descale=k_descale, v_descale=v_descale,
                )
                assert torch.isfinite(output).all().item()
                assert torch.count_nonzero(output).item() > 0
                torch.testing.assert_close(output.float(), reference.float(), atol=5e-2, rtol=5e-2)
                assert (output.data_ptr(), scratch.data_ptr(), page_table.data_ptr(), cache_seqlens.data_ptr()) == addresses
            del graph
        finally:
            result.close()
