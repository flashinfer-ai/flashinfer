"""Regression tests for dense split decode under CUDA graphs.

Covers the gqa_group_size > 8 case (e.g. MiniMax-M3 dense layers: 64 q heads /
4 KV heads = gqa 16, head_dim 128) where the single-qtile / regularized decode
graph epilogues previously only stored MMA row_slot 0 (packed rows 0-7),
dropping q heads 8..group_size-1 of every KV group.

Note the CUDA-graph metadata contract exercised here: with
copy_runtime_metadata (the default), the copy from the caller's metadata
tensors into the scratch's static buffers is captured inside the graph, so the
metadata tensors passed at capture must keep stable addresses across replays —
callers update their contents in place (this is what a serving engine does with
persistent buffers).
"""
from __future__ import annotations

import math

import torch

from b12x.attention.paged.reference import paged_attention_reference
from b12x.attention._shared.contiguous.api import clear_attention_caches
from b12x.attention.paged._forward import paged_attention_forward
from b12x.attention.paged._scratch import B12XPagedAttentionScratchCaps, plan_paged_attention_scratch
from b12x.attention.paged.planner import create_paged_plan

from b12x.testing.reference.helpers import require_b12x
from b12x.testing.reference.paged_attention_helpers import make_paged_inputs


class _ForcedSplitDecodeGraphHarness:
    """Decode-graph harness using the default dense split planner policy."""

    def __init__(self, q, k_cache, v_cache):
        self.q = q
        self.k_cache = k_cache
        self.v_cache = v_cache
        self._scratch_plan = None
        self._scratch = None
        self._output = None
        self._page_table = None
        self._cache_seqlens = None
        self._cu_seqlens_q = None
        self._last_scratch = None

    def prepare(self, page_table, cache_seqlens, cu_seqlens_q):
        plan = create_paged_plan(
            self.q,
            self.k_cache,
            self.v_cache,
            page_table,
            cache_seqlens,
            cu_seqlens_q,
            mode="decode",
            enable_cuda_graph=True,
            graph_chunk_policy=True,
        )
        assert plan.split_kv
        if self._scratch_plan is None:
            batch = page_table.shape[0]
            self._scratch_plan = plan_paged_attention_scratch(
                B12XPagedAttentionScratchCaps(
                    device=self.q.device,
                    mode="decode",
                    dtype=self.q.dtype,
                    kv_dtype=self.k_cache.dtype,
                    num_q_heads=self.q.shape[1],
                    num_kv_heads=self.k_cache.shape[2],
                    head_dim_qk=self.q.shape[2],
                    head_dim_vo=self.v_cache.shape[3],
                    page_size=self.k_cache.shape[1],
                    max_total_q=plan.total_q,
                    max_batch=batch,
                    max_page_table_width=page_table.shape[1],
                    # split decode graphs pad work items to the CTA policy
                    # (~2 CTAs/SM), so size generously rather than from the
                    # bootstrap plan
                    max_work_items=max(plan.new_batch_size, 2048),
                    max_partial_rows=max(plan.total_num_partial_rows, 2048),
                    num_cache_pages=self.k_cache.shape[0],
                    use_cuda_graph=True,
                )
            )
            self._scratch = tuple(
                torch.empty(shape, dtype=dtype, device=self.q.device)
                for shape, dtype in self._scratch_plan.shapes_and_dtypes()
            )
            self._scratch_plan.prepare_decode_graph_replay_state(
                batch=batch,
                max_page_table_width=page_table.shape[1],
                force_split_kv=True,
            )
        self._page_table = page_table
        self._cache_seqlens = cache_seqlens
        self._cu_seqlens_q = cu_seqlens_q
        if self._output is not None:
            self._bind()

    def _bind(self):
        binding = self._scratch_plan.bind(
            scratch=self._scratch,
            q=self.q,
            k_cache=self.k_cache,
            v_cache=self.v_cache,
            output=self._output,
            page_table=self._page_table,
            cache_seqlens=self._cache_seqlens,
            cu_seqlens_q=self._cu_seqlens_q,
            k_descale=None,
            v_descale=None,
        )
        self._last_scratch = binding.scratch
        return binding

    def run(self, output):
        self._output = output
        return paged_attention_forward(binding=self._bind())


def _run_forced_split_decode_graph(
    *,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    cache_seqlens_capture: list[int],
    cache_seqlens_replay: list[int] | None = None,
    seed: int = 73,
    num_pages: int = 4096,
) -> None:
    clear_attention_caches()
    batch = len(cache_seqlens_capture)
    max_len = max(cache_seqlens_capture + (cache_seqlens_replay or []))
    page_size = 64
    width = max_len // page_size + 2
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = make_paged_inputs(
        q_seqlens=[1] * batch,
        cache_seqlens=cache_seqlens_capture,
        page_size=page_size,
        seed=seed,
        q_heads=q_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        page_table_width=width,
        num_pages=num_pages,
    )
    harness = _ForcedSplitDecodeGraphHarness(q, k_cache, v_cache)
    harness.prepare(page_table, cache_seqlens, cu_seqlens_q)
    output = torch.empty_like(q)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        harness.run(output)

    ref_out, _ = paged_attention_reference(
        q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q, causal=True
    )
    graph.replay()
    torch.cuda.synchronize()
    assert (output - ref_out).abs().max().item() <= 0.02

    if cache_seqlens_replay is None:
        return

    q2, k2, v2, pt2, csl2, cu2 = make_paged_inputs(
        q_seqlens=[1] * batch,
        cache_seqlens=cache_seqlens_replay,
        page_size=page_size,
        seed=seed + 6,
        q_heads=q_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        page_table_width=page_table.shape[1],
        num_pages=k_cache.shape[0],
    )
    # keep metadata tensor addresses stable; update contents in place
    q.copy_(q2)
    k_cache.copy_(k2)
    v_cache.copy_(v2)
    page_table.copy_(pt2)
    cache_seqlens.copy_(csl2)
    cu_seqlens_q.copy_(cu2)
    harness.prepare(page_table, cache_seqlens, cu_seqlens_q)

    ref_out2, _ = paged_attention_reference(
        q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q, causal=True
    )
    graph.replay()
    torch.cuda.synchronize()
    assert (output - ref_out2).abs().max().item() <= 0.02


@torch.inference_mode()
def test_forced_split_decode_graph_gqa16_head_dim128() -> None:
    """MiniMax-M3 dense-layer shape: 64 q heads / 4 KV heads / head_dim 128."""
    require_b12x()
    _run_forced_split_decode_graph(
        q_heads=64,
        kv_heads=4,
        head_dim=128,
        cache_seqlens_capture=[4096, 4096, 4096, 4096],
        cache_seqlens_replay=[512, 1536, 3072, 4096],
    )


@torch.inference_mode()
def test_forced_split_decode_graph_gqa12_head_dim128() -> None:
    """gqa 12: second MMA row slot partially occupied."""
    require_b12x()
    _run_forced_split_decode_graph(
        q_heads=48,
        kv_heads=4,
        head_dim=128,
        cache_seqlens_capture=[4096, 4096, 4096, 4096],
    )


@torch.inference_mode()
def test_forced_split_decode_graph_gqa8_head_dim256() -> None:
    """gqa 8 / head_dim 256 control (single row slot fast path)."""
    require_b12x()
    _run_forced_split_decode_graph(
        q_heads=8,
        kv_heads=1,
        head_dim=256,
        cache_seqlens_capture=[4096, 4096, 4096, 4096],
        cache_seqlens_replay=[512, 1536, 3072, 4096],
    )


@torch.inference_mode()
def test_qwen38_gqa6_head_dim256_split_decode_graph_matches_reference() -> None:
    """Qwen3.8 TP1 full attention: 24 Q heads / 4 KV heads / H256."""
    require_b12x()
    _run_forced_split_decode_graph(
        q_heads=24,
        kv_heads=4,
        head_dim=256,
        cache_seqlens_capture=[128],
        cache_seqlens_replay=[65],
        num_pages=8,
    )


@torch.inference_mode()
def test_forced_split_decode_graph_gqa16_head_dim128_batch2() -> None:
    """batch != 4 exercises the bf16/128 regular-decode merge bdy=3 config."""
    require_b12x()
    _run_forced_split_decode_graph(
        q_heads=64,
        kv_heads=4,
        head_dim=128,
        cache_seqlens_capture=[4096, 2048],
        cache_seqlens_replay=[1024, 4096],
    )


@torch.inference_mode()
def test_direct_decode_graph_gqa128_writes_every_query_tile() -> None:
    """GQA128 / CTA16 must consume all eight q tiles and write every head."""
    device = require_b12x()
    clear_attention_caches()
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = (
        make_paged_inputs(
            q_seqlens=[1],
            cache_seqlens=[193],
            page_size=64,
            seed=197,
            q_heads=128,
            kv_heads=1,
            head_dim=256,
            page_table_width=4,
            num_pages=8,
        )
    )
    bootstrap_plan = create_paged_plan(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        mode="decode",
        disable_split_kv=True,
        enable_cuda_graph=True,
        graph_chunk_policy=True,
    )
    assert bootstrap_plan.cta_tile_q == 16
    assert tuple(bootstrap_plan.qo_tile_indices) == tuple(range(8))

    scratch_plan = plan_paged_attention_scratch(
        B12XPagedAttentionScratchCaps(
            device=device,
            mode="decode",
            dtype=q.dtype,
            kv_dtype=k_cache.dtype,
            num_q_heads=q.shape[1],
            num_kv_heads=k_cache.shape[2],
            head_dim_qk=q.shape[2],
            head_dim_vo=v_cache.shape[3],
            page_size=k_cache.shape[1],
            max_total_q=1,
            max_batch=1,
            max_page_table_width=page_table.shape[1],
            max_work_items=8,
            max_partial_rows=0,
            num_cache_pages=k_cache.shape[0],
            use_cuda_graph=True,
            copy_runtime_metadata=True,
        )
    )
    scratch_plan.prepare_decode_graph_replay_state(
        batch=1,
        total_q_capacity=1,
        max_page_table_width=page_table.shape[1],
        max_cache_page_count=page_table.shape[1],
        force_split_kv=False,
    )
    scratch = tuple(
        torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
        for spec in scratch_plan.scratch_specs()
    )
    output = torch.full_like(q, torch.nan)

    def bind():
        return scratch_plan.bind(
            scratch=scratch,
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            output=output,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            disable_split_kv=True,
            active_total_q=1,
        )

    warm_binding = bind()
    warm_binding.run()
    torch.cuda.synchronize(device)
    assert warm_binding.scratch.plan.split_kv is False
    assert warm_binding.scratch.plan.cta_tile_q == 16
    assert tuple(warm_binding.scratch.plan.qo_tile_indices) == tuple(range(8))

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_binding = bind()
        _, captured_lse = captured_binding.run()

    # Poison every destination after capture.  The old qtile-blind kernel
    # rewrote heads 0..63 twice and left the upper half poisoned.
    output.fill_(torch.nan)
    captured_lse.fill_(torch.nan)
    graph.replay()
    torch.cuda.synchronize(device)

    reference_output, reference_lse = paged_attention_reference(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        causal=True,
    )
    assert torch.isfinite(output).all().item()
    assert torch.isfinite(captured_lse).all().item()
    assert output[:, :64].abs().max().item() > 0
    assert output[:, 64:].abs().max().item() > 0
    torch.testing.assert_close(
        output.to(torch.float32),
        reference_output.to(torch.float32),
        atol=3e-2,
        rtol=3e-2,
    )
    torch.testing.assert_close(
        captured_lse.to(torch.float32) * math.log(2.0),
        reference_lse.to(torch.float32),
        atol=5e-2,
        rtol=5e-2,
    )
