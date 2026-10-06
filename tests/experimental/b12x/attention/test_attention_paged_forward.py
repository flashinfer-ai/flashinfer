from __future__ import annotations

import math

import pytest
import torch

from b12x._lib.runtime_control import kernel_resolution_guard
from benchmarks.experimental.b12x.benchmark_paged_attention import (
    _capture_backend_graph,
    _make_uniform_paged_inputs,
    _quantize_paged_kv_cache_global_e4m3,
)
from b12x.attention.paged.reference import paged_attention_reference
from b12x.attention.paged._forward import _build_extend_forward_kernel
from b12x.attention.paged.traits import select_paged_forward_traits_from_plan
from b12x.attention._shared.contiguous.api import clear_attention_caches
from b12x.attention.paged._forward import paged_attention_forward
from b12x.attention.paged._scratch import B12XPagedAttentionScratchCaps, plan_paged_attention_scratch
from b12x.attention.paged.planner import create_paged_plan

from b12x.testing.reference.helpers import require_b12x
from b12x.testing.reference.paged_attention_helpers import quantize_paged_kv_cache_e4m3
from .test_attention_paged_planner import _make_inputs


_ALLOCATOR_COUNTERS = (
    "allocation.all.allocated",
    "allocation.all.freed",
    "segment.all.allocated",
    "segment.all.freed",
    "num_alloc_retries",
    "num_ooms",
)


def _allocator_counters(device: torch.device) -> dict[str, int]:
    stats = torch.cuda.memory_stats(device)
    return {name: int(stats.get(name, 0)) for name in _ALLOCATOR_COUNTERS}


def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f = a.to(torch.float32).reshape(-1)
    b_f = b.to(torch.float32).reshape(-1)
    return torch.nn.functional.cosine_similarity(a_f, b_f, dim=0).item()


class _PagedScratchHarness:
    def __init__(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        *,
        mode: str,
    ) -> None:
        self.q = q
        self.k_cache = k_cache
        self.v_cache = v_cache
        self.mode = mode
        self.plan = None
        self._scratch_plan = None
        self._scratch = None
        self._prepare_kwargs = {}
        self._page_table = None
        self._cache_seqlens = None
        self._cu_seqlens_q = None

    def prepare(
        self,
        page_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        **kwargs,
    ) -> None:
        self.plan = create_paged_plan(
            self.q,
            self.k_cache,
            self.v_cache,
            page_table,
            cache_seqlens,
            cu_seqlens_q,
            mode=self.mode,
            **kwargs,
        )
        self._scratch_plan = plan_paged_attention_scratch(
            B12XPagedAttentionScratchCaps(
                device=self.q.device,
                mode=self.mode,
                dtype=self.q.dtype,
                kv_dtype=self.k_cache.dtype,
                num_q_heads=self.q.shape[1],
                num_kv_heads=self.k_cache.shape[2],
                head_dim_qk=self.q.shape[2],
                head_dim_vo=self.v_cache.shape[3],
                page_size=self.k_cache.shape[1],
                max_total_q=self.plan.total_q,
                max_batch=page_table.shape[0],
                max_page_table_width=page_table.shape[1],
                max_work_items=max(self.plan.new_batch_size, 1),
                max_partial_rows=self.plan.total_num_partial_rows,
                num_cache_pages=self.k_cache.shape[0],
            )
        )
        self._scratch = tuple(
            torch.empty(shape, dtype=dtype, device=self.q.device)
            for shape, dtype in self._scratch_plan.shapes_and_dtypes()
        )
        self._prepare_kwargs = dict(kwargs)
        self._page_table = page_table
        self._cache_seqlens = cache_seqlens
        self._cu_seqlens_q = cu_seqlens_q

    def run(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        *,
        output: torch.Tensor,
        k_descale: torch.Tensor | None = None,
        v_descale: torch.Tensor | None = None,
        attention_sink_bias: torch.Tensor | None = None,
        relative_attention_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self._scratch_plan is not None
        assert self._scratch is not None
        binding = self._scratch_plan.bind(
            scratch=self._scratch,
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            output=output,
            page_table=self._page_table,
            cache_seqlens=self._cache_seqlens,
            cu_seqlens_q=self._cu_seqlens_q,
            active_total_q=q.shape[0],
            k_descale=k_descale,
            v_descale=v_descale,
            attention_sink_bias=attention_sink_bias,
            relative_attention_bias=relative_attention_bias,
            **self._prepare_kwargs,
        )
        self.plan = binding.scratch.plan
        return paged_attention_forward(binding=binding)


def _make_workspace(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    mode: str,
) -> _PagedScratchHarness:
    return _PagedScratchHarness(q, k_cache, v_cache, mode=mode)


def _run_decode_graph_check(
    *,
    batch: int = 8,
    cache_seqlen: int,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    (
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        capture_page_table,
        capture_cache_seqlens,
        cu_seqlens_q,
    ) = _make_uniform_paged_inputs(
        batch=batch,
        q_seqlen=1,
        cache_seqlen=cache_seqlen,
        capture_cache_seqlen=None,
        page_size=64,
        q_heads=8,
        kv_heads=1,
        head_dim=256,
        dtype=torch.bfloat16,
        seed=1,
    )
    k_fp8, v_fp8, k_descale, v_descale, k_scale, v_scale = _quantize_paged_kv_cache_global_e4m3(
        k_cache,
        v_cache,
        batch=batch,
        kv_heads=1,
    )
    backend = _capture_backend_graph(
        q=q,
        k_cache=k_fp8,
        v_cache=v_fp8,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        capture_page_table=capture_page_table,
        capture_cache_seqlens=capture_cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        fixed_split_pages=None,
        k_descale=k_descale,
        v_descale=v_descale,
        warmup=1,
        graph_ctas_per_sm=None,
        window_left=-1,
    )
    reference, _ = paged_attention_reference(
        q, k_fp8, v_fp8, page_table, cache_seqlens, cu_seqlens_q,
        k_descale=k_descale, v_descale=v_descale, causal=True,
    )
    return backend.output, reference, backend.plan_desc


def _run_decode_reference_check(
    *,
    batch: int = 8,
    cache_seqlen: int,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    (
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        capture_page_table,
        capture_cache_seqlens,
        cu_seqlens_q,
    ) = _make_uniform_paged_inputs(
        batch=batch,
        q_seqlen=1,
        cache_seqlen=cache_seqlen,
        capture_cache_seqlen=None,
        page_size=64,
        q_heads=8,
        kv_heads=1,
        head_dim=256,
        dtype=torch.bfloat16,
        seed=1,
    )
    k_fp8, v_fp8, k_descale, v_descale, _k_scale, _v_scale = _quantize_paged_kv_cache_global_e4m3(
        k_cache,
        v_cache,
        batch=batch,
        kv_heads=1,
    )
    backend = _capture_backend_graph(
        q=q,
        k_cache=k_fp8,
        v_cache=v_fp8,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        capture_page_table=capture_page_table,
        capture_cache_seqlens=capture_cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        fixed_split_pages=None,
        k_descale=k_descale,
        v_descale=v_descale,
        warmup=1,
        graph_ctas_per_sm=None,
        window_left=-1,
    )
    backend.graph.replay()
    torch.cuda.synchronize()
    ref_out, _ref_lse = paged_attention_reference(
        q,
        k_fp8,
        v_fp8,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        k_descale=k_descale,
        v_descale=v_descale,
        causal=True,
    )
    return backend.output, ref_out, backend.plan_desc


@torch.inference_mode()
def test_paged_forward_matches_reference_decode_short_context() -> None:
    require_b12x()
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_inputs(
        q_seqlens=[1, 1, 1],
        cache_seqlens=[64, 128, 192],
        dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
    )
    workspace = _make_workspace(q, k_cache, v_cache, mode="decode")
    workspace.prepare(page_table, cache_seqlens, cu_seqlens_q)
    output, lse_base2 = workspace.run(
        q,
        k_cache,
        v_cache,
        output=torch.empty_like(q),
    )
    torch.cuda.synchronize()

    ref_out, ref_lse = paged_attention_reference(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        causal=True,
    )
    lse_natural = lse_base2 * math.log(2.0)
    assert (output - ref_out).abs().max().item() <= 0.03
    assert (lse_natural - ref_lse).abs().max().item() <= 0.05
    assert _cosine_similarity(output, ref_out) >= 0.99999


@torch.inference_mode()
def test_qwen38_gqa6_head_dim256_decode_matches_reference() -> None:
    """Qwen3.8 full-attention geometry must avoid the H128 GQA6 row fast path."""
    require_b12x()
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_inputs(
        q_seqlens=[1],
        cache_seqlens=[128],
        q_heads=24,
        kv_heads=4,
        head_dim_qk=256,
        head_dim_vo=256,
        dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
    )
    workspace = _make_workspace(q, k_cache, v_cache, mode="decode")
    workspace.prepare(
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        disable_split_kv=True,
    )
    output, _lse = workspace.run(
        q,
        k_cache,
        v_cache,
        output=torch.empty_like(q),
    )
    torch.cuda.synchronize()

    ref_out, _ref_lse = paged_attention_reference(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        causal=True,
    )
    assert (output - ref_out).abs().max().item() <= 0.03
    assert _cosine_similarity(output, ref_out) >= 0.99999


@torch.inference_mode()
def test_paged_forward_matches_reference_decode_dense_page128() -> None:
    require_b12x()
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_inputs(
        q_seqlens=[1, 1],
        cache_seqlens=[200, 384],
        page_size=128,
        q_heads=64,
        kv_heads=4,
        head_dim_qk=128,
        head_dim_vo=128,
        dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
    )
    workspace = _make_workspace(q, k_cache, v_cache, mode="decode")
    workspace.prepare(page_table, cache_seqlens, cu_seqlens_q)
    assert workspace.plan.page_size == 128
    assert workspace.plan.msa_block_sparse is False
    output, lse_base2 = workspace.run(
        q,
        k_cache,
        v_cache,
        output=torch.empty_like(q),
    )
    torch.cuda.synchronize()

    ref_out, ref_lse = paged_attention_reference(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        causal=True,
    )
    lse_natural = lse_base2 * math.log(2.0)
    assert (output - ref_out).abs().max().item() <= 0.03
    assert (lse_natural - ref_lse).abs().max().item() <= 0.05
    assert _cosine_similarity(output, ref_out) >= 0.99999


@torch.inference_mode()
def test_paged_forward_matches_reference_fp8_decode_short_context_batch8() -> None:
    require_b12x()
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_inputs(
        q_seqlens=[1, 1, 1, 1, 1, 1, 1, 1],
        cache_seqlens=[64, 64, 64, 64, 64, 64, 64, 64],
        dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
    )
    k_fp8, v_fp8, k_descale, v_descale = quantize_paged_kv_cache_e4m3(
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
    )
    workspace = _make_workspace(q, k_fp8, v_fp8, mode="decode")
    workspace.prepare(page_table, cache_seqlens, cu_seqlens_q)
    output, lse_base2 = workspace.run(
        q,
        k_fp8,
        v_fp8,
        output=torch.empty_like(q),
        k_descale=k_descale,
        v_descale=v_descale,
    )
    torch.cuda.synchronize()

    ref_out, ref_lse = paged_attention_reference(
        q,
        k_fp8,
        v_fp8,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        k_descale=k_descale,
        v_descale=v_descale,
        causal=True,
    )
    lse_natural = lse_base2 * math.log(2.0)
    assert (output - ref_out).abs().max().item() <= 0.05
    assert (lse_natural - ref_lse).abs().max().item() <= 0.08
    assert _cosine_similarity(output, ref_out) >= 0.999


@torch.inference_mode()
def test_paged_forward_matches_reference_decode_with_sliding_window_and_sink() -> None:
    require_b12x()
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_inputs(
        q_seqlens=[1, 1, 1],
        cache_seqlens=[128, 192, 256],
        dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
    )
    window_left = 80
    attention_sink_bias = torch.linspace(-0.2, 0.2, q.shape[1], dtype=torch.float32, device=q.device)
    workspace = _make_workspace(q, k_cache, v_cache, mode="decode")
    workspace.prepare(page_table, cache_seqlens, cu_seqlens_q, window_left=window_left)
    output, lse_base2 = workspace.run(
        q,
        k_cache,
        v_cache,
        output=torch.empty_like(q),
        attention_sink_bias=attention_sink_bias,
    )
    torch.cuda.synchronize()

    ref_out, ref_lse = paged_attention_reference(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        causal=True,
        window_left=window_left,
        attention_sink_bias=attention_sink_bias,
    )
    lse_natural = lse_base2 * math.log(2.0)
    assert (output - ref_out).abs().max().item() <= 0.03
    assert (lse_natural - ref_lse).abs().max().item() <= 0.05
    assert _cosine_similarity(output, ref_out) >= 0.99999


@torch.inference_mode()
def test_paged_forward_matches_reference_decode_with_relative_bias() -> None:
    require_b12x()
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_inputs(
        q_seqlens=[1, 1, 1],
        cache_seqlens=[128, 256, 384],
        page_size=128,
        q_heads=8,
        kv_heads=1,
        head_dim_qk=128,
        head_dim_vo=128,
        dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
    )
    relative_attention_bias = torch.randn(
        q.shape[0],
        q.shape[1],
        1024,
        dtype=q.dtype,
        device=q.device,
    )
    workspace = _make_workspace(q, k_cache, v_cache, mode="decode")
    workspace.prepare(page_table, cache_seqlens, cu_seqlens_q)
    output, lse_base2 = workspace.run(
        q,
        k_cache,
        v_cache,
        output=torch.empty_like(q),
        relative_attention_bias=relative_attention_bias,
    )
    torch.cuda.synchronize()

    ref_out, ref_lse = paged_attention_reference(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        causal=True,
        relative_attention_bias=relative_attention_bias,
    )
    lse_natural = lse_base2 * math.log(2.0)
    assert (output - ref_out).abs().max().item() <= 0.03
    assert (lse_natural - ref_lse).abs().max().item() <= 0.05
    assert _cosine_similarity(output, ref_out) >= 0.99999


@pytest.mark.parametrize(("mode", "q_seqlen"), [("decode", 1), ("extend", 6)])
@torch.inference_mode()
def test_paged_forward_large_relative_bias_is_numerically_stable(
    mode: str,
    q_seqlen: int,
) -> None:
    require_b12x()
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_inputs(
        q_seqlens=[q_seqlen],
        cache_seqlens=[222],
        page_size=128,
        q_heads=8,
        kv_heads=2,
        head_dim_qk=128,
        head_dim_vo=128,
        dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
    )
    combined_cache = torch.empty(
        k_cache.shape[0],
        2,
        *k_cache.shape[1:],
        dtype=k_cache.dtype,
        device=k_cache.device,
    )
    combined_cache[:, 0].copy_(k_cache)
    combined_cache[:, 1].copy_(v_cache)
    k_cache, v_cache = combined_cache.unbind(dim=1)

    bias_generator = torch.Generator(device=q.device).manual_seed(17)
    relative_attention_bias = (
        torch.randn(
            q.shape[0],
            q.shape[1],
            512,
            dtype=torch.float32,
            device=q.device,
            generator=bias_generator,
        )
        .mul_(1.0e9)
        .to(q.dtype)
    )
    workspace = _make_workspace(q, k_cache, v_cache, mode=mode)
    workspace.prepare(
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        disable_split_kv=True,
        window_left=511,
    )
    output, lse_base2 = workspace.run(
        q,
        k_cache,
        v_cache,
        output=torch.empty_like(q),
        relative_attention_bias=relative_attention_bias,
    )
    torch.cuda.synchronize()

    ref_out, _ = paged_attention_reference(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        causal=True,
        window_left=511,
        relative_attention_bias=relative_attention_bias,
    )
    assert torch.isfinite(output).all().item()
    assert torch.isfinite(lse_base2).all().item()
    torch.testing.assert_close(
        output.to(torch.float32),
        ref_out.to(torch.float32),
        atol=3e-2,
        rtol=3e-2,
    )
    assert _cosine_similarity(output, ref_out) >= 0.99999


@pytest.mark.parametrize(
    ("kv_heads", "window_left", "relative_extent"),
    [
        (1, -1, 1024),
        (2, 511, 512),
    ],
)
@torch.inference_mode()
def test_paged_forward_decode_graph_replays_with_relative_bias(
    kv_heads: int,
    window_left: int,
    relative_extent: int,
) -> None:
    require_b12x()
    clear_attention_caches()
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_inputs(
        q_seqlens=[1],
        cache_seqlens=[384],
        page_size=128,
        q_heads=8,
        kv_heads=kv_heads,
        head_dim_qk=128,
        head_dim_vo=128,
        dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
    )
    max_page_table_width = 8192
    capture_page_table = torch.empty(
        (1, max_page_table_width),
        dtype=page_table.dtype,
        device=page_table.device,
    )
    capture_page_table.copy_(page_table[:, -1:])
    capture_page_table[:, : page_table.shape[1]].copy_(page_table)
    page_table = capture_page_table
    relative_attention_bias = torch.randn(
        1,
        q.shape[1],
        relative_extent,
        dtype=q.dtype,
        device=q.device,
    )
    scratch_plan = plan_paged_attention_scratch(
        B12XPagedAttentionScratchCaps(
            device=q.device,
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
            max_page_table_width=max_page_table_width,
            max_work_items=512,
            max_partial_rows=0,
            num_cache_pages=max_page_table_width,
            use_cuda_graph=True,
            copy_runtime_metadata=True,
        )
    )
    scratch_plan.prepare_decode_graph_replay_state(
        batch=1,
        total_q_capacity=1,
        max_page_table_width=max_page_table_width,
        max_cache_page_count=max_page_table_width,
        window_left=window_left,
    )
    (scratch_spec,) = scratch_plan.scratch_specs()
    scratch = torch.empty(
        scratch_spec.shape,
        dtype=scratch_spec.dtype,
        device=scratch_spec.device,
    )
    output = torch.empty_like(q)

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
            window_left=window_left,
            active_total_q=1,
            relative_attention_bias=relative_attention_bias,
        )

    bind().run()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        binding = bind()
        binding.run()

    for cache_seqlen in range(1, 385):
        cache_seqlens.fill_(cache_seqlen)
        graph.replay()
    torch.cuda.synchronize()

    ref_out, _ = paged_attention_reference(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        causal=True,
        window_left=window_left,
        relative_attention_bias=relative_attention_bias,
    )
    assert binding.scratch.plan.split_kv is False
    assert torch.allclose(
        output.to(torch.float32),
        ref_out.to(torch.float32),
        atol=3e-2,
        rtol=3e-2,
    )
    assert _cosine_similarity(output, ref_out) >= 0.99999


@torch.inference_mode()
def test_paged_forward_matches_reference_decode_mimo_gqa_shape_with_sliding_window_and_sink() -> None:
    require_b12x()
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_inputs(
        q_seqlens=[1, 1, 1],
        cache_seqlens=[128, 192, 256],
        q_heads=64,
        kv_heads=8,
        head_dim_qk=192,
        head_dim_vo=128,
        dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
    )
    window_left = 80
    attention_sink_bias = torch.linspace(-0.2, 0.2, q.shape[1], dtype=torch.float32, device=q.device)
    workspace = _make_workspace(q, k_cache, v_cache, mode="decode")
    workspace.prepare(page_table, cache_seqlens, cu_seqlens_q, window_left=window_left)
    output, lse_base2 = workspace.run(
        q,
        k_cache,
        v_cache,
        output=torch.empty(q.shape[0], q.shape[1], v_cache.shape[3], dtype=q.dtype, device=q.device),
        attention_sink_bias=attention_sink_bias,
    )
    torch.cuda.synchronize()

    ref_out, ref_lse = paged_attention_reference(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        causal=True,
        window_left=window_left,
        attention_sink_bias=attention_sink_bias,
    )
    lse_natural = lse_base2 * math.log(2.0)
    assert (output - ref_out).abs().max().item() <= 0.02
    assert (lse_natural - ref_lse).abs().max().item() <= 0.03
    assert _cosine_similarity(output, ref_out) >= 0.9999


@torch.inference_mode()
def test_paged_forward_attention_sink_affects_denominator_only() -> None:
    require_b12x()
    q_heads = 8
    kv_heads = 1
    head_dim = 256
    page_size = 64
    q = torch.zeros((1, q_heads, head_dim), dtype=torch.bfloat16, device="cuda")
    k_cache = torch.zeros((1, page_size, kv_heads, head_dim), dtype=torch.bfloat16, device="cuda")
    v_cache = torch.zeros((1, page_size, kv_heads, head_dim), dtype=torch.bfloat16, device="cuda")
    v_cache[:, 0, :, :].fill_(1.0)
    page_table = torch.zeros((1, 1), dtype=torch.int32, device="cuda")
    cache_seqlens = torch.ones((1,), dtype=torch.int32, device="cuda")
    cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    attention_sink_bias = torch.full((q_heads,), math.log(3.0), dtype=torch.float32, device="cuda")

    workspace = _make_workspace(q, k_cache, v_cache, mode="decode")
    workspace.prepare(page_table, cache_seqlens, cu_seqlens_q)
    output, lse_base2 = workspace.run(
        q,
        k_cache,
        v_cache,
        output=torch.empty_like(q),
        attention_sink_bias=attention_sink_bias,
    )
    torch.cuda.synchronize()

    expected_output = torch.full_like(output, 0.25)
    expected_lse = torch.full((1, q_heads), math.log(4.0), dtype=torch.float32, device="cuda")
    lse_natural = lse_base2 * math.log(2.0)
    assert (output - expected_output).abs().max().item() <= 0.002
    assert (lse_natural - expected_lse).abs().max().item() <= 0.002


@torch.inference_mode()
def test_paged_forward_native_fp8_qkv_matches_reference_fp8_decode_short_context_batch8(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    require_b12x()
    monkeypatch.setenv("B12X_TURBO_ATTN", "1")
    output, ref_out, _plan_desc = _run_decode_reference_check(cache_seqlen=64)
    assert (output - ref_out).abs().max().item() <= 0.02
    assert _cosine_similarity(output, ref_out) >= 0.995


@torch.inference_mode()
def test_paged_forward_matches_reference_without_split_bf16_extend() -> None:
    require_b12x()
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_inputs(
        q_seqlens=[6, 5],
        cache_seqlens=[64, 64],
        dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
    )
    workspace = _make_workspace(q, k_cache, v_cache, mode="extend")
    workspace.prepare(
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        disable_split_kv=True,
    )
    output, lse_base2 = workspace.run(q, k_cache, v_cache, output=torch.empty_like(q))
    torch.cuda.synchronize()

    ref_out, ref_lse = paged_attention_reference(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        causal=True,
    )
    lse_natural = lse_base2 * math.log(2.0)
    assert (output - ref_out).abs().max().item() <= 0.03
    assert (lse_natural - ref_lse).abs().max().item() <= 0.05
    assert _cosine_similarity(output, ref_out) >= 0.99999


@torch.inference_mode()
def test_paged_forward_bf16_extend_dual_tma_tail_matches_reference() -> None:
    """Replay the one-stage BF16 K/V TMA specialization as a serving graph."""
    require_b12x()
    device = torch.device("cuda")
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_inputs(
        q_seqlens=[4],
        cache_seqlens=[4096],
        dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
    )
    workspace = _make_workspace(q, k_cache, v_cache, mode="extend")
    workspace.prepare(
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        disable_split_kv=True,
    )

    traits = select_paged_forward_traits_from_plan(workspace.plan)
    kernel = _build_extend_forward_kernel(
        traits,
        False,
        False,
        workspace.plan.window_left,
        False,
        False,
        False,
        False,
        int(workspace.plan.page_size),
    )
    assert traits.cta_tile_q == 16
    assert traits.cta_tile_kv == 64
    assert traits.num_warps_q == 1
    assert traits.num_warps_kv == 4
    assert kernel.num_stages == 1
    assert kernel.use_paged_k_tma is True
    assert kernel.use_paged_v_tma is True
    assert kernel.use_paged_kv_tma_fp8_raw_issue is False

    scenario_q = (
        q.clone(),
        q.float().mul(-0.75).add(0.03125).to(torch.bfloat16),
    )
    scenario_k = (
        k_cache.clone(),
        k_cache.float().mul(0.5).add(0.015625).to(torch.bfloat16),
    )
    scenario_v = (
        v_cache.clone(),
        v_cache.float().mul(-0.5).add(0.0625).to(torch.bfloat16),
    )
    assert not torch.equal(scenario_q[0], scenario_q[1])
    assert not torch.equal(scenario_k[0], scenario_k[1])
    assert not torch.equal(scenario_v[0], scenario_v[1])

    metadata = {
        "page_table": page_table,
        "cache_seqlens": cache_seqlens,
        "cu_seqlens_q": cu_seqlens_q,
    }
    metadata_snapshots = {name: tensor.clone() for name, tensor in metadata.items()}

    def install_scenario(index: int) -> None:
        q.copy_(scenario_q[index])
        k_cache.copy_(scenario_k[index])
        v_cache.copy_(scenario_v[index])

    output = torch.empty_like(q)
    assert workspace._scratch_plan is not None
    assert workspace._scratch is not None
    binding = workspace._scratch_plan.bind(
        scratch=workspace._scratch,
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        output=output,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        active_total_q=q.shape[0],
        disable_split_kv=True,
    )
    workspace.plan = binding.scratch.plan

    def run_bound() -> tuple[torch.Tensor, torch.Tensor]:
        return paged_attention_forward(binding=binding)

    install_scenario(0)
    warm_output, warm_lse = run_bound()
    torch.cuda.synchronize(device)
    assert warm_output.data_ptr() == output.data_ptr()
    assert bool(torch.isfinite(warm_lse).all().item())

    with kernel_resolution_guard('BF16 paged dual-TMA serving replay must use the warmed specialization'):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured_output, captured_lse = run_bound()

    assert captured_output.data_ptr() == output.data_ptr()
    assert captured_lse.data_ptr() == warm_lse.data_ptr()
    stable_tensors = {
        "q": q,
        "k_cache": k_cache,
        "v_cache": v_cache,
        **metadata,
        "output": output,
        "lse": captured_lse,
        **{
            f"scratch_{index}": tensor
            for index, tensor in enumerate(workspace._scratch)
        },
    }
    stable_addresses = {
        name: tensor.data_ptr() for name, tensor in stable_tensors.items()
    }

    scenario_outputs: list[torch.Tensor] = []
    for index in range(2):
        install_scenario(index)
        ref_out, ref_lse = paged_attention_reference(
            q,
            k_cache,
            v_cache,
            page_table,
            cache_seqlens,
            cu_seqlens_q,
            causal=True,
        )
        torch.cuda.synchronize(device)

        output.fill_(float("nan"))
        captured_lse.fill_(float("nan"))
        torch.cuda.synchronize(device)
        allocator_before = _allocator_counters(device)
        graph.replay()
        torch.cuda.synchronize(device)
        allocator_after = _allocator_counters(device)

        assert allocator_after == allocator_before
        assert {
            name: tensor.data_ptr() for name, tensor in stable_tensors.items()
        } == stable_addresses
        assert bool(torch.isfinite(output).all().item())
        assert bool(torch.isfinite(captured_lse).all().item())
        assert bool((output != 0).any().item())
        torch.testing.assert_close(q, scenario_q[index], rtol=0.0, atol=0.0)
        torch.testing.assert_close(k_cache, scenario_k[index], rtol=0.0, atol=0.0)
        torch.testing.assert_close(v_cache, scenario_v[index], rtol=0.0, atol=0.0)
        for name, tensor in metadata.items():
            torch.testing.assert_close(
                tensor,
                metadata_snapshots[name],
                rtol=0.0,
                atol=0.0,
            )

        lse_natural = captured_lse * math.log(2.0)
        assert (output - ref_out).abs().max().item() <= 0.03
        assert (lse_natural - ref_lse).abs().max().item() <= 0.05
        assert _cosine_similarity(output, ref_out) >= 0.99999
        scenario_outputs.append(output.clone())

    assert not torch.equal(scenario_outputs[0], scenario_outputs[1])


@torch.inference_mode()
def test_paged_forward_matches_reference_extend_dense_page128() -> None:
    require_b12x()
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_inputs(
        q_seqlens=[6, 5],
        cache_seqlens=[256, 384],
        page_size=128,
        q_heads=64,
        kv_heads=4,
        head_dim_qk=128,
        head_dim_vo=128,
        dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
    )
    workspace = _make_workspace(q, k_cache, v_cache, mode="extend")
    workspace.prepare(
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        disable_split_kv=True,
    )
    assert workspace.plan.page_size == 128
    assert workspace.plan.msa_block_sparse is False
    output, lse_base2 = workspace.run(q, k_cache, v_cache, output=torch.empty_like(q))
    torch.cuda.synchronize()

    ref_out, ref_lse = paged_attention_reference(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        causal=True,
    )
    lse_natural = lse_base2 * math.log(2.0)
    assert (output - ref_out).abs().max().item() <= 0.03
    assert (lse_natural - ref_lse).abs().max().item() <= 0.05
    assert _cosine_similarity(output, ref_out) >= 0.99999


@torch.inference_mode()
def test_paged_extend_dense_page128_compile_key_uses_fixed_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    require_b12x()
    clear_attention_caches()

    import b12x.attention.paged._forward as paged_api

    forward_specs: list[str] = []

    def fake_launch(
        _func,
        *,
        compile_spec,
        compile_args,
        runtime_args,
        compile_kwargs=None,
    ):
        if compile_spec.kernel_id == "attention.paged.forward":
            forward_specs.append(repr(compile_spec))

    monkeypatch.setattr(paged_api, "b12x_launch", fake_launch)

    batch = 4
    page_size = 128
    page_table_width = 4
    num_cache_pages = 64
    max_total_q = 64
    q_heads = 16
    kv_heads = 1
    head_dim = 128
    device = torch.device("cuda")
    dtype = torch.bfloat16

    def make_kv_cache(pages: int) -> tuple[torch.Tensor, torch.Tensor]:
        k_cache = torch.empty(
            pages,
            page_size,
            kv_heads,
            head_dim,
            dtype=dtype,
            device=device,
        )
        v_cache = torch.empty_like(k_cache)
        return k_cache, v_cache

    scratch_plan = plan_paged_attention_scratch(
        B12XPagedAttentionScratchCaps(
            device=device,
            mode="extend",
            dtype=dtype,
            kv_dtype=dtype,
            num_q_heads=q_heads,
            num_kv_heads=kv_heads,
            head_dim_qk=head_dim,
            head_dim_vo=head_dim,
            page_size=page_size,
            max_total_q=max_total_q,
            max_batch=batch,
            max_page_table_width=page_table_width,
            max_work_items=128,
            max_partial_rows=0,
            num_cache_pages=num_cache_pages,
        )
    )
    scratch = tuple(
        torch.empty(shape, dtype=scratch_dtype, device=device)
        for shape, scratch_dtype in scratch_plan.shapes_and_dtypes()
    )

    def make_metadata(
        q_lens: list[int],
        cache_lens: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        total_q = sum(q_lens)
        q = torch.randn(total_q, q_heads, head_dim, dtype=dtype, device=device)
        page_table = torch.empty(
            batch, page_table_width, dtype=torch.int32, device=device
        )
        for req_idx, cache_len in enumerate(cache_lens):
            req_pages = math.ceil(cache_len / page_size)
            assert req_pages <= page_table_width
            page_ids = torch.arange(
                req_idx * page_table_width,
                req_idx * page_table_width + req_pages,
                dtype=torch.int32,
                device=device,
            )
            page_table[req_idx, :req_pages] = page_ids
            page_table[req_idx, req_pages:] = page_ids[-1]
        cache_seqlens = torch.tensor(cache_lens, dtype=torch.int32, device=device)
        cu_seqlens_q = torch.tensor(
            [0, *torch.tensor(q_lens, dtype=torch.int32).cumsum(0).tolist()],
            dtype=torch.int32,
            device=device,
        )
        return q, page_table, cache_seqlens, cu_seqlens_q

    for k_cache, v_cache in (
        make_kv_cache(num_cache_pages),
        make_kv_cache(num_cache_pages + 32),
    ):
        for q_lens, cache_lens in (
            ([3, 2, 1, 4], [129, 130, 131, 132]),
            ([16, 8, 4, 1], [400, 401, 402, 403]),
        ):
            q, page_table, cache_seqlens, cu_seqlens_q = make_metadata(
                q_lens,
                cache_lens,
            )
            output = torch.empty_like(q)
            binding = scratch_plan.bind(
                scratch=scratch,
                q=q,
                k_cache=k_cache,
                v_cache=v_cache,
                output=output,
                page_table=page_table,
                cache_seqlens=cache_seqlens,
                cu_seqlens_q=cu_seqlens_q,
                active_total_q=q.shape[0],
            )
            paged_attention_forward(binding=binding)

    assert len(forward_specs) == 4
    assert len(set(forward_specs)) == 1


@torch.inference_mode()
def test_paged_forward_matches_reference_extend_with_sliding_window_and_sink() -> None:
    require_b12x()
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_inputs(
        q_seqlens=[6, 5],
        cache_seqlens=[320, 384],
        dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
    )
    window_left = 96
    attention_sink_bias = torch.linspace(0.1, -0.1, q.shape[1], dtype=torch.float32, device=q.device)
    workspace = _make_workspace(q, k_cache, v_cache, mode="extend")
    workspace.prepare(
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        disable_split_kv=True,
        window_left=window_left,
    )
    output, lse_base2 = workspace.run(
        q,
        k_cache,
        v_cache,
        output=torch.empty_like(q),
        attention_sink_bias=attention_sink_bias,
    )
    torch.cuda.synchronize()

    ref_out, ref_lse = paged_attention_reference(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        causal=True,
        window_left=window_left,
        attention_sink_bias=attention_sink_bias,
    )
    lse_natural = lse_base2 * math.log(2.0)
    assert (output - ref_out).abs().max().item() <= 0.03
    assert (lse_natural - ref_lse).abs().max().item() <= 0.05
    assert _cosine_similarity(output, ref_out) >= 0.99999


@torch.inference_mode()
def test_paged_forward_matches_reference_extend_with_relative_bias() -> None:
    require_b12x()
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_inputs(
        q_seqlens=[6, 5],
        cache_seqlens=[320, 384],
        page_size=128,
        q_heads=8,
        kv_heads=2,
        head_dim_qk=128,
        head_dim_vo=128,
        dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
    )
    window_left = 511
    relative_attention_bias = torch.randn(
        q.shape[0],
        q.shape[1],
        512,
        dtype=q.dtype,
        device=q.device,
    )
    workspace = _make_workspace(q, k_cache, v_cache, mode="extend")
    workspace.prepare(
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        disable_split_kv=True,
        window_left=window_left,
    )
    output, lse_base2 = workspace.run(
        q,
        k_cache,
        v_cache,
        output=torch.empty_like(q),
        relative_attention_bias=relative_attention_bias,
    )
    torch.cuda.synchronize()

    ref_out, ref_lse = paged_attention_reference(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        causal=True,
        window_left=window_left,
        relative_attention_bias=relative_attention_bias,
    )
    lse_natural = lse_base2 * math.log(2.0)
    assert (output - ref_out).abs().max().item() <= 0.03
    assert (lse_natural - ref_lse).abs().max().item() <= 0.05
    assert _cosine_similarity(output, ref_out) >= 0.99999


@torch.inference_mode()
def test_paged_forward_matches_reference_with_fp8_kv_extend() -> None:
    require_b12x()
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_inputs(
        q_seqlens=[6, 5],
        cache_seqlens=[2048, 4096],
        dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
    )
    k_fp8, v_fp8, k_descale, v_descale = quantize_paged_kv_cache_e4m3(
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
    )
    workspace = _make_workspace(q, k_fp8, v_fp8, mode="extend")
    workspace.prepare(page_table, cache_seqlens, cu_seqlens_q)
    assert workspace.plan.split_kv is False
    output, lse_base2 = workspace.run(
        q,
        k_fp8,
        v_fp8,
        output=torch.empty_like(q),
        k_descale=k_descale,
        v_descale=v_descale,
    )
    torch.cuda.synchronize()

    ref_out, ref_lse = paged_attention_reference(
        q,
        k_fp8,
        v_fp8,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        k_descale=k_descale,
        v_descale=v_descale,
        causal=True,
    )
    lse_natural = lse_base2 * math.log(2.0)
    assert (output - ref_out).abs().max().item() <= 0.05
    assert (lse_natural - ref_lse).abs().max().item() <= 0.08
    assert _cosine_similarity(output, ref_out) >= 0.999


@torch.inference_mode()
def test_paged_forward_matches_reference_with_split_fp8_decode() -> None:
    require_b12x()
    output, reference, plan_desc = _run_decode_graph_check(cache_seqlen=512)
    assert plan_desc.endswith(",split")
    assert (output - reference).abs().max().item() <= 0.01
    assert _cosine_similarity(output, reference) >= 0.999


@torch.inference_mode()
def test_paged_forward_native_fp8_qkv_matches_reference_with_split_fp8_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    require_b12x()
    monkeypatch.setenv("B12X_TURBO_ATTN", "1")
    output, ref_out, plan_desc = _run_decode_reference_check(cache_seqlen=8192)
    assert plan_desc.endswith(",split")
    assert (output - ref_out).abs().max().item() <= 0.01
    assert _cosine_similarity(output, ref_out) >= 0.995


@torch.inference_mode()
def test_paged_forward_matches_reference_with_bf16_kv_extend() -> None:
    require_b12x()
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_inputs(
        q_seqlens=[6, 5],
        cache_seqlens=[2048, 4096],
        dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
    )
    workspace = _make_workspace(q, k_cache, v_cache, mode="extend")
    workspace.prepare(page_table, cache_seqlens, cu_seqlens_q)
    assert workspace.plan.split_kv is False
    output, lse_base2 = workspace.run(q, k_cache, v_cache, output=torch.empty_like(q))
    torch.cuda.synchronize()

    ref_out, ref_lse = paged_attention_reference(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        causal=True,
    )
    lse_natural = lse_base2 * math.log(2.0)
    assert (output - ref_out).abs().max().item() <= 0.03
    assert (lse_natural - ref_lse).abs().max().item() <= 0.05
    assert _cosine_similarity(output, ref_out) >= 0.99999


@torch.inference_mode()
@pytest.mark.parametrize(
    "kv_dtype",
    [torch.bfloat16, torch.float8_e4m3fn],
)
def test_qwen38_full_prefill_terminal_v_copy_matches_reference(
    kv_dtype: torch.dtype,
) -> None:
    """Full prefill must drain a terminal partial V tile before its PV MMA."""
    require_b12x()
    q_len = 256
    batch = 8
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_inputs(
        q_seqlens=[q_len] * batch,
        cache_seqlens=[q_len] * batch,
        q_heads=24,
        kv_heads=4,
        head_dim_qk=256,
        head_dim_vo=256,
        dtype=torch.bfloat16,
        kv_dtype=kv_dtype,
    )
    workspace = _make_workspace(q, k_cache, v_cache, mode="extend")
    workspace.prepare(page_table, cache_seqlens, cu_seqlens_q)
    assert workspace.plan.split_kv is False
    descale = (
        torch.ones(batch, dtype=torch.float32, device=q.device)
        if kv_dtype == torch.float8_e4m3fn
        else None
    )

    ref_out, _ref_lse = paged_attention_reference(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        k_descale=descale,
        v_descale=descale,
        causal=True,
    )
    output = torch.empty_like(q)
    for _ in range(3):
        output.fill_(float("nan"))
        workspace.run(
            q,
            k_cache,
            v_cache,
            output=output,
            k_descale=descale,
            v_descale=descale,
        )
        torch.cuda.synchronize()
        assert torch.isfinite(output).all()
        assert (output - ref_out).abs().max().item() <= 0.05
        assert _cosine_similarity(output, ref_out) >= 0.999
