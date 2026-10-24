from __future__ import annotations

from dataclasses import replace
import math
from types import SimpleNamespace

import pytest
import torch

from b12x._lib.compile_plan import forbid_lowering
from b12x._lib.runtime_control import kernel_resolution_guard
from b12x.attention import paged
import b12x.attention.paged._forward as paged_api
import b12x.attention.paged._scratch as scratch_api
from b12x.attention.paged.reference import paged_attention_reference
from b12x.attention.paged._scratch import (
    B12XPagedAttentionBinding,
    B12XPagedAttentionScratchCaps,
    plan_paged_attention_scratch,
)

from b12x.testing.reference.helpers import require_b12x


def _caps() -> B12XPagedAttentionScratchCaps:
    return B12XPagedAttentionScratchCaps(
        device="cpu",
        mode="decode",
        dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
        num_q_heads=2,
        num_kv_heads=1,
        head_dim_qk=16,
        head_dim_vo=16,
        page_size=4,
        max_total_q=2,
        max_batch=2,
        max_page_table_width=3,
        max_work_items=4,
        max_partial_rows=4,
        num_cache_pages=8,
    )


def _runtime_tensors():
    q = torch.empty((2, 2, 16), dtype=torch.bfloat16)
    k_cache = torch.empty((8, 4, 1, 16), dtype=torch.bfloat16)
    v_cache = torch.empty((8, 4, 1, 16), dtype=torch.bfloat16)
    output = torch.empty((2, 2, 16), dtype=torch.bfloat16)
    page_table = torch.zeros((2, 3), dtype=torch.int32)
    cache_seqlens = torch.ones((2,), dtype=torch.int32)
    cu_seqlens_q = torch.arange(3, dtype=torch.int32)
    return q, k_cache, v_cache, output, page_table, cache_seqlens, cu_seqlens_q


def _prepared_cpu_decode_graph_scratch() -> scratch_api.B12XPagedAttentionScratch:
    return scratch_api.B12XPagedAttentionScratch(
        shared_scratch=torch.empty(1, dtype=torch.uint8),
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
        mode="decode",
        num_q_heads=2,
        num_kv_heads=1,
        head_dim_qk=16,
        head_dim_vo=16,
        page_size=4,
        max_total_q=2,
        max_batch=2,
        max_page_table_width=3,
        max_work_items=4,
        max_partial_rows=4,
        num_cache_pages=8,
        use_cuda_graph=True,
        _plan=SimpleNamespace(
            page_table_shape=(2, 3),
            total_q=2,
            window_left=-1,
        ),
        _plan_metadata_cache=object(),
        _decode_graph_chunk_pages_lut=torch.ones(4, dtype=torch.int32),
    )


def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f = a.to(torch.float32).reshape(-1)
    b_f = b.to(torch.float32).reshape(-1)
    return torch.nn.functional.cosine_similarity(a_f, b_f, dim=0).item()


def _lse_base2_to_natural(lse: torch.Tensor) -> torch.Tensor:
    return lse * math.log(2.0)


def _make_mimo_packed_inputs(
    *,
    q_seqlens: list[int],
    cache_seqlens: list[int],
    page_size: int = 64,
    seed: int = 0,
    page_table_width: int = 8,
    num_pages: int = 64,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    if len(q_seqlens) != len(cache_seqlens):
        raise ValueError("q_seqlens and cache_seqlens must have the same length")
    torch.manual_seed(seed)
    device = "cuda"
    q_heads = 16
    kv_heads = 1
    head_dim_qk = 192
    head_dim_vo = 128
    total_q = sum(q_seqlens)
    q = (
        torch.randn(
            total_q,
            q_heads,
            head_dim_qk,
            device=device,
            dtype=torch.bfloat16,
        )
        / 4
    )

    pages_per_request = [
        (cache_len + page_size - 1) // page_size for cache_len in cache_seqlens
    ]
    if page_table_width < max(pages_per_request, default=0):
        raise ValueError("page_table_width is too small for cache_seqlens")
    if num_pages < sum(pages_per_request):
        raise ValueError("num_pages is too small for cache_seqlens")

    k_compact = (
        torch.randn(
            num_pages,
            page_size,
            kv_heads,
            head_dim_qk,
            device=device,
            dtype=torch.bfloat16,
        )
        / 4
    )
    v_compact = (
        torch.randn(
            num_pages,
            page_size,
            kv_heads,
            head_dim_vo,
            device=device,
            dtype=torch.bfloat16,
        )
        / 4
    )
    packed_cache = torch.empty(
        (*k_compact.shape[:-1], head_dim_qk + head_dim_vo),
        dtype=torch.bfloat16,
        device=device,
    )
    k_cache = packed_cache[..., :head_dim_qk]
    v_cache = packed_cache[..., head_dim_qk:]
    k_cache.copy_(k_compact)
    v_cache.copy_(v_compact)

    page_table = torch.zeros(
        len(q_seqlens),
        page_table_width,
        dtype=torch.int32,
        device=device,
    )
    page_order = torch.randperm(num_pages, device=device)
    cursor = 0
    for request_idx, num_req_pages in enumerate(pages_per_request):
        if num_req_pages == 0:
            continue
        page_ids = page_order[cursor : cursor + num_req_pages].to(torch.int32)
        cursor += num_req_pages
        page_table[request_idx, :num_req_pages] = page_ids
        page_table[request_idx, num_req_pages:] = page_ids[-1]

    cache_seqlens_t = torch.tensor(cache_seqlens, dtype=torch.int32, device=device)
    q_offsets = [0]
    for q_len in q_seqlens:
        q_offsets.append(q_offsets[-1] + q_len)
    cu_seqlens_q = torch.tensor(q_offsets, dtype=torch.int32, device=device)
    return q, k_cache, v_cache, page_table, cache_seqlens_t, cu_seqlens_q


def _make_strided_q_view(q: torch.Tensor) -> torch.Tensor:
    storage = torch.empty(
        (*q.shape[:-1], q.shape[-1] + 8),
        dtype=q.dtype,
        device=q.device,
    )
    q_strided = storage[..., : q.shape[-1]]
    q_strided.copy_(q)
    assert q_strided.stride() != q.stride()
    return q_strided


def test_paged_attention_scratch_plan_exposes_one_opaque_scratch_spec() -> None:
    plan = plan_paged_attention_scratch(_caps())

    specs = plan.scratch_specs()
    assert len(specs) == 1
    assert specs[0].name == "paged_attention.scratch"
    assert specs[0].dtype == torch.uint8
    assert specs[0].shape == plan.shapes_and_dtypes()[0][0]
    assert specs[0].nbytes == specs[0].shape[0]
    assert plan.caps.max_total_q == 2


def test_decode_graph_scratch_plan_requires_exact_batch_bucket() -> None:
    plan = plan_paged_attention_scratch(replace(_caps(), use_cuda_graph=True))

    with pytest.raises(ValueError, match="must exactly match its scratch bucket"):
        plan.prepare_decode_graph_replay_state(
            batch=1,
            total_q_capacity=1,
            max_page_table_width=3,
            max_cache_page_count=3,
        )


def test_decode_graph_scratch_plan_requires_one_query_per_request() -> None:
    plan = plan_paged_attention_scratch(replace(_caps(), use_cuda_graph=True))

    with pytest.raises(ValueError, match="must cover one query per request"):
        plan.prepare_decode_graph_replay_state(
            batch=2,
            total_q_capacity=1,
            max_page_table_width=3,
            max_cache_page_count=3,
        )


def test_decode_graph_scratch_plan_rejects_preparation_during_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = plan_paged_attention_scratch(replace(_caps(), use_cuda_graph=True))
    object.__setattr__(plan.caps, "device", torch.device("cuda"))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)

    with pytest.raises(RuntimeError, match="before CUDA graph capture"):
        plan.prepare_decode_graph_replay_state(
            batch=2,
            total_q_capacity=2,
            max_page_table_width=3,
            max_cache_page_count=3,
        )


@pytest.mark.parametrize("invalid_kind", ["dtype", "contiguity"])
def test_prepared_decode_graph_scratch_requires_fixed_metadata_contract(
    invalid_kind: str,
) -> None:
    scratch = _prepared_cpu_decode_graph_scratch()
    page_table = torch.empty((2, 3), dtype=torch.int32)
    if invalid_kind == "dtype":
        page_table = page_table.to(torch.int64)
        expected = "torch.int32"
    else:
        page_table = torch.empty((2, 6), dtype=torch.int32)[:, ::2]
        assert not page_table.is_contiguous()
        expected = "contiguous"

    with pytest.raises((TypeError, ValueError), match=expected):
        scratch.prepare(
            page_table,
            torch.empty((2,), dtype=torch.int32),
            torch.empty((3,), dtype=torch.int32),
            active_total_q=2,
        )


def test_prepared_decode_graph_scratch_accepts_narrower_copied_page_table() -> None:
    scratch = _prepared_cpu_decode_graph_scratch()

    scratch._validate_decode_graph_runtime_contract(
        page_table=torch.empty((2, 2), dtype=torch.int32),
        cache_seqlens=torch.empty((2,), dtype=torch.int32),
        cu_seqlens_q=torch.empty((3,), dtype=torch.int32),
        active_total_q=2,
    )


def test_prepared_decode_graph_scratch_rejects_narrower_referenced_page_table() -> None:
    scratch = _prepared_cpu_decode_graph_scratch()
    scratch.copy_runtime_metadata = False

    with pytest.raises(ValueError, match="referenced page_table must exactly match"):
        scratch._validate_decode_graph_runtime_contract(
            page_table=torch.empty((2, 2), dtype=torch.int32),
            cache_seqlens=torch.empty((2,), dtype=torch.int32),
            cu_seqlens_q=torch.empty((3,), dtype=torch.int32),
            active_total_q=2,
        )


@pytest.mark.parametrize(
    ("page_table_shape", "cache_shape", "cu_shape", "active_total_q", "match"),
    [
        ((1, 3), (2,), (3,), 2, "page_table must exactly match"),
        ((2, 3), (1,), (3,), 2, "cache_seqlens must exactly match"),
        ((2, 3), (2,), (2,), 2, "cu_seqlens_q must exactly match"),
        ((2, 3), (2,), (3,), 1, "active_total_q must exactly match"),
    ],
)
def test_prepared_decode_graph_scratch_rejects_partial_runtime_bucket(
    page_table_shape: tuple[int, int],
    cache_shape: tuple[int],
    cu_shape: tuple[int],
    active_total_q: int,
    match: str,
) -> None:
    scratch = _prepared_cpu_decode_graph_scratch()

    with pytest.raises(ValueError, match=match):
        scratch.prepare(
            torch.empty(page_table_shape, dtype=torch.int32),
            torch.empty(cache_shape, dtype=torch.int32),
            torch.empty(cu_shape, dtype=torch.int32),
            active_total_q=active_total_q,
        )


@pytest.mark.parametrize("short_tensor", ["q", "output", "active_total_q"])
def test_prepared_decode_graph_binding_rejects_partial_total_q(
    short_tensor: str,
) -> None:
    scratch = _prepared_cpu_decode_graph_scratch()
    q, k_cache, v_cache, output, *_ = _runtime_tensors()
    if short_tensor == "q":
        q = q[:1]
    elif short_tensor == "output":
        output = output[:1]

    with pytest.raises(ValueError, match="total_q must exactly match"):
        scratch.bind(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            output=output,
            active_total_q=1 if short_tensor == "active_total_q" else None,
        )


def test_decode_graph_capture_prepare_and_forward_insert_one_updater(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scratch = _prepared_cpu_decode_graph_scratch()
    calls = 0

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    monkeypatch.setattr(
        scratch_api.B12XPagedAttentionScratch,
        "_bind_runtime_metadata",
        lambda self, page_table, cache_seqlens, cu_seqlens_q: None,
    )
    monkeypatch.setattr(
        scratch_api.B12XPagedAttentionScratch,
        "_copy_cached_plan_metadata",
        lambda self, cache: None,
    )

    def update(self: scratch_api.B12XPagedAttentionScratch) -> None:
        nonlocal calls
        calls += 1

    monkeypatch.setattr(
        scratch_api.B12XPagedAttentionScratch,
        "update_decode_graph_replay_metadata_from_runtime_cache_seqlens",
        update,
    )

    scratch.prepare(
        torch.empty((2, 3), dtype=torch.int32),
        torch.empty((2,), dtype=torch.int32),
        torch.empty((3,), dtype=torch.int32),
        active_total_q=2,
    )
    assert calls == 0

    scratch.device = torch.device("cuda")
    paged_api._capture_decode_graph_replay_metadata_if_needed(scratch)
    assert calls == 1


def test_paged_attention_scratch_bind_returns_common_binding_type(
    monkeypatch,
) -> None:
    plan = plan_paged_attention_scratch(_caps())
    (spec,) = plan.scratch_specs()
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
    q, k_cache, v_cache, output, page_table, cache_seqlens, cu_seqlens_q = (
        _runtime_tensors()
    )
    calls = {}

    def fake_prepare(
        self, page_table_arg, cache_seqlens_arg, cu_seqlens_q_arg, **kwargs
    ):
        calls["page_table"] = page_table_arg
        calls["cache_seqlens"] = cache_seqlens_arg
        calls["cu_seqlens_q"] = cu_seqlens_q_arg
        calls["kwargs"] = kwargs
        self.page_table = page_table_arg
        self.cache_seqlens = cache_seqlens_arg
        self.cu_seqlens_q = cu_seqlens_q_arg
        self._plan = object()
        return self

    monkeypatch.setattr(scratch_api.B12XPagedAttentionScratch, "prepare", fake_prepare)

    binding = plan.bind(
        scratch=scratch,
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        output=output,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        window_left=7,
        active_total_q=2,
    )

    assert isinstance(binding, B12XPagedAttentionBinding)
    assert isinstance(binding.scratch, scratch_api.B12XPagedAttentionScratch)
    assert binding.q is q
    assert binding.output is output
    assert calls["page_table"] is page_table
    assert calls["cache_seqlens"] is cache_seqlens
    assert calls["cu_seqlens_q"] is cu_seqlens_q
    assert calls["kwargs"]["window_left"] == 7
    assert calls["kwargs"]["active_total_q"] == 2


def test_paged_attention_binding_run_uses_function_binding_argument(
    monkeypatch,
) -> None:
    scratch = object()
    q, k_cache, v_cache, output, *_ = _runtime_tensors()
    binding = B12XPagedAttentionBinding(
        scratch=scratch,
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        output=output,
    )
    calls = {}

    def fake_forward(**kwargs):
        calls.update(kwargs)
        return "out", "lse"

    monkeypatch.setattr(paged_api, "paged_attention_forward", fake_forward)

    assert binding.run() == ("out", "lse")
    assert calls["binding"] is binding


@torch.inference_mode()
def test_heuristic_extend_binding_compiles_before_frozen_runs_high_page_ids() -> None:
    device = require_b12x()
    page_size, head_dim, q_heads, kv_heads, batch, q_capacity = 128, 128, 8, 1, 2, 128
    max_cache_seqlen = page_size + 1
    page_table_width = math.ceil(max_cache_seqlen / page_size)
    live_page_count = batch * page_table_width
    page_shape = (2, page_size, kv_heads, head_dim)
    high_page_id = torch.iinfo(torch.int32).max // math.prod(page_shape) + 2
    num_cache_pages = high_page_id + live_page_count
    combined_kv_cache = torch.empty(
        (num_cache_pages, *page_shape), dtype=torch.bfloat16, device=device
    )
    k_cache, v_cache = combined_kv_cache[:, 0], combined_kv_cache[:, 1]
    # Low pages differ from the live ones so a truncated page id cannot match.
    combined_kv_cache[:live_page_count].fill_(3)
    combined_kv_cache[high_page_id:].normal_(std=0.25)
    page_table = torch.arange(
        high_page_id, num_cache_pages, dtype=torch.int32, device=device
    ).view(batch, page_table_width)
    q = (
        torch.randn(
            (q_capacity, q_heads, head_dim), dtype=torch.bfloat16, device=device
        )
        / 4
    )
    output = torch.full_like(q, torch.nan)

    capacity = paged.extend_graph_capacity(
        device=device,
        q_dtype=q.dtype,
        kv_dtype=k_cache.dtype,
        num_q_heads=q_heads,
        num_kv_heads=kv_heads,
        head_dim_qk=head_dim,
        head_dim_vo=head_dim,
        page_size=page_size,
        batch=batch,
        total_q_capacity=q_capacity,
        max_cache_page_count=page_table_width,
        window_left=-1,
    )
    plan = paged.plan(
        paged.Caps(
            device=device,
            mode="extend",
            dtype=q.dtype,
            kv_dtype=k_cache.dtype,
            num_q_heads=q_heads,
            num_kv_heads=kv_heads,
            head_dim_qk=head_dim,
            head_dim_vo=head_dim,
            page_size=page_size,
            max_total_q=q_capacity,
            max_batch=batch,
            max_page_table_width=page_table_width,
            max_work_items=capacity.max_work_items,
            max_partial_rows=0,
            num_cache_pages=num_cache_pages,
            use_cuda_graph=True,
            copy_runtime_metadata=False,
        )
    )
    plan.prepare_graph_replay_state(
        page_table=page_table,
        cache_seqlens=torch.full(
            (batch,), max_cache_seqlen, dtype=torch.int32, device=device
        ),
        cu_seqlens_q=torch.tensor(
            [0, q_capacity // 2, q_capacity], dtype=torch.int32, device=device
        ),
        active_total_q=q_capacity,
        window_left=-1,
    )
    (spec,) = plan.scratch_specs()
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=device)

    def live(q_seqlens: list[int], cache_seqlens: list[int]):
        rows = sum(q_seqlens)
        lens = torch.tensor(cache_seqlens, dtype=torch.int32, device=device)
        cu_seqlens_q = torch.tensor(
            [0, q_seqlens[0], rows], dtype=torch.int32, device=device
        )
        binding = plan.bind(
            scratch=scratch,
            q=q[:rows],
            k_cache=k_cache,
            v_cache=v_cache,
            output=output[:rows],
            page_table=page_table,
            cache_seqlens=lens,
            cu_seqlens_q=cu_seqlens_q,
            window_left=-1,
        )
        reference, _ = paged_attention_reference(
            q[:rows],
            k_cache,
            v_cache,
            page_table,
            lens,
            cu_seqlens_q,
            causal=True,
        )
        return binding, reference

    cases = (
        ([40, 24], [max_cache_seqlen, page_size]),
        ([9, 8], [page_size, max_cache_seqlen]),
    )
    binding, _ = live(*cases[0])
    paged.compile(binding=binding)
    assert torch.isnan(output).all().item()

    # forbid_lowering installs b12x's Triton hook, so the guard also rejects Triton JIT misses.
    with (
        kernel_resolution_guard("paged.compile primed this binding"),
        forbid_lowering(),
    ):
        for q_seqlens, cache_seqlens in cases:
            binding, reference = live(q_seqlens, cache_seqlens)
            output.fill_(torch.nan)
            out, _ = paged.run(binding=binding)
            torch.testing.assert_close(
                out.float(), reference.float(), atol=2e-2, rtol=2e-2
            )


def test_paged_attention_forward_rejects_binding_plus_runtime_tensors() -> None:
    scratch = object()
    q, k_cache, v_cache, output, *_ = _runtime_tensors()
    binding = B12XPagedAttentionBinding(
        scratch=scratch,
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        output=output,
    )

    with pytest.raises(ValueError, match="binding owns runtime tensors"):
        paged_api.paged_attention_forward(binding=binding, q=q)


@pytest.mark.parametrize(
    ("mode", "q_seqlens", "cache_seqlens", "window_left", "use_attention_sink"),
    [
        ("extend", [5], [5], -1, False),
        ("extend", [5], [5], 127, True),
        ("decode", [1], [5], -1, False),
        ("decode", [1], [5], 127, True),
    ],
)
def test_mimo_v25_packed_diffkv_scratch_matches_reference(
    mode: str,
    q_seqlens: list[int],
    cache_seqlens: list[int],
    window_left: int,
    use_attention_sink: bool,
) -> None:
    require_b12x()
    paged_api.clear_paged_caches()

    q, k_cache, v_cache, page_table, cache_seqlens_t, cu_seqlens_q = (
        _make_mimo_packed_inputs(
            q_seqlens=q_seqlens,
            cache_seqlens=cache_seqlens,
            seed=900 + len(q_seqlens) + max(window_left, 0) + int(use_attention_sink),
        )
    )
    assert tuple(k_cache.stride()) == (20480, 320, 320, 1)
    assert tuple(v_cache.stride()) == (20480, 320, 320, 1)

    attention_sink_bias = None
    if use_attention_sink:
        attention_sink_bias = torch.linspace(
            -1.0,
            1.0,
            q.shape[1],
            device=q.device,
            dtype=torch.float32,
        )

    max_total_q = max(int(q.shape[0]), len(q_seqlens))
    max_batch = len(q_seqlens)
    max_partial_rows = max_batch * 64 if mode == "decode" else 0
    plan = plan_paged_attention_scratch(
        B12XPagedAttentionScratchCaps(
            device=q.device,
            mode=mode,
            dtype=q.dtype,
            kv_dtype=k_cache.dtype,
            num_q_heads=16,
            num_kv_heads=1,
            head_dim_qk=192,
            head_dim_vo=128,
            page_size=64,
            max_total_q=max_total_q,
            max_batch=max_batch,
            max_page_table_width=int(page_table.shape[1]),
            max_work_items=128,
            max_partial_rows=max_partial_rows,
            num_cache_pages=int(k_cache.shape[0]),
        )
    )
    (spec,) = plan.scratch_specs()
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
    output = torch.empty(
        (*q.shape[:-1], v_cache.shape[-1]),
        dtype=q.dtype,
        device=q.device,
    )

    binding = plan.bind(
        scratch=scratch,
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        output=output,
        page_table=page_table,
        cache_seqlens=cache_seqlens_t,
        cu_seqlens_q=cu_seqlens_q,
        fixed_split_size=1 if mode == "decode" else None,
        window_left=window_left,
        active_total_q=int(q.shape[0]),
        attention_sink_bias=attention_sink_bias,
    )
    out, lse = binding.run()
    ref_out, ref_lse = paged_attention_reference(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens_t,
        cu_seqlens_q,
        causal=True,
        window_left=window_left,
        attention_sink_bias=attention_sink_bias,
    )

    assert torch.allclose(
        out.to(torch.float32),
        ref_out.to(torch.float32),
        atol=2e-2,
        rtol=2e-2,
    )
    assert _cosine_similarity(out, ref_out) >= 0.99999
    assert torch.allclose(
        _lse_base2_to_natural(lse),
        ref_lse,
        atol=3e-2,
        rtol=3e-2,
    )


@pytest.mark.parametrize("mode", ["extend", "decode"])
def test_mimo_v25_packed_diffkv_scratch_matches_reference_with_strided_q(
    mode: str,
) -> None:
    require_b12x()
    paged_api.clear_paged_caches()

    q_seqlens = [5] if mode == "extend" else [1]
    cache_seqlens = [5]
    q, k_cache, v_cache, page_table, cache_seqlens_t, cu_seqlens_q = (
        _make_mimo_packed_inputs(
            q_seqlens=q_seqlens,
            cache_seqlens=cache_seqlens,
            seed=1231,
        )
    )
    q = _make_strided_q_view(q)

    max_total_q = 512 if mode == "extend" else 4
    max_batch = 4
    max_partial_rows = max_batch * 64 if mode == "decode" else 0
    max_work_items = 516 if mode == "extend" else 256
    plan = plan_paged_attention_scratch(
        B12XPagedAttentionScratchCaps(
            device=q.device,
            mode=mode,
            dtype=q.dtype,
            kv_dtype=k_cache.dtype,
            num_q_heads=16,
            num_kv_heads=1,
            head_dim_qk=192,
            head_dim_vo=128,
            page_size=64,
            max_total_q=max_total_q,
            max_batch=max_batch,
            max_page_table_width=int(page_table.shape[1]),
            max_work_items=max_work_items,
            max_partial_rows=max_partial_rows,
            num_cache_pages=1,
        )
    )
    (spec,) = plan.scratch_specs()
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
    output = torch.empty(
        (*q.shape[:-1], v_cache.shape[-1]),
        dtype=q.dtype,
        device=q.device,
    )

    binding = plan.bind(
        scratch=scratch,
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        output=output,
        page_table=page_table,
        cache_seqlens=cache_seqlens_t,
        cu_seqlens_q=cu_seqlens_q,
        fixed_split_size=1 if mode == "decode" else None,
        active_total_q=int(q.shape[0]),
    )
    out, _ = binding.run()
    ref_out, _ = paged_attention_reference(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens_t,
        cu_seqlens_q,
        causal=True,
    )

    assert torch.allclose(
        out.to(torch.float32),
        ref_out.to(torch.float32),
        atol=2e-2,
        rtol=2e-2,
    )
    assert _cosine_similarity(out, ref_out) >= 0.99999


def test_mimo_v25_extend_replans_from_warmup_to_single_short_prompt() -> None:
    require_b12x()
    paged_api.clear_paged_caches()

    plan = plan_paged_attention_scratch(
        B12XPagedAttentionScratchCaps(
            device="cuda",
            mode="extend",
            dtype=torch.bfloat16,
            kv_dtype=torch.bfloat16,
            num_q_heads=16,
            num_kv_heads=1,
            head_dim_qk=192,
            head_dim_vo=128,
            page_size=64,
            max_total_q=512,
            max_batch=4,
            max_page_table_width=8,
            max_work_items=516,
            max_partial_rows=0,
            num_cache_pages=1,
        )
    )
    (spec,) = plan.scratch_specs()
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)

    for q_seqlens, cache_seqlens, seed in (
        ([2, 2, 2, 2], [2, 2, 2, 2], 1401),
        ([5], [5], 1402),
    ):
        q, k_cache, v_cache, page_table, cache_seqlens_t, cu_seqlens_q = (
            _make_mimo_packed_inputs(
                q_seqlens=q_seqlens,
                cache_seqlens=cache_seqlens,
                seed=seed,
                num_pages=8,
            )
        )
        output = torch.empty(
            (*q.shape[:-1], v_cache.shape[-1]),
            dtype=q.dtype,
            device=q.device,
        )

        binding = plan.bind(
            scratch=scratch,
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            output=output,
            page_table=page_table,
            cache_seqlens=cache_seqlens_t,
            cu_seqlens_q=cu_seqlens_q,
            active_total_q=int(q.shape[0]),
        )
        out, _ = binding.run()
        ref_out, _ = paged_attention_reference(
            q,
            k_cache,
            v_cache,
            page_table,
            cache_seqlens_t,
            cu_seqlens_q,
            causal=True,
        )

        assert torch.allclose(
            out.to(torch.float32),
            ref_out.to(torch.float32),
            atol=2e-2,
            rtol=2e-2,
        )
        assert _cosine_similarity(out, ref_out) >= 0.99999


@torch.inference_mode()
def test_mimo_v25_packed_diffkv_scratch_cuda_graph_replays_with_updated_metadata() -> (
    None
):
    require_b12x()
    paged_api.clear_paged_caches()

    batch = 4
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = (
        _make_mimo_packed_inputs(
            q_seqlens=[1] * batch,
            cache_seqlens=[129, 193, 257, 385],
            page_size=64,
            seed=1729,
            page_table_width=8,
            num_pages=128,
        )
    )
    attention_sink_bias = torch.linspace(
        -0.25,
        0.35,
        steps=q.shape[1],
        device=q.device,
        dtype=torch.float32,
    )
    plan = plan_paged_attention_scratch(
        B12XPagedAttentionScratchCaps(
            device=q.device,
            mode="decode",
            dtype=q.dtype,
            kv_dtype=k_cache.dtype,
            num_q_heads=16,
            num_kv_heads=1,
            head_dim_qk=192,
            head_dim_vo=128,
            page_size=64,
            max_total_q=batch,
            max_batch=batch,
            max_page_table_width=int(page_table.shape[1]),
            max_work_items=512,
            max_partial_rows=512,
            num_cache_pages=int(k_cache.shape[0]),
            use_cuda_graph=True,
            copy_runtime_metadata=True,
        )
    )
    plan.prepare_decode_graph_replay_state(
        batch=batch,
        total_q_capacity=batch,
        max_page_table_width=int(page_table.shape[1]),
        max_cache_page_count=int(page_table.shape[1]),
        window_left=128,
    )
    (spec,) = plan.scratch_specs()
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
    output = torch.empty(
        (*q.shape[:-1], v_cache.shape[-1]),
        dtype=q.dtype,
        device=q.device,
    )

    plan.bind(
        scratch=scratch,
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        output=output,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        window_left=128,
        active_total_q=batch,
        attention_sink_bias=attention_sink_bias,
    ).run()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        binding = plan.bind(
            scratch=scratch,
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            output=output,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            window_left=128,
            active_total_q=batch,
            attention_sink_bias=attention_sink_bias,
        )
        binding.run()

    assert binding.scratch._decode_graph_metadata_captured_in_graph is True

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
        window_left=128,
        attention_sink_bias=attention_sink_bias,
    )
    assert torch.allclose(
        output.to(torch.float32),
        ref_out.to(torch.float32),
        atol=2e-2,
        rtol=2e-2,
    )
    assert _cosine_similarity(output, ref_out) >= 0.99999

    q_2, k_cache_2, v_cache_2, page_table_2, cache_seqlens_2, cu_seqlens_q_2 = (
        _make_mimo_packed_inputs(
            q_seqlens=[1] * batch,
            cache_seqlens=[193, 257, 385, 449],
            page_size=64,
            seed=1733,
            page_table_width=int(page_table.shape[1]),
            num_pages=int(k_cache.shape[0]),
        )
    )
    q.copy_(q_2)
    k_cache.copy_(k_cache_2)
    v_cache.copy_(v_cache_2)
    page_table.copy_(page_table_2)
    cache_seqlens.copy_(cache_seqlens_2)
    cu_seqlens_q.copy_(cu_seqlens_q_2)

    graph.replay()
    torch.cuda.synchronize()
    ref_out_2, _ = paged_attention_reference(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        causal=True,
        window_left=128,
        attention_sink_bias=attention_sink_bias,
    )
    assert torch.allclose(
        output.to(torch.float32),
        ref_out_2.to(torch.float32),
        atol=2e-2,
        rtol=2e-2,
    )
    assert _cosine_similarity(output, ref_out_2) >= 0.99999


def _run_mimo_v25_fp8_decode_graph_case(
    window_left: int,
    seed: int,
    *,
    batch: int = 4,
    cache_seqlens_list: list[int] | None = None,
    page_table_width: int = 8,
    num_pages: int = 128,
    expect_page_table_width: int | None = None,
) -> None:
    if cache_seqlens_list is None:
        cache_seqlens_list = [129, 193, 257, 385]
    if len(cache_seqlens_list) != batch:
        raise ValueError("cache_seqlens_list length must match batch")
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = (
        _make_mimo_packed_inputs(
            q_seqlens=[1] * batch,
            cache_seqlens=cache_seqlens_list,
            page_size=64,
            seed=seed,
            page_table_width=page_table_width,
            num_pages=num_pages,
        )
    )
    k_cache = k_cache.to(torch.float8_e4m3fn)
    v_cache = v_cache.to(torch.float8_e4m3fn)
    k_descale = torch.ones((batch, 1), device=q.device, dtype=torch.float32)
    v_descale = torch.ones((batch, 1), device=q.device, dtype=torch.float32)

    plan = plan_paged_attention_scratch(
        B12XPagedAttentionScratchCaps(
            device=q.device,
            mode="decode",
            dtype=q.dtype,
            kv_dtype=k_cache.dtype,
            num_q_heads=16,
            num_kv_heads=1,
            head_dim_qk=192,
            head_dim_vo=128,
            page_size=64,
            max_total_q=batch,
            max_batch=batch,
            max_page_table_width=int(page_table.shape[1]),
            max_work_items=512,
            max_partial_rows=0,
            num_cache_pages=int(k_cache.shape[0]),
            use_cuda_graph=True,
            copy_runtime_metadata=True,
        )
    )
    plan.prepare_decode_graph_replay_state(
        batch=batch,
        total_q_capacity=batch,
        max_page_table_width=int(page_table.shape[1]),
        max_cache_page_count=int(page_table.shape[1]),
        window_left=window_left,
    )
    (spec,) = plan.scratch_specs()
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
    output = torch.empty(
        (*q.shape[:-1], v_cache.shape[-1]),
        dtype=q.dtype,
        device=q.device,
    )

    def bind():
        return plan.bind(
            scratch=scratch,
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            output=output,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            window_left=window_left,
            active_total_q=batch,
            k_descale=k_descale,
            v_descale=v_descale,
        )

    bind().run()
    torch.cuda.synchronize()
    eager = output.clone()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        binding = bind()
        binding.run()
    with pytest.raises(RuntimeError, match="cannot replace decode graph replay state"):
        plan.prepare_decode_graph_replay_state(
            batch=batch,
            total_q_capacity=batch,
            max_page_table_width=int(page_table.shape[1]),
            max_cache_page_count=int(page_table.shape[1]),
            window_left=window_left,
        )
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
        k_descale=k_descale,
        v_descale=v_descale,
    )
    assert binding.scratch._plan.split_kv is False
    if expect_page_table_width is not None:
        assert binding.scratch.page_table is not None
        assert int(binding.scratch.page_table.shape[1]) == expect_page_table_width
    assert torch.allclose(output, eager, atol=0, rtol=0)
    assert torch.allclose(
        output.to(torch.float32),
        ref_out.to(torch.float32),
        atol=5e-2,
        rtol=5e-2,
    )
    assert _cosine_similarity(output, ref_out) >= 0.99999


@torch.inference_mode()
def test_mimo_v25_decode_graph_no_split_compile_key_tracks_batch_bucket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    require_b12x()
    paged_api.clear_paged_caches()

    captured_specs = []

    def capture_launch(
        kernel: object,
        *,
        compile_spec: object,
        compile_args: tuple[object, ...],
        runtime_args: tuple[object, ...],
    ) -> None:
        del kernel, compile_args, runtime_args
        captured_specs.append(compile_spec)

    monkeypatch.setattr(paged_api, "b12x_launch", capture_launch)

    def bind_no_split_decode(batch: int) -> None:
        q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = (
            _make_mimo_packed_inputs(
                q_seqlens=[1] * batch,
                cache_seqlens=[129] * batch,
                page_size=64,
                seed=3500 + batch,
                page_table_width=8,
                num_pages=128,
            )
        )
        plan = plan_paged_attention_scratch(
            B12XPagedAttentionScratchCaps(
                device=q.device,
                mode="decode",
                dtype=q.dtype,
                kv_dtype=k_cache.dtype,
                num_q_heads=16,
                num_kv_heads=1,
                head_dim_qk=192,
                head_dim_vo=128,
                page_size=64,
                max_total_q=batch,
                max_batch=batch,
                max_page_table_width=int(page_table.shape[1]),
                max_work_items=512,
                max_partial_rows=0,
                num_cache_pages=int(k_cache.shape[0]),
                use_cuda_graph=True,
                copy_runtime_metadata=True,
            )
        )
        plan.prepare_decode_graph_replay_state(
            batch=batch,
            total_q_capacity=batch,
            max_page_table_width=int(page_table.shape[1]),
            max_cache_page_count=int(page_table.shape[1]),
            window_left=-1,
            force_split_kv=False,
        )
        (spec,) = plan.scratch_specs()
        scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
        output = torch.empty(
            (*q.shape[:-1], v_cache.shape[-1]),
            dtype=q.dtype,
            device=q.device,
        )
        binding = plan.bind(
            scratch=scratch,
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            output=output,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            active_total_q=batch,
        )
        assert binding.scratch._plan.split_kv is False
        binding.run()

    bind_no_split_decode(8)
    bind_no_split_decode(4)

    assert len(captured_specs) == 2
    assert captured_specs[0] != captured_specs[1]


@torch.inference_mode()
def test_mimo_v25_fp8_decode_graph_non_split_ignores_worklist_bucket_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    require_b12x()
    paged_api.clear_paged_caches()

    def fail_regular_metadata_update(**kwargs: object) -> None:
        raise AssertionError("non-split decode should not update chunk metadata")

    monkeypatch.setattr(
        "b12x.attention.paged.graph_replay.update_regular_decode_graph_chunk_metadata",
        fail_regular_metadata_update,
    )
    cache_seqlens = [129, 193, 257, 385, 449, 513, 577]
    _run_mimo_v25_fp8_decode_graph_case(
        window_left=-1,
        seed=3100,
        batch=7,
        cache_seqlens_list=cache_seqlens,
        page_table_width=10,
    )
    _run_mimo_v25_fp8_decode_graph_case(
        window_left=127,
        seed=3200,
        batch=7,
        cache_seqlens_list=cache_seqlens,
        page_table_width=10,
    )


@torch.inference_mode()
def test_mimo_v25_fp8_decode_graph_pads_large_power2_page_table_width(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    require_b12x()
    paged_api.clear_paged_caches()

    def fail_regular_metadata_update(**kwargs: object) -> None:
        raise AssertionError(
            "full-window non-split decode should use cached graph metadata"
        )

    monkeypatch.setattr(
        "b12x.attention.paged.graph_replay.update_regular_decode_graph_chunk_metadata",
        fail_regular_metadata_update,
    )
    _run_mimo_v25_fp8_decode_graph_case(
        window_left=-1,
        seed=3300,
        batch=8,
        cache_seqlens_list=[1] * 8,
        page_table_width=16_384,
        num_pages=2048,
        expect_page_table_width=16_385,
    )
    _run_mimo_v25_fp8_decode_graph_case(
        window_left=127,
        seed=3400,
        batch=8,
        cache_seqlens_list=[129, 193, 257, 385, 449, 513, 577, 641],
        page_table_width=16_384,
        num_pages=2048,
        expect_page_table_width=16_385,
    )
