from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from b12x.attention.paged._tuning import GqaConfig
from b12x.attention.paged.planner import (
    PagedPlanBudget,
    build_decode_chunk_pages_lut,
    create_paged_plan,
    decode_chunk_pages_for_graph,
    infer_paged_mode,
    plan_decode_graph_capacity,
    plan_extend_graph_capacity,
    plan_verify_graph_capacity,
    resolve_decode_graph_ctas_per_sm,
    use_paged_extend_fp8_pv_repack,
)
from b12x.preparation import FrozenMapping
from b12x.attention.paged._scratch import (
    B12XPagedAttentionScratchCaps,
    _paged_attention_scratch_layout,
    plan_decode_graph_scratch_envelope,
    plan_paged_attention_scratch,
)


def _make_inputs(
    *,
    q_seqlens: list[int],
    cache_seqlens: list[int],
    page_size: int = 64,
    q_heads: int = 8,
    kv_heads: int = 1,
    head_dim_qk: int = 256,
    head_dim_vo: int = 256,
    dtype: torch.dtype = torch.bfloat16,
    kv_dtype: torch.dtype = torch.float8_e4m3fn,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = "cuda"
    batch = len(q_seqlens)
    total_q = sum(q_seqlens)
    q = torch.randn(total_q, q_heads, head_dim_qk, dtype=dtype, device=device)
    max_pages = max((cache_len + page_size - 1) // page_size for cache_len in cache_seqlens)
    num_pages = sum((cache_len + page_size - 1) // page_size for cache_len in cache_seqlens) + 8
    k_cache = torch.randn(
        num_pages, page_size, kv_heads, head_dim_qk, dtype=torch.float32, device=device
    ).to(kv_dtype)
    v_cache = torch.randn(
        num_pages, page_size, kv_heads, head_dim_vo, dtype=torch.float32, device=device
    ).to(kv_dtype)
    page_table = torch.zeros(batch, max_pages, dtype=torch.int32, device=device)
    cursor = 0
    for request_idx, cache_len in enumerate(cache_seqlens):
        req_pages = (cache_len + page_size - 1) // page_size
        page_ids = torch.arange(cursor, cursor + req_pages, dtype=torch.int32, device=device)
        cursor += req_pages
        page_table[request_idx, :req_pages] = page_ids
        page_table[request_idx, req_pages:] = page_ids[-1]
    cache_seqlens_t = torch.tensor(cache_seqlens, dtype=torch.int32, device=device)
    offsets = [0]
    for q_len in q_seqlens:
        offsets.append(offsets[-1] + q_len)
    cu_seqlens_q = torch.tensor(offsets, dtype=torch.int32, device=device)
    return q, k_cache, v_cache, page_table, cache_seqlens_t, cu_seqlens_q


def _fp8_pv_repack_plan(*, num_qo_tiles: int) -> SimpleNamespace:
    return SimpleNamespace(
        mode="extend",
        dtype=torch.bfloat16,
        kv_dtype=torch.float8_e4m3fn,
        head_dim_qk=256,
        head_dim_vo=256,
        page_size=64,
        cta_tile_q=64,
        window_left=-1,
        msa_block_sparse=False,
        num_qo_tiles=num_qo_tiles,
        num_kv_heads=4,
    )


def test_fp8_pv_repack_policy_uses_sm_wave_cutoff() -> None:
    assert use_paged_extend_fp8_pv_repack(
        _fp8_pv_repack_plan(num_qo_tiles=384),
        resident_ctas_per_sm=2,
        num_sms=48,
    )
    assert not use_paged_extend_fp8_pv_repack(
        _fp8_pv_repack_plan(num_qo_tiles=385),
        resident_ctas_per_sm=2,
        num_sms=48,
    )


def test_fp8_pv_repack_policy_rejects_other_contracts() -> None:
    plan = _fp8_pv_repack_plan(num_qo_tiles=192)
    plan.head_dim_vo = 128
    assert not use_paged_extend_fp8_pv_repack(
        plan,
        resident_ctas_per_sm=2,
        num_sms=48,
    )


def test_paged_infers_decode_mode() -> None:
    _, _, _, _, _, cu_seqlens_q = _make_inputs(q_seqlens=[1, 1, 1], cache_seqlens=[64, 128, 192])
    assert infer_paged_mode(cu_seqlens_q) == "decode"


def test_paged_infers_extend_mode() -> None:
    _, _, _, _, _, cu_seqlens_q = _make_inputs(q_seqlens=[1, 6], cache_seqlens=[64, 128])
    assert infer_paged_mode(cu_seqlens_q) == "extend"


def test_paged_decode_plan_ignores_fixed_split_metadata() -> None:
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_inputs(
        q_seqlens=[1, 1],
        cache_seqlens=[2048, 4096],
    )
    plan = create_paged_plan(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        fixed_split_size=8,
    )

    assert plan.mode == "decode"
    assert plan.cta_tile_q == 16
    assert plan.split_kv is False
    assert plan.kv_chunk_size == 64 * 64
    assert plan.request_indices == (0, 1)
    assert plan.qo_tile_indices == (0, 0)
    assert plan.kv_tile_indices == (0, 0)
    assert plan.merge_indptr == (0, 1, 2)
    assert plan.o_indptr == (0, 1, 2)
    assert plan.total_num_partial_rows == 0


def test_paged_eager_decode_plan_honors_explicit_forced_split_kv() -> None:
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_inputs(
        q_seqlens=[1, 1],
        cache_seqlens=[2048, 4096],
    )

    plan = create_paged_plan(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        fixed_split_size=8,
        force_split_kv=True,
    )

    assert plan.split_kv is True
    assert plan.kv_chunk_size == 8 * 64
    assert plan.new_batch_size == 12
    assert plan.total_num_partial_rows == 12


def test_paged_scratch_shapes_follow_plan_metadata() -> None:
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_inputs(
        q_seqlens=[1, 6],
        cache_seqlens=[2048, 8192],
    )
    plan = create_paged_plan(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
    )
    scratch_plan = plan_paged_attention_scratch(
        B12XPagedAttentionScratchCaps(
            device=q.device,
            mode="extend",
            dtype=q.dtype,
            kv_dtype=k_cache.dtype,
            num_q_heads=q.shape[1],
            num_kv_heads=k_cache.shape[2],
            head_dim_qk=q.shape[2],
            head_dim_vo=v_cache.shape[3],
            page_size=k_cache.shape[1],
            max_total_q=plan.total_q,
            max_batch=page_table.shape[0],
            max_page_table_width=page_table.shape[1],
            max_work_items=plan.new_batch_size,
            max_partial_rows=plan.total_num_partial_rows,
            num_cache_pages=k_cache.shape[0],
        )
    )
    scratch = tuple(
        torch.empty(shape, dtype=dtype, device=q.device)
        for shape, dtype in scratch_plan.shapes_and_dtypes()
    )
    binding = scratch_plan.bind(
        scratch=scratch,
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        output=torch.empty((q.shape[0], q.shape[1], v_cache.shape[3]), dtype=q.dtype, device=q.device),
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
    )
    scratch_views = binding.scratch

    assert scratch_views.lse.shape == (8, 7)
    assert scratch_views.kv_chunk_size_ptr.item() == 128 * 64
    assert scratch_views.total_num_rows_ptr.item() == 7
    assert scratch_views.request_indices.shape[0] == plan.new_batch_size
    assert scratch_views.merge_indptr.shape[0] == plan.total_q + 1
    assert scratch_views.o_indptr.shape[0] == page_table.shape[0] + 1
    assert scratch_views.tmp_output is None
    assert scratch_views.tmp_lse is None


def test_paged_fp8_decode_plan_uses_single_kv_span() -> None:
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_inputs(
        q_seqlens=[1] * 8,
        cache_seqlens=[8192] * 8,
    )
    plan = create_paged_plan(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
    )

    assert plan.mode == "decode"
    assert plan.kv_chunk_size == 8192
    assert plan.new_batch_size == page_table.shape[0]
    assert plan.total_num_partial_rows == 0
    assert plan.split_kv is False


@pytest.mark.parametrize("kv_dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_paged_short_extend_plan_uses_cta_tile_q_16(
    kv_dtype: torch.dtype,
) -> None:
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_inputs(
        q_seqlens=[4],
        cache_seqlens=[4096],
        kv_dtype=kv_dtype,
    )
    plan = create_paged_plan(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
    )

    assert plan.mode == "extend"
    assert plan.cta_tile_q == 16
    assert plan.split_kv is False


@pytest.mark.parametrize("kv_dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_paged_verify_plan_uses_decode_style_split_kv(
    kv_dtype: torch.dtype,
) -> None:
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_inputs(
        q_seqlens=[4],
        cache_seqlens=[65536],
        kv_dtype=kv_dtype,
    )
    plan = create_paged_plan(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        mode="verify",
        enable_cuda_graph=True,
        graph_chunk_policy=True,
    )

    assert plan.mode == "verify"
    assert plan.cta_tile_q == 16
    assert plan.split_kv is True
    assert plan.kv_chunk_size == 22 * 64
    assert plan.graph_ctas_per_sm == 2
    assert plan.total_num_partial_rows > 0
    assert plan.new_batch_size > 2


def test_paged_plan_disables_split_kv_when_merge_backend_is_unsupported() -> None:
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_inputs(
        q_seqlens=[6] * 8,
        cache_seqlens=[8192] * 8,
        q_heads=48,
        kv_heads=8,
        head_dim_qk=128,
        head_dim_vo=128,
        kv_dtype=torch.bfloat16,
    )
    plan = create_paged_plan(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        enable_cuda_graph=True,
        graph_chunk_policy=True,
    )

    assert plan.split_kv is False


def test_paged_graph_budget_is_independent_of_cache_length() -> None:
    short_inputs = _make_inputs(
        q_seqlens=[1] * 8,
        cache_seqlens=[2048] * 8,
        kv_dtype=torch.bfloat16,
    )
    long_inputs = _make_inputs(
        q_seqlens=[1] * 8,
        cache_seqlens=[32768] * 8,
        kv_dtype=torch.bfloat16,
    )

    short_plan = create_paged_plan(
        *short_inputs,
        enable_cuda_graph=True,
        graph_chunk_policy=True,
        graph_ctas_per_sm=2,
    )
    long_plan = create_paged_plan(
        *long_inputs,
        enable_cuda_graph=True,
        graph_chunk_policy=True,
        graph_ctas_per_sm=2,
    )

    expected_budget = int(torch.cuda.get_device_properties("cuda").multi_processor_count) * 2
    assert short_plan.graph_ctas_per_sm == 2
    assert long_plan.graph_ctas_per_sm == 2
    assert short_plan.max_batch_size_if_split == expected_budget
    assert long_plan.max_batch_size_if_split == expected_budget
    assert short_plan.padded_batch_size == long_plan.padded_batch_size == expected_budget


def test_paged_graph_mode_falls_back_when_heuristic_overflows_budget() -> None:
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_inputs(
        q_seqlens=[6] * 8,
        cache_seqlens=[128000] * 8,
        kv_dtype=torch.bfloat16,
    )
    plan = create_paged_plan(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        enable_cuda_graph=True,
        graph_chunk_policy=True,
        graph_ctas_per_sm=2,
    )

    assert plan.new_batch_size <= plan.padded_batch_size


def test_paged_extend_plan_respects_fixed_partial_row_budget() -> None:
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_inputs(
        q_seqlens=[128],
        cache_seqlens=[128 * 64],
        kv_dtype=torch.bfloat16,
    )
    plan = create_paged_plan(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        plan_budget=PagedPlanBudget(
            max_total_q=128,
            max_batch=1,
            max_page_table_width=128,
            max_work_items=4096,
            max_partial_rows=512,
        ),
    )

    assert plan.mode == "extend"
    assert plan.split_kv is False
    assert plan.total_num_partial_rows == 0


def test_paged_extend_plan_budget_can_force_nosplit() -> None:
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_inputs(
        q_seqlens=[64],
        cache_seqlens=[256 * 64],
        kv_dtype=torch.bfloat16,
    )
    plan = create_paged_plan(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        plan_budget=PagedPlanBudget(
            max_total_q=64,
            max_batch=1,
            max_page_table_width=256,
            max_work_items=4096,
            max_partial_rows=0,
        ),
    )

    assert plan.mode == "extend"
    assert plan.split_kv is False
    assert plan.total_num_partial_rows == 0


def test_paged_extend_plan_rejects_fixed_split_smaller_than_full_span() -> None:
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_inputs(
        q_seqlens=[6] * 8,
        cache_seqlens=[8192] * 8,
        kv_dtype=torch.bfloat16,
    )

    with pytest.raises(
        ValueError,
        match="extend fixed_split_size must cover the full effective KV span",
    ):
        create_paged_plan(
            q,
            k_cache,
            v_cache,
            page_table,
            cache_seqlens,
            cu_seqlens_q,
            fixed_split_size=8,
        )


def test_paged_graph_mode_defaults_to_decode_graph_split_heuristic() -> None:
    batch = 7
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_inputs(
        q_seqlens=[1] * batch,
        cache_seqlens=[8192] * batch,
        kv_dtype=torch.bfloat16,
    )
    plan = create_paged_plan(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        enable_cuda_graph=True,
        graph_chunk_policy=True,
    )
    expected_budget = int(torch.cuda.get_device_properties("cuda").multi_processor_count) * 2
    assert plan.graph_ctas_per_sm == 2
    assert plan.max_batch_size_if_split == expected_budget
    expected_chunk_pages = decode_chunk_pages_for_graph(
        q_dtype=q.dtype,
        kv_dtype=k_cache.dtype,
        batch=batch,
        page_size=int(k_cache.shape[1]),
        head_dim_qk=int(q.shape[2]),
        head_dim_vo=int(v_cache.shape[3]),
        gqa_group_size=int(q.shape[1] // k_cache.shape[2]),
        max_effective_kv_pages=8192 // 64,
        max_chunks_per_req=expected_budget // batch,
    )
    assert expected_chunk_pages is not None
    assert plan.split_kv is True
    assert plan.kv_chunk_size == expected_chunk_pages * 64
    assert plan.new_batch_size > batch
    assert plan.total_num_partial_rows == plan.new_batch_size


def test_paged_graph_decode_auto_split_preserves_explicit_opt_outs() -> None:
    inputs = _make_inputs(
        q_seqlens=[1] * 4,
        cache_seqlens=[8192] * 4,
        kv_dtype=torch.bfloat16,
    )

    explicit_false = create_paged_plan(
        *inputs,
        enable_cuda_graph=True,
        graph_chunk_policy=True,
        force_split_kv=False,
    )
    explicit_disable = create_paged_plan(
        *inputs,
        enable_cuda_graph=True,
        graph_chunk_policy=True,
        disable_split_kv=True,
    )

    assert explicit_false.split_kv is False
    assert explicit_disable.split_kv is False
    assert explicit_false.new_batch_size == explicit_disable.new_batch_size == 4


def test_paged_graph_decode_auto_split_uses_short_window_chunk_policy() -> None:
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_inputs(
        q_seqlens=[1],
        cache_seqlens=[8192],
        page_size=128,
        q_heads=36,
        kv_heads=4,
        head_dim_qk=128,
        head_dim_vo=128,
        kv_dtype=torch.float8_e4m3fn,
    )

    short_window_plan = create_paged_plan(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        enable_cuda_graph=True,
        graph_chunk_policy=True,
        window_left=512,
    )
    explicit_direct_plan = create_paged_plan(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        enable_cuda_graph=True,
        graph_chunk_policy=True,
        window_left=512,
        force_split_kv=False,
    )
    long_full_plan = create_paged_plan(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        enable_cuda_graph=True,
        graph_chunk_policy=True,
        window_left=-1,
    )

    assert short_window_plan.split_kv is True
    assert short_window_plan.kv_chunk_size == 128
    assert short_window_plan.new_batch_size == 5
    assert explicit_direct_plan.split_kv is False
    assert explicit_direct_plan.new_batch_size == 1
    assert long_full_plan.split_kv is True
    assert long_full_plan.new_batch_size > 1


def test_paged_graph_decode_auto_split_falls_back_with_fixed_capacity() -> None:
    inputs = _make_inputs(
        q_seqlens=[1] * 8,
        cache_seqlens=[8192] * 8,
        kv_dtype=torch.bfloat16,
    )
    no_partial_budget = PagedPlanBudget(
        max_total_q=8,
        max_batch=8,
        max_page_table_width=128,
        max_work_items=16,
        max_partial_rows=0,
    )

    plan = create_paged_plan(
        *inputs,
        enable_cuda_graph=True,
        graph_chunk_policy=True,
        max_batch_size_if_split=16,
        plan_budget=no_partial_budget,
    )

    assert plan.split_kv is False
    assert plan.new_batch_size == 8
    assert plan.total_num_partial_rows == 0

    with pytest.raises(ValueError, match="workspace budget"):
        create_paged_plan(
            *inputs,
            enable_cuda_graph=True,
            graph_chunk_policy=True,
            max_batch_size_if_split=16,
            plan_budget=no_partial_budget,
            force_split_kv=True,
        )


def test_decode_graph_capacity_is_static_exact_and_capacity_aware() -> None:
    capacity = plan_decode_graph_capacity(
        device=torch.device("cuda"),
        q_dtype=torch.bfloat16,
        kv_dtype=torch.float8_e4m3fn,
        num_q_heads=36,
        num_kv_heads=4,
        head_dim_qk=128,
        head_dim_vo=128,
        page_size=128,
        batch=1,
        max_cache_page_count=512,
        window_left=512,
    )

    assert capacity.query_tiles_per_request == 1
    assert capacity.max_effective_kv_pages == 5
    assert len(capacity.chunk_pages_lut) == 5
    assert capacity.max_chunks_per_request == 5
    assert capacity.max_work_items == 5
    assert capacity.max_partial_rows == 5

    direct_only = plan_decode_graph_capacity(
        device=torch.device("cuda"),
        q_dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
        num_q_heads=8,
        num_kv_heads=1,
        head_dim_qk=256,
        head_dim_vo=256,
        page_size=64,
        batch=8,
        max_cache_page_count=128,
        max_work_items=8,
        max_partial_rows=0,
    )

    assert direct_only.max_chunks_per_request == 1
    assert direct_only.max_work_items == 8
    assert direct_only.max_partial_rows == 0
    assert all(
        (page_count + chunk_pages - 1) // chunk_pages == 1
        for page_count, chunk_pages in enumerate(
            direct_only.chunk_pages_lut, start=1
        )
    )


def test_decode_graph_capacity_counts_multiple_query_tiles() -> None:
    capacity = plan_decode_graph_capacity(
        device=torch.device("cuda"),
        q_dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
        num_q_heads=128,
        num_kv_heads=1,
        head_dim_qk=256,
        head_dim_vo=256,
        page_size=64,
        batch=2,
        max_cache_page_count=64,
        max_work_items=16,
        max_partial_rows=0,
    )

    assert capacity.cta_tile_q == 16
    assert capacity.query_tiles_per_request == 8
    assert capacity.max_chunks_per_request == 1
    assert capacity.max_work_items == 16
    assert capacity.max_partial_rows == 0


@pytest.mark.parametrize(
    ("total_q_capacity", "expected_work_items"),
    (
        (128, 94),
        (1024, 96),
        (8192, 768),
    ),
)
def test_extend_graph_capacity_buckets_match_laguna_full_geometry(
    monkeypatch: pytest.MonkeyPatch,
    total_q_capacity: int,
    expected_work_items: int,
) -> None:
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(multi_processor_count=188),
    )
    monkeypatch.setattr(
        torch.cuda,
        "get_device_capability",
        lambda _device: (12, 0),
    )

    capacity = plan_extend_graph_capacity(
        device=torch.device("cuda:0"),
        q_dtype=torch.bfloat16,
        kv_dtype=torch.float8_e4m3fn,
        num_q_heads=24,
        num_kv_heads=4,
        head_dim_qk=128,
        head_dim_vo=128,
        page_size=128,
        batch=1,
        total_q_capacity=total_q_capacity,
        max_cache_page_count=2048,
    )

    assert capacity.graph_ctas_per_sm == 2
    assert capacity.cta_tile_q == 64
    assert capacity.max_work_items == expected_work_items
    assert capacity.max_effective_kv_pages == 2048
    assert capacity.representative_cache_seqlen == 262144


def test_extend_graph_capacity_includes_batch_packing_and_window_span(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(multi_processor_count=188),
    )

    capacity = plan_extend_graph_capacity(
        device=torch.device("cuda:0"),
        q_dtype=torch.bfloat16,
        kv_dtype=torch.float8_e4m3fn,
        num_q_heads=36,
        num_kv_heads=4,
        head_dim_qk=128,
        head_dim_vo=128,
        page_size=128,
        batch=8,
        total_q_capacity=8192,
        max_cache_page_count=2048,
        window_left=511,
    )

    assert capacity.graph_ctas_per_sm == 2
    assert capacity.cta_tile_q == 128
    assert capacity.max_work_items == 583
    assert capacity.max_effective_kv_pages == 68


@pytest.mark.parametrize(
    (
        "page_size",
        "num_q_heads",
        "max_cache_pages",
        "window_left",
        "expected_effective_pages",
        "expected_chunk_pages",
        "expected_partial_rows",
    ),
    [
        (64, 24, 2048, -1, 2048, 67, 248),
        (128, 36, 1024, 511, 6, 1, 48),
    ],
)
def test_verify_graph_capacity_matches_laguna_geometry(
    monkeypatch: pytest.MonkeyPatch,
    page_size: int,
    num_q_heads: int,
    max_cache_pages: int,
    window_left: int,
    expected_effective_pages: int,
    expected_chunk_pages: int,
    expected_partial_rows: int,
) -> None:
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(multi_processor_count=188),
    )

    capacity = plan_verify_graph_capacity(
        device=torch.device("cuda", 0),
        q_dtype=torch.bfloat16,
        kv_dtype=torch.float8_e4m3fn,
        num_q_heads=num_q_heads,
        num_kv_heads=4,
        head_dim_qk=128,
        head_dim_vo=128,
        page_size=page_size,
        batch=1,
        query_len=8,
        max_cache_page_count=max_cache_pages,
        window_left=window_left,
    )

    assert capacity.cta_tile_q == 16
    assert capacity.max_work_items == 94
    assert capacity.max_effective_kv_pages == expected_effective_pages
    assert capacity.kv_chunk_size_pages == expected_chunk_pages
    assert capacity.max_partial_rows == expected_partial_rows
    expected_cache_seqlen = (
        max_cache_pages * page_size
        if window_left < 0
        else (max_cache_pages - 1) * page_size + 1
    )
    assert capacity.representative_cache_seqlen == expected_cache_seqlen


@pytest.mark.parametrize(
    (
        "batch",
        "max_cache_pages",
        "expected_work_items",
        "expected_chunks_per_request",
        "expected_partial_rows",
    ),
    (
        (1, 512, 47, 47, 376),
        (1, 2048, 94, 94, 752),
        (2, 2048, 94, 47, 752),
        (4, 2048, 94, 23, 736),
        (8, 2048, 94, 11, 704),
    ),
)
def test_verify_graph_capacity_reserves_exact_laguna_split_rectangle(
    monkeypatch: pytest.MonkeyPatch,
    batch: int,
    max_cache_pages: int,
    expected_work_items: int,
    expected_chunks_per_request: int,
    expected_partial_rows: int,
) -> None:
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(multi_processor_count=188),
    )
    monkeypatch.setattr(
        torch.cuda,
        "get_device_capability",
        lambda _device: (12, 0),
    )

    capacity = plan_verify_graph_capacity(
        device=torch.device("cuda:0"),
        q_dtype=torch.bfloat16,
        kv_dtype=torch.float8_e4m3fn,
        num_q_heads=24,
        num_kv_heads=4,
        head_dim_qk=128,
        head_dim_vo=128,
        page_size=128,
        batch=batch,
        query_len=8,
        max_cache_page_count=max_cache_pages,
    )

    assert capacity.cta_tile_q == 64
    assert capacity.max_work_items == expected_work_items
    assert capacity.max_partial_rows == expected_partial_rows
    assert capacity.max_partial_rows // (batch * 8) == expected_chunks_per_request


def test_decode_graph_scratch_envelope_covers_every_batch_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Pin the architecture budget so this regression is deterministic on both
    # DGX Spark and workstation Blackwell.  This Laguna-like geometry has a
    # larger mid-batch split layout than its batch-256 direct layout.
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(
            multi_processor_count=159,
            major=12,
            minor=0,
        ),
    )
    geometry = dict(
        device=torch.device("cuda:0"),
        q_dtype=torch.bfloat16,
        kv_dtype=torch.float8_e4m3fn,
        num_q_heads=36,
        num_kv_heads=4,
        head_dim_qk=128,
        head_dim_vo=128,
        page_size=128,
        max_cache_page_count=512,
    )
    max_batch = 256

    bucket_nbytes: list[int] = []
    for batch in range(1, max_batch + 1):
        capacity = plan_decode_graph_capacity(**geometry, batch=batch)
        caps = B12XPagedAttentionScratchCaps(
            device=geometry["device"],
            mode="decode",
            dtype=geometry["q_dtype"],
            kv_dtype=geometry["kv_dtype"],
            num_q_heads=geometry["num_q_heads"],
            num_kv_heads=geometry["num_kv_heads"],
            head_dim_qk=geometry["head_dim_qk"],
            head_dim_vo=geometry["head_dim_vo"],
            page_size=geometry["page_size"],
            max_total_q=batch,
            max_batch=batch,
            max_page_table_width=geometry["max_cache_page_count"],
            max_work_items=capacity.max_work_items,
            max_partial_rows=capacity.max_partial_rows,
            num_cache_pages=1,
            use_cuda_graph=True,
            copy_runtime_metadata=True,
        )
        bucket_nbytes.append(int(_paged_attention_scratch_layout(caps).nbytes))

    envelope = plan_decode_graph_scratch_envelope(
        **geometry,
        max_batch=max_batch,
        max_page_table_width=geometry["max_cache_page_count"],
    )
    expected_nbytes = max(bucket_nbytes)
    expected_witness = bucket_nbytes.index(expected_nbytes) + 1

    assert expected_nbytes > bucket_nbytes[-1]
    assert envelope.nbytes == expected_nbytes
    assert envelope.witness_batch == expected_witness


def test_decode_graph_scratch_envelope_applies_direct_only_storage_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(multi_processor_count=159),
    )
    monkeypatch.setattr(
        torch.cuda,
        "get_device_capability",
        lambda _device: (12, 0),
    )
    envelope = plan_decode_graph_scratch_envelope(
        device=torch.device("cuda:0"),
        q_dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
        num_q_heads=8,
        num_kv_heads=1,
        head_dim_qk=256,
        head_dim_vo=256,
        page_size=64,
        max_batch=8,
        max_page_table_width=128,
        max_partial_rows=0,
    )
    batch_eight_capacity = plan_decode_graph_capacity(
        device=torch.device("cuda:0"),
        q_dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
        num_q_heads=8,
        num_kv_heads=1,
        head_dim_qk=256,
        head_dim_vo=256,
        page_size=64,
        batch=8,
        max_cache_page_count=128,
        max_partial_rows=0,
    )

    assert batch_eight_capacity.max_partial_rows == 0
    assert 1 <= envelope.witness_batch <= 8
    assert envelope.nbytes > 0


def test_prepare_decode_graph_replay_state_respects_direct_only_caps() -> None:
    caps = B12XPagedAttentionScratchCaps(
        device=torch.device("cuda"),
        mode="decode",
        dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
        num_q_heads=8,
        num_kv_heads=1,
        head_dim_qk=256,
        head_dim_vo=256,
        page_size=64,
        max_total_q=8,
        max_batch=8,
        max_page_table_width=128,
        max_work_items=8,
        max_partial_rows=0,
        num_cache_pages=128,
        use_cuda_graph=True,
    )
    scratch_plan = plan_paged_attention_scratch(caps)

    scratch_plan.prepare_decode_graph_replay_state(
        batch=8,
        max_page_table_width=128,
        max_cache_page_count=128,
    )

    assert scratch_plan.plan.split_kv is False
    assert scratch_plan.plan.new_batch_size == 8
    assert scratch_plan.plan.total_num_partial_rows == 0


def test_prepare_windowed_decode_uses_nonmonotone_lut_worst_page_count() -> None:
    capacity = plan_decode_graph_capacity(
        device=torch.device("cuda"),
        q_dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
        num_q_heads=128,
        num_kv_heads=1,
        head_dim_qk=256,
        head_dim_vo=256,
        page_size=64,
        batch=3,
        max_cache_page_count=67,
        window_left=1023,
    )
    # This policy is intentionally non-monotone: the schedule maximum occurs
    # before the longest representable windowed cache span.
    assert capacity.worst_page_count < capacity.max_effective_kv_pages

    scratch_plan = plan_paged_attention_scratch(
        B12XPagedAttentionScratchCaps(
            device=torch.device("cuda"),
            mode="decode",
            dtype=torch.bfloat16,
            kv_dtype=torch.bfloat16,
            num_q_heads=128,
            num_kv_heads=1,
            head_dim_qk=256,
            head_dim_vo=256,
            page_size=64,
            max_total_q=3,
            max_batch=3,
            max_page_table_width=67,
            max_work_items=capacity.max_work_items,
            max_partial_rows=capacity.max_partial_rows,
            num_cache_pages=67,
            use_cuda_graph=True,
        )
    )
    scratch_plan.prepare_decode_graph_replay_state(
        batch=3,
        max_page_table_width=67,
        max_cache_page_count=67,
        window_left=1023,
    )

    assert scratch_plan.plan.new_batch_size == capacity.max_work_items
    assert (
        scratch_plan.plan.total_num_partial_rows == capacity.max_partial_rows
    )


def test_decode_graph_chunk_pages_for_graph_uses_heuristic() -> None:
    assert (
        decode_chunk_pages_for_graph(
            q_dtype=torch.bfloat16,
            kv_dtype=torch.bfloat16,
            batch=4,
            page_size=64,
            head_dim_qk=256,
            head_dim_vo=256,
            gqa_group_size=8,
            max_effective_kv_pages=32,
        )
        == 1
    )
    assert (
        decode_chunk_pages_for_graph(
            q_dtype=torch.bfloat16,
            kv_dtype=torch.bfloat16,
            batch=4,
            page_size=64,
            head_dim_qk=256,
            head_dim_vo=256,
            gqa_group_size=8,
            max_effective_kv_pages=256,
        )
        == 6
    )
    assert (
        decode_chunk_pages_for_graph(
            q_dtype=torch.bfloat16,
            kv_dtype=torch.bfloat16,
            batch=8,
            page_size=64,
            head_dim_qk=192,
            head_dim_vo=128,
            gqa_group_size=16,
            max_effective_kv_pages=256,
        )
        == 32
    )


def test_decode_graph_chunk_pages_uses_finer_fp8_minimax_bs1_splits() -> None:
    assert (
        decode_chunk_pages_for_graph(
            q_dtype=torch.bfloat16,
            kv_dtype=torch.float8_e4m3fn,
            batch=1,
            page_size=64,
            head_dim_qk=128,
            head_dim_vo=128,
            gqa_group_size=6,
            max_effective_kv_pages=257,
        )
        == 6
    )
    assert (
        decode_chunk_pages_for_graph(
            q_dtype=torch.bfloat16,
            kv_dtype=torch.float8_e4m3fn,
            batch=2,
            page_size=64,
            head_dim_qk=128,
            head_dim_vo=128,
            gqa_group_size=6,
            max_effective_kv_pages=257,
        )
        == 9
    )


def test_decode_graph_bf16_gqa8_bs1_scales_capacity_budgets_with_sms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(multi_processor_count=48),
    )
    monkeypatch.setattr(
        torch.cuda,
        "get_device_capability",
        lambda _device: (12, 1),
    )

    capacity = plan_decode_graph_capacity(
        device=torch.device("cuda:0"),
        q_dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
        num_q_heads=8,
        num_kv_heads=1,
        head_dim_qk=256,
        head_dim_vo=256,
        page_size=64,
        batch=1,
        max_cache_page_count=513,
    )

    assert capacity.architecture_max_chunks_per_request == 288
    assert capacity.max_chunks_per_request == 14
    assert capacity.max_work_items == 14
    assert capacity.max_partial_rows == 14
    assert capacity.chunk_pages_lut[-1] == 37

    shorter_bucket = plan_decode_graph_capacity(
        device=torch.device("cuda:0"),
        q_dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
        num_q_heads=8,
        num_kv_heads=1,
        head_dim_qk=256,
        head_dim_vo=256,
        page_size=64,
        batch=1,
        max_cache_page_count=257,
    )
    assert shorter_bucket.max_chunks_per_request == 19
    assert shorter_bucket.max_work_items == 19
    assert shorter_bucket.max_partial_rows == 19
    assert shorter_bucket.chunk_pages_lut[-1] == 14

    longer_bucket = plan_decode_graph_capacity(
        device=torch.device("cuda:0"),
        q_dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
        num_q_heads=8,
        num_kv_heads=1,
        head_dim_qk=256,
        head_dim_vo=256,
        page_size=64,
        batch=1,
        max_cache_page_count=1025,
    )
    assert longer_bucket.max_chunks_per_request == 17
    assert longer_bucket.max_work_items == 17
    assert longer_bucket.max_partial_rows == 17
    assert longer_bucket.chunk_pages_lut[-1] == 61

    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(multi_processor_count=188),
    )
    monkeypatch.setattr(
        torch.cuda,
        "get_device_capability",
        lambda _device: (12, 0),
    )
    for max_cache_page_count, expected_chunks in (
        (257, 74),
        (513, 55),
        (1025, 67),
    ):
        capacity = plan_decode_graph_capacity(
            device=torch.device("cuda:0"),
            q_dtype=torch.bfloat16,
            kv_dtype=torch.bfloat16,
            num_q_heads=8,
            num_kv_heads=1,
            head_dim_qk=256,
            head_dim_vo=256,
            page_size=64,
            batch=1,
            max_cache_page_count=max_cache_page_count,
        )
        assert capacity.max_chunks_per_request == expected_chunks
        assert capacity.max_work_items == expected_chunks
        assert capacity.max_partial_rows == expected_chunks


def test_decode_graph_fp8_gqa8_bs1_scales_capacity_budgets_with_sms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(multi_processor_count=48),
    )
    monkeypatch.setattr(
        torch.cuda,
        "get_device_capability",
        lambda _device: (12, 1),
    )

    capacity = plan_decode_graph_capacity(
        device=torch.device("cuda:0"),
        q_dtype=torch.bfloat16,
        kv_dtype=torch.float8_e4m3fn,
        num_q_heads=8,
        num_kv_heads=1,
        head_dim_qk=256,
        head_dim_vo=256,
        page_size=64,
        batch=1,
        max_cache_page_count=513,
    )

    assert capacity.architecture_max_chunks_per_request == 96
    assert capacity.max_chunks_per_request == 18
    assert capacity.max_work_items == 18
    assert capacity.max_partial_rows == 18
    assert capacity.chunk_pages_lut[-1] == 29

    shorter_bucket = plan_decode_graph_capacity(
        device=torch.device("cuda:0"),
        q_dtype=torch.bfloat16,
        kv_dtype=torch.float8_e4m3fn,
        num_q_heads=8,
        num_kv_heads=1,
        head_dim_qk=256,
        head_dim_vo=256,
        page_size=64,
        batch=1,
        max_cache_page_count=257,
    )
    assert shorter_bucket.max_chunks_per_request == 22
    assert shorter_bucket.max_work_items == 22
    assert shorter_bucket.max_partial_rows == 22
    assert shorter_bucket.chunk_pages_lut[-1] == 12

    longer_bucket = plan_decode_graph_capacity(
        device=torch.device("cuda:0"),
        q_dtype=torch.bfloat16,
        kv_dtype=torch.float8_e4m3fn,
        num_q_heads=8,
        num_kv_heads=1,
        head_dim_qk=256,
        head_dim_vo=256,
        page_size=64,
        batch=1,
        max_cache_page_count=1025,
    )
    assert longer_bucket.max_chunks_per_request == 18
    assert longer_bucket.max_work_items == 18
    assert longer_bucket.max_partial_rows == 18
    assert longer_bucket.chunk_pages_lut[-1] == 57

    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(multi_processor_count=188),
    )
    monkeypatch.setattr(
        torch.cuda,
        "get_device_capability",
        lambda _device: (12, 0),
    )
    for max_cache_page_count, expected_chunks in (
        (257, 86),
        (513, 71),
        (1025, 71),
    ):
        capacity = plan_decode_graph_capacity(
            device=torch.device("cuda:0"),
            q_dtype=torch.bfloat16,
            kv_dtype=torch.float8_e4m3fn,
            num_q_heads=8,
            num_kv_heads=1,
            head_dim_qk=256,
            head_dim_vo=256,
            page_size=64,
            batch=1,
            max_cache_page_count=max_cache_page_count,
        )
        assert capacity.max_chunks_per_request == expected_chunks
        assert capacity.max_work_items == expected_chunks
        assert capacity.max_partial_rows == expected_chunks


def test_decode_graph_page128_laguna_keeps_adaptive_one_wave_grid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = dict(
        q_dtype=torch.bfloat16,
        kv_dtype=torch.float8_e4m3fn,
        batch=1,
        page_size=128,
        head_dim_qk=128,
        head_dim_vo=128,
        gqa_group_size=9,
        max_chunks_per_req=94,
    )

    assert decode_chunk_pages_for_graph(
        max_effective_kv_pages=511, **kwargs
    ) == 11
    assert decode_chunk_pages_for_graph(
        max_effective_kv_pages=512, **kwargs
    ) == 11
    assert decode_chunk_pages_for_graph(
        max_effective_kv_pages=513, **kwargs
    ) == 11

    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(multi_processor_count=188),
    )
    monkeypatch.setattr(
        torch.cuda,
        "get_device_capability",
        lambda _device: (12, 0),
    )
    capacity = plan_decode_graph_capacity(
        device=torch.device("cuda:0"),
        q_dtype=torch.bfloat16,
        kv_dtype=torch.float8_e4m3fn,
        num_q_heads=36,
        num_kv_heads=4,
        head_dim_qk=128,
        head_dim_vo=128,
        page_size=128,
        batch=1,
        max_cache_page_count=513,
    )
    assert capacity.architecture_max_chunks_per_request == 94
    assert capacity.max_chunks_per_request == 47
    assert capacity.max_work_items == 47
    assert capacity.max_partial_rows == 47
    assert capacity.worst_page_count == 47
    assert capacity.chunk_pages_lut[45:49] == (1, 1, 2, 2)
    assert capacity.chunk_pages_lut[510:513] == (11, 11, 11)


@pytest.mark.parametrize(
    (
        "batch",
        "architecture_chunks",
        "max_request_chunks",
        "total_work_items",
    ),
    (
        (1, 94, 94, 94),
        (2, 47, 47, 94),
        (4, 23, 23, 92),
        (8, 11, 11, 88),
    ),
)
def test_decode_graph_page128_laguna_gqa6_uses_measured_chunk_budget(
    monkeypatch: pytest.MonkeyPatch,
    batch: int,
    architecture_chunks: int,
    max_request_chunks: int,
    total_work_items: int,
) -> None:
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(multi_processor_count=188),
    )
    monkeypatch.setattr(
        torch.cuda,
        "get_device_capability",
        lambda _device: (12, 0),
    )

    capacity = plan_decode_graph_capacity(
        device=torch.device("cuda:0"),
        q_dtype=torch.bfloat16,
        kv_dtype=torch.float8_e4m3fn,
        num_q_heads=24,
        num_kv_heads=4,
        head_dim_qk=128,
        head_dim_vo=128,
        page_size=128,
        batch=batch,
        max_cache_page_count=2049,
    )

    assert capacity.architecture_max_chunks_per_request == architecture_chunks
    assert capacity.max_chunks_per_request == max_request_chunks
    assert capacity.max_work_items == total_work_items
    assert capacity.max_partial_rows == total_work_items


@pytest.mark.parametrize(
    ("graph_ctas_per_sm", "max_request_chunks", "total_work_items"),
    ((1, 24, 47), (3, 71, 141)),
)
def test_decode_graph_page128_laguna_gqa6_preserves_uneven_work_budget(
    monkeypatch: pytest.MonkeyPatch,
    graph_ctas_per_sm: int,
    max_request_chunks: int,
    total_work_items: int,
) -> None:
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(
            major=12,
            minor=0,
            multi_processor_count=188,
            name="Synthetic SM120",
        ),
    )
    monkeypatch.setattr(
        torch.cuda,
        "get_device_capability",
        lambda _device: (12, 0),
    )

    capacity = plan_decode_graph_capacity(
        device=torch.device("cuda:0"),
        q_dtype=torch.bfloat16,
        kv_dtype=torch.float8_e4m3fn,
        num_q_heads=24,
        num_kv_heads=4,
        head_dim_qk=128,
        head_dim_vo=128,
        page_size=128,
        batch=2,
        max_cache_page_count=128,
        graph_ctas_per_sm=graph_ctas_per_sm,
    )

    assert capacity.architecture_max_chunks_per_request == max_request_chunks
    assert capacity.max_chunks_per_request == max_request_chunks
    assert capacity.max_work_items == total_work_items
    assert capacity.max_partial_rows == total_work_items


def test_decode_graph_ctas_per_sm_uses_smaller_minimax_bs1_to_bs4_budget() -> None:
    for kv_dtype in (torch.bfloat16, torch.float8_e4m3fn):
        assert (
            resolve_decode_graph_ctas_per_sm(
                kv_dtype=kv_dtype,
                batch=1,
                page_size=64,
                head_dim_qk=128,
                head_dim_vo=128,
                gqa_group_size=6,
            )
            == 1
        )
        assert (
            resolve_decode_graph_ctas_per_sm(
                kv_dtype=kv_dtype,
                batch=5,
                page_size=64,
                head_dim_qk=128,
                head_dim_vo=128,
                gqa_group_size=6,
            )
            == 2
        )
    assert (
        resolve_decode_graph_ctas_per_sm(
            kv_dtype=torch.bfloat16,
            batch=2,
            page_size=64,
            head_dim_qk=128,
            head_dim_vo=128,
            gqa_group_size=6,
        )
        == 6
    )
    assert (
        resolve_decode_graph_ctas_per_sm(
            kv_dtype=torch.float8_e4m3fn,
            batch=2,
            page_size=64,
            head_dim_qk=128,
            head_dim_vo=128,
            gqa_group_size=6,
        )
        == 1
    )


@pytest.mark.parametrize(
    ("capability", "sm_count", "expected_chunks"),
    (
        ((12, 1), 48, {1: 12, 2: 6, 4: 3, 8: 3}),
        ((12, 0), 188, {1: 47, 2: 23, 4: 11, 8: 5}),
    ),
)
def test_sm12x_h256_decode_uses_one_wave_only_when_it_fills_most_sms(
    monkeypatch,
    capability: tuple[int, int],
    sm_count: int,
    expected_chunks: dict[int, int],
) -> None:
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(
            major=capability[0],
            minor=capability[1],
            multi_processor_count=sm_count,
            name="Synthetic GPU",
        ),
    )
    monkeypatch.setattr(
        torch.cuda,
        "get_device_capability",
        lambda _device: capability,
    )

    for kv_dtype in (torch.bfloat16, torch.float8_e4m3fn):
        for batch, chunks in expected_chunks.items():
            capacity = plan_decode_graph_capacity(
                device=torch.device("cuda:0"),
                q_dtype=torch.bfloat16,
                kv_dtype=kv_dtype,
                num_q_heads=24,
                num_kv_heads=4,
                head_dim_qk=256,
                head_dim_vo=256,
                page_size=64,
                batch=batch,
                max_cache_page_count=512,
            )
            assert capacity.max_chunks_per_request == chunks

    explicit_two_wave = plan_decode_graph_capacity(
        device=torch.device("cuda:0"),
        q_dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
        num_q_heads=24,
        num_kv_heads=4,
        head_dim_qk=256,
        head_dim_vo=256,
        page_size=64,
        batch=1,
        max_cache_page_count=512,
        graph_ctas_per_sm=2,
    )
    assert explicit_two_wave.max_chunks_per_request == sm_count // 2


def test_build_decode_chunk_pages_lut_uses_heuristic() -> None:
    lut = build_decode_chunk_pages_lut(
        q_dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
        batch=8,
        page_size=64,
        head_dim_qk=192,
        head_dim_vo=128,
        gqa_group_size=16,
        max_effective_kv_pages=16,
    )

    assert lut[:8] == (1, 1, 1, 1, 1, 1, 1, 1)
    assert lut[8:] == (2, 2, 2, 2, 2, 2, 2, 2)


def test_gqa_config_factors_chunk_lut_losslessly() -> None:
    max_pages = 2_048
    max_chunks = 24
    base_lut = tuple(
        1 if page_count <= 16 else 6 if page_count <= 256 else 24
        for page_count in range(1, max_pages + 1)
    )
    chunk_pages_lut = tuple(
        max(base, (page_count + max_chunks - 1) // max_chunks)
        for page_count, base in enumerate(base_lut, start=1)
    )
    capacity = SimpleNamespace(
        graph_ctas_per_sm=2,
        cta_tile_q=16,
        query_tiles_per_request=1,
        architecture_max_chunks_per_request=96,
        max_chunks_per_request=max_chunks,
        max_work_items=max_chunks,
        max_partial_rows=max_chunks,
        max_effective_kv_pages=max_pages,
        worst_page_count=max_pages,
        chunk_pages_lut=chunk_pages_lut,
    )

    config = GqaConfig.from_capacity(capacity)
    round_trip = GqaConfig.from_config(FrozenMapping(config.to_dict()))

    assert config.chunk_pages_lut() == chunk_pages_lut
    assert round_trip == config
    assert len(config.base_chunk_pages_runs) < 10


def test_decode_graph_page128_reuses_page64_lut_policy() -> None:
    kwargs = dict(
        q_dtype=torch.bfloat16,
        kv_dtype=torch.float8_e4m3fn,
        batch=1,
        head_dim_qk=128,
        head_dim_vo=128,
        gqa_group_size=16,
        max_effective_kv_pages=16,
    )

    assert decode_chunk_pages_for_graph(page_size=128, **kwargs) == (
        decode_chunk_pages_for_graph(page_size=64, **kwargs)
    )
    assert build_decode_chunk_pages_lut(page_size=128, **kwargs) == (
        build_decode_chunk_pages_lut(page_size=64, **kwargs)
    )


@pytest.mark.parametrize(
    ("q_seqlens", "cache_seqlens", "kv_dtype"),
    [
        ([1] * 8, [8192] * 8, torch.float8_e4m3fn),
        ([1] * 8, [32768] * 8, torch.bfloat16),
        ([6] * 8, [8192] * 8, torch.float8_e4m3fn),
        ([6] * 8, [32768] * 8, torch.bfloat16),
    ],
)
def test_paged_non_policy_chunk_selection_still_produces_valid_plans(
    q_seqlens: list[int],
    cache_seqlens: list[int],
    kv_dtype: torch.dtype,
) -> None:
    q, k_cache, v_cache, page_table, cache_seqlens_t, cu_seqlens_q = _make_inputs(
        q_seqlens=q_seqlens,
        cache_seqlens=cache_seqlens,
        kv_dtype=kv_dtype,
    )
    plan = create_paged_plan(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens_t,
        cu_seqlens_q,
    )

    assert plan.kv_chunk_size > 0
    assert plan.new_batch_size >= page_table.shape[0]
    assert plan.total_num_partial_rows == 0
    assert plan.split_kv is False
