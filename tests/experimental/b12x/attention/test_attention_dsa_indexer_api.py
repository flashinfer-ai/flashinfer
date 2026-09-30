from __future__ import annotations

import inspect
from dataclasses import replace

import pytest
import torch

from b12x._lib.runtime_control import kernel_resolution_guard
from b12x.attention import dsa_indexer
from b12x.attention.dsa_indexer import tiled_topk as tiled_topk_module
from b12x.attention.dsa_indexer.kernel import (
    PAGED_MQA_LOGITS_SCHEDULE_PAGES_PER_SPLIT,
    _split_index_k_cache_runtime_views,
    run_paged_tiled_logits_kernel,
    run_paged_supertile_logits_kernel,
)
from b12x.attention.dsa_indexer.contiguous_kernel import (
    build_indexer_contiguous_logits_kernel_binding,
)
from b12x.attention.dsa_indexer.tiled_topk import run_row_topk
from b12x.attention.dsa_indexer.reference import (
    contiguous_logits_reference,
    pack_index_k_cache_reference,
    paged_decode_logits_reference,
)
from b12x.attention.dsa_indexer._impl import (
    IndexerContiguousMetadata,
    build_paged_mqa_schedule_metadata,
    clear_indexer_caches,
    contiguous_logits,
    contiguous_tiled_topk,
    paged_decode_logits,
    uses_paged_mqa_schedule,
)
from b12x.attention.dsa_indexer.contiguous_kernel import (
    resolve_contiguous_prefill_block_k,
)
from b12x.attention.dsa_indexer.scratch import (
    B12XIndexerContiguousScratchCaps,
    B12XIndexerPagedScratchCaps,
    INDEXER_PAGED_ROUTE_TILED,
    plan_indexer_contiguous_scratch,
    plan_indexer_paged_scratch,
    resolve_paged_prefill_k_rows,
)
from b12x._lib.compiler import clear_compile_cache, compile_cache_info


_FP8_E4M3_MAX = float(torch.finfo(torch.float8_e4m3fn).max)


def _public_dsa_inputs(rows=2, heads=4, topk=2):
    return dict(
        q_fp8=torch.empty((rows, heads, 128), dtype=torch.float8_e4m3fn),
        query_weights=torch.empty((rows, heads), dtype=torch.bfloat16),
        index_k_cache=torch.empty((8, 64 * (128 + 4)), dtype=torch.uint8),
        page_table=torch.zeros((rows, 4), dtype=torch.int32),
        cache_lengths=torch.full((rows,), 64, dtype=torch.int32),
        active_width=torch.full((1,), 256, dtype=torch.int32),
        output_indices=torch.empty((rows, topk), dtype=torch.int32),
        output_scores=torch.empty((rows, topk), dtype=torch.float32),
    )


def test_public_plan_is_declarative_and_frozen_bind_rejects_it() -> None:
    caps = dsa_indexer.Caps(
        device="cpu", num_q_heads=4, max_q_rows=2, max_page_table_width=4, topk=2
    )
    inputs = _public_dsa_inputs()
    declaration = dsa_indexer.plan(
        caps, invocation=dsa_indexer.invocation_from_tensors(caps, **inputs)
    )
    from b12x.preparation import Plan

    assert isinstance(declaration, Plan)
    assert declaration.prepared is None
    assert declaration.selection is None
    with kernel_resolution_guard("frozen"), pytest.raises(RuntimeError, match="not prepared"):
        dsa_indexer.bind(declaration, scratch=torch.empty(1, dtype=torch.uint8), **inputs)
    assert tuple(inspect.signature(dsa_indexer.run).parameters) == ("binding",)
    assert "index_topk_fp8" not in dsa_indexer.__all__
    assert "source_layout" not in inspect.signature(dsa_indexer.Caps).parameters
    assert "resolve_paged_prefill_k_rows" not in dsa_indexer.__all__


def test_public_selection_metadata_ignores_pool_size_but_preserves_layout() -> None:
    caps = dsa_indexer.Caps(
        device="cpu", num_q_heads=4, max_q_rows=2, max_page_table_width=4, topk=2
    )
    inputs = _public_dsa_inputs()
    base = dsa_indexer.invocation_from_tensors(caps, **inputs)
    width = inputs["index_k_cache"].shape[1]
    inputs["index_k_cache"] = torch.empty((32, width), dtype=torch.uint8)
    grown = dsa_indexer.invocation_from_tensors(caps, **inputs)
    assert grown == base
    inputs["index_k_cache"] = torch.empty_strided(
        (32, width), (2 * width, 1), dtype=torch.uint8
    )
    interleaved = dsa_indexer.invocation_from_tensors(caps, **inputs)
    assert interleaved != base
    assert interleaved["operands"]["index_k_cache"]["strides"] == (2 * width, 1)


@pytest.mark.parametrize(
    ("output_index_space", "output_physical_slots"),
    [("logical", False), ("physical", True)],
)
def test_public_output_index_space_is_frozen_in_declaration(
    output_index_space: str, output_physical_slots: bool
) -> None:
    caps = dsa_indexer.Caps(
        device="cpu", num_q_heads=4, max_q_rows=2, max_page_table_width=4,
        topk=2, output_index_space=output_index_space,
    )
    declaration = dsa_indexer.plan(
        caps, invocation=dsa_indexer.invocation_from_tensors(caps, **_public_dsa_inputs())
    )
    assert declaration.query.output_physical_slots is output_physical_slots


def test_public_output_index_space_rejects_unknown_semantics() -> None:
    with pytest.raises(ValueError, match="output_index_space"):
        dsa_indexer.Caps(
            device="cpu", num_q_heads=4, max_q_rows=2, max_page_table_width=4,
            topk=2, output_index_space="request_relative",
        )


def test_topk_candidate_capacity_is_compile_time_topk_policy() -> None:
    assert tiled_topk_module._resolve_smem_candidate_capacity(topk=512) == 1024
    assert tiled_topk_module._resolve_smem_candidate_capacity(topk=1024) == 8192
    assert tiled_topk_module._resolve_smem_candidate_capacity(topk=2048) == 8192


def _make_real_page_table(
    *,
    page_starts: list[int],
    seqlens: list[int],
    width_blocks: int,
    device: torch.device,
) -> torch.Tensor:
    real_page_table = torch.full(
        (len(seqlens), width_blocks),
        -1,
        dtype=torch.int32,
        device=device,
    )
    for row_idx, (page_start, seq_len) in enumerate(
        zip(page_starts, seqlens, strict=True)
    ):
        block_count = (int(seq_len) + 63) // 64
        if block_count:
            real_page_table[row_idx, :block_count] = torch.arange(
                page_start,
                page_start + block_count,
                dtype=torch.int32,
                device=device,
            )
    return real_page_table


def _quantize_rows_to_kv_fp8(k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = k.abs().amax(dim=1) / _FP8_E4M3_MAX
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    quant = (
        (k / scale.unsqueeze(1))
        .clamp(-_FP8_E4M3_MAX, _FP8_E4M3_MAX)
        .to(torch.float8_e4m3fn)
    )
    return quant, scale.to(torch.float32)


def _assert_logits_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


def _one_scratch(plan):
    (spec,) = plan.scratch_specs()
    return torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)


def _bind_paged_decode(
    *,
    real_page_table: torch.Tensor,
    cache_seqlens_int32: torch.Tensor,
    num_q_heads: int,
    schedule_metadata: torch.Tensor | None = None,
    active_width: torch.Tensor | None = None,
    topk: int = 1,
):
    plan = plan_indexer_paged_scratch(
        B12XIndexerPagedScratchCaps(
            device=real_page_table.device,
            num_q_heads=int(num_q_heads),
            max_q_rows=int(real_page_table.shape[0]),
            max_page_table_width=int(real_page_table.shape[1]),
            topk=int(topk),
            reserve_paged_logits=False,
            route=INDEXER_PAGED_ROUTE_TILED,
        )
    )
    return plan.bind(
        scratch=_one_scratch(plan),
        real_page_table=real_page_table,
        cache_seqlens_int32=cache_seqlens_int32,
        active_width=active_width,
        schedule_metadata=schedule_metadata,
        expected_num_q_heads=int(num_q_heads),
    )


def _bind_contiguous_topk(
    *,
    kv_fp8: tuple[torch.Tensor, torch.Tensor],
    k_start: torch.Tensor,
    k_end: torch.Tensor,
    num_q_heads: int,
    topk: int,
    q_rows: int | None = None,
    supertile_k: int = 32768,
    strict: bool = True,
):
    k_quant, k_scale = kv_fp8
    max_q_rows = int(q_rows) if q_rows is not None else int(k_start.shape[0])
    plan = plan_indexer_contiguous_scratch(
        B12XIndexerContiguousScratchCaps(
            device=k_quant.device,
            num_q_heads=int(num_q_heads),
            max_q_rows=max_q_rows,
            max_k_rows=int(k_quant.shape[0]),
            topk=int(topk),
            supertile_k=int(supertile_k),
        )
    )
    binding = plan.bind(
        scratch=_one_scratch(plan),
        k_start=k_start,
        k_end=k_end,
        gather_rows=int(k_quant.shape[0]),
        topk=int(topk),
    )
    scratch = binding.scratch
    scratch.k_quant[: int(k_quant.shape[0])].copy_(k_quant)
    scratch.k_scale[: int(k_scale.shape[0])].copy_(k_scale)
    if not strict:
        binding = replace(binding, strict=False)
    return binding, (
        scratch.k_quant[: int(k_quant.shape[0])],
        scratch.k_scale[: int(k_scale.shape[0])],
    )


def _bind_staged_contiguous_logits(
    *,
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    kv_fp8: tuple[torch.Tensor, torch.Tensor],
    k_start: torch.Tensor,
    k_end: torch.Tensor,
    prefill_block_k: int | None,
):
    k_quant, k_scale = kv_fp8
    q_rows = int(k_start.shape[0])
    k_rows = int(k_quant.shape[0])
    plan = plan_indexer_contiguous_scratch(
        B12XIndexerContiguousScratchCaps(
            device=q_fp8.device,
            num_q_heads=int(q_fp8.shape[1]),
            max_q_rows=q_rows,
            max_k_rows=k_rows,
            topk=1,
            supertile_k=max(k_rows, 256),
        )
    )
    plan_binding = plan.bind(
        scratch=_one_scratch(plan),
        k_start=k_start,
        k_end=k_end,
        gather_rows=k_rows,
        topk=1,
    )
    scratch = plan_binding.scratch
    scratch.k_quant[:k_rows].copy_(k_quant)
    scratch.k_scale[:k_rows].copy_(k_scale)
    scratch.prepare_k_padding(k_rows=k_rows)
    scratch_k_quant = scratch.k_quant[:k_rows]
    scratch_k_scale = scratch.k_scale[:k_rows]
    q_bytes = q_fp8.view(torch.uint8)
    q_u32 = q_bytes.view(torch.uint32).view(
        int(q_fp8.shape[0]),
        int(q_fp8.shape[1]),
        128 // 4,
    )
    out = torch.empty((q_rows, k_rows), dtype=torch.float32, device=q_fp8.device)
    kernel_binding = build_indexer_contiguous_logits_kernel_binding(
        q_fp8=q_fp8,
        weights=weights,
        k_quant=scratch_k_quant,
        k_scale=scratch_k_scale,
        k_start=k_start,
        k_end=k_end,
        preinitialize_invalid_logits=False,
        prefill_block_k=prefill_block_k,
        q_u32=q_u32,
        q_bytes=q_bytes,
        weights_kernel=weights,
        k_quant_bytes=scratch.k_quant.view(torch.uint8),
        k_scale_kernel=scratch.k_scale,
        k_start_kernel=k_start,
        k_end_kernel=k_end,
        out_kernel=out,
        out_view=out,
        k_tma_desc_ptrs=scratch.k_tma_desc_ptrs,
        k_tma_prefill_desc_ptrs=scratch.k_tma_prefill_desc_ptrs,
    )
    return kernel_binding, out, (scratch_k_quant, scratch_k_scale)


def _paged_mqa_schedule_reference(
    context_lens: torch.Tensor,
    *,
    block_kv: int,
    num_sms: int,
) -> torch.Tensor:
    rows = (
        context_lens[:, -1].tolist()
        if context_lens.ndim == 2
        else context_lens.tolist()
    )
    split_kv = block_kv * PAGED_MQA_LOGITS_SCHEDULE_PAGES_PER_SPLIT
    prefix_sum: list[int] = []
    total = 0
    for row in rows:
        total += max((int(row) + split_kv - 1) // split_kv, 0)
        prefix_sum.append(total)

    q, r = divmod(total, num_sms)
    out: list[list[int]] = []
    for sm_idx in range(num_sms + 1):
        seg_start = sm_idx * q + min(sm_idx, r)
        q_idx = 0
        while q_idx < len(prefix_sum) and prefix_sum[q_idx] <= seg_start:
            q_idx += 1
        kv_split_idx = seg_start if q_idx == 0 else seg_start - prefix_sum[q_idx - 1]
        out.append([q_idx, kv_split_idx])
    return torch.tensor(out, dtype=torch.int32)


def test_dsa_index_runtime_views_preserve_page_stride() -> None:
    device = torch.device("cpu")
    page_count = 3
    page_bytes = 64 * (128 + 4)
    index_k_cache = torch.arange(
        page_count * page_bytes, dtype=torch.uint8, device=device
    ).view(
        page_count,
        page_bytes,
    )

    quant, scales = _split_index_k_cache_runtime_views(index_k_cache)

    assert quant.shape == (page_count, 64, 128)
    assert quant.stride() == (page_bytes, 128, 1)
    assert (
        quant.untyped_storage().data_ptr() == index_k_cache.untyped_storage().data_ptr()
    )
    assert quant[1, 2, 127].item() == index_k_cache[1, 2 * 128 + 127].item()

    data_bytes = 64 * 128
    assert scales.shape == (page_count, 64)
    assert scales.stride() == (page_bytes // 4, 1)
    assert (
        scales.untyped_storage().data_ptr()
        == index_k_cache.untyped_storage().data_ptr()
    )
    scale_bytes = scales.view(torch.uint8).view(page_count, 64, 4)
    assert scale_bytes[1, 2, 0].item() == index_k_cache[1, data_bytes + 2 * 4].item()


def test_contiguous_prefill512_policy_allows_padded_k_rows() -> None:
    assert (
        resolve_contiguous_prefill_block_k(
            valid_q_rows=1536,
            k_rows=5001,
            num_heads=32,
        )
        == 512
    )


def test_paged_prefill_k_rows_match_plan_capacity_and_override(monkeypatch) -> None:
    assert resolve_paged_prefill_k_rows(max_page_table_width=128) == 8192
    assert resolve_paged_prefill_k_rows(max_page_table_width=1024) == 32768

    monkeypatch.setenv("B12X_PAGED_INDEX_SUPERTILE_K", "16384")
    assert resolve_paged_prefill_k_rows(max_page_table_width=128) == 16384


def test_paged_dsa_glm_front_door_does_not_expose_paged_window_contract() -> None:
    glm_params = inspect.signature(run_paged_tiled_logits_kernel).parameters
    paged_params = inspect.signature(run_paged_supertile_logits_kernel).parameters

    assert "source_page_offset" not in glm_params
    assert "output_width_tokens" not in glm_params
    assert "source_page_offset" in paged_params
    assert "output_width_tokens" in paged_params


def test_build_paged_mqa_schedule_metadata_matches_deepgemm_partitioning() -> None:
    context_lens_1d = torch.tensor([0, 64, 4096, 4097, 16384], dtype=torch.int32)
    schedule_1d = build_paged_mqa_schedule_metadata(context_lens_1d, 64, 5)
    expected_1d = _paged_mqa_schedule_reference(context_lens_1d, block_kv=64, num_sms=5)
    assert schedule_1d.shape == (6, 2)
    assert schedule_1d.dtype == torch.int32
    assert schedule_1d.is_contiguous()
    assert torch.equal(schedule_1d.cpu(), expected_1d)

    context_lens_2d = torch.tensor([[64, 65], [0, 8192], [128, 129]], dtype=torch.int32)
    schedule_2d = build_paged_mqa_schedule_metadata(context_lens_2d, 64, 7)
    expected_2d = _paged_mqa_schedule_reference(context_lens_2d, block_kv=64, num_sms=7)
    assert schedule_2d.shape == (8, 2)
    assert schedule_2d.dtype == torch.int32
    assert schedule_2d.is_contiguous()
    assert torch.equal(schedule_2d.cpu(), expected_2d)


def test_uses_paged_mqa_schedule_only_for_long_rows() -> None:
    assert not uses_paged_mqa_schedule(q_rows=0, max_pages=2048)
    assert not uses_paged_mqa_schedule(q_rows=1, max_pages=128)
    assert not uses_paged_mqa_schedule(q_rows=2, max_pages=512)
    assert not uses_paged_mqa_schedule(q_rows=9, max_pages=2048)
    assert uses_paged_mqa_schedule(q_rows=1, max_pages=2048)
    assert uses_paged_mqa_schedule(q_rows=2, max_pages=2048)
    assert uses_paged_mqa_schedule(q_rows=8, max_pages=2048)


def test_dsa_contiguous_prefill_block_k_auto_targets_long_bs1_prefill(
    monkeypatch,
) -> None:
    monkeypatch.delenv("B12X_DSA_CONTIGUOUS_PREFILL_THRESHOLD", raising=False)
    monkeypatch.delenv("B12X_DSA_CONTIGUOUS_PREFILL_BLOCK_K", raising=False)

    assert (
        resolve_contiguous_prefill_block_k(
            valid_q_rows=2048,
            k_rows=65536,
            num_heads=32,
        )
        == 512
    )
    assert (
        resolve_contiguous_prefill_block_k(
            valid_q_rows=512,
            k_rows=65536,
            num_heads=64,
        )
        == 256
    )
    assert (
        resolve_contiguous_prefill_block_k(
            valid_q_rows=128,
            k_rows=65536,
            num_heads=64,
        )
        is None
    )


def test_dsa_contiguous_prefill_block_k_env_overrides(monkeypatch) -> None:
    monkeypatch.delenv("B12X_DSA_CONTIGUOUS_PREFILL_THRESHOLD", raising=False)
    monkeypatch.setenv("B12X_DSA_CONTIGUOUS_PREFILL_BLOCK_K", "256")
    assert (
        resolve_contiguous_prefill_block_k(
            valid_q_rows=2048,
            k_rows=65536,
            num_heads=32,
        )
        == 256
    )

    monkeypatch.setenv("B12X_DSA_CONTIGUOUS_PREFILL_BLOCK_K", "512")
    assert (
        resolve_contiguous_prefill_block_k(
            valid_q_rows=2048,
            k_rows=65536,
            num_heads=32,
        )
        == 512
    )
    with pytest.raises(ValueError, match="unsupported"):
        resolve_contiguous_prefill_block_k(
            valid_q_rows=512,
            k_rows=65536,
            num_heads=64,
        )

    monkeypatch.setenv("B12X_DSA_CONTIGUOUS_PREFILL_BLOCK_K", "bad")
    with pytest.raises(ValueError, match="auto, 256, or 512"):
        resolve_contiguous_prefill_block_k(
            valid_q_rows=2048,
            k_rows=65536,
            num_heads=64,
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for paged kernel coverage"
)
def test_paged_decode_logits_cuda_kernel_matches_reference() -> None:
    device = torch.device("cuda")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(72_101)

    q_rows = 4
    num_heads = 8
    page_starts = [2, 8, 12, 16]
    num_tokens = (max(page_starts) + 3) * 64
    seqlens = torch.tensor([65, 96, 128, 191], dtype=torch.int32, device=device)
    real_page_table = _make_real_page_table(
        page_starts=page_starts,
        seqlens=seqlens.tolist(),
        width_blocks=3,
        device=device,
    )
    q_fp8 = (
        torch.randn((q_rows, num_heads, 128), generator=gen, dtype=torch.float32).to(
            device=device
        )
        / 2
    ).to(torch.float8_e4m3fn)
    weights = torch.randn((q_rows, num_heads), generator=gen, dtype=torch.float32).to(
        device=device
    )
    index_k_cache = pack_index_k_cache_reference(
        torch.randn((num_tokens, 128), generator=gen, dtype=torch.float32).to(
            device=device
        )
        / 3
    )

    binding = _bind_paged_decode(
        real_page_table=real_page_table,
        cache_seqlens_int32=seqlens,
        num_q_heads=num_heads,
        schedule_metadata=build_paged_mqa_schedule_metadata(seqlens, 64, 8),
    )

    actual = paged_decode_logits(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        binding=binding,
    )
    expected = paged_decode_logits_reference(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        real_page_table=real_page_table,
        query_row_to_batch=torch.arange(q_rows, dtype=torch.int32, device=device),
        seqlens_per_query=seqlens,
    )

    torch.cuda.synchronize(device)
    _assert_logits_close(actual, expected)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for paged kernel coverage"
)
def test_paged_decode_logits_cuda_schedule_kernel_matches_reference() -> None:
    device = torch.device("cuda")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(72_111)

    q_rows = 2
    num_heads = 8
    width_blocks = 1024
    page_starts = [2, 1100]
    num_tokens = (max(page_starts) + 40) * 64
    seqlens = torch.tensor([2048, 2304], dtype=torch.int32, device=device)
    real_page_table = _make_real_page_table(
        page_starts=page_starts,
        seqlens=seqlens.tolist(),
        width_blocks=width_blocks,
        device=device,
    )
    q_fp8 = (
        torch.randn((q_rows, num_heads, 128), generator=gen, dtype=torch.float32).to(
            device=device
        )
        / 2
    ).to(torch.float8_e4m3fn)
    weights = torch.randn((q_rows, num_heads), generator=gen, dtype=torch.float32).to(
        device=device
    )
    index_k_cache = pack_index_k_cache_reference(
        torch.randn((num_tokens, 128), generator=gen, dtype=torch.float32).to(
            device=device
        )
        / 3
    )

    binding = _bind_paged_decode(
        real_page_table=real_page_table,
        cache_seqlens_int32=seqlens,
        num_q_heads=num_heads,
        schedule_metadata=build_paged_mqa_schedule_metadata(seqlens, 64, 8),
    )

    actual = paged_decode_logits(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        binding=binding,
    )
    expected = paged_decode_logits_reference(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        real_page_table=real_page_table,
        query_row_to_batch=torch.arange(q_rows, dtype=torch.int32, device=device),
        seqlens_per_query=seqlens,
    )

    torch.cuda.synchronize(device)
    _assert_logits_close(actual, expected)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for graph capture coverage"
)
def test_paged_decode_logits_cuda_graph_replay_tracks_live_width_without_stale_output() -> (
    None
):
    device = torch.device("cuda")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(72_102)

    rows = 2
    num_heads = 8
    num_tokens = 1024
    graph_width_blocks = 4
    live_width_blocks = 3

    q_fp8 = (
        torch.randn((rows, num_heads, 128), generator=gen, dtype=torch.float32).to(
            device=device
        )
        / 2
    ).to(torch.float8_e4m3fn)
    weights = torch.randn((rows, num_heads), generator=gen, dtype=torch.float32).to(
        device=device
    )
    index_k_cache = pack_index_k_cache_reference(
        torch.randn((num_tokens, 128), generator=gen, dtype=torch.float32).to(
            device=device
        )
        / 3
    )
    live_real_page_table0 = _make_real_page_table(
        page_starts=[2, 8],
        seqlens=[150, 129],
        width_blocks=live_width_blocks,
        device=device,
    )
    live_real_page_table1 = _make_real_page_table(
        page_starts=[4, 9],
        seqlens=[65, 64],
        width_blocks=live_width_blocks,
        device=device,
    )
    graph_real_page_table = torch.full(
        (rows, graph_width_blocks),
        -1,
        dtype=torch.int32,
        device=device,
    )
    graph_seqlens = torch.empty((rows,), dtype=torch.int32, device=device)
    graph_schedule_metadata = torch.empty((9, 2), dtype=torch.int32, device=device)
    graph_active_width = torch.empty((1,), dtype=torch.int32, device=device)

    def prepare(page_table: torch.Tensor, seqlens: torch.Tensor) -> None:
        graph_real_page_table[:, :live_width_blocks].copy_(page_table)
        graph_seqlens.copy_(seqlens)
        graph_active_width.copy_(
            torch.clamp(
                graph_seqlens.amax().reshape(1), min=0, max=graph_width_blocks * 64
            )
        )
        build_paged_mqa_schedule_metadata(
            graph_seqlens, 64, 8, out=graph_schedule_metadata
        )

    binding = _bind_paged_decode(
        real_page_table=graph_real_page_table,
        cache_seqlens_int32=graph_seqlens,
        num_q_heads=num_heads,
        schedule_metadata=graph_schedule_metadata,
        active_width=graph_active_width,
    )

    clear_indexer_caches()
    prepare(
        live_real_page_table0,
        torch.tensor([150, 129], dtype=torch.int32, device=device),
    )
    paged_decode_logits(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        binding=binding,
    )
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_out = paged_decode_logits(
            q_fp8=q_fp8,
            weights=weights,
            index_k_cache=index_k_cache,
            binding=binding,
        )
    graph.replay()
    torch.cuda.synchronize(device)
    actual0 = captured_out.clone()
    expected0 = paged_decode_logits_reference(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        real_page_table=graph_real_page_table,
        query_row_to_batch=torch.arange(rows, dtype=torch.int32, device=device),
        seqlens_per_query=graph_seqlens,
    )
    _assert_logits_close(actual0, expected0)

    prepare(
        live_real_page_table1, torch.tensor([65, 64], dtype=torch.int32, device=device)
    )
    graph.replay()
    torch.cuda.synchronize(device)
    actual1 = captured_out.clone()
    expected1 = paged_decode_logits_reference(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        real_page_table=graph_real_page_table,
        query_row_to_batch=torch.arange(rows, dtype=torch.int32, device=device),
        seqlens_per_query=graph_seqlens,
    )
    _assert_logits_close(actual1, expected1)
    assert torch.isneginf(actual1[:, 65:]).all()


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for contiguous kernel coverage"
)
def test_contiguous_logits_matches_reference() -> None:
    device = torch.device("cuda")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(72_103)

    q_rows = 5
    num_heads = 3
    k_rows = 64
    q_fp8 = (
        torch.randn(
            (q_rows + 1, num_heads, 128), generator=gen, dtype=torch.float32
        ).to(device=device)
        / 2
    ).to(torch.float8_e4m3fn)
    weights = torch.randn(
        (q_rows + 1, num_heads), generator=gen, dtype=torch.float32
    ).to(device=device)
    k = (
        torch.randn((k_rows, 128), generator=gen, dtype=torch.float32).to(device=device)
        / 3
    )
    kv_fp8 = _quantize_rows_to_kv_fp8(k)
    k_start = torch.tensor([0, 5, 12, 12, 40], dtype=torch.int32, device=device)
    k_end = torch.tensor([8, 16, 20, 12, 55], dtype=torch.int32, device=device)

    actual = contiguous_logits(
        q_fp8=q_fp8,
        weights=weights,
        kv_fp8=kv_fp8,
        metadata=IndexerContiguousMetadata(
            k_start=k_start,
            k_end=k_end,
        ),
    )
    expected = contiguous_logits_reference(
        q_fp8=q_fp8,
        weights=weights,
        kv_fp8=kv_fp8,
        k_start=k_start,
        k_end=k_end,
    )

    _assert_logits_close(actual, expected)
    assert torch.isneginf(actual[-1]).all()


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for contiguous kernel coverage"
)
def test_contiguous_logits_matches_reference_for_sparse_tile_ranges() -> None:
    device = torch.device("cuda")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(72_104)

    q_rows = 40
    num_heads = 4
    k_rows = 130
    q_fp8 = (
        torch.randn((q_rows, num_heads, 128), generator=gen, dtype=torch.float32).to(
            device=device
        )
        / 2
    ).to(torch.float8_e4m3fn)
    weights = torch.randn((q_rows, num_heads), generator=gen, dtype=torch.float32).to(
        device=device
    )
    k = (
        torch.randn((k_rows, 128), generator=gen, dtype=torch.float32).to(device=device)
        / 3
    )
    kv_fp8 = _quantize_rows_to_kv_fp8(k)
    k_start = torch.tensor(([0] * 32) + ([128] * 8), dtype=torch.int32, device=device)
    k_end = torch.tensor(([32] * 32) + ([130] * 8), dtype=torch.int32, device=device)

    actual = contiguous_logits(
        q_fp8=q_fp8,
        weights=weights,
        kv_fp8=kv_fp8,
        metadata=IndexerContiguousMetadata(
            k_start=k_start,
            k_end=k_end,
        ),
    )
    expected = contiguous_logits_reference(
        q_fp8=q_fp8,
        weights=weights,
        kv_fp8=kv_fp8,
        k_start=k_start,
        k_end=k_end,
    )

    _assert_logits_close(actual, expected)
    assert torch.isneginf(actual[:32, 32:]).all()
    assert torch.isneginf(actual[32:, :128]).all()


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for graph capture coverage"
)
@pytest.mark.parametrize(
    "q_rows,num_heads,k_rows,prefill_block_k",
    [
        pytest.param(5, 3, 64, None, id="decode"),
        pytest.param(256, 8, 512, 256, id="prefill256"),
        pytest.param(1024, 32, 4096, 512, id="prefill512-h32"),
    ],
)
def test_contiguous_logits_staged_binding_graph_replay_tracks_live_weights(
    monkeypatch,
    q_rows: int,
    num_heads: int,
    k_rows: int,
    prefill_block_k: int | None,
) -> None:
    device = torch.device("cuda")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(73_000 + q_rows * 31 + num_heads * 17 + k_rows)

    q_fp8 = (
        torch.randn((q_rows, num_heads, 128), generator=gen, dtype=torch.float32).to(
            device=device
        )
        / 2
    ).to(torch.float8_e4m3fn)
    weights = torch.randn((q_rows, num_heads), generator=gen, dtype=torch.float32).to(
        device=device
    )
    live_weights = torch.randn(
        (q_rows, num_heads), generator=gen, dtype=torch.float32
    ).to(device=device)
    k = (
        torch.randn((k_rows, 128), generator=gen, dtype=torch.float32).to(device=device)
        / 3
    )
    kv_fp8 = _quantize_rows_to_kv_fp8(k)
    k_start = torch.zeros(q_rows, dtype=torch.int32, device=device)
    k_end = torch.full((q_rows,), k_rows, dtype=torch.int32, device=device)
    binding, out, bound_kv_fp8 = _bind_staged_contiguous_logits(
        q_fp8=q_fp8,
        weights=weights,
        kv_fp8=kv_fp8,
        k_start=k_start,
        k_end=k_end,
        prefill_block_k=prefill_block_k,
    )
    k_quant, k_scale = bound_kv_fp8
    sampled_q = torch.tensor(
        sorted({0, q_rows // 2, q_rows - 1}), dtype=torch.long, device=device
    )
    sampled_k = torch.tensor(
        sorted({0, 1, k_rows // 2, k_rows - 1}), dtype=torch.long, device=device
    )

    def sampled_reference() -> torch.Tensor:
        scores = torch.einsum(
            "qhd,kd->qhk",
            q_fp8[sampled_q].to(torch.float32),
            k_quant[sampled_k].to(torch.float32),
        )
        return (torch.relu(scores) * weights[sampled_q, :, None]).sum(dim=1) * k_scale[
            sampled_k
        ][None, :]

    clear_indexer_caches()
    binding.run()
    torch.cuda.synchronize(device)
    warm_compile_misses = compile_cache_info()["compile_misses"]

    # A staged replay must not manufacture the historical 1x1/1x1x1 CUDA
    # placeholder tensors. This guard makes the regression observable even
    # though CUDA graph pools can otherwise hide a capture-time allocation.
    real_empty = torch.empty

    def reject_cuda_dummy_empty(*args, **kwargs):
        requested_shape = (
            tuple(args[0]) if args and isinstance(args[0], tuple) else None
        )
        requested_device = torch.device(kwargs.get("device", "cpu"))
        if requested_device.type == "cuda" and requested_shape in ((1, 1), (1, 1, 1)):
            raise AssertionError(
                f"staged contiguous replay allocated CUDA dummy {requested_shape}"
            )
        return real_empty(*args, **kwargs)

    with kernel_resolution_guard('staged contiguous graph replay must use warmed kernels'):
        with monkeypatch.context() as patch:
            patch.setattr(torch, "empty", reject_cuda_dummy_empty)
            guarded_out = binding.run()
        torch.cuda.synchronize(device)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured_out = binding.run()

    assert guarded_out.data_ptr() == out.data_ptr()
    assert captured_out.data_ptr() == out.data_ptr()
    input_ptrs = (
        q_fp8.data_ptr(),
        weights.data_ptr(),
        k_quant.data_ptr(),
        k_scale.data_ptr(),
        k_start.data_ptr(),
        k_end.data_ptr(),
        out.data_ptr(),
    )

    graph.replay()
    torch.cuda.synchronize(device)
    actual0 = out[sampled_q[:, None], sampled_k[None, :]].clone()
    expected0 = sampled_reference()
    _assert_logits_close(actual0, expected0)

    weights.copy_(live_weights)
    graph.replay()
    torch.cuda.synchronize(device)
    actual1 = out[sampled_q[:, None], sampled_k[None, :]].clone()
    expected1 = sampled_reference()
    _assert_logits_close(actual1, expected1)
    assert not torch.equal(actual0, actual1)
    assert compile_cache_info()["compile_misses"] == warm_compile_misses
    assert input_ptrs == (
        q_fp8.data_ptr(),
        weights.data_ptr(),
        k_quant.data_ptr(),
        k_scale.data_ptr(),
        k_start.data_ptr(),
        k_end.data_ptr(),
        out.data_ptr(),
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for contiguous kernel coverage"
)
@pytest.mark.parametrize("num_heads", [16, 32, 64])
def test_contiguous_logits_cuda_matches_reference_for_large_head_counts(
    num_heads: int,
) -> None:
    device = torch.device("cuda")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(72_105 + num_heads)

    q_rows = 8
    k_rows = 257
    q_fp8 = (
        torch.randn((q_rows, num_heads, 128), generator=gen, dtype=torch.float32).to(
            device=device
        )
        / 2
    ).to(torch.float8_e4m3fn)
    weights = torch.randn((q_rows, num_heads), generator=gen, dtype=torch.float32).to(
        device=device
    )
    k = (
        torch.randn((k_rows, 128), generator=gen, dtype=torch.float32).to(device=device)
        / 3
    )
    kv_fp8 = _quantize_rows_to_kv_fp8(k)
    k_start = torch.tensor(
        [0, 192, 16, 128, 32, 224, 0, 64],
        dtype=torch.int32,
        device=device,
    )
    k_end = torch.tensor(
        [33, 257, 80, 192, 96, 257, 1, 65],
        dtype=torch.int32,
        device=device,
    )

    actual = contiguous_logits(
        q_fp8=q_fp8,
        weights=weights,
        kv_fp8=kv_fp8,
        metadata=IndexerContiguousMetadata(
            k_start=k_start,
            k_end=k_end,
        ),
    )
    expected = contiguous_logits_reference(
        q_fp8=q_fp8,
        weights=weights,
        kv_fp8=kv_fp8,
        k_start=k_start,
        k_end=k_end,
    )

    torch.cuda.synchronize(device)
    _assert_logits_close(actual, expected)
    assert torch.isneginf(actual[0, 33:192]).all()
    assert torch.isneginf(actual[1, :192]).all()
    assert torch.isneginf(actual[6, 1:]).all()


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for contiguous kernel coverage"
)
@pytest.mark.parametrize(
    "q_rows, k_rows",
    [
        (256, 4096),  # shortest prefill q, >2048 k — crosses the K-tile budget.
        (512, 8192),  # multi-Q-tile prefill over long context.
        (1024, 3072),  # many Q-tiles, mid-length K.
    ],
)
def test_contiguous_logits_cuda_matches_reference_for_long_prefill(
    q_rows: int, k_rows: int
) -> None:
    device = torch.device("cuda")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(72_200 + q_rows * 17 + k_rows)

    num_heads = 64
    q_fp8 = (
        torch.randn((q_rows, num_heads, 128), generator=gen, dtype=torch.float32).to(
            device=device
        )
        / 2
    ).to(torch.float8_e4m3fn)
    weights = torch.randn((q_rows, num_heads), generator=gen, dtype=torch.float32).to(
        device=device
    )
    k = (
        torch.randn((k_rows, 128), generator=gen, dtype=torch.float32).to(device=device)
        / 3
    )
    kv_fp8 = _quantize_rows_to_kv_fp8(k)

    # Causal ragged ranges: row q sees k ∈ [0, q+1). Spans the full k_rows range
    # for tail rows and exercises per-row sparse-range liveness for early rows.
    positions = torch.arange(q_rows, dtype=torch.int32, device=device)
    k_start = torch.zeros(q_rows, dtype=torch.int32, device=device)
    k_end = torch.clamp(positions + 1, max=k_rows).to(torch.int32)

    actual = contiguous_logits(
        q_fp8=q_fp8,
        weights=weights,
        kv_fp8=kv_fp8,
        metadata=IndexerContiguousMetadata(
            k_start=k_start,
            k_end=k_end,
        ),
    )
    actual_no_fill = contiguous_logits(
        q_fp8=q_fp8,
        weights=weights,
        kv_fp8=kv_fp8,
        metadata=IndexerContiguousMetadata(
            k_start=k_start,
            k_end=k_end,
        ),
        preinitialize_invalid_logits=False,
    )
    expected = contiguous_logits_reference(
        q_fp8=q_fp8,
        weights=weights,
        kv_fp8=kv_fp8,
        k_start=k_start,
        k_end=k_end,
    )

    torch.cuda.synchronize(device)
    _assert_logits_close(actual, expected)
    _assert_logits_close(actual_no_fill, expected)
    # Out-of-range positions must stay -inf all the way out to k_rows.
    for q in (0, q_rows // 2, q_rows - 1):
        ke = min(q + 1, k_rows)
        if ke < k_rows:
            assert torch.isneginf(actual[q, ke:]).all(), (
                f"row {q} leaked non-neginf beyond k_end={ke}"
            )
            assert torch.isneginf(actual_no_fill[q, ke:]).all(), (
                f"no-fill row {q} leaked non-neginf beyond k_end={ke}"
            )


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for contiguous kernel coverage"
)
@pytest.mark.parametrize(
    "q_rows, k_rows",
    [
        (256, 3072),  # slightly over the old hardcoded 2048 K-tile budget.
        (256, 8192),  # well past it — exercises full K-grid scaling.
    ],
)
def test_contiguous_logits_cuda_matches_reference_for_dense_long_prefill(
    q_rows: int, k_rows: int
) -> None:
    """Dense (non-causal) long-K prefill: every q row sees every k row.

    Exercises the K-tile grid scaling — with a fixed K_GROUPS=4 launcher and a fixed
    inner K-tile loop of 4, the kernel can only cover 2048 K-rows per Q-tile; anything
    beyond that silently stays at -inf and this test catches it.
    """
    device = torch.device("cuda")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(72_400 + q_rows * 31 + k_rows)

    num_heads = 64
    q_fp8 = (
        torch.randn((q_rows, num_heads, 128), generator=gen, dtype=torch.float32).to(
            device=device
        )
        / 2
    ).to(torch.float8_e4m3fn)
    weights = torch.randn((q_rows, num_heads), generator=gen, dtype=torch.float32).to(
        device=device
    )
    k = (
        torch.randn((k_rows, 128), generator=gen, dtype=torch.float32).to(device=device)
        / 3
    )
    kv_fp8 = _quantize_rows_to_kv_fp8(k)
    k_start = torch.zeros(q_rows, dtype=torch.int32, device=device)
    k_end = torch.full((q_rows,), k_rows, dtype=torch.int32, device=device)

    actual = contiguous_logits(
        q_fp8=q_fp8,
        weights=weights,
        kv_fp8=kv_fp8,
        metadata=IndexerContiguousMetadata(k_start=k_start, k_end=k_end),
    )
    expected = contiguous_logits_reference(
        q_fp8=q_fp8,
        weights=weights,
        kv_fp8=kv_fp8,
        k_start=k_start,
        k_end=k_end,
    )
    torch.cuda.synchronize(device)
    _assert_logits_close(actual, expected)
    # No position should have silently fallen back to -inf.
    assert torch.isfinite(actual).all(), "kernel left finite positions as -inf"


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for tiled topk coverage"
)
def test_contiguous_tiled_topk_graph_replay_tracks_live_weights(monkeypatch) -> None:
    device = torch.device("cuda")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(73_620)

    q_rows = 32
    num_heads = 8
    k_rows = 1024
    topk = 512
    q_fp8 = (
        torch.randn((q_rows, num_heads, 128), generator=gen, dtype=torch.float32).to(
            device=device
        )
        / 2
    ).to(torch.float8_e4m3fn)
    weights = torch.randn((q_rows, num_heads), generator=gen, dtype=torch.float32).to(
        device=device
    )
    live_weights = torch.randn(
        (q_rows, num_heads), generator=gen, dtype=torch.float32
    ).to(device=device)
    k = (
        torch.randn((k_rows, 128), generator=gen, dtype=torch.float32).to(device=device)
        / 3
    )
    kv_fp8 = _quantize_rows_to_kv_fp8(k)
    k_start = torch.zeros(q_rows, dtype=torch.int32, device=device)
    k_end = torch.full((q_rows,), k_rows, dtype=torch.int32, device=device)
    binding, bound_kv_fp8 = _bind_contiguous_topk(
        kv_fp8=kv_fp8,
        k_start=k_start,
        k_end=k_end,
        num_q_heads=num_heads,
        topk=topk,
        q_rows=q_rows,
        supertile_k=k_rows,
    )

    def expected_topk_values() -> torch.Tensor:
        logits = contiguous_logits_reference(
            q_fp8=q_fp8,
            weights=weights,
            kv_fp8=bound_kv_fp8,
            k_start=k_start,
            k_end=k_end,
        )
        return torch.topk(logits, k=topk, dim=1, largest=True, sorted=False).values

    clear_indexer_caches()
    contiguous_tiled_topk(
        q_fp8=q_fp8,
        weights=weights,
        kv_fp8=bound_kv_fp8,
        binding=binding,
    )
    torch.cuda.synchronize(device)
    warm_compile_misses = compile_cache_info()["compile_misses"]

    real_empty = torch.empty

    def reject_cuda_dummy_empty(*args, **kwargs):
        requested_shape = (
            tuple(args[0]) if args and isinstance(args[0], tuple) else None
        )
        requested_device = torch.device(kwargs.get("device", "cpu"))
        if requested_device.type == "cuda" and requested_shape in ((1, 1), (1, 1, 1)):
            raise AssertionError(
                f"staged tiled-topk replay allocated CUDA dummy {requested_shape}"
            )
        return real_empty(*args, **kwargs)

    with kernel_resolution_guard('staged tiled top-k graph replay must use warmed kernels'):
        with monkeypatch.context() as patch:
            patch.setattr(torch, "empty", reject_cuda_dummy_empty)
            guarded_out = contiguous_tiled_topk(
                q_fp8=q_fp8,
                weights=weights,
                kv_fp8=bound_kv_fp8,
                binding=binding,
            )
        torch.cuda.synchronize(device)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured_out = contiguous_tiled_topk(
                q_fp8=q_fp8,
                weights=weights,
                kv_fp8=bound_kv_fp8,
                binding=binding,
            )

    assert binding.output_indices is not None
    assert binding.output_values is not None
    assert guarded_out.data_ptr() == binding.output_indices.data_ptr()
    assert captured_out.data_ptr() == binding.output_indices.data_ptr()

    graph.replay()
    torch.cuda.synchronize(device)
    actual0 = binding.output_values.clone()
    expected0 = expected_topk_values()
    torch.testing.assert_close(
        torch.sort(actual0, dim=1).values,
        torch.sort(expected0, dim=1).values,
        atol=1e-4,
        rtol=1e-4,
    )

    weights.copy_(live_weights)
    graph.replay()
    torch.cuda.synchronize(device)
    actual1 = binding.output_values.clone()
    expected1 = expected_topk_values()
    torch.testing.assert_close(
        torch.sort(actual1, dim=1).values,
        torch.sort(expected1, dim=1).values,
        atol=1e-4,
        rtol=1e-4,
    )
    assert not torch.equal(actual0, actual1)
    assert compile_cache_info()["compile_misses"] == warm_compile_misses


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for tiled topk coverage"
)
def test_contiguous_tiled_topk_matches_scatter_logits(monkeypatch) -> None:
    monkeypatch.setenv("B12X_DSA_TOPK_SUPERTILE_K", "3072")

    device = torch.device("cuda")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(72_620)

    q_rows = 256
    num_heads = 8
    k_rows = 4096
    topk = 2048
    q_fp8 = (
        torch.randn((q_rows, num_heads, 128), generator=gen, dtype=torch.float32).to(
            device=device
        )
        / 2
    ).to(torch.float8_e4m3fn)
    weights = torch.randn((q_rows, num_heads), generator=gen, dtype=torch.float32).to(
        device=device
    )
    k = (
        torch.randn((k_rows, 128), generator=gen, dtype=torch.float32).to(device=device)
        / 3
    )
    kv_fp8 = _quantize_rows_to_kv_fp8(k)
    k_start = torch.zeros(q_rows, dtype=torch.int32, device=device)
    k_end = torch.full((q_rows,), k_rows, dtype=torch.int32, device=device)
    metadata = IndexerContiguousMetadata(k_start=k_start, k_end=k_end)
    binding, bound_kv_fp8 = _bind_contiguous_topk(
        kv_fp8=kv_fp8,
        k_start=k_start,
        k_end=k_end,
        num_q_heads=num_heads,
        topk=topk,
        q_rows=q_rows,
        supertile_k=3072,
    )

    actual = contiguous_tiled_topk(
        q_fp8=q_fp8,
        weights=weights,
        kv_fp8=bound_kv_fp8,
        binding=binding,
    )
    logits = contiguous_logits(
        q_fp8=q_fp8,
        weights=weights,
        kv_fp8=kv_fp8,
        metadata=metadata,
        preinitialize_invalid_logits=False,
    )
    expected = torch.topk(logits, k=topk, dim=1, largest=True, sorted=False).indices.to(
        torch.int32
    )

    torch.cuda.synchronize(device)
    assert actual.shape == (q_rows, topk)
    assert torch.equal(
        torch.sort(actual, dim=1).values, torch.sort(expected, dim=1).values
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for tiled topk coverage"
)
def test_contiguous_tiled_topk_streaming_fold_many_chunks(monkeypatch) -> None:
    # Force >= 3 supertile chunks so the fold exercises the middle (is_first=False)
    # kernel specialization and the full carry ping-pong (not just the 2-chunk
    # first+last pair). k_rows / supertile_k = 5 chunks regardless of block_k.
    monkeypatch.setenv("B12X_DSA_TOPK_SUPERTILE_K", "2048")

    device = torch.device("cuda")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(20_260_603)

    q_rows = 128
    num_heads = 8
    k_rows = 10240
    topk = 512
    q_fp8 = (
        torch.randn((q_rows, num_heads, 128), generator=gen, dtype=torch.float32).to(
            device=device
        )
        / 2
    ).to(torch.float8_e4m3fn)
    weights = torch.randn((q_rows, num_heads), generator=gen, dtype=torch.float32).to(
        device=device
    )
    k = (
        torch.randn((k_rows, 128), generator=gen, dtype=torch.float32).to(device=device)
        / 3
    )
    kv_fp8 = _quantize_rows_to_kv_fp8(k)
    # Non-uniform per-row windows exercise clipping + carry padding across chunks
    # while keeping every row's range >= topk so the result is fully determined.
    k_start = torch.zeros(q_rows, dtype=torch.int32, device=device)
    k_end = torch.randint(
        topk + 1, k_rows + 1, (q_rows,), generator=gen, dtype=torch.int32
    ).to(device=device)
    metadata = IndexerContiguousMetadata(k_start=k_start, k_end=k_end)
    binding, bound_kv_fp8 = _bind_contiguous_topk(
        kv_fp8=kv_fp8,
        k_start=k_start,
        k_end=k_end,
        num_q_heads=num_heads,
        topk=topk,
        q_rows=q_rows,
        supertile_k=2048,
    )

    actual = contiguous_tiled_topk(
        q_fp8=q_fp8,
        weights=weights,
        kv_fp8=bound_kv_fp8,
        binding=binding,
    )
    logits = contiguous_logits(
        q_fp8=q_fp8,
        weights=weights,
        kv_fp8=kv_fp8,
        metadata=metadata,
        preinitialize_invalid_logits=False,
    )
    expected = torch.topk(logits, k=topk, dim=1, largest=True, sorted=False).indices.to(
        torch.int32
    )

    torch.cuda.synchronize(device)
    assert actual.shape == (q_rows, topk)
    # Correctness invariant under ties: fp8-derived logits contain many equal
    # values, so the exact top-k set is ambiguous at the rank-`topk` boundary and
    # the radix kernel may pick different tied indices than torch.topk. The
    # tie-robust check is that the selected logit-VALUE multiset matches the
    # reference per row — that proves the fold returned a valid exact top-k.
    actual_values = torch.sort(torch.gather(logits, 1, actual.long()), dim=1).values
    expected_values = torch.sort(torch.gather(logits, 1, expected.long()), dim=1).values
    assert torch.equal(actual_values, expected_values)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for indexer compile-cache coverage",
)
def test_contiguous_tiled_topk_live_rows_do_not_resolve_new_kernel(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("B12X_COMPILE_CACHE_DIR", str(tmp_path / "cute-cache"))
    monkeypatch.setenv("B12X_DSA_TOPK_SUPERTILE_K", "32768")

    device = torch.device("cuda")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(72_621)
    clear_compile_cache()
    clear_indexer_caches()

    num_heads = 32
    topk = 512

    def make_inputs(q_rows: int, k_rows: int):
        q_fp8 = (
            torch.randn(
                (q_rows, num_heads, 128), generator=gen, dtype=torch.float32
            ).to(device=device)
            / 2
        ).to(torch.float8_e4m3fn)
        weights = torch.randn(
            (q_rows, num_heads), generator=gen, dtype=torch.float32
        ).to(device=device)
        k = (
            torch.randn((k_rows, 128), generator=gen, dtype=torch.float32).to(
                device=device
            )
            / 3
        )
        kv_fp8 = _quantize_rows_to_kv_fp8(k)
        k_start = torch.zeros(q_rows, dtype=torch.int32, device=device)
        k_end = torch.full((q_rows,), k_rows, dtype=torch.int32, device=device)
        metadata = IndexerContiguousMetadata(k_start=k_start, k_end=k_end)
        binding, bound_kv_fp8 = _bind_contiguous_topk(
            kv_fp8=kv_fp8,
            k_start=k_start,
            k_end=k_end,
            num_q_heads=num_heads,
            topk=topk,
            q_rows=q_rows,
            supertile_k=32768,
        )
        return q_fp8, weights, bound_kv_fp8, binding, metadata

    warm_q, warm_weights, warm_kv, warm_binding, _ = make_inputs(2048, 4096)
    contiguous_tiled_topk(
        q_fp8=warm_q,
        weights=warm_weights,
        kv_fp8=warm_kv,
        binding=warm_binding,
    )
    torch.cuda.synchronize(device)
    warm_misses = compile_cache_info()["compile_misses"]

    live_q, live_weights, live_kv, live_binding, _ = make_inputs(1536, 5001)
    with kernel_resolution_guard('indexer contiguous live rows and padded K rows should be runtime'):
        actual = contiguous_tiled_topk(
            q_fp8=live_q,
            weights=live_weights,
            kv_fp8=live_kv,
            binding=live_binding,
        )
        torch.cuda.synchronize(device)

    assert actual.shape == (1536, topk)
    assert compile_cache_info()["compile_misses"] == warm_misses


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for indexer compile-cache coverage",
)
def test_row_topk_indices_only_preserves_value_buffer() -> None:
    device = torch.device("cuda")
    generator = torch.Generator(device="cpu").manual_seed(73_621)
    rows, width, topk = 17, 1024, 512
    row_logits = torch.randn(
        (rows, width), generator=generator, dtype=torch.float32
    ).to(device)
    lengths = torch.full((rows,), width, dtype=torch.int32, device=device)
    output_values = torch.full(
        (rows, topk), 12345.0, dtype=torch.float32, device=device
    )
    output_indices = torch.empty((rows, topk), dtype=torch.int32, device=device)

    run_row_topk(
        row_logits=row_logits,
        lengths=lengths,
        topk=topk,
        output_values=output_values,
        output_indices=output_indices,
        write_values=False,
    )
    torch.cuda.synchronize(device)

    expected = torch.topk(row_logits, k=topk, dim=1, largest=True, sorted=False)
    assert bool((output_values == 12345.0).all())
    assert torch.equal(
        torch.sort(output_indices, dim=1).values.to(torch.long),
        torch.sort(expected.indices, dim=1).values,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for padded row top-k coverage",
)
def test_row_topk_padded_overflow_is_unique_and_graph_safe() -> None:
    """Validate a gathered row fold with fewer than top-k live candidates."""
    device = torch.device("cuda")
    generator = torch.Generator(device="cpu").manual_seed(73_623)
    rows, width, topk = 2304, 2560, 512
    live_counts = (
        torch.arange(rows, dtype=torch.int64, device=device)
        .div(4, rounding_mode="floor")
        .clamp(min=1)
    )
    columns = torch.arange(width, dtype=torch.int64, device=device).view(1, -1)
    live_mask = columns < live_counts.view(-1, 1)
    row_logits = torch.randn(
        (rows, width), generator=generator, dtype=torch.float32
    ).to(device)
    row_logits.masked_fill_(~live_mask, float("-inf"))
    gather_table = (
        torch.arange(width, dtype=torch.int32, device=device)
        .expand(rows, -1)
        .contiguous()
    )
    gather_table.masked_fill_(~live_mask, -1)
    lengths = torch.full((rows,), width, dtype=torch.int32, device=device)
    output_indices = torch.empty((rows, topk), dtype=torch.int32, device=device)
    output_values = torch.full((rows, topk), 12345.0, dtype=torch.float32, device=device)

    run_row_topk(
        row_logits=row_logits,
        lengths=lengths,
        topk=topk,
        output_indices=output_indices,
        output_values=output_values,
        output_gather_table=gather_table,
        write_values=False,
    )
    torch.cuda.synchronize(device)

    with kernel_resolution_guard('padded row top-k graph replay must use the warmed kernel'):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run_row_topk(
                row_logits=row_logits,
                lengths=lengths,
                topk=topk,
                output_indices=output_indices,
                output_values=output_values,
                output_gather_table=gather_table,
                write_values=False,
            )
    pointers = (
        row_logits.data_ptr(), output_values.data_ptr(), output_indices.data_ptr(),
    )
    output_indices.fill_(-2)
    torch.cuda.synchronize(device)
    before = torch.cuda.memory_stats(device)
    for _ in range(8):
        graph.replay()
    torch.cuda.synchronize(device)
    after = torch.cuda.memory_stats(device)
    for key in ("allocation.all.allocated", "allocated_bytes.all.allocated"):
        assert before[key] == after[key]
    assert pointers == (
        row_logits.data_ptr(), output_values.data_ptr(), output_indices.data_ptr(),
    )
    assert bool((output_values == 12345.0).all())

    logical = output_indices.to(torch.int64)
    valid = logical >= 0
    assert torch.equal(valid.sum(dim=1), live_counts.clamp(max=topk))
    assert not bool(((logical >= live_counts.view(-1, 1)) & valid).any())
    occurrences = torch.zeros((rows, width), dtype=torch.int16, device=device)
    occurrences.scatter_add_(1, logical.clamp_min(0), valid.to(torch.int16))
    assert int(occurrences.max()) == 1

    full_rows = live_counts >= topk
    expected = torch.topk(
        row_logits[full_rows], k=topk, dim=1, largest=True, sorted=False
    ).indices
    assert torch.equal(
        torch.sort(logical[full_rows], dim=1).values,
        torch.sort(expected, dim=1).values,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for indexer compile-cache coverage",
)
def test_row_topk_graph_replay_tracks_live_logits_and_lengths() -> None:
    device = torch.device("cuda")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(73_622)

    rows = 17
    width = 1024
    topk = 512
    row_logits = torch.randn((rows, width), generator=gen, dtype=torch.float32).to(
        device=device
    )
    live_logits = torch.randn((rows, width), generator=gen, dtype=torch.float32).to(
        device=device
    )
    lengths = torch.linspace(topk, width, rows, dtype=torch.float32, device=device).to(
        torch.int32
    )
    live_lengths = torch.flip(lengths, dims=(0,)).contiguous()
    output_values = torch.empty((rows, topk), dtype=torch.float32, device=device)
    output_indices = torch.empty((rows, topk), dtype=torch.int32, device=device)

    def expected_topk() -> torch.return_types.topk:
        columns = torch.arange(width, dtype=torch.int32, device=device)
        masked = torch.where(
            columns[None, :] < lengths[:, None],
            row_logits,
            torch.full_like(row_logits, float("-inf")),
        )
        return torch.topk(masked, k=topk, dim=1, largest=True, sorted=False)

    clear_indexer_caches()
    run_row_topk(
        row_logits=row_logits,
        lengths=lengths,
        topk=topk,
        output_values=output_values,
        output_indices=output_indices,
    )
    torch.cuda.synchronize(device)
    warm_compile_misses = compile_cache_info()["compile_misses"]

    with kernel_resolution_guard('row top-k graph replay must use the warmed kernel'):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured_values, captured_indices = run_row_topk(
                row_logits=row_logits,
                lengths=lengths,
                topk=topk,
                output_values=output_values,
                output_indices=output_indices,
            )

    assert captured_values.data_ptr() == output_values.data_ptr()
    assert captured_indices.data_ptr() == output_indices.data_ptr()

    graph.replay()
    torch.cuda.synchronize(device)
    expected0 = expected_topk()
    assert torch.equal(
        torch.sort(output_values, dim=1).values,
        torch.sort(expected0.values, dim=1).values,
    )
    assert torch.equal(
        torch.sort(output_indices, dim=1).values.to(torch.long),
        torch.sort(expected0.indices, dim=1).values,
    )

    row_logits.copy_(live_logits)
    lengths.copy_(live_lengths)
    graph.replay()
    torch.cuda.synchronize(device)
    expected1 = expected_topk()
    assert torch.equal(
        torch.sort(output_values, dim=1).values,
        torch.sort(expected1.values, dim=1).values,
    )
    assert torch.equal(
        torch.sort(output_indices, dim=1).values.to(torch.long),
        torch.sort(expected1.indices, dim=1).values,
    )
    assert compile_cache_info()["compile_misses"] == warm_compile_misses


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for indexer compile-cache coverage",
)
@pytest.mark.parametrize("topk", [512, 1024])
def test_row_topk_live_rows_do_not_resolve_new_kernel(
    tmp_path, monkeypatch, topk: int
) -> None:
    monkeypatch.setenv("B12X_COMPILE_CACHE_DIR", str(tmp_path / "cute-cache"))

    device = torch.device("cuda")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(72_622)
    clear_compile_cache()
    clear_indexer_caches()

    width = 1024

    def make_inputs(rows: int) -> tuple[torch.Tensor, torch.Tensor]:
        logits = torch.randn(
            (rows, width),
            generator=gen,
            dtype=torch.float32,
            device="cpu",
        ).to(device=device)
        lengths = torch.full((rows,), width, dtype=torch.int32, device=device)
        return logits, lengths

    warm_logits, warm_lengths = make_inputs(256)
    run_row_topk(row_logits=warm_logits, lengths=warm_lengths, topk=topk)
    torch.cuda.synchronize(device)
    warm_misses = compile_cache_info()["compile_misses"]

    live_logits, live_lengths = make_inputs(113)
    with kernel_resolution_guard('row topk live rows should be runtime'):
        values, indices = run_row_topk(
            row_logits=live_logits,
            lengths=live_lengths,
            topk=topk,
        )
        torch.cuda.synchronize(device)

    expected = torch.topk(live_logits, k=topk, dim=1, largest=True, sorted=False)
    assert compile_cache_info()["compile_misses"] == warm_misses
    assert torch.equal(
        torch.sort(indices, dim=1).values.to(torch.long),
        torch.sort(expected.indices, dim=1).values,
    )
    torch.testing.assert_close(
        torch.sort(values, dim=1).values,
        torch.sort(expected.values, dim=1).values,
        atol=0,
        rtol=0,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for indexer compile-cache coverage",
)
def test_row_topk_gather_table_width_and_stride_are_runtime(monkeypatch) -> None:
    monkeypatch.setenv("B12X_COMPILE_DISK_CACHE", "0")
    monkeypatch.setenv("B12X_COMPILE_MEMORY_CACHE", "1")
    clear_compile_cache()
    clear_indexer_caches()

    device = torch.device("cuda")
    rows, topk = 17, 512

    def select(width: int) -> None:
        row_logits = (
            torch.arange(width, dtype=torch.float32, device=device)
            .expand(rows, -1)
            .contiguous()
        )
        lengths = torch.full((rows,), width, dtype=torch.int32, device=device)
        gather_table = torch.arange(
            rows * width, dtype=torch.int32, device=device
        ).reshape(rows, width)
        _, indices = run_row_topk(
            row_logits=row_logits,
            lengths=lengths,
            topk=topk,
            output_gather_table=gather_table,
        )
        torch.cuda.synchronize(device)
        logical = torch.topk(
            row_logits, k=topk, dim=1, largest=True, sorted=False
        ).indices
        expected = torch.gather(gather_table, 1, logical)
        assert torch.equal(
            torch.sort(indices, dim=1).values,
            torch.sort(expected, dim=1).values,
        )

    select(512)
    misses_after_narrow = compile_cache_info()["compile_misses"]
    select(1024)
    assert compile_cache_info()["compile_misses"] == misses_after_narrow


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for BK512 prefill coverage"
)
def test_contiguous_logits_cuda_prefill512_sampled_logits(monkeypatch) -> None:
    monkeypatch.setenv("B12X_DSA_CONTIGUOUS_PREFILL_BLOCK_K", "512")

    device = torch.device("cuda")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(72_512)

    q_rows = 1024
    k_rows = 32768
    num_heads = 32
    q_fp8 = (
        torch.randn((q_rows, num_heads, 128), generator=gen, dtype=torch.float32).to(
            device=device
        )
        / 2
    ).to(torch.float8_e4m3fn)
    weights = torch.randn((q_rows, num_heads), generator=gen, dtype=torch.float32).to(
        device=device
    )
    k = (
        torch.randn((k_rows, 128), generator=gen, dtype=torch.float32).to(device=device)
        / 3
    )
    kv_fp8 = _quantize_rows_to_kv_fp8(k)

    k_start = torch.zeros(q_rows, dtype=torch.int32, device=device)
    k_end = torch.zeros(q_rows, dtype=torch.int32, device=device)
    ranges = {
        0: (0, k_rows),
        31: (256, 1024),
        32: (512, 1536),
        255: (8192, 12288),
        512: (16384, k_rows),
        1023: (k_rows - 1024, k_rows),
    }
    for q_idx, (start, end) in ranges.items():
        k_start[q_idx] = start
        k_end[q_idx] = end

    actual = contiguous_logits(
        q_fp8=q_fp8,
        weights=weights,
        kv_fp8=kv_fp8,
        metadata=IndexerContiguousMetadata(k_start=k_start, k_end=k_end),
    )
    torch.cuda.synchronize(device)

    k_quant, k_scale = kv_fp8

    def assert_sampled_logits(q_idx: int, cols: list[int]) -> None:
        k_cols = torch.tensor(cols, dtype=torch.long, device=device)
        scores = torch.matmul(
            q_fp8[q_idx].to(torch.float32), k_quant[k_cols].to(torch.float32).T
        )
        expected = (torch.relu(scores) * weights[q_idx].unsqueeze(1)).sum(
            dim=0
        ) * k_scale[k_cols]
        torch.testing.assert_close(
            actual[q_idx, k_cols], expected, atol=1e-4, rtol=1e-4
        )

    assert_sampled_logits(0, [0, 255, 256, 511, 512, 8191, 16384, 32767])
    assert_sampled_logits(31, [256, 511, 512, 1023])
    assert_sampled_logits(32, [512, 1024, 1535])
    assert_sampled_logits(255, [8192, 8193, 12287])
    assert_sampled_logits(512, [16384, 32767])
    assert_sampled_logits(1023, [31744, 32767])

    assert torch.isneginf(actual[31, torch.tensor([0, 1024], device=device)]).all()
    assert torch.isneginf(actual[32, torch.tensor([511, 1536], device=device)]).all()
    assert torch.isneginf(actual[255, torch.tensor([8191, 12288], device=device)]).all()
    assert torch.isneginf(actual[512, torch.tensor([0, 16383], device=device)]).all()
    assert torch.isneginf(actual[1023, torch.tensor([31743], device=device)]).all()
    assert torch.isneginf(actual[1]).all()
    assert torch.isneginf(actual[64]).all()
