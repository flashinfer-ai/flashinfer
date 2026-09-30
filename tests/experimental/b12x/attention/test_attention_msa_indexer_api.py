from __future__ import annotations

import pytest
import torch

from b12x._lib.runtime_control import kernel_resolution_guard
from b12x.attention.dsa_indexer._impl import (
    IndexerContiguousMetadata,
    IndexerPagedDecodeMetadata,
    clear_indexer_caches,
    msa_contiguous_block_scores,
    msa_decode_query_positions,
    msa_paged_decode_block_scores,
    msa_q2k_indices_decode,
    msa_q2k_indices_prefill,
    quantize_msa_q_fp8,
)
from b12x.attention.dsa_indexer.msa_reference import MSA_SM_SCALE
from b12x.attention.dsa_indexer.contiguous_kernel import (
    run_contiguous_block_scores_kernel,
)
from b12x.attention.dsa_indexer.msa_reference import (
    msa_contiguous_block_scores_reference,
    msa_paged_decode_block_scores_reference,
    msa_q2k_indices_reference,
)
from b12x.attention.dsa_indexer.reference import pack_index_k_cache_reference
from b12x.attention.dsa_indexer.scratch import (
    B12XIndexerContiguousScratchCaps,
    B12XIndexerPagedScratchCaps,
    plan_indexer_contiguous_scratch,
    plan_indexer_paged_scratch,
)
from b12x._lib.compiler import compile_cache_info


def _one_scratch(plan):
    (spec,) = plan.scratch_specs()
    return torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)


def _make_real_page_table(
    *,
    rows: int,
    width_pages: int,
    device: torch.device,
) -> torch.Tensor:
    page_ids = torch.arange(width_pages, dtype=torch.int32, device=device)
    return page_ids.unsqueeze(0).expand(rows, width_pages).contiguous()


def _make_msa_decode_case(
    *,
    rows: int,
    heads: int,
    width_pages: int,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, IndexerPagedDecodeMetadata]:
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    q = (
        torch.randn(
            (rows, heads, 128), generator=gen, dtype=torch.float32, device=device
        )
        / 3
    )
    q_fp8, q_scale = quantize_msa_q_fp8(q)
    k_rows = max(width_pages * 64, 1)
    k = (
        torch.randn((k_rows, 128), generator=gen, dtype=torch.float32, device=device)
        / 3
    )
    if rows and heads:
        # Ensure at least one finite block max is negative, which catches accidental ReLU.
        q0 = torch.ones((128,), dtype=torch.float32, device=device) / 4
        k0 = -torch.ones((128, 128), dtype=torch.float32, device=device) / 4
        q_fp8[:1, :1], q_scale[:1, :1] = quantize_msa_q_fp8(q0.view(1, 1, 128))
        k[:128].copy_(k0)
    index_k_cache = pack_index_k_cache_reference(k)
    real_page_table = _make_real_page_table(
        rows=rows, width_pages=width_pages, device=device
    )
    seqlens = torch.full((rows,), width_pages * 64, dtype=torch.int32, device=device)
    if rows >= 3:
        seqlens[1] = max(width_pages * 64 - 1, 0)
        seqlens[2] = min(width_pages * 64, 129)
    metadata = IndexerPagedDecodeMetadata(
        real_page_table=real_page_table,
        cache_seqlens_int32=seqlens,
    )
    return q_fp8, q_scale, index_k_cache, metadata


def _quantize_rows_to_kv_fp8(k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    fp8_max = float(torch.finfo(torch.float8_e4m3fn).max)
    scale = k.abs().amax(dim=1) / fp8_max
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    quant = (k / scale.unsqueeze(1)).clamp(-fp8_max, fp8_max)
    return quant.to(torch.float8_e4m3fn), scale.to(torch.float32)


def _make_msa_contiguous_case(
    *,
    rows: int,
    heads: int,
    k_rows: int,
    device: torch.device,
    seed: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    tuple[torch.Tensor, torch.Tensor],
    IndexerContiguousMetadata,
]:
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    q = (
        torch.randn(
            (rows, heads, 128), generator=gen, dtype=torch.float32, device=device
        )
        / 3
    )
    q_fp8, q_scale = quantize_msa_q_fp8(q)
    k = (
        torch.randn((k_rows, 128), generator=gen, dtype=torch.float32, device=device)
        / 3
    )
    if rows and heads and k_rows >= 128:
        q0 = torch.ones((128,), dtype=torch.float32, device=device) / 4
        k[:128].fill_(-0.25)
        q_fp8[:1, :1], q_scale[:1, :1] = quantize_msa_q_fp8(q0.view(1, 1, 128))
    kv_fp8 = _quantize_rows_to_kv_fp8(k)
    k_start = torch.zeros((rows,), dtype=torch.int32, device=device)
    k_end = torch.arange(1, rows + 1, dtype=torch.int32, device=device).clamp(
        max=k_rows
    )
    if rows >= 3:
        k_start[2:] = 128
        k_end[2:] = torch.arange(
            129, 129 + rows - 2, dtype=torch.int32, device=device
        ).clamp(max=k_rows)
    metadata = IndexerContiguousMetadata(k_start=k_start, k_end=k_end)
    return q_fp8, q_scale, kv_fp8, metadata


def test_msa_decode_query_positions() -> None:
    seqlens = torch.tensor([1, 128, 129], dtype=torch.int32)
    assert torch.equal(msa_decode_query_positions(seqlens), torch.tensor([0, 127, 128]))


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for MSA decode coverage"
)
def test_msa_paged_scratch_binding_owns_decode_outputs() -> None:
    device = torch.device("cuda")
    q_fp8, q_scale, index_k_cache, metadata = _make_msa_decode_case(
        rows=2,
        heads=4,
        width_pages=5,
        device=device,
        seed=92_155,
    )
    plan = plan_indexer_paged_scratch(
        B12XIndexerPagedScratchCaps(
            device=device,
            num_q_heads=1,
            num_idx_heads=4,
            max_q_rows=2,
            max_page_table_width=5,
            topk=16,
            score_mode="msa",
        )
    )
    binding = plan.bind_msa(
        scratch=_one_scratch(plan),
        real_page_table=metadata.real_page_table,
        cache_seqlens_int32=metadata.cache_seqlens_int32,
    )

    actual = msa_q2k_indices_decode(
        q_fp8=q_fp8,
        q_scale=q_scale,
        index_k_cache=index_k_cache,
        binding=binding,
    )
    expected = msa_q2k_indices_reference(
        q_fp8=q_fp8,
        q_scale=q_scale,
        index_k_cache=index_k_cache,
        real_page_table=metadata.real_page_table,
        cache_seqlens_int32=metadata.cache_seqlens_int32,
        query_positions=metadata.cache_seqlens_int32 - 1,
    )

    assert actual.data_ptr() == binding.q2k_indices.data_ptr()
    assert binding.block_scores is not None
    assert binding.page_scores is not None
    assert binding.block_scores.shape == (4, 2, 3)
    assert binding.page_scores.shape == (4, 2, 6)
    assert torch.equal(actual, expected)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for MSA decode coverage"
)
def test_msa_paged_binding_rejects_duplicate_outputs_and_runs_valid_kernel() -> None:
    device = torch.device("cuda")
    q_fp8, q_scale, index_k_cache, metadata = _make_msa_decode_case(
        rows=2,
        heads=4,
        width_pages=5,
        device=device,
        seed=92_156,
    )
    plan = plan_indexer_paged_scratch(
        B12XIndexerPagedScratchCaps(
            device=device,
            num_q_heads=1,
            num_idx_heads=4,
            max_q_rows=2,
            max_page_table_width=5,
            topk=16,
            score_mode="msa",
        )
    )
    binding = plan.bind_msa(
        scratch=_one_scratch(plan),
        real_page_table=metadata.real_page_table,
        cache_seqlens_int32=metadata.cache_seqlens_int32,
    )
    out_scores = torch.empty((4, 2, 3), dtype=torch.float32, device=device)
    out_indices = torch.empty((4, 2, 16), dtype=torch.int32, device=device)

    with pytest.raises(ValueError, match="binding owns metadata"):
        msa_paged_decode_block_scores(
            q_fp8=q_fp8,
            q_scale=q_scale,
            index_k_cache=index_k_cache,
            metadata=metadata,
            binding=binding,
        )
    with pytest.raises(
        ValueError, match="binding owns metadata and block-score buffers"
    ):
        msa_paged_decode_block_scores(
            q_fp8=q_fp8,
            q_scale=q_scale,
            index_k_cache=index_k_cache,
            out=out_scores,
            binding=binding,
        )
    with pytest.raises(ValueError, match="binding owns metadata and q2k output"):
        msa_q2k_indices_decode(
            q_fp8=q_fp8,
            q_scale=q_scale,
            index_k_cache=index_k_cache,
            out_indices=out_indices,
            binding=binding,
        )

    nsa_plan = plan_indexer_paged_scratch(
        B12XIndexerPagedScratchCaps(
            device=device,
            num_q_heads=1,
            max_q_rows=2,
            max_page_table_width=5,
            topk=16,
        )
    )
    with pytest.raises(RuntimeError, match="score_mode='msa'"):
        nsa_plan.bind_msa(
            scratch=_one_scratch(nsa_plan),
            real_page_table=metadata.real_page_table,
            cache_seqlens_int32=metadata.cache_seqlens_int32,
        )

    # Host-side rejection checks must not turn this into a CPU-only migration
    # test: finish through the same valid binding with the physical kernel.
    actual = msa_q2k_indices_decode(
        q_fp8=q_fp8,
        q_scale=q_scale,
        index_k_cache=index_k_cache,
        binding=binding,
    )
    expected = msa_q2k_indices_reference(
        q_fp8=q_fp8,
        q_scale=q_scale,
        index_k_cache=index_k_cache,
        real_page_table=metadata.real_page_table,
        cache_seqlens_int32=metadata.cache_seqlens_int32,
        query_positions=metadata.cache_seqlens_int32 - 1,
    )
    torch.cuda.synchronize(device)
    assert torch.equal(actual, expected)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for MSA prefill coverage"
)
def test_msa_contiguous_scratch_binding_owns_prefill_outputs() -> None:
    device = torch.device("cuda")
    q_fp8, q_scale, kv_fp8, metadata = _make_msa_contiguous_case(
        rows=3,
        heads=4,
        k_rows=257,
        device=device,
        seed=92_255,
    )
    plan = plan_indexer_contiguous_scratch(
        B12XIndexerContiguousScratchCaps(
            device=device,
            num_q_heads=1,
            num_idx_heads=4,
            max_q_rows=3,
            max_k_rows=257,
            topk=16,
            score_mode="msa",
        )
    )
    binding = plan.bind_msa(
        scratch=_one_scratch(plan),
        k_start=metadata.k_start,
        k_end=metadata.k_end,
    )
    k_quant, k_scale = kv_fp8
    k_rows = int(k_quant.shape[0])
    binding.scratch.k_quant[:k_rows].copy_(k_quant)
    binding.scratch.k_scale[:k_rows].copy_(k_scale)
    scratch_kv_fp8 = (
        binding.scratch.k_quant[:k_rows],
        binding.scratch.k_scale[:k_rows],
    )

    actual = msa_q2k_indices_prefill(
        q_fp8=q_fp8,
        q_scale=q_scale,
        kv_fp8=scratch_kv_fp8,
        binding=binding,
    )
    expected = msa_q2k_indices_reference(
        q_fp8=q_fp8,
        q_scale=q_scale,
        kv_fp8=scratch_kv_fp8,
        k_start=metadata.k_start,
        k_end=metadata.k_end,
        query_positions=metadata.k_end - 1,
        block_base=torch.div(metadata.k_start, 128, rounding_mode="floor"),
    )

    assert actual.data_ptr() == binding.q2k_indices.data_ptr()
    assert binding.block_scores is not None
    assert binding.block_scores.shape == (4, 3, 3)
    assert torch.equal(actual, expected)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for MSA prefill coverage"
)
def test_msa_contiguous_binding_rejects_duplicate_outputs_and_runs_valid_kernel() -> (
    None
):
    device = torch.device("cuda")
    q_fp8, q_scale, kv_fp8, metadata = _make_msa_contiguous_case(
        rows=3,
        heads=4,
        k_rows=257,
        device=device,
        seed=92_256,
    )
    plan = plan_indexer_contiguous_scratch(
        B12XIndexerContiguousScratchCaps(
            device=device,
            num_q_heads=1,
            num_idx_heads=4,
            max_q_rows=3,
            max_k_rows=257,
            topk=16,
            score_mode="msa",
        )
    )
    binding = plan.bind_msa(
        scratch=_one_scratch(plan),
        k_start=metadata.k_start,
        k_end=metadata.k_end,
    )
    out_scores = torch.empty((4, 3, 3), dtype=torch.float32, device=device)
    out_indices = torch.empty((4, 3, 16), dtype=torch.int32, device=device)

    with pytest.raises(ValueError, match="binding owns metadata"):
        msa_contiguous_block_scores(
            q_fp8=q_fp8,
            q_scale=q_scale,
            kv_fp8=kv_fp8,
            metadata=metadata,
            binding=binding,
        )
    with pytest.raises(
        ValueError, match="binding owns metadata and block-score buffers"
    ):
        msa_contiguous_block_scores(
            q_fp8=q_fp8,
            q_scale=q_scale,
            kv_fp8=kv_fp8,
            out=out_scores,
            binding=binding,
        )
    with pytest.raises(ValueError, match="binding owns metadata and q2k output"):
        msa_q2k_indices_prefill(
            q_fp8=q_fp8,
            q_scale=q_scale,
            kv_fp8=kv_fp8,
            out_indices=out_indices,
            binding=binding,
        )

    nsa_plan = plan_indexer_contiguous_scratch(
        B12XIndexerContiguousScratchCaps(
            device=device,
            num_q_heads=1,
            max_q_rows=3,
            max_k_rows=257,
            topk=16,
        )
    )
    with pytest.raises(RuntimeError, match="score_mode='msa'"):
        nsa_plan.bind_msa(
            scratch=_one_scratch(nsa_plan),
            k_start=metadata.k_start,
            k_end=metadata.k_end,
        )

    # Exercise the physical implementation after the host-side validation
    # assertions.  The valid binding owns the staged K/V and output storage.
    k_quant, k_scale = kv_fp8
    k_rows = int(k_quant.shape[0])
    binding.scratch.k_quant[:k_rows].copy_(k_quant)
    binding.scratch.k_scale[:k_rows].copy_(k_scale)
    scratch_kv_fp8 = (
        binding.scratch.k_quant[:k_rows],
        binding.scratch.k_scale[:k_rows],
    )
    actual = msa_q2k_indices_prefill(
        q_fp8=q_fp8,
        q_scale=q_scale,
        kv_fp8=scratch_kv_fp8,
        binding=binding,
    )
    expected = msa_q2k_indices_reference(
        q_fp8=q_fp8,
        q_scale=q_scale,
        kv_fp8=scratch_kv_fp8,
        k_start=metadata.k_start,
        k_end=metadata.k_end,
        query_positions=metadata.k_end - 1,
        block_base=torch.div(metadata.k_start, 128, rounding_mode="floor"),
    )
    torch.cuda.synchronize(device)
    assert torch.equal(actual, expected)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for MSA decode scorer coverage",
)
@pytest.mark.parametrize(
    ("rows", "heads", "width_pages"),
    [
        (2, 1, 8),
        (1, 4, 1024),
        (4, 4, 1024),
    ],
)
def test_msa_paged_decode_block_scores_cuda_matches_reference(
    rows: int,
    heads: int,
    width_pages: int,
) -> None:
    clear_indexer_caches()
    device = torch.device("cuda")
    q_fp8, q_scale, index_k_cache, metadata = _make_msa_decode_case(
        rows=rows,
        heads=heads,
        width_pages=width_pages,
        device=device,
        seed=92_201 + rows * 17 + heads,
    )

    actual = msa_paged_decode_block_scores(
        q_fp8=q_fp8,
        q_scale=q_scale,
        index_k_cache=index_k_cache,
        metadata=metadata,
    )
    expected = msa_paged_decode_block_scores_reference(
        q_fp8=q_fp8,
        q_scale=q_scale,
        index_k_cache=index_k_cache,
        real_page_table=metadata.real_page_table,
        cache_seqlens_int32=metadata.cache_seqlens_int32,
    )

    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
    finite = expected[0, 0].isfinite()
    assert torch.any(actual[0, 0, finite] < 0)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for MSA decode scorer coverage",
)
def test_msa_q2k_indices_decode_cuda_matches_reference() -> None:
    clear_indexer_caches()
    device = torch.device("cuda")
    q_fp8, q_scale, index_k_cache, metadata = _make_msa_decode_case(
        rows=4,
        heads=4,
        width_pages=9,
        device=device,
        seed=92_301,
    )

    actual = msa_q2k_indices_decode(
        q_fp8=q_fp8,
        q_scale=q_scale,
        index_k_cache=index_k_cache,
        metadata=metadata,
        topk=8,
    )
    expected = msa_q2k_indices_reference(
        q_fp8=q_fp8,
        q_scale=q_scale,
        index_k_cache=index_k_cache,
        real_page_table=metadata.real_page_table,
        cache_seqlens_int32=metadata.cache_seqlens_int32,
        query_positions=metadata.cache_seqlens_int32 - 1,
        topk=8,
    )

    assert torch.equal(actual, expected)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for MSA contiguous scorer coverage",
)
@pytest.mark.parametrize("rows", [4, 257])
def test_msa_contiguous_block_scores_cuda_matches_reference(rows: int) -> None:
    clear_indexer_caches()
    device = torch.device("cuda")
    q_fp8, q_scale, kv_fp8, metadata = _make_msa_contiguous_case(
        rows=rows,
        heads=4,
        k_rows=384,
        device=device,
        seed=92_401 + rows,
    )

    actual = msa_contiguous_block_scores(
        q_fp8=q_fp8,
        q_scale=q_scale,
        kv_fp8=kv_fp8,
        metadata=metadata,
    )
    expected = msa_contiguous_block_scores_reference(
        q_fp8=q_fp8,
        q_scale=q_scale,
        kv_fp8=kv_fp8,
        k_start=metadata.k_start,
        k_end=metadata.k_end,
    )

    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
    assert actual.shape[0] == 4


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for MSA contiguous graph coverage",
)
def test_msa_contiguous_block_scores_graph_replay_tracks_live_weights() -> None:
    clear_indexer_caches()
    device = torch.device("cuda")
    rows = 257
    heads = 4
    k_rows = 384
    q_fp8, q_scale, kv_fp8, metadata = _make_msa_contiguous_case(
        rows=rows,
        heads=heads,
        k_rows=k_rows,
        device=device,
        seed=92_758,
    )
    plan = plan_indexer_contiguous_scratch(
        B12XIndexerContiguousScratchCaps(
            device=device,
            num_q_heads=1,
            num_idx_heads=heads,
            max_q_rows=rows,
            max_k_rows=k_rows,
            topk=16,
            score_mode="msa",
        )
    )
    binding = plan.bind_msa(
        scratch=_one_scratch(plan),
        k_start=metadata.k_start,
        k_end=metadata.k_end,
    )
    scratch = binding.scratch
    k_quant, k_scale = kv_fp8
    scratch.k_quant[:k_rows].copy_(k_quant)
    scratch.k_scale[:k_rows].copy_(k_scale)
    scratch.prepare_k_padding(k_rows=k_rows)
    bound_k_quant = scratch.k_quant[:k_rows]
    bound_k_scale = scratch.k_scale[:k_rows]
    weights = (q_scale * MSA_SM_SCALE).contiguous()
    live_q_scale = (q_scale * 1.25).contiguous()
    q_bytes = q_fp8.view(torch.uint8)
    q_u32 = q_bytes.view(torch.uint32).view(rows, heads, 128 // 4)
    assert binding.block_scores is not None
    block_scores = binding.block_scores

    def run_staged() -> torch.Tensor:
        return run_contiguous_block_scores_kernel(
            q_fp8=q_fp8,
            weights=weights,
            k_quant=bound_k_quant,
            k_scale=bound_k_scale,
            k_start=metadata.k_start,
            k_end=metadata.k_end,
            block_scores=block_scores,
            num_blocks_out=int(block_scores.shape[2]),
            q_u32=q_u32,
            q_bytes=q_bytes,
            weights_kernel=weights,
            k_quant_bytes=scratch.k_quant.view(torch.uint8),
            k_scale_kernel=scratch.k_scale,
            k_start_kernel=metadata.k_start,
            k_end_kernel=metadata.k_end,
            out_kernel=scratch.dummy_logits,
            tile_logits_kernel=scratch.tile_logits,
            k_tma_prefill_desc_ptrs=scratch.k_tma_prefill_desc_ptrs,
        )

    run_staged()
    torch.cuda.synchronize(device)
    warm_compile_misses = compile_cache_info()["compile_misses"]

    with kernel_resolution_guard('MSA contiguous graph replay must use the warmed kernel'):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured_out = run_staged()

    assert captured_out.data_ptr() == block_scores.data_ptr()
    graph.replay()
    torch.cuda.synchronize(device)
    expected0 = msa_contiguous_block_scores_reference(
        q_fp8=q_fp8,
        q_scale=q_scale,
        kv_fp8=(bound_k_quant, bound_k_scale),
        k_start=metadata.k_start,
        k_end=metadata.k_end,
    )
    torch.testing.assert_close(block_scores, expected0, atol=1e-4, rtol=1e-4)

    q_scale.copy_(live_q_scale)
    weights.copy_(q_scale * MSA_SM_SCALE)
    graph.replay()
    torch.cuda.synchronize(device)
    expected1 = msa_contiguous_block_scores_reference(
        q_fp8=q_fp8,
        q_scale=q_scale,
        kv_fp8=(bound_k_quant, bound_k_scale),
        k_start=metadata.k_start,
        k_end=metadata.k_end,
    )
    torch.testing.assert_close(block_scores, expected1, atol=1e-4, rtol=1e-4)
    assert compile_cache_info()["compile_misses"] == warm_compile_misses
