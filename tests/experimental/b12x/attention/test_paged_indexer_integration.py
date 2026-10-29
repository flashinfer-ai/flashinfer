from __future__ import annotations

import pytest
import torch

from b12x.attention.dsa_indexer._impl import clear_indexer_caches
from b12x.attention.dsa_indexer.paged import (
    _paged_indexer_chunk_geometry,
    _plan_two_level_fold,
    index_topk_fp8,
    pack_paged_index_k_cache_reference,
    paged_index_logits_reference,
    prepare_paged_indexer_metadata,
    resolve_replicated_num_q_heads,
)
from b12x.attention.dsa_indexer.scratch import (
    B12XIndexerPagedScratchCaps,
    plan_indexer_paged_scratch,
)
from b12x.attention.dsa_indexer.tiled_topk import run_tiled_topk


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
    return real_page_table.contiguous()


def _rand_fp8_q(
    shape: tuple[int, int, int],
    *,
    gen: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    return (
        torch.randn(shape, generator=gen, dtype=torch.float32).to(device=device) / 2
    ).to(torch.float8_e4m3fn)


def _expected_paged_index_topk(
    *,
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    index_k_cache: torch.Tensor,
    real_page_table: torch.Tensor,
    seqlens: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    logits = paged_index_logits_reference(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        real_page_table=real_page_table,
        query_row_to_batch=torch.arange(
            q_fp8.shape[0],
            dtype=torch.int32,
            device=q_fp8.device,
        ),
        seqlens_per_query=seqlens,
    )
    raw = torch.topk(logits, k=topk, dim=1, largest=True, sorted=False).indices.to(
        torch.int32
    )
    return raw


def _paged_index_logits(
    *,
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    index_k_cache: torch.Tensor,
    real_page_table: torch.Tensor,
    seqlens: torch.Tensor,
) -> torch.Tensor:
    return paged_index_logits_reference(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        real_page_table=real_page_table,
        query_row_to_batch=torch.arange(
            q_fp8.shape[0],
            dtype=torch.int32,
            device=q_fp8.device,
        ),
        seqlens_per_query=seqlens,
    )


def _logical_to_physical(
    logical: torch.Tensor,
    page_table: torch.Tensor,
    *,
    page_size: int = 64,
) -> torch.Tensor:
    safe_logical = logical.clamp_min(0)
    page_cols = torch.div(safe_logical, page_size, rounding_mode="floor").long()
    page_ids = torch.gather(page_table, 1, page_cols)
    physical = page_ids * page_size + torch.remainder(safe_logical, page_size)
    return torch.where(logical >= 0, physical, torch.full_like(physical, -1))


def _bind_paged_indexer(
    *,
    device: torch.device,
    num_heads: int,
    rows: int,
    width_blocks: int,
    topk: int,
    real_page_table: torch.Tensor,
    seqlens: torch.Tensor,
    supertile_k: int = 512,
    active_width: torch.Tensor | None = None,
    schedule_metadata: torch.Tensor | None = None,
    shared_page_table: bool = False,
    route: str = "paged_tiled",
    output_physical_slots: bool = False,
):
    plan = plan_indexer_paged_scratch(
        B12XIndexerPagedScratchCaps(
            device=device,
            num_q_heads=num_heads,
            max_q_rows=rows,
            max_page_table_width=width_blocks,
            topk=topk,
            page_size=64,
            reserve_paged_logits=False,
            paged_tile_logits_k_rows=supertile_k,
            mode="prefill" if shared_page_table else "decode",
            shared_page_table=shared_page_table,
            route=route,
        )
    )
    scratch = [
        torch.empty(shape, dtype=dtype, device=device)
        for shape, dtype in plan.shapes_and_dtypes()
    ]
    return plan.bind(
        scratch=scratch,
        real_page_table=real_page_table,
        cache_seqlens_int32=seqlens,
        active_width=active_width,
        schedule_metadata=schedule_metadata,
        expected_num_q_heads=num_heads,
        shared_page_table=shared_page_table,
        output_physical_slots=output_physical_slots,
    )


def test_resolve_replicated_num_q_heads_for_tensor_parallel() -> None:
    assert (
        resolve_replicated_num_q_heads(global_num_q_heads=64, tensor_parallel_size=2)
        == 64
    )
    assert (
        resolve_replicated_num_q_heads(global_num_q_heads=64, tensor_parallel_size=1)
        == 64
    )
    with pytest.raises(ValueError, match="must be positive"):
        resolve_replicated_num_q_heads(global_num_q_heads=0, tensor_parallel_size=2)
    with pytest.raises(ValueError, match="tensor_parallel_size must be positive"):
        resolve_replicated_num_q_heads(global_num_q_heads=64, tensor_parallel_size=0)


def test_paged_index_decode_rejects_sharded_selector_heads() -> None:
    device = torch.device("cpu")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(91_002)

    replicated_heads = resolve_replicated_num_q_heads(
        global_num_q_heads=64,
        tensor_parallel_size=2,
    )
    real_page_table = torch.tensor([[0]], dtype=torch.int32, device=device)
    seqlens = torch.tensor([1], dtype=torch.int32, device=device)
    binding = _bind_paged_indexer(
        device=device,
        num_heads=replicated_heads,
        rows=1,
        width_blocks=1,
        topk=512,
        real_page_table=real_page_table,
        seqlens=seqlens,
    )
    q_fp8 = _rand_fp8_q((1, 32, 128), gen=gen, device=device)
    weights = torch.randn((1, 32), generator=gen, dtype=torch.float32, device=device)
    index_k_cache = pack_paged_index_k_cache_reference(
        torch.randn((64, 128), generator=gen, dtype=torch.float32, device=device)
    )

    with pytest.raises(ValueError, match="expected indexer head count 64"):
        index_topk_fp8(
            q_fp8=q_fp8,
            weights=weights,
            index_k_cache=index_k_cache,
            topk=512,
            binding=binding,
        )


def test_paged_index_metadata_rejects_clamp_to_one_lengths() -> None:
    real_page_table = torch.full((1, 2), -1, dtype=torch.int32)
    clamped_seqlens = torch.tensor([1], dtype=torch.int32)

    with pytest.raises(ValueError, match="raw unclamped paged-index lengths"):
        prepare_paged_indexer_metadata(
            real_page_table=real_page_table,
            cache_seqlens_int32=clamped_seqlens,
        )


def test_paged_index_plan_binding_keeps_metadata_aliases() -> None:
    device = torch.device("cpu")
    rows = 1
    num_heads = 64
    width_blocks = 16
    q_fp8 = torch.empty(
        (rows, num_heads, 128), dtype=torch.float8_e4m3fn, device=device
    )
    weights = torch.empty((rows, num_heads), dtype=torch.float32, device=device)
    del q_fp8, weights
    real_page_table = torch.empty(
        (rows, width_blocks), dtype=torch.int32, device=device
    )
    seqlens = torch.empty((rows,), dtype=torch.int32, device=device)
    active_width = torch.empty((1,), dtype=torch.int32, device=device)
    schedule = torch.empty((4, 2), dtype=torch.int32, device=device)

    binding = _bind_paged_indexer(
        device=device,
        num_heads=num_heads,
        rows=rows,
        width_blocks=width_blocks,
        topk=512,
        real_page_table=real_page_table,
        seqlens=seqlens,
        active_width=active_width,
        schedule_metadata=schedule,
    )

    assert binding.real_page_table.data_ptr() == real_page_table.data_ptr()
    assert binding.cache_seqlens_int32.data_ptr() == seqlens.data_ptr()
    assert binding.active_width.data_ptr() == active_width.data_ptr()
    assert binding.schedule_metadata.data_ptr() == schedule.data_ptr()


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for graph capture"
)
def test_index_topk_fp8_graph_matches_reference(
    monkeypatch,
) -> None:
    monkeypatch.setenv("B12X_PAGED_INDEX_SUPERTILE_K", "512")

    device = torch.device("cuda")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(91_004)

    rows = 2
    num_heads = 64
    width_blocks = 16
    topk = 512
    graph_real_page_table = torch.full(
        (rows, width_blocks),
        -1,
        dtype=torch.int32,
        device=device,
    )
    graph_seqlens = torch.empty((rows,), dtype=torch.int32, device=device)
    q_fp8 = _rand_fp8_q((rows, num_heads, 128), gen=gen, device=device)
    weights = torch.randn((rows, num_heads), generator=gen, dtype=torch.float32).to(
        device=device
    )
    api_weights = weights.unsqueeze(-1)
    index_k_cache = pack_paged_index_k_cache_reference(
        torch.randn((80 * 64, 128), generator=gen, dtype=torch.float32).to(
            device=device
        )
        / 3
    )
    actual = torch.empty((rows, topk), dtype=torch.int32, device=device)
    actual_scores = torch.empty((rows, topk), dtype=torch.float32, device=device)

    def prepare(page_starts: list[int], seqlens_list: list[int]):
        live_table = _make_real_page_table(
            page_starts=page_starts,
            seqlens=seqlens_list,
            width_blocks=width_blocks,
            device=device,
        )
        graph_real_page_table.copy_(live_table)
        graph_seqlens.copy_(
            torch.tensor(seqlens_list, dtype=torch.int32, device=device)
        )
        return prepare_paged_indexer_metadata(
            real_page_table=graph_real_page_table,
            cache_seqlens_int32=graph_seqlens,
            expected_num_q_heads=num_heads,
            build_schedule=False,
        )

    binding = _bind_paged_indexer(
        device=device,
        num_heads=num_heads,
        rows=rows,
        width_blocks=width_blocks,
        topk=topk,
        real_page_table=graph_real_page_table,
        seqlens=graph_seqlens,
        supertile_k=512,
    )

    clear_indexer_caches()
    metadata = prepare([2, 40], [900, 960])
    assert metadata.real_page_table.data_ptr() == binding.real_page_table.data_ptr()
    index_topk_fp8(
        q_fp8=q_fp8,
        weights=api_weights,
        index_k_cache=index_k_cache,
        topk=topk,
        expected_num_q_heads=num_heads,
        binding=binding,
        out_indices=actual,
        out_scores=actual_scores,
        supertile_k=512,
    )
    torch.cuda.synchronize(device)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        index_topk_fp8(
            q_fp8=q_fp8,
            weights=api_weights,
            index_k_cache=index_k_cache,
            topk=topk,
            expected_num_q_heads=num_heads,
            binding=binding,
            out_indices=actual,
            out_scores=actual_scores,
            supertile_k=512,
        )
    graph.replay()
    torch.cuda.synchronize(device)
    expected_raw0 = _expected_paged_index_topk(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        real_page_table=graph_real_page_table,
        seqlens=graph_seqlens,
        topk=topk,
    )
    assert torch.equal(
        torch.sort(actual, dim=1).values,
        torch.sort(expected_raw0, dim=1).values,
    )
    logits0 = _paged_index_logits(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        real_page_table=graph_real_page_table,
        seqlens=graph_seqlens,
    )
    torch.testing.assert_close(
        actual_scores,
        torch.gather(logits0, 1, actual.to(torch.int64)),
        rtol=0,
        atol=1e-2,
    )

    prepare([4, 8], [640, 768])
    graph.replay()
    torch.cuda.synchronize(device)
    expected_raw1 = _expected_paged_index_topk(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        real_page_table=graph_real_page_table,
        seqlens=graph_seqlens,
        topk=topk,
    )
    assert torch.equal(
        torch.sort(actual, dim=1).values,
        torch.sort(expected_raw1, dim=1).values,
    )
    logits1 = _paged_index_logits(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        real_page_table=graph_real_page_table,
        seqlens=graph_seqlens,
    )
    torch.testing.assert_close(
        actual_scores,
        torch.gather(logits1, 1, actual.to(torch.int64)),
        rtol=0,
        atol=1e-2,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("route", ["paged_fused", "paged_tiled"])
def test_index_topk_fp8_overwrites_every_output_slot(route: str) -> None:
    device = torch.device("cuda")
    generator = torch.Generator(device="cpu").manual_seed(91_010)
    rows, num_heads, width_blocks, topk = 2, 32, 16, 512
    seqlens = torch.tensor([0, 321], dtype=torch.int32, device=device)
    page_table = _make_real_page_table(
        page_starts=[0, 2],
        seqlens=seqlens.cpu().tolist(),
        width_blocks=width_blocks,
        device=device,
    )
    q_fp8 = _rand_fp8_q((rows, num_heads, 128), gen=generator, device=device)
    weights = torch.randn(
        (rows, num_heads), generator=generator, dtype=torch.float32
    ).to(device)
    index_k_cache = pack_paged_index_k_cache_reference(
        torch.randn((32 * 64, 128), generator=generator, dtype=torch.float32).to(device)
        / 3
    )
    sentinel = 0x5A5A5A5A
    output = torch.full((rows, topk), sentinel, dtype=torch.int32, device=device)
    binding = _bind_paged_indexer(
        device=device,
        num_heads=num_heads,
        rows=rows,
        width_blocks=width_blocks,
        topk=topk,
        real_page_table=page_table,
        seqlens=seqlens,
        route=route,
    )

    index_topk_fp8(
        q_fp8=q_fp8,
        weights=weights.unsqueeze(-1),
        index_k_cache=index_k_cache,
        topk=topk,
        expected_num_q_heads=num_heads,
        binding=binding,
        out_indices=output,
        supertile_k=512,
    )
    torch.cuda.synchronize(device)

    assert not torch.any(output == sentinel)
    assert torch.all(output[0] == -1)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for graph capture"
)
def test_paged_tiled_indexer_emits_physical_slots_in_final_fold(
    monkeypatch,
) -> None:
    monkeypatch.setenv("B12X_PAGED_INDEX_SUPERTILE_K", "512")

    device = torch.device("cuda")
    gen = torch.Generator(device="cpu").manual_seed(91_009)
    rows, num_heads, width_blocks, topk = 4, 32, 32, 512
    seqlens = torch.tensor([0, 1472, 1536, 1600], dtype=torch.int32, device=device)
    shared_storage = torch.full((1, width_blocks), -1, dtype=torch.int32, device=device)
    shared_storage[0, :25] = torch.arange(5, 30, dtype=torch.int32, device=device).flip(
        0
    )
    page_table = shared_storage.expand(rows, width_blocks)
    assert page_table.stride(0) == 0

    q_fp8 = _rand_fp8_q((rows, num_heads, 128), gen=gen, device=device)
    weights = torch.randn((rows, num_heads), generator=gen, dtype=torch.float32).to(
        device
    )
    index_k_cache = pack_paged_index_k_cache_reference(
        torch.randn((64 * 64, 128), generator=gen, dtype=torch.float32).to(device) / 3
    )
    actual = torch.empty((rows, topk), dtype=torch.int32, device=device)
    binding = _bind_paged_indexer(
        device=device,
        num_heads=num_heads,
        rows=rows,
        width_blocks=width_blocks,
        topk=topk,
        real_page_table=page_table,
        seqlens=seqlens,
        supertile_k=512,
        shared_page_table=True,
        route="paged_tiled",
        output_physical_slots=True,
    )

    clear_indexer_caches()
    index_topk_fp8(
        q_fp8=q_fp8,
        weights=weights.unsqueeze(-1),
        index_k_cache=index_k_cache,
        topk=topk,
        expected_num_q_heads=num_heads,
        binding=binding,
        out_indices=actual,
        supertile_k=512,
    )
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        index_topk_fp8(
            q_fp8=q_fp8,
            weights=weights.unsqueeze(-1),
            index_k_cache=index_k_cache,
            topk=topk,
            expected_num_q_heads=num_heads,
            binding=binding,
            out_indices=actual,
            supertile_k=512,
        )
    graph.replay()
    torch.cuda.synchronize(device)

    logical = _expected_paged_index_topk(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        real_page_table=page_table,
        seqlens=seqlens,
        topk=topk,
    )
    expected = _logical_to_physical(logical, page_table)
    expected[0].fill_(-1)
    assert torch.equal(
        torch.sort(actual, dim=1).values,
        torch.sort(expected, dim=1).values,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_tiled_topk_physical_slots_use_runtime_page_table_row_stride(
    monkeypatch,
) -> None:
    """A cached cubin must support both narrow and full-capacity page tables."""
    from b12x._lib.compiler import clear_compile_cache, compile_cache_info

    monkeypatch.setenv("B12X_COMPILE_DISK_CACHE", "0")
    monkeypatch.setenv("B12X_COMPILE_MEMORY_CACHE", "1")
    clear_compile_cache()

    device = torch.device("cuda")
    rows, block_q, block_k, topk = 17, 32, 512, 512
    lengths = torch.full((rows,), 32, dtype=torch.int32, device=device)
    page_ids = torch.arange(1000, 1000 + rows, dtype=torch.int32, device=device)
    tile_logits = torch.zeros(
        block_q * block_k,
        dtype=torch.float32,
        device=device,
    )

    def select(
        width: int, *, shared: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        selected_page_ids = page_ids
        if shared:
            selected_page_ids = torch.full_like(page_ids, 2000)
            page_table = torch.full((1, width), -1, dtype=torch.int32, device=device)
            page_table[:, 0] = selected_page_ids[0]
            page_table = page_table.expand(rows, -1)
        else:
            page_table = torch.full((rows, width), -1, dtype=torch.int32, device=device)
            page_table[:, 0] = selected_page_ids
        _, indices = run_tiled_topk(
            tile_logits=tile_logits,
            k_start=None,
            lengths=lengths,
            topk=topk,
            block_q=block_q,
            block_k=block_k,
            num_k_tiles=1,
            zero_row_start=True,
            output_page_table=page_table,
            output_page_size=64,
        )
        torch.cuda.synchronize(device)
        return indices, selected_page_ids

    narrow, narrow_page_ids = select(1)
    misses_after_narrow = compile_cache_info()["compile_misses"]
    wide, wide_page_ids = select(8192)
    shared, shared_page_ids = select(8192, shared=True)
    misses_after_reuse = compile_cache_info()["compile_misses"]

    def expected(selected_page_ids: torch.Tensor) -> torch.Tensor:
        result = torch.full((rows, topk), -1, dtype=torch.int32, device=device)
        result[:, :32] = (
            selected_page_ids[:, None] * 64
            + torch.arange(32, dtype=torch.int32, device=device)[None, :]
        )
        return torch.sort(result, dim=1).values

    assert torch.equal(torch.sort(narrow, dim=1).values, expected(narrow_page_ids))
    assert torch.equal(torch.sort(wide, dim=1).values, expected(wide_page_ids))
    assert torch.equal(torch.sort(shared, dim=1).values, expected(shared_page_ids))
    assert misses_after_reuse == misses_after_narrow


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for graph capture"
)
def test_paged_index_shared_supertile_prefill_graph_matches_reference(
    monkeypatch,
) -> None:
    monkeypatch.setenv("B12X_PAGED_INDEX_SUPERTILE_K", "4096")

    device = torch.device("cuda")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(91_006)

    rows = 1024
    num_heads = 32
    width_blocks = 64
    topk = 512
    supertile_k = 4096
    graph_real_page_table = torch.full(
        (rows, width_blocks),
        -1,
        dtype=torch.int32,
        device=device,
    )
    graph_shared_page_table = torch.full(
        (1, width_blocks),
        -1,
        dtype=torch.int32,
        device=device,
    )
    graph_seqlens = torch.empty((rows,), dtype=torch.int32, device=device)
    q_fp8 = _rand_fp8_q((rows, num_heads, 128), gen=gen, device=device)
    weights = torch.randn((rows, num_heads), generator=gen, dtype=torch.float32).to(
        device=device
    )
    api_weights = weights.unsqueeze(-1)
    index_k_cache = pack_paged_index_k_cache_reference(
        torch.randn((128 * 64, 128), generator=gen, dtype=torch.float32).to(
            device=device
        )
        / 3
    )
    padded_index_k_cache = torch.empty(
        (int(index_k_cache.shape[0]), int(index_k_cache.shape[1]) + 16),
        dtype=index_k_cache.dtype,
        device=device,
    )
    padded_index_k_cache[:, : int(index_k_cache.shape[1])].copy_(index_k_cache)
    index_k_cache = padded_index_k_cache.as_strided(
        tuple(index_k_cache.shape),
        (int(padded_index_k_cache.stride(0)), 1),
    )
    assert not index_k_cache.is_contiguous()
    actual = torch.empty((rows, topk), dtype=torch.int32, device=device)
    expected = torch.empty((rows, topk), dtype=torch.int32, device=device)
    shared_table_binding = graph_shared_page_table.expand(rows, -1)
    binding = _bind_paged_indexer(
        device=device,
        num_heads=num_heads,
        rows=rows,
        width_blocks=width_blocks,
        topk=topk,
        real_page_table=shared_table_binding,
        seqlens=graph_seqlens,
        supertile_k=supertile_k,
        shared_page_table=True,
        route="packed_contiguous",
    )
    reference_binding = _bind_paged_indexer(
        device=device,
        num_heads=num_heads,
        rows=rows,
        width_blocks=width_blocks,
        topk=topk,
        real_page_table=graph_real_page_table,
        seqlens=graph_seqlens,
        supertile_k=supertile_k,
        shared_page_table=False,
    )

    def prepare(page_start: int, seq_len: int, *, shared_page_table: bool):
        base_table = _make_real_page_table(
            page_starts=[page_start],
            seqlens=[seq_len],
            width_blocks=width_blocks,
            device=device,
        )
        if shared_page_table:
            graph_shared_page_table.copy_(base_table)
            real_page_table = shared_table_binding
        else:
            graph_real_page_table.copy_(base_table.expand(rows, -1))
            real_page_table = graph_real_page_table
        graph_seqlens.fill_(seq_len)
        return prepare_paged_indexer_metadata(
            real_page_table=real_page_table,
            cache_seqlens_int32=graph_seqlens,
            expected_num_q_heads=num_heads,
            build_schedule=False,
            shared_page_table=shared_page_table,
        )

    clear_indexer_caches()
    metadata = prepare(3, 4096, shared_page_table=True)
    reference_metadata = prepare(3, 4096, shared_page_table=False)
    assert metadata.real_page_table.data_ptr() == binding.real_page_table.data_ptr()
    assert (
        reference_metadata.real_page_table.data_ptr()
        == reference_binding.real_page_table.data_ptr()
    )
    index_topk_fp8(
        q_fp8=q_fp8,
        weights=api_weights,
        index_k_cache=index_k_cache,
        topk=topk,
        expected_num_q_heads=num_heads,
        binding=reference_binding,
        out_indices=expected,
        supertile_k=supertile_k,
    )
    index_topk_fp8(
        q_fp8=q_fp8,
        weights=api_weights,
        index_k_cache=index_k_cache,
        topk=topk,
        expected_num_q_heads=num_heads,
        binding=binding,
        out_indices=actual,
        supertile_k=supertile_k,
    )
    torch.cuda.synchronize(device)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        index_topk_fp8(
            q_fp8=q_fp8,
            weights=api_weights,
            index_k_cache=index_k_cache,
            topk=topk,
            expected_num_q_heads=num_heads,
            binding=binding,
            out_indices=actual,
            supertile_k=supertile_k,
        )
    graph.replay()
    torch.cuda.synchronize(device)
    assert torch.equal(
        torch.sort(actual, dim=1).values,
        torch.sort(expected, dim=1).values,
    )

    metadata = prepare(24, 3584, shared_page_table=True)
    reference_metadata = prepare(24, 3584, shared_page_table=False)
    index_topk_fp8(
        q_fp8=q_fp8,
        weights=api_weights,
        index_k_cache=index_k_cache,
        topk=topk,
        expected_num_q_heads=num_heads,
        binding=reference_binding,
        out_indices=expected,
        supertile_k=supertile_k,
    )
    graph.replay()
    torch.cuda.synchronize(device)
    assert torch.equal(
        torch.sort(actual, dim=1).values,
        torch.sort(expected, dim=1).values,
    )


def test_paged_index_supertile_scratch_sizes_candidate_carry_buffer() -> None:
    device = torch.device("cpu")
    page_size = 64
    page_table_width = 1056
    supertile_k = 8192
    # The streaming fold replaced the old map+merge candidate slab (one half per
    # supertile chunk) with a fixed two-half carry double-buffer (read prev / write
    # next), so the candidate buffer no longer scales with the chunk count.
    chunk_count = (page_table_width * page_size + supertile_k - 1) // supertile_k
    assert chunk_count > 2  # this config is genuinely multi-chunk

    plan = plan_indexer_paged_scratch(
        B12XIndexerPagedScratchCaps(
            device=device,
            num_q_heads=64,
            max_q_rows=16,
            max_page_table_width=page_table_width,
            topk=512,
            page_size=page_size,
            reserve_paged_logits=False,
            paged_tile_logits_k_rows=supertile_k,
            route="paged_tiled",
        )
    )
    scratch = [
        torch.empty(shape, dtype=dtype, device=device)
        for shape, dtype in plan.shapes_and_dtypes()
    ]
    real_page_table = torch.empty(
        (16, page_table_width), dtype=torch.int32, device=device
    )
    seqlens = torch.empty((16,), dtype=torch.int32, device=device)
    binding = plan.bind(
        scratch=scratch,
        real_page_table=real_page_table,
        cache_seqlens_int32=seqlens,
        expected_num_q_heads=64,
    )
    candidate_values, candidate_indices = (
        binding.scratch.get_indexer_contiguous_candidate_buffers()
    )
    # Fold carry double-buffer: exactly two halves, decoupled from chunk_count.
    assert candidate_values.shape[0] == 2
    assert candidate_indices.shape[0] == 2


def test_paged_index_supertile_plan_records_launch_contract() -> None:
    device = torch.device("cpu")
    plan = plan_indexer_paged_scratch(
        B12XIndexerPagedScratchCaps(
            device=device,
            num_q_heads=64,
            max_q_rows=2,
            max_page_table_width=16,
            topk=512,
            page_size=64,
            reserve_paged_logits=False,
            paged_tile_logits_k_rows=512,
            route="paged_tiled",
        )
    )

    assert plan.layout.route == "paged_tiled"
    assert plan.layout.supertile_tokens == 512
    assert plan.layout.max_chunks == 2
    assert plan.layout.tile_logits_elements == 32 * 512


def test_two_level_fold_auto_respects_candidate_memory_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("B12X_INDEXER_TWO_LEVEL_FOLD", "auto")
    monkeypatch.setenv("B12X_INDEXER_TWO_LEVEL_FOLD_MAX_MIB", "1")
    common = {
        "topk": 512,
        "page_size": 64,
        "page_table_width": 1024,
        "supertile_pages": 256,
        "output_physical_slots": False,
    }

    below = _plan_two_level_fold(q_rows=63, **common)
    above = _plan_two_level_fold(q_rows=64, **common)

    assert below.slices == ((1, 0), (1, 1), (1, 2), (1, 3))
    assert below.total_slices == 4
    assert below.candidate_nbytes == 63 * (4 * 512 * 8 + 4)
    assert below.candidate_nbytes <= 1024 * 1024
    assert below.reason == "within memory budget"
    assert above.slices == ()
    assert above.total_slices == 0
    assert above.candidate_nbytes == 64 * (4 * 512 * 8 + 4)
    assert above.candidate_nbytes > 1024 * 1024
    assert above.reason == "candidate slab exceeds the memory budget"


def test_two_level_fold_forced_mode_ignores_memory_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("B12X_INDEXER_TWO_LEVEL_FOLD", "on")
    monkeypatch.setenv("B12X_INDEXER_TWO_LEVEL_FOLD_MAX_MIB", "0")

    plan = _plan_two_level_fold(
        q_rows=64,
        topk=512,
        page_size=64,
        page_table_width=1024,
        supertile_pages=256,
        output_physical_slots=False,
    )

    assert plan.slices == ((1, 0), (1, 1), (1, 2), (1, 3))
    assert plan.total_slices == 4
    assert plan.reason == "forced"


def test_paged_indexer_chunk_geometry_handles_partial_final_chunk() -> None:
    chunk = _paged_indexer_chunk_geometry(
        chunk_idx=4,
        page_table_width=1025,
        supertile_pages=256,
        page_size=64,
    )

    assert chunk.page_begin == 1024
    assert chunk.page_end == 1025
    assert chunk.page_count == 1
    assert chunk.token_begin == 65536
    assert chunk.token_count == 64


@pytest.mark.parametrize(
    ("output_physical_slots", "width", "mode", "reason"),
    [
        (True, 1024, "auto", "physical output requires streaming carry"),
        (False, 128, "auto", "context is below the two-level threshold"),
        (False, 1024, "off", "two-level fold is disabled"),
    ],
)
def test_two_level_fold_ineligible_routes_use_streaming_carry(
    monkeypatch: pytest.MonkeyPatch,
    output_physical_slots: bool,
    width: int,
    mode: str,
    reason: str,
) -> None:
    monkeypatch.setenv("B12X_INDEXER_TWO_LEVEL_FOLD", mode)

    plan = _plan_two_level_fold(
        q_rows=16,
        topk=512,
        page_size=64,
        page_table_width=width,
        supertile_pages=256,
        output_physical_slots=output_physical_slots,
    )

    assert plan.slices == ()
    assert plan.reason == reason


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("B12X_INDEXER_TWO_LEVEL_FOLD", "sometimes"),
        ("B12X_INDEXER_TWO_LEVEL_FOLD_MAX_MIB", "-1"),
        ("B12X_INDEXER_TWO_LEVEL_FOLD_MAX_MIB", "many"),
    ],
)
def test_two_level_fold_rejects_invalid_policy(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: str,
) -> None:
    monkeypatch.setenv(name, value)

    with pytest.raises(ValueError):
        _plan_two_level_fold(
            q_rows=1,
            topk=512,
            page_size=64,
            page_table_width=1024,
            supertile_pages=256,
            output_physical_slots=False,
        )


def test_two_level_fold_rejects_invalid_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("B12X_INDEXER_TWO_LEVEL_FOLD", "auto")

    with pytest.raises(ValueError, match="q_rows"):
        _plan_two_level_fold(
            q_rows=0,
            topk=512,
            page_size=64,
            page_table_width=1024,
            supertile_pages=256,
            output_physical_slots=False,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_paged_index_two_level_fold_clips_rounded_final_slice() -> None:
    device = torch.device("cuda")
    rows, num_heads, topk = 1, 16, 512
    supertile_k = 33 * 512
    width_blocks = 2 * (supertile_k // 64)
    width_tokens = width_blocks * 64

    seqlens = torch.full((rows,), width_tokens, dtype=torch.int32, device=device)
    page_table = torch.arange(width_blocks, dtype=torch.int32, device=device).unsqueeze(
        0
    )
    q_fp8 = torch.zeros(
        (rows, num_heads, 128), dtype=torch.float8_e4m3fn, device=device
    )
    q_fp8[:, 0, 0] = 1
    weights = torch.zeros((rows, num_heads), dtype=torch.float32, device=device)
    weights[:, 0] = -1
    k = torch.zeros((width_tokens, 128), dtype=torch.float32, device=device)
    k[:, 0] = torch.linspace(1, 2, width_tokens, device=device)
    index_k_cache = pack_paged_index_k_cache_reference(k)
    actual = torch.empty((rows, topk), dtype=torch.int32, device=device)
    binding = _bind_paged_indexer(
        device=device,
        num_heads=num_heads,
        rows=rows,
        width_blocks=width_blocks,
        topk=topk,
        real_page_table=page_table,
        seqlens=seqlens,
        supertile_k=supertile_k,
    )

    index_topk_fp8(
        q_fp8=q_fp8,
        weights=weights.unsqueeze(-1),
        index_k_cache=index_k_cache,
        topk=topk,
        expected_num_q_heads=num_heads,
        binding=binding,
        out_indices=actual,
        supertile_k=supertile_k,
    )
    torch.cuda.synchronize(device)

    expected = _expected_paged_index_topk(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        real_page_table=page_table,
        seqlens=seqlens,
        topk=topk,
    )
    assert torch.equal(
        torch.sort(actual, dim=1).values,
        torch.sort(expected, dim=1).values,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for graph capture"
)
def test_index_topk_fp8_graph_unaligned_single_chunk(
    monkeypatch,
) -> None:
    monkeypatch.setenv("B12X_PAGED_INDEX_SUPERTILE_K", "1536")

    device = torch.device("cuda")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(91_005)

    rows = 2
    num_heads = 64
    width_blocks = 17
    supertile_blocks = 24
    topk = 512
    graph_real_page_table = torch.full(
        (rows, width_blocks),
        -1,
        dtype=torch.int32,
        device=device,
    )
    graph_seqlens = torch.empty((rows,), dtype=torch.int32, device=device)
    q_fp8 = _rand_fp8_q((rows, num_heads, 128), gen=gen, device=device)
    weights = torch.randn((rows, num_heads), generator=gen, dtype=torch.float32).to(
        device=device
    )
    api_weights = weights.unsqueeze(-1)
    index_k_cache = pack_paged_index_k_cache_reference(
        torch.randn((96 * 64, 128), generator=gen, dtype=torch.float32).to(
            device=device
        )
        / 3
    )
    actual = torch.empty((rows, topk), dtype=torch.int32, device=device)

    def prepare(page_starts: list[int], seqlens_list: list[int]):
        live_table = _make_real_page_table(
            page_starts=page_starts,
            seqlens=seqlens_list,
            width_blocks=width_blocks,
            device=device,
        )
        graph_real_page_table.copy_(live_table)
        graph_seqlens.copy_(
            torch.tensor(seqlens_list, dtype=torch.int32, device=device)
        )
        return prepare_paged_indexer_metadata(
            real_page_table=graph_real_page_table,
            cache_seqlens_int32=graph_seqlens,
            expected_num_q_heads=num_heads,
            build_schedule=False,
        )

    binding = _bind_paged_indexer(
        device=device,
        num_heads=num_heads,
        rows=rows,
        width_blocks=supertile_blocks,
        topk=topk,
        real_page_table=graph_real_page_table,
        seqlens=graph_seqlens,
        supertile_k=1536,
    )
    clear_indexer_caches()
    metadata = prepare([2, 48], [960, 1024])
    assert metadata.real_page_table.data_ptr() == binding.real_page_table.data_ptr()
    index_topk_fp8(
        q_fp8=q_fp8,
        weights=api_weights,
        index_k_cache=index_k_cache,
        topk=topk,
        expected_num_q_heads=num_heads,
        binding=binding,
        out_indices=actual,
        supertile_k=1536,
    )
    torch.cuda.synchronize(device)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        index_topk_fp8(
            q_fp8=q_fp8,
            weights=api_weights,
            index_k_cache=index_k_cache,
            topk=topk,
            expected_num_q_heads=num_heads,
            binding=binding,
            out_indices=actual,
            supertile_k=1536,
        )
    graph.replay()
    torch.cuda.synchronize(device)
    expected_raw0 = _expected_paged_index_topk(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        real_page_table=graph_real_page_table,
        seqlens=graph_seqlens,
        topk=topk,
    )
    assert torch.equal(
        torch.sort(actual, dim=1).values,
        torch.sort(expected_raw0, dim=1).values,
    )

    prepare([4, 12], [640, 704])
    graph.replay()
    torch.cuda.synchronize(device)
    expected_raw1 = _expected_paged_index_topk(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        real_page_table=graph_real_page_table,
        seqlens=graph_seqlens,
        topk=topk,
    )
    assert torch.equal(
        torch.sort(actual, dim=1).values,
        torch.sort(expected_raw1, dim=1).values,
    )
