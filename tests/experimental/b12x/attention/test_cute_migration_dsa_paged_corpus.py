"""Exact GPU/graph corpus for the DSA paged, fused, and persistent kernels.

Each test captures one production CUTLASS specialization, replays two changed
live scenarios, and compares the replayed result with an independent GPU
oracle.  Outputs and scratch are caller-owned except for the scheduled
TOKEN_LOGITS API, whose production wrapper currently creates one fixed-capacity
graph-owned output during capture.  Those tests prove that address stays stable
and that replay performs no caching-allocator allocation; caller-owned scheduled
output plumbing remains a separate serving-API gap.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
import torch

from b12x._lib.runtime_control import kernel_resolution_guard
from b12x.attention.dsa_indexer._impl import (
    build_paged_mqa_schedule_metadata,
    clear_indexer_caches,
)
from b12x.attention.dsa_indexer.fused_indexer import (
    _COOP_STATE_WORDS,
    fused_indexer_scratch_capacity,
    run_fused_paged_indexer,
)
from b12x.attention.dsa_indexer.kernel import (
    _split_index_k_cache_runtime_views,
    build_indexer_paged_logits_kernel_binding,
    build_indexer_paged_supertile_logits_kernel_binding,
    build_indexer_paged_tiled_logits_kernel_binding,
)
from b12x.attention.dsa_indexer.persistent_topk import (
    persistent_topk2048_scratch_nbytes,
    run_persistent_topk2048,
)
from b12x.attention.dsa_indexer.reference import (
    pack_index_k_cache_reference,
    paged_decode_logits_reference,
)
from b12x._lib.compiler import compile_cache_info


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for the CUTLASS migration corpus",
)

_PAGE_SIZE = 64
_HEAD_DIM = 128
_TILE_Q = 32
_TILE_K = 512
_ALLOCATOR_COUNTERS = (
    "allocation.all.allocated",
    "allocation.all.freed",
    "segment.all.allocated",
    "segment.all.freed",
    "num_alloc_retries",
    "num_ooms",
)


def _rand_fp8(
    shape: tuple[int, ...],
    *,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    return (torch.randn(shape, generator=generator, dtype=torch.float32) / 3).to(
        device=device, dtype=torch.float8_e4m3fn
    )


def _rand_positive_weights(
    shape: tuple[int, ...],
    *,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    return (torch.rand(shape, generator=generator, dtype=torch.float32) + 0.25).to(
        device=device
    )


def _packed_k_cache(
    num_pages: int,
    *,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    source = torch.randn(
        (num_pages * _PAGE_SIZE, _HEAD_DIM),
        generator=generator,
        dtype=torch.float32,
    ).to(device=device)
    return pack_index_k_cache_reference(source / 3)


def _page_table(
    *,
    rows: int,
    max_pages: int,
    active_pages: tuple[int, ...],
    cache_pages: int,
    offset: int,
    device: torch.device,
) -> torch.Tensor:
    assert len(active_pages) == rows
    table = torch.full(
        (rows, max_pages),
        -1,
        dtype=torch.int32,
        device=device,
    )
    for row, count in enumerate(active_pages):
        page_ids = (
            torch.arange(count, dtype=torch.int32, device=device)
            + offset
            + row * (max(active_pages) + 3)
        ) % cache_pages
        table[row, :count].copy_(page_ids)
    return table


def _capture_once(
    run: Callable[[], Any],
    *,
    device: torch.device,
    reason: str,
) -> tuple[torch.cuda.CUDAGraph, Any]:
    warm_output = run()
    torch.cuda.synchronize(device)
    warm_compile_misses = int(compile_cache_info()["compile_misses"])
    del warm_output

    with kernel_resolution_guard(reason):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured_output = run()
    torch.cuda.synchronize(device)
    assert int(compile_cache_info()["compile_misses"]) == warm_compile_misses
    return graph, captured_output


def _allocator_counters(device: torch.device) -> dict[str, int]:
    stats = torch.cuda.memory_stats(device)
    return {name: int(stats.get(name, 0)) for name in _ALLOCATOR_COUNTERS}


def _replay_without_allocator_activity(
    graph: torch.cuda.CUDAGraph,
    *,
    device: torch.device,
) -> None:
    before = _allocator_counters(device)
    graph.replay()
    torch.cuda.synchronize(device)
    after = _allocator_counters(device)
    assert after == before


def _paged_reference(
    *,
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    index_k_cache: torch.Tensor,
    page_table: torch.Tensor,
    seqlens: torch.Tensor,
) -> torch.Tensor:
    rows = int(q_fp8.shape[0])
    return paged_decode_logits_reference(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        real_page_table=page_table,
        query_row_to_batch=torch.arange(
            rows,
            dtype=torch.int32,
            device=q_fp8.device,
        ),
        seqlens_per_query=seqlens,
    )


def _assert_valid_logits(
    actual: torch.Tensor,
    expected: torch.Tensor,
    seqlens: torch.Tensor,
    *,
    require_invalid_neginf: bool,
) -> None:
    for row in range(int(seqlens.numel())):
        length = int(seqlens[row].item())
        actual_valid = actual[row, :length]
        expected_valid = expected[row, :length]
        assert bool(torch.isfinite(actual_valid).all().item())
        assert bool((actual_valid != 0).any().item())
        torch.testing.assert_close(actual_valid, expected_valid, atol=1e-4, rtol=1e-4)
        if require_invalid_neginf:
            assert bool(torch.isneginf(actual[row, length:]).all().item())


def _copy_paged_scenario(
    *,
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    page_table: torch.Tensor,
    seqlens: torch.Tensor,
    active_width: torch.Tensor,
    scenario: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    scenario_q, scenario_weights, scenario_table, scenario_lengths = scenario
    q_fp8.copy_(scenario_q)
    weights.copy_(scenario_weights)
    page_table.copy_(scenario_table)
    seqlens.copy_(scenario_lengths)
    active_width.fill_(int(scenario_lengths.max().item()))


def _make_paged_scenarios(
    *,
    rows: int,
    heads: int,
    max_pages: int,
    cache_pages: int,
    lengths_a: tuple[int, ...],
    lengths_b: tuple[int, ...],
    generator: torch.Generator,
    device: torch.device,
) -> tuple[
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
]:
    def make(
        lengths: tuple[int, ...],
        *,
        offset: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        active_pages = tuple(
            (length + _PAGE_SIZE - 1) // _PAGE_SIZE for length in lengths
        )
        return (
            _rand_fp8((rows, heads, _HEAD_DIM), generator=generator, device=device),
            _rand_positive_weights((rows, heads), generator=generator, device=device),
            _page_table(
                rows=rows,
                max_pages=max_pages,
                active_pages=active_pages,
                cache_pages=cache_pages,
                offset=offset,
                device=device,
            ),
            torch.tensor(lengths, dtype=torch.int32, device=device),
        )

    return make(lengths_a, offset=5), make(lengths_b, offset=37)


def _capture_prototype(
    scenario: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Make capture-time values distinct from both post-capture scenarios."""
    scenario_q, scenario_weights, scenario_table, scenario_lengths = scenario
    prototype_lengths = scenario_lengths - 17
    prototype_table = scenario_table.clone()
    for row in range(int(prototype_lengths.numel())):
        active_pages = (
            int(prototype_lengths[row].item()) + _PAGE_SIZE - 1
        ) // _PAGE_SIZE
        prototype_table[row, :active_pages].copy_(
            torch.roll(prototype_table[row, :active_pages], shifts=1)
        )
    return (
        (-scenario_q.to(torch.float32)).to(torch.float8_e4m3fn),
        torch.flip(scenario_weights, dims=(1,)),
        prototype_table,
        prototype_lengths,
    )


def test_dsa_paged_base_graph_live_gpu_oracle() -> None:
    """Unscheduled tiled scorer: one DSAPagedLogitsKernel specialization."""
    device = torch.device("cuda")
    generator = torch.Generator(device="cpu").manual_seed(83_101)
    rows, heads, max_pages, cache_pages = 4, 8, 8, 96
    index_k_cache = _packed_k_cache(cache_pages, generator=generator, device=device)
    scenarios = _make_paged_scenarios(
        rows=rows,
        heads=heads,
        max_pages=max_pages,
        cache_pages=cache_pages,
        lengths_a=(509, 447, 383, 319),
        lengths_b=(257, 191, 127, 63),
        generator=generator,
        device=device,
    )
    q_fp8 = torch.empty_like(scenarios[0][0])
    weights = torch.empty_like(scenarios[0][1])
    page_table = torch.empty_like(scenarios[0][2])
    seqlens = torch.empty_like(scenarios[0][3])
    active_width = torch.empty((1,), dtype=torch.int32, device=device)
    tile_logits = torch.empty((_TILE_Q * _TILE_K,), dtype=torch.float32, device=device)
    _copy_paged_scenario(
        q_fp8=q_fp8,
        weights=weights,
        page_table=page_table,
        seqlens=seqlens,
        active_width=active_width,
        scenario=_capture_prototype(scenarios[0]),
    )
    binding = build_indexer_paged_tiled_logits_kernel_binding(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        real_page_table=page_table,
        seqlens_per_query=seqlens,
        active_width=active_width,
        tile_logits=tile_logits,
        tile_block_q=_TILE_Q,
        tile_block_k=_TILE_K,
        preinitialize_tile_logits=False,
    )

    clear_indexer_caches()
    graph, captured = _capture_once(
        binding.run,
        device=device,
        reason="DSA paged-base migration replay must use the warmed specialization",
    )
    assert captured.data_ptr() == tile_logits.data_ptr()
    output_ptr = tile_logits.data_ptr()

    for scenario in scenarios:
        _copy_paged_scenario(
            q_fp8=q_fp8,
            weights=weights,
            page_table=page_table,
            seqlens=seqlens,
            active_width=active_width,
            scenario=scenario,
        )
        tile_logits.fill_(float("nan"))
        _replay_without_allocator_activity(graph, device=device)
        assert tile_logits.data_ptr() == output_ptr
        actual = tile_logits.view(_TILE_Q, _TILE_K)[:rows]
        expected = _paged_reference(
            q_fp8=q_fp8,
            weights=weights,
            index_k_cache=index_k_cache,
            page_table=page_table,
            seqlens=seqlens,
        )
        _assert_valid_logits(
            actual,
            expected,
            seqlens,
            require_invalid_neginf=False,
        )


def _run_scheduled_graph_case(*, rows: int, seed: int) -> None:
    device = torch.device("cuda")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    heads, max_pages, cache_pages = 8, 1024, 160
    if rows == 1:
        lengths_a = (4093,)
        lengths_b = (3071,)
    else:
        lengths_a = (4093, 3581)
        lengths_b = (2815, 3327)
    index_k_cache = _packed_k_cache(cache_pages, generator=generator, device=device)
    scenarios = _make_paged_scenarios(
        rows=rows,
        heads=heads,
        max_pages=max_pages,
        cache_pages=cache_pages,
        lengths_a=lengths_a,
        lengths_b=lengths_b,
        generator=generator,
        device=device,
    )
    q_fp8 = torch.empty_like(scenarios[0][0])
    weights = torch.empty_like(scenarios[0][1])
    page_table = torch.empty_like(scenarios[0][2])
    seqlens = torch.empty_like(scenarios[0][3])
    active_width = torch.empty((1,), dtype=torch.int32, device=device)
    schedule = torch.empty((9, 2), dtype=torch.int32, device=device)

    def install(
        scenario: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        _copy_paged_scenario(
            q_fp8=q_fp8,
            weights=weights,
            page_table=page_table,
            seqlens=seqlens,
            active_width=active_width,
            scenario=scenario,
        )
        build_paged_mqa_schedule_metadata(
            seqlens,
            _PAGE_SIZE,
            8,
            out=schedule,
        )

    install(_capture_prototype(scenarios[0]))
    binding = build_indexer_paged_logits_kernel_binding(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        real_page_table=page_table,
        seqlens_per_query=seqlens,
        schedule_metadata=schedule,
        active_width=active_width,
        preinitialize_invalid_logits=True,
    )

    clear_indexer_caches()
    graph, captured = _capture_once(
        binding.run,
        device=device,
        reason="scheduled DSA migration replay must use the warmed specialization",
    )
    output_ptr = captured.data_ptr()
    assert tuple(captured.shape) == (rows, max_pages * _PAGE_SIZE)

    for scenario in scenarios:
        install(scenario)
        captured.fill_(float("nan"))
        _replay_without_allocator_activity(graph, device=device)
        assert captured.data_ptr() == output_ptr
        expected = _paged_reference(
            q_fp8=q_fp8,
            weights=weights,
            index_k_cache=index_k_cache,
            page_table=page_table,
            seqlens=seqlens,
        )
        _assert_valid_logits(
            captured,
            expected,
            seqlens,
            require_invalid_neginf=True,
        )


def test_dsa_paged_scheduled_single_graph_live_gpu_oracle() -> None:
    """Long one-row decode: one scheduled-single specialization."""
    _run_scheduled_graph_case(rows=1, seed=83_201)


def test_dsa_paged_scheduled_multi_graph_live_gpu_oracle() -> None:
    """Long two-row decode: one scheduled-multi specialization."""
    _run_scheduled_graph_case(rows=2, seed=83_301)


def test_dsa_paged_stream_graph_live_gpu_oracle(monkeypatch) -> None:
    """Streamed supertile scorer without the tiled-top-k selector helper."""
    monkeypatch.setenv("B12X_INDEXER_STREAM_SCORER", "1")
    device = torch.device("cuda")
    generator = torch.Generator(device="cpu").manual_seed(83_401)
    rows, heads, max_pages, cache_pages = 2, 64, 8, 96
    index_k_cache = _packed_k_cache(cache_pages, generator=generator, device=device)
    scenarios = _make_paged_scenarios(
        rows=rows,
        heads=heads,
        max_pages=max_pages,
        cache_pages=cache_pages,
        lengths_a=(509, 443),
        lengths_b=(251, 187),
        generator=generator,
        device=device,
    )
    q_fp8 = torch.empty_like(scenarios[0][0])
    weights = torch.empty_like(scenarios[0][1])
    page_table = torch.empty_like(scenarios[0][2])
    seqlens = torch.empty_like(scenarios[0][3])
    active_width = torch.empty((1,), dtype=torch.int32, device=device)
    tile_logits = torch.empty((_TILE_Q * _TILE_K,), dtype=torch.float32, device=device)
    _copy_paged_scenario(
        q_fp8=q_fp8,
        weights=weights,
        page_table=page_table,
        seqlens=seqlens,
        active_width=active_width,
        scenario=_capture_prototype(scenarios[0]),
    )
    binding = build_indexer_paged_supertile_logits_kernel_binding(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        real_page_table=page_table,
        seqlens_per_query=seqlens,
        active_width=active_width,
        tile_logits=tile_logits,
        source_page_offset=0,
        output_width_tokens=_TILE_K,
        tile_block_q=_TILE_Q,
        tile_block_k=_TILE_K,
        preinitialize_tile_logits=False,
    )

    clear_indexer_caches()
    graph, captured = _capture_once(
        binding.run,
        device=device,
        reason="DSA paged-stream migration replay must use the warmed specialization",
    )
    assert captured.data_ptr() == tile_logits.data_ptr()
    output_ptr = tile_logits.data_ptr()

    for scenario in scenarios:
        _copy_paged_scenario(
            q_fp8=q_fp8,
            weights=weights,
            page_table=page_table,
            seqlens=seqlens,
            active_width=active_width,
            scenario=scenario,
        )
        tile_logits.fill_(float("nan"))
        _replay_without_allocator_activity(graph, device=device)
        assert tile_logits.data_ptr() == output_ptr
        actual = tile_logits.view(_TILE_Q, _TILE_K)[:rows]
        expected = _paged_reference(
            q_fp8=q_fp8,
            weights=weights,
            index_k_cache=index_k_cache,
            page_table=page_table,
            seqlens=seqlens,
        )
        _assert_valid_logits(
            actual,
            expected,
            seqlens,
            require_invalid_neginf=False,
        )


def _fused_topk_reference(
    *,
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    k_quant_bytes: torch.Tensor,
    k_scales: torch.Tensor,
    page_table: torch.Tensor,
    seqlens: torch.Tensor,
    topk: int,
) -> tuple[torch.Tensor, list[set[int]]]:
    k_fp8 = k_quant_bytes.view(torch.float8_e4m3fn)
    expected_values: list[torch.Tensor] = []
    expected_indices: list[set[int]] = []
    for row in range(int(q_fp8.shape[0])):
        length = int(seqlens[row].item())
        page_count = (length + _PAGE_SIZE - 1) // _PAGE_SIZE
        pages = page_table[row, :page_count].to(torch.long)
        k_row = k_fp8[pages].reshape(-1, _HEAD_DIM)[:length].to(torch.float32)
        scale_row = k_scales[pages].reshape(-1)[:length]
        scores = torch.einsum(
            "hd,td->ht",
            q_fp8[row].to(torch.float32),
            k_row,
        )
        logits = (torch.relu(scores) * weights[row].unsqueeze(1)).sum(dim=0) * scale_row
        selected = torch.topk(logits, topk, largest=True, sorted=True)
        expected_values.append(selected.values)
        expected_indices.append(set(selected.indices.tolist()))
    return torch.stack(expected_values), expected_indices


def test_dsa_fused_graph_live_gpu_oracle() -> None:
    """Paged score+top-k with one fixed-capacity cooperative-merge policy."""
    device = torch.device("cuda")
    generator = torch.Generator(device="cpu").manual_seed(83_501)
    rows, heads, topk, max_pages, cache_pages = 2, 16, 512, 32, 128
    ctas_per_group = 4
    index_k_cache = _packed_k_cache(cache_pages, generator=generator, device=device)
    k_quant_bytes, k_scales = _split_index_k_cache_runtime_views(index_k_cache)
    scenarios = _make_paged_scenarios(
        rows=rows,
        heads=heads,
        max_pages=max_pages,
        cache_pages=cache_pages,
        lengths_a=(2045, 1917),
        lengths_b=(1533, 1405),
        generator=generator,
        device=device,
    )
    q_fp8 = torch.empty_like(scenarios[0][0])
    weights = torch.empty_like(scenarios[0][1])
    page_table = torch.empty_like(scenarios[0][2])
    seqlens = torch.empty_like(scenarios[0][3])
    unused_active_width = torch.empty((1,), dtype=torch.int32, device=device)
    _copy_paged_scenario(
        q_fp8=q_fp8,
        weights=weights,
        page_table=page_table,
        seqlens=seqlens,
        active_width=unused_active_width,
        scenario=_capture_prototype(scenarios[0]),
    )
    out_indices = torch.empty((rows, topk), dtype=torch.int32, device=device)
    out_values = torch.empty((rows, topk), dtype=torch.float32, device=device)
    pack_elems, state_words = fused_indexer_scratch_capacity(
        rows,
        topk,
        rows * ctas_per_group,
    )
    assert pack_elems == rows * ctas_per_group * topk
    assert state_words == rows * _COOP_STATE_WORDS
    pack_values = torch.empty(pack_elems, dtype=torch.float32, device=device)
    pack_indices = torch.empty(pack_elems, dtype=torch.int32, device=device)
    merge_state = torch.zeros(state_words, dtype=torch.int32, device=device)
    scratch_ptrs = (
        pack_values.data_ptr(),
        pack_indices.data_ptr(),
        merge_state.data_ptr(),
    )

    def run() -> tuple[torch.Tensor, torch.Tensor]:
        return run_fused_paged_indexer(
            q_bytes=q_fp8.view(torch.uint8),
            weights=weights,
            k_quant_bytes=k_quant_bytes,
            k_scales=k_scales,
            real_page_table=page_table,
            seqlens=seqlens,
            num_heads=heads,
            topk=topk,
            out_indices=out_indices,
            out_values=out_values,
            ctas_per_group=ctas_per_group,
            merge_threshold=0,
            pack_values=pack_values,
            pack_indices=pack_indices,
            merge_state=merge_state,
            merge_state_preinitialized=True,
        )

    clear_indexer_caches()
    graph, captured = _capture_once(
        run,
        device=device,
        reason="DSA fused migration replay must use the warmed specialization",
    )
    assert captured[0].data_ptr() == out_indices.data_ptr()
    assert captured[1].data_ptr() == out_values.data_ptr()
    output_ptrs = (out_indices.data_ptr(), out_values.data_ptr())

    for scenario in scenarios:
        _copy_paged_scenario(
            q_fp8=q_fp8,
            weights=weights,
            page_table=page_table,
            seqlens=seqlens,
            active_width=unused_active_width,
            scenario=scenario,
        )
        out_indices.fill_(-1)
        out_values.fill_(float("nan"))
        _replay_without_allocator_activity(graph, device=device)
        assert (out_indices.data_ptr(), out_values.data_ptr()) == output_ptrs
        assert (
            pack_values.data_ptr(),
            pack_indices.data_ptr(),
            merge_state.data_ptr(),
        ) == scratch_ptrs
        expected_values, expected_indices = _fused_topk_reference(
            q_fp8=q_fp8,
            weights=weights,
            k_quant_bytes=k_quant_bytes,
            k_scales=k_scales,
            page_table=page_table,
            seqlens=seqlens,
            topk=topk,
        )
        assert bool(torch.isfinite(out_values).all().item())
        assert bool((out_values != 0).any().item())
        torch.testing.assert_close(
            torch.sort(out_values, dim=1, descending=True).values,
            expected_values,
            atol=1e-2,
            rtol=0,
        )
        for row in range(rows):
            assert set(out_indices[row].tolist()) == expected_indices[row]


def _persistent_topk_reference(
    *,
    logits: torch.Tensor,
    lengths: torch.Tensor,
    page_table: torch.Tensor,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    positions = torch.arange(
        logits.shape[1],
        dtype=torch.int64,
        device=logits.device,
    )
    masked = torch.where(
        positions.unsqueeze(0) < lengths.unsqueeze(1),
        logits,
        torch.full_like(logits, float("-inf")),
    )
    selected = torch.topk(masked, topk, dim=1, largest=True, sorted=False)
    return selected.values, torch.gather(page_table, 1, selected.indices)


def test_dsa_persistent_topk_graph_live_gpu_oracle() -> None:
    """Paged-output persistent radix top-k above the 32K route threshold."""
    device = torch.device("cuda")
    generator = torch.Generator(device="cpu").manual_seed(83_601)
    rows, width, topk = 2, 33_792, 512
    logits = torch.empty((rows, width), dtype=torch.float32, device=device)
    lengths = torch.empty((rows,), dtype=torch.int32, device=device)
    page_table = torch.empty((rows, width), dtype=torch.int32, device=device)
    output = torch.empty((rows, topk), dtype=torch.int32, device=device)
    scratch_nbytes = persistent_topk2048_scratch_nbytes(
        rows,
        width,
        device=device,
    )
    assert scratch_nbytes % torch.empty((), dtype=torch.int32).element_size() == 0
    scratch = torch.empty(
        (scratch_nbytes // torch.empty((), dtype=torch.int32).element_size(),),
        dtype=torch.int32,
        device=device,
    )
    scratch_ptr = scratch.data_ptr()
    scenario_logits = (
        torch.randn(
            (2, rows, width),
            generator=generator,
            dtype=torch.float32,
        )
        .to(device=device)
        .unbind(0)
    )
    scenario_lengths = (
        torch.tensor((33_791, 32_901), dtype=torch.int32, device=device),
        torch.tensor((32_777, 33_333), dtype=torch.int32, device=device),
    )
    logical = torch.arange(width, dtype=torch.int32, device=device)
    scenario_tables = (
        torch.stack((logical + 10_000, torch.flip(logical, dims=(0,)) + 50_000)),
        torch.stack((torch.flip(logical, dims=(0,)) + 90_000, logical + 130_000)),
    )
    prototype_logits = -scenario_logits[0]
    prototype_lengths = scenario_lengths[0] - 17
    prototype_table = torch.roll(scenario_tables[0], shifts=1, dims=1)

    def install(index: int) -> None:
        logits.copy_(scenario_logits[index])
        lengths.copy_(scenario_lengths[index])
        page_table.copy_(scenario_tables[index])

    def run() -> torch.Tensor:
        return run_persistent_topk2048(
            logits,
            lengths,
            page_table_1=page_table,
            output_indices=output,
            scratch=scratch,
            max_seq_len=width,
            topk=topk,
        )

    logits.copy_(prototype_logits)
    lengths.copy_(prototype_lengths)
    page_table.copy_(prototype_table)
    clear_indexer_caches()
    graph, captured = _capture_once(
        run,
        device=device,
        reason="DSA persistent-top-k migration replay must use the warmed specialization",
    )
    assert captured.data_ptr() == output.data_ptr()
    output_ptr = output.data_ptr()

    for index in range(2):
        install(index)
        output.fill_(-1)
        _replay_without_allocator_activity(graph, device=device)
        assert output.data_ptr() == output_ptr
        assert scratch.data_ptr() == scratch_ptr
        expected_values, expected_indices = _persistent_topk_reference(
            logits=logits,
            lengths=lengths,
            page_table=page_table,
            topk=topk,
        )
        assert bool(torch.isfinite(expected_values).all().item())
        assert bool((expected_values != 0).any().item())
        assert bool((output >= 0).all().item())
        for row in range(rows):
            assert set(output[row].tolist()) == set(expected_indices[row].tolist())
