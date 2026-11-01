"""Page-size-32 paged indexer parity tests (GLM-5.3 kpool serving profile).

The kpool pool cache allocates 32-pool pages while the paged indexer was
historically 64-only. These tests pin the packed-contiguous shared-prefill
route at page_size=32 against the pure-torch paged reference, exercise the
DCP2 sharded-pool contract (per-rank local top-k + exact cross-rank merge ==
global top-k), and cover the 64-bit big-page-id addressing rule from
AGENTS.md for the triton supertile gather.
"""

import pytest
import torch

from b12x.attention.nsa_indexer import (
    PAGED_INDEX_PAGE_SIZES,
    index_topk_fp8,
)
from b12x.attention.nsa_indexer.paged import (
    pack_paged_index_k_cache_reference,
    paged_index_logits_reference,
)
from b12x.attention.nsa_indexer.scratch import (
    B12XIndexerPagedScratchCaps,
    plan_indexer_paged_scratch,
)

_PAGE = 32
_NUM_HEADS = 16
_HEAD_DIM = 128


def _rand_fp8_q(
    shape: tuple[int, int, int],
    *,
    gen: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    return (
        torch.randn(shape, generator=gen, dtype=torch.float32).to(device=device) / 2
    ).to(torch.float8_e4m3fn)


def _bind_paged_indexer(
    *,
    device: torch.device,
    num_heads: int,
    rows: int,
    width_blocks: int,
    topk: int,
    real_page_table: torch.Tensor,
    seqlens: torch.Tensor,
    supertile_k: int = 4096,
    shared_page_table: bool = True,
):
    plan = plan_indexer_paged_scratch(
        B12XIndexerPagedScratchCaps(
            device=device,
            num_q_heads=num_heads,
            max_q_rows=rows,
            max_page_table_width=width_blocks,
            topk=topk,
            page_size=_PAGE,
            reserve_paged_logits=False,
            paged_tile_logits_k_rows=supertile_k,
            mode="prefill" if shared_page_table else "decode",
            shared_page_table=shared_page_table,
            route="packed_contiguous" if shared_page_table else "paged_tiled",
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
        expected_num_q_heads=num_heads,
        shared_page_table=shared_page_table,
    )


def _reference_topk(
    *,
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    index_k_cache: torch.Tensor,
    real_page_table: torch.Tensor,
    seqlens: torch.Tensor,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dense reference logits + exact top-(values, indices) per row."""
    logits = paged_index_logits_reference(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        real_page_table=real_page_table,
        query_row_to_batch=torch.arange(
            q_fp8.shape[0], dtype=torch.int32, device=q_fp8.device
        ),
        seqlens_per_query=seqlens,
        page_size=_PAGE,
    )
    values, indices = torch.topk(logits, k=min(topk, logits.shape[1]), dim=1)
    return values, indices.to(torch.int32)


def _assert_topk_parity(
    *,
    got_indices: torch.Tensor,
    got_scores: torch.Tensor,
    ref_values: torch.Tensor,
    ref_indices: torch.Tensor,
    rtol: float = 2e-3,
) -> None:
    """Exact-indices-or-tie tolerance against the dense reference top-k."""
    assert got_scores.dtype == torch.float32
    order = torch.argsort(got_scores, dim=1, descending=True)
    got_sorted = torch.gather(got_scores, 1, order)
    k = ref_values.shape[1]
    torch.testing.assert_close(
        got_sorted[:, :k],
        ref_values,
        rtol=rtol,
        atol=rtol * max(1.0, float(ref_values.abs().max())),
    )
    for row in range(got_indices.shape[0]):
        boundary = float(ref_values[row, -1])
        eps = rtol * max(1.0, abs(boundary))
        ref_allowed = set(ref_indices[row][ref_values[row] >= boundary - eps].tolist())
        got_set = set(
            int(i)
            for i, s in zip(
                got_indices[row].tolist(), got_scores[row].tolist(), strict=True
            )
            if s > boundary - eps and i >= 0
        )
        assert len(got_set) == len(ref_allowed), (
            f"row {row}: selected {len(got_set)} candidates above the reference "
            f"boundary, reference allows {len(ref_allowed)}"
        )
        assert got_set == ref_allowed, f"row {row}: index sets disagree"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_page32_packed_contiguous_matches_reference(monkeypatch) -> None:
    monkeypatch.setenv("B12X_PAGED_INDEX_SUPERTILE_K", "2048")
    device = torch.device("cuda")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(202_609)

    rows = 128
    seqlen = 4096  # pools
    topk = 512
    width_blocks = seqlen // _PAGE

    q_fp8 = _rand_fp8_q((rows, _NUM_HEADS, _HEAD_DIM), gen=gen, device=device)
    weights = torch.randn((rows, _NUM_HEADS), generator=gen).to(device)
    k_matrix = (torch.randn((seqlen, _HEAD_DIM), generator=gen) / 3).to(device)
    # Per-pool magnitude spread keeps the reference top-k separated well past
    # kernel-vs-reference accumulation noise.
    k_matrix = k_matrix * (1.0 + torch.arange(seqlen, device=device)[:, None] / seqlen)
    index_k_cache = pack_paged_index_k_cache_reference(k_matrix, page_size=_PAGE)

    page_table = torch.arange(width_blocks, dtype=torch.int32, device=device)
    page_table = page_table.view(1, width_blocks).expand(rows, width_blocks)
    seqlens_t = torch.full((rows,), seqlen, dtype=torch.int32, device=device)

    binding = _bind_paged_indexer(
        device=device,
        num_heads=_NUM_HEADS,
        rows=rows,
        width_blocks=width_blocks,
        topk=topk,
        real_page_table=page_table,
        seqlens=seqlens_t,
        supertile_k=2048,
    )
    out_indices = torch.empty((rows, topk), dtype=torch.int32, device=device)
    out_scores = torch.empty((rows, topk), dtype=torch.float32, device=device)
    index_topk_fp8(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        binding=binding,
        page_size=_PAGE,
        expected_num_q_heads=_NUM_HEADS,
        out_indices=out_indices,
        out_scores=out_scores,
    )
    torch.cuda.synchronize(device)

    ref_values, ref_indices = _reference_topk(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        real_page_table=page_table,
        seqlens=seqlens_t,
        topk=topk,
    )
    _assert_topk_parity(
        got_indices=out_indices,
        got_scores=out_scores,
        ref_values=ref_values,
        ref_indices=ref_indices,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_page32_sharded_local_topk_merge_matches_global_reference(
    monkeypatch,
) -> None:
    """DCP2 contract: per-rank local top-k + exact merge == global top-k.

    Simulates the vllm kpool DCP route: each rank holds an interleaved shard
    of the pool context (cp_kv_cache_interleave_size=1: global pool g lives on
    rank g %% world), runs a paged top-k over its LOCAL pool cache, and the
    ranks merge their candidates by score (stable, rank-major) and truncate.
    """
    monkeypatch.setenv("B12X_PAGED_INDEX_SUPERTILE_K", "1024")
    device = torch.device("cuda")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(313_131)

    world = 2
    rows = 32
    select_k = 256
    work_k = 512
    total_pools = 2048  # global pool context (padded pool granularity)
    local_pools = total_pools // world
    width_blocks = local_pools // _PAGE

    q_fp8 = _rand_fp8_q((rows, _NUM_HEADS, _HEAD_DIM), gen=gen, device=device)
    weights = torch.randn((rows, _NUM_HEADS), generator=gen).to(device)
    k_global = (torch.randn((total_pools, _HEAD_DIM), generator=gen) / 3).to(device)
    k_global = k_global * (
        1.0 + torch.arange(total_pools, device=device)[:, None] / total_pools
    )

    # Per-row GLOBAL context length (pools): each query row attends to
    # row_ctx[r] pools of the global sequence.
    row_ctx = torch.arange(
        local_pools, local_pools + rows, dtype=torch.int64, device=device
    )

    # Global reference: one page table over the global pool sequence.
    global_width_blocks = total_pools // _PAGE
    global_table = (
        torch.arange(global_width_blocks, dtype=torch.int32, device=device)
        .view(1, global_width_blocks)
        .expand(rows, global_width_blocks)
    )
    global_cache = pack_paged_index_k_cache_reference(k_global, page_size=_PAGE)
    ref_values, ref_indices = _reference_topk(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=global_cache,
        real_page_table=global_table,
        seqlens=row_ctx.to(torch.int32),
        topk=select_k,
    )

    # Per-rank local shard: local pool position p maps to global pool
    # p * world + rank (interleave size 1).
    local_positions = torch.arange(local_pools, dtype=torch.int64, device=device)
    local_len = torch.full((rows,), local_pools, dtype=torch.int32, device=device)
    per_rank: list[tuple[torch.Tensor, torch.Tensor]] = []
    for rank in range(world):
        local_rows = k_global[local_positions * world + rank]
        local_cache = pack_paged_index_k_cache_reference(local_rows, page_size=_PAGE)
        local_table = (
            torch.arange(width_blocks, dtype=torch.int32, device=device)
            .view(1, width_blocks)
            .expand(rows, width_blocks)
        )
        # Local causal length for a row with global context C: number of local
        # positions p with p * world + rank < C.
        local_ctx = (row_ctx - rank + world - 1) // world
        local_ctx = local_ctx.clamp(min=0, max=local_pools)
        binding = _bind_paged_indexer(
            device=device,
            num_heads=_NUM_HEADS,
            rows=rows,
            width_blocks=width_blocks,
            topk=work_k,
            real_page_table=local_table,
            seqlens=local_ctx.to(torch.int32),
            supertile_k=1024,
        )
        out_indices = torch.empty((rows, work_k), dtype=torch.int32, device=device)
        out_scores = torch.empty((rows, work_k), dtype=torch.float32, device=device)
        index_topk_fp8(
            q_fp8=q_fp8,
            weights=weights,
            index_k_cache=local_cache,
            binding=binding,
            page_size=_PAGE,
            expected_num_q_heads=_NUM_HEADS,
            out_indices=out_indices,
            out_scores=out_scores,
        )
        per_rank.append((out_indices, out_scores))
    torch.cuda.synchronize(device)

    # Exact cross-rank merge (vllm _merge_b12x_dcp_topk semantics): local ->
    # global ids, stable lexsort by (score desc, candidate position asc),
    # truncate to select_k.
    cand_scores = []
    cand_indices = []
    for rank, (idx, scores) in enumerate(per_rank):
        global_ids = idx.to(torch.int64) * world + rank
        global_ids = torch.where(idx >= 0, global_ids, torch.full_like(global_ids, -1))
        cand_indices.append(global_ids)
        cand_scores.append(scores)
    all_scores = torch.cat(cand_scores, dim=1)
    all_indices = torch.cat(cand_indices, dim=1)
    order = torch.argsort(all_scores, dim=-1, descending=True, stable=True)
    merged_scores = torch.gather(all_scores, 1, order)[:, :select_k]
    merged_indices = torch.gather(all_indices, 1, order)[:, :select_k]

    _assert_topk_parity(
        got_indices=merged_indices.to(torch.int32),
        got_scores=merged_scores,
        ref_values=ref_values,
        ref_indices=ref_indices,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_page32_big_page_id_gather_survives_2gib_offsets(monkeypatch) -> None:
    """AGENTS.md 64-bit rule: live pages parked past 2**31 / stride bytes.

    The triton supertile gather scales page ids into byte offsets; a 32-bit
    product silently corrupts (or faults) once page_id * row_stride crosses
    2**31. Page-32 row stride is 4224 bytes, so ids past ~508k pages leave the
    32-bit envelope.
    """
    monkeypatch.setenv("B12X_PAGED_INDEX_SUPERTILE_K", "1024")
    device = torch.device("cuda")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(64_511)

    rows = 4
    seqlen = 2048  # pools
    topk = 512
    width_blocks = seqlen // _PAGE
    row_stride = _PAGE * (_HEAD_DIM + 4)
    # First page id whose byte offset exceeds 2**31.
    big_base = (1 << 31) // row_stride + 4096
    num_pages = big_base + width_blocks

    q_fp8 = _rand_fp8_q((rows, _NUM_HEADS, _HEAD_DIM), gen=gen, device=device)
    weights = torch.randn((rows, _NUM_HEADS), generator=gen).to(device)
    k_matrix = (torch.randn((seqlen, _HEAD_DIM), generator=gen) / 3).to(device)
    k_matrix = k_matrix * (1.0 + torch.arange(seqlen, device=device)[:, None] / seqlen)
    packed = pack_paged_index_k_cache_reference(k_matrix, page_size=_PAGE)

    # Mostly-uninitialized pool; only the parked tail pages hold valid data.
    cache = torch.empty((num_pages, row_stride), dtype=torch.uint8, device=device)
    cache[num_pages - width_blocks :].copy_(packed)

    page_ids = torch.arange(
        num_pages - width_blocks, num_pages, dtype=torch.int32, device=device
    )
    assert int(page_ids[0]) * row_stride > (1 << 31)
    page_table = page_ids.view(1, width_blocks).expand(rows, width_blocks)
    seqlens_t = torch.full((rows,), seqlen, dtype=torch.int32, device=device)

    binding = _bind_paged_indexer(
        device=device,
        num_heads=_NUM_HEADS,
        rows=rows,
        width_blocks=width_blocks,
        topk=topk,
        real_page_table=page_table,
        seqlens=seqlens_t,
        supertile_k=1024,
    )
    out_indices = torch.empty((rows, topk), dtype=torch.int32, device=device)
    out_scores = torch.empty((rows, topk), dtype=torch.float32, device=device)
    index_topk_fp8(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=cache,
        binding=binding,
        page_size=_PAGE,
        expected_num_q_heads=_NUM_HEADS,
        out_indices=out_indices,
        out_scores=out_scores,
    )
    torch.cuda.synchronize(device)

    ref_values, ref_indices = _reference_topk(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=cache,
        real_page_table=page_table,
        seqlens=seqlens_t,
        topk=topk,
    )
    _assert_topk_parity(
        got_indices=out_indices,
        got_scores=out_scores,
        ref_values=ref_values,
        ref_indices=ref_indices,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_page32_rejected_on_64only_routes(monkeypatch) -> None:
    """Page-32 support is scoped to the packed-contiguous prefill route."""
    monkeypatch.setenv("B12X_PAGED_INDEX_SUPERTILE_K", "512")
    device = torch.device("cuda")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(7)

    rows = 1
    seqlen = 64
    width_blocks = seqlen // _PAGE
    q_fp8 = _rand_fp8_q((rows, _NUM_HEADS, _HEAD_DIM), gen=gen, device=device)
    weights = torch.randn((rows, _NUM_HEADS), generator=gen).to(device)
    k_matrix = torch.randn((seqlen, _HEAD_DIM), generator=gen).to(device) / 3
    cache = pack_paged_index_k_cache_reference(k_matrix, page_size=_PAGE)
    table = torch.arange(width_blocks, dtype=torch.int32, device=device).view(
        1, width_blocks
    )
    seqlens_t = torch.full((rows,), seqlen, dtype=torch.int32, device=device)
    binding = _bind_paged_indexer(
        device=device,
        num_heads=_NUM_HEADS,
        rows=rows,
        width_blocks=width_blocks,
        topk=8,
        real_page_table=table,
        seqlens=seqlens_t,
        supertile_k=512,
        shared_page_table=False,
    )
    with pytest.raises(RuntimeError, match="only supports page_size=64"):
        index_topk_fp8(
            q_fp8=q_fp8,
            weights=weights,
            index_k_cache=cache,
            binding=binding,
            page_size=_PAGE,
            expected_num_q_heads=_NUM_HEADS,
        )


def test_page_size_constants() -> None:
    assert PAGED_INDEX_PAGE_SIZES == (32, 64)
