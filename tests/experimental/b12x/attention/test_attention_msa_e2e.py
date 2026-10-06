from __future__ import annotations

import math

import torch

from b12x.attention.dsa_indexer.reference import pack_index_k_cache_reference
from b12x.attention.paged.reference import msa_attention_reference
from b12x.attention._shared.contiguous.api import clear_attention_caches
from b12x.attention.paged._forward import paged_attention_forward
from b12x.attention.paged._scratch import (
    B12XPagedAttentionScratchCaps,
    plan_paged_attention_scratch,
)
from b12x.attention.paged.planner import create_paged_plan

from b12x.testing.reference.helpers import require_b12x
from b12x.testing.reference.paged_attention_helpers import make_paged_inputs


_INDEX_HEAD_DIM = 128
_INDEX_PAGE_SIZE = 64
_MSA_BLOCK_TOKENS = 128
_MSA_TOPK = 16
_MSA_SCALE = 1.0 / math.sqrt(float(_INDEX_HEAD_DIM))


def _pack_index_cache_from_attention_k(k_cache: torch.Tensor) -> torch.Tensor:
    k_idx = k_cache[:, :, 0, :].contiguous().view(-1, k_cache.shape[-1])
    return pack_index_k_cache_reference(k_idx, page_size=int(k_cache.shape[1]))


def _unpack_index_cache_pages(index_k_cache: torch.Tensor) -> torch.Tensor:
    num_pages = int(index_k_cache.shape[0])
    data_bytes = _INDEX_PAGE_SIZE * _INDEX_HEAD_DIM
    k_quant = (
        index_k_cache[:, :data_bytes]
        .contiguous()
        .view(num_pages, _INDEX_PAGE_SIZE, _INDEX_HEAD_DIM)
        .view(torch.float8_e4m3fn)
        .to(torch.float32)
    )
    k_scale = (
        index_k_cache[:, data_bytes : data_bytes + _INDEX_PAGE_SIZE * 4]
        .contiguous()
        .view(torch.float32)
        .view(num_pages, _INDEX_PAGE_SIZE)
    )
    return k_quant * k_scale.unsqueeze(-1)


def _select_msa_topk_blocks(
    block_scores: torch.Tensor,
    query_positions: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    num_heads, total_q, num_blocks = block_scores.shape
    topk = min(_MSA_TOPK, int(num_blocks))
    local_blocks = torch.div(
        query_positions.to(torch.long),
        _MSA_BLOCK_TOKENS,
        rounding_mode="floor",
    ).clamp_(min=0, max=max(int(num_blocks) - 1, 0))
    local = local_blocks.view(1, total_q, 1).expand(num_heads, total_q, 1)

    forced = block_scores.clone()
    forced.scatter_(2, local, float("inf"))
    values, indices = torch.topk(forced, k=topk, dim=2)
    del values

    gathered = block_scores.gather(2, indices)
    valid = torch.isfinite(gathered) | indices.eq(
        local.expand(num_heads, total_q, topk)
    )
    sentinel = torch.full_like(indices, torch.iinfo(torch.int32).max)
    selected = torch.where(valid, indices, sentinel)
    selected, _ = selected.sort(dim=2)
    selected = torch.where(selected.eq(sentinel), -1, selected).to(torch.int32)

    if out is None:
        out = torch.full(
            (num_heads, total_q, _MSA_TOPK),
            -1,
            dtype=torch.int32,
            device=block_scores.device,
        )
    else:
        out.fill_(-1)
    out[:, :, :topk].copy_(selected)
    return out


def _decode_msa_q2k_from_index_cache(
    *,
    q: torch.Tensor,
    index_k_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    batch, page_table_width = page_table.shape
    kv_heads = int(out.shape[0])
    q_per_kv = int(q.shape[1]) // kv_heads
    width = int(page_table_width) * _INDEX_PAGE_SIZE
    num_blocks = (width + _MSA_BLOCK_TOKENS - 1) // _MSA_BLOCK_TOKENS

    k_pages = _unpack_index_cache_pages(index_k_cache)
    page_ids = page_table.to(torch.long).clamp_(
        min=0, max=max(int(k_pages.shape[0]) - 1, 0)
    )
    k_rows = k_pages.index_select(0, page_ids.reshape(-1)).view(
        batch,
        page_table_width * _INDEX_PAGE_SIZE,
        _INDEX_HEAD_DIM,
    )
    q_idx = q.view(batch, kv_heads, q_per_kv, _INDEX_HEAD_DIM)[:, :, 0, :].to(
        torch.float32
    )
    scores = torch.matmul(q_idx, k_rows.transpose(1, 2)) * _MSA_SCALE

    token_pos = torch.arange(width, dtype=torch.long, device=q.device)
    valid = token_pos.view(1, 1, width) < cache_seqlens.to(torch.long).view(batch, 1, 1)
    scores = scores.masked_fill(~valid, float("-inf"))
    block_scores = scores.view(batch, kv_heads, num_blocks, _MSA_BLOCK_TOKENS).amax(
        dim=3
    )
    block_scores = block_scores.permute(1, 0, 2).contiguous()
    query_positions = (cache_seqlens - 1).clamp_min(0)
    return _select_msa_topk_blocks(block_scores, query_positions, out=out)


def _prefill_msa_q2k_from_index_cache(
    *,
    q: torch.Tensor,
    index_k_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    num_kv_heads: int,
) -> torch.Tensor:
    q_offsets = [int(v) for v in cu_seqlens_q.detach().cpu().tolist()]
    cache_lengths = [int(v) for v in cache_seqlens.detach().cpu().tolist()]
    total_q = q_offsets[-1]
    q2k = torch.full(
        (num_kv_heads, total_q, _MSA_TOPK),
        -1,
        dtype=torch.int32,
        device=q.device,
    )
    k_pages = _unpack_index_cache_pages(index_k_cache)
    q_per_kv = int(q.shape[1]) // int(num_kv_heads)
    q_idx = q.view(total_q, num_kv_heads, q_per_kv, _INDEX_HEAD_DIM)[:, :, 0, :].to(
        torch.float32
    )

    for request_idx, (q_start, q_end) in enumerate(zip(q_offsets[:-1], q_offsets[1:])):
        qo_len = q_end - q_start
        cache_len = cache_lengths[request_idx]
        num_pages = (cache_len + _INDEX_PAGE_SIZE - 1) // _INDEX_PAGE_SIZE
        page_ids = page_table[request_idx, :num_pages].to(torch.long)
        k_rows = k_pages.index_select(0, page_ids).view(-1, _INDEX_HEAD_DIM)[:cache_len]
        for q_row in range(q_start, q_end):
            token_local = q_row - q_start
            visible = max(token_local + cache_len - qo_len + 1, 1)
            num_blocks = (visible + _MSA_BLOCK_TOKENS - 1) // _MSA_BLOCK_TOKENS
            scores = (
                torch.matmul(q_idx[q_row], k_rows[:visible].transpose(0, 1))
                * _MSA_SCALE
            )
            padded = torch.full(
                (num_kv_heads, num_blocks * _MSA_BLOCK_TOKENS),
                float("-inf"),
                dtype=torch.float32,
                device=q.device,
            )
            padded[:, :visible].copy_(scores)
            block_scores = padded.view(
                num_kv_heads, num_blocks, _MSA_BLOCK_TOKENS
            ).amax(dim=2)
            selected = _select_msa_topk_blocks(
                block_scores.unsqueeze(1),
                torch.tensor([visible - 1], dtype=torch.int32, device=q.device),
            )
            q2k[:, q_row : q_row + 1, :].copy_(selected)
    return q2k.contiguous()


def _run_msa_extend_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    q2k_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    plan = create_paged_plan(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        mode="extend",
        msa_block_sparse=True,
    )
    assert plan.msa_union_tile is True
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
            max_work_items=max(plan.new_batch_size, 1),
            max_partial_rows=0,
            num_cache_pages=k_cache.shape[0],
            msa_block_sparse=True,
        )
    )
    scratch = tuple(
        torch.empty(shape, dtype=dtype, device=q.device)
        for shape, dtype in scratch_plan.shapes_and_dtypes()
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
        q2k_indices=q2k_indices,
    )
    out, lse_base2 = paged_attention_forward(binding=binding)
    return out, lse_base2 * math.log(2.0)


@torch.inference_mode()
def test_msa_decode_indexer_selection_to_attention_graph_replay_contract() -> None:
    require_b12x()
    clear_attention_caches()

    batch = 2
    page_table_width = 80
    num_pages = 160

    def make_case(cache_lens: list[int], *, seed: int):
        return make_paged_inputs(
            q_seqlens=[1] * batch,
            cache_seqlens=cache_lens,
            page_size=_INDEX_PAGE_SIZE,
            q_heads=64,
            kv_heads=4,
            head_dim=_INDEX_HEAD_DIM,
            dtype=torch.bfloat16,
            seed=seed,
            page_table_width=page_table_width,
            num_pages=num_pages,
        )

    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = make_case(
        [2048, 5000],
        seed=3101,
    )
    index_k_cache = _pack_index_cache_from_attention_k(k_cache)
    q2k_indices = torch.empty((4, batch, _MSA_TOPK), dtype=torch.int32, device=q.device)

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
            max_total_q=batch,
            max_batch=batch,
            max_page_table_width=page_table_width,
            max_work_items=batch * 32,
            max_partial_rows=batch * 32,
            num_cache_pages=num_pages,
            use_cuda_graph=True,
            msa_block_sparse=True,
        )
    )
    scratch_plan.prepare_decode_graph_replay_state(
        batch=batch,
        max_page_table_width=page_table_width,
        max_cache_page_count=page_table_width,
    )
    scratch = tuple(
        torch.empty(shape, dtype=dtype, device=q.device)
        for shape, dtype in scratch_plan.shapes_and_dtypes()
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
        q2k_indices=q2k_indices,
    )
    runtime_page_table = binding.scratch.page_table
    runtime_cache_seqlens = binding.scratch.cache_seqlens
    assert runtime_page_table is not None
    assert runtime_cache_seqlens is not None

    _decode_msa_q2k_from_index_cache(
        q=q,
        index_k_cache=index_k_cache,
        page_table=runtime_page_table,
        cache_seqlens=runtime_cache_seqlens,
        out=q2k_indices,
    )
    paged_attention_forward(binding=binding)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _decode_msa_q2k_from_index_cache(
            q=q,
            index_k_cache=index_k_cache,
            page_table=runtime_page_table,
            cache_seqlens=runtime_cache_seqlens,
            out=q2k_indices,
        )
        paged_attention_forward(binding=binding)

    graph.replay()
    torch.cuda.synchronize()
    expected_q2k = torch.empty_like(q2k_indices)
    _decode_msa_q2k_from_index_cache(
        q=q,
        index_k_cache=index_k_cache,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        out=expected_q2k,
    )
    torch.testing.assert_close(q2k_indices, expected_q2k)
    ref_out, ref_lse = msa_attention_reference(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        q2k_indices,
    )
    torch.testing.assert_close(
        binding.scratch.current_lse_view() * math.log(2.0),
        ref_lse,
        rtol=2e-3,
        atol=2e-3,
    )
    assert (
        torch.nn.functional.cosine_similarity(
            output.to(torch.float32).reshape(-1),
            ref_out.to(torch.float32).reshape(-1),
            dim=0,
        ).item()
        >= 0.999
    )

    q_next, k_next, v_next, page_table_next, cache_seqlens_next, cu_seqlens_q_next = (
        make_case(
            [1, 129],
            seed=3102,
        )
    )
    q.copy_(q_next)
    k_cache.copy_(k_next)
    v_cache.copy_(v_next)
    index_k_cache.copy_(_pack_index_cache_from_attention_k(k_next))
    binding = scratch_plan.bind(
        scratch=scratch,
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        output=output,
        page_table=page_table_next,
        cache_seqlens=cache_seqlens_next,
        cu_seqlens_q=cu_seqlens_q_next,
        q2k_indices=q2k_indices,
    )

    graph.replay()
    torch.cuda.synchronize()
    _decode_msa_q2k_from_index_cache(
        q=q,
        index_k_cache=index_k_cache,
        page_table=page_table_next,
        cache_seqlens=cache_seqlens_next,
        out=expected_q2k,
    )
    torch.testing.assert_close(q2k_indices, expected_q2k)
    ref_out_next, ref_lse_next = msa_attention_reference(
        q,
        k_cache,
        v_cache,
        page_table_next,
        cache_seqlens_next,
        cu_seqlens_q_next,
        q2k_indices,
    )
    torch.testing.assert_close(
        binding.scratch.current_lse_view() * math.log(2.0),
        ref_lse_next,
        rtol=2e-3,
        atol=2e-3,
    )
    assert (
        torch.nn.functional.cosine_similarity(
            output.to(torch.float32).reshape(-1),
            ref_out_next.to(torch.float32).reshape(-1),
            dim=0,
        ).item()
        >= 0.999
    )


@torch.inference_mode()
def test_msa_prefill_indexer_selection_to_union_attention_contract() -> None:
    require_b12x()
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = make_paged_inputs(
        q_seqlens=[8, 5],
        cache_seqlens=[384, 512],
        page_size=_INDEX_PAGE_SIZE,
        q_heads=64,
        kv_heads=4,
        head_dim=_INDEX_HEAD_DIM,
        dtype=torch.bfloat16,
        seed=3201,
    )
    index_k_cache = _pack_index_cache_from_attention_k(k_cache)
    q2k_indices = _prefill_msa_q2k_from_index_cache(
        q=q,
        index_k_cache=index_k_cache,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        num_kv_heads=4,
    )

    out, lse = _run_msa_extend_attention(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        q2k_indices,
    )
    ref_out, ref_lse = msa_attention_reference(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        q2k_indices,
    )
    torch.testing.assert_close(lse, ref_lse, rtol=2e-3, atol=2e-3)
    assert (
        torch.nn.functional.cosine_similarity(
            out.to(torch.float32).reshape(-1),
            ref_out.to(torch.float32).reshape(-1),
            dim=0,
        ).item()
        >= 0.999
    )
