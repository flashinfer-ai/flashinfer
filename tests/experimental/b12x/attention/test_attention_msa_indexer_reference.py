from __future__ import annotations

import math

import torch

from b12x.attention.dsa_indexer._impl import msa_topk_blocks
from b12x.attention.dsa_indexer.msa_reference import (
    MSA_BLOCK_TOKENS,
    MSA_SM_SCALE,
    MSA_TOPK_BLOCKS,
    msa_contiguous_block_scores_reference,
    msa_paged_decode_block_scores_reference,
    msa_q2k_indices_reference,
    msa_select_blocks_reference,
    quantize_msa_q_fp8_reference,
)
from b12x.attention.dsa_indexer.reference import (
    pack_index_k_cache_reference,
    unpack_index_k_cache_reference,
)


_FP8_E4M3_MAX = float(torch.finfo(torch.float8_e4m3fn).max)


def _quantize_rows_to_kv_fp8(k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = k.abs().amax(dim=1) / _FP8_E4M3_MAX
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    quant = (k / scale.unsqueeze(1)).clamp(-_FP8_E4M3_MAX, _FP8_E4M3_MAX)
    return quant.to(torch.float8_e4m3fn), scale.to(torch.float32)


def _make_inputs(
    *,
    q_rows: int,
    heads: int,
    k_rows: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    q = torch.randn((q_rows, heads, 128), generator=gen, dtype=torch.float32) / 3
    k = torch.randn((k_rows, 128), generator=gen, dtype=torch.float32) / 3
    q_fp8, q_scale = quantize_msa_q_fp8_reference(q)
    k_fp8, k_scale = _quantize_rows_to_kv_fp8(k)
    k_dequant = k_fp8.to(torch.float32) * k_scale.unsqueeze(1)
    return q_fp8, q_scale, k_fp8, k_scale, k_dequant


def _manual_contiguous_block_scores(
    *,
    q_fp8: torch.Tensor,
    q_scale: torch.Tensor,
    k_dequant: torch.Tensor,
    k_start: torch.Tensor,
    k_end: torch.Tensor,
) -> torch.Tensor:
    q_rows, heads, _ = q_fp8.shape
    num_blocks = math.ceil(k_dequant.shape[0] / MSA_BLOCK_TOKENS)
    out = torch.full((heads, q_rows, num_blocks), float("-inf"), dtype=torch.float32)
    q_dequant = q_fp8.to(torch.float32) * q_scale.unsqueeze(2)
    for q_idx in range(k_start.numel()):
        begin = max(0, int(k_start[q_idx]))
        end = min(int(k_end[q_idx]), k_dequant.shape[0])
        for token_idx in range(begin, end):
            block_idx = token_idx // MSA_BLOCK_TOKENS
            score = torch.matmul(q_dequant[q_idx], k_dequant[token_idx]) * MSA_SM_SCALE
            out[:, q_idx, block_idx] = torch.maximum(out[:, q_idx, block_idx], score)
    return out


def _manual_paged_block_scores(
    *,
    q_fp8: torch.Tensor,
    q_scale: torch.Tensor,
    k_dequant: torch.Tensor,
    real_page_table: torch.Tensor,
    cache_seqlens_int32: torch.Tensor,
) -> torch.Tensor:
    q_rows, heads, _ = q_fp8.shape
    width_tokens = real_page_table.shape[1] * 64
    num_blocks = math.ceil(width_tokens / MSA_BLOCK_TOKENS)
    out = torch.full((heads, q_rows, num_blocks), float("-inf"), dtype=torch.float32)
    q_dequant = q_fp8.to(torch.float32) * q_scale.unsqueeze(2)
    for q_idx in range(real_page_table.shape[0]):
        seq_len = min(max(0, int(cache_seqlens_int32[q_idx])), width_tokens)
        for logical_pos in range(seq_len):
            page_id = int(real_page_table[q_idx, logical_pos // 64])
            if page_id < 0:
                continue
            physical = page_id * 64 + logical_pos % 64
            if physical >= k_dequant.shape[0]:
                continue
            block_idx = logical_pos // MSA_BLOCK_TOKENS
            score = torch.matmul(q_dequant[q_idx], k_dequant[physical]) * MSA_SM_SCALE
            out[:, q_idx, block_idx] = torch.maximum(out[:, q_idx, block_idx], score)
    return out


def test_quantize_msa_q_fp8_reference_uses_positive_scales_for_zero_rows() -> None:
    q = torch.zeros((2, 4, 128), dtype=torch.float32)
    q_fp8, q_scale = quantize_msa_q_fp8_reference(q)
    assert q_fp8.dtype == torch.float8_e4m3fn
    assert q_scale.dtype == torch.float32
    assert torch.equal(q_scale, torch.ones_like(q_scale))


def test_msa_contiguous_block_scores_reference_matches_token_loop() -> None:
    q_fp8, q_scale, k_fp8, k_scale, k_dequant = _make_inputs(
        q_rows=5,
        heads=4,
        k_rows=384,
        seed=91_001,
    )
    k_start = torch.tensor([0, 1, 127, 128, 255], dtype=torch.int32)
    k_end = torch.tensor([1, 127, 129, 255, 256], dtype=torch.int32)

    actual = msa_contiguous_block_scores_reference(
        q_fp8=q_fp8,
        q_scale=q_scale,
        kv_fp8=(k_fp8, k_scale),
        k_start=k_start,
        k_end=k_end,
    )
    expected = _manual_contiguous_block_scores(
        q_fp8=q_fp8,
        q_scale=q_scale,
        k_dequant=k_dequant,
        k_start=k_start,
        k_end=k_end,
    )

    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    assert torch.isneginf(actual[:, 0, 1:]).all()
    assert torch.isfinite(actual[:, 2, 0]).all()
    assert torch.isfinite(actual[:, 2, 1]).all()


def test_msa_paged_decode_block_scores_reference_matches_token_loop() -> None:
    q_fp8, q_scale, _, _, _ = _make_inputs(q_rows=3, heads=4, k_rows=1, seed=91_101)
    page_starts = [3, 6, 9]
    width_pages = 5
    total_pages = max(page_starts) + width_pages
    gen = torch.Generator(device="cpu")
    gen.manual_seed(91_102)
    k = torch.randn((total_pages * 64, 128), generator=gen, dtype=torch.float32) / 4
    index_k_cache = pack_index_k_cache_reference(k)
    k_dequant = unpack_index_k_cache_reference(
        index_k_cache, num_tokens=total_pages * 64
    )
    real_page_table = torch.full((3, width_pages), -1, dtype=torch.int32)
    seqlens = torch.tensor([63, 129, 255], dtype=torch.int32)
    for row, start in enumerate(page_starts):
        pages = (int(seqlens[row]) + 63) // 64
        real_page_table[row, :pages] = torch.arange(
            start, start + pages, dtype=torch.int32
        )

    actual = msa_paged_decode_block_scores_reference(
        q_fp8=q_fp8,
        q_scale=q_scale,
        index_k_cache=index_k_cache,
        real_page_table=real_page_table,
        cache_seqlens_int32=seqlens,
    )
    expected = _manual_paged_block_scores(
        q_fp8=q_fp8,
        q_scale=q_scale,
        k_dequant=k_dequant,
        real_page_table=real_page_table,
        cache_seqlens_int32=seqlens,
    )

    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    assert torch.isneginf(actual[:, 0, 1:]).all()
    assert torch.isfinite(actual[:, 1, 1]).all()


def test_msa_select_blocks_reference_forces_local_sorts_and_pads() -> None:
    scores = torch.arange(20, dtype=torch.float32).view(1, 1, 20)
    actual = msa_select_blocks_reference(
        block_scores=scores,
        query_positions=torch.tensor([0], dtype=torch.int32),
        topk=MSA_TOPK_BLOCKS,
    )
    expected = torch.tensor(
        [[[0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]]]
    )
    assert torch.equal(actual, expected.to(torch.int32))

    all_inf = torch.full((1, 1, 4), float("-inf"), dtype=torch.float32)
    forced = msa_select_blocks_reference(
        block_scores=all_inf,
        query_positions=torch.tensor([2 * MSA_BLOCK_TOKENS], dtype=torch.int32),
        topk=6,
    )
    assert torch.equal(
        forced, torch.tensor([[[2, -1, -1, -1, -1, -1]]], dtype=torch.int32)
    )


def test_msa_topk_blocks_matches_reference_with_block_base() -> None:
    scores = torch.full((4, 2, 8), float("-inf"), dtype=torch.float32)
    jitter = torch.arange(scores.numel(), dtype=torch.float32).view_as(scores) * 1.0e-6
    scores[:, 0, 4:8] = torch.tensor([0.1, 0.4, 0.2, 0.3])
    scores[:, 1, 2:6] = torch.tensor([0.5, 0.1, 0.3, 0.2])
    scores = scores + jitter
    query_positions = torch.tensor(
        [5 * MSA_BLOCK_TOKENS, 3 * MSA_BLOCK_TOKENS], dtype=torch.int32
    )
    block_base = torch.tensor([4, 2], dtype=torch.int32)

    actual = msa_topk_blocks(
        block_scores=scores,
        query_positions=query_positions,
        block_base=block_base,
        topk=6,
    )
    expected = msa_select_blocks_reference(
        block_scores=scores,
        query_positions=query_positions,
        block_base=block_base,
        topk=6,
    )
    assert torch.equal(actual, expected)
    assert (actual[:, :, 0] >= 0).all()
    assert torch.equal(actual[:, 0, :4], torch.tensor([[0, 1, 2, 3]]).expand(4, -1))


def test_msa_q2k_indices_reference_composes_paged_decode() -> None:
    q_fp8, q_scale, _, _, _ = _make_inputs(q_rows=2, heads=1, k_rows=1, seed=91_201)
    k = torch.eye(128, dtype=torch.float32).repeat(2, 1)
    index_k_cache = pack_index_k_cache_reference(k)
    real_page_table = torch.tensor([[0, 1, 2], [0, 1, -1]], dtype=torch.int32)
    seqlens = torch.tensor([129, 64], dtype=torch.int32)

    actual = msa_q2k_indices_reference(
        q_fp8=q_fp8,
        q_scale=q_scale,
        index_k_cache=index_k_cache,
        real_page_table=real_page_table,
        cache_seqlens_int32=seqlens,
        query_positions=seqlens - 1,
        topk=4,
    )

    assert actual.shape == (1, 2, 4)
    assert actual[0, 0, 1].item() == 1
    assert actual[0, 1, 0].item() == 0
