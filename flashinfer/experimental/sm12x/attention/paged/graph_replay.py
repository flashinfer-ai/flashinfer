# SPDX-FileCopyrightText: 2026 FlashInfer team
# SPDX-License-Identifier: Apache-2.0
# Ported from b12x b12x/attention/paged/graph_replay.py @ 3a6c424b (2026-06-13) -- one-time curated port.
# Upstream b12x is a research sandbox; this tree is the canonical home.
"""Device-side decode graph replay helpers for the paged attention backend."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import triton
import triton.language as tl

_DECODE_BLOCK_CHUNKS = 128
_DECODE_BLOCK_PAGES = 128
_PREFILL_BLOCK_WORK_ITEMS = 128
_PREFILL_BLOCK_ROWS = 128
_MSA_UNION_TOPK = 16
_MSA_UNION_TOKENS_PER_TILE = 8
_MSA_UNION_MAX_BLOCKS = _MSA_UNION_TOPK * _MSA_UNION_TOKENS_PER_TILE


@triton.jit
def stage_decode_cuda_graph_metadata_triton(
    req_to_token_ptr,
    req_pool_indices_ptr,
    seq_lens_ptr,
    cache_seqlens_ptr,
    cu_seqlens_q_ptr,
    page_table_ptr,
    swa_page_table_ptr,
    swa_index_mapping_ptr,
    req_to_token_row_stride,
    req_to_token_numel,
    page_table_row_stride,
    swa_page_table_row_stride,
    req_pool_indices_stride,
    seq_lens_stride,
    swa_index_mapping_size,
    PAGE_SIZE: tl.constexpr,
    BATCH: tl.constexpr,
    MAX_PAGES: tl.constexpr,
    HAS_SWA: tl.constexpr,
    BLOCK_PAGES: tl.constexpr,
):
    req_idx = tl.program_id(axis=0)
    page_block_idx = tl.program_id(axis=1)

    if page_block_idx == 0:
        seq_len = tl.load(seq_lens_ptr + req_idx * seq_lens_stride).to(tl.int32)
        tl.store(cache_seqlens_ptr + req_idx, seq_len)
        tl.store(cu_seqlens_q_ptr + req_idx, req_idx)
        tl.store(cu_seqlens_q_ptr + BATCH, BATCH, mask=req_idx == 0)

    page_offsets = page_block_idx * BLOCK_PAGES + tl.arange(0, BLOCK_PAGES)
    page_mask = page_offsets < MAX_PAGES
    req_pool_idx = tl.load(req_pool_indices_ptr + req_idx * req_pool_indices_stride).to(
        tl.int64
    )
    flat_token_offsets = (
        req_pool_idx * req_to_token_row_stride + page_offsets.to(tl.int64) * PAGE_SIZE
    )
    flat_token_offsets = tl.minimum(
        tl.maximum(flat_token_offsets, 0),
        req_to_token_numel - 1,
    )
    token_indices = tl.load(
        req_to_token_ptr + flat_token_offsets, mask=page_mask, other=0
    )
    tl.store(
        page_table_ptr + req_idx * page_table_row_stride + page_offsets,
        (token_indices // PAGE_SIZE).to(tl.int32),
        mask=page_mask,
    )

    if HAS_SWA:
        swa_indices = tl.where(
            token_indices < 0,
            swa_index_mapping_size - 1,
            token_indices,
        )
        swa_indices = tl.minimum(tl.maximum(swa_indices, 0), swa_index_mapping_size - 1)
        swa_token_indices = tl.load(
            swa_index_mapping_ptr + swa_indices,
            mask=page_mask,
            other=-1,
        )
        tl.store(
            swa_page_table_ptr + req_idx * swa_page_table_row_stride + page_offsets,
            (swa_token_indices // PAGE_SIZE).to(tl.int32),
            mask=page_mask,
        )


@triton.jit
def patch_decode_cuda_graph_current_pages_triton(
    cache_seqlens_ptr,
    page_table_ptr,
    swa_page_table_ptr,
    out_cache_loc_ptr,
    out_cache_loc_swa_ptr,
    swa_index_mapping_ptr,
    page_table_row_stride,
    swa_page_table_row_stride,
    out_cache_loc_stride,
    out_cache_loc_swa_stride,
    swa_index_mapping_size,
    PAGE_SIZE: tl.constexpr,
    FILL_VALUE: tl.constexpr,
    MAX_PAGES: tl.constexpr,
    HAS_SWA: tl.constexpr,
    HAS_OUT_CACHE_LOC_SWA: tl.constexpr,
    BLOCK_PAGES: tl.constexpr,
):
    req_idx = tl.program_id(axis=0)
    page_block_idx = tl.program_id(axis=1)

    cache_len = tl.load(cache_seqlens_ptr + req_idx).to(tl.int32)
    out_cache_loc = tl.load(out_cache_loc_ptr + req_idx * out_cache_loc_stride).to(
        tl.int64
    )
    valid = (cache_len != FILL_VALUE) | (out_cache_loc != 0)
    logical_page = tl.minimum(
        tl.maximum((cache_len - 1) // PAGE_SIZE, 0),
        MAX_PAGES - 1,
    )

    page_offsets = page_block_idx * BLOCK_PAGES + tl.arange(0, BLOCK_PAGES)
    page_mask = page_offsets < MAX_PAGES
    invalid_page_mask = page_mask & ~valid
    tl.store(
        page_table_ptr + req_idx * page_table_row_stride + page_offsets,
        tl.zeros((BLOCK_PAGES,), tl.int32),
        mask=invalid_page_mask,
    )
    if HAS_SWA:
        tl.store(
            swa_page_table_ptr + req_idx * swa_page_table_row_stride + page_offsets,
            tl.zeros((BLOCK_PAGES,), tl.int32),
            mask=invalid_page_mask,
        )

    if page_block_idx == 0:
        tl.store(cache_seqlens_ptr + req_idx, tl.where(valid, cache_len, FILL_VALUE))
        current_page = (out_cache_loc // PAGE_SIZE).to(tl.int32)
        tl.store(
            page_table_ptr + req_idx * page_table_row_stride + logical_page,
            current_page,
            mask=valid,
        )
        if HAS_SWA:
            if HAS_OUT_CACHE_LOC_SWA:
                out_cache_loc_swa = tl.load(
                    out_cache_loc_swa_ptr + req_idx * out_cache_loc_swa_stride
                ).to(tl.int64)
            else:
                swa_mapping_idx = tl.minimum(
                    tl.maximum(out_cache_loc, 0),
                    swa_index_mapping_size - 1,
                )
                out_cache_loc_swa = tl.load(swa_index_mapping_ptr + swa_mapping_idx).to(
                    tl.int64
                )
            tl.store(
                swa_page_table_ptr + req_idx * swa_page_table_row_stride + logical_page,
                (out_cache_loc_swa // PAGE_SIZE).to(tl.int32),
                mask=valid,
            )


@triton.jit
def build_decode_graph_page_table_full_triton(
    req_to_token_ptr,
    req_pool_indices_ptr,
    page_table_ptr,
    req_to_token_row_stride,
    req_to_token_numel,
    page_table_row_stride,
    PAGE_SIZE: tl.constexpr,
    MAX_PAGES: tl.constexpr,
    BLOCK_PAGES: tl.constexpr,
):
    req_idx = tl.program_id(axis=0)
    page_block_idx = tl.program_id(axis=1)

    req_pool_idx = tl.load(req_pool_indices_ptr + req_idx).to(tl.int64)
    page_offsets = page_block_idx * BLOCK_PAGES + tl.arange(0, BLOCK_PAGES)
    page_mask = page_offsets < MAX_PAGES
    flat_token_offsets = (
        req_pool_idx * req_to_token_row_stride + page_offsets.to(tl.int64) * PAGE_SIZE
    )
    flat_token_offsets = tl.minimum(
        tl.maximum(flat_token_offsets, 0),
        req_to_token_numel - 1,
    )
    token_indices = tl.load(
        req_to_token_ptr + flat_token_offsets, mask=page_mask, other=0
    )
    tl.store(
        page_table_ptr + req_idx * page_table_row_stride + page_offsets,
        (token_indices // PAGE_SIZE).to(tl.int32),
        mask=page_mask,
    )


@triton.jit
def update_decode_graph_metadata_fused_triton(
    cache_seqlens_ptr,
    request_indices_ptr,
    qo_tile_indices_ptr,
    kv_tile_indices_ptr,
    merge_indptr_ptr,
    o_indptr_ptr,
    block_valid_mask_ptr,
    kv_chunk_size_ptr,
    kv_window_start_tokens_ptr,
    decode_chunk_pages_lut_ptr,
    max_chunks_per_req,
    block_valid_capacity,
    lut_size,
    PAGE_SIZE: tl.constexpr,
    WINDOW_PAGE_SPAN: tl.constexpr,
    WINDOW_LEFT: tl.constexpr,
    BATCH: tl.constexpr,
    BLOCK_BATCH: tl.constexpr,
    BLOCK_WORK_ITEMS: tl.constexpr,
):
    batch_offsets = tl.arange(0, BLOCK_BATCH)
    batch_mask = batch_offsets < BATCH
    cache_len = tl.load(cache_seqlens_ptr + batch_offsets, mask=batch_mask, other=0).to(
        tl.int32
    )
    num_pages = tl.maximum((cache_len + (PAGE_SIZE - 1)) // PAGE_SIZE, 1)
    window_start_page = tl.full((BLOCK_BATCH,), 0, tl.int32)
    if WINDOW_LEFT >= 0:
        window_start_token = tl.maximum(cache_len - 1 - WINDOW_LEFT, 0)
        window_start_page = window_start_token // PAGE_SIZE
    effective_pages = tl.maximum(num_pages - window_start_page, 1)
    policy_pages = tl.max(tl.where(batch_mask, effective_pages, 1), axis=0)
    if WINDOW_PAGE_SPAN > 0:
        policy_pages = tl.minimum(policy_pages, WINDOW_PAGE_SPAN)
    policy_pages = tl.minimum(tl.maximum(policy_pages, 1), lut_size - 1)
    chunk_pages = tl.load(decode_chunk_pages_lut_ptr + policy_pages).to(tl.int32)
    num_chunks = tl.maximum((effective_pages + chunk_pages - 1) // chunk_pages, 1)
    prefix = tl.cumsum(tl.where(batch_mask, num_chunks, 0), 0)

    tl.store(merge_indptr_ptr, 0)
    tl.store(o_indptr_ptr, 0)
    tl.store(
        merge_indptr_ptr + batch_offsets + 1,
        prefix,
        mask=batch_mask,
    )
    tl.store(
        o_indptr_ptr + batch_offsets + 1,
        prefix,
        mask=batch_mask,
    )
    tl.store(
        kv_window_start_tokens_ptr + batch_offsets,
        window_start_page * PAGE_SIZE,
        mask=batch_mask,
    )
    tl.store(kv_chunk_size_ptr, chunk_pages * PAGE_SIZE)

    work_offsets = tl.arange(0, BLOCK_WORK_ITEMS)
    block_valid_mask = work_offsets < block_valid_capacity
    tl.store(
        block_valid_mask_ptr + work_offsets,
        tl.zeros((BLOCK_WORK_ITEMS,), tl.int32),
        mask=block_valid_mask,
    )


@triton.jit
def build_msa_prefill_union_metadata_triton(
    q2k_indices_ptr,
    cache_seqlens_ptr,
    cu_seqlens_q_ptr,
    request_indices_ptr,
    qo_tile_indices_ptr,
    block_valid_mask_ptr,
    union_blocks_ptr,
    union_masks_ptr,
    union_counts_ptr,
    total_q_capacity,
    block_valid_capacity,
    NUM_KV_HEADS: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
    TOPK: tl.constexpr,
    TOKENS_PER_TILE: tl.constexpr,
    MAX_UNION_BLOCKS: tl.constexpr,
):
    work_idx = tl.program_id(axis=0)
    kv_head_idx = tl.program_id(axis=1)
    work_valid = work_idx < block_valid_capacity
    block_valid = tl.load(block_valid_mask_ptr + work_idx, mask=work_valid, other=0).to(
        tl.int32
    )
    active = work_valid & (block_valid != 0)
    union_base = (work_idx * NUM_KV_HEADS + kv_head_idx) * MAX_UNION_BLOCKS
    union_count_offset = work_idx * NUM_KV_HEADS + kv_head_idx
    tl.store(union_counts_ptr + union_count_offset, 0, mask=work_valid)

    request_idx = tl.load(request_indices_ptr + work_idx, mask=active, other=0).to(
        tl.int32
    )
    qo_tile_idx = tl.load(qo_tile_indices_ptr + work_idx, mask=active, other=0).to(
        tl.int32
    )
    q_start = tl.load(cu_seqlens_q_ptr + request_idx, mask=active, other=0).to(tl.int32)
    q_end = tl.load(cu_seqlens_q_ptr + request_idx + 1, mask=active, other=0).to(
        tl.int32
    )
    qo_len = q_end - q_start
    cache_len = tl.load(cache_seqlens_ptr + request_idx, mask=active, other=0).to(
        tl.int32
    )
    tile_first_token = qo_tile_idx * TOKENS_PER_TILE
    tile_token_count = tl.minimum(
        tl.maximum(qo_len - tile_first_token, 0), TOKENS_PER_TILE
    )
    union_count = tl.full((), 0, tl.int32)

    for token_idx in tl.static_range(0, TOKENS_PER_TILE):
        token_valid = active & (token_idx < tile_token_count)
        q_row_idx = q_start + tile_first_token + token_idx
        visible_len = cache_len - qo_len + tile_first_token + token_idx + 1
        visible_len = tl.maximum(visible_len, 1)
        bit = tl.full((), 1 << token_idx, tl.int32)
        for list_idx in tl.static_range(0, TOPK):
            q2k_offset = (kv_head_idx * total_q_capacity + q_row_idx) * TOPK + list_idx
            block_id = tl.load(
                q2k_indices_ptr + q2k_offset, mask=token_valid, other=-1
            ).to(tl.int32)
            valid_block = (
                token_valid & (block_id >= 0) & (block_id * BLOCK_TOKENS < visible_len)
            )
            found = tl.full((), False, tl.int1)
            found_slot = tl.full((), 0, tl.int32)
            for union_idx in tl.static_range(0, MAX_UNION_BLOCKS):
                probe = union_idx < union_count
                existing = tl.load(
                    union_blocks_ptr + union_base + union_idx,
                    mask=valid_block & probe,
                    other=-1,
                ).to(tl.int32)
                match = valid_block & probe & (existing == block_id)
                found_slot = tl.where(match & ~found, union_idx, found_slot)
                found = found | match

            do_update = valid_block & found
            old_mask = tl.load(
                union_masks_ptr + union_base + found_slot,
                mask=do_update,
                other=0,
            ).to(tl.int32)
            tl.store(
                union_masks_ptr + union_base + found_slot,
                old_mask | bit,
                mask=do_update,
            )

            do_store = valid_block & ~found & (union_count < MAX_UNION_BLOCKS)
            tl.store(
                union_blocks_ptr + union_base + union_count, block_id, mask=do_store
            )
            tl.store(union_masks_ptr + union_base + union_count, bit, mask=do_store)
            union_count += tl.where(do_store, 1, 0)

    tl.store(union_counts_ptr + union_count_offset, union_count, mask=work_valid)


@triton.jit
def update_decode_graph_compact_work_metadata_triton(
    request_indices_ptr,
    qo_tile_indices_ptr,
    kv_tile_indices_ptr,
    o_indptr_ptr,
    block_valid_mask_ptr,
    work_items_capacity,
    block_valid_capacity,
    BATCH: tl.constexpr,
    BLOCK_CHUNKS: tl.constexpr,
):
    req_idx = tl.program_id(axis=0)
    chunk_block_idx = tl.program_id(axis=1)
    chunk_offsets = chunk_block_idx * BLOCK_CHUNKS + tl.arange(0, BLOCK_CHUNKS)
    req_start = tl.load(o_indptr_ptr + req_idx).to(tl.int32)
    req_end = tl.load(o_indptr_ptr + req_idx + 1).to(tl.int32)
    num_chunks = req_end - req_start
    work_offsets = req_start + chunk_offsets
    active = (req_idx < BATCH) & (chunk_offsets < num_chunks)
    work_mask = active & (work_offsets < work_items_capacity)
    valid_mask = active & (work_offsets < block_valid_capacity)
    tl.store(request_indices_ptr + work_offsets, req_idx, mask=work_mask)
    tl.store(qo_tile_indices_ptr + work_offsets, 0, mask=work_mask)
    tl.store(
        kv_tile_indices_ptr + work_offsets, chunk_offsets.to(tl.int32), mask=work_mask
    )
    tl.store(block_valid_mask_ptr + work_offsets, 1, mask=valid_mask)


@triton.jit
def update_msa_decode_graph_metadata_fused_triton(
    cache_seqlens_ptr,
    merge_indptr_ptr,
    o_indptr_ptr,
    block_valid_mask_ptr,
    kv_chunk_size_ptr,
    kv_window_start_tokens_ptr,
    kv_chunk_size,
    block_valid_capacity,
    PAGE_SIZE: tl.constexpr,
    PAGES_PER_BLOCK: tl.constexpr,
    BATCH: tl.constexpr,
    BLOCK_BATCH: tl.constexpr,
    BLOCK_WORK_ITEMS: tl.constexpr,
):
    batch_offsets = tl.arange(0, BLOCK_BATCH)
    batch_mask = batch_offsets < BATCH
    cache_len = tl.load(cache_seqlens_ptr + batch_offsets, mask=batch_mask, other=0).to(
        tl.int32
    )
    active = batch_mask & (cache_len > 0)
    visible_blocks = tl.maximum((cache_len + 127) // 128, 1)
    selected_blocks = tl.minimum(visible_blocks, 16)
    tail_tokens = tl.maximum(cache_len - (visible_blocks - 1) * 128, 1)
    tail_pages = tl.minimum(
        tl.maximum((tail_tokens + PAGE_SIZE - 1) // PAGE_SIZE, 1), PAGES_PER_BLOCK
    )
    effective_pages = tl.maximum(
        (selected_blocks - 1) * PAGES_PER_BLOCK + tail_pages, 1
    )
    selected_tokens = effective_pages * PAGE_SIZE
    num_chunks = tl.maximum((selected_tokens + kv_chunk_size - 1) // kv_chunk_size, 1)
    num_chunks = tl.where(active, num_chunks, 0)
    prefix = tl.cumsum(tl.where(batch_mask, num_chunks, 0), 0)

    tl.store(merge_indptr_ptr, 0)
    tl.store(o_indptr_ptr, 0)
    tl.store(merge_indptr_ptr + batch_offsets + 1, prefix, mask=batch_mask)
    tl.store(o_indptr_ptr + batch_offsets + 1, prefix, mask=batch_mask)
    tl.store(
        kv_window_start_tokens_ptr + batch_offsets,
        tl.zeros((BLOCK_BATCH,), tl.int32),
        mask=batch_mask,
    )
    tl.store(kv_chunk_size_ptr, kv_chunk_size)

    work_offsets = tl.arange(0, BLOCK_WORK_ITEMS)
    tl.store(
        block_valid_mask_ptr + work_offsets,
        tl.zeros((BLOCK_WORK_ITEMS,), tl.int32),
        mask=work_offsets < block_valid_capacity,
    )


@triton.jit
def update_regular_decode_graph_metadata_fused_triton(
    cache_seqlens_ptr,
    merge_indptr_ptr,
    o_indptr_ptr,
    kv_chunk_size_ptr,
    kv_window_start_tokens_ptr,
    decode_chunk_pages_lut_ptr,
    kv_chunk_size_tensor_ptr,
    lut_size,
    PAGE_SIZE: tl.constexpr,
    WINDOW_PAGE_SPAN: tl.constexpr,
    WINDOW_LEFT: tl.constexpr,
    BATCH: tl.constexpr,
    BLOCK_BATCH: tl.constexpr,
    USE_LUT: tl.constexpr,
    HAS_KV_CHUNK_SIZE_TENSOR: tl.constexpr,
    FIXED_KV_CHUNK_SIZE: tl.constexpr,
):
    batch_offsets = tl.arange(0, BLOCK_BATCH)
    batch_mask = batch_offsets < BATCH
    cache_len = tl.load(cache_seqlens_ptr + batch_offsets, mask=batch_mask, other=0).to(
        tl.int32
    )
    num_pages = tl.maximum((cache_len + (PAGE_SIZE - 1)) // PAGE_SIZE, 1)
    window_start_page = tl.full((BLOCK_BATCH,), 0, tl.int32)
    if WINDOW_LEFT >= 0:
        window_start_token = tl.maximum(cache_len - 1 - WINDOW_LEFT, 0)
        window_start_page = window_start_token // PAGE_SIZE
    effective_pages = tl.maximum(num_pages - window_start_page, 1)

    if USE_LUT:
        policy_pages = tl.max(tl.where(batch_mask, effective_pages, 1), axis=0)
        if WINDOW_PAGE_SPAN > 0:
            policy_pages = tl.minimum(policy_pages, WINDOW_PAGE_SPAN)
        policy_pages = tl.minimum(tl.maximum(policy_pages, 1), lut_size - 1)
        chunk_pages = tl.load(decode_chunk_pages_lut_ptr + policy_pages).to(tl.int32)
        kv_chunk_size = chunk_pages * PAGE_SIZE
    elif HAS_KV_CHUNK_SIZE_TENSOR:
        kv_chunk_size = tl.load(kv_chunk_size_tensor_ptr).to(tl.int32)
    else:
        kv_chunk_size = tl.full((), FIXED_KV_CHUNK_SIZE, tl.int32)

    effective_tokens = effective_pages * PAGE_SIZE
    num_chunks = tl.maximum((effective_tokens + kv_chunk_size - 1) // kv_chunk_size, 1)
    prefix = tl.cumsum(tl.where(batch_mask, num_chunks, 0), 0)

    tl.store(merge_indptr_ptr, 0)
    tl.store(o_indptr_ptr, 0)
    tl.store(
        merge_indptr_ptr + batch_offsets + 1,
        prefix,
        mask=batch_mask,
    )
    tl.store(
        o_indptr_ptr + batch_offsets + 1,
        prefix,
        mask=batch_mask,
    )
    tl.store(
        kv_window_start_tokens_ptr + batch_offsets,
        window_start_page * PAGE_SIZE,
        mask=batch_mask,
    )
    tl.store(kv_chunk_size_ptr, kv_chunk_size)


@triton.jit
def update_decode_graph_window_start_tokens_triton(
    cache_seqlens_ptr,
    kv_window_start_tokens_ptr,
    PAGE_SIZE: tl.constexpr,
    WINDOW_LEFT: tl.constexpr,
    BATCH: tl.constexpr,
    BLOCK_BATCH: tl.constexpr,
):
    batch_offsets = tl.arange(0, BLOCK_BATCH)
    batch_mask = batch_offsets < BATCH
    cache_len = tl.load(cache_seqlens_ptr + batch_offsets, mask=batch_mask, other=0).to(
        tl.int32
    )
    window_start_token = tl.maximum(cache_len - 1 - WINDOW_LEFT, 0)
    window_start_page = window_start_token // PAGE_SIZE
    tl.store(
        kv_window_start_tokens_ptr + batch_offsets,
        window_start_page * PAGE_SIZE,
        mask=batch_mask,
    )


@triton.jit
def update_prefill_graph_work_metadata_triton(
    cache_seqlens_ptr,
    cu_seqlens_q_ptr,
    request_indices_ptr,
    qo_tile_indices_ptr,
    kv_tile_indices_ptr,
    o_indptr_ptr,
    block_valid_mask_ptr,
    kv_chunk_size_ptr,
    kv_window_start_tokens_ptr,
    work_items_capacity: tl.constexpr,
    block_valid_capacity: tl.constexpr,
    BATCH: tl.constexpr,
    MAX_Q_TILES_PER_REQ: tl.constexpr,
    MAX_CHUNKS_PER_Q_TILE: tl.constexpr,
    CTA_TILE_Q: tl.constexpr,
    GQA_GROUP_SIZE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    WINDOW_LEFT: tl.constexpr,
    SPLIT_KV: tl.constexpr,
    BLOCK_WORK_ITEMS: tl.constexpr,
):
    block_idx = tl.program_id(axis=0)
    offsets = block_idx * BLOCK_WORK_ITEMS + tl.arange(0, BLOCK_WORK_ITEMS)
    slots_per_req = MAX_Q_TILES_PER_REQ * MAX_CHUNKS_PER_Q_TILE
    req_idx = offsets // slots_per_req
    req_local = offsets - req_idx * slots_per_req
    q_tile_idx = req_local // MAX_CHUNKS_PER_Q_TILE
    kv_tile_idx = req_local - q_tile_idx * MAX_CHUNKS_PER_Q_TILE

    in_block_capacity = offsets < block_valid_capacity
    in_work_capacity = offsets < work_items_capacity
    in_batch = req_idx < BATCH
    usable = in_block_capacity & in_work_capacity & in_batch

    q_start = tl.load(cu_seqlens_q_ptr + req_idx, mask=usable, other=0).to(tl.int32)
    q_end = tl.load(cu_seqlens_q_ptr + req_idx + 1, mask=usable, other=0).to(tl.int32)
    q_len = tl.maximum(q_end - q_start, 0)
    cache_len = tl.load(cache_seqlens_ptr + req_idx, mask=usable, other=0).to(tl.int32)

    packed_q_len = q_len * GQA_GROUP_SIZE
    num_q_tiles = (packed_q_len + CTA_TILE_Q - 1) // CTA_TILE_Q
    num_pages = tl.maximum((cache_len + PAGE_SIZE - 1) // PAGE_SIZE, 1)

    window_start_page = tl.full((BLOCK_WORK_ITEMS,), 0, tl.int32)
    if WINDOW_LEFT >= 0:
        first_causal_key = tl.maximum(cache_len - tl.maximum(q_len, 1), 0)
        first_window_key = tl.maximum(first_causal_key - WINDOW_LEFT, 0)
        window_start_page = first_window_key // PAGE_SIZE
        window_start_page = tl.minimum(window_start_page, tl.maximum(num_pages - 1, 0))
    effective_pages = tl.maximum(num_pages - window_start_page, 1)

    kv_chunk_size = tl.load(kv_chunk_size_ptr).to(tl.int32)
    chunk_pages = tl.maximum((kv_chunk_size + PAGE_SIZE - 1) // PAGE_SIZE, 1)
    num_chunks = tl.full((BLOCK_WORK_ITEMS,), 1, tl.int32)
    if SPLIT_KV:
        num_chunks = tl.maximum((effective_pages + chunk_pages - 1) // chunk_pages, 1)

    active = (
        usable
        & (q_tile_idx < num_q_tiles)
        & (kv_tile_idx < num_chunks)
        & (q_len > 0)
        & (cache_len > 0)
    )
    tl.store(
        block_valid_mask_ptr + offsets, active.to(tl.int32), mask=in_block_capacity
    )
    tl.store(request_indices_ptr + offsets, req_idx.to(tl.int32), mask=in_work_capacity)
    tl.store(
        qo_tile_indices_ptr + offsets, q_tile_idx.to(tl.int32), mask=in_work_capacity
    )
    tl.store(
        kv_tile_indices_ptr + offsets, kv_tile_idx.to(tl.int32), mask=in_work_capacity
    )

    first_slot_for_req = usable & (req_local == 0)
    tl.store(
        kv_window_start_tokens_ptr + req_idx,
        (window_start_page * PAGE_SIZE).to(tl.int32),
        mask=first_slot_for_req,
    )
    tl.store(
        o_indptr_ptr + req_idx + 1,
        (q_len * num_chunks).to(tl.int32),
        mask=first_slot_for_req,
    )
    tl.store(o_indptr_ptr + offsets, 0, mask=offsets == 0)


@triton.jit
def update_prefill_graph_row_indptr_triton(
    cu_seqlens_q_ptr,
    o_indptr_ptr,
    merge_indptr_ptr,
    BATCH: tl.constexpr,
    MAX_Q_ROWS_PER_REQ: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    req_idx = tl.program_id(axis=0)
    row_block_idx = tl.program_id(axis=1)
    rows = row_block_idx * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)

    q_start = tl.load(cu_seqlens_q_ptr + req_idx).to(tl.int32)
    q_end = tl.load(cu_seqlens_q_ptr + req_idx + 1).to(tl.int32)
    q_len = tl.maximum(q_end - q_start, 0)
    request_partial_start = tl.load(o_indptr_ptr + req_idx).to(tl.int32)
    request_partial_end = tl.load(o_indptr_ptr + req_idx + 1).to(tl.int32)
    safe_q_len = tl.maximum(q_len, 1)
    num_chunks = tl.maximum(
        (request_partial_end - request_partial_start) // safe_q_len, 1
    )

    row_mask = rows < q_len
    tl.store(
        merge_indptr_ptr + q_start + rows + 1,
        request_partial_start + (rows + 1) * num_chunks,
        mask=row_mask,
    )
    if BATCH > 0:
        tl.store(merge_indptr_ptr, 0, mask=(req_idx == 0) & (row_block_idx == 0))


def make_decode_chunk_pages_lut_tensor(
    decode_chunk_pages_lut: Sequence[int],
    *,
    device: torch.device,
) -> torch.Tensor:
    if not decode_chunk_pages_lut:
        raise ValueError("decode chunk-pages LUT must be non-empty")
    if any(int(chunk_pages) <= 0 for chunk_pages in decode_chunk_pages_lut):
        raise ValueError("decode chunk-pages LUT must contain only positive values")
    return torch.tensor(
        (
            int(decode_chunk_pages_lut[0]),
            *(int(chunk_pages) for chunk_pages in decode_chunk_pages_lut),
        ),
        dtype=torch.int32,
        device=device,
    )


def summarize_decode_chunk_pages_lut(
    decode_chunk_pages_lut: Sequence[int],
) -> tuple[int, int]:
    if not decode_chunk_pages_lut:
        raise ValueError("decode chunk-pages LUT must be non-empty")
    worst_page_count = 1
    max_chunks_per_req = 1
    for page_count, chunk_pages in enumerate(decode_chunk_pages_lut, start=1):
        num_chunks = (page_count + int(chunk_pages) - 1) // int(chunk_pages)
        if num_chunks > max_chunks_per_req:
            max_chunks_per_req = num_chunks
            worst_page_count = page_count
    return int(worst_page_count), int(max_chunks_per_req)


def build_msa_prefill_union_metadata(
    *,
    q2k_indices: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    request_indices: torch.Tensor,
    qo_tile_indices: torch.Tensor,
    block_valid_mask: torch.Tensor,
    union_blocks: torch.Tensor,
    union_masks: torch.Tensor,
    union_counts: torch.Tensor,
) -> None:
    if q2k_indices.ndim != 3 or int(q2k_indices.shape[2]) != _MSA_UNION_TOPK:
        raise ValueError("q2k_indices must have shape [kv_heads, total_q_capacity, 16]")
    if q2k_indices.dtype != torch.int32 or not q2k_indices.is_contiguous():
        raise TypeError("q2k_indices must be contiguous torch.int32")
    if union_blocks.dtype != torch.int32 or union_masks.dtype != torch.int32:
        raise TypeError("MSA union block/mask buffers must be torch.int32")
    if union_counts.dtype != torch.int32:
        raise TypeError("MSA union count buffer must be torch.int32")
    if union_blocks.ndim != 3 or union_masks.ndim != 3:
        raise ValueError("MSA union block/mask buffers must be rank-3")
    if union_counts.ndim != 2:
        raise ValueError("MSA union count buffer must be rank-2")
    if tuple(union_blocks.shape) != tuple(union_masks.shape):
        raise ValueError("MSA union block and mask buffers must have matching shapes")
    if int(union_blocks.shape[1]) != int(q2k_indices.shape[0]):
        raise ValueError(
            "MSA union buffers must use the same kv-head count as q2k_indices"
        )
    if int(union_blocks.shape[2]) < _MSA_UNION_MAX_BLOCKS:
        raise ValueError("MSA union buffers need capacity for 128 blocks per tile")
    if tuple(union_counts.shape) != tuple(union_blocks.shape[:2]):
        raise ValueError("MSA union count buffer shape must match union_blocks[:2]")
    work_capacity = int(block_valid_mask.shape[0])
    if int(union_blocks.shape[0]) < work_capacity:
        raise ValueError("MSA union buffers are smaller than block_valid_mask capacity")
    if (
        int(request_indices.shape[0]) < work_capacity
        or int(qo_tile_indices.shape[0]) < work_capacity
    ):
        raise ValueError(
            "MSA worklist buffers are smaller than block_valid_mask capacity"
        )
    device = q2k_indices.device
    tensors = (
        cache_seqlens,
        cu_seqlens_q,
        request_indices,
        qo_tile_indices,
        block_valid_mask,
        union_blocks,
        union_masks,
        union_counts,
    )
    if any(t.device != device for t in tensors):
        raise ValueError("MSA union metadata tensors must all be on the q2k device")
    build_msa_prefill_union_metadata_triton[(work_capacity, int(q2k_indices.shape[0]))](
        q2k_indices,
        cache_seqlens,
        cu_seqlens_q,
        request_indices,
        qo_tile_indices,
        block_valid_mask,
        union_blocks,
        union_masks,
        union_counts,
        int(q2k_indices.shape[1]),
        work_capacity,
        NUM_KV_HEADS=int(q2k_indices.shape[0]),
        BLOCK_TOKENS=128,
        TOPK=_MSA_UNION_TOPK,
        TOKENS_PER_TILE=_MSA_UNION_TOKENS_PER_TILE,
        MAX_UNION_BLOCKS=_MSA_UNION_MAX_BLOCKS,
        num_warps=1,
    )


def stage_decode_cuda_graph_metadata(
    *,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    page_table: torch.Tensor,
    page_size: int,
    swa_page_table: torch.Tensor | None = None,
    swa_index_mapping: torch.Tensor | None = None,
) -> None:
    device = page_table.device
    if req_to_token.device != device:
        raise ValueError("req_to_token and page_table must be on the same device")
    if req_pool_indices.device != device:
        raise ValueError("req_pool_indices and page_table must be on the same device")
    if seq_lens.device != device or cache_seqlens.device != device:
        raise ValueError(
            "seq_lens/cache_seqlens and page_table must be on the same device"
        )
    if cu_seqlens_q.device != device:
        raise ValueError("cu_seqlens_q and page_table must be on the same device")
    if page_size <= 0:
        raise ValueError("page_size must be positive")
    if page_table.ndim != 2:
        raise ValueError("page_table must be rank-2")

    bs = int(page_table.shape[0])
    max_pages = int(page_table.shape[1])
    if bs <= 0:
        raise ValueError("decode graph staging requires bs > 0")
    if max_pages <= 0:
        raise ValueError("page_table must have at least one page column")
    if int(req_pool_indices.shape[0]) < bs:
        raise ValueError("req_pool_indices is smaller than the graph batch")
    if int(seq_lens.shape[0]) < bs:
        raise ValueError("seq_lens is smaller than the graph batch")
    if int(cache_seqlens.shape[0]) < bs:
        raise ValueError("cache_seqlens is smaller than the graph batch")
    if int(cu_seqlens_q.shape[0]) < bs + 1:
        raise ValueError("cu_seqlens_q is smaller than the graph batch")

    has_swa = swa_page_table is not None
    if has_swa != (swa_index_mapping is not None):
        raise ValueError(
            "swa_page_table and swa_index_mapping must be provided together"
        )
    if has_swa:
        assert swa_page_table is not None
        assert swa_index_mapping is not None
        if swa_page_table.device != device or swa_index_mapping.device != device:
            raise ValueError("SWA staging tensors must be on the page_table device")
        if tuple(swa_page_table.shape) != tuple(page_table.shape):
            raise ValueError("swa_page_table shape must match page_table shape")
        if swa_index_mapping.numel() <= 0:
            raise ValueError("swa_index_mapping must be non-empty")

    page_blocks = triton.cdiv(max_pages, _DECODE_BLOCK_PAGES)
    stage_decode_cuda_graph_metadata_triton[(bs, page_blocks)](
        req_to_token,
        req_pool_indices,
        seq_lens,
        cache_seqlens,
        cu_seqlens_q,
        page_table,
        page_table if swa_page_table is None else swa_page_table,
        req_to_token if swa_index_mapping is None else swa_index_mapping,
        req_to_token.stride(0),
        req_to_token.numel(),
        page_table.stride(0),
        page_table.stride(0) if swa_page_table is None else swa_page_table.stride(0),
        req_pool_indices.stride(0),
        seq_lens.stride(0),
        1 if swa_index_mapping is None else swa_index_mapping.numel(),
        PAGE_SIZE=page_size,
        BATCH=bs,
        MAX_PAGES=max_pages,
        HAS_SWA=has_swa,
        BLOCK_PAGES=_DECODE_BLOCK_PAGES,
    )


def patch_decode_cuda_graph_current_pages(
    *,
    cache_seqlens: torch.Tensor,
    page_table: torch.Tensor,
    out_cache_loc: torch.Tensor,
    page_size: int,
    fill_value: int = 1,
    swa_page_table: torch.Tensor | None = None,
    out_cache_loc_swa: torch.Tensor | None = None,
    swa_index_mapping: torch.Tensor | None = None,
) -> None:
    device = page_table.device
    if cache_seqlens.device != device or out_cache_loc.device != device:
        raise ValueError(
            "cache_seqlens/out_cache_loc and page_table must share a device"
        )
    if page_size <= 0:
        raise ValueError("page_size must be positive")
    if page_table.ndim != 2:
        raise ValueError("page_table must be rank-2")

    bs = int(cache_seqlens.shape[0])
    max_pages = int(page_table.shape[1])
    if bs <= 0:
        raise ValueError("decode graph current-page patch requires bs > 0")
    if int(page_table.shape[0]) < bs:
        raise ValueError("page_table is smaller than cache_seqlens")
    if max_pages <= 0:
        raise ValueError("page_table must have at least one page column")
    if int(out_cache_loc.shape[0]) < bs:
        raise ValueError("out_cache_loc is smaller than the graph batch")

    has_swa = swa_page_table is not None
    has_out_cache_loc_swa = out_cache_loc_swa is not None
    if has_swa:
        assert swa_page_table is not None
        if swa_page_table.device != device:
            raise ValueError("swa_page_table and page_table must share a device")
        if tuple(swa_page_table.shape) != tuple(page_table.shape):
            raise ValueError("swa_page_table shape must match page_table shape")
        if has_out_cache_loc_swa:
            assert out_cache_loc_swa is not None
            if out_cache_loc_swa.device != device:
                raise ValueError("out_cache_loc_swa and page_table must share a device")
            if int(out_cache_loc_swa.shape[0]) < bs:
                raise ValueError("out_cache_loc_swa is smaller than the graph batch")
        elif swa_index_mapping is None:
            raise ValueError(
                "swa_index_mapping is required when out_cache_loc_swa is not provided"
            )
        else:
            if swa_index_mapping.device != device:
                raise ValueError("swa_index_mapping and page_table must share a device")
            if swa_index_mapping.numel() <= 0:
                raise ValueError("swa_index_mapping must be non-empty")

    page_blocks = triton.cdiv(max_pages, _DECODE_BLOCK_PAGES)
    patch_decode_cuda_graph_current_pages_triton[(bs, page_blocks)](
        cache_seqlens,
        page_table,
        page_table if swa_page_table is None else swa_page_table,
        out_cache_loc,
        out_cache_loc if out_cache_loc_swa is None else out_cache_loc_swa,
        page_table if swa_index_mapping is None else swa_index_mapping,
        page_table.stride(0),
        page_table.stride(0) if swa_page_table is None else swa_page_table.stride(0),
        out_cache_loc.stride(0),
        out_cache_loc.stride(0)
        if out_cache_loc_swa is None
        else out_cache_loc_swa.stride(0),
        1 if swa_index_mapping is None else swa_index_mapping.numel(),
        PAGE_SIZE=page_size,
        FILL_VALUE=int(fill_value),
        MAX_PAGES=max_pages,
        HAS_SWA=has_swa,
        HAS_OUT_CACHE_LOC_SWA=has_out_cache_loc_swa,
        BLOCK_PAGES=_DECODE_BLOCK_PAGES,
    )


def update_decode_graph_chunk_metadata_fused(
    *,
    cache_seqlens: torch.Tensor,
    request_indices: torch.Tensor,
    qo_tile_indices: torch.Tensor,
    kv_tile_indices: torch.Tensor,
    merge_indptr: torch.Tensor,
    o_indptr: torch.Tensor,
    block_valid_mask: torch.Tensor,
    kv_chunk_size_ptr: torch.Tensor,
    kv_window_start_tokens: torch.Tensor,
    decode_chunk_pages_lut: torch.Tensor,
    page_size: int,
    window_page_span: int = 0,
    window_left: int = -1,
) -> None:
    device = cache_seqlens.device
    if request_indices.device != device:
        raise ValueError("request_indices and cache_seqlens must be on the same device")
    if qo_tile_indices.device != device or kv_tile_indices.device != device:
        raise ValueError(
            "tile index buffers and cache_seqlens must be on the same device"
        )
    if merge_indptr.device != device or o_indptr.device != device:
        raise ValueError("indptr buffers and cache_seqlens must be on the same device")
    if block_valid_mask.device != device or kv_chunk_size_ptr.device != device:
        raise ValueError(
            "decode graph buffers and cache_seqlens must be on the same device"
        )
    if kv_window_start_tokens.device != device:
        raise ValueError(
            "kv_window_start_tokens and cache_seqlens must be on the same device"
        )
    if decode_chunk_pages_lut.device != device:
        raise ValueError(
            "decode_chunk_pages_lut and cache_seqlens must be on the same device"
        )
    if page_size <= 0:
        raise ValueError("page_size must be positive")
    if window_left < -1:
        raise ValueError("window_left must be -1 or non-negative")

    bs = int(cache_seqlens.shape[0])
    if bs <= 0:
        raise ValueError("decode graph replay requires bs > 0")
    work_items_capacity = int(request_indices.shape[0])
    block_valid_capacity = int(block_valid_mask.shape[0])
    if (
        int(qo_tile_indices.shape[0]) != work_items_capacity
        or int(kv_tile_indices.shape[0]) != work_items_capacity
    ):
        raise RuntimeError(
            "decode graph tile index buffers must match request_indices shape"
        )
    if work_items_capacity % bs != 0:
        raise RuntimeError(
            "decode graph workspace request_indices shape is incompatible with the batch bucket"
        )
    max_chunks_per_req = work_items_capacity // bs
    if max_chunks_per_req <= 0:
        raise RuntimeError(
            "decode graph workspace must allocate at least one chunk per request"
        )
    if int(merge_indptr.shape[0]) < bs + 1 or int(o_indptr.shape[0]) < bs + 1:
        raise RuntimeError(
            "decode graph indptr buffers are smaller than the graph batch"
        )
    if int(kv_window_start_tokens.shape[0]) < bs:
        raise RuntimeError(
            "decode graph kv_window_start_tokens is smaller than the graph batch"
        )
    if int(decode_chunk_pages_lut.shape[0]) < 2:
        raise RuntimeError(
            "decode graph chunk-pages LUT must contain at least two entries"
        )

    update_decode_graph_metadata_fused_triton[(1,)](
        cache_seqlens,
        request_indices,
        qo_tile_indices,
        kv_tile_indices,
        merge_indptr,
        o_indptr,
        block_valid_mask,
        kv_chunk_size_ptr,
        kv_window_start_tokens,
        decode_chunk_pages_lut,
        max_chunks_per_req,
        block_valid_capacity,
        decode_chunk_pages_lut.shape[0],
        PAGE_SIZE=page_size,
        WINDOW_PAGE_SPAN=int(window_page_span),
        WINDOW_LEFT=int(window_left),
        BATCH=bs,
        BLOCK_BATCH=triton.next_power_of_2(bs),
        BLOCK_WORK_ITEMS=triton.next_power_of_2(
            max(work_items_capacity, block_valid_capacity)
        ),
    )
    update_decode_graph_compact_work_metadata_triton[
        (bs, triton.cdiv(max_chunks_per_req, _DECODE_BLOCK_CHUNKS))
    ](
        request_indices,
        qo_tile_indices,
        kv_tile_indices,
        o_indptr,
        block_valid_mask,
        work_items_capacity,
        block_valid_capacity,
        BATCH=bs,
        BLOCK_CHUNKS=_DECODE_BLOCK_CHUNKS,
    )


def update_decode_graph_replay_metadata(
    *,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    request_indices: torch.Tensor,
    qo_tile_indices: torch.Tensor,
    kv_tile_indices: torch.Tensor,
    merge_indptr: torch.Tensor,
    o_indptr: torch.Tensor,
    block_valid_mask: torch.Tensor,
    kv_chunk_size_ptr: torch.Tensor,
    kv_window_start_tokens: torch.Tensor,
    decode_chunk_pages_lut: torch.Tensor,
    page_size: int,
    window_page_span: int = 0,
    window_left: int = -1,
) -> None:
    if req_to_token.device != page_table.device:
        raise ValueError("req_to_token and page_table must be on the same device")
    if req_pool_indices.device != page_table.device:
        raise ValueError("req_pool_indices and page_table must be on the same device")
    if cache_seqlens.device != page_table.device:
        raise ValueError("cache_seqlens and page_table must be on the same device")
    if (
        qo_tile_indices.device != page_table.device
        or kv_tile_indices.device != page_table.device
    ):
        raise ValueError("tile index buffers and page_table must be on the same device")
    if decode_chunk_pages_lut.device != page_table.device:
        raise ValueError(
            "decode_chunk_pages_lut and page_table must be on the same device"
        )
    if page_size <= 0:
        raise ValueError("page_size must be positive")

    bs = int(cache_seqlens.shape[0])
    if bs <= 0:
        raise ValueError("decode graph replay requires bs > 0")
    if int(req_pool_indices.shape[0]) != bs:
        raise ValueError("req_pool_indices shape must match cache_seqlens batch")
    work_items_capacity = int(request_indices.shape[0])
    if (
        int(qo_tile_indices.shape[0]) != work_items_capacity
        or int(kv_tile_indices.shape[0]) != work_items_capacity
    ):
        raise RuntimeError(
            "decode graph tile index buffers must match request_indices shape"
        )
    if work_items_capacity % bs != 0:
        raise RuntimeError(
            "decode graph workspace request_indices shape is incompatible with the batch bucket"
        )
    max_chunks_per_req = work_items_capacity // bs
    if max_chunks_per_req <= 0:
        raise RuntimeError(
            "decode graph workspace must allocate at least one chunk per request"
        )

    page_blocks = triton.cdiv(int(page_table.shape[1]), _DECODE_BLOCK_PAGES)
    build_decode_graph_page_table_full_triton[(bs, page_blocks)](
        req_to_token,
        req_pool_indices,
        page_table,
        req_to_token.stride(0),
        req_to_token.numel(),
        page_table.stride(0),
        PAGE_SIZE=page_size,
        MAX_PAGES=int(page_table.shape[1]),
        BLOCK_PAGES=_DECODE_BLOCK_PAGES,
    )

    update_decode_graph_chunk_metadata_fused(
        cache_seqlens=cache_seqlens,
        request_indices=request_indices,
        qo_tile_indices=qo_tile_indices,
        kv_tile_indices=kv_tile_indices,
        merge_indptr=merge_indptr,
        o_indptr=o_indptr,
        block_valid_mask=block_valid_mask,
        kv_chunk_size_ptr=kv_chunk_size_ptr,
        kv_window_start_tokens=kv_window_start_tokens,
        decode_chunk_pages_lut=decode_chunk_pages_lut,
        page_size=page_size,
        window_page_span=window_page_span,
        window_left=window_left,
    )


def update_msa_decode_graph_chunk_metadata(
    *,
    cache_seqlens: torch.Tensor,
    request_indices: torch.Tensor,
    qo_tile_indices: torch.Tensor,
    kv_tile_indices: torch.Tensor,
    merge_indptr: torch.Tensor,
    o_indptr: torch.Tensor,
    block_valid_mask: torch.Tensor,
    kv_chunk_size_ptr: torch.Tensor,
    kv_window_start_tokens: torch.Tensor,
    kv_chunk_size: int,
    page_size: int,
) -> None:
    device = cache_seqlens.device
    if request_indices.device != device:
        raise ValueError("request_indices and cache_seqlens must be on the same device")
    if qo_tile_indices.device != device or kv_tile_indices.device != device:
        raise ValueError(
            "tile index buffers and cache_seqlens must be on the same device"
        )
    if merge_indptr.device != device or o_indptr.device != device:
        raise ValueError("indptr buffers and cache_seqlens must be on the same device")
    if block_valid_mask.device != device or kv_chunk_size_ptr.device != device:
        raise ValueError(
            "MSA decode graph buffers and cache_seqlens must be on the same device"
        )
    if kv_window_start_tokens.device != device:
        raise ValueError(
            "kv_window_start_tokens and cache_seqlens must be on the same device"
        )
    if page_size not in (64, 128):
        raise ValueError(
            "MSA decode graph replay requires page_size=64 or page_size=128"
        )
    kv_chunk_size = int(kv_chunk_size)
    # Chunk boundaries are 64-token-tile aligned in the compacted virtual key
    # space regardless of the physical page size.
    if kv_chunk_size <= 0 or kv_chunk_size % 64 != 0:
        raise ValueError(
            "MSA decode graph kv_chunk_size must be a positive multiple of 64 tokens"
        )

    bs = int(cache_seqlens.shape[0])
    if bs <= 0:
        raise ValueError("MSA decode graph replay requires bs > 0")
    work_items_capacity = int(request_indices.shape[0])
    block_valid_capacity = int(block_valid_mask.shape[0])
    if (
        int(qo_tile_indices.shape[0]) != work_items_capacity
        or int(kv_tile_indices.shape[0]) != work_items_capacity
    ):
        raise RuntimeError(
            "MSA decode graph tile index buffers must match request_indices shape"
        )
    if work_items_capacity % bs != 0:
        raise RuntimeError(
            "MSA decode graph workspace request_indices shape is incompatible with the batch bucket"
        )
    max_chunks_per_req = work_items_capacity // bs
    if max_chunks_per_req <= 0:
        raise RuntimeError(
            "MSA decode graph workspace must allocate at least one chunk per request"
        )
    max_required_chunks = triton.cdiv(16 * 128, kv_chunk_size)
    if max_chunks_per_req < max_required_chunks:
        raise RuntimeError(
            "MSA decode graph workspace request_indices capacity is too small for the chunk size"
        )
    if int(merge_indptr.shape[0]) < bs + 1 or int(o_indptr.shape[0]) < bs + 1:
        raise RuntimeError(
            "MSA decode graph indptr buffers are smaller than the graph batch"
        )
    if int(kv_window_start_tokens.shape[0]) < bs:
        raise RuntimeError(
            "MSA decode graph kv_window_start_tokens is smaller than the graph batch"
        )

    update_msa_decode_graph_metadata_fused_triton[(1,)](
        cache_seqlens,
        merge_indptr,
        o_indptr,
        block_valid_mask,
        kv_chunk_size_ptr,
        kv_window_start_tokens,
        kv_chunk_size,
        block_valid_capacity,
        PAGE_SIZE=page_size,
        PAGES_PER_BLOCK=128 // int(page_size),
        BATCH=bs,
        BLOCK_BATCH=triton.next_power_of_2(bs),
        BLOCK_WORK_ITEMS=triton.next_power_of_2(
            max(work_items_capacity, block_valid_capacity)
        ),
    )
    update_decode_graph_compact_work_metadata_triton[
        (bs, triton.cdiv(max_chunks_per_req, _DECODE_BLOCK_CHUNKS))
    ](
        request_indices,
        qo_tile_indices,
        kv_tile_indices,
        o_indptr,
        block_valid_mask,
        work_items_capacity,
        block_valid_capacity,
        BATCH=bs,
        BLOCK_CHUNKS=_DECODE_BLOCK_CHUNKS,
    )


def update_msa_decode_graph_replay_metadata(
    *,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    request_indices: torch.Tensor,
    qo_tile_indices: torch.Tensor,
    kv_tile_indices: torch.Tensor,
    merge_indptr: torch.Tensor,
    o_indptr: torch.Tensor,
    block_valid_mask: torch.Tensor,
    kv_chunk_size_ptr: torch.Tensor,
    kv_window_start_tokens: torch.Tensor,
    kv_chunk_size: int,
    page_size: int,
) -> None:
    if req_to_token.device != page_table.device:
        raise ValueError("req_to_token and page_table must be on the same device")
    if req_pool_indices.device != page_table.device:
        raise ValueError("req_pool_indices and page_table must be on the same device")
    if cache_seqlens.device != page_table.device:
        raise ValueError("cache_seqlens and page_table must be on the same device")
    if page_size not in (64, 128):
        raise ValueError(
            "MSA decode graph replay requires page_size=64 or page_size=128"
        )

    bs = int(cache_seqlens.shape[0])
    if bs <= 0:
        raise ValueError("MSA decode graph replay requires bs > 0")
    if int(req_pool_indices.shape[0]) != bs:
        raise ValueError("req_pool_indices shape must match cache_seqlens batch")

    page_blocks = triton.cdiv(int(page_table.shape[1]), _DECODE_BLOCK_PAGES)
    build_decode_graph_page_table_full_triton[(bs, page_blocks)](
        req_to_token,
        req_pool_indices,
        page_table,
        req_to_token.stride(0),
        req_to_token.numel(),
        page_table.stride(0),
        PAGE_SIZE=page_size,
        MAX_PAGES=int(page_table.shape[1]),
        BLOCK_PAGES=_DECODE_BLOCK_PAGES,
    )
    update_msa_decode_graph_chunk_metadata(
        cache_seqlens=cache_seqlens,
        request_indices=request_indices,
        qo_tile_indices=qo_tile_indices,
        kv_tile_indices=kv_tile_indices,
        merge_indptr=merge_indptr,
        o_indptr=o_indptr,
        block_valid_mask=block_valid_mask,
        kv_chunk_size_ptr=kv_chunk_size_ptr,
        kv_window_start_tokens=kv_window_start_tokens,
        kv_chunk_size=kv_chunk_size,
        page_size=page_size,
    )


def update_regular_decode_graph_replay_metadata(
    *,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    merge_indptr: torch.Tensor,
    o_indptr: torch.Tensor,
    kv_chunk_size_ptr: torch.Tensor,
    kv_window_start_tokens: torch.Tensor,
    decode_chunk_pages_lut: torch.Tensor,
    page_size: int,
    window_page_span: int = 0,
    window_left: int = -1,
) -> None:
    if req_to_token.device != page_table.device:
        raise ValueError("req_to_token and page_table must be on the same device")
    if req_pool_indices.device != page_table.device:
        raise ValueError("req_pool_indices and page_table must be on the same device")
    if cache_seqlens.device != page_table.device:
        raise ValueError("cache_seqlens and page_table must be on the same device")
    if decode_chunk_pages_lut.device != page_table.device:
        raise ValueError(
            "decode_chunk_pages_lut and page_table must be on the same device"
        )
    if page_size <= 0:
        raise ValueError("page_size must be positive")

    bs = int(cache_seqlens.shape[0])
    if bs <= 0:
        raise ValueError("decode graph replay requires bs > 0")
    if int(req_pool_indices.shape[0]) != bs:
        raise ValueError("req_pool_indices shape must match cache_seqlens batch")

    page_blocks = triton.cdiv(int(page_table.shape[1]), _DECODE_BLOCK_PAGES)
    build_decode_graph_page_table_full_triton[(bs, page_blocks)](
        req_to_token,
        req_pool_indices,
        page_table,
        req_to_token.stride(0),
        req_to_token.numel(),
        page_table.stride(0),
        PAGE_SIZE=page_size,
        MAX_PAGES=int(page_table.shape[1]),
        BLOCK_PAGES=_DECODE_BLOCK_PAGES,
    )

    update_regular_decode_graph_chunk_metadata_from_lut(
        cache_seqlens=cache_seqlens,
        merge_indptr=merge_indptr,
        o_indptr=o_indptr,
        kv_chunk_size_ptr=kv_chunk_size_ptr,
        kv_window_start_tokens=kv_window_start_tokens,
        decode_chunk_pages_lut=decode_chunk_pages_lut,
        page_size=page_size,
        window_page_span=window_page_span,
        window_left=window_left,
    )


def _launch_regular_decode_graph_metadata_fused(
    *,
    cache_seqlens: torch.Tensor,
    merge_indptr: torch.Tensor,
    o_indptr: torch.Tensor,
    kv_chunk_size_ptr: torch.Tensor,
    kv_window_start_tokens: torch.Tensor,
    page_size: int,
    window_page_span: int,
    window_left: int,
    decode_chunk_pages_lut: torch.Tensor | None = None,
    kv_chunk_size: int | torch.Tensor | None = None,
) -> None:
    bs = int(cache_seqlens.shape[0])
    if bs <= 0:
        raise ValueError("decode graph replay requires bs > 0")
    if page_size <= 0:
        raise ValueError("page_size must be positive")
    if window_left < -1:
        raise ValueError("window_left must be -1 or non-negative")

    use_lut = decode_chunk_pages_lut is not None
    has_kv_chunk_size_tensor = isinstance(kv_chunk_size, torch.Tensor)
    fixed_kv_chunk_size = 1
    kv_chunk_size_tensor = kv_chunk_size_ptr
    if use_lut:
        if decode_chunk_pages_lut.device != cache_seqlens.device:
            raise ValueError(
                "decode_chunk_pages_lut and cache_seqlens must be on the same device"
            )
        if int(decode_chunk_pages_lut.shape[0]) < 2:
            raise RuntimeError(
                "decode graph chunk-pages LUT must contain at least two entries"
            )
        lut_size = int(decode_chunk_pages_lut.shape[0])
    else:
        decode_chunk_pages_lut = kv_chunk_size_ptr
        lut_size = 0
        if has_kv_chunk_size_tensor:
            assert isinstance(kv_chunk_size, torch.Tensor)
            if kv_chunk_size.device != cache_seqlens.device:
                raise ValueError(
                    "kv_chunk_size tensor and cache_seqlens must be on the same device"
                )
            if kv_chunk_size.numel() != 1:
                raise ValueError(
                    "kv_chunk_size tensor must contain exactly one element"
                )
            kv_chunk_size_tensor = kv_chunk_size.reshape(1)
        else:
            if kv_chunk_size is None:
                raise ValueError(
                    "kv_chunk_size is required when decode_chunk_pages_lut is not provided"
                )
            fixed_kv_chunk_size = int(kv_chunk_size)
            if fixed_kv_chunk_size <= 0:
                raise ValueError("kv_chunk_size must be positive")

    update_regular_decode_graph_metadata_fused_triton[(1,)](
        cache_seqlens,
        merge_indptr,
        o_indptr,
        kv_chunk_size_ptr,
        kv_window_start_tokens,
        decode_chunk_pages_lut,
        kv_chunk_size_tensor,
        lut_size,
        PAGE_SIZE=page_size,
        WINDOW_PAGE_SPAN=int(window_page_span),
        WINDOW_LEFT=int(window_left),
        BATCH=bs,
        BLOCK_BATCH=triton.next_power_of_2(bs),
        USE_LUT=use_lut,
        HAS_KV_CHUNK_SIZE_TENSOR=has_kv_chunk_size_tensor,
        FIXED_KV_CHUNK_SIZE=fixed_kv_chunk_size,
    )


def update_decode_graph_window_start_tokens(
    *,
    cache_seqlens: torch.Tensor,
    kv_window_start_tokens: torch.Tensor,
    page_size: int,
    window_left: int,
) -> None:
    device = cache_seqlens.device
    if kv_window_start_tokens.device != device:
        raise ValueError(
            "kv_window_start_tokens and cache_seqlens must be on the same device"
        )
    if page_size <= 0:
        raise ValueError("page_size must be positive")
    if window_left < 0:
        raise ValueError("window_left must be non-negative")

    bs = int(cache_seqlens.shape[0])
    if bs <= 0:
        raise ValueError("decode graph replay requires bs > 0")
    if int(kv_window_start_tokens.shape[0]) < bs:
        raise RuntimeError(
            "decode graph kv_window_start_tokens is smaller than the graph batch"
        )

    update_decode_graph_window_start_tokens_triton[(1,)](
        cache_seqlens,
        kv_window_start_tokens,
        PAGE_SIZE=page_size,
        WINDOW_LEFT=int(window_left),
        BATCH=bs,
        BLOCK_BATCH=triton.next_power_of_2(bs),
    )


def update_regular_decode_graph_chunk_metadata_from_lut(
    *,
    cache_seqlens: torch.Tensor,
    merge_indptr: torch.Tensor,
    o_indptr: torch.Tensor,
    kv_chunk_size_ptr: torch.Tensor,
    kv_window_start_tokens: torch.Tensor,
    decode_chunk_pages_lut: torch.Tensor,
    page_size: int,
    window_page_span: int = 0,
    window_left: int = -1,
) -> None:
    device = cache_seqlens.device
    if merge_indptr.device != device or o_indptr.device != device:
        raise ValueError("indptr buffers and cache_seqlens must be on the same device")
    if kv_chunk_size_ptr.device != device:
        raise ValueError(
            "decode graph buffers and cache_seqlens must be on the same device"
        )
    if kv_window_start_tokens.device != device:
        raise ValueError(
            "kv_window_start_tokens and cache_seqlens must be on the same device"
        )

    _launch_regular_decode_graph_metadata_fused(
        cache_seqlens=cache_seqlens,
        merge_indptr=merge_indptr,
        o_indptr=o_indptr,
        kv_chunk_size_ptr=kv_chunk_size_ptr,
        kv_window_start_tokens=kv_window_start_tokens,
        decode_chunk_pages_lut=decode_chunk_pages_lut,
        page_size=page_size,
        window_page_span=window_page_span,
        window_left=window_left,
    )


def update_regular_decode_graph_chunk_metadata(
    *,
    cache_seqlens: torch.Tensor,
    merge_indptr: torch.Tensor,
    o_indptr: torch.Tensor,
    kv_chunk_size_ptr: torch.Tensor,
    kv_chunk_size: int | torch.Tensor,
    kv_window_start_tokens: torch.Tensor,
    max_chunks_per_req: int,
    page_size: int,
    window_page_span: int = 0,
    window_left: int = -1,
) -> None:
    device = cache_seqlens.device
    if merge_indptr.device != device or o_indptr.device != device:
        raise ValueError("indptr buffers and cache_seqlens must be on the same device")
    if kv_chunk_size_ptr.device != device:
        raise ValueError(
            "decode graph buffers and cache_seqlens must be on the same device"
        )
    if kv_window_start_tokens.device != device:
        raise ValueError(
            "kv_window_start_tokens and cache_seqlens must be on the same device"
        )
    if max_chunks_per_req <= 0:
        raise ValueError("max_chunks_per_req must be positive")
    if page_size <= 0:
        raise ValueError("page_size must be positive")

    bs = int(cache_seqlens.shape[0])
    if bs <= 0:
        raise ValueError("decode graph replay requires bs > 0")

    _launch_regular_decode_graph_metadata_fused(
        cache_seqlens=cache_seqlens,
        merge_indptr=merge_indptr,
        o_indptr=o_indptr,
        kv_chunk_size_ptr=kv_chunk_size_ptr,
        kv_window_start_tokens=kv_window_start_tokens,
        kv_chunk_size=kv_chunk_size,
        page_size=page_size,
        window_page_span=window_page_span,
        window_left=window_left,
    )


def update_decode_graph_chunk_metadata(
    *,
    cache_seqlens: torch.Tensor,
    request_indices: torch.Tensor,
    qo_tile_indices: torch.Tensor,
    kv_tile_indices: torch.Tensor,
    merge_indptr: torch.Tensor,
    o_indptr: torch.Tensor,
    block_valid_mask: torch.Tensor,
    kv_chunk_size_ptr: torch.Tensor,
    kv_window_start_tokens: torch.Tensor,
    decode_chunk_pages_lut: torch.Tensor,
    page_size: int,
    window_page_span: int = 0,
    window_left: int = -1,
) -> None:
    device = cache_seqlens.device
    if request_indices.device != device:
        raise ValueError("request_indices and cache_seqlens must be on the same device")
    if qo_tile_indices.device != device or kv_tile_indices.device != device:
        raise ValueError(
            "tile index buffers and cache_seqlens must be on the same device"
        )
    if merge_indptr.device != device or o_indptr.device != device:
        raise ValueError("indptr buffers and cache_seqlens must be on the same device")
    if block_valid_mask.device != device or kv_chunk_size_ptr.device != device:
        raise ValueError(
            "decode graph buffers and cache_seqlens must be on the same device"
        )
    if decode_chunk_pages_lut.device != device:
        raise ValueError(
            "decode_chunk_pages_lut and cache_seqlens must be on the same device"
        )
    if page_size <= 0:
        raise ValueError("page_size must be positive")

    update_decode_graph_chunk_metadata_fused(
        cache_seqlens=cache_seqlens,
        request_indices=request_indices,
        qo_tile_indices=qo_tile_indices,
        kv_tile_indices=kv_tile_indices,
        merge_indptr=merge_indptr,
        o_indptr=o_indptr,
        block_valid_mask=block_valid_mask,
        kv_chunk_size_ptr=kv_chunk_size_ptr,
        kv_window_start_tokens=kv_window_start_tokens,
        decode_chunk_pages_lut=decode_chunk_pages_lut,
        page_size=page_size,
        window_page_span=window_page_span,
        window_left=window_left,
    )


def update_prefill_graph_chunk_metadata(
    *,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    request_indices: torch.Tensor,
    qo_tile_indices: torch.Tensor,
    kv_tile_indices: torch.Tensor,
    merge_indptr: torch.Tensor,
    o_indptr: torch.Tensor,
    block_valid_mask: torch.Tensor,
    kv_chunk_size_ptr: torch.Tensor,
    kv_window_start_tokens: torch.Tensor,
    total_num_rows_ptr: torch.Tensor,
    batch: int,
    max_q_tiles_per_req: int,
    max_chunks_per_q_tile: int,
    max_q_rows_per_req: int,
    cta_tile_q: int,
    gqa_group_size: int,
    page_size: int,
    split_kv: bool,
    window_left: int = -1,
) -> None:
    device = cache_seqlens.device
    if cu_seqlens_q.device != device:
        raise ValueError("cu_seqlens_q and cache_seqlens must be on the same device")
    if request_indices.device != device:
        raise ValueError("request_indices and cache_seqlens must be on the same device")
    if qo_tile_indices.device != device or kv_tile_indices.device != device:
        raise ValueError(
            "tile index buffers and cache_seqlens must be on the same device"
        )
    if merge_indptr.device != device or o_indptr.device != device:
        raise ValueError("indptr buffers and cache_seqlens must be on the same device")
    if block_valid_mask.device != device or kv_chunk_size_ptr.device != device:
        raise ValueError(
            "prefill graph buffers and cache_seqlens must be on the same device"
        )
    if kv_window_start_tokens.device != device or total_num_rows_ptr.device != device:
        raise ValueError(
            "prefill graph scalar buffers and cache_seqlens must be on the same device"
        )
    if page_size <= 0:
        raise ValueError("page_size must be positive")
    if cta_tile_q <= 0:
        raise ValueError("cta_tile_q must be positive")
    if gqa_group_size <= 0:
        raise ValueError("gqa_group_size must be positive")
    if window_left < -1:
        raise ValueError("window_left must be -1 or non-negative")

    bs = int(batch)
    if bs <= 0:
        raise ValueError("prefill graph replay requires batch > 0")
    if int(cache_seqlens.shape[0]) < bs:
        raise ValueError("cache_seqlens is smaller than the graph batch")
    if int(cu_seqlens_q.shape[0]) < bs + 1:
        raise ValueError("cu_seqlens_q is smaller than the graph batch")
    work_items_capacity = int(request_indices.shape[0])
    block_valid_capacity = int(block_valid_mask.shape[0])
    if (
        int(qo_tile_indices.shape[0]) != work_items_capacity
        or int(kv_tile_indices.shape[0]) != work_items_capacity
    ):
        raise RuntimeError(
            "prefill graph tile index buffers must match request_indices shape"
        )
    if max_q_tiles_per_req <= 0:
        raise ValueError("max_q_tiles_per_req must be positive")
    if max_chunks_per_q_tile <= 0:
        raise ValueError("max_chunks_per_q_tile must be positive")
    if max_q_rows_per_req <= 0:
        raise ValueError("max_q_rows_per_req must be positive")
    required_work_items = bs * int(max_q_tiles_per_req) * int(max_chunks_per_q_tile)
    if required_work_items > work_items_capacity:
        raise RuntimeError(
            "prefill graph workspace request_indices capacity is too small for the graph plan"
        )
    if required_work_items > block_valid_capacity:
        raise RuntimeError(
            "prefill graph workspace block_valid capacity is too small for the graph plan"
        )
    if int(o_indptr.shape[0]) < bs + 1:
        raise RuntimeError("prefill graph workspace o_indptr capacity is too small")
    if int(kv_window_start_tokens.shape[0]) < bs:
        raise RuntimeError(
            "prefill graph workspace kv_window_start_tokens capacity is too small"
        )

    work_blocks = triton.cdiv(block_valid_capacity, _PREFILL_BLOCK_WORK_ITEMS)
    update_prefill_graph_work_metadata_triton[(work_blocks,)](
        cache_seqlens,
        cu_seqlens_q,
        request_indices,
        qo_tile_indices,
        kv_tile_indices,
        o_indptr,
        block_valid_mask,
        kv_chunk_size_ptr,
        kv_window_start_tokens,
        work_items_capacity=work_items_capacity,
        block_valid_capacity=block_valid_capacity,
        BATCH=bs,
        MAX_Q_TILES_PER_REQ=int(max_q_tiles_per_req),
        MAX_CHUNKS_PER_Q_TILE=int(max_chunks_per_q_tile),
        CTA_TILE_Q=int(cta_tile_q),
        GQA_GROUP_SIZE=int(gqa_group_size),
        PAGE_SIZE=int(page_size),
        WINDOW_LEFT=int(window_left),
        SPLIT_KV=bool(split_kv),
        BLOCK_WORK_ITEMS=_PREFILL_BLOCK_WORK_ITEMS,
    )
    torch.cumsum(
        o_indptr[1 : bs + 1],
        dim=0,
        out=o_indptr[1 : bs + 1],
    )
    total_num_rows_ptr[:1].copy_(cu_seqlens_q[bs : bs + 1])

    row_blocks = triton.cdiv(int(max_q_rows_per_req), _PREFILL_BLOCK_ROWS)
    update_prefill_graph_row_indptr_triton[(bs, row_blocks)](
        cu_seqlens_q,
        o_indptr,
        merge_indptr,
        BATCH=bs,
        MAX_Q_ROWS_PER_REQ=int(max_q_rows_per_req),
        BLOCK_ROWS=_PREFILL_BLOCK_ROWS,
    )
