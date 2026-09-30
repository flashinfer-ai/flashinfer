"""Auxiliary Triton stages for the QSA selection and state transaction."""

from __future__ import annotations

import contextlib
import contextvars
import functools
import inspect
import math

from collections.abc import Mapping

import torch
import triton
import triton.language as tl

from b12x._lib.compile_plan import launch_triton as _jit_launch


@triton.jit
def _round_fp32_to_bf16(value):
    """Round FP32 to BF16 and widen without permitting arithmetic fusion."""
    bits = value.to(tl.uint32, bitcast=True)
    rounding_bias = 0x7FFF + ((bits >> 16) & 1)
    rounded_bits = (bits + rounding_bias) & 0xFFFF0000
    return rounded_bits.to(tl.float32, bitcast=True)


@triton.jit(do_not_specialize=["index_query_row_stride"])
def _prepare_index_query_kernel(
    index_query,
    request_ids,
    norm_weight,
    rope_positions,
    rope_cos,
    rope_sin,
    prepared_query,
    rope_position_rows,
    eps,
    rope_position_row_stride,
    rope_position_axis_stride,
    rope_cos_row_stride,
    rope_sin_row_stride,
    index_query_row_stride,
    INDEX_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROTARY_DIM: tl.constexpr,
    POSITION_AXES: tl.constexpr,
    MROPE_INTERLEAVED: tl.constexpr,
    MROPE_SECTION_0: tl.constexpr,
    MROPE_SECTION_1: tl.constexpr,
    ROPE_IS_BF16: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    program = tl.program_id(0)
    row = program // INDEX_HEADS
    head = program % INDEX_HEADS
    dims = tl.arange(0, BLOCK_D)
    dim_mask = dims < HEAD_DIM
    valid_row = tl.load(request_ids + row).to(tl.int64) >= 0
    offsets = (row.to(tl.int64) * INDEX_HEADS + head) * HEAD_DIM + dims
    query_base = row.to(tl.int64) * index_query_row_stride + head * HEAD_DIM
    values = tl.load(
        index_query + query_base + dims,
        mask=valid_row & dim_mask,
        other=0.0,
    ).to(tl.float32)
    variance = tl.sum(values * values, axis=0) / HEAD_DIM
    inv_rms = tl.rsqrt(variance + eps)
    weight = tl.load(norm_weight + dims, mask=dim_mask, other=0.0).to(tl.float32)
    normalized = (values * inv_rms * (1.0 + weight)).to(tl.bfloat16).to(tl.float32)

    half_rotary = ROTARY_DIM // 2
    in_rotary = dims < ROTARY_DIM
    pair = dims % half_rotary
    partner_dim = tl.where(dims < half_rotary, dims + half_rotary, dims - half_rotary)
    partner = tl.load(
        index_query + query_base + partner_dim,
        mask=valid_row & in_rotary,
        other=0.0,
    ).to(tl.float32)
    partner_weight = tl.load(
        norm_weight + partner_dim,
        mask=in_rotary,
        other=0.0,
    ).to(tl.float32)
    partner = (
        (partner * inv_rms * (1.0 + partner_weight)).to(tl.bfloat16).to(tl.float32)
    )

    if POSITION_AXES == 1:
        axis = tl.zeros((BLOCK_D,), tl.int32)
    elif MROPE_INTERLEAVED:
        is_height = (pair % 3 == 1) & (pair < 3 * MROPE_SECTION_1)
        is_width = (pair % 3 == 2) & (
            pair < 3 * (ROTARY_DIM // 2 - MROPE_SECTION_0 - MROPE_SECTION_1)
        )
        axis = tl.where(is_height, 1, tl.where(is_width, 2, 0))
    else:
        axis = tl.where(
            pair < MROPE_SECTION_0,
            0,
            tl.where(pair < MROPE_SECTION_0 + MROPE_SECTION_1, 1, 2),
        )
    position = tl.load(
        rope_positions
        + row * rope_position_row_stride
        + axis * rope_position_axis_stride,
        mask=in_rotary,
        other=0,
    ).to(tl.int64)
    valid_position = (position >= 0) & (position < rope_position_rows)
    cosine = tl.load(
        rope_cos + position * rope_cos_row_stride + pair,
        mask=valid_row & in_rotary & valid_position,
        other=1.0,
    ).to(tl.float32)
    sine = tl.load(
        rope_sin + position * rope_sin_row_stride + pair,
        mask=valid_row & in_rotary & valid_position,
        other=0.0,
    ).to(tl.float32)
    rotated_partner = tl.where(dims < half_rotary, -partner, partner)
    if ROPE_IS_BF16:
        direct_product = _round_fp32_to_bf16(normalized * cosine)
        partner_product = _round_fp32_to_bf16(rotated_partner * sine)
        rotated = _round_fp32_to_bf16(direct_product + partner_product)
    else:
        rotated = normalized * cosine + rotated_partner * sine
    result = tl.where(in_rotary, rotated, normalized)
    tl.store(prepared_query + offsets, result, mask=dim_mask)


@triton.jit(do_not_specialize=["raw_key_row_stride"])
def _compress_completed_groups_kernel(
    raw_index_key,
    query_positions,
    rope_positions,
    request_ids,
    query_start_loc,
    raw_state_slot_ids,
    raw_k_ring,
    raw_logical_positions,
    raw_rope_positions,
    key_norm_weight,
    rope_cos,
    rope_sin,
    compressed_cache,
    compressed_block_table,
    rope_position_rows,
    rope_position_row_stride,
    rope_position_axis_stride,
    rope_cos_row_stride,
    rope_sin_row_stride,
    raw_state_slot_stride,
    raw_k_slot_stride,
    raw_k_ring_stride,
    raw_position_slot_stride,
    raw_rope_slot_stride,
    raw_rope_ring_stride,
    compressed_page_stride,
    compressed_token_stride,
    compressed_table_stride,
    eps,
    raw_key_row_stride,
    INDEX_HEAD_DIM: tl.constexpr,
    ROTARY_DIM: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    RING_CAPACITY: tl.constexpr,
    COMPRESSED_PAGE_SIZE: tl.constexpr,
    DCP_SIZE: tl.constexpr,
    DCP_RANK: tl.constexpr,
    CP_INTERLEAVE: tl.constexpr,
    POSITION_AXES: tl.constexpr,
    MROPE_INTERLEAVED: tl.constexpr,
    MROPE_SECTION_0: tl.constexpr,
    MROPE_SECTION_1: tl.constexpr,
    ROPE_IS_BF16: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    request = tl.load(request_ids + row).to(tl.int64)
    position = tl.load(query_positions + row).to(tl.int64)
    real_request = request >= 0
    state_slot = tl.load(
        raw_state_slot_ids + request * raw_state_slot_stride,
        mask=real_request,
        other=-1,
    ).to(tl.int64)
    group_id = position // COMPRESS_RATIO
    if DCP_SIZE == 1:
        complete = ((position + 1) % COMPRESS_RATIO) == 0
    else:
        group_interleave = CP_INTERLEAVE // COMPRESS_RATIO
        dcp_round = DCP_SIZE * group_interleave
        owner = (group_id // group_interleave) % DCP_SIZE
        complete = (((position + 1) % COMPRESS_RATIO) == 0) & (
            owner == DCP_RANK
        )
    if real_request & complete & (state_slot >= 0):
        request_start = tl.load(query_start_loc + request).to(tl.int64)
        current_first = tl.load(query_positions + request_start).to(tl.int64)
        group_first = position - COMPRESS_RATIO + 1
        dims = tl.arange(0, BLOCK_D)
        dim_mask = dims < INDEX_HEAD_DIM
        half_rotary = ROTARY_DIM // 2
        partner_dim = tl.where(
            dims < half_rotary, dims + half_rotary, dims - half_rotary
        )
        total = tl.zeros((BLOCK_D,), tl.float32)
        partner_total = tl.zeros((BLOCK_D,), tl.float32)
        for offset in tl.static_range(0, COMPRESS_RATIO):
            source_position = group_first + offset
            from_current = source_position >= current_first
            current_row = request_start + source_position - current_first
            current = tl.load(
                raw_index_key + current_row * raw_key_row_stride + dims,
                mask=from_current & dim_mask,
                other=0.0,
            ).to(tl.float32)
            current_partner = tl.load(
                raw_index_key + current_row * raw_key_row_stride + partner_dim,
                mask=from_current & (dims < ROTARY_DIM),
                other=0.0,
            ).to(tl.float32)
            ring_slot = source_position % RING_CAPACITY
            key_base = (
                state_slot * raw_k_slot_stride + ring_slot * raw_k_ring_stride
            ).to(tl.int64)
            prior = tl.load(
                raw_k_ring + key_base + dims,
                mask=(~from_current) & dim_mask,
                other=0.0,
            ).to(tl.float32)
            prior_partner = tl.load(
                raw_k_ring + key_base + partner_dim,
                mask=(~from_current) & (dims < ROTARY_DIM),
                other=0.0,
            ).to(tl.float32)
            total += tl.where(from_current, current, prior)
            partner_total += tl.where(from_current, current_partner, prior_partner)

        pooled = (total / COMPRESS_RATIO).to(tl.bfloat16)
        pooled_fp32 = pooled.to(tl.float32)
        variance = tl.sum(pooled_fp32 * pooled_fp32, axis=0) / INDEX_HEAD_DIM
        inv_rms = tl.rsqrt(variance + eps)
        weight = tl.load(key_norm_weight + dims, mask=dim_mask, other=0.0).to(
            tl.float32
        )
        normalized = (
            (pooled_fp32 * inv_rms * (1.0 + weight)).to(tl.bfloat16).to(tl.float32)
        )

        in_rotary = dims < ROTARY_DIM
        pair = dims % half_rotary
        partner_pooled = (partner_total / COMPRESS_RATIO).to(tl.bfloat16).to(tl.float32)
        partner_weight = tl.load(
            key_norm_weight + partner_dim,
            mask=in_rotary,
            other=0.0,
        ).to(tl.float32)
        partner = (
            (partner_pooled * inv_rms * (1.0 + partner_weight))
            .to(tl.bfloat16)
            .to(tl.float32)
        )

        if POSITION_AXES == 1:
            axis = tl.zeros((BLOCK_D,), tl.int32)
        elif MROPE_INTERLEAVED:
            is_height = (pair % 3 == 1) & (pair < 3 * MROPE_SECTION_1)
            is_width = (pair % 3 == 2) & (
                pair < 3 * (ROTARY_DIM // 2 - MROPE_SECTION_0 - MROPE_SECTION_1)
            )
            axis = tl.where(is_height, 1, tl.where(is_width, 2, 0))
        else:
            axis = tl.where(
                pair < MROPE_SECTION_0,
                0,
                tl.where(pair < MROPE_SECTION_0 + MROPE_SECTION_1, 1, 2),
            )
        first_from_current = group_first >= current_first
        first_current_row = request_start + group_first - current_first
        first_ring_slot = group_first % RING_CAPACITY
        rope_base = (
            state_slot * raw_rope_slot_stride + first_ring_slot * raw_rope_ring_stride
        ).to(tl.int64)
        current_rope = tl.load(
            rope_positions
            + first_current_row * rope_position_row_stride
            + axis * rope_position_axis_stride,
            mask=first_from_current & in_rotary,
            other=0,
        ).to(tl.int64)
        ring_rope = tl.load(
            raw_rope_positions + rope_base + axis,
            mask=(~first_from_current) & in_rotary,
            other=0,
        ).to(tl.int64)
        first_rope_position = tl.where(first_from_current, current_rope, ring_rope)
        valid_position = (first_rope_position >= 0) & (
            first_rope_position < rope_position_rows
        )
        cosine = tl.load(
            rope_cos + first_rope_position * rope_cos_row_stride + pair,
            mask=in_rotary & valid_position,
            other=1.0,
        ).to(tl.float32)
        sine = tl.load(
            rope_sin + first_rope_position * rope_sin_row_stride + pair,
            mask=in_rotary & valid_position,
            other=0.0,
        ).to(tl.float32)
        rotated_partner = tl.where(dims < half_rotary, -partner, partner)
        if ROPE_IS_BF16:
            direct_product = _round_fp32_to_bf16(normalized * cosine)
            partner_product = _round_fp32_to_bf16(rotated_partner * sine)
            rotated = _round_fp32_to_bf16(direct_product + partner_product)
        else:
            rotated = normalized * cosine + rotated_partner * sine
        representative = tl.where(in_rotary, rotated, normalized)
        if DCP_SIZE == 1:
            local_group = group_id
        else:
            local_group = (
                (group_id // dcp_round) * group_interleave
                + group_id % group_interleave
            )
        logical_page = local_group // COMPRESSED_PAGE_SIZE
        page_offset = local_group % COMPRESSED_PAGE_SIZE
        table_offset = (request * compressed_table_stride + logical_page).to(tl.int64)
        physical_page = tl.load(compressed_block_table + table_offset).to(tl.int64)
        if physical_page >= 0:
            cache_base = (
                physical_page * compressed_page_stride
                + page_offset * compressed_token_stride
            ).to(tl.int64)
            tl.store(
                compressed_cache + cache_base + dims,
                representative,
                mask=dim_mask,
            )


@triton.jit(do_not_specialize=["raw_key_row_stride"])
def _commit_raw_ring_kernel(
    raw_index_key,
    query_positions,
    rope_positions,
    request_ids,
    query_start_loc,
    sequence_lengths,
    is_prefilling,
    raw_state_slot_ids,
    raw_k_ring,
    raw_logical_positions,
    raw_rope_positions,
    raw_interval_start_positions,
    rope_position_row_stride,
    rope_position_axis_stride,
    raw_state_slot_stride,
    raw_k_slot_stride,
    raw_k_ring_stride,
    raw_position_slot_stride,
    raw_rope_slot_stride,
    raw_rope_ring_stride,
    raw_interval_start_stride,
    raw_key_row_stride,
    INDEX_HEAD_DIM: tl.constexpr,
    POSITION_AXES: tl.constexpr,
    RING_CAPACITY: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    request = tl.program_id(0)
    suffix_offset = tl.program_id(1)
    request_start = tl.load(query_start_loc + request).to(tl.int64)
    request_end = tl.load(query_start_loc + request + 1).to(tl.int64)
    query_length = request_end - request_start
    suffix_length = tl.minimum(query_length, RING_CAPACITY)
    row = request_end - suffix_length + suffix_offset
    active = suffix_offset < suffix_length
    observed_request = tl.load(request_ids + row, mask=active, other=-1).to(tl.int64)
    real_request = active & (observed_request == request)
    state_slot = tl.load(
        raw_state_slot_ids + request * raw_state_slot_stride,
        mask=real_request,
        other=-1,
    ).to(tl.int64)
    if real_request & (state_slot >= 0):
        position = tl.load(query_positions + row).to(tl.int64)
        ring_slot = position % RING_CAPACITY
        dims = tl.arange(0, BLOCK_D)
        dim_mask = dims < INDEX_HEAD_DIM
        key_base = (state_slot * raw_k_slot_stride + ring_slot * raw_k_ring_stride).to(
            tl.int64
        )
        key = tl.load(
            raw_index_key + row.to(tl.int64) * raw_key_row_stride + dims,
            mask=dim_mask,
            other=0.0,
        )
        tl.store(raw_k_ring + key_base + dims, key, mask=dim_mask)
        tag_offset = (state_slot * raw_position_slot_stride + ring_slot).to(tl.int64)
        tl.store(raw_logical_positions + tag_offset, position)
        axes = tl.arange(0, 4)
        axis_mask = axes < POSITION_AXES
        rope = tl.load(
            rope_positions
            + row * rope_position_row_stride
            + axes * rope_position_axis_stride,
            mask=axis_mask,
            other=-1,
        )
        rope_base = (
            state_slot * raw_rope_slot_stride + ring_slot * raw_rope_ring_stride
        ).to(tl.int64)
        tl.store(raw_rope_positions + rope_base + axes, rope, mask=axis_mask)
        if suffix_offset == 0:
            interval_start_offset = (state_slot * raw_interval_start_stride).to(
                tl.int64
            )
            prefill = tl.load(is_prefilling + request).to(tl.int1)
            sequence_length = tl.load(sequence_lengths + request).to(tl.int64)
            anchor = tl.where(prefill, sequence_length - 1, position)
            tl.store(
                raw_interval_start_positions + interval_start_offset,
                anchor,
            )


@triton.jit
def _score_representatives_kernel(
    prepared_query,
    query_positions,
    request_ids,
    sequence_lengths,
    compressed_cache,
    compressed_block_table,
    scores,
    eligible_counts,
    merge_lengths,
    compressed_page_stride,
    compressed_token_stride,
    compressed_table_stride,
    score_row_stride,
    MAX_GROUPS: tl.constexpr,
    GROUP_OFFSET: tl.constexpr,
    GROUP_COUNT: tl.constexpr,
    GROUP_BUDGET: tl.constexpr,
    INDEX_HEADS: tl.constexpr,
    INDEX_HEAD_DIM: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    COMPRESSED_PAGE_SIZE: tl.constexpr,
    DCP_SIZE: tl.constexpr,
    DCP_RANK: tl.constexpr,
    CP_INTERLEAVE: tl.constexpr,
    BLOCK_G: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    group_block = tl.program_id(1)
    request = tl.load(request_ids + row).to(tl.int64)
    position = tl.load(query_positions + row).to(tl.int64)
    real_request = request >= 0
    sequence_length = tl.load(
        sequence_lengths + request,
        mask=real_request,
        other=0,
    ).to(tl.int64)
    global_eligible = tl.minimum(
        (position + 1) // COMPRESS_RATIO,
        sequence_length // COMPRESS_RATIO,
    )
    if DCP_SIZE == 1:
        eligible = global_eligible
    else:
        group_interleave = CP_INTERLEAVE // COMPRESS_RATIO
        dcp_round = DCP_SIZE * group_interleave
        complete_rounds = global_eligible // dcp_round
        remainder = global_eligible - complete_rounds * dcp_round
        rank_remainder = tl.minimum(
            tl.maximum(remainder - DCP_RANK * group_interleave, 0),
            group_interleave,
        )
        eligible = complete_rounds * group_interleave + rank_remainder
    eligible = tl.minimum(eligible, MAX_GROUPS)
    eligible = tl.where(real_request, eligible, 0)
    prior_eligible = tl.minimum(eligible, GROUP_OFFSET)
    carry_count = tl.minimum(prior_eligible, GROUP_BUDGET)
    chunk_eligible = tl.minimum(tl.maximum(eligible - GROUP_OFFSET, 0), GROUP_COUNT)
    if group_block == 0:
        tl.store(eligible_counts + row, eligible)
        tl.store(merge_lengths + row, carry_count + chunk_eligible)
    local_groups = group_block * BLOCK_G + tl.arange(0, BLOCK_G)
    groups = GROUP_OFFSET + local_groups
    group_mask = local_groups < GROUP_COUNT
    active = group_mask & (groups < eligible) & real_request
    logical_pages = groups // COMPRESSED_PAGE_SIZE
    page_offsets = groups % COMPRESSED_PAGE_SIZE
    table_offsets = request * compressed_table_stride + logical_pages.to(tl.int64)
    physical_pages = tl.load(
        compressed_block_table + table_offsets,
        mask=active,
        other=-1,
    ).to(tl.int64)
    valid_pages = physical_pages >= 0
    dims = tl.arange(0, BLOCK_D)
    dim_mask = dims < INDEX_HEAD_DIM
    cache_offsets = (
        physical_pages[:, None] * compressed_page_stride
        + page_offsets[:, None].to(tl.int64) * compressed_token_stride
        + dims[None, :]
    )
    keys = tl.load(
        compressed_cache + cache_offsets,
        mask=active[:, None] & valid_pages[:, None] & dim_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    score = tl.zeros((BLOCK_G,), tl.float32)
    for head in tl.static_range(0, INDEX_HEADS):
        query_offsets = (row * INDEX_HEADS + head) * INDEX_HEAD_DIM + dims
        query = tl.load(
            prepared_query + query_offsets,
            mask=dim_mask,
            other=0.0,
        ).to(tl.float32)
        dot = tl.sum(keys * query[None, :], axis=1)
        score += tl.maximum(dot, 0.0)
    score *= 1.0 / math.sqrt(INDEX_HEAD_DIM)
    score = tl.where(active & valid_pages, score, -float("inf"))
    output_columns = carry_count + local_groups
    tl.store(
        scores + row * score_row_stride + output_columns,
        score,
        mask=group_mask,
    )


@triton.jit
def _stage_topk_carry_kernel(
    prior_values,
    eligible_counts,
    scores,
    score_row_stride,
    GROUP_OFFSET: tl.constexpr,
    GROUP_BUDGET: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK_K)
    eligible = tl.load(eligible_counts + row)
    carry_count = tl.minimum(tl.minimum(eligible, GROUP_OFFSET), GROUP_BUDGET)
    values = tl.load(
        prior_values + row * GROUP_BUDGET + columns,
        mask=columns < carry_count,
        other=-float("inf"),
    )
    tl.store(
        scores + row * score_row_stride + columns,
        values,
        mask=columns < carry_count,
    )


@triton.jit
def _remap_topk_group_ids_kernel(
    local_ids,
    prior_ids,
    eligible_counts,
    merge_lengths,
    GROUP_OFFSET: tl.constexpr,
    GROUP_BUDGET: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK_K)
    eligible = tl.load(eligible_counts + row)
    carry_count = tl.minimum(tl.minimum(eligible, GROUP_OFFSET), GROUP_BUDGET)
    output_count = tl.minimum(tl.load(merge_lengths + row), GROUP_BUDGET)
    local = tl.load(
        local_ids + row * GROUP_BUDGET + columns,
        mask=columns < GROUP_BUDGET,
        other=-1,
    )
    carried = tl.load(
        prior_ids + row * GROUP_BUDGET + local,
        mask=(columns < output_count) & (local >= 0) & (local < carry_count),
        other=-1,
    )
    global_id = tl.where(
        local < carry_count,
        carried,
        GROUP_OFFSET + local - carry_count,
    )
    global_id = tl.where((columns < output_count) & (local >= 0), global_id, -1)
    tl.store(
        local_ids + row * GROUP_BUDGET + columns,
        global_id,
        mask=columns < GROUP_BUDGET,
    )


@triton.jit
def _copy_stable_topk_kernel(
    stable_values,
    stable_ids,
    topk_values,
    topk_ids,
    GROUP_BUDGET: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK_K)
    mask = columns < GROUP_BUDGET
    values = tl.load(
        stable_values + row * GROUP_BUDGET + columns,
        mask=mask,
        other=-float("inf"),
    )
    ids = tl.load(
        stable_ids + row * GROUP_BUDGET + columns,
        mask=mask,
        other=-1,
    )
    score_bits = values.to(tl.uint32, bitcast=True).to(tl.uint64)
    id_key = (0xFFFFFFFF - ids.to(tl.uint32)).to(tl.uint64)
    keys = tl.where(
        (columns < GROUP_BUDGET) & (ids >= 0), (score_bits << 32) | id_key, 0
    )
    keys = tl.sort(keys, dim=0, descending=True)
    sorted_ids = (0xFFFFFFFF - (keys & 0xFFFFFFFF).to(tl.uint32)).to(tl.int32)
    sorted_values = (keys >> 32).to(tl.uint32).to(tl.float32, bitcast=True)
    valid = mask & (keys != 0)
    tl.store(
        topk_values + row * GROUP_BUDGET + columns,
        tl.where(valid, sorted_values, -float("inf")),
        mask=mask,
    )
    tl.store(
        topk_ids + row * GROUP_BUDGET + columns,
        tl.where(valid, sorted_ids, -1),
        mask=mask,
    )


@triton.jit
def _expand_selected_groups_kernel(
    topk_group_ids,
    eligible_counts,
    query_positions,
    selected_positions,
    topk_row_stride,
    selected_row_stride,
    GROUP_BUDGET: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    SELECTION_WIDTH: tl.constexpr,
    DCP_SIZE: tl.constexpr,
    DCP_RANK: tl.constexpr,
    CP_INTERLEAVE: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK_W)
    column_mask = columns < SELECTION_WIDTH
    eligible = tl.load(eligible_counts + row)
    selected_groups = tl.minimum(eligible, GROUP_BUDGET)
    expanded_count = selected_groups * COMPRESS_RATIO
    group_columns = columns // COMPRESS_RATIO
    group_ids = tl.load(
        topk_group_ids + row * topk_row_stride + group_columns,
        mask=column_mask & (columns < expanded_count),
        other=-1,
    )
    expanded = group_ids * COMPRESS_RATIO + columns % COMPRESS_RATIO
    position = tl.load(query_positions + row).to(tl.int64)
    tail_start = ((position + 1) // COMPRESS_RATIO) * COMPRESS_RATIO
    if DCP_SIZE == 1:
        local_tail_start = tail_start
        tail_length = position + 1 - tail_start
    else:
        global_tail_group = tail_start // COMPRESS_RATIO
        group_interleave = CP_INTERLEAVE // COMPRESS_RATIO
        dcp_round = DCP_SIZE * group_interleave
        tail_owner = (global_tail_group // group_interleave) % DCP_SIZE
        local_tail_group = (
            (global_tail_group // dcp_round) * group_interleave
            + global_tail_group % group_interleave
        )
        local_tail_start = local_tail_group * COMPRESS_RATIO
        tail_length = tl.where(
            tail_owner == DCP_RANK,
            position + 1 - tail_start,
            0,
        )
    tail_column = columns - expanded_count
    in_tail = (tail_column >= 0) & (tail_column < tail_length)
    result = tl.where(
        columns < expanded_count,
        expanded,
        tl.where(in_tail, local_tail_start + tail_column, -1),
    )
    tl.store(
        selected_positions + row * selected_row_stride + columns,
        result,
        mask=column_mask,
    )


@triton.jit
def _expand_global_selected_groups_kernel(
    topk_group_ids,
    query_positions,
    selected_positions,
    topk_row_stride,
    selected_row_stride,
    GROUP_BUDGET: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    SELECTION_WIDTH: tl.constexpr,
    DCP_SIZE: tl.constexpr,
    DCP_RANK: tl.constexpr,
    CP_INTERLEAVE: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK_W)
    column_mask = columns < SELECTION_WIDTH
    group_columns = columns // COMPRESS_RATIO
    global_group = tl.load(
        topk_group_ids + row * topk_row_stride + group_columns,
        mask=column_mask & (group_columns < GROUP_BUDGET),
        other=-1,
    ).to(tl.int64)
    group_interleave = CP_INTERLEAVE // COMPRESS_RATIO
    dcp_round = DCP_SIZE * group_interleave
    safe_group = tl.maximum(global_group, 0)
    owner = (safe_group // group_interleave) % DCP_SIZE
    local_group = (
        (safe_group // dcp_round) * group_interleave
        + safe_group % group_interleave
    )
    expanded = local_group * COMPRESS_RATIO + columns % COMPRESS_RATIO
    selected = tl.where(
        (columns < GROUP_BUDGET * COMPRESS_RATIO)
        & (global_group >= 0)
        & (owner == DCP_RANK),
        expanded,
        -1,
    )

    position = tl.load(query_positions + row).to(tl.int64)
    global_tail_start = ((position + 1) // COMPRESS_RATIO) * COMPRESS_RATIO
    global_tail_group = global_tail_start // COMPRESS_RATIO
    tail_owner = (global_tail_group // group_interleave) % DCP_SIZE
    local_tail_group = (
        (global_tail_group // dcp_round) * group_interleave
        + global_tail_group % group_interleave
    )
    tail_column = columns - GROUP_BUDGET * COMPRESS_RATIO
    tail_length = position + 1 - global_tail_start
    in_tail = (
        (tail_owner == DCP_RANK)
        & (tail_column >= 0)
        & (tail_column < tail_length)
    )
    selected = tl.where(
        in_tail,
        local_tail_group * COMPRESS_RATIO + tail_column,
        selected,
    )
    tl.store(
        selected_positions + row * selected_row_stride + columns,
        selected,
        mask=column_mask,
    )


_SUPPORT_KERNEL_KEYS = {
    _prepare_index_query_kernel: "prepare_index_query",
    _compress_completed_groups_kernel: "compress_completed_groups",
    _commit_raw_ring_kernel: "commit_raw_ring",
    _score_representatives_kernel: "score_representatives",
    _stage_topk_carry_kernel: "stage_topk_carry",
    _remap_topk_group_ids_kernel: "remap_topk_group_ids",
    _copy_stable_topk_kernel: "stable_topk_copy",
    _expand_selected_groups_kernel: "expand_selected_groups",
    _expand_global_selected_groups_kernel: "expand_global_selected_groups",
}

_support_launch_context: contextvars.ContextVar[
    tuple[Mapping[str, object], bool] | None
] = contextvars.ContextVar("_support_launch_context", default=None)


def _support_kernel_key(kernel: object, constexprs: Mapping[str, object]) -> str:
    key = _SUPPORT_KERNEL_KEYS[kernel]
    # Planned score chunks have distinct static offsets and final chunk sizes.
    if kernel is _score_representatives_kernel:
        key = f"{key}/groups-{int(constexprs['GROUP_COUNT'])}"
    if "GROUP_OFFSET" in constexprs:
        key = f"{key}/offset-{int(constexprs['GROUP_OFFSET'])}"
    return key


def _launch_triton(kernel: object, grid: tuple[int, ...], *args: object, **constexprs: object) -> object:
    """Launch a declared QSA support program without resolving JIT at runtime."""
    context = _support_launch_context.get()
    if context is None:
        return _jit_launch(kernel, grid, *args, **constexprs)
    programs, compiling = context
    key = _support_kernel_key(kernel, constexprs)
    if compiling:
        if not isinstance(programs, dict):
            raise TypeError("QSA support compilation requires a mutable program map")
        program = _jit_launch(kernel, grid, *args, **constexprs)
        programs[key] = program
        return program
    try:
        program = programs[key]
    except KeyError:
        raise RuntimeError(f"prepared QSA support programs are missing {key!r}") from None
    # Native Triton launchers retain their static ABI; only runtime pointers and
    # scalars cross this boundary, on Triton's required three-dimensional grid.
    return program[tuple((*grid, 1, 1)[:3])](*args)


@contextlib.contextmanager
def _support_context(programs: Mapping[str, object], *, compiling: bool):
    if compiling:
        from b12x._lib.compile_plan import compile_only_launches_enabled
        if not compile_only_launches_enabled():
            raise RuntimeError("QSA support extraction requires compile-only preparation")
    token = _support_launch_context.set((programs, compiling))
    try:
        yield
    finally:
        _support_launch_context.reset(token)


def _prepared_support_wrapper(fn):
    """Accept a prepared support map without changing each legacy ABI."""
    signature = inspect.signature(fn)

    @functools.wraps(fn)
    def wrapped(*args, _prepared: Mapping[str, object] | None = None, **kwargs):
        context = _support_launch_context.get()
        if _prepared is None or (context is not None and context[1]):
            return fn(*args, **kwargs)
        with _support_context(_prepared, compiling=False):
            return fn(*args, **kwargs)

    wrapped.__signature__ = signature.replace(
        parameters=[
            *signature.parameters.values(),
            inspect.Parameter(
                "_prepared",
                inspect.Parameter.KEYWORD_ONLY,
                default=None,
                annotation=Mapping[str, object] | None,
            ),
        ]
    )
    return wrapped


def _mrope_sections(caps) -> tuple[int, int]:
    if int(caps.position_axes) == 1:
        return 0, 0
    assert caps.mrope_sections is not None
    return int(caps.mrope_sections[0]), int(caps.mrope_sections[1])


def launch_prepare_index_query(
    *,
    index_query: torch.Tensor,
    request_ids: torch.Tensor,
    norm_weight: torch.Tensor,
    rope_positions: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    prepared_query: torch.Tensor,
    caps,
) -> None:
    rows = int(index_query.shape[0])
    section0, section1 = _mrope_sections(caps)
    _launch_triton(_prepare_index_query_kernel, (rows * int(caps.index_heads),), index_query,
    request_ids,
    norm_weight,
    rope_positions,
    rope_cos,
    rope_sin,
    prepared_query,
    int(rope_cos.shape[0]),
    float(caps.rms_norm_eps),
    int(rope_positions.stride(0)),
    int(rope_positions.stride(1)),
    int(rope_cos.stride(0)),
    int(rope_sin.stride(0)),
    int(index_query.stride(0)),
    INDEX_HEADS=int(caps.index_heads),
    HEAD_DIM=int(caps.index_head_dim),
    ROTARY_DIM=int(caps.index_rotary_dim),
    POSITION_AXES=int(caps.position_axes),
    MROPE_INTERLEAVED=bool(caps.mrope_interleaved),
    MROPE_SECTION_0=section0,
    MROPE_SECTION_1=section1,
    ROPE_IS_BF16=rope_cos.dtype == torch.bfloat16,
    BLOCK_D=triton.next_power_of_2(int(caps.index_head_dim)),
    num_warps=4,)


def launch_compress_completed_groups(
    *,
    raw_index_key: torch.Tensor,
    query_positions: torch.Tensor,
    rope_positions: torch.Tensor,
    request_ids: torch.Tensor,
    query_start_loc: torch.Tensor,
    raw_state_slot_ids: torch.Tensor,
    raw_k_ring: torch.Tensor,
    raw_logical_positions: torch.Tensor,
    raw_rope_positions: torch.Tensor,
    key_norm_weight: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    compressed_cache: torch.Tensor,
    compressed_block_table: torch.Tensor,
    caps,
) -> None:
    rows = int(raw_index_key.shape[0])
    section0, section1 = _mrope_sections(caps)
    _launch_triton(_compress_completed_groups_kernel, (rows,), raw_index_key,
    query_positions,
    rope_positions,
    request_ids,
    query_start_loc,
    raw_state_slot_ids,
    raw_k_ring,
    raw_logical_positions,
    raw_rope_positions,
    key_norm_weight,
    rope_cos,
    rope_sin,
    compressed_cache,
    compressed_block_table,
    int(rope_cos.shape[0]),
    int(rope_positions.stride(0)),
    int(rope_positions.stride(1)),
    int(rope_cos.stride(0)),
    int(rope_sin.stride(0)),
    int(raw_state_slot_ids.stride(0)),
    int(raw_k_ring.stride(0)),
    int(raw_k_ring.stride(1)),
    int(raw_logical_positions.stride(0)),
    int(raw_rope_positions.stride(0)),
    int(raw_rope_positions.stride(1)),
    int(compressed_cache.stride(0)),
    int(compressed_cache.stride(1)),
    int(compressed_block_table.stride(0)),
    float(caps.rms_norm_eps),
    int(raw_index_key.stride(0)),
    INDEX_HEAD_DIM=int(caps.index_head_dim),
    ROTARY_DIM=int(caps.index_rotary_dim),
    COMPRESS_RATIO=int(caps.compress_ratio),
    RING_CAPACITY=int(caps.raw_ring_capacity),
    COMPRESSED_PAGE_SIZE=int(caps.compressed_page_size),
    DCP_SIZE=int(getattr(caps, "dcp_size", 1)),
    DCP_RANK=int(getattr(caps, "dcp_rank", 0)),
    CP_INTERLEAVE=int(getattr(caps, "cp_kv_cache_interleave_size", 1)),
    POSITION_AXES=int(caps.position_axes),
    MROPE_INTERLEAVED=bool(caps.mrope_interleaved),
    MROPE_SECTION_0=section0,
    MROPE_SECTION_1=section1,
    ROPE_IS_BF16=rope_cos.dtype == torch.bfloat16,
    BLOCK_D=triton.next_power_of_2(int(caps.index_head_dim)),
    num_warps=4,)


def launch_commit_raw_ring(
    *,
    raw_index_key: torch.Tensor,
    query_positions: torch.Tensor,
    rope_positions: torch.Tensor,
    request_ids: torch.Tensor,
    query_start_loc: torch.Tensor,
    sequence_lengths: torch.Tensor,
    is_prefilling: torch.Tensor,
    raw_state_slot_ids: torch.Tensor,
    raw_k_ring: torch.Tensor,
    raw_logical_positions: torch.Tensor,
    raw_rope_positions: torch.Tensor,
    raw_interval_start_positions: torch.Tensor,
    caps,
) -> None:
    _launch_triton(_commit_raw_ring_kernel, (int(caps.max_batch), int(caps.raw_ring_capacity)), raw_index_key,
    query_positions,
    rope_positions,
    request_ids,
    query_start_loc,
    sequence_lengths,
    is_prefilling,
    raw_state_slot_ids,
    raw_k_ring,
    raw_logical_positions,
    raw_rope_positions,
    raw_interval_start_positions,
    int(rope_positions.stride(0)),
    int(rope_positions.stride(1)),
    int(raw_state_slot_ids.stride(0)),
    int(raw_k_ring.stride(0)),
    int(raw_k_ring.stride(1)),
    int(raw_logical_positions.stride(0)),
    int(raw_rope_positions.stride(0)),
    int(raw_rope_positions.stride(1)),
    int(raw_interval_start_positions.stride(0)),
    int(raw_index_key.stride(0)),
    INDEX_HEAD_DIM=int(caps.index_head_dim),
    POSITION_AXES=int(caps.position_axes),
    RING_CAPACITY=int(caps.raw_ring_capacity),
    BLOCK_D=triton.next_power_of_2(int(caps.index_head_dim)),
    num_warps=4,)


def launch_score_representatives(
    *,
    prepared_query: torch.Tensor,
    query_positions: torch.Tensor,
    request_ids: torch.Tensor,
    sequence_lengths: torch.Tensor,
    compressed_cache: torch.Tensor,
    compressed_block_table: torch.Tensor,
    scores: torch.Tensor,
    eligible_counts: torch.Tensor,
    merge_lengths: torch.Tensor,
    group_offset: int,
    group_count: int,
    caps,
) -> None:
    rows = int(prepared_query.shape[0])
    block_g = 32
    _launch_triton(_score_representatives_kernel, (rows, triton.cdiv(int(group_count), block_g)), prepared_query,
    query_positions,
    request_ids,
    sequence_lengths,
    compressed_cache,
    compressed_block_table,
    scores,
    eligible_counts,
    merge_lengths,
    int(compressed_cache.stride(0)),
    int(compressed_cache.stride(1)),
    int(compressed_block_table.stride(0)),
    int(scores.stride(0)),
    MAX_GROUPS=int(caps.max_groups),
    GROUP_OFFSET=int(group_offset),
    GROUP_COUNT=int(group_count),
    GROUP_BUDGET=int(caps.group_budget),
    INDEX_HEADS=int(caps.index_heads),
    INDEX_HEAD_DIM=int(caps.index_head_dim),
    COMPRESS_RATIO=int(caps.compress_ratio),
    COMPRESSED_PAGE_SIZE=int(caps.compressed_page_size),
    DCP_SIZE=int(getattr(caps, "dcp_size", 1)),
    DCP_RANK=int(getattr(caps, "dcp_rank", 0)),
    CP_INTERLEAVE=int(getattr(caps, "cp_kv_cache_interleave_size", 1)),
    BLOCK_G=block_g,
    BLOCK_D=triton.next_power_of_2(int(caps.index_head_dim)),
    num_warps=4,)


def launch_stage_topk_carry(
    *,
    prior_values: torch.Tensor,
    eligible_counts: torch.Tensor,
    scores: torch.Tensor,
    group_offset: int,
    group_budget: int,
) -> None:
    rows = int(scores.shape[0])
    _launch_triton(_stage_topk_carry_kernel, (rows,), prior_values,
    eligible_counts,
    scores,
    int(scores.stride(0)),
    GROUP_OFFSET=int(group_offset),
    GROUP_BUDGET=int(group_budget),
    BLOCK_K=triton.next_power_of_2(int(group_budget)),
    num_warps=8,)


def launch_topk_groups(
    *,
    scores: torch.Tensor,
    eligible_counts: torch.Tensor,
    topk_values: torch.Tensor,
    topk_group_ids: torch.Tensor,
    group_budget: int,
) -> None:
    # This is the exact single-row radix kernel, not the cooperative multi-CTA
    # persistent path.
    from ..dsa_indexer.tiled_topk import run_row_topk

    context = _support_launch_context.get()
    options = {}
    if context is not None:
        programs, compiling = context
        if compiling:
            if not isinstance(programs, dict):
                raise TypeError("QSA top-k compilation requires a mutable program map")
            options["launcher_sink"] = programs.setdefault("row_topk", {})
        else:
            options["launcher"] = programs["row_topk"]
    run_row_topk(
        row_logits=scores,
        lengths=eligible_counts,
        topk=int(group_budget),
        output_values=topk_values,
        output_indices=topk_group_ids,
        **options,
    )


def launch_remap_topk_group_ids(
    *,
    local_ids: torch.Tensor,
    prior_ids: torch.Tensor,
    eligible_counts: torch.Tensor,
    merge_lengths: torch.Tensor,
    group_offset: int,
    group_budget: int,
) -> None:
    rows = int(local_ids.shape[0])
    _launch_triton(_remap_topk_group_ids_kernel, (rows,), local_ids,
    prior_ids,
    eligible_counts,
    merge_lengths,
    GROUP_OFFSET=int(group_offset),
    GROUP_BUDGET=int(group_budget),
    BLOCK_K=triton.next_power_of_2(int(group_budget)),
    num_warps=8,)


def launch_stabilize_topk(
    *,
    scores: torch.Tensor,
    merge_lengths: torch.Tensor,
    prior_ids: torch.Tensor,
    eligible_counts: torch.Tensor,
    topk_values: torch.Tensor,
    topk_group_ids: torch.Tensor,
    tie_counts: torch.Tensor,
    greater_counts: torch.Tensor,
    stable_values: torch.Tensor,
    stable_ids: torch.Tensor,
    thresholds: torch.Tensor,
    greater_totals: torch.Tensor,
    group_offset: int,
    group_budget: int,
) -> None:
    """Make threshold ties exact and stable by retaining lower group IDs."""
    rows = int(scores.shape[0])
    block_k = triton.next_power_of_2(int(group_budget))
    from ._stable_select_cute import launch_stable_selection

    context = _support_launch_context.get()
    prepared = None
    if context is not None and not context[1]:
        prepared = context[0]["stable_selection"]
    raw = launch_stable_selection(
        scores=scores, merge_lengths=merge_lengths, prior_ids=prior_ids,
        eligible_counts=eligible_counts, topk_values=topk_values,
        stable_values=stable_values, stable_ids=stable_ids,
        group_offset=group_offset, group_budget=group_budget, prepared=prepared,
    )
    if context is not None and context[1]:
        context[0]["stable_selection"] = raw
    _launch_triton(_copy_stable_topk_kernel, (rows,), stable_values,
    stable_ids,
    topk_values,
    topk_group_ids,
    GROUP_BUDGET=int(group_budget),
    BLOCK_K=block_k,
    num_warps=8,)


def launch_expand_selected_groups(
    *,
    topk_group_ids: torch.Tensor,
    eligible_counts: torch.Tensor,
    query_positions: torch.Tensor,
    selected_positions: torch.Tensor,
    caps,
) -> None:
    rows = int(query_positions.shape[0])
    _launch_triton(_expand_selected_groups_kernel, (rows,), topk_group_ids,
    eligible_counts,
    query_positions,
    selected_positions,
    int(topk_group_ids.stride(0)),
    int(selected_positions.stride(0)),
    GROUP_BUDGET=int(caps.group_budget),
    COMPRESS_RATIO=int(caps.compress_ratio),
    SELECTION_WIDTH=int(caps.selection_width),
    DCP_SIZE=int(getattr(caps, "dcp_size", 1)),
    DCP_RANK=int(getattr(caps, "dcp_rank", 0)),
    CP_INTERLEAVE=int(getattr(caps, "cp_kv_cache_interleave_size", 1)),
    BLOCK_W=triton.next_power_of_2(int(caps.selection_width)),
    num_warps=8,)


def launch_expand_global_selected_groups(
    *,
    topk_group_ids: torch.Tensor,
    query_positions: torch.Tensor,
    selected_positions: torch.Tensor,
    caps,
) -> None:
    rows = int(query_positions.shape[0])
    _launch_triton(_expand_global_selected_groups_kernel, (rows,), topk_group_ids,
    query_positions,
    selected_positions,
    int(topk_group_ids.stride(0)),
    int(selected_positions.stride(0)),
    GROUP_BUDGET=int(caps.group_budget),
    COMPRESS_RATIO=int(caps.compress_ratio),
    SELECTION_WIDTH=int(caps.selection_width),
    DCP_SIZE=int(getattr(caps, "dcp_size", 1)),
    DCP_RANK=int(getattr(caps, "dcp_rank", 0)),
    CP_INTERLEAVE=int(getattr(caps, "cp_kv_cache_interleave_size", 1)),
    BLOCK_W=triton.next_power_of_2(int(caps.selection_width)),
    num_warps=8,)


_SUPPORT_WRAPPER_NAMES = (
    "launch_prepare_index_query",
    "launch_compress_completed_groups",
    "launch_commit_raw_ring",
    "launch_score_representatives",
    "launch_stage_topk_carry",
    "launch_topk_groups",
    "launch_remap_topk_group_ids",
    "launch_stabilize_topk",
    "launch_expand_selected_groups",
    "launch_expand_global_selected_groups",
)

for _support_wrapper_name in _SUPPORT_WRAPPER_NAMES:
    globals()[_support_wrapper_name] = _prepared_support_wrapper(
        globals()[_support_wrapper_name]
    )
del _support_wrapper_name


__all__ = [
    "launch_prepare_index_query",
    "launch_compress_completed_groups",
    "launch_commit_raw_ring",
    "launch_score_representatives",
    "launch_stage_topk_carry",
    "launch_topk_groups",
    "launch_remap_topk_group_ids",
    "launch_stabilize_topk",
    "launch_expand_selected_groups",
    "launch_expand_global_selected_groups",
]
