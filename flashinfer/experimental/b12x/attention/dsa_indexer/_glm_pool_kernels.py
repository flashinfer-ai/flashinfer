# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM C4 softmax pooling, tail updates, and FP8 query rotation."""

import triton
import triton.language as tl


@triton.jit
def _fwht_stage(x, N: tl.constexpr, GROUPS: tl.constexpr, STRIDE: tl.constexpr):
    x3 = tl.reshape(x, (GROUPS, 2, STRIDE))
    x3 = tl.trans(x3, 0, 2, 1)
    a, b = tl.split(x3)
    x3 = tl.join(a + b, a - b)
    x3 = tl.trans(x3, 0, 2, 1)
    return tl.reshape(x3, (N,))


@triton.jit
def _fwht128(x):
    x = _fwht_stage(x, 128, 64, 1)
    x = _fwht_stage(x, 128, 32, 2)
    x = _fwht_stage(x, 128, 16, 4)
    x = _fwht_stage(x, 128, 8, 8)
    x = _fwht_stage(x, 128, 4, 16)
    x = _fwht_stage(x, 128, 2, 32)
    x = _fwht_stage(x, 128, 1, 64)
    return x * 0.08838834764831845


@triton.jit
def _fwht_quant_kernel(
    q,
    q_out,
    scale_out,
    weights,
    rows,
    HEAD_DIM: tl.constexpr,
    FP8_MAX: tl.constexpr,
    BLOCK_R: tl.constexpr,
    SCALE_WEIGHTS: tl.constexpr,
    WEIGHT_NORM: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid * BLOCK_R + tl.arange(0, BLOCK_R)
    row_mask = row < rows
    dim = tl.arange(0, HEAD_DIM)
    x = tl.load(
        q + row[:, None] * HEAD_DIM + dim[None, :],
        mask=row_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    n: tl.constexpr = BLOCK_R * HEAD_DIM
    x = tl.reshape(x, (n,))
    x = _fwht_stage(x, n, BLOCK_R * 64, 1)
    x = _fwht_stage(x, n, BLOCK_R * 32, 2)
    x = _fwht_stage(x, n, BLOCK_R * 16, 4)
    x = _fwht_stage(x, n, BLOCK_R * 8, 8)
    x = _fwht_stage(x, n, BLOCK_R * 4, 16)
    x = _fwht_stage(x, n, BLOCK_R * 2, 32)
    x = _fwht_stage(x, n, BLOCK_R, 64)
    x = (x * 0.08838834764831845).to(tl.bfloat16).to(tl.float32)
    x = tl.reshape(x, (BLOCK_R, HEAD_DIM))
    absmax = tl.maximum(tl.max(tl.abs(x), axis=1), 1e-4)
    scale = tl.exp2(tl.ceil(tl.log2(absmax / FP8_MAX)))
    quantized = tl.minimum(tl.maximum(x / scale[:, None], -FP8_MAX), FP8_MAX)
    tl.store(
        q_out + row[:, None] * HEAD_DIM + dim[None, :],
        quantized,
        mask=row_mask[:, None],
    )
    tl.store(scale_out + row, scale, mask=row_mask)
    if SCALE_WEIGHTS:
        weight = tl.load(weights + row, mask=row_mask, other=0.0).to(tl.float32)
        weight *= scale
        weight *= WEIGHT_NORM
        tl.store(weights + row, weight, mask=row_mask)


@triton.jit
def _write_pool(
    cache_fp8,
    cache_fp32,
    cache_location,
    x,
    dim,
    dim_mask,
    write_mask,
    page_stride_bytes,
    PAGE_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    x = _fwht128(x.to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    absmax = tl.maximum(tl.max(tl.abs(x), axis=0), 1e-4)
    scale = tl.exp2(tl.ceil(tl.log2(absmax / FP8_MAX)))
    quantized = tl.minimum(tl.maximum(x / scale, -FP8_MAX), FP8_MAX)
    location = cache_location.to(tl.int64)
    page = location // PAGE_SIZE
    offset = location - page * PAGE_SIZE
    page_bytes = page * page_stride_bytes.to(tl.int64)
    k_offsets = page_bytes + offset * HEAD_DIM + dim.to(tl.int64)
    scale_byte_offset = page_bytes + PAGE_SIZE * HEAD_DIM + offset * 4
    tl.store(cache_fp8 + k_offsets, quantized, mask=dim_mask & write_mask)
    tl.store(cache_fp32 + scale_byte_offset // 4, scale, mask=write_mask)


@triton.jit
def _decode_update_kernel(
    cache_fp8,
    cache_fp32,
    tail,
    state_slots,
    query_start_loc,
    key,
    gate,
    ape,
    slot_mapping,
    positions,
    page_stride_bytes,
    model_block_size,
    parent_stride_pages,
    tail_stride_0,
    tail_stride_1,
    tail_stride_2,
    key_stride_0,
    gate_stride_0,
    ape_stride_0,
    PAGE_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    POOL_SIZE: tl.constexpr,
    TAIL_CAPACITY: tl.constexpr,
    FP8_MAX: tl.constexpr,
    BLOCK_D: tl.constexpr,
    PACKED_MAIN_SLOTS: tl.constexpr,
):
    request = tl.program_id(0)
    state_slot = tl.load(state_slots + request).to(tl.int64)
    start = tl.load(query_start_loc + request)
    end = tl.load(query_start_loc + request + 1)
    dim = tl.arange(0, BLOCK_D)
    dim_mask = (dim < HEAD_DIM) & (state_slot >= 0)
    for row in tl.range(start, end):
        position = tl.load(positions + row).to(tl.int64)
        physical_slot = position % TAIL_CAPACITY
        current_key = tl.load(
            key + row * key_stride_0 + dim, mask=dim_mask, other=0.0
        ).to(tl.float32)
        current_gate = tl.load(
            gate + row * gate_stride_0 + dim, mask=dim_mask, other=0.0
        ).to(tl.float32)
        cache_location = tl.load(slot_mapping + row).to(tl.int64)
        if PACKED_MAIN_SLOTS:
            parent_page = cache_location // model_block_size
            model_page_offset = cache_location - parent_page * model_block_size
            pool_offset = model_page_offset // POOL_SIZE
            child_page = pool_offset // PAGE_SIZE
            child_offset = pool_offset - child_page * PAGE_SIZE
            cache_location = (
                parent_page * parent_stride_pages + child_page
            ) * PAGE_SIZE + child_offset
            write_pool = (
                (position % POOL_SIZE == POOL_SIZE - 1)
                & (cache_location >= 0)
                & (state_slot >= 0)
            )
        else:
            write_pool = (cache_location >= 0) & (state_slot >= 0)
        pool_mask = (dim < HEAD_DIM) & write_pool
        maximum = tl.full((BLOCK_D,), -float("inf"), tl.float32)
        for slot in tl.static_range(0, POOL_SIZE):
            score = current_gate
            if slot != POOL_SIZE - 1:
                tail_offset = (
                    state_slot * tail_stride_0
                    + tail_stride_1
                    + ((position - POOL_SIZE + 1 + slot) % TAIL_CAPACITY)
                    * tail_stride_2
                    + dim
                )
                score = tl.load(tail + tail_offset, mask=pool_mask, other=0.0).to(
                    tl.float32
                )
            score += tl.load(
                ape + slot * ape_stride_0 + dim,
                mask=pool_mask,
                other=0.0,
            ).to(tl.float32)
            maximum = tl.maximum(maximum, score)
        numerator = tl.zeros((BLOCK_D,), tl.float32)
        denominator = tl.zeros((BLOCK_D,), tl.float32)
        for slot in tl.static_range(0, POOL_SIZE):
            value = current_key
            score = current_gate
            if slot != POOL_SIZE - 1:
                base = (
                    state_slot * tail_stride_0
                    + ((position - POOL_SIZE + 1 + slot) % TAIL_CAPACITY)
                    * tail_stride_2
                    + dim
                )
                value = tl.load(tail + base, mask=pool_mask, other=0.0).to(tl.float32)
                score = tl.load(
                    tail + base + tail_stride_1, mask=pool_mask, other=0.0
                ).to(tl.float32)
            score += tl.load(
                ape + slot * ape_stride_0 + dim,
                mask=pool_mask,
                other=0.0,
            ).to(tl.float32)
            weight = tl.exp(score - maximum)
            numerator += value * weight
            denominator += weight
        pooled = numerator / tl.maximum(denominator, 1e-20)
        _write_pool(
            cache_fp8,
            cache_fp32,
            cache_location,
            pooled,
            dim,
            dim < HEAD_DIM,
            write_pool,
            page_stride_bytes,
            PAGE_SIZE,
            HEAD_DIM,
            FP8_MAX,
        )
        base = state_slot * tail_stride_0 + physical_slot * tail_stride_2
        tl.store(tail + base + dim, current_key, mask=dim_mask)
        tl.store(tail + base + tail_stride_1 + dim, current_gate, mask=dim_mask)


@triton.jit
def _prefill_pool_kernel(
    cache_fp8,
    cache_fp32,
    tail,
    state_slots,
    query_start_loc,
    key,
    gate,
    ape,
    slot_mapping,
    positions,
    request_offset,
    page_stride_bytes,
    model_block_size,
    parent_stride_pages,
    tail_stride_0,
    tail_stride_1,
    tail_stride_2,
    key_stride_0,
    gate_stride_0,
    ape_stride_0,
    PAGE_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    POOL_SIZE: tl.constexpr,
    TAIL_CAPACITY: tl.constexpr,
    FP8_MAX: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    request = request_offset + tl.program_id(0)
    pool_ordinal = tl.program_id(1)
    state_slot = tl.load(state_slots + request).to(tl.int64)
    start = tl.load(query_start_loc + request).to(tl.int64)
    end = tl.load(query_start_loc + request + 1).to(tl.int64)
    has_rows = end > start
    first_position = tl.load(positions + start, mask=has_rows, other=0).to(tl.int64)
    first_physical_slot = first_position % POOL_SIZE
    first_completion = start + (POOL_SIZE - 1 - first_physical_slot)
    completion_row = first_completion + pool_ordinal * POOL_SIZE
    write_pool = has_rows & (completion_row < end) & (state_slot >= 0)
    completion_position = tl.load(
        positions + completion_row, mask=write_pool, other=-1
    ).to(tl.int64)
    write_pool &= completion_position % POOL_SIZE == POOL_SIZE - 1
    main_cache_location = tl.load(
        slot_mapping + completion_row, mask=write_pool, other=-1
    ).to(tl.int64)
    write_pool &= main_cache_location >= 0
    parent_page = main_cache_location // model_block_size
    model_page_offset = main_cache_location - parent_page * model_block_size
    pool_offset = model_page_offset // POOL_SIZE
    child_page = pool_offset // PAGE_SIZE
    child_offset = pool_offset - child_page * PAGE_SIZE
    cache_location = (
        parent_page * parent_stride_pages + child_page
    ) * PAGE_SIZE + child_offset

    dim = tl.arange(0, BLOCK_D)
    dim_mask = dim < HEAD_DIM
    maximum = tl.full((BLOCK_D,), -float("inf"), tl.float32)
    for slot in tl.static_range(0, POOL_SIZE):
        source_row = completion_row - (POOL_SIZE - 1 - slot)
        from_current_chunk = source_row >= start
        tail_slot = (completion_position - POOL_SIZE + 1 + slot) % TAIL_CAPACITY
        tail_offset = (
            state_slot * tail_stride_0 + tail_stride_1 + tail_slot * tail_stride_2 + dim
        )
        current_score = tl.load(
            gate + source_row * gate_stride_0 + dim,
            mask=dim_mask & write_pool & from_current_chunk,
            other=0.0,
        ).to(tl.float32)
        previous_score = tl.load(
            tail + tail_offset,
            mask=dim_mask & write_pool & ~from_current_chunk,
            other=0.0,
        ).to(tl.float32)
        score = tl.where(from_current_chunk, current_score, previous_score)
        score += tl.load(
            ape + slot * ape_stride_0 + dim,
            mask=dim_mask & write_pool,
            other=0.0,
        ).to(tl.float32)
        maximum = tl.maximum(maximum, score)

    numerator = tl.zeros((BLOCK_D,), tl.float32)
    denominator = tl.zeros((BLOCK_D,), tl.float32)
    for slot in tl.static_range(0, POOL_SIZE):
        source_row = completion_row - (POOL_SIZE - 1 - slot)
        from_current_chunk = source_row >= start
        tail_slot = (completion_position - POOL_SIZE + 1 + slot) % TAIL_CAPACITY
        tail_offset = state_slot * tail_stride_0 + tail_slot * tail_stride_2 + dim
        current_value = tl.load(
            key + source_row * key_stride_0 + dim,
            mask=dim_mask & write_pool & from_current_chunk,
            other=0.0,
        ).to(tl.float32)
        previous_value = tl.load(
            tail + tail_offset,
            mask=dim_mask & write_pool & ~from_current_chunk,
            other=0.0,
        ).to(tl.float32)
        value = tl.where(from_current_chunk, current_value, previous_value)
        current_score = tl.load(
            gate + source_row * gate_stride_0 + dim,
            mask=dim_mask & write_pool & from_current_chunk,
            other=0.0,
        ).to(tl.float32)
        previous_score = tl.load(
            tail + tail_offset + tail_stride_1,
            mask=dim_mask & write_pool & ~from_current_chunk,
            other=0.0,
        ).to(tl.float32)
        score = tl.where(from_current_chunk, current_score, previous_score)
        score += tl.load(
            ape + slot * ape_stride_0 + dim,
            mask=dim_mask & write_pool,
            other=0.0,
        ).to(tl.float32)
        weight = tl.exp(score - maximum)
        numerator += value * weight
        denominator += weight

    pooled = numerator / tl.maximum(denominator, 1e-20)
    _write_pool(
        cache_fp8,
        cache_fp32,
        cache_location,
        pooled,
        dim,
        dim_mask,
        write_pool,
        page_stride_bytes,
        PAGE_SIZE,
        HEAD_DIM,
        FP8_MAX,
    )


@triton.jit
def _prefill_tail_kernel(
    tail,
    state_slots,
    query_start_loc,
    key,
    gate,
    positions,
    request_offset,
    tail_stride_0,
    tail_stride_1,
    tail_stride_2,
    key_stride_0,
    gate_stride_0,
    HEAD_DIM: tl.constexpr,
    TAIL_CAPACITY: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    request = request_offset + tl.program_id(0)
    state_slot = tl.load(state_slots + request).to(tl.int64)
    start = tl.load(query_start_loc + request).to(tl.int64)
    end = tl.load(query_start_loc + request + 1).to(tl.int64)
    last_row = end - 1
    has_rows = end > start
    last_position = tl.load(positions + last_row, mask=has_rows, other=0).to(tl.int64)
    last_physical_slot = last_position % TAIL_CAPACITY
    dim = tl.arange(0, BLOCK_D)
    dim_mask = (dim < HEAD_DIM) & has_rows & (state_slot >= 0)
    for slot in tl.static_range(0, TAIL_CAPACITY):
        distance = (last_physical_slot - slot + TAIL_CAPACITY) % TAIL_CAPACITY
        source_row = last_row - distance
        source_in_chunk = source_row >= start
        source_position = tl.load(
            positions + source_row,
            mask=has_rows & source_in_chunk,
            other=-1,
        ).to(tl.int64)
        write_slot = (
            dim_mask & source_in_chunk & (source_position % TAIL_CAPACITY == slot)
        )
        value = tl.load(
            key + source_row * key_stride_0 + dim,
            mask=write_slot,
            other=0.0,
        )
        score = tl.load(
            gate + source_row * gate_stride_0 + dim,
            mask=write_slot,
            other=0.0,
        )
        tail_offset = state_slot * tail_stride_0 + slot * tail_stride_2 + dim
        tl.store(tail + tail_offset, value, mask=write_slot)
        tl.store(tail + tail_offset + tail_stride_1, score, mask=write_slot)
