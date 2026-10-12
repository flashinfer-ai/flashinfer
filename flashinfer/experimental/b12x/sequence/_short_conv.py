"""Shared packed-request history and causal depthwise convolution kernels."""

import triton
import triton.language as tl


@triton.jit
def _decode_short_conv_kernel(
    input_ptr,
    conv_weight_ptr,
    query_start_loc_ptr,
    state_slot_ids_ptr,
    state_is_fresh_ptr,
    num_seqs_ptr,
    num_tokens_ptr,
    conv_state_ptr,
    out_ptr,
    CHANNELS: tl.constexpr,
    KERNEL_SIZE: tl.constexpr,
    DILATION: tl.constexpr,
    INPUT_ROW_STRIDE: tl.constexpr,
    STATE_SLOT_STRIDE: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    token = tl.program_id(0)
    channels = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    channel_mask = channels < CHANNELS
    num_tokens = tl.load(num_tokens_ptr).to(tl.int32)
    num_seqs = tl.load(num_seqs_ptr).to(tl.int32)
    token_live = (token < num_tokens) & (num_seqs > 0)
    low = tl.zeros((), tl.int32)
    high = tl.maximum(num_seqs, 1)
    while low + 1 < high:
        middle = (low + high) // 2
        start = tl.load(query_start_loc_ptr + middle, mask=token_live, other=0)
        low = tl.where(start <= token, middle, low)
        high = tl.where(start <= token, high, middle)
    slot = tl.load(state_slot_ids_ptr + low, mask=token_live, other=-1).to(tl.int64)
    fresh = tl.load(state_is_fresh_ptr + low, mask=token_live, other=True).to(tl.int1)
    live = token_live & (slot >= 0)
    STATE_LENGTH: tl.constexpr = DILATION * (KERNEL_SIZE - 1)
    state_offsets = slot * STATE_SLOT_STRIDE + channels.to(tl.int64) * STATE_LENGTH

    # Each channel owns its history; load it before the in-place shift.
    history = ()
    for position in tl.static_range(STATE_LENGTH):
        value = tl.load(
            conv_state_ptr + state_offsets + position,
            mask=live & channel_mask & ~fresh,
            other=0.0,
        )
        history += (value,)
    current = tl.load(
        input_ptr + token.to(tl.int64) * INPUT_ROW_STRIDE + channels.to(tl.int64),
        mask=live & channel_mask,
        other=0.0,
    )
    accumulator = tl.zeros((BLOCK_C,), tl.float32)
    for tap in tl.static_range(KERNEL_SIZE):
        if tap == KERNEL_SIZE - 1:
            value = current
        else:
            value = history[tap * DILATION]
        weight = tl.load(
            conv_weight_ptr + channels.to(tl.int64) * KERNEL_SIZE + tap,
            mask=channel_mask,
            other=0.0,
        ).to(tl.float32)
        accumulator += value.to(tl.float32) * weight
    convolved = accumulator.to(tl.bfloat16).to(tl.float32)
    activated = (convolved * tl.sigmoid(convolved)).to(tl.bfloat16)
    tl.store(
        out_ptr + token.to(tl.int64) * CHANNELS + channels.to(tl.int64),
        tl.where(live, activated, 0.0),
        mask=channel_mask,
    )
    for position in tl.static_range(STATE_LENGTH):
        if position == STATE_LENGTH - 1:
            value = current
        else:
            value = history[position + 1]
        tl.store(
            conv_state_ptr + state_offsets + position,
            value,
            mask=live & channel_mask,
        )


@triton.jit
def _speculative_decode_short_conv_kernel(
    input_ptr,
    conv_weight_ptr,
    query_start_loc_ptr,
    state_slot_ids_ptr,
    state_is_fresh_ptr,
    num_accepted_tokens_ptr,
    num_seqs_ptr,
    num_tokens_ptr,
    conv_state_ptr,
    out_ptr,
    CHANNELS: tl.constexpr,
    STATE_LENGTH: tl.constexpr,
    STATE_CAPACITY: tl.constexpr,
    stride_state_slot: tl.constexpr,
    stride_state_channel: tl.constexpr,
    stride_state_position: tl.constexpr,
    MAX_SPECULATIVE: tl.constexpr,
    KERNEL_SIZE: tl.constexpr,
    DILATION: tl.constexpr,
    INPUT_ROW_STRIDE: tl.constexpr,
    MAX_SEQS: tl.constexpr,
    MAX_TOKENS: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Speculative decode in one launch: the history gather, convolution and state update that
    ``_prepare_history``, ``_short_conv`` and ``_update_state`` perform, with identical arithmetic.

    Each program owns one request's channels, so it reads that request's history before
    rewriting it in place. Program ``MAX_SEQS`` zeroes the rows past the packed tokens.
    """
    request = tl.program_id(0)
    channels = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    channel_mask = channels < CHANNELS
    if request == MAX_SEQS:
        num_tokens = tl.load(num_tokens_ptr).to(tl.int32)
        for token in range(num_tokens, MAX_TOKENS):
            tl.store(
                out_ptr + token * CHANNELS + channels.to(tl.int64),
                tl.zeros((BLOCK_C,), tl.bfloat16),
                mask=channel_mask,
            )
    else:
        num_seqs = tl.load(num_seqs_ptr).to(tl.int32)
        request_live = request < num_seqs
        slot = tl.load(state_slot_ids_ptr + request, mask=request_live, other=-1).to(tl.int64)
        fresh = tl.load(state_is_fresh_ptr + request, mask=request_live, other=True).to(tl.int1)
        live = request_live & (slot >= 0)
        accepted = tl.load(num_accepted_tokens_ptr + request, mask=request_live, other=0).to(tl.int32)
        rollback = tl.where(request_live, tl.minimum(accepted - 1, MAX_SPECULATIVE), 0)
        query_start = tl.load(query_start_loc_ptr + request, mask=request_live, other=0).to(tl.int32)
        query_length = tl.load(query_start_loc_ptr + request + 1, mask=request_live, other=0).to(tl.int32) - query_start
        state_base = slot * stride_state_slot + channels.to(tl.int64) * stride_state_channel

        history = ()
        for position in tl.static_range(STATE_LENGTH):
            source_position = position + rollback
            history += (
                tl.load(
                    conv_state_ptr + state_base + source_position.to(tl.int64) * stride_state_position,
                    mask=live & channel_mask & ~fresh & (source_position < STATE_CAPACITY),
                    other=0.0,
                ),
            )
        query = ()
        for relative in tl.static_range(MAX_SPECULATIVE + 1):
            query += (
                tl.load(
                    input_ptr + (query_start + relative).to(tl.int64) * INPUT_ROW_STRIDE + channels.to(tl.int64),
                    mask=request_live & channel_mask & (relative < query_length),
                    other=0.0,
                ),
            )

        for relative in tl.static_range(MAX_SPECULATIVE + 1):
            accumulator = tl.zeros((BLOCK_C,), tl.float32)
            for tap in tl.static_range(KERNEL_SIZE):
                if relative >= (KERNEL_SIZE - 1 - tap) * DILATION:
                    source = query[relative - (KERNEL_SIZE - 1 - tap) * DILATION].to(tl.float32)
                elif STATE_LENGTH + relative >= (KERNEL_SIZE - 1 - tap) * DILATION:
                    source = history[STATE_LENGTH + relative - (KERNEL_SIZE - 1 - tap) * DILATION].to(tl.float32)
                else:
                    source = tl.zeros((BLOCK_C,), tl.float32)
                weight = tl.load(
                    conv_weight_ptr + channels.to(tl.int64) * KERNEL_SIZE + tap,
                    mask=channel_mask,
                    other=0.0,
                ).to(tl.float32)
                accumulator += source * weight
            convolved = accumulator.to(tl.bfloat16).to(tl.float32)
            activated = (convolved * tl.sigmoid(convolved)).to(tl.bfloat16)
            tl.store(
                out_ptr + (query_start + relative).to(tl.int64) * CHANNELS + channels.to(tl.int64),
                tl.where(live, activated, 0.0),
                mask=request_live & channel_mask & (relative < query_length),
            )

        # Commit the first query token and keep the rest as the speculative tail.
        for position in tl.static_range(STATE_LENGTH):
            if position + 1 < STATE_LENGTH:
                value = history[position + 1]
            else:
                value = query[position + 1 - STATE_LENGTH]
            tl.store(
                conv_state_ptr + state_base + position * stride_state_position,
                value,
                mask=live & channel_mask & (query_length > 0),
            )
        for candidate in tl.static_range(MAX_SPECULATIVE):
            tl.store(
                conv_state_ptr + state_base + (STATE_LENGTH + candidate) * stride_state_position,
                query[candidate + 1],
                mask=live & channel_mask & (query_length > 0),
            )


@triton.jit
def _request_ids_kernel(
    query_start_loc_ptr,
    num_seqs_ptr,
    num_tokens_ptr,
    request_ids_ptr,
    MAX_TOKENS: tl.constexpr,
):
    token = tl.program_id(0)
    num_tokens = tl.load(num_tokens_ptr).to(tl.int32)
    num_seqs = tl.load(num_seqs_ptr).to(tl.int32)
    live = token < num_tokens

    low = tl.zeros((), tl.int32)
    high = tl.maximum(num_seqs, 1)
    while low + 1 < high:
        middle = (low + high) // 2
        start = tl.load(query_start_loc_ptr + middle, mask=live, other=0).to(tl.int32)
        low = tl.where(start <= token, middle, low)
        high = tl.where(start <= token, high, middle)
    request = tl.where(live & (num_seqs > 0), low, -1)
    tl.store(request_ids_ptr + token, request)


@triton.jit
def _prepare_history_kernel(
    conv_state_ptr,
    state_slot_ids_ptr,
    state_is_fresh_ptr,
    num_accepted_tokens_ptr,
    request_is_prefill_ptr,
    num_seqs_ptr,
    gathered_state_ptr,
    CHANNELS: tl.constexpr,
    STATE_LENGTH: tl.constexpr,
    STATE_CAPACITY: tl.constexpr,
    stride_state_slot: tl.constexpr,
    stride_state_channel: tl.constexpr,
    stride_state_position: tl.constexpr,
    MAX_SPECULATIVE: tl.constexpr,
    DECODE: tl.constexpr,
    MIXED: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    request = tl.program_id(0)
    channel_block = tl.program_id(1)
    channels = channel_block * BLOCK_C + tl.arange(0, BLOCK_C)
    channel_mask = channels < CHANNELS
    num_seqs = tl.load(num_seqs_ptr).to(tl.int32)
    request_live = request < num_seqs
    slot = tl.load(state_slot_ids_ptr + request, mask=request_live, other=-1).to(
        tl.int64
    )
    fresh = tl.load(state_is_fresh_ptr + request, mask=request_live, other=True).to(
        tl.int1
    )
    live = request_live & (slot >= 0)
    if MIXED:
        request_decode = request_live & ~tl.load(
            request_is_prefill_ptr + request,
            mask=request_live,
            other=True,
        ).to(tl.int1)
    else:
        request_decode = request_live & DECODE
    if DECODE or MIXED:
        accepted = tl.load(
            num_accepted_tokens_ptr + request, mask=request_live, other=0
        ).to(tl.int32)
        rollback = tl.minimum(accepted - 1, MAX_SPECULATIVE)
        rollback = tl.where(request_decode, rollback, 0)
    else:
        rollback = tl.zeros((), tl.int32)

    for position in tl.static_range(0, STATE_LENGTH):
        source_position = position + rollback
        source_offsets = (
            slot.to(tl.int64) * stride_state_slot
            + channels.to(tl.int64) * stride_state_channel
            + source_position.to(tl.int64) * stride_state_position
        )
        value = tl.load(
            conv_state_ptr + source_offsets,
            mask=live & channel_mask & ~fresh & (source_position < STATE_CAPACITY),
            other=0.0,
        )
        gathered_offsets = (
            request.to(tl.int64) * CHANNELS + channels.to(tl.int64)
        ) * STATE_LENGTH + position
        tl.store(
            gathered_state_ptr + gathered_offsets,
            value,
            mask=channel_mask,
        )


@triton.jit
def _short_conv_kernel(
    input_ptr,
    gathered_state_ptr,
    conv_weight_ptr,
    query_start_loc_ptr,
    request_ids_ptr,
    state_slot_ids_ptr,
    num_tokens_ptr,
    out_ptr,
    CHANNELS: tl.constexpr,
    STATE_LENGTH: tl.constexpr,
    KERNEL_SIZE: tl.constexpr,
    DILATION: tl.constexpr,
    BLOCK_C: tl.constexpr,
    ADD_RESIDUAL: tl.constexpr,
    INPUT_ROW_STRIDE: tl.constexpr = None,
):
    if INPUT_ROW_STRIDE is None:
        INPUT_ROW_STRIDE = CHANNELS
    token = tl.program_id(0)
    channel_block = tl.program_id(1)
    channels = channel_block * BLOCK_C + tl.arange(0, BLOCK_C)
    channel_mask = channels < CHANNELS
    num_tokens = tl.load(num_tokens_ptr).to(tl.int32)
    token_live = token < num_tokens
    request = tl.load(request_ids_ptr + token).to(tl.int32)
    slot = tl.load(
        state_slot_ids_ptr + request,
        mask=token_live & (request >= 0),
        other=-1,
    ).to(tl.int64)
    live = token_live & (request >= 0) & (slot >= 0)
    query_start = tl.load(
        query_start_loc_ptr + request,
        mask=live,
        other=0,
    ).to(tl.int32)
    query_relative = token - query_start

    accumulator = tl.zeros((BLOCK_C,), tl.float32)
    for tap in tl.static_range(0, KERNEL_SIZE):
        history_distance = (KERNEL_SIZE - 1 - tap) * DILATION
        source_relative = query_relative - history_distance
        from_query = source_relative >= 0
        query_token = query_start + source_relative
        query_offsets = query_token.to(tl.int64) * INPUT_ROW_STRIDE + channels.to(
            tl.int64
        )
        query_value = tl.load(
            input_ptr + query_offsets,
            mask=live & channel_mask & from_query,
            other=0.0,
        ).to(tl.float32)
        history_position = STATE_LENGTH + source_relative
        history_offsets = (
            request.to(tl.int64) * CHANNELS + channels.to(tl.int64)
        ) * STATE_LENGTH + history_position
        history_value = tl.load(
            gathered_state_ptr + history_offsets,
            mask=(
                live
                & channel_mask
                & ~from_query
                & (history_position >= 0)
                & (history_position < STATE_LENGTH)
            ),
            other=0.0,
        ).to(tl.float32)
        source = tl.where(from_query, query_value, history_value)
        weight = tl.load(
            conv_weight_ptr + channels.to(tl.int64) * KERNEL_SIZE + tap,
            mask=channel_mask,
            other=0.0,
        ).to(tl.float32)
        accumulator += source * weight

    conv_bf16 = accumulator.to(tl.bfloat16)
    conv_float = conv_bf16.to(tl.float32)
    activated = (conv_float * tl.sigmoid(conv_float)).to(tl.bfloat16)
    output_offsets = token.to(tl.int64) * CHANNELS + channels.to(tl.int64)
    contribution = activated
    if ADD_RESIDUAL:
        u = tl.load(out_ptr + output_offsets, mask=channel_mask, other=0.0).to(
            tl.bfloat16
        )
        contribution = (u + activated).to(tl.bfloat16)
    tl.store(
        out_ptr + output_offsets,
        tl.where(live, contribution, 0.0),
        mask=channel_mask,
    )


@triton.jit
def _short_conv_tiled_kernel(
    input_ptr,
    gathered_state_ptr,
    conv_weight_ptr,
    query_start_loc_ptr,
    request_ids_ptr,
    state_slot_ids_ptr,
    num_tokens_ptr,
    out_ptr,
    CHANNELS: tl.constexpr,
    STATE_LENGTH: tl.constexpr,
    KERNEL_SIZE: tl.constexpr,
    DILATION: tl.constexpr,
    BLOCK_C: tl.constexpr,
    INPUT_ROW_STRIDE: tl.constexpr,
    BLOCK_T: tl.constexpr,
    MAX_TOKENS: tl.constexpr,
):
    """``_short_conv_kernel`` over a tile of tokens, with identical arithmetic."""
    tokens = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    channels = tl.max_contiguous(
        tl.multiple_of(tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C), BLOCK_C),
        BLOCK_C,
    )
    channel_mask = channels < CHANNELS
    num_tokens = tl.load(num_tokens_ptr).to(tl.int32)
    token_live = tokens < num_tokens
    request = tl.load(request_ids_ptr + tokens, mask=token_live, other=-1).to(tl.int32)
    slot = tl.load(
        state_slot_ids_ptr + request,
        mask=token_live & (request >= 0),
        other=-1,
    ).to(tl.int64)
    live = token_live & (request >= 0) & (slot >= 0)
    query_start = tl.load(query_start_loc_ptr + request, mask=live, other=0).to(tl.int32)
    query_relative = tokens - query_start

    accumulator = tl.zeros((BLOCK_T, BLOCK_C), tl.float32)
    # Only tokens within the first taps of their request read carried history;
    # tiles without such a token skip that gather.
    history_reach = (KERNEL_SIZE - 1) * DILATION
    if tl.min(tl.where(live, query_relative, history_reach), axis=0) < history_reach:
        for tap in tl.static_range(0, KERNEL_SIZE):
            history_distance = (KERNEL_SIZE - 1 - tap) * DILATION
            source_relative = query_relative - history_distance
            from_query = source_relative >= 0
            query_rows = tl.multiple_of(
                (query_start + source_relative).to(tl.int64) * INPUT_ROW_STRIDE, 16
            )
            query_value = tl.load(
                input_ptr + query_rows[:, None] + channels[None, :],
                mask=(live & from_query)[:, None] & channel_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            history_position = STATE_LENGTH + source_relative
            history_offsets = (
                request.to(tl.int64)[:, None] * CHANNELS
                + channels.to(tl.int64)[None, :]
            ) * STATE_LENGTH + history_position[:, None]
            history_value = tl.load(
                gathered_state_ptr + history_offsets,
                mask=(
                    live
                    & ~from_query
                    & (history_position >= 0)
                    & (history_position < STATE_LENGTH)
                )[:, None]
                & channel_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            source = tl.where(from_query[:, None], query_value, history_value)
            weight = tl.load(
                conv_weight_ptr + channels.to(tl.int64) * KERNEL_SIZE + tap,
                mask=channel_mask,
                other=0.0,
            ).to(tl.float32)
            accumulator += source * weight[None, :]
    else:
        for tap in tl.static_range(0, KERNEL_SIZE):
            source_relative = query_relative - (KERNEL_SIZE - 1 - tap) * DILATION
            query_rows = tl.multiple_of(
                (query_start + source_relative).to(tl.int64) * INPUT_ROW_STRIDE, 16
            )
            query_value = tl.load(
                input_ptr + query_rows[:, None] + channels[None, :],
                mask=live[:, None] & channel_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            weight = tl.load(
                conv_weight_ptr + channels.to(tl.int64) * KERNEL_SIZE + tap,
                mask=channel_mask,
                other=0.0,
            ).to(tl.float32)
            accumulator += query_value * weight[None, :]

    conv_float = accumulator.to(tl.bfloat16).to(tl.float32)
    activated = (conv_float * tl.sigmoid(conv_float)).to(tl.bfloat16)
    output_rows = tl.multiple_of(tokens.to(tl.int64) * CHANNELS, 16)
    tl.store(
        out_ptr + output_rows[:, None] + channels[None, :],
        tl.where(live[:, None], activated, 0.0),
        mask=(tokens < MAX_TOKENS)[:, None] & channel_mask[None, :],
    )


@triton.jit
def _update_state_kernel(
    input_ptr,
    gathered_state_ptr,
    query_start_loc_ptr,
    state_slot_ids_ptr,
    request_is_prefill_ptr,
    num_seqs_ptr,
    conv_state_ptr,
    CHANNELS: tl.constexpr,
    STATE_LENGTH: tl.constexpr,
    STATE_CAPACITY: tl.constexpr,
    stride_state_slot: tl.constexpr,
    stride_state_channel: tl.constexpr,
    stride_state_position: tl.constexpr,
    MAX_SPECULATIVE: tl.constexpr,
    DECODE: tl.constexpr,
    MIXED: tl.constexpr,
    BLOCK_C: tl.constexpr,
    INPUT_ROW_STRIDE: tl.constexpr = None,
):
    if INPUT_ROW_STRIDE is None:
        INPUT_ROW_STRIDE = CHANNELS
    request = tl.program_id(0)
    channel_block = tl.program_id(1)
    channels = channel_block * BLOCK_C + tl.arange(0, BLOCK_C)
    channel_mask = channels < CHANNELS
    num_seqs = tl.load(num_seqs_ptr).to(tl.int32)
    request_live = request < num_seqs
    slot = tl.load(state_slot_ids_ptr + request, mask=request_live, other=-1).to(
        tl.int64
    )
    live = request_live & (slot >= 0)
    query_start = tl.load(query_start_loc_ptr + request, mask=request_live, other=0).to(
        tl.int32
    )
    query_end = tl.load(
        query_start_loc_ptr + request + 1, mask=request_live, other=0
    ).to(tl.int32)
    query_length = query_end - query_start
    if MIXED:
        request_decode = request_live & ~tl.load(
            request_is_prefill_ptr + request,
            mask=request_live,
            other=True,
        ).to(tl.int1)
    else:
        request_decode = request_live & DECODE

    for position in tl.static_range(0, STATE_LENGTH):
        if DECODE or MIXED:
            commit_current = tl.where(query_length > 0, 1, 0)
            decode_source_position = position + commit_current
            prefill_source_position = position + query_length
            source_position = tl.where(
                request_decode,
                decode_source_position,
                prefill_source_position,
            )
        else:
            source_position = position + query_length
        from_query = source_position >= STATE_LENGTH
        query_relative = source_position - STATE_LENGTH
        query_token = query_start + query_relative
        query_offsets = query_token.to(tl.int64) * INPUT_ROW_STRIDE + channels.to(
            tl.int64
        )
        query_value = tl.load(
            input_ptr + query_offsets,
            mask=(
                live
                & channel_mask
                & from_query
                & (query_relative >= 0)
                & (query_relative < query_length)
            ),
            other=0.0,
        )
        history_offsets = (
            request.to(tl.int64) * CHANNELS + channels.to(tl.int64)
        ) * STATE_LENGTH + source_position
        history_value = tl.load(
            gathered_state_ptr + history_offsets,
            mask=live & channel_mask & ~from_query,
            other=0.0,
        )
        value = tl.where(from_query, query_value, history_value)
        destination_offsets = (
            slot.to(tl.int64) * stride_state_slot
            + channels.to(tl.int64) * stride_state_channel
            + position * stride_state_position
        )
        tl.store(
            conv_state_ptr + destination_offsets,
            value,
            mask=live & channel_mask & (query_length > 0),
        )

    for candidate in tl.static_range(0, MAX_SPECULATIVE):
        if DECODE or MIXED:
            query_relative = candidate + 1
            query_token = query_start + query_relative
            query_offsets = query_token.to(tl.int64) * INPUT_ROW_STRIDE + channels.to(
                tl.int64
            )
            candidate_value = tl.load(
                input_ptr + query_offsets,
                mask=(
                    live
                    & request_decode
                    & channel_mask
                    & (query_relative < query_length)
                ),
                other=0.0,
            )
        else:
            candidate_value = tl.zeros((BLOCK_C,), tl.float32)
        destination_offsets = (
            slot.to(tl.int64) * stride_state_slot
            + channels.to(tl.int64) * stride_state_channel
            + (STATE_LENGTH + candidate) * stride_state_position
        )
        tl.store(
            conv_state_ptr + destination_offsets,
            candidate_value,
            mask=live & channel_mask & (query_length > 0),
        )
