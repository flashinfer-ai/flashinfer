# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the FlashInfer project

import functools
from pathlib import Path

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cuda.bindings.driver as cuda
from cutlass._mlir.dialects import arith as mlir_arith
from cutlass._mlir.dialects import math as mlir_math
from cutlass.cutlass_dsl import dsl_user_op

from ..jit.cute_dsl_core import build_and_load_cute_dsl_kernel

BLOCK_T = 16
BLOCK_C = 128
_SOURCE_FILES = (str(Path(__file__).resolve()),)


@dsl_user_op
def _fma(a, b, c, *, loc=None, ip=None):
    return cutlass.Float32(
        mlir_math.fma(
            a.ir_value(loc=loc, ip=ip),
            b.ir_value(loc=loc, ip=ip),
            c.ir_value(loc=loc, ip=ip),
            fastmath=mlir_arith.FastMathFlags.none,
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def _step(h0, h1, h2, current, w0, w1, w2, w3, bias):
    value = _fma(h0, w0, bias)
    value = _fma(h1, w1, value)
    value = _fma(h2, w2, value)
    value = _fma(current, w3, value)
    return value / (cutlass.Float32(1.0) + cute.exp(-value, fastmath=True))


@cute.jit
def _token_tile(
    x,
    out,
    state,
    slot,
    channel,
    start,
    stop,
    h0,
    h1,
    h2,
    w0,
    w1,
    w2,
    w3,
    bias,
    tile: cutlass.Constexpr,
    speculative: cutlass.Constexpr,
    full: cutlass.Constexpr,
    state_tile,
    vector_state: cutlass.Constexpr,
    defer_state: cutlass.Constexpr,
):
    if cutlass.const_expr(defer_state):
        prior1, prior2 = h1.to(cutlass.BFloat16), h2.to(cutlass.BFloat16)
    inputs = cute.make_rmem_tensor((tile,), cutlass.BFloat16)
    for i in cutlass.range_constexpr(tile):
        if cutlass.const_expr(full):
            inputs[i] = x[start + i, channel]
        else:
            inputs[i] = cutlass.BFloat16(0.0)
            if start + i < stop:
                inputs[i] = x[start + i, channel]
    for i in cutlass.range_constexpr(tile):
        if full or start + i < stop:
            current = inputs[i].to(cutlass.Float32)
            value = _step(h0, h1, h2, current, w0, w1, w2, w3, bias)
            out[start + i, channel] = value.to(cutlass.BFloat16)
            h0, h1, h2 = h1, h2, current
            if cutlass.const_expr(speculative):
                if cutlass.const_expr(vector_state):
                    tid, _, _ = cute.arch.thread_idx()
                    state_tile[i + 2, tid] = inputs[i]
                elif cutlass.const_expr(not defer_state):
                    state[slot, channel, i + 2] = inputs[i]
    if cutlass.const_expr(defer_state):
        for i in cutlass.range_constexpr(tile):
            if start + i < stop:
                state[slot, channel, i + 2] = inputs[i]
        state[slot, channel, 0] = prior1
        state[slot, channel, 1] = prior2


@cute.jit
def _store_state_tile(
    state,
    state_tile,
    slot,
    channel_base,
    length,
    max_tokens: cutlass.Constexpr,
    block_c: cutlass.Constexpr,
):
    tid, _, _ = cute.arch.thread_idx()
    cute.arch.sync_threads()
    for i in cutlass.range_constexpr((max_tokens + 9) // 8):
        offset = (i * block_c + tid) * 8
        row = offset // block_c
        column = offset % block_c
        if row < length + 2:
            values = cute.make_rmem_tensor((8,), cutlass.BFloat16)
            source = cute.make_tensor(
                state_tile.iterator + offset, cute.make_layout((8,))
            )
            target_offset = (
                slot * state.stride[0] + row * state.stride[2] + channel_base + column
            )
            target = cute.make_tensor(
                state.iterator + cute.assume(target_offset, divby=8),
                cute.make_layout((8,)),
            )
            cute.autovec_copy(source, values)
            cute.autovec_copy(values, target)


@cute.jit
def _warp_chunk_prefix(offsets, base, block_t: cutlass.Constexpr):
    lane = cute.arch.lane_idx()
    seq = base + lane
    value = cutlass.Int32(0)
    if seq + 1 < offsets.shape[0]:
        length = cutlass.Int64(offsets[seq + 1]) - offsets[seq]
        value = cutlass.Int32((length + block_t - 1) // block_t)
    for shift in cutlass.range_constexpr(5):
        delta = 1 << shift
        lower = cute.arch.shuffle_sync_up(value, offset=delta, mask_and_clamp=0)
        if lane >= delta:
            value += lower
    return value


@cute.kernel
def _chunk_prefix(
    offsets: cute.Tensor, prefix: cute.Tensor, block_t: cutlass.Constexpr
):
    lane = cute.arch.lane_idx()
    count = offsets.shape[0] - 1
    carry = cutlass.Int32(0)
    if lane == 0:
        prefix[0] = carry
    for base in cutlass.range(0, count, 32):
        seq = base + lane
        value = _warp_chunk_prefix(offsets, base, block_t)
        if seq < count:
            prefix[seq + 1] = carry + value
        carry += cute.arch.shuffle_sync(value, 31)


@cute.jit
def _prefix_launch(
    offsets: cute.Tensor,
    prefix: cute.Tensor,
    block_t: cutlass.Constexpr,
    stream: cuda.CUstream,
):
    _chunk_prefix(offsets, prefix, block_t).launch(
        grid=(1, 1, 1), block=(32, 1, 1), stream=stream
    )


@cute.jit
def _warp_metadata(
    offsets,
    slots,
    accepted,
    seq,
    packed: cutlass.Constexpr,
    speculative: cutlass.Constexpr,
):
    lane = cute.arch.lane_idx()
    source = slots.iterator + seq
    if cutlass.const_expr(speculative):
        if lane == 1:
            source = accepted.iterator + seq
    if cutlass.const_expr(packed):
        if lane == 2:
            source = offsets.iterator + seq
        if lane == 3:
            source = offsets.iterator + seq + 1
    value = cutlass.Int32(0)
    if lane < 4:
        value = cute.make_tensor(source, cute.make_layout((1,)))[0]
    slot = cutlass.Int64(cute.arch.shuffle_sync(value, 0))
    count = cutlass.Int32(1)
    if cutlass.const_expr(speculative):
        count = cute.arch.shuffle_sync(value, 1)
    begin = cutlass.Int64(seq)
    end = begin + 1
    if cutlass.const_expr(packed):
        begin = cutlass.Int64(cute.arch.shuffle_sync(value, 2))
        end = cutlass.Int64(cute.arch.shuffle_sync(value, 3))
    return begin, end, slot, count


@cute.kernel
def _conv(
    x: cute.Tensor,
    weight: cute.Tensor,
    state: cute.Tensor,
    offsets: cute.Tensor,
    slots: cute.Tensor,
    initial: cute.Tensor,
    accepted: cute.Tensor,
    bias: cute.Tensor,
    out: cute.Tensor,
    prefix: cute.Tensor,
    prefill: cutlass.Constexpr,
    packed: cutlass.Constexpr,
    has_initial: cutlass.Constexpr,
    speculative: cutlass.Constexpr,
    has_bias: cutlass.Constexpr,
    max_tokens: cutlass.Constexpr,
    inline_prefix: cutlass.Constexpr,
    block_t: cutlass.Constexpr,
    block_c: cutlass.Constexpr,
    full_tile: cutlass.Constexpr,
    vector_state: cutlass.Constexpr,
    defer_state: cutlass.Constexpr,
    warp_metadata: cutlass.Constexpr,
    single_sequence: cutlass.Constexpr,
):
    task, channel_tile, _ = cute.arch.block_idx()
    tid, _, _ = cute.arch.thread_idx()
    channel = cutlass.Int64(channel_tile) * block_c + tid
    if cutlass.const_expr(vector_state):
        state_tile = utils.SmemAllocator().allocate_tensor(
            cutlass.BFloat16,
            cute.make_layout((max_tokens + 2, block_c), stride=(block_c, 1)),
            byte_alignment=16,
        )
    else:
        state_tile = state
    seq = task
    chunk = cutlass.Int32(0)
    active_task = task < slots.shape[0]
    if cutlass.const_expr(prefill):
        if cutlass.const_expr(single_sequence):
            seq = cutlass.Int32(0)
            chunk = task
            active_task = cutlass.Int64(task) * block_t < x.shape[0]
        elif cutlass.const_expr(inline_prefix):
            cumulative = _warp_chunk_prefix(offsets, cutlass.Int32(0), block_t)
            active_task = task < cute.arch.shuffle_sync(cumulative, 31)
            if active_task:
                seq = cute.arch.popc(cute.arch.vote_ballot_sync(cumulative <= task))
                previous = cutlass.Int32(0)
                if seq > 0:
                    previous = cute.arch.shuffle_sync(cumulative, seq - 1)
                chunk = task - previous
        else:
            active_task = task < prefix[slots.shape[0]]
        if (
            cutlass.const_expr(not inline_prefix and not single_sequence)
            and active_task
        ):
            lo = cutlass.Int32(0)
            hi = cutlass.Int32(slots.shape[0])
            while lo < hi:
                mid = lo + (hi - lo) // 2
                if prefix[mid + 1] <= task:
                    lo = mid + 1
                else:
                    hi = mid
            seq = lo
            chunk = task - prefix[seq]
    begin = cutlass.Int64(seq)
    end = begin + 1
    slot = cutlass.Int64(-1)
    count = cutlass.Int32(1)
    if cutlass.const_expr(warp_metadata):
        if active_task:
            begin, end, slot, count = _warp_metadata(
                offsets, slots, accepted, seq, packed, speculative
            )
    if active_task and channel < x.shape[1]:
        if cutlass.const_expr(not warp_metadata):
            begin = cutlass.Int64(seq)
            end = begin + 1
            if cutlass.const_expr(prefill and single_sequence):
                end = cutlass.Int64(x.shape[0])
            elif cutlass.const_expr(packed):
                begin = cutlass.Int64(offsets[seq])
                end = cutlass.Int64(offsets[seq + 1])
            slot = cutlass.Int64(slots[seq])
            count = cutlass.Int32(1)
            if cutlass.const_expr(speculative):
                count = accepted[seq]
        start = begin
        stop = end
        if cutlass.const_expr(prefill):
            start += cutlass.Int64(chunk) * block_t
            stop = cute.min(start + block_t, end)
        history_offset = cutlass.Int64(0)
        live = slot != -1
        if cutlass.const_expr(speculative):
            live = live and count > 0
            history_offset = cutlass.Int64(count) - 1
        if start < stop:
            if live:
                h0 = cutlass.Float32(0.0)
                h1 = cutlass.Float32(0.0)
                h2 = cutlass.Float32(0.0)
                if cutlass.const_expr(prefill):
                    if chunk > 0:
                        h0 = x[start - 3, channel].to(cutlass.Float32)
                        h1 = x[start - 2, channel].to(cutlass.Float32)
                        h2 = x[start - 1, channel].to(cutlass.Float32)
                    elif cutlass.const_expr(has_initial):
                        if initial[seq] != 0:
                            h0 = state[slot, channel, 0].to(cutlass.Float32)
                            h1 = state[slot, channel, 1].to(cutlass.Float32)
                            h2 = state[slot, channel, 2].to(cutlass.Float32)
                else:
                    h0 = state[slot, channel, history_offset].to(cutlass.Float32)
                    h1 = state[slot, channel, history_offset + 1].to(cutlass.Float32)
                    h2 = state[slot, channel, history_offset + 2].to(cutlass.Float32)

                if cutlass.const_expr(prefill):
                    if chunk == 0:
                        length = end - begin
                        tail0, tail1 = h1, h2
                        if length >= 3:
                            tail0 = x[end - 3, channel].to(cutlass.Float32)
                            tail1 = x[end - 2, channel].to(cutlass.Float32)
                        elif length == 2:
                            tail0 = h2
                            tail1 = x[begin, channel].to(cutlass.Float32)
                        state[slot, channel, 0] = tail0.to(cutlass.BFloat16)
                        state[slot, channel, 1] = tail1.to(cutlass.BFloat16)
                        state[slot, channel, 2] = x[end - 1, channel]
                elif cutlass.const_expr(speculative):
                    if cutlass.const_expr(vector_state):
                        state_tile[0, tid] = h1.to(cutlass.BFloat16)
                        state_tile[1, tid] = h2.to(cutlass.BFloat16)
                    elif cutlass.const_expr(not defer_state):
                        state[slot, channel, 0] = h1.to(cutlass.BFloat16)
                        state[slot, channel, 1] = h2.to(cutlass.BFloat16)

                taps = cute.make_rmem_tensor((4,), weight.element_type)
                source = cute.make_tensor(
                    weight.iterator + cute.assume(channel * 4, divby=4),
                    cute.make_layout((4,)),
                )
                cute.autovec_copy(source, taps)
                w0 = taps[0].to(cutlass.Float32)
                w1 = taps[1].to(cutlass.Float32)
                w2 = taps[2].to(cutlass.Float32)
                w3 = taps[3].to(cutlass.Float32)
                b = cutlass.Float32(0.0)
                if cutlass.const_expr(has_bias):
                    b = bias[channel].to(cutlass.Float32)
                if cutlass.const_expr(prefill or max_tokens > 0):
                    tile = block_t if cutlass.const_expr(prefill) else max_tokens
                    if full_tile and stop - start == tile:
                        _token_tile(
                            x,
                            out,
                            state,
                            slot,
                            channel,
                            start,
                            stop,
                            h0,
                            h1,
                            h2,
                            w0,
                            w1,
                            w2,
                            w3,
                            b,
                            tile,
                            speculative,
                            True,
                            state_tile,
                            vector_state,
                            defer_state,
                        )
                    else:
                        _token_tile(
                            x,
                            out,
                            state,
                            slot,
                            channel,
                            start,
                            stop,
                            h0,
                            h1,
                            h2,
                            w0,
                            w1,
                            w2,
                            w3,
                            b,
                            tile,
                            speculative,
                            False,
                            state_tile,
                            vector_state,
                            defer_state,
                        )
                else:
                    for token in cutlass.range(start, stop):
                        current = x[token, channel].to(cutlass.Float32)
                        value = _step(h0, h1, h2, current, w0, w1, w2, w3, b)
                        out[token, channel] = value.to(cutlass.BFloat16)
                        h0, h1, h2 = h1, h2, current
                        if cutlass.const_expr(speculative):
                            state[slot, channel, token - begin + 2] = x[token, channel]
                if cutlass.const_expr(vector_state):
                    _store_state_tile(
                        state,
                        state_tile,
                        slot,
                        cutlass.Int64(channel_tile) * block_c,
                        stop - start,
                        max_tokens,
                        block_c,
                    )
                if cutlass.const_expr(not prefill and not speculative):
                    state[slot, channel, 0] = h0.to(cutlass.BFloat16)
                    state[slot, channel, 1] = h1.to(cutlass.BFloat16)
                    state[slot, channel, 2] = h2.to(cutlass.BFloat16)
            else:
                for token in cutlass.range(start, stop):
                    out[token, channel] = cutlass.BFloat16(0.0)


@cute.jit
def _launch(
    x,
    weight,
    state,
    offsets,
    slots,
    initial,
    accepted,
    bias,
    out,
    prefix,
    prefill: cutlass.Constexpr,
    packed: cutlass.Constexpr,
    has_initial: cutlass.Constexpr,
    speculative: cutlass.Constexpr,
    has_bias: cutlass.Constexpr,
    max_tokens: cutlass.Constexpr,
    inline_prefix: cutlass.Constexpr,
    block_t: cutlass.Constexpr,
    block_c: cutlass.Constexpr,
    full_tile: cutlass.Constexpr,
    vector_state: cutlass.Constexpr,
    defer_state: cutlass.Constexpr,
    warp_metadata: cutlass.Constexpr,
    single_sequence: cutlass.Constexpr,
    stream: cuda.CUstream,
):
    tasks = slots.shape[0]
    if cutlass.const_expr(prefill):
        rows = cutlass.Int64(x.shape[0])
        tasks = cutlass.Int32(
            cute.min(rows, (rows + block_t - 1) // block_t + slots.shape[0])
        )
        if cutlass.const_expr(single_sequence):
            tasks = cutlass.Int32((rows + block_t - 1) // block_t)
    _conv(
        x,
        weight,
        state,
        offsets,
        slots,
        initial,
        accepted,
        bias,
        out,
        prefix,
        prefill,
        packed,
        has_initial,
        speculative,
        has_bias,
        max_tokens,
        inline_prefix,
        block_t,
        block_c,
        full_tile,
        vector_state,
        defer_state,
        warp_metadata,
        single_sequence,
    ).launch(
        grid=(tasks, cute.ceil_div(x.shape[1], block_c), 1),
        block=(block_c, 1, 1),
        stream=stream,
    )


def _vector(dtype):
    return cute.runtime.make_fake_tensor(
        dtype, (cute.sym_int(),), (1,), assumed_align=max(1, dtype.width // 8)
    )


@functools.cache
def get_prefix_kernel(arch, device_index, block_t=BLOCK_T):
    return build_and_load_cute_dsl_kernel(
        "causal_conv1d",
        f"prefix_t{block_t}_{arch}",
        lambda: cute.compile(
            _prefix_launch,
            _vector(cutlass.Int32),
            _vector(cutlass.Int32),
            block_t,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options=f"--enable-tvm-ffi --gpu-arch={arch}",
        ),
        extra_key_files=_SOURCE_FILES,
    )


@functools.cache
def get_conv_kernel(
    prefill,
    packed,
    has_initial,
    speculative,
    weight_fp32,
    bias_kind,
    arch,
    device_index,
    channel_contiguous,
    max_tokens,
    inline_prefix,
    weight_alignment,
    block_t=BLOCK_T,
    block_c=BLOCK_C,
    full_tile=False,
    static_channels=0,
    vector_state=False,
    static_history=0,
    warp_metadata=False,
    static_strides=(),
    single_sequence=False,
):
    defer_state = (
        speculative
        and not vector_state
        and max_tokens > 0
        and (channel_contiguous or max_tokens <= 8)
    )
    rows, channels, slots, history = (cute.sym_int() for _ in range(4))
    if static_channels:
        channels = static_channels
    input_stride = static_channels or cute.sym_int64()
    output_stride = static_channels or cute.sym_int64()
    state_strides = (
        (cute.sym_int64(), 1, static_channels or cute.sym_int64())
        if channel_contiguous
        else (cute.sym_int64(), static_history or cute.sym_int64(), 1)
    )
    if static_strides:
        input_stride, output_stride, channel_stride, history_stride = static_strides
        state_strides = (cute.sym_int64(), channel_stride, history_stride)
    x = cute.runtime.make_fake_tensor(
        cutlass.BFloat16,
        (rows, channels),
        (input_stride, 1),
        assumed_align=2,
    )
    state = cute.runtime.make_fake_tensor(
        cutlass.BFloat16,
        (slots, channels, history),
        state_strides,
        assumed_align=16 if vector_state else 2,
    )
    weight_dtype = cutlass.Float32 if weight_fp32 else cutlass.BFloat16
    bias_dtype = cutlass.Float32 if bias_kind == "fp32" else cutlass.BFloat16
    weight = cute.runtime.make_fake_tensor(
        weight_dtype, (channels, 4), (4, 1), assumed_align=weight_alignment
    )
    bias = _vector(bias_dtype)
    initial_dtype = cutlass.Uint8 if has_initial else cutlass.Int32
    output = cute.runtime.make_fake_tensor(
        cutlass.BFloat16,
        (rows, channels),
        (output_stride, 1),
        assumed_align=2,
    )
    name = (
        f"w4_p{int(prefill)}_r{int(packed)}_h{int(has_initial)}"
        f"_s{int(speculative)}_w{int(weight_fp32)}_b{bias_kind}"
        f"_l{int(channel_contiguous)}_u{max_tokens}_pfx{int(inline_prefix)}"
        f"_a{weight_alignment}_t{block_t}_c{block_c}_f{int(full_tile)}"
        f"_d{static_channels}_svec{int(vector_state)}_dh{static_history}_wm{int(warp_metadata)}"
        f"_str{'x'.join(map(str, static_strides))}_ss{int(single_sequence)}_{arch}"
    )
    return build_and_load_cute_dsl_kernel(
        "causal_conv1d",
        name,
        lambda: cute.compile(
            _launch,
            x,
            weight,
            state,
            _vector(cutlass.Int32),
            _vector(cutlass.Int32),
            _vector(initial_dtype),
            _vector(cutlass.Int32),
            bias,
            output,
            _vector(cutlass.Int32),
            prefill,
            packed,
            has_initial,
            speculative,
            bias_kind != "none",
            max_tokens,
            inline_prefix,
            block_t,
            block_c,
            full_tile,
            vector_state,
            defer_state,
            warp_metadata,
            single_sequence,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options=f"--enable-tvm-ffi --gpu-arch={arch}",
        ),
        extra_key_files=_SOURCE_FILES,
    )
