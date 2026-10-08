# Copyright (c) 2026 KDA Team
# SPDX-License-Identifier: MIT
# See LICENSE.kda-for-kda.txt for the full license.

"""Independent exact recurrent KDA kernel for partial-tail sequences."""

from __future__ import annotations

import triton
import triton.language as tl

HEAD_DIM = 128
VALUE_BLOCK = 32


@triton.jit(do_not_specialize=["num_seqs"])
def _recurrent_kda_kernel(
    q,
    k,
    v,
    g,
    beta,
    a_log,
    dt_bias,
    initial_state,
    out,
    final_state,
    cu_seqlens,
    num_seqs: tl.int64,
    num_heads: tl.constexpr,
    scale: tl.constexpr,
    lower_bound: tl.constexpr,
    use_initial_state: tl.constexpr,
    store_final_state: tl.constexpr,
):
    program = tl.program_id(0)
    value_block = program % 4
    sequence_head = program // 4
    sequence = sequence_head // num_heads
    head = sequence_head % num_heads

    begin = tl.load(cu_seqlens + sequence).to(tl.int64)
    end = tl.load(cu_seqlens + sequence + 1).to(tl.int64)
    key_offset = tl.arange(0, 128)
    value_offset = value_block * 32 + tl.arange(0, 32)

    state = tl.zeros((32, 128), dtype=tl.float32)
    if use_initial_state:
        state_ptr = (
            initial_state
            + (sequence * num_heads + head) * 128 * 128
            + value_offset[:, None] * 128
            + key_offset[None, :]
        )
        state += tl.load(state_ptr).to(tl.float32)

    q_ptr = q + (begin * num_heads + head) * 128 + key_offset
    k_ptr = k + (begin * num_heads + head) * 128 + key_offset
    v_ptr = v + (begin * num_heads + head) * 128 + value_offset
    g_ptr = g + (begin * num_heads + head) * 128 + key_offset
    beta_ptr = beta + begin * num_heads + head
    out_ptr = out + (begin * num_heads + head) * 128 + value_offset
    bias = tl.load(dt_bias + head * 128 + key_offset).to(tl.float32)
    decay_scale = tl.exp(tl.load(a_log + head).to(tl.float32))

    for _ in tl.range(begin, end, num_stages=2):
        query = tl.load(q_ptr).to(tl.float32)
        key = tl.load(k_ptr).to(tl.float32)
        query *= tl.rsqrt(tl.sum(query * query) + 1.0e-6) * scale
        key *= tl.rsqrt(tl.sum(key * key) + 1.0e-6)

        gate = tl.load(g_ptr).to(tl.float32) + bias
        log_decay = lower_bound * tl.sigmoid(decay_scale * gate)
        state *= tl.exp(log_decay[None, :])

        residual = tl.load(v_ptr).to(tl.float32)
        residual -= tl.sum(state * key[None, :], axis=1)
        residual *= tl.sigmoid(tl.load(beta_ptr).to(tl.float32))
        state += residual[:, None] * key[None, :]
        output = tl.sum(state * query[None, :], axis=1)
        tl.store(out_ptr, output)

        q_ptr += num_heads * 128
        k_ptr += num_heads * 128
        v_ptr += num_heads * 128
        g_ptr += num_heads * 128
        beta_ptr += num_heads
        out_ptr += num_heads * 128

    if store_final_state:
        final_ptr = (
            final_state
            + (sequence * num_heads + head) * 128 * 128
            + value_offset[:, None] * 128
            + key_offset[None, :]
        )
        tl.store(final_ptr, state)


def launch_recurrent(
    *,
    q,
    k,
    v,
    g,
    beta,
    a_log,
    dt_bias,
    out,
    initial_state,
    final_state,
    cu_seqlens,
    scale,
    lower_bound,
):
    """Launch the exact token recurrence without an FLA dependency."""

    num_seqs = cu_seqlens.numel() - 1
    num_heads = q.shape[-2]
    pointer = q
    _recurrent_kda_kernel[(num_seqs * num_heads * (HEAD_DIM // VALUE_BLOCK),)](
        q,
        k,
        v,
        g,
        beta,
        a_log,
        dt_bias,
        initial_state if initial_state is not None else pointer,
        out,
        final_state if final_state is not None else pointer,
        cu_seqlens,
        num_seqs,
        num_heads=num_heads,
        scale=float(scale),
        lower_bound=float(lower_bound),
        use_initial_state=initial_state is not None,
        store_final_state=final_state is not None,
        num_warps=4,
        num_stages=2,
    )


__all__ = ["launch_recurrent"]
