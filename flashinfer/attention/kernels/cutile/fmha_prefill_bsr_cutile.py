# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Migrated from Ocean cuTile suite to public cuda.tile API.
# Source: ocean/src/tilegym/suites/flashinfer/cutile/fmha_prefill_bsr.py
# Changes vs Ocean source:
#   - Removed `from tilegym.backend import register_impl` (TileGym-private)
#   - Removed @register_impl decorator; function exposed directly as fmha_prefill_bsr_cutile
#   - Stripped {$nv-internal-release-nvt} comment lines (2 per-line markers, not whole-file)
#   - Added output buffer pre-zeroing per PR #3426 pattern

import math
from types import SimpleNamespace
from typing import Optional

import cuda.tile as ct
import torch
from cuda.tile import RoundingMode as RMd
from cuda.tile.tune import exhaustive_search

# Module-level tune caches for prefill kernels
_prefill_paged_lpt_tune_cache: dict = {}
_prefill_paged_tune_cache: dict = {}
_prefill_ragged_lpt_tune_cache: dict = {}
_prefill_ragged_tune_cache: dict = {}

INV_LOG_2 = 1.0 / math.log(2)

ConstInt = ct.Constant[int]
ConstBool = ct.Constant[bool]
ConstFloat = ct.Constant[float]


def _get_prefill_autotune_configs(page_size=None):
    configs = [
        SimpleNamespace(BLOCK_M=128, BLOCK_N=32, occupancy=1, num_ctas=1),
        SimpleNamespace(BLOCK_M=128, BLOCK_N=64, occupancy=1, num_ctas=1),
        SimpleNamespace(BLOCK_M=128, BLOCK_N=128, occupancy=1, num_ctas=1),
    ]

    if torch.cuda.get_device_capability()[0] != 9:
        configs.extend(
            [
                SimpleNamespace(BLOCK_M=256, BLOCK_N=128, occupancy=1, num_ctas=1),
                SimpleNamespace(BLOCK_M=256, BLOCK_N=64, occupancy=1, num_ctas=1),
                SimpleNamespace(BLOCK_M=128, BLOCK_N=16, occupancy=1, num_ctas=1),
                SimpleNamespace(BLOCK_M=128, BLOCK_N=16, occupancy=2, num_ctas=1),
                SimpleNamespace(BLOCK_M=128, BLOCK_N=32, occupancy=2, num_ctas=1),
                SimpleNamespace(BLOCK_M=128, BLOCK_N=64, occupancy=2, num_ctas=1),
                SimpleNamespace(BLOCK_M=128, BLOCK_N=128, occupancy=2, num_ctas=1),
            ]
        )

    for cfg in configs:
        if page_size is not None and cfg.BLOCK_N > page_size:
            continue
        yield cfg


def _load_page_prefill(
    cache,
    block_tables,
    page_table_offset,
    page,
    token,
    off_kv_h,
    NUM_PAGES,
    LOAD_BLOCK_N,
    BLOCK_D,
    _PAGE_SIZE,
    dim3_offset=0,
    LATENCY=3,
):
    """
    Load data from paged cache via TMA for prefill attention.

    For single page, issues one TMA load.
    For multiple pages, issues N independent TMA loads and concatenates via ct.cat.
    """
    PAD_ZERO = ct.PaddingMode.ZERO
    if NUM_PAGES == 1:
        page_id = ct.gather(block_tables, (page_table_offset + page,), padding_value=0).item()
        data = ct.reshape(
            ct.load(
                cache,
                index=(page_id, token // LOAD_BLOCK_N, off_kv_h, dim3_offset),
                shape=(1, LOAD_BLOCK_N, 1, BLOCK_D),
                order=(0, 1, 2, 3),
                allow_tma=True,
                latency=LATENCY,
                padding_mode=PAD_ZERO,
            ),
            (LOAD_BLOCK_N, BLOCK_D),
        )
    elif NUM_PAGES == 2:
        pg0 = ct.gather(block_tables, (page_table_offset + page,), padding_value=0).item()
        d0 = ct.reshape(
            ct.load(
                cache,
                index=(pg0, token // LOAD_BLOCK_N, off_kv_h, dim3_offset),
                shape=(1, LOAD_BLOCK_N, 1, BLOCK_D),
                order=(0, 1, 2, 3),
                allow_tma=True,
                latency=LATENCY,
                padding_mode=PAD_ZERO,
            ),
            (LOAD_BLOCK_N, BLOCK_D),
        )
        pg1 = ct.gather(block_tables, (page_table_offset + page + 1,), padding_value=0).item()
        d1 = ct.reshape(
            ct.load(
                cache,
                index=(pg1, 0, off_kv_h, dim3_offset),
                shape=(1, LOAD_BLOCK_N, 1, BLOCK_D),
                order=(0, 1, 2, 3),
                allow_tma=True,
                latency=LATENCY,
                padding_mode=PAD_ZERO,
            ),
            (LOAD_BLOCK_N, BLOCK_D),
        )
        data = ct.cat((d0, d1), 0)
    elif NUM_PAGES == 4:
        pg0 = ct.gather(block_tables, (page_table_offset + page,), padding_value=0).item()
        d0 = ct.reshape(
            ct.load(
                cache,
                index=(pg0, token // LOAD_BLOCK_N, off_kv_h, dim3_offset),
                shape=(1, LOAD_BLOCK_N, 1, BLOCK_D),
                order=(0, 1, 2, 3),
                allow_tma=True,
                latency=LATENCY,
                padding_mode=PAD_ZERO,
            ),
            (LOAD_BLOCK_N, BLOCK_D),
        )
        pg1 = ct.gather(block_tables, (page_table_offset + page + 1,), padding_value=0).item()
        d1 = ct.reshape(
            ct.load(
                cache,
                index=(pg1, 0, off_kv_h, dim3_offset),
                shape=(1, LOAD_BLOCK_N, 1, BLOCK_D),
                order=(0, 1, 2, 3),
                allow_tma=True,
                latency=LATENCY,
                padding_mode=PAD_ZERO,
            ),
            (LOAD_BLOCK_N, BLOCK_D),
        )
        pg2 = ct.gather(block_tables, (page_table_offset + page + 2,), padding_value=0).item()
        d2 = ct.reshape(
            ct.load(
                cache,
                index=(pg2, 0, off_kv_h, dim3_offset),
                shape=(1, LOAD_BLOCK_N, 1, BLOCK_D),
                order=(0, 1, 2, 3),
                allow_tma=True,
                latency=LATENCY,
                padding_mode=PAD_ZERO,
            ),
            (LOAD_BLOCK_N, BLOCK_D),
        )
        pg3 = ct.gather(block_tables, (page_table_offset + page + 3,), padding_value=0).item()
        d3 = ct.reshape(
            ct.load(
                cache,
                index=(pg3, 0, off_kv_h, dim3_offset),
                shape=(1, LOAD_BLOCK_N, 1, BLOCK_D),
                order=(0, 1, 2, 3),
                allow_tma=True,
                latency=LATENCY,
                padding_mode=PAD_ZERO,
            ),
            (LOAD_BLOCK_N, BLOCK_D),
        )
        data = ct.cat((ct.cat((d0, d1), 0), ct.cat((d2, d3), 0)), 0)
    elif NUM_PAGES == 8:
        pg0 = ct.gather(block_tables, (page_table_offset + page,), padding_value=0).item()
        d0 = ct.reshape(
            ct.load(
                cache,
                index=(pg0, token // LOAD_BLOCK_N, off_kv_h, dim3_offset),
                shape=(1, LOAD_BLOCK_N, 1, BLOCK_D),
                order=(0, 1, 2, 3),
                allow_tma=True,
                latency=LATENCY,
                padding_mode=PAD_ZERO,
            ),
            (LOAD_BLOCK_N, BLOCK_D),
        )
        pg1 = ct.gather(block_tables, (page_table_offset + page + 1,), padding_value=0).item()
        d1 = ct.reshape(
            ct.load(
                cache,
                index=(pg1, 0, off_kv_h, dim3_offset),
                shape=(1, LOAD_BLOCK_N, 1, BLOCK_D),
                order=(0, 1, 2, 3),
                allow_tma=True,
                latency=LATENCY,
                padding_mode=PAD_ZERO,
            ),
            (LOAD_BLOCK_N, BLOCK_D),
        )
        pg2 = ct.gather(block_tables, (page_table_offset + page + 2,), padding_value=0).item()
        d2 = ct.reshape(
            ct.load(
                cache,
                index=(pg2, 0, off_kv_h, dim3_offset),
                shape=(1, LOAD_BLOCK_N, 1, BLOCK_D),
                order=(0, 1, 2, 3),
                allow_tma=True,
                latency=LATENCY,
                padding_mode=PAD_ZERO,
            ),
            (LOAD_BLOCK_N, BLOCK_D),
        )
        pg3 = ct.gather(block_tables, (page_table_offset + page + 3,), padding_value=0).item()
        d3 = ct.reshape(
            ct.load(
                cache,
                index=(pg3, 0, off_kv_h, dim3_offset),
                shape=(1, LOAD_BLOCK_N, 1, BLOCK_D),
                order=(0, 1, 2, 3),
                allow_tma=True,
                latency=LATENCY,
                padding_mode=PAD_ZERO,
            ),
            (LOAD_BLOCK_N, BLOCK_D),
        )
        pg4 = ct.gather(block_tables, (page_table_offset + page + 4,), padding_value=0).item()
        d4 = ct.reshape(
            ct.load(
                cache,
                index=(pg4, 0, off_kv_h, dim3_offset),
                shape=(1, LOAD_BLOCK_N, 1, BLOCK_D),
                order=(0, 1, 2, 3),
                allow_tma=True,
                latency=LATENCY,
                padding_mode=PAD_ZERO,
            ),
            (LOAD_BLOCK_N, BLOCK_D),
        )
        pg5 = ct.gather(block_tables, (page_table_offset + page + 5,), padding_value=0).item()
        d5 = ct.reshape(
            ct.load(
                cache,
                index=(pg5, 0, off_kv_h, dim3_offset),
                shape=(1, LOAD_BLOCK_N, 1, BLOCK_D),
                order=(0, 1, 2, 3),
                allow_tma=True,
                latency=LATENCY,
                padding_mode=PAD_ZERO,
            ),
            (LOAD_BLOCK_N, BLOCK_D),
        )
        pg6 = ct.gather(block_tables, (page_table_offset + page + 6,), padding_value=0).item()
        d6 = ct.reshape(
            ct.load(
                cache,
                index=(pg6, 0, off_kv_h, dim3_offset),
                shape=(1, LOAD_BLOCK_N, 1, BLOCK_D),
                order=(0, 1, 2, 3),
                allow_tma=True,
                latency=LATENCY,
                padding_mode=PAD_ZERO,
            ),
            (LOAD_BLOCK_N, BLOCK_D),
        )
        pg7 = ct.gather(block_tables, (page_table_offset + page + 7,), padding_value=0).item()
        d7 = ct.reshape(
            ct.load(
                cache,
                index=(pg7, 0, off_kv_h, dim3_offset),
                shape=(1, LOAD_BLOCK_N, 1, BLOCK_D),
                order=(0, 1, 2, 3),
                allow_tma=True,
                latency=LATENCY,
                padding_mode=PAD_ZERO,
            ),
            (LOAD_BLOCK_N, BLOCK_D),
        )
        data = ct.cat(
            (
                ct.cat((ct.cat((d0, d1), 0), ct.cat((d2, d3), 0)), 0),
                ct.cat((ct.cat((d4, d5), 0), ct.cat((d6, d7), 0)), 0),
            ),
            0,
        )
    return data


def _load_page_wrapper_prefill(
    curr_n,
    cache,
    block_tables,
    page_table_offset,
    off_kv_h,
    PAGE_SIZE,
    BLOCK_N,
    BLOCK_D,
    LOAD_BLOCK_N,
    dim3_offset=0,
    LATENCY=3,
):
    NUM_PAGES = BLOCK_N // LOAD_BLOCK_N
    page = curr_n // PAGE_SIZE
    token = curr_n % PAGE_SIZE
    return _load_page_prefill(
        cache,
        block_tables,
        page_table_offset,
        page,
        token,
        off_kv_h,
        NUM_PAGES,
        LOAD_BLOCK_N,
        BLOCK_D,
        PAGE_SIZE,
        dim3_offset,
        LATENCY,
    )


def _prefill_attention_paged_body(
    batch_id,
    head_id,
    seq_block_id,
    query,
    key_cache,
    value_cache,
    actual_seq_lens_q,
    actual_seq_lens_kv,
    batch_offsets,
    block_tables,
    output,
    lse_output,
    k_scale: ConstFloat,
    v_scale: ConstFloat,
    N_KV_HEADS: ConstInt,
    PAGE_SIZE: ConstInt,
    BLOCK_M: ConstInt,
    BLOCK_N: ConstInt,
    BLOCK_D: ConstInt,
    BLOCK_R: ConstInt,
    QUERY_GROUP_SIZE: ConstInt,
    stride_block_table,
    IS_CAUSAL: ConstBool,
    LOAD_BLOCK_N: ConstInt,
):
    # Load sequence info
    seq_start_idx_tile = ct.gather(batch_offsets, (batch_id,), padding_value=0)
    seq_start_index = seq_start_idx_tile.item()

    seq_len_q_tile = ct.gather(actual_seq_lens_q, (batch_id,), padding_value=0)
    seq_len_q = seq_len_q_tile.item()

    seq_len_kv_tile = ct.gather(actual_seq_lens_kv, (batch_id,), padding_value=0)
    seq_len_kv = seq_len_kv_tile.item()

    start_m = BLOCK_M * seq_block_id

    if start_m >= seq_len_q:
        return

    off_kv_h = head_id // QUERY_GROUP_SIZE
    qk_scale = k_scale * INV_LOG_2
    PAD_ZERO = ct.PaddingMode.ZERO

    page_table_offset = batch_id * stride_block_table

    q_seq = query.slice(axis=0, start=seq_start_index, stop=seq_start_index + seq_len_q)
    o_seq = output.slice(axis=0, start=seq_start_index, stop=seq_start_index + seq_len_q)

    q_tile = ct.load(
        q_seq,
        index=(seq_block_id, head_id, 0),
        shape=(BLOCK_M, 1, BLOCK_D),
        order=(0, 1, 2),
        allow_tma=True,
        latency=2,
        padding_mode=PAD_ZERO,
    )
    q = ct.reshape(q_tile, (BLOCK_M, BLOCK_D))

    q_pe = None
    if BLOCK_R > 0:
        q_pe_tile = ct.load(
            q_seq,
            index=(seq_block_id, head_id, BLOCK_D // BLOCK_R),
            shape=(BLOCK_M, 1, BLOCK_R),
            order=(0, 1, 2),
            allow_tma=True,
            latency=2,
            padding_mode=PAD_ZERO,
        )
        q_pe = ct.reshape(q_pe_tile, (BLOCK_M, BLOCK_R))

    # Initialize accumulators
    m_i = ct.full((BLOCK_M,), -math.inf, dtype=ct.float32)
    l_i = ct.full((BLOCK_M,), 1.0, dtype=ct.float32)
    acc = ct.full((BLOCK_M, BLOCK_D), 0.0, dtype=ct.float32)

    # Pre-allocate zero accumulator for QK (hoisted outside loop)
    qk_zeros = ct.full((BLOCK_M, BLOCK_N), 0.0, dtype=ct.float32)

    offs_n_base = ct.arange(BLOCK_N, dtype=ct.int32)
    offs_m = start_m + ct.arange(BLOCK_M, dtype=ct.int32)

    # Compute bounds for two-stage causal split
    if IS_CAUSAL:
        # Off-band: everything before the diagonal block (fully unmasked)
        off_band_hi = ct.minimum(seq_len_kv, start_m)
        # On-band: the diagonal block itself
        on_band_lo = start_m
        on_band_hi = ct.minimum(seq_len_kv, start_m + BLOCK_M)
    else:
        # Non-causal: process everything in off-band loop
        off_band_hi = seq_len_kv
        on_band_lo = 0
        on_band_hi = 0

    off_band_iters = (off_band_hi + BLOCK_N - 1) // BLOCK_N
    for iter_idx in range(off_band_iters):
        curr_n = iter_idx * BLOCK_N

        k = _load_page_wrapper_prefill(
            curr_n,
            key_cache,
            block_tables,
            page_table_offset,
            off_kv_h,
            PAGE_SIZE,
            BLOCK_N,
            BLOCK_D,
            LOAD_BLOCK_N,
            0,
            3,
        )

        qk = ct.mma(q, ct.transpose(k), acc=qk_zeros)

        if BLOCK_R > 0:
            k_pe = _load_page_wrapper_prefill(
                curr_n,
                key_cache,
                block_tables,
                page_table_offset,
                off_kv_h,
                PAGE_SIZE,
                BLOCK_N,
                BLOCK_R,
                LOAD_BLOCK_N,
                BLOCK_D // BLOCK_R,
                3,
            )
            qk = ct.mma(q_pe, ct.transpose(k_pe), acc=qk)

        # Online softmax with flush_to_zero for IR parity with Triton
        qk_max = ct.max(qk, axis=1, keepdims=False)
        m_ij = ct.maximum(m_i, (qk_max * qk_scale))
        p = ct.exp2(qk * qk_scale - ct.reshape(m_ij, (BLOCK_M, 1)), flush_to_zero=True)

        alpha = ct.exp2((m_i - m_ij), flush_to_zero=True)
        l_i = l_i * alpha + ct.sum(p, axis=1, keepdims=False)
        acc = acc * ct.reshape(alpha, (BLOCK_M, 1))

        v = _load_page_wrapper_prefill(
            curr_n,
            value_cache,
            block_tables,
            page_table_offset,
            off_kv_h,
            PAGE_SIZE,
            BLOCK_N,
            BLOCK_D,
            LOAD_BLOCK_N,
            0,
            4,
        )

        acc = ct.mma(ct.astype(p, q.dtype), v, acc=acc)
        m_i = m_ij

    if IS_CAUSAL:
        on_band_iters = (on_band_hi - on_band_lo + BLOCK_N - 1) // BLOCK_N
        for iter_idx in range(on_band_iters):
            curr_n = on_band_lo + iter_idx * BLOCK_N

            k = _load_page_wrapper_prefill(
                curr_n,
                key_cache,
                block_tables,
                page_table_offset,
                off_kv_h,
                PAGE_SIZE,
                BLOCK_N,
                BLOCK_D,
                LOAD_BLOCK_N,
                0,
                3,
            )

            qk = ct.mma(q, ct.transpose(k), acc=qk_zeros)

            if BLOCK_R > 0:
                k_pe = _load_page_wrapper_prefill(
                    curr_n,
                    key_cache,
                    block_tables,
                    page_table_offset,
                    off_kv_h,
                    PAGE_SIZE,
                    BLOCK_N,
                    BLOCK_R,
                    LOAD_BLOCK_N,
                    BLOCK_D // BLOCK_R,
                    3,
                )
                qk = ct.mma(q_pe, ct.transpose(k_pe), acc=qk)

            offs_n = curr_n + offs_n_base
            causal_mask = ct.reshape(offs_m, (BLOCK_M, 1)) >= ct.reshape(offs_n, (1, BLOCK_N))
            qk = ct.where(causal_mask, qk, ct.full((BLOCK_M, BLOCK_N), -1.0e6, dtype=ct.float32))

            qk_max = ct.max(qk, axis=1, keepdims=False)
            m_ij = ct.maximum(m_i, (qk_max * qk_scale))
            p = ct.exp2(qk * qk_scale - ct.reshape(m_ij, (BLOCK_M, 1)), flush_to_zero=True)

            alpha = ct.exp2((m_i - m_ij), flush_to_zero=True)
            l_i = l_i * alpha + ct.sum(p, axis=1, keepdims=False)
            acc = acc * ct.reshape(alpha, (BLOCK_M, 1))

            v = _load_page_wrapper_prefill(
                curr_n,
                value_cache,
                block_tables,
                page_table_offset,
                off_kv_h,
                PAGE_SIZE,
                BLOCK_N,
                BLOCK_D,
                LOAD_BLOCK_N,
                0,
                4,
            )

            acc = ct.mma(ct.astype(p, q.dtype), v, acc=acc)
            m_i = m_ij

    # Epilogue: normalize and store with RMd.APPROX
    l_i_rcp = ct.truediv(v_scale, l_i, flush_to_zero=True, rounding_mode=RMd.APPROX)
    acc = acc * ct.reshape(l_i_rcp, (BLOCK_M, 1))
    lse = m_i + ct.log2(l_i)

    # Store output using TMA
    acc_out = ct.astype(acc, output.dtype)
    acc_3d = ct.reshape(acc_out, (BLOCK_M, 1, BLOCK_D))
    ct.store(
        o_seq,
        index=(seq_block_id, head_id, 0),
        tile=acc_3d,
        order=(0, 1, 2),
        allow_tma=True,
        latency=2,
    )

    # Store LSE - lse_output is 2D [total_tokens, num_heads]
    lse_scaled = lse * (1.0 / INV_LOG_2)  # multiply by constant instead of dividing
    offs_m_store = ct.arange(BLOCK_M, dtype=ct.int32)
    token_indices = seq_start_index + start_m + offs_m_store
    head_indices = ct.full((BLOCK_M,), head_id, dtype=ct.int32)
    lse_mask = offs_m_store + start_m < seq_len_q
    token_indices_masked = ct.where(lse_mask, token_indices, ct.full((BLOCK_M,), -1, dtype=ct.int32))
    lse_indices = (token_indices_masked, head_indices)
    ct.scatter(lse_output, lse_indices, lse_scaled)


@ct.kernel
def _prefill_attention_paged_kernel(
    query,
    key_cache,
    value_cache,
    actual_seq_lens_q,
    actual_seq_lens_kv,
    batch_offsets,
    block_tables,
    output,
    lse_output,
    K_SCALE: ConstFloat,
    V_SCALE: ConstFloat,
    N_KV_HEADS: ConstInt,
    PAGE_SIZE: ConstInt,
    BLOCK_M: ConstInt,
    BLOCK_N: ConstInt,
    BLOCK_D: ConstInt,
    BLOCK_R: ConstInt,
    QUERY_GROUP_SIZE: ConstInt,
    stride_block_table,
    IS_CAUSAL: ConstBool,
    LOAD_BLOCK_N: ConstInt,
):
    batch_id = ct.bid(0)
    head_id = ct.bid(1)
    seq_block_id = ct.bid(2)

    _prefill_attention_paged_body(
        batch_id,
        head_id,
        seq_block_id,
        query,
        key_cache,
        value_cache,
        actual_seq_lens_q,
        actual_seq_lens_kv,
        batch_offsets,
        block_tables,
        output,
        lse_output,
        K_SCALE,
        V_SCALE,
        N_KV_HEADS,
        PAGE_SIZE,
        BLOCK_M,
        BLOCK_N,
        BLOCK_D,
        BLOCK_R,
        QUERY_GROUP_SIZE,
        stride_block_table,
        IS_CAUSAL,
        LOAD_BLOCK_N,
    )


@ct.kernel
def _prefill_attention_paged_lpt_kernel(
    query,
    key_cache,
    value_cache,
    actual_seq_lens_q,
    actual_seq_lens_kv,
    batch_offsets,
    block_tables,
    output,
    lse_output,
    K_SCALE: ConstFloat,
    V_SCALE: ConstFloat,
    N_KV_HEADS: ConstInt,
    PAGE_SIZE: ConstInt,
    BLOCK_M: ConstInt,
    BLOCK_N: ConstInt,
    BLOCK_D: ConstInt,
    BLOCK_R: ConstInt,
    QUERY_GROUP_SIZE: ConstInt,
    stride_block_table,
    IS_CAUSAL: ConstBool,
    LOAD_BLOCK_N: ConstInt,
    NUM_HEADS: ConstInt,
    NUM_BATCH: ConstInt,
    MAX_SEQ_LEN: ConstInt,
    SWIZZLE: ConstInt,
    NUM_HB_QUOTIENT: ConstInt,
    NUM_HB_REMAINDER: ConstInt,
):
    tile_idx = ct.bid(0)
    NUM_BLOCKS = (MAX_SEQ_LEN + BLOCK_M - 1) // BLOCK_M
    l2_major_blocks = SWIZZLE * NUM_BLOCKS
    bidhb = tile_idx // l2_major_blocks
    l2_mod = tile_idx % l2_major_blocks
    if bidhb < NUM_HB_QUOTIENT:
        block = l2_mod // SWIZZLE
        bidhb_residual = l2_mod % SWIZZLE
    else:
        block = l2_mod // NUM_HB_REMAINDER
        bidhb_residual = l2_mod % NUM_HB_REMAINDER
    bidhb_actual = bidhb * SWIZZLE + bidhb_residual
    batch_id = bidhb_actual // NUM_HEADS
    head_id = bidhb_actual % NUM_HEADS
    seq_block_id = NUM_BLOCKS - 1 - block

    if tile_idx >= NUM_BLOCKS * NUM_HEADS * NUM_BATCH or batch_id >= NUM_BATCH or head_id >= NUM_HEADS:
        return

    _prefill_attention_paged_body(
        batch_id,
        head_id,
        seq_block_id,
        query,
        key_cache,
        value_cache,
        actual_seq_lens_q,
        actual_seq_lens_kv,
        batch_offsets,
        block_tables,
        output,
        lse_output,
        K_SCALE,
        V_SCALE,
        N_KV_HEADS,
        PAGE_SIZE,
        BLOCK_M,
        BLOCK_N,
        BLOCK_D,
        BLOCK_R,
        QUERY_GROUP_SIZE,
        stride_block_table,
        IS_CAUSAL,
        LOAD_BLOCK_N,
    )


def _prefill_attention_ragged_body(
    batch_id,
    head_id,
    seq_block_id,
    query,
    key_cache,
    value_cache,
    actual_seq_lens_q,
    actual_seq_lens_kv,
    batch_offsets,
    output,
    lse_output,
    k_scale: ConstFloat,
    v_scale: ConstFloat,
    N_KV_HEADS: ConstInt,
    BLOCK_M: ConstInt,
    BLOCK_N: ConstInt,
    BLOCK_D: ConstInt,
    BLOCK_R: ConstInt,
    QUERY_GROUP_SIZE: ConstInt,
    IS_CAUSAL: ConstBool,
):
    # Load sequence info
    seq_start_idx_tile = ct.gather(batch_offsets, (batch_id,), padding_value=0)
    seq_start_index = seq_start_idx_tile.item()

    seq_len_q_tile = ct.gather(actual_seq_lens_q, (batch_id,), padding_value=0)
    seq_len_q = seq_len_q_tile.item()

    seq_len_kv_tile = ct.gather(actual_seq_lens_kv, (batch_id,), padding_value=0)
    seq_len_kv = seq_len_kv_tile.item()

    start_m = BLOCK_M * seq_block_id

    if start_m >= seq_len_q:
        return

    off_kv_h = head_id // QUERY_GROUP_SIZE
    qk_scale = k_scale * INV_LOG_2
    PAD_ZERO = ct.PaddingMode.ZERO

    # Create sliced views for ragged tensors - enables TMA with block indices
    # Slice along axis 0 to offset base pointer by seq_start_index
    q_seq = query.slice(axis=0, start=seq_start_index, stop=seq_start_index + seq_len_q)
    k_seq = key_cache.slice(axis=0, start=seq_start_index, stop=seq_start_index + seq_len_kv)
    v_seq = value_cache.slice(axis=0, start=seq_start_index, stop=seq_start_index + seq_len_kv)
    o_seq = output.slice(axis=0, start=seq_start_index, stop=seq_start_index + seq_len_q)

    # Load Q tile using TMA - use seq_block_id as block index
    # q_seq shape: [seq_len_q, num_heads, head_dim_qk + head_dim_rope]
    q_tile = ct.load(
        q_seq,
        index=(seq_block_id, head_id, 0),
        shape=(BLOCK_M, 1, BLOCK_D),
        order=(0, 1, 2),
        allow_tma=True,
        latency=2,
        padding_mode=PAD_ZERO,
    )
    q = ct.reshape(q_tile, (BLOCK_M, BLOCK_D))

    # Load Q_PE if needed
    q_pe = None
    if BLOCK_R > 0:
        q_pe_tile = ct.load(
            q_seq,
            index=(seq_block_id, head_id, BLOCK_D // BLOCK_R),
            shape=(BLOCK_M, 1, BLOCK_R),
            order=(0, 1, 2),
            allow_tma=True,
            latency=2,
            padding_mode=PAD_ZERO,
        )
        q_pe = ct.reshape(q_pe_tile, (BLOCK_M, BLOCK_R))

    # Initialize accumulators
    m_i = ct.full((BLOCK_M,), -math.inf, dtype=ct.float32)
    l_i = ct.full((BLOCK_M,), 1.0, dtype=ct.float32)
    acc = ct.full((BLOCK_M, BLOCK_D), 0.0, dtype=ct.float32)

    # Pre-allocate zero accumulator for QK (hoisted outside loop)
    qk_zeros = ct.full((BLOCK_M, BLOCK_N), 0.0, dtype=ct.float32)

    offs_n_base = ct.arange(BLOCK_N, dtype=ct.int32)
    offs_m = start_m + ct.arange(BLOCK_M, dtype=ct.int32)

    # Compute bounds for two-stage causal split
    if IS_CAUSAL:
        # Off-band: everything before the diagonal block (fully unmasked)
        off_band_hi = ct.minimum(seq_len_kv, start_m)
        # On-band: the diagonal block itself
        on_band_lo = start_m
        on_band_hi = ct.minimum(seq_len_kv, start_m + BLOCK_M)
    else:
        # Non-causal: process everything in off-band loop
        off_band_hi = seq_len_kv
        on_band_lo = 0
        on_band_hi = 0

    off_band_iters = (off_band_hi + BLOCK_N - 1) // BLOCK_N
    for iter_idx in range(off_band_iters):
        curr_n = iter_idx * BLOCK_N

        k_tile = ct.load(
            k_seq,
            index=(iter_idx, off_kv_h, 0),
            shape=(BLOCK_N, 1, BLOCK_D),
            order=(0, 1, 2),
            allow_tma=True,
            latency=2,
            padding_mode=PAD_ZERO,
        )
        k = ct.reshape(k_tile, (BLOCK_N, BLOCK_D))

        qk = ct.mma(q, ct.transpose(k), acc=qk_zeros)

        if BLOCK_R > 0:
            k_pe_tile = ct.load(
                k_seq,
                index=(iter_idx, off_kv_h, BLOCK_D // BLOCK_R),
                shape=(BLOCK_N, 1, BLOCK_R),
                order=(0, 1, 2),
                allow_tma=True,
                latency=2,
                padding_mode=PAD_ZERO,
            )
            k_pe = ct.reshape(k_pe_tile, (BLOCK_N, BLOCK_R))
            qk = ct.mma(q_pe, ct.transpose(k_pe), acc=qk)

        qk_max = ct.max(qk, axis=1, keepdims=False)
        m_ij = ct.maximum(m_i, (qk_max * qk_scale))
        p = ct.exp2(qk * qk_scale - ct.reshape(m_ij, (BLOCK_M, 1)), flush_to_zero=True)

        alpha = ct.exp2((m_i - m_ij), flush_to_zero=True)
        l_i = l_i * alpha + ct.sum(p, axis=1, keepdims=False)
        acc = acc * ct.reshape(alpha, (BLOCK_M, 1))

        v_tile = ct.load(
            v_seq,
            index=(iter_idx, off_kv_h, 0),
            shape=(BLOCK_N, 1, BLOCK_D),
            order=(0, 1, 2),
            allow_tma=True,
            latency=2,
            padding_mode=PAD_ZERO,
        )
        v = ct.reshape(v_tile, (BLOCK_N, BLOCK_D))

        acc = ct.mma(ct.astype(p, q.dtype), v, acc=acc)
        m_i = m_ij

    if IS_CAUSAL:
        on_band_iters = (on_band_hi - on_band_lo + BLOCK_N - 1) // BLOCK_N
        on_band_block_start = on_band_lo // BLOCK_N
        for iter_idx in range(on_band_iters):
            curr_n = on_band_lo + iter_idx * BLOCK_N
            block_idx = on_band_block_start + iter_idx

            k_tile = ct.load(
                k_seq,
                index=(block_idx, off_kv_h, 0),
                shape=(BLOCK_N, 1, BLOCK_D),
                order=(0, 1, 2),
                allow_tma=True,
                latency=2,
                padding_mode=PAD_ZERO,
            )
            k = ct.reshape(k_tile, (BLOCK_N, BLOCK_D))

            qk = ct.mma(q, ct.transpose(k), acc=qk_zeros)

            if BLOCK_R > 0:
                k_pe_tile = ct.load(
                    k_seq,
                    index=(block_idx, off_kv_h, BLOCK_D // BLOCK_R),
                    shape=(BLOCK_N, 1, BLOCK_R),
                    order=(0, 1, 2),
                    allow_tma=True,
                    latency=2,
                    padding_mode=PAD_ZERO,
                )
                k_pe = ct.reshape(k_pe_tile, (BLOCK_N, BLOCK_R))
                qk = ct.mma(q_pe, ct.transpose(k_pe), acc=qk)

            offs_n = curr_n + offs_n_base
            causal_mask = ct.reshape(offs_m, (BLOCK_M, 1)) >= ct.reshape(offs_n, (1, BLOCK_N))
            qk = ct.where(causal_mask, qk, ct.full((BLOCK_M, BLOCK_N), -1.0e6, dtype=ct.float32))

            qk_max = ct.max(qk, axis=1, keepdims=False)
            m_ij = ct.maximum(m_i, (qk_max * qk_scale))
            p = ct.exp2(qk * qk_scale - ct.reshape(m_ij, (BLOCK_M, 1)), flush_to_zero=True)

            alpha = ct.exp2((m_i - m_ij), flush_to_zero=True)
            l_i = l_i * alpha + ct.sum(p, axis=1, keepdims=False)
            acc = acc * ct.reshape(alpha, (BLOCK_M, 1))

            v_tile = ct.load(
                v_seq,
                index=(block_idx, off_kv_h, 0),
                shape=(BLOCK_N, 1, BLOCK_D),
                order=(0, 1, 2),
                allow_tma=True,
                latency=2,
                padding_mode=PAD_ZERO,
            )
            v = ct.reshape(v_tile, (BLOCK_N, BLOCK_D))

            acc = ct.mma(ct.astype(p, q.dtype), v, acc=acc)
            m_i = m_ij

    l_i_rcp = ct.truediv(v_scale, l_i, flush_to_zero=True, rounding_mode=RMd.APPROX)
    acc = acc * ct.reshape(l_i_rcp, (BLOCK_M, 1))
    lse = m_i + ct.log2(l_i)

    acc_out = ct.astype(acc, output.dtype)
    acc_3d = ct.reshape(acc_out, (BLOCK_M, 1, BLOCK_D))
    ct.store(
        o_seq,
        index=(seq_block_id, head_id, 0),
        tile=acc_3d,
        order=(0, 1, 2),
        allow_tma=True,
        latency=2,
    )

    lse_scaled = lse * (1.0 / INV_LOG_2)
    offs_m_store = ct.arange(BLOCK_M, dtype=ct.int32)
    token_indices = seq_start_index + start_m + offs_m_store
    head_indices = ct.full((BLOCK_M,), head_id, dtype=ct.int32)
    lse_mask = offs_m_store + start_m < seq_len_q
    token_indices_masked = ct.where(lse_mask, token_indices, ct.full((BLOCK_M,), -1, dtype=ct.int32))
    lse_indices = (token_indices_masked, head_indices)
    ct.scatter(lse_output, lse_indices, lse_scaled)


@ct.kernel
def _prefill_attention_ragged_kernel(
    query,
    key_cache,
    value_cache,
    actual_seq_lens_q,
    actual_seq_lens_kv,
    batch_offsets,
    output,
    lse_output,
    K_SCALE: ConstFloat,
    V_SCALE: ConstFloat,
    N_KV_HEADS: ConstInt,
    BLOCK_M: ConstInt,
    BLOCK_N: ConstInt,
    BLOCK_D: ConstInt,
    BLOCK_R: ConstInt,
    QUERY_GROUP_SIZE: ConstInt,
    IS_CAUSAL: ConstBool,
):
    """
    Prefill attention kernel with ragged (contiguous) KV cache.
    Optimized with two-stage causal loop split:
    - Stage 1 (off-band): Fully unmasked region before diagonal - no causal mask needed
    - Stage 2 (on-band): Diagonal block where causal mask matters
    """
    seq_block_id = ct.bid(0)
    batch_id = ct.bid(1)
    head_id = ct.bid(2)

    _prefill_attention_ragged_body(
        batch_id,
        head_id,
        seq_block_id,
        query,
        key_cache,
        value_cache,
        actual_seq_lens_q,
        actual_seq_lens_kv,
        batch_offsets,
        output,
        lse_output,
        K_SCALE,
        V_SCALE,
        N_KV_HEADS,
        BLOCK_M,
        BLOCK_N,
        BLOCK_D,
        BLOCK_R,
        QUERY_GROUP_SIZE,
        IS_CAUSAL,
    )


@ct.kernel
def _prefill_attention_ragged_lpt_kernel(
    query,
    key_cache,
    value_cache,
    actual_seq_lens_q,
    actual_seq_lens_kv,
    batch_offsets,
    output,
    lse_output,
    K_SCALE: ConstFloat,
    V_SCALE: ConstFloat,
    N_KV_HEADS: ConstInt,
    BLOCK_M: ConstInt,
    BLOCK_N: ConstInt,
    BLOCK_D: ConstInt,
    BLOCK_R: ConstInt,
    QUERY_GROUP_SIZE: ConstInt,
    IS_CAUSAL: ConstBool,
    NUM_HEADS: ConstInt,
    NUM_BATCH: ConstInt,
    MAX_SEQ_LEN: ConstInt,
    SWIZZLE: ConstInt,
    NUM_HB_QUOTIENT: ConstInt,
    NUM_HB_REMAINDER: ConstInt,
):
    tile_idx = ct.bid(0)
    NUM_BLOCKS = (MAX_SEQ_LEN + BLOCK_M - 1) // BLOCK_M
    l2_major_blocks = SWIZZLE * NUM_BLOCKS
    bidhb = tile_idx // l2_major_blocks
    l2_mod = tile_idx % l2_major_blocks
    if bidhb < NUM_HB_QUOTIENT:
        block = l2_mod // SWIZZLE
        bidhb_residual = l2_mod % SWIZZLE
    else:
        block = l2_mod // NUM_HB_REMAINDER
        bidhb_residual = l2_mod % NUM_HB_REMAINDER
    bidhb_actual = bidhb * SWIZZLE + bidhb_residual
    batch_id = bidhb_actual // NUM_HEADS
    head_id = bidhb_actual % NUM_HEADS
    seq_block_id = NUM_BLOCKS - 1 - block  # LPT: reverse order

    if tile_idx >= NUM_BLOCKS * NUM_HEADS * NUM_BATCH or batch_id >= NUM_BATCH or head_id >= NUM_HEADS:
        return

    _prefill_attention_ragged_body(
        batch_id,
        head_id,
        seq_block_id,
        query,
        key_cache,
        value_cache,
        actual_seq_lens_q,
        actual_seq_lens_kv,
        batch_offsets,
        output,
        lse_output,
        K_SCALE,
        V_SCALE,
        N_KV_HEADS,
        BLOCK_M,
        BLOCK_N,
        BLOCK_D,
        BLOCK_R,
        QUERY_GROUP_SIZE,
        IS_CAUSAL,
    )


def prefill_attention_kv_paged_cutile(
    q,
    k_cache,
    v_cache,
    actual_seq_lens_q,
    actual_seq_lens_kv,
    actual_seq_offset,
    block_tables,
    k_scale,
    v_scale,
    num_batch,
    max_seq_len,
    is_causal: bool = True,
    outputs: Optional[torch.Tensor] = None,
    out_lse: Optional[torch.Tensor] = None,
    use_lpt_scheduler: bool = True,
):
    """
    Prefill attention with paged KV cache (cuTile implementation).
    """
    # KV cache [num_pages, page_size, num_kv_heads, head_dim_qk]
    total_num_pages = k_cache.shape[0]
    page_size = k_cache.shape[1]
    num_kv_heads = k_cache.shape[2]
    num_qo_heads = q.shape[1]
    head_dim_qk = q.shape[-1]
    head_dim_vo = v_cache.shape[-1]

    BLOCK_R = head_dim_qk - head_dim_vo
    QUERY_GROUP_SIZE = num_qo_heads // num_kv_heads

    outputs = (
        torch.zeros(
            [q.shape[0], num_qo_heads, head_dim_vo],
            dtype=q.dtype,
            device=q.device,
        )
        if outputs is None
        else outputs
    )
    out_lse = (
        torch.zeros([q.shape[0], num_qo_heads], dtype=torch.float32, device=q.device) if out_lse is None else out_lse
    )

    # Flatten tensors for kernel
    actual_seq_lens_q_flat = actual_seq_lens_q.reshape(-1).contiguous()
    actual_seq_lens_kv_flat = actual_seq_lens_kv.reshape(-1).contiguous()
    batch_offsets_flat = actual_seq_offset.reshape(-1).contiguous()
    block_tables_flat = block_tables.reshape(-1).contiguous()
    stride_block_table = block_tables.shape[1] if block_tables.dim() > 1 else 1

    if use_lpt_scheduler:
        element_size = q.element_size()
        size_one_kv_head = max_seq_len * (head_dim_qk + head_dim_vo) * element_size
        size_l2 = 50 * 1024 * 1024  # 50 MB for K & V
        if size_l2 < size_one_kv_head:
            swizzle = 1
        else:
            log2_floor = (size_l2 // size_one_kv_head).bit_length() - 1
            swizzle = 1 << log2_floor
        num_hb_quotient = (num_qo_heads * num_batch) // swizzle
        num_hb_remainder = (num_qo_heads * num_batch) % swizzle

        paged_lpt_stream = torch.cuda.current_stream()
        paged_lpt_cache_key = (
            num_batch,
            num_qo_heads,
            num_kv_heads,
            total_num_pages,
            page_size,
            head_dim_qk,
            head_dim_vo,
            BLOCK_R,
            QUERY_GROUP_SIZE,
            max_seq_len,
            is_causal,
            swizzle,
            q.dtype,
            str(q.device),
        )
        if paged_lpt_cache_key not in _prefill_paged_lpt_tune_cache:
            result = exhaustive_search(
                list(_get_prefill_autotune_configs(page_size)),
                paged_lpt_stream,
                lambda cfg: ((max_seq_len + cfg.BLOCK_M - 1) // cfg.BLOCK_M * num_qo_heads * num_batch, 1, 1),
                _prefill_attention_paged_lpt_kernel,
                lambda cfg: (
                    q,
                    k_cache,
                    v_cache,
                    actual_seq_lens_q_flat,
                    actual_seq_lens_kv_flat,
                    batch_offsets_flat,
                    block_tables_flat,
                    outputs,
                    out_lse,
                    k_scale,
                    v_scale,
                    num_kv_heads,
                    page_size,
                    cfg.BLOCK_M,
                    cfg.BLOCK_N,
                    head_dim_vo,
                    BLOCK_R,
                    QUERY_GROUP_SIZE,
                    stride_block_table,
                    is_causal,
                    min(cfg.BLOCK_N, page_size),
                    num_qo_heads,
                    num_batch,
                    max_seq_len,
                    swizzle,
                    num_hb_quotient,
                    max(num_hb_remainder, 1),
                ),
                lambda cfg: {"occupancy": cfg.occupancy},
            )
            best_cfg = result.best.config
            _prefill_paged_lpt_tune_cache[paged_lpt_cache_key] = (
                best_cfg,
                _prefill_attention_paged_lpt_kernel.replace_hints(occupancy=best_cfg.occupancy),
            )
        best_cfg, tuned_kernel = _prefill_paged_lpt_tune_cache[paged_lpt_cache_key]
        ct.launch(
            paged_lpt_stream,
            ((max_seq_len + best_cfg.BLOCK_M - 1) // best_cfg.BLOCK_M * num_qo_heads * num_batch, 1, 1),
            tuned_kernel,
            (
                q,
                k_cache,
                v_cache,
                actual_seq_lens_q_flat,
                actual_seq_lens_kv_flat,
                batch_offsets_flat,
                block_tables_flat,
                outputs,
                out_lse,
                k_scale,
                v_scale,
                num_kv_heads,
                page_size,
                best_cfg.BLOCK_M,
                best_cfg.BLOCK_N,
                head_dim_vo,
                BLOCK_R,
                QUERY_GROUP_SIZE,
                stride_block_table,
                is_causal,
                min(best_cfg.BLOCK_N, page_size),
                num_qo_heads,
                num_batch,
                max_seq_len,
                swizzle,
                num_hb_quotient,
                max(num_hb_remainder, 1),
            ),
        )
    else:
        paged_stream = torch.cuda.current_stream()
        paged_cache_key = (
            num_batch,
            num_qo_heads,
            num_kv_heads,
            total_num_pages,
            page_size,
            head_dim_qk,
            head_dim_vo,
            BLOCK_R,
            QUERY_GROUP_SIZE,
            max_seq_len,
            is_causal,
            q.dtype,
            str(q.device),
        )
        if paged_cache_key not in _prefill_paged_tune_cache:
            result = exhaustive_search(
                list(_get_prefill_autotune_configs(page_size)),
                paged_stream,
                lambda cfg: (num_batch, num_qo_heads, (max_seq_len + cfg.BLOCK_M - 1) // cfg.BLOCK_M),
                _prefill_attention_paged_kernel,
                lambda cfg: (
                    q,
                    k_cache,
                    v_cache,
                    actual_seq_lens_q_flat,
                    actual_seq_lens_kv_flat,
                    batch_offsets_flat,
                    block_tables_flat,
                    outputs,
                    out_lse,
                    k_scale,
                    v_scale,
                    num_kv_heads,
                    page_size,
                    cfg.BLOCK_M,
                    cfg.BLOCK_N,
                    head_dim_vo,
                    BLOCK_R,
                    QUERY_GROUP_SIZE,
                    stride_block_table,
                    is_causal,
                    min(cfg.BLOCK_N, page_size),
                ),
                lambda cfg: {"occupancy": cfg.occupancy},
            )
            best_cfg = result.best.config
            _prefill_paged_tune_cache[paged_cache_key] = (
                best_cfg,
                _prefill_attention_paged_kernel.replace_hints(occupancy=best_cfg.occupancy),
            )
        best_cfg, tuned_kernel = _prefill_paged_tune_cache[paged_cache_key]
        ct.launch(
            paged_stream,
            (num_batch, num_qo_heads, (max_seq_len + best_cfg.BLOCK_M - 1) // best_cfg.BLOCK_M),
            tuned_kernel,
            (
                q,
                k_cache,
                v_cache,
                actual_seq_lens_q_flat,
                actual_seq_lens_kv_flat,
                batch_offsets_flat,
                block_tables_flat,
                outputs,
                out_lse,
                k_scale,
                v_scale,
                num_kv_heads,
                page_size,
                best_cfg.BLOCK_M,
                best_cfg.BLOCK_N,
                head_dim_vo,
                BLOCK_R,
                QUERY_GROUP_SIZE,
                stride_block_table,
                is_causal,
                min(best_cfg.BLOCK_N, page_size),
            ),
        )

    return outputs, out_lse


def prefill_attention_kv_ragged_cutile(
    q,
    k_cache,
    v_cache,
    actual_seq_lens_q,
    actual_seq_lens_kv,
    actual_seq_offset,
    block_tables,
    k_scale,
    v_scale,
    num_batch,
    max_seq_len,
    is_causal: bool = True,
    outputs: Optional[torch.Tensor] = None,
    out_lse: Optional[torch.Tensor] = None,
    use_lpt_scheduler: bool = True,
):
    """
    Prefill attention with ragged KV cache (cuTile implementation).
    """
    # KV cache [total_num_tokens, num_kv_heads, head_dim_qk]
    num_kv_heads = k_cache.shape[1]
    num_qo_heads = q.shape[1]
    head_dim_qk = q.shape[-1]
    head_dim_vo = v_cache.shape[-1]

    BLOCK_R = head_dim_qk - head_dim_vo
    QUERY_GROUP_SIZE = num_qo_heads // num_kv_heads

    outputs = (
        torch.zeros(
            [q.shape[0], num_qo_heads, head_dim_vo],
            device=q.device,
            dtype=q.dtype,
        )
        if outputs is None
        else outputs
    )
    out_lse = (
        torch.zeros([q.shape[0], num_qo_heads], dtype=torch.float32, device=q.device) if out_lse is None else out_lse
    )

    # Flatten tensors for kernel
    actual_seq_lens_q_flat = actual_seq_lens_q.reshape(-1).contiguous()
    actual_seq_lens_kv_flat = actual_seq_lens_kv.reshape(-1).contiguous()
    batch_offsets_flat = actual_seq_offset.reshape(-1).contiguous()

    # Autotune key matching Triton's key for consistent caching across problem sizes
    autotune_key = (
        QUERY_GROUP_SIZE,
        num_kv_heads,
        BLOCK_R,
        head_dim_vo,
        k_scale,
        v_scale,
        max_seq_len,
        num_batch,
        3 if is_causal else 1,  # STAGE
    )

    if use_lpt_scheduler:
        element_size = q.element_size()
        size_one_kv_head = max_seq_len * (head_dim_qk + head_dim_vo) * element_size
        size_l2 = 50 * 1024 * 1024  # 50 MB for K & V
        if size_l2 < size_one_kv_head:
            swizzle = 1
        else:
            log2_floor = (size_l2 // size_one_kv_head).bit_length() - 1
            swizzle = 1 << log2_floor
        num_hb_quotient = (num_qo_heads * num_batch) // swizzle
        num_hb_remainder = (num_qo_heads * num_batch) % swizzle

        ragged_lpt_stream = torch.cuda.current_stream()
        ragged_lpt_cache_key = (autotune_key, swizzle, str(q.device))
        if ragged_lpt_cache_key not in _prefill_ragged_lpt_tune_cache:
            result = exhaustive_search(
                list(_get_prefill_autotune_configs(None)),
                ragged_lpt_stream,
                lambda cfg: ((max_seq_len + cfg.BLOCK_M - 1) // cfg.BLOCK_M * num_qo_heads * num_batch, 1, 1),
                _prefill_attention_ragged_lpt_kernel,
                lambda cfg: (
                    q,
                    k_cache,
                    v_cache,
                    actual_seq_lens_q_flat,
                    actual_seq_lens_kv_flat,
                    batch_offsets_flat,
                    outputs,
                    out_lse,
                    k_scale,
                    v_scale,
                    num_kv_heads,
                    cfg.BLOCK_M,
                    cfg.BLOCK_N,
                    head_dim_vo,
                    BLOCK_R,
                    QUERY_GROUP_SIZE,
                    is_causal,
                    num_qo_heads,
                    num_batch,
                    max_seq_len,
                    swizzle,
                    num_hb_quotient,
                    max(num_hb_remainder, 1),
                ),
                lambda cfg: {"occupancy": cfg.occupancy},
            )
            best_cfg = result.best.config
            _prefill_ragged_lpt_tune_cache[ragged_lpt_cache_key] = (
                best_cfg,
                _prefill_attention_ragged_lpt_kernel.replace_hints(occupancy=best_cfg.occupancy),
            )
        best_cfg, tuned_kernel = _prefill_ragged_lpt_tune_cache[ragged_lpt_cache_key]
        ct.launch(
            ragged_lpt_stream,
            ((max_seq_len + best_cfg.BLOCK_M - 1) // best_cfg.BLOCK_M * num_qo_heads * num_batch, 1, 1),
            tuned_kernel,
            (
                q,
                k_cache,
                v_cache,
                actual_seq_lens_q_flat,
                actual_seq_lens_kv_flat,
                batch_offsets_flat,
                outputs,
                out_lse,
                k_scale,
                v_scale,
                num_kv_heads,
                best_cfg.BLOCK_M,
                best_cfg.BLOCK_N,
                head_dim_vo,
                BLOCK_R,
                QUERY_GROUP_SIZE,
                is_causal,
                num_qo_heads,
                num_batch,
                max_seq_len,
                swizzle,
                num_hb_quotient,
                max(num_hb_remainder, 1),
            ),
        )
    else:
        ragged_stream = torch.cuda.current_stream()
        ragged_cache_key = (autotune_key, str(q.device))
        if ragged_cache_key not in _prefill_ragged_tune_cache:
            result = exhaustive_search(
                list(_get_prefill_autotune_configs(None)),
                ragged_stream,
                lambda cfg: ((max_seq_len + cfg.BLOCK_M - 1) // cfg.BLOCK_M, num_batch, num_qo_heads),
                _prefill_attention_ragged_kernel,
                lambda cfg: (
                    q,
                    k_cache,
                    v_cache,
                    actual_seq_lens_q_flat,
                    actual_seq_lens_kv_flat,
                    batch_offsets_flat,
                    outputs,
                    out_lse,
                    k_scale,
                    v_scale,
                    num_kv_heads,
                    cfg.BLOCK_M,
                    cfg.BLOCK_N,
                    head_dim_vo,
                    BLOCK_R,
                    QUERY_GROUP_SIZE,
                    is_causal,
                ),
                lambda cfg: {"occupancy": cfg.occupancy},
            )
            best_cfg = result.best.config
            _prefill_ragged_tune_cache[ragged_cache_key] = (
                best_cfg,
                _prefill_attention_ragged_kernel.replace_hints(occupancy=best_cfg.occupancy),
            )
        best_cfg, tuned_kernel = _prefill_ragged_tune_cache[ragged_cache_key]
        ct.launch(
            ragged_stream,
            ((max_seq_len + best_cfg.BLOCK_M - 1) // best_cfg.BLOCK_M, num_batch, num_qo_heads),
            tuned_kernel,
            (
                q,
                k_cache,
                v_cache,
                actual_seq_lens_q_flat,
                actual_seq_lens_kv_flat,
                batch_offsets_flat,
                outputs,
                out_lse,
                k_scale,
                v_scale,
                num_kv_heads,
                best_cfg.BLOCK_M,
                best_cfg.BLOCK_N,
                head_dim_vo,
                BLOCK_R,
                QUERY_GROUP_SIZE,
                is_causal,
            ),
        )

    return outputs, out_lse
