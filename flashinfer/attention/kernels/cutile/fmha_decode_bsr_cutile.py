# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Paged / MLA decode attention (block-sparse-row layout) using the public
# cuda.tile API, with split-KV reduction and persistent scheduling.

import math
import os
from types import SimpleNamespace
from typing import Optional
from typing import TypeAlias

import cuda.tile as ct
import torch
from cuda.tile import RoundingMode as RMd
from cuda.tile.tune import exhaustive_search

# Set FLASHINFER_CUTILE_AUTOTUNE_DISABLED=1 to skip exhaustive search
_AUTOTUNE_DISABLED = os.getenv("FLASHINFER_CUTILE_AUTOTUNE_DISABLED", "0") != "0"


def _is_power_of_2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _next_power_of_2(n: int) -> int:
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()


# Module-level tune caches for paged decode and MLA decode
_decode_kv_paged_tune_cache: dict = {}
_decode_mla_paged_tune_cache: dict = {}

INV_LOG_2 = 1.0 / math.log(2)

# Max pages loaded per KV block. _load_page unrolls a balanced ct.cat tree of this
# many per-page TMA loads; capping it bounds kernel size / autotune compile time
# (e.g. page_size=1 + BLOCK_N=128 would otherwise build a 128-page tree).
_MAX_LOAD_PAGES = 8

ConstInt: TypeAlias = ct.Constant[int]
ConstBool: TypeAlias = ct.Constant[bool]
ConstFloat: TypeAlias = ct.Constant[float]


def _tileiras_supports_advanced_indexing() -> bool:
    """True iff the active cuTile toolchain can compile ``ct.load_advanced_indexing``.

    That op requires tileiras >= 13.3. ``_get_max_supported_bytecode_version`` returns
    >= 13.3 only when BOTH the cuda-tile Python frontend lists 13.3 in its supported
    versions AND the resolved tileiras binary accepts it — exactly the condition under
    which the gather-TMA loader below will compile. Any probe failure falls back to the
    per-page tree loader (works on 13.2), so decode never hard-fails on older toolchains.
    """
    try:
        import tempfile

        from cuda.tile._bytecode.version import BytecodeVersion
        from cuda.tile._compile import _get_max_supported_bytecode_version

        return (
            _get_max_supported_bytecode_version(tempfile.gettempdir())
            >= BytecodeVersion.V_13_3
        )
    except Exception:
        return False


# Computed once at import. When True, _load_page uses the single gather-TMA
# ct.load_advanced_indexing (handles arbitrary NUM_PAGES incl. page_size==1);
# otherwise it falls back to the per-page ct.cat tree loader.
_SUPPORTS_ADVANCED_INDEXING = _tileiras_supports_advanced_indexing()


@ct.kernel
def _splitk_reduce_kernel(
    attn_splitk_out,
    lse_splitk_out,
    attn_out,
    actual_seq_lens,
    NUM_HEADS: ConstInt,
    NUM_KV_LEN_PER_SPLIT: ConstInt,
    NUM_KV_SPLITS: ConstInt,
    NUM_KV_SPLITS_POW2: ConstInt,
    BLOCK_D: ConstInt,
):
    batch_id = ct.bid(0)
    head_id = ct.bid(1)
    dtype = attn_out.dtype

    seq_len_tile = ct.load(actual_seq_lens, (batch_id,), shape=(1,))
    seq_len = seq_len_tile.item()
    actual_num_splits = (seq_len + NUM_KV_LEN_PER_SPLIT - 1) // NUM_KV_LEN_PER_SPLIT
    actual_num_splits = ct.minimum(actual_num_splits, NUM_KV_SPLITS)

    lse_vals = ct.load(
        lse_splitk_out, (batch_id, head_id, 0), shape=(1, 1, NUM_KV_SPLITS_POW2)
    )
    lse_vals = ct.reshape(lse_vals, (NUM_KV_SPLITS_POW2,))

    split_indices = ct.arange(NUM_KV_SPLITS_POW2, dtype=ct.int32)
    valid_mask = split_indices < actual_num_splits
    lse_vals = ct.where(
        valid_mask, lse_vals, ct.full((NUM_KV_SPLITS_POW2,), -1e30, dtype=ct.float32)
    )

    lse_max = ct.max(lse_vals)
    weights = ct.exp2(lse_vals - lse_max)
    weights = ct.where(
        valid_mask, weights, ct.zeros((NUM_KV_SPLITS_POW2,), dtype=ct.float32)
    )
    weights_sum = ct.sum(weights)
    weights = weights / weights_sum

    # attn_splitk_out has NUM_KV_SPLITS (<= POW2) splits on axis 0; only lse is
    # padded to POW2 by the caller. Zero-fill the out-of-range split rows here so
    # a non-power-of-two NUM_KV_SPLITS cannot read past the buffer. Those rows
    # carry weight 0 (their lse is padded to -inf), so the zeros do not affect
    # the weighted reduction below.
    out_all = ct.load(
        attn_splitk_out,
        (0, batch_id, head_id, 0),
        shape=(NUM_KV_SPLITS_POW2, 1, 1, BLOCK_D),
        padding_mode=ct.PaddingMode.ZERO,
    )
    out_all = ct.reshape(out_all, (NUM_KV_SPLITS_POW2, BLOCK_D))
    out_all = ct.astype(out_all, ct.float32)

    weights_row = ct.reshape(weights, (1, NUM_KV_SPLITS_POW2))
    acc = ct.mma(weights_row, out_all, ct.zeros((1, BLOCK_D), dtype=ct.float32))
    acc = ct.reshape(acc, (BLOCK_D,))

    result = ct.astype(acc, dtype)
    ct.store(attn_out, (batch_id, head_id, 0), ct.reshape(result, (1, 1, BLOCK_D)))


def _splitk_reduce_with_seq_len(
    attn_splitk_out,
    lse_splitk_out,
    actual_seq_lens,
    num_kv_len_per_split,
    attn_out=None,
):
    NUM_KV_SPLITS, B, num_heads, head_dim = attn_splitk_out.shape

    if attn_out is None:
        attn_out = torch.empty(
            (B, num_heads, head_dim),
            device=attn_splitk_out.device,
            dtype=attn_splitk_out.dtype,
        )

    if NUM_KV_SPLITS == 1:
        attn_out.copy_(attn_splitk_out[0])
        return attn_out

    if NUM_KV_SPLITS == 2:
        lse_0, lse_1 = lse_splitk_out[:, :, 0], lse_splitk_out[:, :, 1]
        lse_max = torch.maximum(lse_0, lse_1)
        w0, w1 = torch.exp2(lse_0 - lse_max), torch.exp2(lse_1 - lse_max)
        w_sum = w0 + w1
        result = (
            attn_splitk_out[0].float() * w0.unsqueeze(-1)
            + attn_splitk_out[1].float() * w1.unsqueeze(-1)
        ) / w_sum.unsqueeze(-1)
        attn_out.copy_(result.to(attn_out.dtype))
        return attn_out

    NUM_KV_SPLITS_POW2 = _next_power_of_2(NUM_KV_SPLITS)
    BLOCK_D = _next_power_of_2(head_dim)

    if NUM_KV_SPLITS < NUM_KV_SPLITS_POW2:
        lse_padded = torch.full(
            (B, num_heads, NUM_KV_SPLITS_POW2),
            float("-inf"),
            device=lse_splitk_out.device,
            dtype=lse_splitk_out.dtype,
        )
        lse_padded[:, :, :NUM_KV_SPLITS] = lse_splitk_out
    else:
        lse_padded = lse_splitk_out

    ct.launch(
        torch.cuda.current_stream(),
        (B, num_heads, 1),
        _splitk_reduce_kernel,
        (
            attn_splitk_out,
            lse_padded,
            attn_out,
            actual_seq_lens,
            num_heads,
            num_kv_len_per_split,
            NUM_KV_SPLITS,
            NUM_KV_SPLITS_POW2,
            BLOCK_D,
        ),
    )
    return attn_out


def _load_page(
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
):
    """
    Load data from paged cache via TMA.

    For single page, issues one TMA load.
    For multiple pages, issues N independent TMA loads and concatenates via ct.cat.
    Each individual load still uses TMA, loading one page at a time.

    Args:
        cache: cache array [total_num_pages, PAGE_SIZE, N_KV_HEADS, BLOCK_D]
        block_tables: flattened page table array
        page_table_offset: offset into block_tables for current batch
        page: starting page index in the page table
        token: token offset within page
        off_kv_h: KV head index
        NUM_PAGES: number of pages to load (compile-time constant; any value)
        LOAD_BLOCK_N: tokens per page load (== PAGE_SIZE)
        BLOCK_D: feature dimension
        _PAGE_SIZE: tokens per page (unused; LOAD_BLOCK_N is used instead)

    Returns:
        Loaded tensor of shape [NUM_PAGES * LOAD_BLOCK_N, BLOCK_D]
    """
    # Multi-page load path. On tileiras >= 13.3 (see _SUPPORTS_ADVANCED_INDEXING)
    # use the single gather-TMA `ct.load_advanced_indexing`, which gathers all
    # NUM_PAGES pages in one op and handles arbitrary NUM_PAGES (incl. page_size==1).
    # On 13.2 fall back to _load_page_tree (a balanced ct.cat over per-page TMA
    # loads) so decode still compiles. The two paths are numerically equivalent;
    # advanced-indexing is used for generality (page_size==1). `_SUPPORTS_ADVANCED_INDEXING`
    # is a module-level Python bool, so exactly one branch is traced at compile time.
    if NUM_PAGES == 1:
        page_id = ct.gather(
            block_tables, (page_table_offset + page,), padding_value=0
        ).item()
        data = ct.reshape(
            ct.load(
                cache,
                index=(page_id, token // LOAD_BLOCK_N, off_kv_h, 0),
                shape=(1, LOAD_BLOCK_N, 1, BLOCK_D),
                order=(0, 1, 2, 3),
                allow_tma=True,
                latency=2,
            ),
            (LOAD_BLOCK_N, BLOCK_D),
        )
    elif _SUPPORTS_ADVANCED_INDEXING:
        p_idx = ct.arange(NUM_PAGES, dtype=ct.int32)
        page_ids = ct.gather(
            block_tables, (page_table_offset + page + p_idx,), padding_value=0
        )
        data = ct.reshape(
            ct.load_advanced_indexing(
                cache,
                (
                    page_ids,
                    ct.Slice(token, LOAD_BLOCK_N),
                    ct.Slice(off_kv_h, 1),
                    ct.Slice(0, BLOCK_D),
                ),
                padding_mode=ct.PaddingMode.ZERO,
            ),
            (NUM_PAGES * LOAD_BLOCK_N, BLOCK_D),
        )
    else:
        data = _load_page_tree(
            cache,
            block_tables,
            page_table_offset,
            page,
            token,
            off_kv_h,
            0,
            NUM_PAGES,
            LOAD_BLOCK_N,
            BLOCK_D,
        )
    return data


def _load_page_tree(
    cache,
    block_tables,
    page_table_offset,
    page,
    token,
    off_kv_h,
    lo,
    hi,
    LOAD_BLOCK_N,
    BLOCK_D,
):
    """Load pages [lo, hi) and return them concatenated along axis 0.

    Balanced-tree recursion over the compile-time page range so the result is a
    single nested ct.cat expression (never a reassigned variable). Page ``lo``
    starts at the intra-page token offset; later pages start at row 0.
    """
    if hi - lo == 1:
        page_id = ct.gather(
            block_tables, (page_table_offset + page + lo,), padding_value=0
        ).item()
        row = token // LOAD_BLOCK_N if lo == 0 else 0
        return ct.reshape(
            ct.load(
                cache,
                index=(page_id, row, off_kv_h, 0),
                shape=(1, LOAD_BLOCK_N, 1, BLOCK_D),
                order=(0, 1, 2, 3),
                allow_tma=True,
                latency=2,
            ),
            (LOAD_BLOCK_N, BLOCK_D),
        )
    mid = (lo + hi) // 2
    left = _load_page_tree(
        cache,
        block_tables,
        page_table_offset,
        page,
        token,
        off_kv_h,
        lo,
        mid,
        LOAD_BLOCK_N,
        BLOCK_D,
    )
    right = _load_page_tree(
        cache,
        block_tables,
        page_table_offset,
        page,
        token,
        off_kv_h,
        mid,
        hi,
        LOAD_BLOCK_N,
        BLOCK_D,
    )
    return ct.cat((left, right), 0)


def _load_page_wrapper(
    curr_n,
    cache,
    block_tables,
    page_table_offset,
    off_kv_h,
    PAGE_SIZE,
    BLOCK_N,
    BLOCK_D,
    LOAD_BLOCK_N,
):
    """
    Load cache data (K or V) for current position.

    Computes page index and token offset from curr_n, then delegates to _load_page.
    """
    NUM_PAGES = BLOCK_N // LOAD_BLOCK_N
    page = curr_n // PAGE_SIZE
    token = curr_n % PAGE_SIZE

    data = _load_page(
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
    )
    return data


@ct.kernel
def _decode_attention_kv_paged_kernel(
    q,
    k_cache,
    v_cache,
    actual_seq_lens,
    block_tables,
    output,
    lse_out,
    K_SCALE: ConstFloat,
    V_SCALE: ConstFloat,
    N_KV_HEADS: ConstInt,
    PAGE_SIZE: ConstInt,
    BLOCK_H: ConstInt,
    BLOCK_N: ConstInt,
    BLOCK_D: ConstInt,
    QUERY_GROUP_SIZE: ConstInt,
    NUM_KV_SPLITS: ConstInt,
    KV_LEN_PER_SPLIT: ConstInt,
    HAS_LSE_OUT: ConstBool,
    stride_block_table,
    NUM_HEAD_BLOCKS: ConstInt,
    TRANS_QK: ConstBool,
    LOAD_BLOCK_N: ConstInt,
    NUM_PAGES_PER_BLOCK: ConstInt,
):
    batch_id = ct.bid(0)
    head_block_id = ct.bid(1)
    kv_split_id = ct.bid(2)

    kv_head_id = head_block_id // NUM_HEAD_BLOCKS
    hb = head_block_id % NUM_HEAD_BLOCKS

    seq_len_tile = ct.gather(actual_seq_lens, (batch_id,), padding_value=0)
    seq_len = seq_len_tile.item()

    qk_scale = K_SCALE * INV_LOG_2
    page_table_offset = batch_id * stride_block_table

    if KV_LEN_PER_SPLIT > 0:
        start_n = KV_LEN_PER_SPLIT * kv_split_id
        end_n = min(start_n + KV_LEN_PER_SPLIT, seq_len)
    else:
        start_n = 0
        end_n = seq_len

    if start_n >= end_n:
        return

    num_iters = (end_n - start_n + BLOCK_N - 1) // BLOCK_N
    offs_n_base = ct.arange(BLOCK_N, dtype=ct.int32)
    tail_n = start_n + ((end_n - start_n) // BLOCK_N) * BLOCK_N

    head_offset = kv_head_id * QUERY_GROUP_SIZE + hb * BLOCK_H
    head_block_idx = head_offset // BLOCK_H

    q_tile = ct.load(
        q,
        index=(batch_id, head_block_idx, 0),
        shape=(1, BLOCK_H, BLOCK_D),
        order=(0, 1, 2),
        allow_tma=True,
        latency=2,
    )
    q_tile = ct.reshape(q_tile, (BLOCK_H, BLOCK_D))

    neg_inf_h = ct.full((BLOCK_H,), -math.inf, dtype=ct.float32)
    ones_h = ct.full((BLOCK_H,), 1.0, dtype=ct.float32)

    if TRANS_QK:
        m_i = neg_inf_h
        l_i = ct.full((BLOCK_N, BLOCK_H), 0.0, dtype=ct.float32)
        acc = ct.full((BLOCK_D, BLOCK_H), 0.0, dtype=ct.float32)
        qk_zeros = ct.full((BLOCK_N, BLOCK_H), 0.0, dtype=ct.float32)
        mask_fill = ct.full((BLOCK_N, BLOCK_H), -1.0e6, dtype=ct.float32)

        for iter_idx in range(num_iters):
            curr_n = start_n + iter_idx * BLOCK_N

            if NUM_PAGES_PER_BLOCK == 1:
                # Single-page path: share page_id between K and V loads
                page = curr_n // PAGE_SIZE
                token = curr_n % PAGE_SIZE
                page_id = ct.gather(
                    block_tables, (page_table_offset + page,), padding_value=0
                ).item()
                token_idx = token // LOAD_BLOCK_N

                k = ct.reshape(
                    ct.load(
                        k_cache,
                        index=(page_id, token_idx, kv_head_id, 0),
                        shape=(1, LOAD_BLOCK_N, 1, BLOCK_D),
                        order=(0, 1, 2, 3),
                        allow_tma=True,
                        latency=2,
                    ),
                    (LOAD_BLOCK_N, BLOCK_D),
                )
            else:
                k = _load_page_wrapper(
                    curr_n,
                    k_cache,
                    block_tables,
                    page_table_offset,
                    kv_head_id,
                    PAGE_SIZE,
                    BLOCK_N,
                    BLOCK_D,
                    LOAD_BLOCK_N,
                )

            qk = ct.mma(k, ct.transpose(q_tile), acc=qk_zeros)

            if curr_n >= tail_n:
                offs_n = curr_n + offs_n_base
                mask = ct.reshape((offs_n < end_n), (BLOCK_N, 1))
                qk = ct.where(mask, qk, mask_fill)

            qk_max = ct.max(qk, axis=0, keepdims=False)
            m_ij = ct.maximum(m_i, (qk_max * qk_scale))
            p = ct.exp2(
                qk * qk_scale - ct.reshape(m_ij, (1, BLOCK_H)), flush_to_zero=True
            )

            alpha = ct.exp2((m_i - m_ij), flush_to_zero=True)
            alpha_2d = ct.reshape(alpha, (1, BLOCK_H))
            l_i = l_i * alpha_2d + p
            acc = acc * alpha_2d

            if NUM_PAGES_PER_BLOCK == 1:
                # Reuse page_id from K load
                v = ct.reshape(
                    ct.load(
                        v_cache,
                        index=(page_id, token_idx, kv_head_id, 0),
                        shape=(1, LOAD_BLOCK_N, 1, BLOCK_D),
                        order=(0, 1, 2, 3),
                        allow_tma=True,
                        latency=2,
                    ),
                    (LOAD_BLOCK_N, BLOCK_D),
                )
            else:
                v = _load_page_wrapper(
                    curr_n,
                    v_cache,
                    block_tables,
                    page_table_offset,
                    kv_head_id,
                    PAGE_SIZE,
                    BLOCK_N,
                    BLOCK_D,
                    LOAD_BLOCK_N,
                )

            acc = ct.mma(ct.transpose(v), ct.astype(p, q.dtype), acc=acc)
            m_i = m_ij

        l_i_sum = ct.sum(l_i, axis=0, keepdims=False)
        l_i_expanded = ct.reshape(l_i_sum, (1, BLOCK_H))
        acc = ct.truediv(
            (acc * V_SCALE), l_i_expanded, flush_to_zero=True, rounding_mode=RMd.APPROX
        )
        acc_out = ct.astype(ct.transpose(acc), output.dtype)
    else:
        m_i = neg_inf_h
        l_i = ones_h
        acc = ct.full((BLOCK_H, BLOCK_D), 0.0, dtype=ct.float32)
        qk_zeros = ct.full((BLOCK_H, BLOCK_N), 0.0, dtype=ct.float32)
        mask_fill = ct.full((BLOCK_H, BLOCK_N), -1.0e6, dtype=ct.float32)

        for iter_idx in range(num_iters):
            curr_n = start_n + iter_idx * BLOCK_N

            if NUM_PAGES_PER_BLOCK == 1:
                # Single-page path: share page_id between K and V loads
                page = curr_n // PAGE_SIZE
                token = curr_n % PAGE_SIZE
                page_id = ct.gather(
                    block_tables, (page_table_offset + page,), padding_value=0
                ).item()
                token_idx = token // LOAD_BLOCK_N

                k = ct.reshape(
                    ct.load(
                        k_cache,
                        index=(page_id, token_idx, kv_head_id, 0),
                        shape=(1, LOAD_BLOCK_N, 1, BLOCK_D),
                        order=(0, 1, 2, 3),
                        allow_tma=True,
                        latency=2,
                    ),
                    (LOAD_BLOCK_N, BLOCK_D),
                )
            else:
                k = _load_page_wrapper(
                    curr_n,
                    k_cache,
                    block_tables,
                    page_table_offset,
                    kv_head_id,
                    PAGE_SIZE,
                    BLOCK_N,
                    BLOCK_D,
                    LOAD_BLOCK_N,
                )

            qk = ct.mma(q_tile, ct.transpose(k), acc=qk_zeros)

            if curr_n >= tail_n:
                offs_n = curr_n + offs_n_base
                mask = ct.reshape((offs_n < end_n), (1, BLOCK_N))
                qk = ct.where(mask, qk, mask_fill)

            qk_max = ct.max(qk, axis=1, keepdims=False)
            m_ij = ct.maximum(m_i, (qk_max * qk_scale))
            p = ct.exp2(
                qk * qk_scale - ct.reshape(m_ij, (BLOCK_H, 1)), flush_to_zero=True
            )

            alpha = ct.exp2((m_i - m_ij), flush_to_zero=True)
            l_i = l_i * alpha + ct.sum(p, axis=1, keepdims=False)

            if NUM_PAGES_PER_BLOCK == 1:
                # Reuse page_id from K load
                v = ct.reshape(
                    ct.load(
                        v_cache,
                        index=(page_id, token_idx, kv_head_id, 0),
                        shape=(1, LOAD_BLOCK_N, 1, BLOCK_D),
                        order=(0, 1, 2, 3),
                        allow_tma=True,
                        latency=2,
                    ),
                    (LOAD_BLOCK_N, BLOCK_D),
                )
            else:
                v = _load_page_wrapper(
                    curr_n,
                    v_cache,
                    block_tables,
                    page_table_offset,
                    kv_head_id,
                    PAGE_SIZE,
                    BLOCK_N,
                    BLOCK_D,
                    LOAD_BLOCK_N,
                )

            acc = acc * ct.reshape(alpha, (BLOCK_H, 1))
            acc = ct.mma(ct.astype(p, q.dtype), v, acc=acc)
            m_i = m_ij

        l_i_expanded = ct.reshape(l_i, (BLOCK_H, 1))
        acc = ct.truediv(
            (acc * V_SCALE), l_i_expanded, flush_to_zero=True, rounding_mode=RMd.APPROX
        )
        acc_out = ct.astype(acc, output.dtype)

    acc_4d = ct.reshape(acc_out, (1, 1, BLOCK_H, BLOCK_D))
    ct.store(
        output,
        index=(kv_split_id, batch_id, head_block_idx, 0),
        tile=acc_4d,
        order=(0, 1, 2, 3),
        allow_tma=True,
        latency=2,
    )

    if HAS_LSE_OUT:
        lse = m_i + ct.log2(
            l_i if not TRANS_QK else ct.sum(l_i, axis=0, keepdims=False)
        )
        offs_h = ct.arange(BLOCK_H, dtype=ct.int32)
        lse_indices = (batch_id, head_offset + offs_h, kv_split_id)
        ct.scatter(lse_out, lse_indices, lse)


def _load_page_mla_tree(
    cache, block_tables, page_table_offset, page, token, lo, hi, LOAD_BLOCK_N, BLOCK_DIM
):
    """Load MLA pages [lo, hi) and return them concatenated along axis 0.

    Balanced-tree recursion over the compile-time page range (mirrors
    ``_load_page_tree`` for the 4D KV cache) so the result is a single nested
    ct.cat expression -- never a reassigned variable, which the cuTile tracer
    rejects. Page ``lo`` starts at the intra-page token offset; later pages
    start at row 0.
    """
    if hi - lo == 1:
        page_id = ct.gather(
            block_tables, (page_table_offset + page + lo,), padding_value=0
        ).item()
        row = token // LOAD_BLOCK_N if lo == 0 else 0
        return ct.reshape(
            ct.load(
                cache,
                index=(page_id, row, 0),
                shape=(1, LOAD_BLOCK_N, BLOCK_DIM),
                order=(0, 1, 2),
                allow_tma=True,
                latency=2,
            ),
            (LOAD_BLOCK_N, BLOCK_DIM),
        )
    mid = (lo + hi) // 2
    left = _load_page_mla_tree(
        cache,
        block_tables,
        page_table_offset,
        page,
        token,
        lo,
        mid,
        LOAD_BLOCK_N,
        BLOCK_DIM,
    )
    right = _load_page_mla_tree(
        cache,
        block_tables,
        page_table_offset,
        page,
        token,
        mid,
        hi,
        LOAD_BLOCK_N,
        BLOCK_DIM,
    )
    return ct.cat((left, right), 0)


def _load_page_mla(
    cache,
    block_tables,
    page_table_offset,
    page,
    token,
    NUM_PAGES,
    LOAD_BLOCK_N,
    BLOCK_DIM,
    _PAGE_SIZE,
):
    """
    Load data from paged MLA cache (3D: [total_num_pages, PAGE_SIZE, dim]) via TMA.

    Issues one TMA load per page and concatenates via a balanced ct.cat tree,
    supporting arbitrary NUM_PAGES. A small page_size yields NUM_PAGES up to 16
    (BLOCK_N // page_size), so the loader must not be limited to a fixed set of
    page counts -- the previous {1, 2, 4} branches fell through to an undefined
    ``data`` for NUM_PAGES in {8, 16} and failed during tracing.

    Args:
        cache: cache array [total_num_pages, PAGE_SIZE, dim]
        block_tables: flattened page table array
        page_table_offset: offset into block_tables for current batch
        page: starting page index in the page table
        token: token offset within page
        NUM_PAGES: number of pages to load (any positive compile-time integer)
        LOAD_BLOCK_N: tokens per page load (== PAGE_SIZE)
        BLOCK_DIM: feature dimension (BLOCK_D or BLOCK_R)
        _PAGE_SIZE: tokens per page (unused; LOAD_BLOCK_N is used instead)

    Returns:
        Loaded tensor of shape [NUM_PAGES * LOAD_BLOCK_N, BLOCK_DIM]
    """
    return _load_page_mla_tree(
        cache,
        block_tables,
        page_table_offset,
        page,
        token,
        0,
        NUM_PAGES,
        LOAD_BLOCK_N,
        BLOCK_DIM,
    )


def _load_page_mla_wrapper(
    curr_n,
    cache,
    block_tables,
    page_table_offset,
    PAGE_SIZE,
    BLOCK_N,
    BLOCK_DIM,
    LOAD_BLOCK_N,
):
    """
    Load MLA cache data (K, V, or K_rope) for current position.

    Computes page index and token offset from curr_n, then delegates to _load_page_mla.
    """
    NUM_PAGES = BLOCK_N // LOAD_BLOCK_N
    page = curr_n // PAGE_SIZE
    token = curr_n % PAGE_SIZE

    data = _load_page_mla(
        cache,
        block_tables,
        page_table_offset,
        page,
        token,
        NUM_PAGES,
        LOAD_BLOCK_N,
        BLOCK_DIM,
        PAGE_SIZE,
    )
    return data


@ct.kernel
def _decode_mla_kv_paged_kernel(
    q_nope,
    q_rope,
    k_cache,
    v_cache,
    k_rope,
    actual_seq_lens,
    block_tables,
    output,
    lse_out,
    K_SCALE: ConstFloat,
    V_SCALE: ConstFloat,
    PAGE_SIZE: ConstInt,
    BLOCK_H: ConstInt,
    BLOCK_N: ConstInt,
    BLOCK_D: ConstInt,
    BLOCK_R: ConstInt,
    QUERY_GROUP_SIZE: ConstInt,
    NUM_KV_SPLITS: ConstInt,
    KV_LEN_PER_SPLIT: ConstInt,
    HAS_LSE_OUT: ConstBool,
    stride_block_table,
    LOAD_BLOCK_N: ConstInt,
    NUM_PAGES_PER_BLOCK: ConstInt,
):
    batch_id = ct.bid(0)
    head_block_id = ct.bid(1)
    kv_split_id = ct.bid(2)

    seq_len_tile = ct.gather(actual_seq_lens, (batch_id,), padding_value=0)
    seq_len = seq_len_tile.item()

    qk_scale = K_SCALE * INV_LOG_2
    page_table_offset = batch_id * stride_block_table

    if KV_LEN_PER_SPLIT > 0:
        start_n = KV_LEN_PER_SPLIT * kv_split_id
        end_n = min(start_n + KV_LEN_PER_SPLIT, seq_len)
    else:
        start_n = 0
        end_n = seq_len

    if start_n >= end_n:
        return

    num_iters = (end_n - start_n + BLOCK_N - 1) // BLOCK_N
    offs_n_base = ct.arange(BLOCK_N, dtype=ct.int32)
    # tail_n = start position of last incomplete block (where masking is needed)
    tail_n = start_n + ((end_n - start_n) // BLOCK_N) * BLOCK_N

    head_block_idx = head_block_id
    head_offset = head_block_id * BLOCK_H

    q_nope_tile = ct.load(
        q_nope,
        index=(batch_id, head_block_idx, 0),
        shape=(1, BLOCK_H, BLOCK_D),
        order=(0, 1, 2),
        allow_tma=True,
        latency=2,
    )
    q_nope_tile = ct.reshape(q_nope_tile, (BLOCK_H, BLOCK_D))

    q_rope_tile = ct.load(
        q_rope,
        index=(batch_id, head_block_idx, 0),
        shape=(1, BLOCK_H, BLOCK_R),
        order=(0, 1, 2),
        allow_tma=True,
        latency=2,
    )
    q_rope_tile = ct.reshape(q_rope_tile, (BLOCK_H, BLOCK_R))

    m_i = ct.full((BLOCK_H,), -math.inf, dtype=ct.float32)
    l_i = ct.full((BLOCK_H,), 1.0, dtype=ct.float32)
    acc = ct.full((BLOCK_H, BLOCK_D), 0.0, dtype=ct.float32)

    for iter_idx in range(num_iters):
        curr_n = start_n + iter_idx * BLOCK_N

        if NUM_PAGES_PER_BLOCK == 1:
            # Single-page path: share page_id between K, K_rope, and V loads
            page_idx = curr_n // PAGE_SIZE
            token_block_idx = (curr_n % PAGE_SIZE) // BLOCK_N
            page_id = ct.gather(
                block_tables, (page_table_offset + page_idx,), padding_value=0
            ).item()

            k_tile = ct.reshape(
                ct.load(
                    k_cache,
                    index=(page_id, token_block_idx, 0),
                    shape=(1, BLOCK_N, BLOCK_D),
                    order=(0, 1, 2),
                    allow_tma=True,
                    latency=2,
                ),
                (BLOCK_N, BLOCK_D),
            )

            k_rope_tile = ct.reshape(
                ct.load(
                    k_rope,
                    index=(page_id, token_block_idx, 0),
                    shape=(1, BLOCK_N, BLOCK_R),
                    order=(0, 1, 2),
                    allow_tma=True,
                    latency=2,
                ),
                (BLOCK_N, BLOCK_R),
            )
        else:
            # Multi-page path: load BLOCK_N tokens across multiple pages
            k_tile = _load_page_mla_wrapper(
                curr_n,
                k_cache,
                block_tables,
                page_table_offset,
                PAGE_SIZE,
                BLOCK_N,
                BLOCK_D,
                LOAD_BLOCK_N,
            )
            k_rope_tile = _load_page_mla_wrapper(
                curr_n,
                k_rope,
                block_tables,
                page_table_offset,
                PAGE_SIZE,
                BLOCK_N,
                BLOCK_R,
                LOAD_BLOCK_N,
            )

        qk = ct.mma(
            q_nope_tile,
            ct.transpose(k_tile),
            acc=ct.full((BLOCK_H, BLOCK_N), 0.0, dtype=ct.float32),
        )
        if BLOCK_R > 0:
            qk = ct.mma(q_rope_tile, ct.transpose(k_rope_tile), acc=qk)

        if curr_n >= tail_n:
            offs_n = curr_n + offs_n_base
            mask = ct.reshape((offs_n < end_n), (1, BLOCK_N))
            qk = ct.where(
                mask, qk, ct.full((BLOCK_H, BLOCK_N), -1.0e6, dtype=ct.float32)
            )

        qk_max = ct.max(qk, axis=1, keepdims=False)
        m_ij = ct.maximum(m_i, (qk_max * qk_scale))
        p = ct.exp2(qk * qk_scale - ct.reshape(m_ij, (BLOCK_H, 1)), flush_to_zero=True)

        alpha = ct.exp2((m_i - m_ij), flush_to_zero=True)
        l_i = l_i * alpha + ct.sum(p, axis=1, keepdims=False)

        if NUM_PAGES_PER_BLOCK == 1:
            # Reuse page_id from K load
            v_tile = ct.reshape(
                ct.load(
                    v_cache,
                    index=(page_id, token_block_idx, 0),
                    shape=(1, BLOCK_N, BLOCK_D),
                    order=(0, 1, 2),
                    allow_tma=True,
                    latency=2,
                ),
                (BLOCK_N, BLOCK_D),
            )
        else:
            v_tile = _load_page_mla_wrapper(
                curr_n,
                v_cache,
                block_tables,
                page_table_offset,
                PAGE_SIZE,
                BLOCK_N,
                BLOCK_D,
                LOAD_BLOCK_N,
            )

        acc = acc * ct.reshape(alpha, (BLOCK_H, 1))
        acc = ct.mma(ct.astype(p, q_nope.dtype), v_tile, acc=acc)
        m_i = m_ij

    l_i_expanded = ct.reshape(l_i, (BLOCK_H, 1))
    acc = ct.truediv(
        (acc * V_SCALE), l_i_expanded, flush_to_zero=True, rounding_mode=RMd.APPROX
    )
    acc_out = ct.astype(acc, output.dtype)

    acc_4d = ct.reshape(acc_out, (1, 1, BLOCK_H, BLOCK_D))
    ct.store(
        output,
        index=(kv_split_id, batch_id, head_block_idx, 0),
        tile=acc_4d,
        order=(0, 1, 2, 3),
        allow_tma=True,
        latency=2,
    )

    if HAS_LSE_OUT:
        lse = m_i + ct.log2(l_i)
        offs_h = ct.arange(BLOCK_H, dtype=ct.int32)
        lse_indices = (batch_id, head_offset + offs_h, kv_split_id)
        ct.scatter(lse_out, lse_indices, lse)


_gqa_decode_autotune_cache = {}


def _get_gqa_decode_autotune_configs(query_group_size: int, page_size: int = 128):
    """Get autotune configurations for GQA decode kernel."""
    cache_key = (query_group_size, page_size)
    if cache_key not in _gqa_decode_autotune_cache:
        configs = []
        # Include small BLOCK_H (2, 4) so common GQA ratios (e.g. 32 qo / 8 kv
        # heads => query_group_size=4, as in Llama-3) get a real autotune space
        # instead of collapsing to the single BLOCK_N=page_size fallback below.
        for BLOCK_H in [2, 4, 8, 16, 32, 64]:
            if query_group_size >= BLOCK_H and query_group_size % BLOCK_H == 0:
                # Allow BLOCK_N values that are multiples of page_size for multi-page loading
                for BLOCK_N in [32, 64, 128]:
                    if page_size > BLOCK_N:
                        continue
                    # BLOCK_N must be a multiple of page_size (or equal)
                    if page_size < BLOCK_N and BLOCK_N % page_size != 0:
                        continue
                    # Cap the per-block page count: _load_page unrolls a balanced
                    # ct.cat tree of NUM_PAGES per-page TMA loads, so a large
                    # NUM_PAGES (e.g. page_size=1 + BLOCK_N=128 -> 128 pages)
                    # compiles a huge kernel and blows up autotune time. 8 keeps
                    # the useful multi-page configs (page_size>=16 fits fully;
                    # page_size=1 falls back to the single-page config below).
                    if BLOCK_N // page_size > _MAX_LOAD_PAGES:
                        continue
                    for occupancy in [1, 2]:
                        configs.append(
                            SimpleNamespace(
                                BLOCK_H=BLOCK_H, BLOCK_N=BLOCK_N, occupancy=occupancy
                            )
                        )
        if not configs:
            # Reached only for odd query_group_size (any even ratio >= 2 gets
            # BLOCK_H=2 from the loop above). BLOCK_H must DIVIDE query_group_size
            # or the head-block grid over/under-covers query heads -- so pick the
            # largest power-of-two divisor (n & -n), which always divides it and
            # is >= 1. The old _next_power_of_2(n)//2 rounded to a non-divisor
            # (e.g. ratio 3 -> 2, which splits 3 heads into a 2+2 grid).
            BLOCK_H = (
                (query_group_size & -query_group_size) if query_group_size > 0 else 1
            )
            configs.append(
                SimpleNamespace(
                    BLOCK_H=BLOCK_H, BLOCK_N=min(64, page_size), occupancy=1
                )
            )
        _gqa_decode_autotune_cache[cache_key] = configs
    return _gqa_decode_autotune_cache[cache_key]


def _gqa_decode_autotune_base(
    stream,
    q,
    k_cache,
    v_cache,
    actual_seq_lens_flat,
    block_tables_flat,
    Att_Out,
    LSE_Out_arg,
    num_batch,
    num_qo_heads,
    num_kv_heads,
    total_num_pages,
    k_scale,
    v_scale,
    page_size,
    head_dim_qk,
    QUERY_GROUP_SIZE,
    NUM_KV_SPLITS,
    kv_len_per_split,
    HAS_LSE_OUT,
    stride_block_table,
    TRANS_QK,
):
    configs = _get_gqa_decode_autotune_configs(QUERY_GROUP_SIZE, page_size)

    cache_key = (
        num_batch,
        num_qo_heads,
        num_kv_heads,
        page_size,
        head_dim_qk,
        QUERY_GROUP_SIZE,
        NUM_KV_SPLITS,
        kv_len_per_split,
        HAS_LSE_OUT,
        TRANS_QK,
        q.dtype,
        str(q.device),
    )
    if cache_key not in _decode_kv_paged_tune_cache:
        if _AUTOTUNE_DISABLED:
            # Respect FLASHINFER_CUTILE_AUTOTUNE_DISABLED: skip the exhaustive
            # search and take the first (deterministic) config. configs is never
            # empty -- _get_gqa_decode_autotune_configs always appends a fallback.
            best_cfg = configs[0]
        else:
            result = exhaustive_search(
                list(configs),
                stream,
                lambda cfg: (
                    num_batch,
                    num_kv_heads * max(QUERY_GROUP_SIZE // cfg.BLOCK_H, 1),
                    NUM_KV_SPLITS,
                ),
                _decode_attention_kv_paged_kernel,
                lambda cfg: (
                    q,
                    k_cache,
                    v_cache,
                    actual_seq_lens_flat,
                    block_tables_flat,
                    Att_Out,
                    LSE_Out_arg,
                    k_scale,
                    v_scale,
                    num_kv_heads,
                    page_size,
                    cfg.BLOCK_H,
                    cfg.BLOCK_N,
                    head_dim_qk,
                    QUERY_GROUP_SIZE,
                    NUM_KV_SPLITS,
                    kv_len_per_split,
                    HAS_LSE_OUT,
                    stride_block_table,
                    max(QUERY_GROUP_SIZE // cfg.BLOCK_H, 1),
                    TRANS_QK,
                    min(cfg.BLOCK_N, page_size),
                    max(cfg.BLOCK_N // page_size, 1),
                ),
                lambda cfg: {"occupancy": cfg.occupancy},
            )
            best_cfg = result.best.config
        _decode_kv_paged_tune_cache[cache_key] = (
            best_cfg,
            _decode_attention_kv_paged_kernel.replace_hints(
                occupancy=best_cfg.occupancy
            ),
        )
    best_cfg, tuned_kernel = _decode_kv_paged_tune_cache[cache_key]
    ct.launch(
        stream,
        (
            num_batch,
            num_kv_heads * max(QUERY_GROUP_SIZE // best_cfg.BLOCK_H, 1),
            NUM_KV_SPLITS,
        ),
        tuned_kernel,
        (
            q,
            k_cache,
            v_cache,
            actual_seq_lens_flat,
            block_tables_flat,
            Att_Out,
            LSE_Out_arg,
            k_scale,
            v_scale,
            num_kv_heads,
            page_size,
            best_cfg.BLOCK_H,
            best_cfg.BLOCK_N,
            head_dim_qk,
            QUERY_GROUP_SIZE,
            NUM_KV_SPLITS,
            kv_len_per_split,
            HAS_LSE_OUT,
            stride_block_table,
            max(QUERY_GROUP_SIZE // best_cfg.BLOCK_H, 1),
            TRANS_QK,
            min(best_cfg.BLOCK_N, page_size),
            max(best_cfg.BLOCK_N // page_size, 1),
        ),
    )
    return Att_Out


def fmha_decode_bsr_cutile(
    q,
    k_cache,
    v_cache,
    actual_seq_lens,
    block_tables,
    k_scale,
    v_scale,
    max_seq_len: int = -1,
    outputs: Optional[torch.Tensor] = None,
    force_split_kv: bool = False,
    force_persistent: bool = False,
):
    num_batch = q.shape[0]
    num_qo_heads = q.shape[1]
    head_dim_qk = q.shape[-1]

    total_num_pages = k_cache.shape[0]
    page_size = k_cache.shape[1]
    num_kv_heads = k_cache.shape[2]
    head_dim_vo = v_cache.shape[-1]

    if num_qo_heads % num_kv_heads != 0:
        # QUERY_GROUP_SIZE below is a floor division; a non-integral GQA ratio
        # would silently drop the trailing query heads.
        raise ValueError(
            "cuTile decode requires num_qo_heads % num_kv_heads == 0 "
            f"(num_qo_heads={num_qo_heads}, num_kv_heads={num_kv_heads})."
        )
    if head_dim_qk != head_dim_vo:
        # The kernel uses a single BLOCK_D (= head_dim_qk) for Q/K loads, the
        # attention accumulator, V loads, and the output store; a different
        # head_dim_vo would produce incorrect output.
        raise NotImplementedError(
            "cuTile decode requires head_dim_qk == head_dim_vo (got "
            f"{head_dim_qk} vs {head_dim_vo})."
        )

    QUERY_GROUP_SIZE = num_qo_heads // num_kv_heads
    TRANS_QK = QUERY_GROUP_SIZE < 64

    if not (_is_power_of_2(head_dim_qk) and _is_power_of_2(head_dim_vo)):
        raise NotImplementedError(
            f"cuTile decode attention requires power-of-2 dimensions. Got head_dim_qk={head_dim_qk}, head_dim_vo={head_dim_vo}."
        )

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    LSE_Out = None
    kv_len_per_split = -1
    NUM_KV_SPLITS = 1

    if max_seq_len < 0:
        max_pages_per_seq = (
            block_tables.shape[1] if block_tables.dim() > 1 else block_tables.shape[0]
        )
        max_seq_len = max_pages_per_seq * page_size

    persistent = num_batch * num_kv_heads > NUM_SMS
    if force_split_kv:
        should_use_split_kv = True
    else:
        # force_persistent (persistent scheduling) is mutually exclusive with
        # split-KV; honor it here so the caller can actually disable split-KV.
        should_use_split_kv = (
            not force_persistent and not persistent and max_seq_len > 2048
        )

    if should_use_split_kv:
        num_split_kv_estimated = max(NUM_SMS // num_batch, 1)
        kv_len_per_split = 1 << (
            (max_seq_len // num_split_kv_estimated - 1).bit_length()
        )
        kv_len_per_split = max(kv_len_per_split, 128)
        NUM_KV_SPLITS = (max_seq_len + kv_len_per_split - 1) // kv_len_per_split

        if num_batch <= 4:
            NUM_KV_SPLITS = min(NUM_KV_SPLITS, 4)
            kv_len_per_split = (max_seq_len + NUM_KV_SPLITS - 1) // NUM_KV_SPLITS
            # Align kv_len_per_split to next power of 2 (ensure alignment with BLOCK_N)
            kv_len_per_split = 1 << (kv_len_per_split - 1).bit_length()

        # Initialize to 0 and -inf so empty splits (where start_n >= end_n) contribute nothing
        Att_Out = torch.zeros(
            (NUM_KV_SPLITS, num_batch, num_qo_heads, head_dim_vo),
            device=q.device,
            dtype=q.dtype,
        )
        LSE_Out = torch.full(
            (num_batch, num_qo_heads, NUM_KV_SPLITS),
            float("-inf"),
            device=q.device,
            dtype=torch.float32,
        )
    else:
        outputs = (
            torch.zeros(
                (num_batch, num_qo_heads, head_dim_vo), device=q.device, dtype=q.dtype
            )
            if outputs is None
            else outputs
        )
        Att_Out = outputs.reshape(NUM_KV_SPLITS, num_batch, num_qo_heads, head_dim_vo)

    actual_seq_lens_flat = actual_seq_lens.reshape(-1).contiguous()
    block_tables_flat = block_tables.reshape(-1).contiguous()
    stride_block_table = block_tables.shape[1] if block_tables.dim() > 1 else 1

    LSE_Out_arg = (
        LSE_Out
        if LSE_Out is not None
        else torch.zeros(1, device=q.device, dtype=torch.float32)
    )
    HAS_LSE_OUT = LSE_Out is not None

    _gqa_decode_autotune_base(
        torch.cuda.current_stream(),
        q,
        k_cache,
        v_cache,
        actual_seq_lens_flat,
        block_tables_flat,
        Att_Out,
        LSE_Out_arg,
        num_batch,
        num_qo_heads,
        num_kv_heads,
        total_num_pages,
        k_scale,
        v_scale,
        page_size,
        head_dim_qk,
        QUERY_GROUP_SIZE,
        NUM_KV_SPLITS,
        kv_len_per_split,
        HAS_LSE_OUT,
        stride_block_table,
        TRANS_QK,
    )

    if should_use_split_kv:
        return _splitk_reduce_with_seq_len(
            Att_Out, LSE_Out, actual_seq_lens_flat, kv_len_per_split, outputs
        )
    return outputs


def _mla_decode_autotune_configs():
    for bh in [16, 32]:
        for bn in [16, 32, 64, 128]:
            for occupancy in [1, 2]:
                yield SimpleNamespace(BLOCK_H=bh, BLOCK_N=bn, occupancy=occupancy)


def _mla_decode_autotune_base(
    stream,
    q,
    q_rope,
    kv_cache,
    k_rope,
    actual_seq_lens_flat,
    block_tables_flat,
    Att_Out,
    LSE_Out_arg,
    num_batch,
    num_qo_heads,
    total_num_pages,
    k_scale,
    v_scale,
    page_size,
    head_dim_qk,
    head_dim_rope,
    QUERY_GROUP_SIZE,
    NUM_KV_SPLITS,
    kv_len_per_split,
    HAS_LSE_OUT,
    stride_block_table,
):
    mla_cache_key = (
        num_batch,
        num_qo_heads,
        page_size,
        head_dim_qk,
        head_dim_rope,
        QUERY_GROUP_SIZE,
        NUM_KV_SPLITS,
        kv_len_per_split,
        HAS_LSE_OUT,
        q.dtype,
        str(q.device),
    )
    if mla_cache_key not in _decode_mla_paged_tune_cache:
        result = exhaustive_search(
            list(_mla_decode_autotune_configs()),
            stream,
            lambda cfg: (
                num_batch,
                (num_qo_heads + cfg.BLOCK_H - 1) // cfg.BLOCK_H,
                NUM_KV_SPLITS,
            ),
            _decode_mla_kv_paged_kernel,
            lambda cfg: (
                q,
                q_rope,
                kv_cache,
                kv_cache,
                k_rope,
                actual_seq_lens_flat,
                block_tables_flat,
                Att_Out,
                LSE_Out_arg,
                k_scale,
                v_scale,
                page_size,
                cfg.BLOCK_H,
                min(cfg.BLOCK_N, page_size),
                head_dim_qk,
                head_dim_rope,
                QUERY_GROUP_SIZE,
                NUM_KV_SPLITS,
                kv_len_per_split,
                HAS_LSE_OUT,
                stride_block_table,
                min(cfg.BLOCK_N, page_size),
                max(cfg.BLOCK_N // page_size, 1),
            ),
            lambda cfg: {"occupancy": cfg.occupancy},
        )
        best_cfg = result.best.config
        _decode_mla_paged_tune_cache[mla_cache_key] = (
            best_cfg,
            _decode_mla_kv_paged_kernel.replace_hints(occupancy=best_cfg.occupancy),
        )
    best_cfg, tuned_kernel = _decode_mla_paged_tune_cache[mla_cache_key]
    ct.launch(
        stream,
        (
            num_batch,
            (num_qo_heads + best_cfg.BLOCK_H - 1) // best_cfg.BLOCK_H,
            NUM_KV_SPLITS,
        ),
        tuned_kernel,
        (
            q,
            q_rope,
            kv_cache,
            kv_cache,
            k_rope,
            actual_seq_lens_flat,
            block_tables_flat,
            Att_Out,
            LSE_Out_arg,
            k_scale,
            v_scale,
            page_size,
            best_cfg.BLOCK_H,
            min(best_cfg.BLOCK_N, page_size),
            head_dim_qk,
            head_dim_rope,
            QUERY_GROUP_SIZE,
            NUM_KV_SPLITS,
            kv_len_per_split,
            HAS_LSE_OUT,
            stride_block_table,
            min(best_cfg.BLOCK_N, page_size),
            max(best_cfg.BLOCK_N // page_size, 1),
        ),
    )
    return Att_Out


def decode_mla_kv_paged_cutile(
    q,
    q_rope,
    kv_cache,
    k_rope,
    actual_seq_lens,
    block_tables,
    k_scale,
    v_scale,
    max_seq_len: int = -1,
    outputs: Optional[torch.Tensor] = None,
    force_split_kv: bool = False,
    force_persistent: bool = False,
):
    num_qo_heads = q.shape[1]
    head_dim_qk = q.shape[-1]
    head_dim_rope = q_rope.shape[-1]
    num_batch = q.shape[0]
    total_num_pages = kv_cache.shape[0]
    page_size = kv_cache.shape[1]

    QUERY_GROUP_SIZE = num_qo_heads

    use_autotune = not _AUTOTUNE_DISABLED
    if use_autotune:
        NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
        num_head_blocks = max(QUERY_GROUP_SIZE // 32, 1)
        total_work = num_batch * num_head_blocks

        if max_seq_len < 0:
            max_pages_per_seq = (
                block_tables.shape[1]
                if block_tables.dim() > 1
                else block_tables.shape[0]
            )
            estimated_seq_len = max_pages_per_seq * page_size
        else:
            estimated_seq_len = max_seq_len

        # Split-KV heuristic: use split-KV when not persistent and seqlen > 256
        # This parallelizes across the sequence dimension for better SM utilization
        if force_split_kv or (
            not force_persistent and estimated_seq_len > 256 and total_work < NUM_SMS
        ):
            should_use_split_kv = True
            num_split_kv_estimated = max(NUM_SMS // num_batch, 1)
            kv_len_per_split = estimated_seq_len // num_split_kv_estimated
            kv_len_per_split = max(
                1 << (kv_len_per_split - 1).bit_length()
                if kv_len_per_split > 0
                else 128,
                128,
            )
            NUM_KV_SPLITS = (
                estimated_seq_len + kv_len_per_split - 1
            ) // kv_len_per_split
            max_seq_len = estimated_seq_len
        else:
            should_use_split_kv = False
            NUM_KV_SPLITS = 1
            kv_len_per_split = -1

        if should_use_split_kv:
            # Initialize to 0 and -inf so empty splits contribute nothing
            Att_Out = torch.zeros(
                (NUM_KV_SPLITS, num_batch, num_qo_heads, head_dim_qk),
                device=q.device,
                dtype=q.dtype,
            )
            LSE_Out = torch.full(
                (num_batch, num_qo_heads, NUM_KV_SPLITS),
                float("-inf"),
                device=q.device,
                dtype=torch.float32,
            )
        else:
            outputs = torch.zeros_like(q) if outputs is None else outputs
            Att_Out = outputs.reshape(1, num_batch, num_qo_heads, head_dim_qk)
            LSE_Out = torch.zeros(1, device=q.device, dtype=torch.float32)

        actual_seq_lens_flat = actual_seq_lens.reshape(-1).contiguous()
        block_tables_flat = block_tables.reshape(-1).contiguous()
        stride_block_table = block_tables.shape[1] if block_tables.dim() > 1 else 1

        HAS_LSE_OUT = should_use_split_kv
        LSE_Out_arg = (
            LSE_Out
            if should_use_split_kv
            else torch.zeros(1, device=q.device, dtype=torch.float32)
        )

        _mla_decode_autotune_base(
            torch.cuda.current_stream(),
            q,
            q_rope,
            kv_cache,
            k_rope,
            actual_seq_lens_flat,
            block_tables_flat,
            Att_Out,
            LSE_Out_arg,
            num_batch,
            num_qo_heads,
            total_num_pages,
            k_scale,
            v_scale,
            page_size,
            head_dim_qk,
            head_dim_rope,
            QUERY_GROUP_SIZE,
            NUM_KV_SPLITS,
            kv_len_per_split,
            HAS_LSE_OUT,
            stride_block_table,
        )

        if should_use_split_kv:
            return _splitk_reduce_with_seq_len(
                Att_Out, LSE_Out, actual_seq_lens_flat, kv_len_per_split, outputs
            )
        return outputs

    use_large_block_h = num_batch >= 16

    if use_large_block_h:
        for bh in [128, 64, 32, 16, 8]:
            if bh <= QUERY_GROUP_SIZE and QUERY_GROUP_SIZE % bh == 0:
                BLOCK_H = bh
                break
        else:
            BLOCK_H = max(QUERY_GROUP_SIZE, 1)
    else:
        for bh in [16, 8, 32]:
            if bh <= QUERY_GROUP_SIZE and QUERY_GROUP_SIZE % bh == 0:
                BLOCK_H = bh
                break
        else:
            BLOCK_H = max(QUERY_GROUP_SIZE, 1)

    while QUERY_GROUP_SIZE % BLOCK_H != 0 and BLOCK_H > 1:
        BLOCK_H = BLOCK_H // 2

    if not _is_power_of_2(BLOCK_H):
        # The divisor from the loop above may be non-power-of-2 (e.g.
        # QUERY_GROUP_SIZE=6 -> BLOCK_H=6). Rounding it (next_pow2//2) can break
        # divisibility (6 -> 4) and drop query heads. The largest power-of-two
        # divisor (n & -n) is both a power of two and divides QUERY_GROUP_SIZE.
        BLOCK_H = QUERY_GROUP_SIZE & -QUERY_GROUP_SIZE

    BLOCK_N = page_size if page_size >= 16 else 16
    LOAD_BLOCK_N = min(BLOCK_N, page_size)
    NUM_PAGES_PER_BLOCK = max(BLOCK_N // page_size, 1)
    num_head_blocks = max(QUERY_GROUP_SIZE // BLOCK_H, 1)

    if not (
        _is_power_of_2(head_dim_qk)
        and _is_power_of_2(head_dim_rope)
        and _is_power_of_2(BLOCK_H)
    ):
        raise NotImplementedError(
            f"cuTile MLA decode requires power-of-2 dimensions. Got head_dim_qk={head_dim_qk}, head_dim_rope={head_dim_rope}, BLOCK_H={BLOCK_H}."
        )

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    LSE_Out = None
    kv_len_per_split = -1
    NUM_KV_SPLITS = 1

    if max_seq_len < 0:
        max_pages_per_seq = (
            block_tables.shape[1] if block_tables.dim() > 1 else block_tables.shape[0]
        )
        estimated_seq_len = max_pages_per_seq * page_size
    else:
        estimated_seq_len = max_seq_len

    # force_split_kv must enter the split path unconditionally (matching the two
    # paged-decode sites above); the degenerate NUM_KV_SPLITS==1 case below still
    # resolves should_use_split_kv=False, so no >=1024 length gate is needed.
    if force_split_kv or (
        not force_persistent
        and estimated_seq_len > 256
        and num_batch * num_head_blocks < NUM_SMS
    ):
        num_split_kv_estimated = max(NUM_SMS // num_batch, 1)
        kv_len_per_split = estimated_seq_len // num_split_kv_estimated
        kv_len_per_split = max(_next_power_of_2(kv_len_per_split), 128)
        NUM_KV_SPLITS = (estimated_seq_len + kv_len_per_split - 1) // kv_len_per_split
        should_use_split_kv = NUM_KV_SPLITS > 1
        max_seq_len = estimated_seq_len
    else:
        should_use_split_kv = False
        kv_len_per_split = estimated_seq_len

    if should_use_split_kv:
        # Initialize to 0 and -inf so empty splits contribute nothing
        Att_Out = torch.zeros(
            (NUM_KV_SPLITS, num_batch, num_qo_heads, head_dim_qk),
            device=q.device,
            dtype=q.dtype,
        )
        LSE_Out = torch.full(
            (num_batch, num_qo_heads, NUM_KV_SPLITS),
            float("-inf"),
            device=q.device,
            dtype=torch.float32,
        )
        grid = (num_batch, num_head_blocks, NUM_KV_SPLITS)
    else:
        outputs = torch.zeros_like(q) if outputs is None else outputs
        Att_Out = outputs.reshape(NUM_KV_SPLITS, num_batch, num_qo_heads, head_dim_qk)
        grid = (num_batch, num_head_blocks, NUM_KV_SPLITS)

    actual_seq_lens_flat = actual_seq_lens.reshape(-1).contiguous()
    block_tables_flat = block_tables.reshape(-1).contiguous()
    stride_block_table = block_tables.shape[1] if block_tables.dim() > 1 else 1

    LSE_Out_arg = (
        LSE_Out
        if LSE_Out is not None
        else torch.zeros(1, device=q.device, dtype=torch.float32)
    )
    HAS_LSE_OUT = LSE_Out is not None

    num_ctas = 2 if (num_batch >= 16 and BLOCK_H >= 64) else None
    kernel = (
        _decode_mla_kv_paged_kernel.replace_hints(num_ctas=num_ctas)
        if num_ctas
        else _decode_mla_kv_paged_kernel
    )

    ct.launch(
        torch.cuda.current_stream(),
        grid,
        kernel,
        (
            q,
            q_rope,
            kv_cache,
            kv_cache,
            k_rope,
            actual_seq_lens_flat,
            block_tables_flat,
            Att_Out,
            LSE_Out_arg,
            k_scale,
            v_scale,
            page_size,
            BLOCK_H,
            BLOCK_N,
            head_dim_qk,
            head_dim_rope,
            QUERY_GROUP_SIZE,
            NUM_KV_SPLITS,
            kv_len_per_split,
            HAS_LSE_OUT,
            stride_block_table,
            LOAD_BLOCK_N,
            NUM_PAGES_PER_BLOCK,
        ),
    )

    if should_use_split_kv:
        return _splitk_reduce_with_seq_len(
            Att_Out, LSE_Out, actual_seq_lens_flat, kv_len_per_split, outputs
        )
    return outputs
