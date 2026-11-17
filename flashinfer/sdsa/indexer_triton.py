# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 by FlashInfer team.

"""The SDSA (Simplified DSA) indexer's Triton kernels: the BF16 and FP8 scorer off
SM10x, such as on Hopper, and the FP8 write of keys and queries."""

import triton  # type: ignore[import]
import triton.language as tl  # type: ignore[import]


@triton.jit
def sdsa_scorer_kernel(
    query_ptr,
    query_stride,
    key_ptr,
    key_page_stride,
    key_row_stride,
    block_table_ptr,
    block_table_stride,
    context_lengths_ptr,
    requests_ptr,
    scores_ptr,
    scores_stride,
    score_scale_ptr,
    ROWS: tl.constexpr,
    SCALED: tl.constexpr,
):
    """Score one 64-key page of a group's request against the group's ROWS
    query rows, padded to 16 rows of zeros for the MMA. Queries and keys are both
    BF16 or both FP8 E4M3, with FP32 accumulation; SCALED
    multiplies the scores by the FP8 score scale at score_scale_ptr."""
    group, page = tl.program_id(0), tl.program_id(1)
    length = tl.load(context_lengths_ptr + group)
    if page * 64 >= length:
        return
    request = tl.load(requests_ptr + group).to(tl.int64)
    block = tl.load(block_table_ptr + request * block_table_stride + page)
    row, key, dim = tl.arange(0, 16), tl.arange(0, 64), tl.arange(0, 128)
    rows = group.to(tl.int64) * ROWS + row
    used = row[:, None] < ROWS
    query = tl.load(
        query_ptr + rows[:, None] * query_stride + dim[None, :], mask=used, other=0.0
    )
    keys = tl.load(
        key_ptr
        + block.to(tl.int64) * key_page_stride
        + key[:, None] * key_row_stride
        + dim[None, :]
    )
    # Triton 3.8 runs this M = 16 FP8 dot on SM90 as mma.sync on E4M3 values
    # converted to FP16, with FP32 accumulation, so the scores match FP64 to
    # FP32 rounding. If a later Triton uses FP8 wgmma, whose accumulator is
    # narrower, recheck against FP64 and split K in two if needed.
    scores = tl.dot(query, tl.trans(keys), out_dtype=tl.float32)
    if SCALED:
        scores *= tl.load(score_scale_ptr)
    key = page * 64 + key
    tl.store(
        scores_ptr + rows[:, None] * scores_stride + key[None, :],
        scores,
        mask=used & (key[None, :] < length),
    )


@triton.jit
def index_qk_fp8_write_kernel(
    key_ptr,
    key_stride,
    slots_ptr,
    keys,
    query_ptr,
    query_stride,
    query_rows_ptr,
    key_cache_ptr,
    key_page_stride,
    key_row_stride,
    key_scale_ptr,
    fp8_query_ptr,
    fp8_query_stride,
    query_scale_ptr,
):
    """Quantize row i to E4M3 at its tensor's scale: key i at slot s = slots[i]
    of the key cache (page s // 64, row s % 64), then query j = i - keys at row
    query_rows[j] of fp8_query. Negative slots and rows are not written. Values
    are divided by the scale in FP32 (rounded to nearest), saturated to +-448 and
    rounded to nearest even (Triton's default for FP32 to FP8). The scale is read
    here, as a block like the row, so a CUDA graph replay sees a scale written
    after capture."""
    i = tl.program_id(0)
    dim = tl.arange(0, 128)
    if i < keys:
        slot = tl.load(slots_ptr + i)
        values = tl.load(key_ptr + i.to(tl.int64) * key_stride + dim).to(tl.float32)
        scale = tl.load(key_scale_ptr + dim * 0)
        out = key_cache_ptr + slot // 64 * key_page_stride + slot % 64 * key_row_stride
    else:
        slot = tl.load(query_rows_ptr + i - keys)
        row = (i - keys).to(tl.int64)
        values = tl.load(query_ptr + row * query_stride + dim).to(tl.float32)
        scale = tl.load(query_scale_ptr + dim * 0)
        out = fp8_query_ptr + slot * fp8_query_stride
    if slot >= 0:
        values = tl.clamp(
            tl.div_rn(values, scale), -448.0, 448.0, propagate_nan=tl.PropagateNan.ALL
        )
        tl.store(out + dim, values.to(tl.float8e4nv))
