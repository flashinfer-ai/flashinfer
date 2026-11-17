# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 by FlashInfer team.

"""The BF16 and FP8 scorer of the SDSA (Simplified DSA) indexer off SM10x, such as
on Hopper."""

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
