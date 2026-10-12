"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import triton
import triton.language as tl


@triton.jit
def segment_scale_128x4_per_group_kernel(
    src_ptr,  # int32 words of the source F8_128x4 blob over all tokens
    dst_ptr,  # int32 words of the destination blob, groups padded to 128 rows
    indptr_ptr,  # int32 [num_groups + 1] row offsets of the groups in the source
    num_groups,
    src_rows,  # rows of the source blob (a multiple of 128)
    num_col_words: tl.constexpr,  # scale columns / 4
    BLOCK_G: tl.constexpr,  # power of two >= num_groups
    BLOCK_C: tl.constexpr,  # column words per program
):
    """One program writes one destination 128-row block for ``BLOCK_C`` column words.

    In the F8_128x4 layout the four scales of row ``r`` in column group ``c`` are one
    32-bit word at ``((r // 128) * C4 + c) * 128 + (r % 32) * 4 + (r // 32) % 4``. The
    destination holds every group padded to whole 128-row blocks, groups in order;
    rows past a group's end, and blocks past the last group, are zero.
    """
    dst_block = tl.program_id(0)
    col_tile = tl.program_id(1)

    g = tl.arange(0, BLOCK_G)
    g_mask = g < num_groups
    lo = tl.load(indptr_ptr + g, mask=g_mask, other=0)
    hi = tl.load(indptr_ptr + g + 1, mask=g_mask, other=0)
    blocks = (hi - lo + 127) // 128
    block_end = tl.cumsum(blocks, axis=0)
    # the group whose padded region holds this block (num_groups if past the last one)
    group = tl.sum((g_mask & (block_end <= dst_block)).to(tl.int32), axis=0)
    in_range = group < num_groups
    sel = g == group
    g_lo = tl.sum(tl.where(sel, lo, 0), axis=0)
    g_rows = tl.sum(tl.where(sel, hi - lo, 0), axis=0)
    g_block_start = tl.sum(tl.where(sel, block_end - blocks, 0), axis=0)

    r = tl.arange(0, 128)
    c = col_tile * BLOCK_C + tl.arange(0, BLOCK_C)
    local = (dst_block - g_block_start) * 128 + r
    row_ok = in_range & (local < g_rows)
    s = tl.minimum(g_lo + local, src_rows - 1)
    src_word = ((s // 128)[:, None] * num_col_words + c[None, :]) * 128 + (
        (s % 32) * 4 + (s // 32) % 4
    )[:, None]
    dst_word = (dst_block * num_col_words + c[None, :]) * 128 + (
        (r % 32) * 4 + (r // 32) % 4
    )[:, None]
    c_ok = c[None, :] < num_col_words
    words = tl.load(src_ptr + src_word, mask=row_ok[:, None] & c_ok, other=0)
    tl.store(dst_ptr + dst_word, words, mask=c_ok)
