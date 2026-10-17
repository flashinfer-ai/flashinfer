# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import triton
import triton.language as tl


@triton.jit
def _expand_variable_block_sparse_indices(
    block_indptr,
    col_indptr,
    kv_indptr,
    kv_indices,
    NUM_ROWS: tl.constexpr,
    NUM_COLS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    block = tl.program_id(0)
    row = block // NUM_COLS
    col = block % NUM_COLS
    head = row // NUM_ROWS
    block_end = tl.load(block_indptr + row * NUM_COLS + col)
    block_start = tl.load(block_indptr + row * NUM_COLS + col - 1, col > 0, other=0)
    length = block_end - block_start
    if length > 0:
        col_idx = head * NUM_COLS + col
        source_start = tl.load(col_indptr + col_idx - 1, col_idx > 0, other=0)
        dest_start = tl.load(kv_indptr + row).to(tl.int64) + block_start
        for start in tl.range(0, length, BLOCK_SIZE):
            offsets = start + tl.arange(0, BLOCK_SIZE)
            tl.store(
                kv_indices + dest_start + offsets,
                source_start + offsets,
                offsets < length,
            )
