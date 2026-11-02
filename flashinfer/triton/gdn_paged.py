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

Per-sequence pool rows of the paged GDN state cache (flashinfer.gdn_paged),
one launch for the whole batch. All operands are non-negative, so integer
``//`` (truncating in Triton) is the floor division the formulas need.
"""

import triton
import triton.language as tl


@triton.jit
def gdn_paged_prefill_slots_kernel(
    cu_seqlens,
    page_indices,
    page_indptr,
    last_page_len,
    read_rows,
    write_rows,
    checkpoint_starts,
    prefix_lens,
    num_seqs,
    page_size,
    BLOCK: tl.constexpr,
):
    seq = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = seq < num_seqs
    seq_len = tl.load(cu_seqlens + seq + 1, mask=valid, other=0) - tl.load(
        cu_seqlens + seq, mask=valid, other=0
    )
    first = tl.load(page_indptr + seq, mask=valid, other=0)
    end = tl.load(page_indptr + seq + 1, mask=valid, other=1)
    total = (end - first - 1) * page_size + tl.load(
        last_page_len + seq, mask=valid, other=0
    )
    prefix = total - seq_len
    # The last prefix token lives on entry (prefix - 1) // page_size.
    read_entry = first + tl.maximum(prefix - 1, 0) // page_size
    read = tl.load(page_indices + read_entry, mask=valid & (prefix > 0), other=-1)
    # A final state that ends on a page boundary is that page's checkpoint.
    write = tl.load(
        page_indices + end - 1,
        mask=valid & (seq_len > 0) & (total % page_size != 0),
        other=-1,
    )
    tl.store(read_rows + seq, read.to(tl.int32), mask=valid)
    tl.store(write_rows + seq, write.to(tl.int32), mask=valid)
    # The first page boundary crossed closes entry prefix // page_size.
    tl.store(
        checkpoint_starts + seq, (first + prefix // page_size).to(tl.int32), mask=valid
    )
    tl.store(prefix_lens + seq, prefix.to(tl.int32), mask=valid)


@triton.jit
def gdn_paged_decode_slots_kernel(
    page_indices,
    page_indptr,
    last_page_len,
    read_rows,
    write_rows,
    num_seqs,
    BLOCK: tl.constexpr,
):
    seq = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = seq < num_seqs
    last_entry = tl.load(page_indptr + seq + 1, mask=valid, other=1) - 1
    write = tl.load(page_indices + last_entry, mask=valid, other=0)
    # A token that opens a new page reads the previous page.
    new_page = tl.load(last_page_len + seq, mask=valid, other=2) == 1
    previous = tl.load(
        page_indices + tl.maximum(last_entry - 1, 0), mask=valid & new_page, other=0
    )
    tl.store(write_rows + seq, write.to(tl.int32), mask=valid)
    tl.store(
        read_rows + seq, tl.where(new_page, previous, write).to(tl.int32), mask=valid
    )
