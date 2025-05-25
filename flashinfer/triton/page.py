"""
Copyright (c) 2025 by FlashInfer team.

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
def get_batch_indices_positions_kernel(
    append_indptr,
    seq_lens_ptr,
    batch_indices_ptr,
    positions_ptr,
    num_stages: tl.constexpr,
):
    batch_idx = tl.program_id(0)

    batch_start = tl.load(append_indptr + batch_idx)
    batch_end = tl.load(append_indptr + batch_idx + 1)
    seq_len = tl.load(seq_lens_ptr + batch_idx)

    for i in tl.range(batch_start, batch_end, 128, num_stages=num_stages):
        offsets = tl.arange(0, 128) + i
        mask = offsets < batch_end
        tl.store(batch_indices_ptr + offsets, batch_idx, mask)
        tl.store(positions_ptr + offsets, offsets + seq_len - batch_end, mask)
