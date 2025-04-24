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

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _build_pos_ids_from_segment_offsets_and_lengths(
    segment_offsets: tl.tensor,
    packed_segment_offsets: tl.tensor,
    out: tl.tensor,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)

    # Get the start and end indices for this segment
    start_idx = tl.load(segment_offsets + pid)
    segment_length = tl.load(packed_segment_offsets + pid + 1) - tl.load(
        packed_segment_offsets + pid
    )
    pos_ids_offset = tl.load(packed_segment_offsets + pid)

    # Process the segment in blocks
    for i in range(0, segment_length, BLOCK_SIZE):
        # Create offsets within the block
        offsets = tl.arange(0, BLOCK_SIZE)

        # Calculate position IDs (i + offsets represents position within the segment)
        pos_ids = i + offsets

        # Store position IDs in output array where valid
        mask = pos_ids < segment_length
        tl.store(out + pos_ids_offset + pos_ids, start_idx + pos_ids, mask=mask)


def build_pos_ids_from_segment_offsets_and_lengths(
    segment_offsets: torch.Tensor,
    packed_segment_offsets: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    num_elements = packed_segment_offsets[-1]
    if out is None:
        out = torch.empty(
            num_elements,
            device=packed_segment_offsets.device,
            dtype=packed_segment_offsets.dtype,
        )
    else:
        assert out.shape == (num_elements,)

    n_segments = segment_offsets.shape[0] - 1
    BLOCK_SIZE = 1024

    # call triton kernel
    _build_pos_ids_from_segment_offsets_and_lengths[(n_segments,)](
        segment_offsets,
        packed_segment_offsets,
        out,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out
