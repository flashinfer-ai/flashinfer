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

"""Die-local scheduling of the CuTe DSL KDA prefill, host side.

When the stock build is kept, how the SM map is encoded for the kernel, and the
scratch layout a caller places memory by.
"""

import pytest
import torch

from flashinfer.kda_prefill_cute import _die1_sm_mask

SM_COUNT = 212
TWO_DOMAINS = [(sm >> 1) & 1 for sm in range(SM_COUNT)]


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "domains, num_sequences",
    [
        (None, 8),
        (TWO_DOMAINS, 1),
        (TWO_DOMAINS[:-1], 8),
        ([0] * SM_COUNT, 8),
        ([sm % 3 for sm in range(SM_COUNT)], 8),
    ],
    ids=["no-map", "one-sequence", "sm-missing", "one-domain", "three-domains"],
)
def test_stock_build_is_kept(domains, num_sequences):
    assert _die1_sm_mask(domains, SM_COUNT, num_sequences) == ()


@pytest.mark.cpu_only
def test_sm_mask_has_one_bit_per_sm():
    mask = _die1_sm_mask(TWO_DOMAINS, SM_COUNT, 2)

    assert len(mask) == 7
    assert all(-(2**31) <= word < 2**31 for word in mask)
    assert [(mask[sm // 32] >> (sm % 32)) & 1 for sm in range(SM_COUNT)] == (
        TWO_DOMAINS
    )


@pytest.mark.cpu_only
def test_workspace_ranges_match_the_partition():
    pytest.importorskip("cutlass")
    from flashinfer.kda_kernels import kda_chunked_bt16 as bt16

    seq_lens, heads, capacity = [40, 16, 7], 3, 6
    chunk_starts = [0, 3, 4, 5]
    workspace = torch.zeros(bt16._decomp_ws_bytes(heads, capacity), dtype=torch.uint8)
    for region in bt16._partition_workspace(workspace, heads, capacity).values():
        rows_per_chunk = region.shape[2] // capacity
        region_bytes = region.view(torch.uint8)
        for seq in range(len(seq_lens)):
            rows = slice(
                chunk_starts[seq] * rows_per_chunk,
                chunk_starts[seq + 1] * rows_per_chunk,
            )
            region_bytes[0, :, rows] = seq + 1

    expected = torch.zeros_like(workspace)
    for seq, ranges in enumerate(bt16.workspace_ranges(seq_lens, heads, capacity)):
        for start, stop in ranges:
            assert not expected[start:stop].any()
            expected[start:stop] = seq + 1
    assert torch.equal(workspace, expected)
