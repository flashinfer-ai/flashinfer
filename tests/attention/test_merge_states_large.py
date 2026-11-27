# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""merge_states over partial buffers past 32-bit offsets and CUDA grid limits.

The cases allocate 2-10 GiB of GPU memory, so this file runs alone on its GPU.
"""

import math

import pytest
import torch

import flashinfer

pytestmark = [
    pytest.mark.gpu_exclusive,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
]


def _ref_merge_rows(v, s, rows):
    """Reference merge of the given rows of (n, chunks, h, d) partials with log2 LSEs."""
    out, lse = [], []
    for r in rows:
        s_r = s[r] * math.log(2)
        w = torch.softmax(s_r, dim=0)
        out.append((w[..., None] * v[r].float()).sum(0))
        lse.append(torch.logsumexp(s_r, dim=0) / math.log(2))
    return torch.stack(out), torch.stack(lse)


@pytest.mark.parametrize(
    "seq_len,num_chunks,num_heads",
    [
        (65_537, 4, 32),  # MergeStatesKernel (num_chunks < seq_len)
        (256, 257, 128),  # MergeStatesLargeNumIndexSetsKernel (num_chunks >= seq_len)
    ],
)
def test_merge_states_full_buffer_past_uint32_boundary(seq_len, num_chunks, num_heads):
    """Verify merge_states indexes the whole partial buffer with 64-bit offsets.

    The partial O holds more than 2^32 elements, so the last token's offsets wrap in 32-bit math.
    """
    d = 512
    numel = seq_len * num_chunks * num_heads * d
    assert numel > 1 << 32
    need = (numel + seq_len * num_heads * d) * 2 + (1 << 30)
    if torch.cuda.mem_get_info()[0] < need:
        pytest.skip(f"needs {need / 2**30:.1f} GiB of free GPU memory")
    gen = torch.Generator(device="cuda").manual_seed(0)
    v = torch.empty(
        (seq_len, num_chunks, num_heads, d), dtype=torch.bfloat16, device="cuda"
    )
    v.normal_(generator=gen)
    s = torch.randn((seq_len, num_chunks, num_heads), device="cuda", generator=gen)
    merged_o, merged_lse = flashinfer.merge_states(v, s)
    rows = [0, seq_len - 2, seq_len - 1]
    ref_o, ref_lse = _ref_merge_rows(v, s, rows)
    torch.testing.assert_close(merged_o[rows].float(), ref_o, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(merged_lse[rows], ref_lse, atol=1e-3, rtol=1e-3)


def test_merge_states_rejects_head_tiles_past_grid_y_limit():
    """Verify more than 65,535 * 32 heads at D=512 fail with a launch error.

    V and the merged output take about 2 GiB each.
    """
    num_heads = 65_535 * 32 + 1
    if torch.cuda.mem_get_info()[0] < (5 << 30):
        pytest.skip("needs 5 GiB of free GPU memory")
    v = torch.empty((1, 1, num_heads, 512), dtype=torch.bfloat16, device="cuda")
    s = torch.zeros((1, 1, num_heads), device="cuda")
    with pytest.raises(
        RuntimeError, match="MergeStates kernel launch failed: invalid argument"
    ):
        flashinfer.merge_states(v, s)
