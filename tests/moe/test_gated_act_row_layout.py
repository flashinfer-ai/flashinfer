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

import pytest
import torch

from flashinfer.fused_moe import (
    deinterleave_gated_rows,
    reorder_rows_for_gated_act_gemm,
)


def _block_interleave(x1: torch.Tensor, x2: torch.Tensor, block: int) -> torch.Tensor:
    chunks = []
    for a, b in zip(x1.split(block, dim=0), x2.split(block, dim=0), strict=True):
        chunks += [a, b]
    return torch.cat(chunks, dim=0)


@pytest.mark.parametrize("rows", [2, 8, 64])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn, torch.uint8])
def test_deinterleave_inverts_reorder_rows_for_gated_act_gemm(rows, dtype):
    w = torch.arange(rows * 4, dtype=torch.float32).reshape(rows, 4).to(dtype)
    interleaved = reorder_rows_for_gated_act_gemm(w)
    out = deinterleave_gated_rows(interleaved, block=1)
    assert out.shape == w.shape
    assert out.dtype == w.dtype
    torch.testing.assert_close(out.view(torch.uint8), w.view(torch.uint8))


@pytest.mark.parametrize("block", [1, 2, 32])
@pytest.mark.parametrize("intermediate_size", [32, 128])
def test_deinterleave_block_layout(block, intermediate_size):
    x1 = torch.randn(intermediate_size, 16)
    x2 = torch.randn(intermediate_size, 16)
    interleaved = _block_interleave(x1, x2, block)
    out = deinterleave_gated_rows(interleaved, block)
    torch.testing.assert_close(out, torch.cat([x1, x2], dim=0))
    assert out.is_contiguous()


def test_deinterleave_three_dim_rows():
    x1 = torch.randn(8, 4, 2)
    x2 = torch.randn(8, 4, 2)
    out = deinterleave_gated_rows(_block_interleave(x1, x2, 4), 4)
    torch.testing.assert_close(out, torch.cat([x1, x2], dim=0))


def test_deinterleave_rejects_bad_shapes():
    with pytest.raises(ValueError):
        deinterleave_gated_rows(torch.zeros(6, 4), block=4)
    with pytest.raises(ValueError):
        deinterleave_gated_rows(torch.zeros(8, 4), block=0)
