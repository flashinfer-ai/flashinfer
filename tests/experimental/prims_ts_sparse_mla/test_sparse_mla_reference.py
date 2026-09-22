# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import pytest
import torch

from flashinfer.testing.sparse_mla import (
    sparse_mla_input_fingerprint,
    sparse_mla_reference,
)


def test_joint_softmax_duplicates_and_sink():
    q = torch.zeros(1, 1, 2, 512, dtype=torch.bfloat16)
    swa = torch.full((2, 1, 512), 2.0, dtype=torch.bfloat16)
    compressed = torch.full((1, 2, 512), 10.0, dtype=torch.bfloat16)
    out, lse = sparse_mla_reference(
        q,
        swa,
        compressed,
        torch.tensor([[[1, 1, -1]]], dtype=torch.int32),
        torch.tensor([[[0, 1]]], dtype=torch.int32),
        compressed_topk_lens=torch.tensor([[1]], dtype=torch.int32),
        sinks=torch.tensor([math.log(2), torch.inf]),
    )
    # Two duplicated SWA entries plus one compressed entry. Sink contributes
    # denominator mass only. This distinguishes a joint softmax from averaging.
    torch.testing.assert_close(
        out[0, 0, 0], torch.full((512,), 14 / 5, dtype=torch.float64)
    )
    assert torch.count_nonzero(out[0, 0, 1]) == 0
    torch.testing.assert_close(
        lse, torch.full((1, 1, 2), math.log(3), dtype=torch.float64)
    )


def test_source_scales_padded_pages_and_empty_queries():
    q = torch.zeros(2, 1, 512, dtype=torch.bfloat16)
    backing = torch.zeros(2, 3, 512, dtype=torch.bfloat16)
    swa = backing[:, :1]
    swa[0].fill_(3)
    swa[1].fill_(7)
    compressed = torch.full((1, 1, 512), 5.0, dtype=torch.bfloat16)
    out, lse = sparse_mla_reference(
        q,
        swa,
        compressed,
        torch.tensor([[1, -1], [-1, -1]], dtype=torch.int32),
        torch.tensor([[0], [0]], dtype=torch.int32),
        compressed_topk_lens=torch.tensor([1, 0], dtype=torch.int32),
        swa_kv_scale=2,
        compressed_kv_scale=4,
    )
    torch.testing.assert_close(out[0], torch.full((1, 512), 17.0, dtype=torch.float64))
    assert torch.count_nonzero(out[1]) == 0
    assert torch.isneginf(lse[1]).all()


def test_rope_channels_contribute_to_values():
    q = torch.zeros(1, 1, 512, dtype=torch.bfloat16)
    kv = torch.zeros(1, 1, 512, dtype=torch.bfloat16)
    kv[..., 448:] = 3
    out, _ = sparse_mla_reference(q, kv, None, torch.zeros(1, 1, dtype=torch.int32))
    torch.testing.assert_close(out, kv.to(torch.float64))


@pytest.mark.parametrize("bad_index", [-2, 1])
def test_invalid_active_index_is_rejected(bad_index):
    q = torch.zeros(1, 1, 512)
    with pytest.raises(ValueError, match="outside its source"):
        sparse_mla_reference(q, q, None, torch.tensor([[bad_index]], dtype=torch.int32))


def test_fingerprint_detects_mutation():
    q = torch.zeros(1, 1, 512, dtype=torch.bfloat16)
    before = sparse_mla_input_fingerprint(q=q, scale=1.0)
    assert before == sparse_mla_input_fingerprint(q=q.clone(), scale=1.0)
    q[..., -1] = 1
    assert before != sparse_mla_input_fingerprint(q=q, scale=1.0)
