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

import math

import pytest
import torch

from flashinfer.sparse import BlockSparseAttentionWrapper
from flashinfer.utils import is_sm90a_supported

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_sm90a_supported(torch.device("cuda")),
    reason="vsa_sm90_blk64 backend requires an SM90 GPU",
)

R = C = 64
HEAD_DIM = 128


@pytest.fixture(scope="module")
def workspace():
    """Allocate the workspace shared by the SM90 attention tests."""
    return torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device="cuda")


def _reference(q, k, v, block_mask, sm_scale=None):
    """Compute a dense block-masked attention reference result."""
    _, num_qo_heads, head_dim = q.shape
    gqa_ratio = num_qo_heads // k.shape[1]
    if sm_scale is None:
        sm_scale = 1 / math.sqrt(head_dim)

    k = k.repeat_interleave(gqa_ratio, dim=1).float().permute(1, 0, 2)
    v = v.repeat_interleave(gqa_ratio, dim=1).float().permute(1, 0, 2)
    scores = torch.matmul(q.float().permute(1, 0, 2), k.transpose(-1, -2))
    token_mask = block_mask.repeat_interleave(R, 1).repeat_interleave(C, 2)
    scores = (scores * sm_scale).masked_fill(~token_mask, float("-inf"))
    valid_rows = token_mask.any(dim=-1)
    lse = torch.logsumexp(scores, dim=-1)
    probs = torch.where(valid_rows.unsqueeze(-1), torch.softmax(scores, dim=-1), 0)
    out = torch.matmul(probs, v).permute(1, 0, 2).to(q.dtype)
    return out, lse.permute(1, 0)


def _bsr_from_mask(mask):
    """Convert a boolean block mask to BSR row pointers and column indices."""
    counts = mask.sum(dim=-1, dtype=torch.int32)
    indptr = torch.nn.functional.pad(counts.cumsum(0, dtype=torch.int32), (1, 0))
    indices = mask.nonzero(as_tuple=False)[:, 1].to(torch.int32)
    return indptr, indices


@pytest.mark.parametrize(
    "dtype,num_qo_heads,num_kv_heads,MB,NB,use_bsr,sm_scale",
    [
        (torch.bfloat16, 4, 4, 8, 8, True, None),
        (torch.float16, 8, 4, 8, 8, True, 0.5),
        (torch.bfloat16, 8, 1, 4, 8, False, None),
    ],
)
def test_vsa_sm90_accuracy(
    dtype,
    num_qo_heads,
    num_kv_heads,
    MB,
    NB,
    use_bsr,
    sm_scale,
    workspace,
):
    """Check SM90 VSA output and LSE against the dense reference."""
    torch.manual_seed(42)
    device = torch.device("cuda")
    q = torch.randn(MB * R, num_qo_heads, HEAD_DIM, dtype=dtype, device=device)
    k = torch.randn(NB * C, num_kv_heads, HEAD_DIM, dtype=dtype, device=device)
    v = torch.randn_like(k)

    mask_heads = 1 if use_bsr else num_qo_heads
    block_mask = torch.rand(mask_heads, MB, NB, device=device) > 0.5
    block_mask[:, :, 0] = True
    reference_mask = block_mask.expand(num_qo_heads, -1, -1)
    expected, expected_lse = _reference(q, k, v, reference_mask, sm_scale)

    wrapper = BlockSparseAttentionWrapper(workspace, backend="vsa_sm90_blk64")
    if use_bsr:
        indptr, indices = _bsr_from_mask(block_mask[0])
        wrapper.plan(
            indptr,
            indices,
            MB * R,
            NB * C,
            R,
            C,
            num_qo_heads,
            num_kv_heads,
            HEAD_DIM,
            q_data_type=dtype,
            sm_scale=sm_scale,
        )
    else:
        wrapper.plan(
            None,
            None,
            MB * R,
            NB * C,
            R,
            C,
            num_qo_heads,
            num_kv_heads,
            HEAD_DIM,
            q_data_type=dtype,
            sm_scale=sm_scale,
            block_mask=block_mask,
        )

    actual, actual_lse = wrapper.run(q, k, v, return_lse=True)
    torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(actual_lse, expected_lse, atol=1e-2, rtol=1e-2)


def test_vsa_sm90_compact_metadata_and_empty_row(workspace):
    """Check compact BSR metadata handling, including an empty query row."""
    torch.manual_seed(43)
    device = torch.device("cuda")
    MB = NB = 4
    num_heads = 4
    q = torch.randn(MB * R, num_heads, HEAD_DIM, dtype=torch.bfloat16, device=device)
    k = torch.randn(NB * C, num_heads, HEAD_DIM, dtype=torch.bfloat16, device=device)
    v = torch.randn_like(k)
    indptr = torch.tensor([0, 0, 2, 3, 5], dtype=torch.int32, device=device)
    indices = torch.tensor([0, 1, 2, 1, 3], dtype=torch.int32, device=device)

    block_mask = torch.zeros(num_heads, MB, NB, dtype=torch.bool, device=device)
    for row in range(MB):
        block_mask[:, row, indices[indptr[row] : indptr[row + 1]]] = True
    expected, expected_lse = _reference(q, k, v, block_mask)

    wrapper = BlockSparseAttentionWrapper(workspace, backend="vsa_sm90_blk64")
    wrapper.plan(
        indptr,
        indices,
        MB * R,
        NB * C,
        R,
        C,
        num_heads,
        num_heads,
        HEAD_DIM,
        q_data_type=torch.bfloat16,
    )
    assert wrapper._vsa_q2k_index.shape[-1] == 2

    actual, actual_lse = wrapper.run(q, k, v, return_lse=True)
    torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(actual_lse, expected_lse, atol=1e-2, rtol=1e-2)
