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

import math

import pytest
import torch

from flashinfer.sparse import BlockSparseAttentionWrapper


def _is_sm100_or_sm103() -> bool:
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability() in ((10, 0), (10, 3))


pytestmark = pytest.mark.skipif(
    not _is_sm100_or_sm103(), reason="Cake VSA requires SM100 or SM103"
)


@pytest.mark.parametrize(
    "block_size,dtype,num_qo_heads,num_kv_heads,head_dim,M,N,selected,return_lse",
    [
        (128, torch.bfloat16, 8, 8, 128, 256, 512, 2, True),
        (64, torch.bfloat16, 8, 8, 128, 128, 256, 2, True),
        (128, torch.float16, 8, 1, 128, 256, 512, 2, False),
        (128, torch.float16, 8, 8, 128, 256, 512, 2, True),
        (128, torch.bfloat16, 8, 2, 128, 256, 512, 2, False),
        (128, torch.bfloat16, 8, 8, 64, 256, 512, 2, False),
        (128, torch.bfloat16, 8, 8, 96, 256, 512, 2, False),
        (128, torch.bfloat16, 8, 8, 128, 128, 16384, 8, False),
    ],
)
def test_cake_vsa_against_dense_reference(
    block_size,
    dtype,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    M,
    N,
    selected,
    return_lse,
):
    torch.manual_seed(0)
    device = torch.device("cuda")
    mb, nb = M // block_size, N // block_size
    mask = torch.zeros((num_qo_heads, mb, nb), dtype=torch.bool, device=device)
    for row in range(mb):
        columns = (torch.arange(selected, device=device) * 7 + row) % nb
        mask[:, row, columns] = True

    q = torch.randn((M, num_qo_heads, head_dim), dtype=dtype, device=device)
    k = torch.randn((N, num_kv_heads, head_dim), dtype=dtype, device=device)
    v = torch.randn((N, num_kv_heads, head_dim), dtype=dtype, device=device)
    workspace = torch.empty((128 * 1024 * 1024,), dtype=torch.uint8, device=device)
    wrapper = BlockSparseAttentionWrapper(workspace, backend="cake")
    wrapper.plan(
        None,
        None,
        M,
        N,
        block_size,
        block_size,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        q_data_type=dtype,
        kv_data_type=dtype,
        block_mask=mask,
    )
    result = wrapper.run(q, k, v, return_lse=return_lse)
    output, lse = result if return_lse else (result, None)

    group = num_qo_heads // num_kv_heads
    k_heads = k.repeat_interleave(group, dim=1)
    v_heads = v.repeat_interleave(group, dim=1)
    scale = 1.0 / math.sqrt(head_dim)
    scores = torch.einsum("mhd,nhd->hmn", q.float(), k_heads.float()) * scale
    token_mask = mask.repeat_interleave(block_size, 1).repeat_interleave(block_size, 2)
    scores.masked_fill_(~token_mask, float("-inf"))
    reference = torch.einsum(
        "hmn,nhd->mhd", torch.softmax(scores, dim=-1), v_heads.float()
    ).to(dtype)
    torch.testing.assert_close(output, reference, atol=1e-2, rtol=1e-2)
    if return_lse:
        torch.testing.assert_close(
            lse, torch.logsumexp(scores, dim=-1).transpose(0, 1), atol=1e-2, rtol=1e-2
        )

    repeated = wrapper.run(q, k, v, return_lse=return_lse)
    repeated_output, repeated_lse = repeated if return_lse else (repeated, None)
    assert (
        repeated_output.untyped_storage().data_ptr()
        != output.untyped_storage().data_ptr()
    )
    torch.testing.assert_close(repeated_output, reference, atol=1e-2, rtol=1e-2)
    if return_lse:
        assert (
            repeated_lse.untyped_storage().data_ptr()
            != lse.untyped_storage().data_ptr()
        )


def test_cake_vsa_blk64_per_head_partial_blocks():
    """FastWan-style per-head top-k must exclude partial-tile padding."""

    torch.manual_seed(20260818)
    device = torch.device("cuda")
    block_size, heads, head_dim = 64, 12, 128
    mb, nb, selected = 5, 9, 2
    M, N = mb * block_size, nb * block_size
    mask = torch.zeros((heads, mb, nb), dtype=torch.bool, device=device)
    offsets = torch.arange(selected, device=device)
    for head in range(heads):
        for row in range(mb):
            mask[head, row, (head * 5 + row * 3 + offsets * 7) % nb] = True

    kv_block_lens = torch.tensor(
        [64, 51, 38, 25, 12, 52, 39, 26, 7],
        dtype=torch.int32,
        device=device,
    )
    q = torch.randn((M, heads, head_dim), dtype=torch.bfloat16, device=device)
    k = torch.randn((N, heads, head_dim), dtype=torch.bfloat16, device=device)
    v = torch.randn((N, heads, head_dim), dtype=torch.bfloat16, device=device)
    for block, valid in enumerate(kv_block_lens.tolist()):
        k[block * block_size + valid : (block + 1) * block_size].fill_(20.0)
        v[block * block_size + valid : (block + 1) * block_size].fill_(20.0)

    workspace = torch.empty((128 * 1024 * 1024,), dtype=torch.uint8, device=device)
    wrapper = BlockSparseAttentionWrapper(workspace, backend="cake")
    wrapper.plan(
        None,
        None,
        M,
        N,
        block_size,
        block_size,
        heads,
        heads,
        head_dim,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
        block_mask=mask,
        kv_block_lens=kv_block_lens,
    )
    output, lse = wrapper.run(q, k, v, return_lse=True)

    q2k_indices = torch.topk(mask.to(torch.int8), selected, dim=-1).indices.to(
        torch.int32
    )
    q2k_num = torch.full((heads, mb), selected, dtype=torch.int32, device=device)
    direct_wrapper = BlockSparseAttentionWrapper(workspace, backend="cake")
    direct_wrapper.plan(
        None,
        None,
        M,
        N,
        block_size,
        block_size,
        heads,
        heads,
        head_dim,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
        q2k_indices=q2k_indices,
        q2k_num=q2k_num,
        kv_block_lens=kv_block_lens,
    )
    direct_output, direct_lse = direct_wrapper.run(q, k, v, return_lse=True)

    scale = 1.0 / math.sqrt(head_dim)
    scores = torch.einsum("mhd,nhd->hmn", q.float(), k.float()) * scale
    token_mask = mask.repeat_interleave(block_size, 1).repeat_interleave(block_size, 2)
    token_offset = torch.arange(N, device=device) % block_size
    block_id = torch.arange(N, device=device) // block_size
    token_mask &= (token_offset < kv_block_lens[block_id])[None, None, :]
    scores.masked_fill_(~token_mask, float("-inf"))
    reference = torch.einsum(
        "hmn,nhd->mhd", torch.softmax(scores, dim=-1), v.float()
    ).to(torch.bfloat16)
    reference_lse = torch.logsumexp(scores, dim=-1).transpose(0, 1)
    torch.testing.assert_close(output, reference, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(lse, reference_lse, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(direct_output, reference, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(direct_lse, reference_lse, atol=1e-2, rtol=1e-2)
