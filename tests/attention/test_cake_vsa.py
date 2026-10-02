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

from flashinfer import cake_vsa
from flashinfer.sparse import BlockSparseAttentionWrapper


def _is_sm100_or_sm103() -> bool:
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability() in ((10, 0), (10, 3))


pytestmark = pytest.mark.skipif(
    not _is_sm100_or_sm103(), reason="Cake VSA requires SM100 or SM103"
)


def _strided_mask(num_qo_heads, mb, nb, selected, device):
    mask = torch.zeros((num_qo_heads, mb, nb), dtype=torch.bool, device=device)
    for row in range(mb):
        columns = (torch.arange(selected, device=device) * 7 + row) % nb
        mask[:, row, columns] = True
    return mask


def _dense_reference(q, k, v, mask, block_size, scale=None):
    group = q.shape[1] // k.shape[1]
    k_heads = k.repeat_interleave(group, dim=1)
    v_heads = v.repeat_interleave(group, dim=1)
    scale = scale if scale is not None else 1.0 / math.sqrt(q.shape[-1])
    scores = torch.einsum("mhd,nhd->hmn", q.float(), k_heads.float()) * scale
    token_mask = mask.repeat_interleave(block_size, 1).repeat_interleave(block_size, 2)
    scores.masked_fill_(~token_mask, float("-inf"))
    output = torch.einsum(
        "hmn,nhd->mhd", torch.softmax(scores, dim=-1), v_heads.float()
    ).to(q.dtype)
    return output, torch.logsumexp(scores, dim=-1).transpose(0, 1)


def _plan(wrapper, M, N, block_size, num_qo_heads, num_kv_heads, head_dim, dtype, **kw):
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
        **kw,
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
        # FP16 GQA direct route beyond the single-tile grid: three selected
        # blocks, two 256-row query tiles, with log-sum-exp.
        (128, torch.float16, 8, 2, 128, 512, 1024, 3, True),
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
    mask = _strided_mask(num_qo_heads, mb, nb, selected, device)

    q = torch.randn((M, num_qo_heads, head_dim), dtype=dtype, device=device)
    k = torch.randn((N, num_kv_heads, head_dim), dtype=dtype, device=device)
    v = torch.randn((N, num_kv_heads, head_dim), dtype=dtype, device=device)
    workspace = torch.empty((128 * 1024 * 1024,), dtype=torch.uint8, device=device)
    wrapper = BlockSparseAttentionWrapper(workspace, backend="cake")
    _plan(wrapper, M, N, block_size, num_qo_heads, num_kv_heads, head_dim, dtype, block_mask=mask)
    result = wrapper.run(q, k, v, return_lse=return_lse)
    output, lse = result if return_lse else (result, None)

    reference, reference_lse = _dense_reference(q, k, v, mask, block_size)
    torch.testing.assert_close(output, reference, atol=1e-2, rtol=1e-2)
    if return_lse:
        torch.testing.assert_close(lse, reference_lse, atol=1e-2, rtol=1e-2)

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


def test_run_does_not_synchronize_after_plan():
    """Every route launches from plan-time metadata; run() performs no host sync."""

    torch.manual_seed(1)
    device = torch.device("cuda")
    workspace = torch.empty((128 * 1024 * 1024,), dtype=torch.uint8, device=device)
    rows = [
        # (block_size, dtype, Hq, Hkv, head_dim, M, N, selected)
        (128, torch.bfloat16, 8, 8, 128, 256, 512, 2),
        (128, torch.float16, 8, 2, 128, 256, 512, 2),
        (64, torch.bfloat16, 4, 4, 128, 128, 256, 2),
        (128, torch.bfloat16, 8, 8, 64, 256, 512, 2),
    ]
    for block_size, dtype, hq, hkv, head_dim, M, N, selected in rows:
        mb, nb = M // block_size, N // block_size
        mask = _strided_mask(hq, mb, nb, selected, device)
        q = torch.randn((M, hq, head_dim), dtype=dtype, device=device)
        k = torch.randn((N, hkv, head_dim), dtype=dtype, device=device)
        v = torch.randn((N, hkv, head_dim), dtype=dtype, device=device)
        wrapper = BlockSparseAttentionWrapper(workspace, backend="cake")
        _plan(wrapper, M, N, block_size, hq, hkv, head_dim, dtype, block_mask=mask)
        # Warm the JIT module outside the guarded region.
        wrapper.run(q, k, v)
        torch.cuda.synchronize()

        torch.cuda.set_sync_debug_mode("error")
        try:
            output = wrapper.run(q, k, v)
        finally:
            torch.cuda.set_sync_debug_mode("default")
        torch.cuda.synchronize()

        reference, _ = _dense_reference(q, k, v, mask, block_size)
        torch.testing.assert_close(output, reference, atol=1e-2, rtol=1e-2)


def test_fp16_gqa_replan_replaces_direct_metadata():
    torch.manual_seed(2)
    device = torch.device("cuda")
    M, N, block_size = 256, 1024, 128
    hq, hkv, head_dim = 8, 2, 128
    mb, nb = M // block_size, N // block_size
    q = torch.randn((M, hq, head_dim), dtype=torch.float16, device=device)
    k = torch.randn((N, hkv, head_dim), dtype=torch.float16, device=device)
    v = torch.randn((N, hkv, head_dim), dtype=torch.float16, device=device)
    workspace = torch.empty((128 * 1024 * 1024,), dtype=torch.uint8, device=device)
    wrapper = BlockSparseAttentionWrapper(workspace, backend="cake")

    first_mask = _strided_mask(hq, mb, nb, 2, device)
    _plan(wrapper, M, N, block_size, hq, hkv, head_dim, torch.float16, block_mask=first_mask)
    first_plan = wrapper._cake_vsa_plan
    assert first_plan["fp16_direct"]["topk"] == 2
    first = wrapper.run(q, k, v)
    assert torch.equal(wrapper.run(q, k, v), first)
    reference, _ = _dense_reference(q, k, v, first_mask, block_size)
    torch.testing.assert_close(first, reference, atol=1e-2, rtol=1e-2)

    second_mask = torch.zeros((hq, mb, nb), dtype=torch.bool, device=device)
    second_mask[:4, :, [1, 5, 6]] = True
    second_mask[4:, :, [0, 3, 7]] = True
    _plan(wrapper, M, N, block_size, hq, hkv, head_dim, torch.float16, block_mask=second_mask)
    second_plan = wrapper._cake_vsa_plan
    assert second_plan is not first_plan
    assert second_plan["fp16_direct"]["topk"] == 3
    second = wrapper.run(q, k, v)
    reference, _ = _dense_reference(q, k, v, second_mask, block_size)
    torch.testing.assert_close(second, reference, atol=1e-2, rtol=1e-2)
    assert not torch.equal(second, first)


def test_custom_sm_scale_reaches_every_route():
    torch.manual_seed(3)
    device = torch.device("cuda")
    workspace = torch.empty((128 * 1024 * 1024,), dtype=torch.uint8, device=device)
    rows = [
        (128, torch.bfloat16, 8, 8, 128, 256, 512, 2),
        (128, torch.float16, 8, 2, 128, 256, 512, 2),
        (64, torch.bfloat16, 4, 4, 128, 128, 256, 2),
    ]
    scale = 0.05
    for block_size, dtype, hq, hkv, head_dim, M, N, selected in rows:
        mb, nb = M // block_size, N // block_size
        mask = _strided_mask(hq, mb, nb, selected, device)
        q = torch.randn((M, hq, head_dim), dtype=dtype, device=device)
        k = torch.randn((N, hkv, head_dim), dtype=dtype, device=device)
        v = torch.randn((N, hkv, head_dim), dtype=dtype, device=device)
        wrapper = BlockSparseAttentionWrapper(workspace, backend="cake")
        _plan(
            wrapper, M, N, block_size, hq, hkv, head_dim, dtype, block_mask=mask, sm_scale=scale
        )
        output = wrapper.run(q, k, v)
        reference, _ = _dense_reference(q, k, v, mask, block_size, scale=scale)
        torch.testing.assert_close(output, reference, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("bad_shape", [(128, 8, 64), (128, 1024)])
def test_blk64_direct_rejects_invalid_output_shape(bad_shape):
    torch.manual_seed(0)
    device = torch.device("cuda")
    M, N = 128, 256
    num_heads, head_dim = 8, 128
    mask = torch.zeros((num_heads, 2, 4), dtype=torch.bool, device=device)
    mask[:, :, :2] = True
    q = torch.randn((M, num_heads, head_dim), dtype=torch.bfloat16, device=device)
    k = torch.randn((N, num_heads, head_dim), dtype=torch.bfloat16, device=device)
    v = torch.randn_like(k)
    workspace = torch.empty((128 * 1024 * 1024,), dtype=torch.uint8, device=device)
    wrapper = BlockSparseAttentionWrapper(workspace, backend="cake")
    _plan(wrapper, M, N, 64, num_heads, num_heads, head_dim, torch.bfloat16, block_mask=mask)
    plan = wrapper._cake_vsa_plan
    assert plan is not None
    assert plan["blk64_profile"] == "blk64_persistent"
    bad_output = torch.empty(bad_shape, dtype=q.dtype, device=device)
    stats = torch.empty((M, num_heads), dtype=torch.float32, device=device)

    with pytest.raises(ValueError, match="out"):
        cake_vsa._run_blk64(plan, q, k, v, bad_output, stats, False)


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
    _plan(
        wrapper,
        M,
        N,
        block_size,
        heads,
        heads,
        head_dim,
        torch.bfloat16,
        block_mask=mask,
        kv_block_lens=kv_block_lens,
    )
    output, lse = wrapper.run(q, k, v, return_lse=True)

    q2k_indices = torch.topk(mask.to(torch.int8), selected, dim=-1).indices.to(
        torch.int32
    )
    q2k_num = torch.full((heads, mb), selected, dtype=torch.int32, device=device)
    direct_wrapper = BlockSparseAttentionWrapper(workspace, backend="cake")
    _plan(
        direct_wrapper,
        M,
        N,
        block_size,
        heads,
        heads,
        head_dim,
        torch.bfloat16,
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


def test_cake_vsa_blk64_full_group_rows_use_weight_stationary_profile():
    """Mixed full-group rows remain correct on the plan-wide WS route."""

    torch.manual_seed(20260823)
    device = torch.device("cuda")
    block_size, heads, head_dim = 64, 2, 128
    mb, nb = 2, 32
    M, N = mb * block_size, nb * block_size
    q2k_indices = (
        torch.arange(nb, dtype=torch.int32, device=device)
        .expand(heads, mb, nb)
        .contiguous()
    )
    q2k_num = torch.tensor([[24, 28], [24, 28]], dtype=torch.int32, device=device)
    q = torch.randn((M, heads, head_dim), dtype=torch.bfloat16, device=device)
    k = torch.randn((N, heads, head_dim), dtype=torch.bfloat16, device=device)
    v = torch.randn((N, heads, head_dim), dtype=torch.bfloat16, device=device)
    inputs = tuple(tensor.clone() for tensor in (q, k, v, q2k_indices, q2k_num))

    workspace = torch.empty((128 * 1024 * 1024,), dtype=torch.uint8, device=device)
    wrapper = BlockSparseAttentionWrapper(workspace, backend="cake")
    _plan(
        wrapper,
        M,
        N,
        block_size,
        heads,
        heads,
        head_dim,
        torch.bfloat16,
        q2k_indices=q2k_indices,
        q2k_num=q2k_num,
    )
    assert wrapper._cake_vsa_plan is not None
    assert wrapper._cake_vsa_plan["blk64_profile"] == "blk64_persistent_ws_m64n256"

    output, lse = wrapper.run(q, k, v, return_lse=True)
    repeated_output, repeated_lse = wrapper.run(q, k, v, return_lse=True)
    assert torch.equal(repeated_output, output)
    assert torch.equal(repeated_lse, lse)

    block_mask = torch.zeros((heads, mb, nb), dtype=torch.bool, device=device)
    for head in range(heads):
        for row in range(mb):
            count = int(q2k_num[head, row].item())
            block_mask[head, row, q2k_indices[head, row, :count].long()] = True
    token_mask = block_mask.repeat_interleave(block_size, 1).repeat_interleave(
        block_size, 2
    )
    scores = torch.einsum("mhd,nhd->hmn", q.float(), k.float()) / math.sqrt(head_dim)
    scores.masked_fill_(~token_mask, float("-inf"))
    reference = torch.einsum(
        "hmn,nhd->mhd", torch.softmax(scores, dim=-1), v.float()
    ).to(torch.bfloat16)
    reference_lse = torch.logsumexp(scores, dim=-1).transpose(0, 1)
    torch.testing.assert_close(output, reference, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(lse, reference_lse, atol=1e-2, rtol=1e-2)

    for tensor, original in zip((q, k, v, q2k_indices, q2k_num), inputs, strict=True):
        assert torch.equal(tensor, original)
