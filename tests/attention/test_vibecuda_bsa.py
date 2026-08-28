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
from flashinfer.vibecuda_bsa import (
    vibecuda_block_sparse_attention,
    vibecuda_bsa_split_g,
    vibecuda_bsa_workspace_numel,
)


def _is_sm100_or_sm103() -> bool:
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability() in ((10, 0), (10, 3))


pytestmark = pytest.mark.skipif(
    not _is_sm100_or_sm103(), reason="VibeCUDA BSA requires SM100 or SM103"
)


def _dense_reference(q, k, v, mask, block_size):
    group = q.shape[1] // k.shape[1]
    k_heads = k.repeat_interleave(group, dim=1)
    v_heads = v.repeat_interleave(group, dim=1)
    scale = 1.0 / math.sqrt(q.shape[2])
    scores = torch.einsum("mhd,nhd->hmn", q.float(), k_heads.float()) * scale
    token_mask = mask.repeat_interleave(block_size, 1).repeat_interleave(
        block_size, 2
    )
    scores.masked_fill_(~token_mask, float("-inf"))
    reference = torch.einsum(
        "hmn,nhd->mhd", torch.softmax(scores, dim=-1), v_heads.float()
    ).to(q.dtype)
    reference_lse = torch.logsumexp(scores, dim=-1).transpose(0, 1)
    return reference, reference_lse


def _make_inputs(block_size, dtype, num_qo_heads, num_kv_heads, head_dim, M, N, selected):
    torch.manual_seed(0)
    device = torch.device("cuda")
    mb, nb = M // block_size, N // block_size
    mask = torch.zeros((num_qo_heads, mb, nb), dtype=torch.bool, device=device)
    offsets = torch.arange(selected, device=device)
    for row in range(mb):
        columns = (offsets * 7 + row) % nb
        mask[:, row, columns] = True
    q = torch.randn((M, num_qo_heads, head_dim), dtype=dtype, device=device)
    k = torch.randn((N, num_kv_heads, head_dim), dtype=dtype, device=device)
    v = torch.randn((N, num_kv_heads, head_dim), dtype=dtype, device=device)
    return q, k, v, mask


# The canonical workload matrix shared with the repository's cake backend test.
_VSA_CASES = [
    (128, torch.bfloat16, 8, 8, 128, 256, 512, 2, True),
    (64, torch.bfloat16, 8, 8, 128, 128, 256, 2, True),
    (128, torch.float16, 8, 1, 128, 256, 512, 2, False),
    (128, torch.float16, 8, 8, 128, 256, 512, 2, True),
    (128, torch.bfloat16, 8, 2, 128, 256, 512, 2, False),
    (128, torch.bfloat16, 8, 8, 64, 256, 512, 2, False),
    (128, torch.bfloat16, 8, 8, 96, 256, 512, 2, False),
    (128, torch.bfloat16, 8, 8, 128, 128, 16384, 8, False),
]


@pytest.mark.parametrize(
    "block_size,dtype,num_qo_heads,num_kv_heads,head_dim,M,N,selected,return_lse",
    _VSA_CASES,
)
def test_vibecuda_bsa_wrapper_against_dense_reference(
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
    q, k, v, mask = _make_inputs(
        block_size, dtype, num_qo_heads, num_kv_heads, head_dim, M, N, selected
    )
    reference, reference_lse = _dense_reference(q, k, v, mask, block_size)

    workspace = torch.empty((128 * 1024 * 1024,), dtype=torch.uint8, device="cuda")
    wrapper = BlockSparseAttentionWrapper(workspace, backend="vibecuda")
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


@pytest.mark.parametrize(
    "block_size,dtype,num_qo_heads,num_kv_heads,head_dim,M,N,selected,return_lse",
    _VSA_CASES,
)
def test_vibecuda_bsa_functional_against_dense_reference(
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
    q, k, v, mask = _make_inputs(
        block_size, dtype, num_qo_heads, num_kv_heads, head_dim, M, N, selected
    )
    reference, reference_lse = _dense_reference(q, k, v, mask, block_size)

    split_g = vibecuda_bsa_split_g(selected, block_size, N)
    ws = None
    ws_numel = vibecuda_bsa_workspace_numel(M, num_qo_heads, head_dim, split_g)
    if ws_numel > 0:
        ws = torch.zeros(ws_numel, dtype=torch.float32, device=q.device)

    result = vibecuda_block_sparse_attention(
        q, k, v, mask, block_size, return_lse=return_lse,
        workspace=ws, split_g=split_g,
    )
    output, lse = result if return_lse else (result, None)
    torch.testing.assert_close(output, reference, atol=1e-2, rtol=1e-2)
    if return_lse:
        torch.testing.assert_close(lse, reference_lse, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_vibecuda_bsa_out_and_sm_scale_parameters(dtype):
    """Preallocated outputs and a non-default sm_scale must be honored."""
    torch.manual_seed(1)
    device = torch.device("cuda")
    block_size, num_qo_heads, num_kv_heads, head_dim, M, N, selected = (
        128,
        8,
        4,
        128,
        256,
        512,
        2,
    )
    q, k, v, mask = _make_inputs(
        block_size, dtype, num_qo_heads, num_kv_heads, head_dim, M, N, selected
    )

    sm_scale = 0.5 / math.sqrt(head_dim)
    out = torch.full_like(q, float("nan"))
    lse = torch.full((M, num_qo_heads), float("nan"), dtype=torch.float32, device=device)
    result = vibecuda_block_sparse_attention(
        q, k, v, mask, block_size, sm_scale=sm_scale, out=out, lse=lse, return_lse=True
    )
    assert result[0].data_ptr() == out.data_ptr()
    assert result[1].data_ptr() == lse.data_ptr()
    # A NaN-poisoned output buffer must be fully overwritten.
    assert torch.isfinite(out.float()).all()
    assert torch.isfinite(lse).all()

    group = num_qo_heads // num_kv_heads
    scores = torch.einsum(
        "mhd,nhd->hmn",
        q.float(),
        k.repeat_interleave(group, dim=1).float(),
    ) * sm_scale
    token_mask = mask.repeat_interleave(block_size, 1).repeat_interleave(block_size, 2)
    scores.masked_fill_(~token_mask, float("-inf"))
    reference = torch.einsum(
        "hmn,nhd->mhd",
        torch.softmax(scores, dim=-1),
        v.repeat_interleave(group, dim=1).float(),
    ).to(dtype)
    torch.testing.assert_close(out, reference, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(
        lse, torch.logsumexp(scores, dim=-1).transpose(0, 1), atol=1e-2, rtol=1e-2
    )


def test_vibecuda_bsa_rejects_unsupported_configuration():
    device = torch.device("cuda")
    block_size, num_qo_heads, num_kv_heads, head_dim, M, N, selected = (
        128,
        8,
        8,
        128,
        256,
        512,
        2,
    )
    q, k, v, mask = _make_inputs(
        block_size, torch.bfloat16, num_qo_heads, num_kv_heads, head_dim, M, N, selected
    )
    workspace = torch.empty((128 * 1024 * 1024,), dtype=torch.uint8, device="cuda")

    # run() before plan() must fail loudly.
    wrapper = BlockSparseAttentionWrapper(workspace, backend="vibecuda")
    with pytest.raises(RuntimeError, match="plan"):
        wrapper.run(q, k, v)

    # A mask with a wrong shape must be rejected at plan time.
    bad_mask = torch.zeros(
        (num_qo_heads, M // block_size, N // block_size + 1),
        dtype=torch.bool,
        device=device,
    )
    with pytest.raises(ValueError, match="block_mask"):
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
            q_data_type=torch.bfloat16,
            block_mask=bad_mask,
        )

    # head_dim outside the supported set must be rejected at plan time.
    with pytest.raises(ValueError, match="head_dim"):
        wrapper.plan(
            None,
            None,
            M,
            N,
            block_size,
            block_size,
            num_qo_heads,
            num_kv_heads,
            192,
            q_data_type=torch.bfloat16,
            block_mask=mask,
        )

    # FP32 inputs are not supported by the functional API either.
    with pytest.raises(ValueError, match="bfloat16 or float16"):
        vibecuda_block_sparse_attention(q.float(), k, v, mask, block_size)

    # A block size that is not a multiple of 64 is rejected.
    with pytest.raises(ValueError, match="multiple of 64"):
        vibecuda_block_sparse_attention(q, k, v, mask, 32)
