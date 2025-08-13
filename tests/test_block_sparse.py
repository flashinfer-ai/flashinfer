"""
Copyright (c) 2024 by FlashInfer team.

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

import numpy as np
import pytest
import scipy as sp
import torch
from jit_utils import gen_decode_attention_modules, gen_prefill_attention_modules

import flashinfer


@pytest.fixture(autouse=True, scope="module")
def warmup_jit():
    flashinfer.jit.build_jit_specs(
        gen_decode_attention_modules(
            [torch.float16],  # q_dtypes
            [torch.float16],  # kv_dtypes
            [128, 256],  # head_dims
            [0],  # pos_encoding_modes
            [False],  # use_sliding_windows
            [False],  # use_logits_soft_caps
        )
        + gen_prefill_attention_modules(
            [torch.float16],  # q_dtypes
            [torch.float16],  # kv_dtypes
            [128, 256],  # head_dims
            [0],  # pos_encoding_modes
            [False],  # use_sliding_windows
            [False],  # use_logits_soft_caps
            [False],  # use_fp16_qk_reductions
        ),
        verbose=False,
    )
    yield


def bsr_attention_ref(
    q,
    k,
    v,
    indptr,
    indices,
    mask_data,
):
    M = q.shape[0]
    N = k.shape[0]
    bsr = sp.sparse.bsr_matrix(
        (mask_data.cpu().numpy(), indices.cpu().numpy(), indptr.cpu().numpy()),
        shape=(M, N),
    )
    dense_mask = torch.tensor(bsr.toarray(), dtype=bool, device=q.device)
    o = flashinfer.prefill.single_prefill_with_kv_cache(q, k, v, custom_mask=dense_mask)
    return o


def set_seed(seed: int = 42):
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


@pytest.mark.parametrize("R", [1, 4, 16])
@pytest.mark.parametrize("C", [1, 4, 16])
@pytest.mark.parametrize("M", [64, 128, 256])
@pytest.mark.parametrize("N", [64, 128, 256])
@pytest.mark.parametrize("num_qo_heads", [1, 4, 16])
@pytest.mark.parametrize("num_kv_heads", [1, 4, 16])
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("mask_inside_block", [True, False])
def test_block_sparse_attention(
    R, C, M, N, num_qo_heads, num_kv_heads, head_dim, mask_inside_block
):
    if num_qo_heads % num_kv_heads != 0:
        pytest.skip("num_qo_heads must be divisible by num_kv_heads")

    set_seed(33)
    rng = np.random.default_rng()

    MB = M // R
    NB = N // C
    S = sp.sparse.random(MB, NB, density=0.25, random_state=rng).tocsr()
    indptr = torch.from_numpy(S.indptr).to(0)
    indices = torch.from_numpy(S.indices).to(0)
    nnz = S.nnz
    if mask_inside_block:
        data_mask = (torch.rand((nnz, R, C)) > 0.5).to(0)
    else:
        data_mask = torch.full((nnz, R, C), True, dtype=bool, device=0)
    q = torch.randn((M, num_qo_heads, head_dim), dtype=torch.float16, device=0)
    k = torch.randn((N, num_kv_heads, head_dim), dtype=torch.float16, device=0)
    v = torch.randn((N, num_kv_heads, head_dim), dtype=torch.float16, device=0)

    o_ref = bsr_attention_ref(q, k, v, indptr, indices, data_mask)
    workspace_buffer = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=0)
    sparse_attention_wrapper = flashinfer.sparse.BlockSparseAttentionWrapper(
        workspace_buffer
    )

    sparse_attention_wrapper.plan(
        indptr,
        indices,
        M,
        N,
        R,
        C,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        mask=data_mask if mask_inside_block else None,
    )

    o = sparse_attention_wrapper.run(q, k, v)
    torch.testing.assert_close(o_ref, o, atol=1e-2, rtol=1e-3)

    # test with pre-allocated output
    o_buffer = torch.empty_like(o)
    sparse_attention_wrapper.run(q, k, v, out=o_buffer)
    torch.testing.assert_close(o_ref, o_buffer, atol=1e-2, rtol=1e-3)


def _ref_attention(
    q: torch.Tensor,  # [gqa_group_size, qo_len, head_dim]
    k: torch.Tensor,  # [1, kv_len, head_dim]
    v: torch.Tensor,  # [1, kv_len, head_dim]
    block_mask_map: torch.Tensor,  # [MB, NB]
    block_row_sz: torch.Tensor,  # [MB]
    block_col_sz: torch.Tensor,  # [NB]
) -> torch.Tensor:
    # convert block mask map to element mask
    def _block_mask_to_element_mask(
        block_mask_map: torch.Tensor,  # [MB, NB] – bool
        block_row_sz: torch.Tensor,  # [MB]     – int (rows per block-row)
        block_col_sz: torch.Tensor,  # [NB]     – int (cols per block-col)
    ) -> torch.Tensor:
        block_row_sz = block_row_sz.to(block_mask_map.device, dtype=torch.long)
        block_col_sz = block_col_sz.to(block_mask_map.device, dtype=torch.long)
        expanded_rows = torch.repeat_interleave(block_mask_map, block_row_sz, dim=0)
        element_mask = torch.repeat_interleave(expanded_rows, block_col_sz, dim=1)

        return element_mask

    dense_mask = _block_mask_to_element_mask(
        block_mask_map, block_row_sz, block_col_sz
    ).to(dtype=torch.bool, device=q.device)

    q = q.transpose(0, 1).contiguous()
    k = k.transpose(0, 1).contiguous()
    v = v.transpose(0, 1).contiguous()
    o = flashinfer.prefill.single_prefill_with_kv_cache(
        q, k, v, custom_mask=dense_mask
    )  # [qo_len, gqa_group_size, head_dim]
    o = o.transpose(0, 1).contiguous()

    return o


@pytest.mark.parametrize("num_qo_heads", [1, 4, 16])
@pytest.mark.parametrize("num_kv_heads", [1, 4, 16])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("seq_len", [256, 4096, 8192])
@pytest.mark.parametrize("num_blocks_row", [10, 20])
@pytest.mark.parametrize("num_blocks_col", [50, 100])
@pytest.mark.parametrize("block_density", [0.2, 0.7, 0.9])
def test_variable_block_sparse_attention_wrapper(
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    seq_len: int,
    num_blocks_row: int,
    num_blocks_col: int,
    block_density: float,
):
    if num_qo_heads % num_kv_heads != 0:
        pytest.skip("num_qo_heads must be divisible by num_kv_heads")
    if seq_len // num_blocks_row < 1:
        pytest.skip("seq_len must be greater than num_blocks_row")
    if seq_len // num_blocks_col < 1:
        pytest.skip("seq_len must be greater than num_blocks_col")

    set_seed(330)

    def random_partition_batch(
        seq_len: int,
        num_blocks: int,
        bsz: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.int32,
    ) -> torch.Tensor:
        assert seq_len >= num_blocks
        sizes = torch.empty((bsz, num_blocks), dtype=dtype, device=device)
        for i in range(bsz):
            cut_pts = torch.randperm(seq_len - 1, device=device)[: num_blocks - 1] + 1
            cut_pts, _ = torch.sort(cut_pts)
            row_sizes = torch.diff(
                torch.cat(
                    (
                        torch.tensor([0], device=device),
                        cut_pts,
                        torch.tensor([seq_len], device=device),
                    )
                )
            )
            sizes[i] = row_sizes

        assert sizes.min() >= 1
        assert sizes.max() <= seq_len
        assert torch.all(sizes.sum(dim=-1) == seq_len)

        return sizes.to(device=device)

    def _test_variable_block_sparse_attention(
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        block_mask_map: torch.Tensor,
        block_row_sz: torch.Tensor,
        block_col_sz: torch.Tensor,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.float16,
    ):
        # qkv: HND
        qo_len = block_row_sz.sum(dim=1)[0].item()
        kv_len = block_col_sz.sum(dim=1)[0].item()
        assert torch.all(block_col_sz.sum(dim=1) == block_col_sz.sum(dim=1)[0])
        assert torch.all(block_row_sz.sum(dim=1) == block_row_sz.sum(dim=1)[0])

        q = torch.randn(num_qo_heads, qo_len, head_dim, device=device, dtype=dtype)
        k = torch.randn(num_kv_heads, kv_len, head_dim, device=device, dtype=dtype)
        v = torch.randn(num_kv_heads, kv_len, head_dim, device=device, dtype=dtype)

        float_workspace_buffer = torch.empty(128 * 1024 * 1024, device=device)
        wrapper = flashinfer.sparse.VariableBlockSparseAttentionWrapper(
            float_workspace_buffer, backend="auto"
        )

        wrapper.plan(
            block_mask_map=block_mask_map,
            block_row_sz=block_row_sz,
            block_col_sz=block_col_sz,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            q_data_type=dtype,
        )

        o: torch.Tensor = wrapper.run(q, k, v)  # [num_qo_heads, qo_len, head_dim]
        o = o.reshape(num_kv_heads, -1, *o.shape[-2:])
        q = q.reshape(num_kv_heads, -1, *q.shape[-2:])
        for kv_head_idx in range(num_kv_heads):
            o_ref = _ref_attention(
                q[kv_head_idx],
                k[kv_head_idx : kv_head_idx + 1, :, :],
                v[kv_head_idx : kv_head_idx + 1, :, :],
                block_mask_map[kv_head_idx],
                block_row_sz[kv_head_idx],
                block_col_sz[kv_head_idx],
            )
            torch.testing.assert_close(o[kv_head_idx], o_ref, atol=1e-2, rtol=1e-2)

    block_row_sz = random_partition_batch(
        seq_len, num_blocks_row, num_kv_heads, device="cuda:0"
    )
    block_col_sz = random_partition_batch(
        seq_len, num_blocks_col, num_kv_heads, device="cuda:0"
    )
    block_mask_map = (
        torch.rand(num_kv_heads, num_blocks_row, num_blocks_col) > block_density
    ).to(device="cuda:0")

    _test_variable_block_sparse_attention(
        num_qo_heads,
        num_kv_heads,
        head_dim,
        block_mask_map,
        block_row_sz,
        block_col_sz,
    )


if __name__ == "__main__":
    # This test verifies the INT32_T overflow issue.
    for seq_len in [16 * 1024, 32 * 1024, 40 * 1024, 48 * 1024, 64 * 1024]:
        test_block_sparse_attention(128, 128, seq_len, seq_len, 1, 1, 128, False)
