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
import torch.nn.functional as F

from flashinfer.sparse import VariableBlockSparseAttentionWrapper


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("group_size", [1, 4])
@pytest.mark.parametrize("metadata_device", ["cpu", "cuda"])
@pytest.mark.parametrize("non_blocking", [False, True])
def test_variable_sparse_plan_matches_dense(
    dtype, group_size, metadata_device, non_blocking
):
    torch.manual_seed(123)
    row_sizes = torch.tensor([[3, 0, 5, 9], [7, 4, 0, 6]], dtype=torch.int32)
    col_sizes = torch.tensor(
        [[0, 1, 257, 3, 2], [128, 0, 4, 130, 1]], dtype=torch.int32
    )
    pattern = torch.tensor(
        [
            [[0, 1, 1, 0, 0], [1, 1, 1, 1, 1], [0, 0, 0, 0, 0], [0, 1, 0, 1, 1]],
            [[1, 0, 0, 0, 1], [0, 1, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 1, 1]],
        ],
        dtype=torch.bool,
    )
    # Preserve non-contiguous mask and size inputs.
    mask_storage = torch.empty((2, 4, 10), dtype=torch.bool, device=metadata_device)
    mask = mask_storage[..., ::2]
    mask.copy_(pattern)
    size_storage = torch.empty((2, 10), dtype=torch.int64, device=metadata_device)
    cols = size_storage[:, ::2]
    cols.copy_(col_sizes)
    rows = row_sizes.to(metadata_device)

    wrapper = VariableBlockSparseAttentionWrapper(
        torch.empty(32 * 1024 * 1024, dtype=torch.uint8, device="cuda"),
        backend="fa2",
    )
    q = torch.randn(2 * group_size, 17, 64, dtype=dtype, device="cuda")
    k = torch.randn(2, 263, 64, dtype=dtype, device="cuda")
    v = torch.randn_like(k)

    # Replanning must replace the pattern even when tensor shapes stay fixed.
    for changed in (False, True):
        expected_pattern = pattern.clone()
        if changed:
            expected_pattern[:, :, 0] = True
            mask.copy_(expected_pattern)
        wrapper.plan(
            mask,
            rows,
            cols,
            2 * group_size,
            2,
            64,
            q_data_type=dtype,
            non_blocking=non_blocking,
        )
        expected_indices = []
        expected_indptr = [0]
        for head in range(2):
            for row in range(4):
                offset = head * 263
                for col, length in enumerate(col_sizes[head].tolist()):
                    if expected_pattern[head, row, col]:
                        expected_indices.extend(range(offset, offset + length))
                    offset += length
                expected_indptr.append(len(expected_indices))
        torch.testing.assert_close(
            wrapper._paged_kv_indices_buf.cpu(),
            torch.tensor(expected_indices, dtype=torch.int32),
        )
        torch.testing.assert_close(
            wrapper._paged_kv_indptr_buf.cpu(),
            torch.tensor(expected_indptr, dtype=torch.int32),
        )

        output = wrapper.run(q, k, v)
        for head in range(2):
            dense_mask = (
                expected_pattern[head]
                .repeat_interleave(row_sizes[head].long(), dim=0)
                .repeat_interleave(col_sizes[head].long(), dim=1)
                .to("cuda")
            )
            reference = F.scaled_dot_product_attention(
                q[head * group_size : (head + 1) * group_size].float(),
                k[head : head + 1].float(),
                v[head : head + 1].float(),
                attn_mask=dense_mask,
            ).to(dtype)
            torch.testing.assert_close(
                output[head * group_size : (head + 1) * group_size],
                reference,
                atol=2e-2,
                rtol=2e-2,
            )

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = wrapper.run(q, k, v)
    q.add_(0.125)
    expected = wrapper.run(q, k, v)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(captured, expected)


def test_variable_sparse_plan_empty_mask():
    wrapper = VariableBlockSparseAttentionWrapper(
        torch.empty(32 * 1024 * 1024, dtype=torch.uint8, device="cuda"),
        backend="fa2",
    )
    wrapper.plan(
        torch.zeros((2, 3, 4), dtype=torch.bool, device="cuda"),
        torch.tensor([[2, 3, 4], [4, 3, 2]], dtype=torch.int32, device="cuda"),
        torch.ones((2, 4), dtype=torch.int32, device="cuda"),
        2,
        2,
        64,
    )
    assert wrapper._paged_kv_indices_buf.numel() == 0
    torch.testing.assert_close(
        wrapper._paged_kv_indptr_buf,
        torch.zeros(7, dtype=torch.int32, device="cuda"),
    )


def test_variable_sparse_plan_torch_fallback(monkeypatch):
    import flashinfer.sparse

    monkeypatch.setattr(flashinfer.sparse, "get_compute_capability", lambda _: (7, 5))
    test_variable_sparse_plan_matches_dense(torch.float16, 4, "cuda", True)


def test_variable_sparse_plan_wide_block_map():
    from flashinfer.sparse import _build_variable_block_sparse_metadata

    cols = 65536
    mask = torch.zeros((1, 1, cols), dtype=torch.bool, device="cuda")
    mask[:, :, ::1000] = True
    _, indptr, indices, _, _ = _build_variable_block_sparse_metadata(
        mask,
        torch.ones((1, 1), dtype=torch.int32, device="cuda"),
        torch.ones((1, cols), dtype=torch.int32, device="cuda"),
    )
    torch.testing.assert_close(
        indices, torch.arange(0, cols, 1000, dtype=torch.int32, device="cuda")
    )
    assert indptr[-1].item() == indices.numel()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires two CUDA devices")
def test_variable_sparse_plan_noncurrent_device():
    from flashinfer.sparse import _build_variable_block_sparse_metadata

    with torch.cuda.device(0):
        result = _build_variable_block_sparse_metadata(
            torch.tensor([[[True, False, True]]], device="cuda:1"),
            torch.tensor([[2]], dtype=torch.int32, device="cuda:1"),
            torch.tensor([[2, 3, 1]], dtype=torch.int32, device="cuda:1"),
        )
        assert torch.cuda.current_device() == 0
        torch.testing.assert_close(
            result[2],
            torch.tensor([0, 1, 5], dtype=torch.int32, device="cuda:1"),
        )
