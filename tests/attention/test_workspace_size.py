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

import pytest
import torch

import flashinfer


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="workspace sizing tests require CUDA"
)


def _paged_kv_inputs(batch_size: int, kv_len: int, page_size: int):
    pages_per_seq = (kv_len + page_size - 1) // page_size
    total_pages = batch_size * pages_per_seq
    indptr = (
        torch.arange(batch_size + 1, dtype=torch.int32, device="cuda") * pages_per_seq
    )
    indices = torch.arange(total_pages, dtype=torch.int32, device="cuda")
    last_page_len = torch.full(
        (batch_size,),
        (kv_len - 1) % page_size + 1,
        dtype=torch.int32,
        device="cuda",
    )
    return indptr, indices, last_page_len


def _byte_workspace(num_bytes: int, device: str = "cuda"):
    return torch.empty((num_bytes,), dtype=torch.uint8, device=device)


@pytest.mark.parametrize("use_cuda_graph", [False, True])
def test_batch_decode_workspace_size_plans_with_exact_buffers(use_cuda_graph):
    batch_size = 4
    kv_len = 4096
    page_size = 16
    num_qo_heads = 16
    num_kv_heads = 4
    head_dim = 128
    dtype = torch.float16

    indptr, indices, last_page_len = _paged_kv_inputs(batch_size, kv_len, page_size)
    kwargs = {}
    if use_cuda_graph:
        kwargs = {
            "use_cuda_graph": True,
            "paged_kv_indptr_buffer": torch.empty(
                batch_size + 1, dtype=torch.int32, device="cuda"
            ),
            "paged_kv_indices_buffer": torch.empty(
                len(indices), dtype=torch.int32, device="cuda"
            ),
            "paged_kv_last_page_len_buffer": torch.empty(
                batch_size, dtype=torch.int32, device="cuda"
            ),
        }
    wrapper = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
        _byte_workspace(32 * 1024 * 1024), **kwargs
    )

    float_workspace_size, int_workspace_size = wrapper.workspace_size(
        indptr,
        indices,
        last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        q_data_type=dtype,
        kv_data_type=dtype,
    )
    assert float_workspace_size > 0
    assert int_workspace_size > 0

    wrapper.reset_workspace_buffer(
        _byte_workspace(float_workspace_size),
        _byte_workspace(int_workspace_size),
    )
    wrapper.plan(
        indptr,
        indices,
        last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        q_data_type=dtype,
        kv_data_type=dtype,
    )
    assert wrapper._plan_info is not None


@pytest.mark.parametrize("use_cuda_graph", [False, True])
def test_batch_prefill_workspace_size_plans_with_exact_buffers(use_cuda_graph):
    batch_size = 3
    qo_len = 64
    kv_len = 1024
    page_size = 16
    num_qo_heads = 16
    num_kv_heads = 4
    head_dim = 128
    fixed_split_size = 16

    qo_indptr = torch.arange(batch_size + 1, dtype=torch.int32, device="cuda") * qo_len
    paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len = _paged_kv_inputs(
        batch_size, kv_len, page_size
    )
    kwargs = {}
    if use_cuda_graph:
        kwargs = {
            "use_cuda_graph": True,
            "qo_indptr_buf": torch.empty(
                batch_size + 1, dtype=torch.int32, device="cuda"
            ),
            "paged_kv_indptr_buf": torch.empty(
                batch_size + 1, dtype=torch.int32, device="cuda"
            ),
            "paged_kv_indices_buf": torch.empty(
                len(paged_kv_indices), dtype=torch.int32, device="cuda"
            ),
            "paged_kv_last_page_len_buf": torch.empty(
                batch_size, dtype=torch.int32, device="cuda"
            ),
        }
    wrapper = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
        _byte_workspace(32 * 1024 * 1024), backend="fa2", **kwargs
    )

    float_workspace_size, int_workspace_size = wrapper.workspace_size(
        qo_indptr,
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        fixed_split_size=fixed_split_size,
        disable_split_kv=False,
    )
    assert float_workspace_size > 0
    assert int_workspace_size > 0

    wrapper.reset_workspace_buffer(
        _byte_workspace(float_workspace_size),
        _byte_workspace(int_workspace_size),
    )
    wrapper.plan(
        qo_indptr,
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        fixed_split_size=fixed_split_size,
        disable_split_kv=False,
    )
    assert wrapper._plan_info is not None
