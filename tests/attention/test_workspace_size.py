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


def test_batch_decode_workspace_size_plans_with_exact_buffers():
    batch_size = 4
    kv_len = 4096
    page_size = 16
    num_qo_heads = 16
    num_kv_heads = 4
    head_dim = 128
    dtype = torch.float16
    logits_soft_cap = 0.0
    window_left = -1

    indptr, indices, last_page_len = _paged_kv_inputs(batch_size, kv_len, page_size)
    wrapper = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
        _byte_workspace(32 * 1024 * 1024)
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

    module = wrapper._cached_module
    assert module is not None
    assert module.workspace_size is not None

    indptr_host = indptr.cpu()
    float_workspace_size, int_workspace_size = module.workspace_size(
        _byte_workspace(0),
        indptr_host,
        batch_size,
        num_qo_heads,
        num_kv_heads,
        page_size,
        False,  # enable_cuda_graph
        window_left,
        logits_soft_cap,
        head_dim,
        head_dim,
        torch.empty(0, dtype=dtype),
        torch.empty(0, dtype=dtype),
    )
    assert float_workspace_size > 0
    assert int_workspace_size > 0

    plan_info = module.plan(
        _byte_workspace(float_workspace_size),
        _byte_workspace(int_workspace_size),
        torch.empty(int_workspace_size, dtype=torch.uint8, pin_memory=True),
        indptr_host,
        batch_size,
        num_qo_heads,
        num_kv_heads,
        page_size,
        False,  # enable_cuda_graph
        window_left,
        logits_soft_cap,
        head_dim,
        head_dim,
        torch.empty(0, dtype=dtype),
        torch.empty(0, dtype=dtype),
    )
    assert len(plan_info) > 0


def test_batch_prefill_workspace_size_plans_with_exact_buffers():
    batch_size = 3
    qo_len = 64
    kv_len = 1024
    page_size = 16
    num_qo_heads = 16
    num_kv_heads = 4
    head_dim = 128
    fixed_split_size = -1

    qo_indptr = torch.arange(batch_size + 1, dtype=torch.int32, device="cuda") * qo_len
    paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len = _paged_kv_inputs(
        batch_size, kv_len, page_size
    )
    wrapper = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
        _byte_workspace(32 * 1024 * 1024), backend="fa2"
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
        disable_split_kv=True,
    )

    module = wrapper._cached_module
    assert module is not None
    assert module.workspace_size is not None

    qo_indptr_host = qo_indptr.cpu()
    paged_kv_indptr_host = paged_kv_indptr.cpu()
    kv_lens_host = torch.full((batch_size,), kv_len, dtype=torch.int32)
    total_num_rows = int(qo_indptr_host[-1].item())
    float_workspace_size, int_workspace_size = module.workspace_size(
        _byte_workspace(0),
        qo_indptr_host,
        paged_kv_indptr_host,
        kv_lens_host,
        total_num_rows,
        batch_size,
        num_qo_heads,
        num_kv_heads,
        page_size,
        False,  # enable_cuda_graph
        head_dim,
        head_dim,
        False,  # causal
        -1,  # window_left
        fixed_split_size,
        True,  # disable_split_kv
        0,  # num_colocated_ctas
    )
    assert float_workspace_size >= 0
    assert int_workspace_size > 0

    plan_info = module.plan(
        _byte_workspace(float_workspace_size),
        _byte_workspace(int_workspace_size),
        torch.empty(int_workspace_size, dtype=torch.uint8, pin_memory=True),
        qo_indptr_host,
        paged_kv_indptr_host,
        kv_lens_host,
        total_num_rows,
        batch_size,
        num_qo_heads,
        num_kv_heads,
        page_size,
        False,  # enable_cuda_graph
        head_dim,
        head_dim,
        False,  # causal
        -1,  # window_left
        fixed_split_size,
        True,  # disable_split_kv
        0,  # num_colocated_ctas
    )
    assert len(plan_info) > 0
