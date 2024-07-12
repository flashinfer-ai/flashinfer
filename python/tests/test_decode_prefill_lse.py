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

import flashinfer
import numpy as np
import torch
import pytest


def test_mlc_failed_case():
    kv_layout = "HND"
    num_pages = 12
    kv_indptr_1 = torch.tensor([0, 0, 9]).int().to(0)
    kv_indices_1 = torch.tensor([3, 4, 5, 6, 7, 8, 9, 10, 11]).int().to(0)
    kv_last_page_len_1 = torch.tensor([0, 1]).int().to(0)
    num_qo_heads = 32
    num_kv_heads = 32
    page_size = 16
    head_dim = 128
    q = torch.randn(2, num_qo_heads, head_dim).to(0).half()
    kv_data = torch.randn(12, 2, num_kv_heads, page_size, head_dim).to(0).half()

    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8).to(0)
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace_buffer, kv_layout)
    wrapper.begin_forward(
        kv_indptr_1,
        kv_indices_1,
        kv_last_page_len_1,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        pos_encoding_mode="NONE",
        data_type=torch.float16,
        q_data_type=torch.float16,
    )
    o_1, lse_1 = wrapper.forward_return_lse(q, kv_data)

    wrapper_tensor_cores = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer, kv_layout, use_tensor_cores=True
    )
    wrapper_tensor_cores.begin_forward(
        kv_indptr_1,
        kv_indices_1,
        kv_last_page_len_1,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        pos_encoding_mode="NONE",
        data_type=torch.float16,
        q_data_type=torch.float16,
    )
    o_1_tc, lse_1_tc = wrapper_tensor_cores.forward_return_lse(q, kv_data)

    np.testing.assert_allclose(
        lse_1.cpu().numpy(), lse_1_tc.cpu().numpy(), rtol=1e-3, atol=1e-3
    )
    np.testing.assert_allclose(
        o_1.cpu().numpy(), o_1_tc.cpu().numpy(), rtol=1e-3, atol=1e-3
    )


if __name__ == "__main__":
    test_mlc_failed_case()
