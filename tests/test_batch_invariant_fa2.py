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

import pytest
import torch
from jit_utils import gen_decode_attention_modules, gen_prefill_attention_modules

import flashinfer


@pytest.fixture(autouse=True, scope="module")
def warmup_jit():
    flashinfer.jit.build_jit_specs(
        gen_decode_attention_modules(
            [torch.float16],  # q_dtypes
            [torch.float16],  # kv_dtypes
            [64, 128, 256],  # head_dims
            [0, 1],  # pos_encoding_modes
            [False],  # use_sliding_windows
            [False],  # use_logits_soft_caps
        )
        + gen_prefill_attention_modules(
            [torch.float16],  # q_dtypes
            [torch.float16],  # kv_dtypes
            [64, 128, 256],  # head_dims
            [0, 1],  # pos_encoding_modes
            [False],  # use_sliding_windows
            [False],  # use_logits_soft_caps
            [False],  # use_fp16_qk_reductions
        ),
        verbose=False,
    )
    yield


@pytest.mark.parametrize("batch_size", [5, 12])
@pytest.mark.parametrize("invariant_bs", [4])
@pytest.mark.parametrize("kv_len", [4096, 8192, 5000])
@pytest.mark.parametrize("fixed_split_size", [2048])
@pytest.mark.parametrize("disable_split_kv", [True, False])
@pytest.mark.parametrize("page_size", [1, 8, 16])
@pytest.mark.parametrize("num_kv_heads", [4])
@pytest.mark.parametrize("group_size", [1, 4, 8])
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("kv_layout", ["HND", "NHD"])
@pytest.mark.parametrize("pos_encoding_mode", ["NONE", "ROPE_LLAMA"])
def test_batch_decode_tensor_cores(
    batch_size: int,
    invariant_bs: int,
    kv_len: int,
    fixed_split_size: int,
    disable_split_kv: bool,
    page_size: int,
    num_kv_heads: int,
    group_size: int,
    head_dim: int,
    kv_layout: str,
    pos_encoding_mode: str,
):
    num_qo_heads = num_kv_heads * group_size
    q = torch.randn(
        batch_size, num_qo_heads, head_dim, device="cuda:0", dtype=torch.float16
    )
    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size
    kv_data = (
        torch.randn(
            total_num_pages,
            2,
            num_kv_heads,
            page_size,
            head_dim,
            device="cuda:0",
            dtype=torch.float16,
        )
        / 10
        if kv_layout == "HND"
        else torch.randn(
            total_num_pages,
            2,
            page_size,
            num_kv_heads,
            head_dim,
            device="cuda:0",
            dtype=torch.float16,
        )
        / 10
    )
    kv_indptr = (
        torch.arange(0, batch_size + 1, device="cuda:0", dtype=torch.int32)
        * num_pages_per_seq
    )
    kv_indices = torch.arange(0, total_num_pages, device="cuda:0", dtype=torch.int32)
    kv_last_page_len = torch.full(
        (batch_size,), (kv_len - 1) % page_size + 1, dtype=torch.int32, device="cuda:0"
    )

    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device="cuda:0")

    wrapper_tensor_cores = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer, kv_layout, use_tensor_cores=True
    )
    wrapper_tensor_cores.plan(
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        pos_encoding_mode=pos_encoding_mode,
        data_type=torch.float16,
        q_data_type=torch.float16,
        fixed_split_size=fixed_split_size if not disable_split_kv else None,
        disable_split_kv=disable_split_kv,
    )
    o_tensor_cores, lse_tensor_cores = wrapper_tensor_cores.run(
        q, kv_data, return_lse=True
    )

    kv_indptr_invariant = kv_indptr[: invariant_bs + 1]
    kv_last_page_len_invariant = kv_last_page_len[:invariant_bs]
    wrapper_tensor_cores.plan(
        kv_indptr_invariant,
        kv_indices,
        kv_last_page_len_invariant,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        pos_encoding_mode=pos_encoding_mode,
        data_type=torch.float16,
        q_data_type=torch.float16,
        fixed_split_size=fixed_split_size if not disable_split_kv else None,
        disable_split_kv=disable_split_kv,
    )
    o_tensor_cores_invariant, lse_tensor_cores_invariant = wrapper_tensor_cores.run(
        q[:invariant_bs], kv_data, return_lse=True
    )
    assert torch.equal(o_tensor_cores[:invariant_bs], o_tensor_cores_invariant)
    assert torch.equal(lse_tensor_cores[:invariant_bs], lse_tensor_cores_invariant)


# test that without fixed split size, precision is different
# TODO: this works for the first 29 cases, but then fails with "illegal memory access"..?

# wrapper_tensor_cores.plan(
#     kv_indptr,
#     kv_indices,
#     kv_last_page_len,
#     num_qo_heads,
#     num_kv_heads,
#     head_dim,
#     page_size,
# )
# o_tensor_cores_invariant, lse_tensor_cores_invariant = wrapper_tensor_cores.run(
#     q[:invariant_bs], kv_data, return_lse=True
# )
# try:
#     torch.testing.assert_close(
#         o_tensor_cores[:invariant_bs], o_tensor_cores_invariant, rtol=1e-7, atol=1e-7
#     )
#     torch.testing.assert_close(
#         lse_tensor_cores[:invariant_bs], lse_tensor_cores_invariant, rtol=1e-7, atol=1e-7
#     )
# except AssertionError:
#     pass
# else:
#     raise AssertionError("Precision is the same without fixed split size")


@pytest.mark.parametrize("batch_size", [3, 4])
@pytest.mark.parametrize("invariant_bs", [2])
@pytest.mark.parametrize("kv_len", [4096, 5000])
@pytest.mark.parametrize("qo_len", [128, 256])
@pytest.mark.parametrize("fixed_split_size", [2048])
@pytest.mark.parametrize("disable_split_kv", [True, False])
@pytest.mark.parametrize("page_size", [1, 8, 16])
@pytest.mark.parametrize("num_kv_heads", [4])
@pytest.mark.parametrize("group_size", [1, 4, 8])
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("kv_layout", ["HND", "NHD"])
@pytest.mark.parametrize("pos_encoding_mode", ["NONE", "ROPE_LLAMA"])
def test_batch_prefill_tensor_cores(
    batch_size: int,
    invariant_bs: int,
    kv_len: int,
    qo_len: int,
    fixed_split_size: int,
    disable_split_kv: bool,
    page_size: int,
    num_kv_heads: int,
    group_size: int,
    head_dim: int,
    kv_layout: str,
    pos_encoding_mode: str,
):
    num_qo_heads = num_kv_heads * group_size
    q = torch.randn(
        batch_size * qo_len,
        num_qo_heads,
        head_dim,
        device="cuda:0",
        dtype=torch.float16,
    )
    q_indptr = (
        torch.arange(0, batch_size + 1, device="cuda:0", dtype=torch.int32) * qo_len
    )
    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size
    kv_data = (
        torch.randn(
            total_num_pages,
            2,
            num_kv_heads,
            page_size,
            head_dim,
            device="cuda:0",
            dtype=torch.float16,
        )
        / 10
        if kv_layout == "HND"
        else torch.randn(
            total_num_pages,
            2,
            page_size,
            num_kv_heads,
            head_dim,
            device="cuda:0",
            dtype=torch.float16,
        )
        / 10
    )
    kv_indptr = (
        torch.arange(0, batch_size + 1, device="cuda:0", dtype=torch.int32)
        * num_pages_per_seq
    )
    kv_indices = torch.arange(0, total_num_pages, device="cuda:0", dtype=torch.int32)
    kv_last_page_len = torch.full(
        (batch_size,), (kv_len - 1) % page_size + 1, dtype=torch.int32, device="cuda:0"
    )

    workspace_buffer = torch.empty(
        1024 * 1024 * 1024, dtype=torch.int8, device="cuda:0"
    )

    wrapper_tensor_cores = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer, kv_layout
    )
    wrapper_tensor_cores.plan(
        q_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        pos_encoding_mode=pos_encoding_mode,
        q_data_type=torch.float16,
        kv_data_type=torch.float16,
        fixed_split_size=fixed_split_size if not disable_split_kv else None,
        disable_split_kv=disable_split_kv,
    )
    o_tensor_cores, lse_tensor_cores = wrapper_tensor_cores.run(
        q, kv_data, return_lse=True
    )

    # Test invariant batch size
    q_indptr_invariant = q_indptr[: invariant_bs + 1]
    kv_indptr_invariant = kv_indptr[: invariant_bs + 1]
    kv_last_page_len_invariant = kv_last_page_len[:invariant_bs]
    wrapper_tensor_cores.plan(
        q_indptr_invariant,
        kv_indptr_invariant,
        kv_indices,
        kv_last_page_len_invariant,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        pos_encoding_mode=pos_encoding_mode,
        q_data_type=torch.float16,
        kv_data_type=torch.float16,
        fixed_split_size=fixed_split_size if not disable_split_kv else None,
        disable_split_kv=disable_split_kv,
    )
    o_tensor_cores_invariant, lse_tensor_cores_invariant = wrapper_tensor_cores.run(
        q[: invariant_bs * qo_len], kv_data, return_lse=True
    )

    # Compare outputs for the invariant batch size
    assert torch.equal(
        o_tensor_cores[: invariant_bs * qo_len], o_tensor_cores_invariant
    )
    assert torch.equal(
        lse_tensor_cores[: invariant_bs * qo_len], lse_tensor_cores_invariant
    )
