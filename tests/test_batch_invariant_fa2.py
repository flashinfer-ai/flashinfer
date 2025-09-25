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
            [torch.bfloat16],  # q_dtypes
            [torch.bfloat16],  # kv_dtypes
            [64, 128, 256],  # head_dims
            [0, 1],  # pos_encoding_modes
            [False],  # use_sliding_windows
            [False],  # use_logits_soft_caps
        )
        + gen_prefill_attention_modules(
            [torch.bfloat16],  # q_dtypes
            [torch.bfloat16],  # kv_dtypes
            [64, 128, 256],  # head_dims
            [0, 1],  # pos_encoding_modes
            [False],  # use_sliding_windows
            [False],  # use_logits_soft_caps
            [False],  # use_fp16_qk_reductions
        ),
        verbose=False,
    )
    yield


# @pytest.mark.parametrize("batch_size", [5, 12])
# @pytest.mark.parametrize("invariant_bs", [4])
# @pytest.mark.parametrize("kv_len", [4096, 8192, 5000])
# @pytest.mark.parametrize("fixed_split_size", [2048])
# @pytest.mark.parametrize("disable_split_kv", [True, False])
# @pytest.mark.parametrize("page_size", [1, 8, 16])
# @pytest.mark.parametrize("num_kv_heads", [4])
# @pytest.mark.parametrize("group_size", [1, 4, 8])
# @pytest.mark.parametrize("head_dim", [128, 256])
# @pytest.mark.parametrize("kv_layout", ["HND", "NHD"])
# @pytest.mark.parametrize("pos_encoding_mode", ["NONE", "ROPE_LLAMA"])
# def test_batch_decode_tensor_cores(
#     batch_size: int,
#     invariant_bs: int,
#     kv_len: int,
#     fixed_split_size: int,
#     disable_split_kv: bool,
#     page_size: int,
#     num_kv_heads: int,
#     group_size: int,
#     head_dim: int,
#     kv_layout: str,
#     pos_encoding_mode: str,
# ):
#     num_qo_heads = num_kv_heads * group_size
#     q = torch.randn(
#         batch_size, num_qo_heads, head_dim, device="cuda:0", dtype=torch.bfloat16
#     )
#     num_pages_per_seq = (kv_len + page_size - 1) // page_size
#     total_num_pages = num_pages_per_seq * batch_size
#     kv_data = (
#         torch.randn(
#             total_num_pages,
#             2,
#             num_kv_heads,
#             page_size,
#             head_dim,
#             device="cuda:0",
#             dtype=torch.bfloat16,
#         )
#         / 10
#         if kv_layout == "HND"
#         else torch.randn(
#             total_num_pages,
#             2,
#             page_size,
#             num_kv_heads,
#             head_dim,
#             device="cuda:0",
#             dtype=torch.bfloat16,
#         )
#         / 10
#     )
#     kv_indptr = (
#         torch.arange(0, batch_size + 1, device="cuda:0", dtype=torch.int32)
#         * num_pages_per_seq
#     )
#     kv_indices = torch.arange(0, total_num_pages, device="cuda:0", dtype=torch.int32)
#     kv_last_page_len = torch.full(
#         (batch_size,), (kv_len - 1) % page_size + 1, dtype=torch.int32, device="cuda:0"
#     )

#     workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device="cuda:0")

#     wrapper_tcore = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
#         workspace_buffer, kv_layout, use_tensor_cores=True
#     )
#     wrapper_tcore.plan(
#         kv_indptr,
#         kv_indices,
#         kv_last_page_len,
#         num_qo_heads,
#         num_kv_heads,
#         head_dim,
#         page_size,
#         pos_encoding_mode=pos_encoding_mode,
#         data_type=torch.bfloat16,
#         q_data_type=torch.bfloat16,
#         fixed_split_size=fixed_split_size if not disable_split_kv else None,
#         disable_split_kv=disable_split_kv,
#     )
#     o_tensor_cores, lse_tensor_cores = wrapper_tcore.run(q, kv_data, return_lse=True)

#     kv_indptr_invariant = kv_indptr[: invariant_bs + 1]
#     kv_last_page_len_invariant = kv_last_page_len[:invariant_bs]
#     wrapper_tcore.plan(
#         kv_indptr_invariant,
#         kv_indices,
#         kv_last_page_len_invariant,
#         num_qo_heads,
#         num_kv_heads,
#         head_dim,
#         page_size,
#         pos_encoding_mode=pos_encoding_mode,
#         data_type=torch.bfloat16,
#         q_data_type=torch.bfloat16,
#         fixed_split_size=fixed_split_size if not disable_split_kv else None,
#         disable_split_kv=disable_split_kv,
#     )
#     o_invariant, lse_invariant = wrapper_tcore.run(
#         q[:invariant_bs], kv_data, return_lse=True
#     )
#     assert torch.equal(o_tensor_cores[:invariant_bs], o_invariant)
#     assert torch.equal(lse_tensor_cores[:invariant_bs], lse_invariant)


# test that without fixed split size, precision is different
# TODO: this works for the first 29 cases, but then fails with "illegal memory access"..?

# wrapper_tcore.plan(
#     kv_indptr,
#     kv_indices,
#     kv_last_page_len,
#     num_qo_heads,
#     num_kv_heads,
#     head_dim,
#     page_size,
# )
# o_invariant, lse_invariant = wrapper_tcore.run(
#     q[:invariant_bs], kv_data, return_lse=True
# )
# try:
#     torch.testing.assert_close(
#         o_tensor_cores[:invariant_bs], o_invariant, rtol=1e-7, atol=1e-7
#     )
#     torch.testing.assert_close(
#         lse_tensor_cores[:invariant_bs], lse_invariant, rtol=1e-7, atol=1e-7
#     )
# except AssertionError:
#     pass
# else:
#     raise AssertionError("Precision is the same without fixed split size")


@pytest.mark.parametrize("batch_size", [3, 4])
@pytest.mark.parametrize("invariant_bs", [2])
@pytest.mark.parametrize("kv_len", [4096, 5000])
# @pytest.mark.parametrize("qo_len", [128, 256])
@pytest.mark.parametrize("qo_len", [2048])
@pytest.mark.parametrize("fixed_split_size", [4096])
@pytest.mark.parametrize("disable_split_kv", [True, False])
@pytest.mark.parametrize("page_size", [1, 8, 16])
@pytest.mark.parametrize("num_kv_heads", [4])
@pytest.mark.parametrize("group_size", [1, 4])
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
        dtype=torch.bfloat16,
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
            dtype=torch.bfloat16,
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
            dtype=torch.bfloat16,
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
        2048 * 1024 * 1024, dtype=torch.int8, device="cuda:0"
    )

    wrapper_tcore = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer, kv_layout
    )

    def default_plan():
        wrapper_tcore.plan(
            q_indptr,
            kv_indptr,
            kv_indices,
            kv_last_page_len,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            pos_encoding_mode=pos_encoding_mode,
            q_data_type=torch.bfloat16,
            kv_data_type=torch.bfloat16,
            fixed_split_size=fixed_split_size if not disable_split_kv else None,
            disable_split_kv=disable_split_kv,
        )

    def invariant_plan():
        wrapper_tcore.plan(
            q_indptr_invariant,
            kv_indptr_invariant,
            kv_indices,
            kv_last_page_len_invariant,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            pos_encoding_mode=pos_encoding_mode,
            q_data_type=torch.bfloat16,
            kv_data_type=torch.bfloat16,
            fixed_split_size=fixed_split_size if not disable_split_kv else None,
            disable_split_kv=disable_split_kv,
        )

    default_plan()
    o_tensor_cores, lse_tensor_cores = wrapper_tcore.run(q, kv_data, return_lse=True)

    # Test invariant batch size
    q_indptr_invariant = q_indptr[: invariant_bs + 1]
    kv_indptr_invariant = kv_indptr[: invariant_bs + 1]
    kv_last_page_len_invariant = kv_last_page_len[:invariant_bs]

    invariant_plan()
    o_invariant, lse_invariant = wrapper_tcore.run(
        q[: invariant_bs * qo_len], kv_data, return_lse=True
    )

    # Compare outputs for the invariant batch size
    assert torch.equal(o_tensor_cores[: invariant_bs * qo_len], o_invariant)
    assert torch.equal(lse_tensor_cores[: invariant_bs * qo_len], lse_invariant)

    if disable_split_kv:
        # Test cuda graph
        del o_invariant, lse_invariant
        g_invariant = torch.cuda.CUDAGraph()
        invariant_plan()
        with torch.cuda.graph(g_invariant):
            o_invariant, lse_invariant = wrapper_tcore.run(
                q[: invariant_bs * qo_len], kv_data, return_lse=True
            )
        invariant_plan()
        g_invariant.replay()

        # capture for full batch
        del o_tensor_cores, lse_tensor_cores
        g = torch.cuda.CUDAGraph()
        default_plan()
        with torch.cuda.graph(g):
            o_tensor_cores, lse_tensor_cores = wrapper_tcore.run(
                q, kv_data, return_lse=True
            )
        default_plan()
        g.replay()
        # compare outputs
        assert torch.equal(o_invariant, o_tensor_cores[: invariant_bs * qo_len])
        assert torch.equal(lse_invariant, lse_tensor_cores[: invariant_bs * qo_len])

    # Test chunked prefill with 1024 chunk size using invariant batch size
    chunk_size = 1024
    chunked_outputs = []
    chunked_lses = []

    for chunk_start in range(0, qo_len, chunk_size):
        chunk_end = min(chunk_start + chunk_size, qo_len)
        current_chunk_size = chunk_end - chunk_start

        # Create chunked q_indptr for invariant batch size
        q_indptr_chunk = (
            torch.arange(0, invariant_bs + 1, device="cuda:0", dtype=torch.int32)
            * current_chunk_size
        )

        # Extract chunked q data for invariant batch size
        q_chunk = torch.empty(
            invariant_bs * current_chunk_size,
            num_qo_heads,
            head_dim,
            device="cuda:0",
            dtype=torch.bfloat16,
        )
        for i in range(invariant_bs):
            start_idx = i * qo_len + chunk_start
            end_idx = i * qo_len + chunk_end
            chunk_start_idx = i * current_chunk_size
            chunk_end_idx = (i + 1) * current_chunk_size
            q_chunk[chunk_start_idx:chunk_end_idx] = q[start_idx:end_idx]

        # Plan and run for this chunk using invariant batch size
        wrapper_tcore.plan(
            q_indptr_chunk,
            kv_indptr_invariant,
            kv_indices,
            kv_last_page_len_invariant,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            pos_encoding_mode=pos_encoding_mode,
            q_data_type=torch.bfloat16,
            kv_data_type=torch.bfloat16,
            fixed_split_size=fixed_split_size if not disable_split_kv else None,
            disable_split_kv=disable_split_kv,
        )
        o_chunk, lse_chunk = wrapper_tcore.run(q_chunk, kv_data, return_lse=True)

        chunked_outputs.append(o_chunk)
        chunked_lses.append(lse_chunk)

    # Concatenate all chunked results
    o_chunked = torch.cat(chunked_outputs, dim=0)
    lse_chunked = torch.cat(chunked_lses, dim=0)

    # Compare chunked results with invariant results
    assert torch.equal(o_tensor_cores[: invariant_bs * qo_len], o_chunked)
    assert torch.equal(lse_tensor_cores[: invariant_bs * qo_len], lse_chunked)
