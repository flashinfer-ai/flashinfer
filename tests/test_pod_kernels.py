"""
Copyright (c) 2023 by FlashInfer team.

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
from flashinfer.jit.attention.pytorch import gen_pod_module


@pytest.fixture(autouse=True, scope="module")
def warmup_jit():
    flashinfer.jit.build_jit_specs(
        gen_decode_attention_modules(
            [torch.float16],  # q_dtypes
            [torch.float16],  # kv_dtypes
            [128],  # head_dims
            [0],  # pos_encoding_modes
            [False],  # use_sliding_windows
            [False],  # use_fp16_qk_reductions
        )
        + gen_prefill_attention_modules(
            [torch.float16],  # q_dtypes
            [
                torch.float16,
            ],  # kv_dtypes
            [128],  # head_dims
            [0],  # pos_encoding_modes
            [False],  # use_sliding_windows
            [False],  # use_logits_soft_cap
            [False],  # use_fp16_qk_reductions
        )
        + [
            gen_pod_module(
                torch.float16,  # dtype_q
                torch.float16,  # dtype_kv
                torch.float16,  # dtype_o
                128,  # head_dim
                0,  # pos_encoding_mode_p
                False,  # use_sliding_window_p
                False,  # use_logits_soft_cap_p
                False,  # use_fp16_qk_reduction
                torch.int32,  # dtype_idx
                0,  # pos_encoding_mode_d
                False,  # use_sliding_window_d
                False,  # use_logits_soft_cap_d
            )
        ],
        verbose=False,
    )
    yield


@pytest.mark.parametrize("kv_len_p", [127, 12288])
@pytest.mark.parametrize("qo_len_p", [127, 12288])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("batch_size_d", [1, 17, 127])
@pytest.mark.parametrize("kv_len_d", [127, 12288])
@pytest.mark.parametrize("page_size_d", [1, 16])
@pytest.mark.parametrize("kv_layout_d", ["NHD"])
@pytest.mark.parametrize("num_kv_heads", [8])
@pytest.mark.parametrize("num_qo_heads", [8, 32])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("pos_encoding_mode", ["NONE"])
@pytest.mark.parametrize("q_dtype", [torch.float16])
@pytest.mark.parametrize("kv_dtype", [torch.float16])
@pytest.mark.parametrize("contiguous_kv", [True])
def test_pod_with_paged_kv_cache(
    # Prefill params
    kv_len_p,
    qo_len_p,
    causal,
    # Decode params
    batch_size_d,
    kv_len_d,
    page_size_d,
    kv_layout_d,
    # Shared params
    num_kv_heads,
    num_qo_heads,
    head_dim,
    pos_encoding_mode,
    q_dtype,
    kv_dtype,
    contiguous_kv,
):
    if causal and qo_len_p > kv_len_p:
        pytest.skip("Causal prefill with qo_len_p > kv_len_p is not supported")
    # Prefill inputs
    q_p = torch.randn(
        qo_len_p, num_qo_heads, head_dim, device="cuda:0", dtype=torch.float16
    )
    k_p = torch.randn(
        kv_len_p, num_kv_heads, head_dim, device="cuda:0", dtype=torch.float16
    )
    v_p = torch.randn(
        kv_len_p, num_kv_heads, head_dim, device="cuda:0", dtype=torch.float16
    )
    # Generate prefill reference output
    o_ref_p = flashinfer.prefill.single_prefill_with_kv_cache(
        q_p,
        k_p,
        v_p,
        causal=causal,
        pos_encoding_mode=pos_encoding_mode,
    )
    # Decode inputs
    q_d = torch.randn(
        batch_size_d, num_qo_heads, head_dim, device="cuda:0", dtype=torch.float16
    )
    num_pages_per_seq = (kv_len_d + page_size_d - 1) // page_size_d
    total_num_pages = num_pages_per_seq * batch_size_d
    if kv_layout_d == "HND":
        kv_shape = [total_num_pages, 2, num_kv_heads, page_size_d, head_dim]
    else:
        kv_shape = [total_num_pages, 2, page_size_d, num_kv_heads, head_dim]
    if not contiguous_kv:
        tmp = [kv_shape[0]]
        for v_d in kv_shape[1:]:
            tmp.append(2)
            tmp.append(v_d)
        kv_shape = tmp
        kv_data_fp32 = torch.randn(*kv_shape, device="cuda:0", dtype=torch.float32)
        kv_data = kv_data_fp32.to(kv_dtype)
        kv_data = kv_data[:, 1, :, 1, :, 1, :, 1, :]
        kv_data_fp32 = kv_data_fp32[:, 1, :, 1, :, 1, :, 1, :]
        # actual data is stored in non-contiguous memory
        assert (
            kv_data.stride(-4)
            != kv_data.shape[-3] * kv_data.shape[-2] * kv_data.shape[-1]
        )
    else:
        kv_data_fp32 = torch.randn(*kv_shape, device="cuda:0", dtype=torch.float32)
        kv_data = kv_data_fp32.to(kv_dtype)
    kv_indptr_d = (
        torch.arange(0, batch_size_d + 1, device="cuda:0", dtype=torch.int32)
        * num_pages_per_seq
    )
    kv_indices_d = torch.arange(0, total_num_pages, device="cuda:0", dtype=torch.int32)
    kv_last_page_len = torch.full(
        (batch_size_d,),
        (kv_len_d - 1) % page_size_d + 1,
        device="cuda:0",
        dtype=torch.int32,
    )

    # Generate decode reference output
    decode_workspace_buffer = torch.empty(
        32 * 1024 * 1024, device="cuda:0", dtype=torch.int8
    )
    decode_wrapper = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
        decode_workspace_buffer, kv_layout_d
    )
    decode_wrapper.plan(
        kv_indptr_d,
        kv_indices_d,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size_d,
        pos_encoding_mode=pos_encoding_mode,
        data_type=kv_dtype,
        q_data_type=q_dtype,
    )
    o_ref_d = decode_wrapper.run(q_d, kv_data)

    workspace_buffer = torch.empty(32 * 1024 * 1024, device="cuda:0", dtype=torch.int8)
    pod_wrapper = flashinfer.PODWithPagedKVCacheWrapper(
        workspace_buffer,
        kv_layout_d,
    )
    pod_wrapper.plan(
        kv_indptr_d,
        kv_indices_d,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size_d,
        pos_encoding_mode=pos_encoding_mode,
        data_type=kv_dtype,
        q_data_type=q_dtype,
    )

    o_p, o_d = pod_wrapper.run(
        q_p,
        k_p,
        v_p,
        q_d,
        kv_data,
        pos_encoding_mode_p=pos_encoding_mode,
        causal_p=causal,
    )
    # Prefill is run with batch size 1
    torch.testing.assert_close(
        o_p, o_ref_p, rtol=1e-3, atol=1e-3, msg="Prefill mismatch"
    )
    # Decode uses all batches at once.
    torch.testing.assert_close(
        o_d, o_ref_d, rtol=1e-3, atol=1e-3, msg="Decode mismatch"
    )


if __name__ == "__main__":
    test_pod_with_paged_kv_cache(
        # Prefill params
        128,
        128,
        True,
        # Decode params
        80,
        12288,
        16,
        "NHD",
        # Other shared params
        8,
        8,
        128,
        "NONE",
        torch.float16,
        torch.float16,
        True,
    )
    test_pod_with_paged_kv_cache(
        # Prefill params
        12288,
        12288,
        True,
        # Decode params
        220,
        12288,
        16,
        "NHD",
        # Other shared params
        4,
        16,
        128,
        "NONE",
        torch.float16,
        torch.float16,
        True,
    )
    test_pod_with_paged_kv_cache(
        # Prefill params
        16384,
        16384,
        True,
        # Decode params
        250,
        12288,
        16,
        "NHD",
        # Other shared params
        4,
        16,
        128,
        "NONE",
        torch.float16,
        torch.float16,
        True,
    )
    print("POD test(s) passed!")
