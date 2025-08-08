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

import flashinfer
from flashinfer.utils import is_sm90a_supported


@pytest.mark.parametrize("seq_len", [11, 99, 1763, 9999, 32767])
@pytest.mark.parametrize("num_qo_heads", [1, 4, 8])
@pytest.mark.parametrize("num_kv_heads", [1, 4, 8])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
@pytest.mark.parametrize("logits_soft_cap", [0.0, 30.0])
def test_single_prefill(
    seq_len, num_qo_heads, num_kv_heads, causal, head_dim, logits_soft_cap
):
    if not is_sm90a_supported(torch.device("cuda")):
        pytest.skip("SM90A is not supported")

    if num_qo_heads % num_kv_heads != 0:
        pytest.skip("num_qo_heads must be divisible by num_kv_heads")
    torch.random.manual_seed(123)
    q = torch.randn(seq_len, num_qo_heads, head_dim, dtype=torch.half, device="cuda")
    k = torch.randn(seq_len, num_kv_heads, head_dim, dtype=torch.half, device="cuda")
    v = torch.randn(seq_len, num_kv_heads, head_dim, dtype=torch.half, device="cuda")

    o_sm80, lse_sm80 = flashinfer.single_prefill_with_kv_cache_return_lse(
        q,
        k,
        v,
        causal=causal,
        logits_soft_cap=logits_soft_cap,
        backend="fa2",
    )

    o_sm90, lse_sm90 = flashinfer.single_prefill_with_kv_cache_return_lse(
        q, k, v, causal=causal, logits_soft_cap=logits_soft_cap, backend="fa3"
    )
    torch.testing.assert_close(lse_sm80, lse_sm90, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(o_sm80, o_sm90, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("batch_size", [1, 4, 8, 16])
@pytest.mark.parametrize("seq_len", [11, 99, 1763, 9999, 32767])
@pytest.mark.parametrize("num_qo_heads", [1, 4, 8])
@pytest.mark.parametrize("num_kv_heads", [1, 4, 8])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("head_dim", [128])  # [64, 128, 256])
@pytest.mark.parametrize("logits_soft_cap", [0.0, 30.0])
def test_batch_ragged_prefill(
    batch_size, seq_len, num_qo_heads, num_kv_heads, causal, head_dim, logits_soft_cap
):
    if not is_sm90a_supported(torch.device("cuda")):
        pytest.skip("SM90A is not supported")

    if num_qo_heads % num_kv_heads != 0:
        pytest.skip("num_qo_heads must be divisible by num_kv_heads")
    torch.random.manual_seed(42)
    q = torch.randn(
        batch_size * seq_len, num_qo_heads, head_dim, dtype=torch.half, device="cuda"
    )
    k = torch.randn(
        batch_size * seq_len, num_kv_heads, head_dim, dtype=torch.half, device="cuda"
    )
    v = torch.randn(
        batch_size * seq_len, num_kv_heads, head_dim, dtype=torch.half, device="cuda"
    )

    workspace_buffer = torch.empty(
        256 * 1024 * 1024, dtype=torch.uint8, device="cuda:0"
    )

    wrapper_sm80 = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        workspace_buffer, backend="fa2"
    )

    wrapper_sm90 = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        workspace_buffer, backend="fa3"
    )

    qo_indptr = torch.arange(0, batch_size * seq_len + 1, seq_len).int()
    kv_indptr = torch.arange(0, batch_size * seq_len + 1, seq_len).int()

    wrapper_sm80.plan(
        qo_indptr,
        kv_indptr,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        causal=causal,
        logits_soft_cap=logits_soft_cap,
    )
    o_sm80, lse_sm80 = wrapper_sm80.run_return_lse(q, k, v)

    wrapper_sm90.plan(
        qo_indptr,
        kv_indptr,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        causal=causal,
        logits_soft_cap=logits_soft_cap,
    )
    o_sm90, lse_sm90 = wrapper_sm90.run_return_lse(q, k, v)

    torch.testing.assert_close(lse_sm80, lse_sm90, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(o_sm80, o_sm90, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("batch_size", [1, 4, 8, 16])
@pytest.mark.parametrize("seq_len", [11, 99, 1763, 9999, 32767])
@pytest.mark.parametrize("num_heads", [4, 32, 128])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dtype", [torch.half, torch.bfloat16])
def test_deepseek_prefill(
    batch_size,
    seq_len,
    num_heads,
    causal,
    dtype,
):
    if not is_sm90a_supported(torch.device("cuda")):
        pytest.skip("SM90A is not supported")

    if batch_size * seq_len > 131072:
        pytest.skip()
    head_dim_qk = 192
    head_dim_vo = 128
    torch.random.manual_seed(42)
    q = torch.randn(
        batch_size * seq_len, num_heads, head_dim_qk, dtype=dtype, device="cuda"
    )
    k = torch.randn(
        batch_size * seq_len, num_heads, head_dim_qk, dtype=dtype, device="cuda"
    )
    v = torch.randn(
        batch_size * seq_len, num_heads, head_dim_vo, dtype=dtype, device="cuda"
    )

    workspace_buffer = torch.empty(
        256 * 1024 * 1024, dtype=torch.uint8, device="cuda:0"
    )

    wrapper_sm80 = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        workspace_buffer, backend="fa2"
    )

    wrapper_sm90 = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        workspace_buffer, backend="fa3"
    )

    qo_indptr = torch.arange(0, batch_size * seq_len + 1, seq_len).int()
    kv_indptr = torch.arange(0, batch_size * seq_len + 1, seq_len).int()

    wrapper_sm80.plan(
        qo_indptr,
        kv_indptr,
        num_heads,
        num_heads,
        head_dim_qk,
        causal=causal,
        head_dim_vo=head_dim_vo,
        q_data_type=dtype,
        kv_data_type=dtype,
    )
    o_sm80, lse_sm80 = wrapper_sm80.run_return_lse(q, k, v)

    wrapper_sm90.plan(
        qo_indptr,
        kv_indptr,
        num_heads,
        num_heads,
        head_dim_qk,
        causal=causal,
        head_dim_vo=head_dim_vo,
        q_data_type=dtype,
        kv_data_type=dtype,
    )
    o_sm90, lse_sm90 = wrapper_sm90.run_return_lse(q, k, v)

    torch.testing.assert_close(lse_sm80, lse_sm90, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(o_sm80, o_sm90, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("batch_size", [1, 4, 8, 16])
@pytest.mark.parametrize("seq_len", [11, 12, 99, 1763, 9999, 32767])
@pytest.mark.parametrize("page_size", [1])  # [1, 16])
@pytest.mark.parametrize("num_qo_heads", [1, 4, 8])
@pytest.mark.parametrize("num_kv_heads", [1, 4, 8])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
@pytest.mark.parametrize("logits_soft_cap", [0.0, 30.0])
def test_batch_paged_prefill(
    batch_size,
    seq_len,
    page_size,
    num_qo_heads,
    num_kv_heads,
    causal,
    head_dim,
    logits_soft_cap,
):
    if not is_sm90a_supported(torch.device("cuda")):
        pytest.skip("SM90A is not supported")

    if num_qo_heads % num_kv_heads != 0:
        pytest.skip("num_qo_heads must be divisible by num_kv_heads")
    torch.random.manual_seed(42)
    q = torch.randn(
        batch_size * seq_len, num_qo_heads, head_dim, dtype=torch.half, device="cuda"
    )
    num_pages_per_request = (seq_len + page_size - 1) // page_size
    k = torch.randn(
        batch_size * num_pages_per_request,
        page_size,
        num_kv_heads,
        head_dim,
        dtype=torch.half,
        device="cuda",
    )
    v = torch.randn(
        batch_size * num_pages_per_request,
        page_size,
        num_kv_heads,
        head_dim,
        dtype=torch.half,
        device="cuda",
    )

    workspace_buffer = torch.empty(
        256 * 1024 * 1024, dtype=torch.uint8, device="cuda:0"
    )

    wrapper_sm80 = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer, backend="fa2"
    )

    wrapper_sm90 = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer, backend="fa3"
    )

    last_page_len = seq_len - (num_pages_per_request - 1) * page_size
    qo_indptr = torch.arange(0, batch_size * seq_len + 1, seq_len).int()
    kv_indptr = torch.arange(
        0, batch_size * num_pages_per_request + 1, num_pages_per_request
    ).int()
    # NOTE(Zihao): pad 256 elements to avoid out-of-bound because we didn't check the boundary in the kernel
    kv_indices = torch.arange(0, batch_size * num_pages_per_request + 256).int()
    last_page_len = torch.full((batch_size,), last_page_len, dtype=torch.int32)

    wrapper_sm80.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        causal=causal,
        logits_soft_cap=logits_soft_cap,
    )
    o_sm80, lse_sm80 = wrapper_sm80.run_return_lse(q, (k, v))

    wrapper_sm90.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        causal=causal,
        logits_soft_cap=logits_soft_cap,
    )
    o_sm90, lse_sm90 = wrapper_sm90.run_return_lse(q, (k, v))

    torch.testing.assert_close(lse_sm80, lse_sm90, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(o_sm80, o_sm90, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize(
    "kv_len, qo_len, prefix_len_ptr, token_pos_in_items_ptr, token_pos_in_items_len, max_item_len_ptr",
    [
        (54, 37, 17, list(range(17)) + list(range(19)) + [0], 100, [18]),
        (97, 81, 16, list(range(80)) + [0], 97, [79]),
    ],
)
@pytest.mark.parametrize("page_size", [1, 5, 16])
@pytest.mark.parametrize("num_kv_heads", [4])
@pytest.mark.parametrize("num_qo_heads", [4, 32])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("causal", [True])
@pytest.mark.parametrize("kv_layout", ["NHD"])
@pytest.mark.parametrize("logits_soft_cap", [0.0, 30.0])
@pytest.mark.parametrize("return_lse", [True, False])
def test_batch_prefill_with_paged_kv_cache_multi_item_scoring_fa3(
    batch_size,
    kv_len,
    qo_len,
    prefix_len_ptr,
    token_pos_in_items_ptr,
    token_pos_in_items_len,
    max_item_len_ptr,
    page_size,
    num_kv_heads,
    num_qo_heads,
    head_dim,
    causal,
    kv_layout,
    logits_soft_cap,
    return_lse,
):
    q = torch.randn(batch_size * qo_len, num_qo_heads, head_dim).to(0).half()
    q_indptr_cpu = torch.arange(0, batch_size + 1).int() * qo_len
    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size
    kv_data = (
        torch.randn(total_num_pages, 2, num_kv_heads, page_size, head_dim).to(0).half()
        if kv_layout == "HND"
        else torch.randn(total_num_pages, 2, page_size, num_kv_heads, head_dim)
        .to(0)
        .half()
    )
    kv_indptr_cpu = torch.arange(0, batch_size + 1).int() * num_pages_per_seq
    kv_indices_cpu = torch.arange(0, total_num_pages).int()
    kv_last_page_len_cpu = torch.full(
        (batch_size,), (kv_len - 1) % page_size + 1, dtype=torch.int32
    )

    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8).to(0)
    q_indptr_gpu = q_indptr_cpu.to(0)
    kv_indptr_gpu = kv_indptr_cpu.to(0)
    kv_indices_gpu = kv_indices_cpu.to(0)
    kv_last_page_len_gpu = kv_last_page_len_cpu.to(0)

    wrapper_fa2 = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer, kv_layout, backend="fa2"
    )
    wrapper_fa2.plan(
        q_indptr_gpu,
        kv_indptr_gpu,
        kv_indices_gpu,
        kv_last_page_len_gpu,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        causal=causal,
        logits_soft_cap=logits_soft_cap,
        prefix_len_ptr=torch.tensor(prefix_len_ptr).to(dtype=torch.uint32).to(0),
        token_pos_in_items_ptr=torch.tensor(token_pos_in_items_ptr)
        .to(dtype=torch.uint16)
        .to(0),
        token_pos_in_items_len=torch.tensor(token_pos_in_items_len)
        .to(dtype=torch.uint32)
        .to(0),
        max_item_len_ptr=torch.tensor(max_item_len_ptr).to(dtype=torch.uint16).to(0),
    )
    o_fa2, lse_fa2 = wrapper_fa2.run_return_lse(q, kv_data)

    wrapper_fa3 = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer, kv_layout, backend="fa3"
    )
    wrapper_fa3.plan(
        q_indptr_gpu,
        kv_indptr_gpu,
        kv_indices_gpu,
        kv_last_page_len_gpu,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        causal=causal,
        logits_soft_cap=logits_soft_cap,
        prefix_len_ptr=torch.tensor(prefix_len_ptr).to(dtype=torch.uint32).to(0),
        token_pos_in_items_ptr=torch.tensor(token_pos_in_items_ptr)
        .to(dtype=torch.uint16)
        .to(0),
        token_pos_in_items_len=torch.tensor(token_pos_in_items_len)
        .to(dtype=torch.uint32)
        .to(0),
        max_item_len_ptr=torch.tensor(max_item_len_ptr).to(dtype=torch.uint16).to(0),
    )

    o_fa3, lse_fa3 = wrapper_fa3.run_return_lse(q, kv_data)

    torch.testing.assert_close(lse_fa2, lse_fa3, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(o_fa2, o_fa3, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize(
    "kv_len, qo_len, prefix_len_ptr, token_pos_in_items_ptr, token_pos_in_items_len, max_item_len_ptr",
    [
        (
            54,
            37,
            [17, 17],
            list(range(17))
            + list(range(19))
            + [0]
            + [0] * 63
            + list(range(15))
            + list(range(21))
            + [0],
            100,
            [18, 20],
        ),
        (
            97,
            81,
            [16, 16],
            list(range(80)) + [0] + [0] * 16 + list(range(76)) + [0],
            97,
            [79, 75],
        ),
    ],
)
@pytest.mark.parametrize("page_size", [1, 5, 16])
@pytest.mark.parametrize("num_kv_heads", [4])
@pytest.mark.parametrize("num_qo_heads", [4, 32])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("causal", [True])
@pytest.mark.parametrize("kv_layout", ["NHD"])
@pytest.mark.parametrize("logits_soft_cap", [0.0, 30.0])
@pytest.mark.parametrize("return_lse", [True, False])
def test_batch_prefill_with_paged_kv_cache_multi_item_scoring_fa3_bsz2(
    batch_size,
    kv_len,
    qo_len,
    prefix_len_ptr,
    token_pos_in_items_ptr,
    token_pos_in_items_len,
    max_item_len_ptr,
    page_size,
    num_kv_heads,
    num_qo_heads,
    head_dim,
    causal,
    kv_layout,
    logits_soft_cap,
    return_lse,
):
    q = torch.randn(batch_size * qo_len, num_qo_heads, head_dim).to(0).half()
    q_indptr_cpu = torch.arange(0, batch_size + 1).int() * qo_len
    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size
    kv_data = (
        torch.randn(total_num_pages, 2, num_kv_heads, page_size, head_dim).to(0).half()
        if kv_layout == "HND"
        else torch.randn(total_num_pages, 2, page_size, num_kv_heads, head_dim)
        .to(0)
        .half()
    )
    kv_indptr_cpu = torch.arange(0, batch_size + 1).int() * num_pages_per_seq
    kv_indices_cpu = torch.arange(0, total_num_pages).int()
    kv_last_page_len_cpu = torch.full(
        (batch_size,), (kv_len - 1) % page_size + 1, dtype=torch.int32
    )

    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8).to(0)
    q_indptr_gpu = q_indptr_cpu.to(0)
    kv_indptr_gpu = kv_indptr_cpu.to(0)
    kv_indices_gpu = kv_indices_cpu.to(0)
    kv_last_page_len_gpu = kv_last_page_len_cpu.to(0)

    wrapper_fa2 = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer, kv_layout, backend="fa2"
    )
    wrapper_fa2.plan(
        q_indptr_gpu,
        kv_indptr_gpu,
        kv_indices_gpu,
        kv_last_page_len_gpu,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        causal=causal,
        logits_soft_cap=logits_soft_cap,
        prefix_len_ptr=torch.tensor(prefix_len_ptr).to(dtype=torch.uint32).to(0),
        token_pos_in_items_ptr=torch.tensor(token_pos_in_items_ptr)
        .to(dtype=torch.uint16)
        .to(0),
        token_pos_in_items_len=torch.tensor(token_pos_in_items_len)
        .to(dtype=torch.uint32)
        .to(0),
        max_item_len_ptr=torch.tensor(max_item_len_ptr).to(dtype=torch.uint16).to(0),
    )
    o_fa2, lse_fa2 = wrapper_fa2.run_return_lse(q, kv_data)

    wrapper_fa3 = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer, kv_layout, backend="fa3"
    )
    wrapper_fa3.plan(
        q_indptr_gpu,
        kv_indptr_gpu,
        kv_indices_gpu,
        kv_last_page_len_gpu,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        causal=causal,
        logits_soft_cap=logits_soft_cap,
        prefix_len_ptr=torch.tensor(prefix_len_ptr).to(dtype=torch.uint32).to(0),
        token_pos_in_items_ptr=torch.tensor(token_pos_in_items_ptr)
        .to(dtype=torch.uint16)
        .to(0),
        token_pos_in_items_len=torch.tensor(token_pos_in_items_len)
        .to(dtype=torch.uint32)
        .to(0),
        max_item_len_ptr=torch.tensor(max_item_len_ptr).to(dtype=torch.uint16).to(0),
    )

    o_fa3, lse_fa3 = wrapper_fa3.run_return_lse(q, kv_data)

    torch.testing.assert_close(lse_fa2, lse_fa3, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(o_fa2, o_fa3, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    # test_batch_prefill(14, 64, 32, 32, False, 128)
    # test_batch_prefill(1, 32767, 8, 8, True, 128)
    # test_single_prefill(64, 1, 1, False, 256)
    # test_batch_paged_prefill(2, 32768, 1, 1, 1, False, 128)
    test_batch_paged_prefill(16, 32767, 1, 8, 8, True, 128, 0)
