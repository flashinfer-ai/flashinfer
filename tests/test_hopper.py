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


@pytest.mark.parametrize("seq_len", [11, 99, 1763, 9999, 32767])
@pytest.mark.parametrize("num_qo_heads", [1, 4, 8])
@pytest.mark.parametrize("num_kv_heads", [1, 4, 8])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
@pytest.mark.parametrize("logits_soft_cap", [0.0, 30.0])
def test_single_prefill(
    seq_len, num_qo_heads, num_kv_heads, causal, head_dim, logits_soft_cap
):
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


if __name__ == "__main__":
    # test_batch_prefill(14, 64, 32, 32, False, 128)
    # test_batch_prefill(1, 32767, 8, 8, True, 128)
    # test_single_prefill(64, 1, 1, False, 256)
    # test_batch_paged_prefill(2, 32768, 1, 1, 1, False, 128)
    test_batch_paged_prefill(16, 32767, 1, 8, 8, True, 128, 0)
