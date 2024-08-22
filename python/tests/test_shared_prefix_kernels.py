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

import numpy
import pytest
import torch

import flashinfer


def ceil_div(a, b):
    return (a + b - 1) // b


@pytest.mark.parametrize("stage", ["decode", "append"])
@pytest.mark.parametrize("batch_size", [12, 17])
@pytest.mark.parametrize("unique_kv_len", [37, 17])
@pytest.mark.parametrize("shared_kv_len", [128, 512, 2048])
@pytest.mark.parametrize("num_heads", [8, 16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("page_size", [1, 16])
def test_batch_attention_with_shared_prefix_paged_kv_cache(
    stage,
    batch_size,
    unique_kv_len,
    shared_kv_len,
    num_heads,
    causal,
    head_dim,
    page_size,
):
    if stage == "decode" and causal == True:
        pytest.skip("Causal attention is not required in decode stage")
    assert shared_kv_len % page_size == 0
    kv_layout = "NHD"
    if stage == "append":
        q = torch.randn(batch_size * unique_kv_len, num_heads, head_dim).to(0).half()
        q_indptr = torch.arange(0, batch_size + 1).to(0).int() * unique_kv_len
    else:
        q = torch.randn(batch_size, num_heads, head_dim).to(0).half()
        q_indptr = torch.arange(0, batch_size + 1).to(0).int()
    k_shared = torch.randn(shared_kv_len, num_heads, head_dim).to(0).half()
    v_shared = torch.randn(shared_kv_len, num_heads, head_dim).to(0).half()
    k_unique = torch.randn(batch_size * unique_kv_len, num_heads, head_dim).to(0).half()
    v_unique = torch.randn(batch_size * unique_kv_len, num_heads, head_dim).to(0).half()

    kv_data = (
        torch.zeros(
            ceil_div(shared_kv_len, page_size)
            + batch_size * ceil_div(unique_kv_len, page_size),
            2,
            page_size,
            num_heads,
            head_dim,
        )
        .to(0)
        .half()
    )
    shared_kv_indices = torch.arange(0, ceil_div(shared_kv_len, page_size)).to(0).int()
    shared_append_indptr = torch.arange(0, 2).to(0).int() * shared_kv_len
    shared_kv_indptr = torch.arange(0, 2).to(0).int() * ceil_div(
        shared_kv_len, page_size
    )
    shared_last_page_len = torch.full(
        (1,), (shared_kv_len - 1) % page_size + 1, dtype=torch.int32
    ).to(0)
    flashinfer.append_paged_kv_cache(
        k_shared,
        v_shared,
        shared_append_indptr,
        kv_data,
        shared_kv_indices,
        shared_kv_indptr,
        shared_last_page_len,
        kv_layout,
    )
    unique_kv_indices = torch.arange(
        0, batch_size * ceil_div(unique_kv_len, page_size)
    ).to(0).int() + ceil_div(shared_kv_len, page_size)
    unique_append_indptr = torch.arange(0, batch_size + 1).to(0).int() * unique_kv_len
    unique_kv_indptr = torch.arange(0, batch_size + 1).to(0).int() * ceil_div(
        unique_kv_len, page_size
    )
    unique_last_page_len = torch.full(
        (batch_size,), (unique_kv_len - 1) % page_size + 1, dtype=torch.int32
    ).to(0)
    flashinfer.append_paged_kv_cache(
        k_unique,
        v_unique,
        unique_append_indptr,
        kv_data,
        unique_kv_indices,
        unique_kv_indptr,
        unique_last_page_len,
        kv_layout,
    )

    if stage == "decode":
        multi_level_wrapper = flashinfer.MultiLevelCascadeAttentionWrapper(
            2, torch.empty(32 * 1024 * 1024, dtype=torch.int8).to(0), kv_layout
        )
        shared_prefix_decode_wrapper = (
            flashinfer.BatchDecodeWithSharedPrefixPagedKVCacheWrapper(
                torch.empty(32 * 1024 * 1024, dtype=torch.int8).to(0), kv_layout
            )
        )
    else:
        multi_level_wrapper = flashinfer.MultiLevelCascadeAttentionWrapper(
            2, torch.empty(32 * 1024 * 1024, dtype=torch.int8).to(0), kv_layout
        )
        shared_prefix_prefill_wrapper = (
            flashinfer.BatchPrefillWithSharedPrefixPagedKVCacheWrapper(
                torch.empty(32 * 1024 * 1024, dtype=torch.int8).to(0), kv_layout
            )
        )

    qo_indptr_top = torch.tensor([0, q.shape[0]], dtype=torch.int32).to(0)
    if stage == "decode":
        qo_indptr_bottom = torch.arange(0, batch_size + 1).to(0)
        multi_level_wrapper.begin_forward(
            [qo_indptr_top, qo_indptr_bottom],
            [shared_kv_indptr, unique_kv_indptr],
            [shared_kv_indices, unique_kv_indices],
            [shared_last_page_len, unique_last_page_len],
            num_heads,
            num_heads,
            head_dim,
            page_size,
        )
        o_multi_level = multi_level_wrapper.forward(q, kv_data)
    else:
        qo_indptr_bottom = torch.arange(0, batch_size + 1).to(0) * unique_kv_len
        multi_level_wrapper.begin_forward(
            [qo_indptr_top, qo_indptr_bottom],
            [shared_kv_indptr, unique_kv_indptr],
            [shared_kv_indices, unique_kv_indices],
            [shared_last_page_len, unique_last_page_len],
            num_heads,
            num_heads,
            head_dim,
            page_size,
        )
        o_multi_level = multi_level_wrapper.forward(q, kv_data, causal=causal)

    if stage == "decode":
        shared_prefix_decode_wrapper.begin_forward(
            unique_kv_indptr,
            unique_kv_indices,
            unique_last_page_len,
            num_heads,
            num_heads,
            head_dim,
            page_size,
        )
        o_two_level = shared_prefix_decode_wrapper.forward(
            q, k_shared, v_shared, kv_data
        )
    else:
        shared_prefix_prefill_wrapper.begin_forward(
            q_indptr,
            unique_kv_indptr,
            unique_kv_indices,
            unique_last_page_len,
            num_heads,
            num_heads,
            head_dim,
            page_size,
        )
        o_two_level = shared_prefix_prefill_wrapper.forward(
            q, k_shared, v_shared, kv_data, causal=causal
        )

    numpy.testing.assert_allclose(
        o_multi_level.cpu().numpy(), o_two_level.cpu().numpy(), rtol=1e-3, atol=1e-3
    )


@pytest.mark.parametrize("seed", [0])
@pytest.mark.parametrize("num_tries", [50])
def test_merge_state_in_place_with_mask(seed, num_tries):
    seq_len = 512
    num_heads = 32
    head_dim = 128
    va = torch.randn(seq_len, num_heads, head_dim).half().to("cuda:0")
    sa = torch.randn(seq_len, num_heads, dtype=torch.float32).to("cuda:0")
    vb = torch.randn(seq_len, num_heads, head_dim).half().to("cuda:0")
    sb = torch.randn(seq_len, num_heads, dtype=torch.float32).to("cuda:0")
    va_orginal = va.clone()
    sa_original = sa.clone()

    # No mask.
    flashinfer.merge_state_in_place(va, sa, vb, sb)
    va_merged_ref = va.clone()
    sa_merged_ref = sa.clone()
    assert not torch.allclose(va_merged_ref, va_orginal)
    assert not torch.allclose(sa_merged_ref, sa_original)

    # Mask with all 1s. Should be identical to no mask.
    mask = torch.ones(seq_len, dtype=torch.bool).to("cuda:0")
    va = va_orginal.clone()
    sa = sa_original.clone()
    flashinfer.merge_state_in_place(va, sa, vb, sb, mask=mask)
    va_merged = va
    sa_merged = sa
    numpy.testing.assert_allclose(
        va_merged.cpu().numpy(), va_merged_ref.cpu().numpy(), rtol=1e-3, atol=1e-3
    )
    numpy.testing.assert_allclose(
        sa_merged.cpu().numpy(), sa_merged_ref.cpu().numpy(), rtol=1e-3, atol=1e-3
    )

    # Mask with all zeros. Input and output should be identical.
    mask = torch.zeros(seq_len, dtype=torch.bool).to("cuda:0")
    va = va_orginal.clone()
    sa = sa_original.clone()
    flashinfer.merge_state_in_place(va, sa, vb, sb, mask=mask)
    va_merged = va
    sa_merged = sa
    numpy.testing.assert_allclose(
        va_merged.cpu().numpy(), va_orginal.cpu().numpy(), rtol=1e-3, atol=1e-3
    )
    numpy.testing.assert_allclose(
        sa_merged.cpu().numpy(), sa_original.cpu().numpy(), rtol=1e-3, atol=1e-3
    )

    # Test some random masks.
    randgen = torch.Generator(device="cuda:0")
    randgen.manual_seed(seed)
    for _ in range(num_tries):
        rand_mask = (
            torch.rand(seq_len, generator=randgen, dtype=torch.float32, device="cuda:0")
            > 0.5
        ).to(dtype=torch.bool)
        true_indices = rand_mask.nonzero()
        false_indices = (rand_mask == 0).nonzero()
        va = va_orginal.clone()
        sa = sa_original.clone()
        flashinfer.merge_state_in_place(va, sa, vb, sb, mask=rand_mask)
        va_merged = va
        sa_merged = sa

        numpy.testing.assert_allclose(
            va_merged[false_indices].cpu().numpy(),
            va_orginal[false_indices].cpu().numpy(),
            rtol=1e-3,
            atol=1e-3,
        )
        numpy.testing.assert_allclose(
            sa_merged[false_indices].cpu().numpy(),
            sa_original[false_indices].cpu().numpy(),
            rtol=1e-3,
            atol=1e-3,
        )
        numpy.testing.assert_allclose(
            va_merged[true_indices].cpu().numpy(),
            va_merged_ref[true_indices].cpu().numpy(),
            rtol=1e-3,
            atol=1e-3,
        )
        numpy.testing.assert_allclose(
            sa_merged[true_indices].cpu().numpy(),
            sa_merged_ref[true_indices].cpu().numpy(),
            rtol=1e-3,
            atol=1e-3,
        )


if __name__ == "__main__":
    test_batch_attention_with_shared_prefix_paged_kv_cache(
        "decode", 12, 37, 128, 8, False, 128, 16
    )
    test_batch_attention_with_shared_prefix_paged_kv_cache(
        "apppend", 12, 37, 128, 8, True, 128, 16
    )
