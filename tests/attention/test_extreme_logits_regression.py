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

import math

import pytest
import torch

import flashinfer

# Regression tests for the finite mask sentinel bugs #4267/#4450/#4451/#4452.
#
# Masked logits are represented as IEEE -inf, so a legitimate logit of any
# magnitude must still win over a masked position. The cases below cover:
# 1. valid rows whose every logit lies below the historical -5e4 sentinel,
# 2. a masked key that would dominate every valid key if masking leaked,
# 3. a one-key ragged row with an extreme negative logit (#4451),
# 4. fully masked rows of a bottom-right aligned causal prefill (#4452),
# 5. paged decode with extreme negative logits (#4450).
#
# Note: the FA2 running maximum is tracked over the *unscaled* QK dot product,
# so the target logits below refer to raw dot products; sm_scale is applied
# inside the exponentials and to the final LSE.


def ref_single_prefill(q, k, v, causal=False):
    """FP64 reference of single-request prefill attention (NHD layout).

    Returns (output, lse) where lse is the base-2 logsumexp; fully masked rows
    get lse = -inf and zero output.
    """
    qo_len, num_qo_heads, head_dim = q.shape
    kv_len, num_kv_heads, _ = k.shape
    group_size = num_qo_heads // num_kv_heads
    scale = head_dim**-0.5

    q64 = q.float().to(torch.float64)
    k64 = k.float().to(torch.float64).repeat_interleave(group_size, dim=1)
    v64 = v.float().to(torch.float64).repeat_interleave(group_size, dim=1)

    scores = torch.einsum("qhd,khd->hqk", q64, k64) * scale
    if causal:
        q_pos = torch.arange(qo_len, dtype=torch.float64, device=q.device)
        k_pos = torch.arange(kv_len, dtype=torch.float64, device=q.device)
        mask = k_pos[None, :] - (kv_len - qo_len) > q_pos[:, None]
        scores = scores.masked_fill(mask[None, :, :], float("-inf"))

    row_max = scores.amax(dim=-1)
    weights = torch.softmax(scores, dim=-1)
    out = torch.einsum("hqk,khd->qhd", weights.to(torch.float64), v64)
    sum_exp = torch.exp(scores - row_max[..., None]).sum(dim=-1)
    lse = row_max + torch.log2(sum_exp)
    lse = torch.where(
        torch.isneginf(row_max), torch.full_like(row_max, float("-inf")), lse
    )
    return out.to(q.dtype), lse.t().to(torch.float32)


def _fill_with_max_logit(target_logit, dtype, kv_len, head_dim):
    """Q/K whose raw QK dot product is target_logit for every position.

    K[0] produces a slightly larger logit so the row maximum is unique.
    """
    q = torch.ones(1, 1, head_dim, dtype=dtype, device="cuda")
    k = torch.full(
        (kv_len, 1, head_dim), target_logit / head_dim, dtype=dtype, device="cuda"
    )
    k[0] = (target_logit + 1024.0) / head_dim
    return q, k


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "target_logit", [-1e4, -5e4, -5.5e4, -6e4, -1e5, -2e5, -2.45e5]
)
def test_single_prefill_extreme_negative_logits(dtype, target_logit):
    torch.manual_seed(0)
    num_heads, head_dim, kv_len = 16, 128, 64
    q, k = _fill_with_max_logit(target_logit, dtype, kv_len, head_dim)
    q = q.repeat(1, num_heads, 1)
    v = torch.randn(kv_len, 1, head_dim, dtype=dtype, device="cuda")

    o = flashinfer.single_prefill_with_kv_cache(q, k, v, backend="fa2")
    o_ref, _ = ref_single_prefill(q, k, v)

    # The pre-fix bug produced an all-zero output once every logit fell below
    # the -5e4 sentinel; guard against that in addition to the value check.
    assert o.float().norm() > 0
    torch.testing.assert_close(o, o_ref, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_single_prefill_masked_key_never_dominates(dtype):
    torch.manual_seed(0)
    qo_len, num_heads, head_dim = 32, 4, 128
    # Every valid key yields a mildly negative logit; masked (future) keys are
    # given a much larger logit so a leaking mask would change the output.
    q = torch.ones(qo_len, num_heads, head_dim, dtype=dtype, device="cuda") * (
        8.0 / head_dim
    )
    k = torch.full((qo_len, 1, head_dim), -8.0, dtype=dtype, device="cuda")
    k[qo_len // 2 :] = 64.0
    v = torch.randn(qo_len, 1, head_dim, dtype=dtype, device="cuda")

    o = flashinfer.single_prefill_with_kv_cache(q, k, v, causal=True, backend="fa2")
    o_ref, _ = ref_single_prefill(q, k, v, causal=True)
    torch.testing.assert_close(o, o_ref, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_ragged_prefill_one_valid_key(dtype):
    # Deterministic #4451 fixture: one valid causal key with raw score
    # -524288 (scaled -46341). Softmax over one key is exactly 1, so the
    # output must equal V regardless of the score magnitude.
    num_qo_heads, num_kv_heads, head_dim = 32, 8, 128
    sm_scale = 1.0 / math.sqrt(head_dim)
    q = torch.full((1, num_qo_heads, head_dim), 64.0, dtype=dtype, device="cuda")
    k = torch.full((1, num_kv_heads, head_dim), -64.0, dtype=dtype, device="cuda")
    v = torch.ones(1, num_kv_heads, head_dim, dtype=dtype, device="cuda")

    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        workspace, kv_layout="NHD", backend="fa2"
    )
    wrapper.plan(
        qo_indptr=torch.tensor([0, 1], dtype=torch.int32, device="cuda"),
        kv_indptr=torch.tensor([0, 1], dtype=torch.int32, device="cuda"),
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim,
        causal=True,
        sm_scale=sm_scale,
        q_data_type=dtype,
        kv_data_type=dtype,
    )
    o, lse = wrapper.run(q, k, v, return_lse=True)

    k_gqa = k.repeat_interleave(num_qo_heads // num_kv_heads, dim=1)
    ref_lse = (q.float() * k_gqa.float()).sum(-1) * sm_scale / math.log(2.0)
    assert not o.isnan().any() and not lse.isnan().any()
    torch.testing.assert_close(
        o, v.repeat_interleave(num_qo_heads // num_kv_heads, dim=1), rtol=0, atol=0
    )
    torch.testing.assert_close(lse, ref_lse, rtol=1e-5, atol=1e-3)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_paged_prefill_fully_masked_rows(dtype):
    # Deterministic #4452 fixture: bottom-right causal alignment with
    # qo_len=34 > kv_len=1 leaves 33 rows with no attendable key. Fully
    # masked rows must produce zero output and LSE=-inf, while the final
    # row still attends the single key.
    qo_len, kv_len = 34, 1
    num_qo_heads, num_kv_heads, head_dim = 32, 8, 128
    page_size, num_pages = 1, 2
    q = torch.zeros(qo_len, num_qo_heads, head_dim, dtype=dtype, device="cuda")
    k_cache = torch.zeros(
        num_pages, page_size, num_kv_heads, head_dim, dtype=dtype, device="cuda"
    )
    v_cache = torch.ones_like(k_cache)

    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace, kv_layout="NHD", backend="fa2"
    )
    wrapper.plan(
        qo_indptr=torch.tensor([0, qo_len], dtype=torch.int32, device="cuda"),
        paged_kv_indptr=torch.tensor([0, 1], dtype=torch.int32, device="cuda"),
        paged_kv_indices=torch.tensor([1], dtype=torch.int32, device="cuda"),
        paged_kv_last_page_len=torch.tensor([1], dtype=torch.int32, device="cuda"),
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim,
        page_size=page_size,
        causal=True,
        sm_scale=1.0 / math.sqrt(head_dim),
        q_data_type=dtype,
        kv_data_type=dtype,
    )
    o, lse = wrapper.run(q, (k_cache, v_cache), return_lse=True)

    num_masked = qo_len - kv_len
    assert not o.isnan().any() and not lse.isnan().any()
    torch.testing.assert_close(
        o[:num_masked], torch.zeros_like(o[:num_masked]), rtol=0, atol=0
    )
    assert torch.isneginf(lse[:num_masked]).all()
    torch.testing.assert_close(
        o[num_masked:], torch.ones_like(o[num_masked:]), rtol=0, atol=0
    )
    torch.testing.assert_close(
        lse[num_masked:], torch.zeros_like(lse[num_masked:]), rtol=0, atol=0
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_paged_prefill_split_kv_empty_chunk(dtype):
    # Multi-token causal prefill (qo_len=2, kv_len=129) with split-KV: the
    # first token's rows have no attendable key in the last kv chunk, so the
    # merge path must combine a finite partial state with an empty one
    # (partial lse=-inf, d=0) without producing NaN.
    bs, qo_len, kv_len = 1, 2, 129
    num_qo_heads, num_kv_heads, head_dim = 8, 2, 128
    page_size = 16
    pages_per = (kv_len + page_size - 1) // page_size
    q = (
        torch.randn(bs * qo_len, num_qo_heads, head_dim, dtype=dtype, device="cuda")
        / 10
    )
    kv_data = (
        torch.randn(
            pages_per, 2, page_size, num_kv_heads, head_dim, dtype=dtype, device="cuda"
        )
        / 10
    )
    qo_indptr = torch.tensor([0, bs * qo_len], dtype=torch.int32, device="cuda")

    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace, kv_layout="NHD", backend="fa2"
    )
    wrapper.plan(
        qo_indptr=qo_indptr,
        paged_kv_indptr=torch.tensor([0, pages_per], dtype=torch.int32, device="cuda"),
        paged_kv_indices=torch.arange(pages_per, dtype=torch.int32, device="cuda"),
        paged_kv_last_page_len=torch.tensor(
            [(kv_len - 1) % page_size + 1], dtype=torch.int32, device="cuda"
        ),
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim,
        page_size=page_size,
        causal=True,
        q_data_type=dtype,
        kv_data_type=dtype,
    )
    o, lse = wrapper.run(q, kv_data, return_lse=True)

    k = kv_data[:, 0].reshape(-1, num_kv_heads, head_dim)
    v = kv_data[:, 1].reshape(-1, num_kv_heads, head_dim)
    o_ref, lse_ref = ref_single_prefill(q, k[:kv_len], v[:kv_len], causal=True)
    assert not o.isnan().any() and not lse.isnan().any()
    torch.testing.assert_close(o, o_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(lse, lse_ref, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_paged_decode_extreme_negative_logits(dtype):
    # Deterministic #4450 fixture: classic (non-tensor-core) decode with two
    # valid keys whose raw scores (-524288) sit far below the historical -5e4
    # sentinel.
    num_qo_heads, num_kv_heads, head_dim = 32, 4, 128
    page_size, num_pages = 1, 17
    sm_scale = 1.0 / math.sqrt(head_dim)
    q = torch.full((1, num_qo_heads, head_dim), 64.0, dtype=dtype, device="cuda")
    k_cache = torch.full(
        (num_pages, page_size, num_kv_heads, head_dim),
        -64.0,
        dtype=dtype,
        device="cuda",
    )
    v_cache = torch.ones_like(k_cache)

    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace, kv_layout="NHD", use_tensor_cores=False
    )
    wrapper.plan(
        indptr=torch.tensor([0, 2], dtype=torch.int32, device="cuda"),
        indices=torch.tensor([15, 16], dtype=torch.int32, device="cuda"),
        last_page_len=torch.tensor([1], dtype=torch.int32, device="cuda"),
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        page_size=page_size,
        pos_encoding_mode="NONE",
        sm_scale=sm_scale,
        q_data_type=dtype,
        kv_data_type=dtype,
    )
    o, lse = wrapper.run(q, (k_cache, v_cache), return_lse=True)

    k_gqa = k_cache[[15, 16], 0].repeat_interleave(num_qo_heads // num_kv_heads, dim=1)
    logits = torch.einsum("hd,khd->hk", q[0].float(), k_gqa.float()) * sm_scale
    ref_lse = (torch.logsumexp(logits, dim=-1) / math.log(2.0)).unsqueeze(0)
    assert not o.isnan().any() and not lse.isnan().any()
    torch.testing.assert_close(o, torch.ones_like(o), rtol=0, atol=0)
    torch.testing.assert_close(lse, ref_lse, rtol=1e-5, atol=1e-3)


if __name__ == "__main__":
    test_single_prefill_extreme_negative_logits(torch.float16, -6e4)
    test_single_prefill_extreme_negative_logits(torch.bfloat16, -2.45e5)
    test_single_prefill_masked_key_never_dominates(torch.bfloat16)
    test_ragged_prefill_one_valid_key(torch.bfloat16)
    test_paged_prefill_fully_masked_rows(torch.bfloat16)
    test_paged_prefill_split_kv_empty_chunk(torch.bfloat16)
    test_paged_decode_extreme_negative_logits(torch.bfloat16)
