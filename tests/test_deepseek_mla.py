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

import math

import pytest
import torch

import flashinfer


def attention_ref(
    batch_size,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
    sm_scale: float,
) -> torch.Tensor:
    qo_len = q.shape[0] // batch_size
    kv_len = k.shape[0] // batch_size
    num_qo_heads = q.shape[1]
    head_dim_qk = q.shape[2]
    head_dim_vo = v.shape[2]
    logits = (
        torch.einsum(
            "bmhd,bnhd->bhmn",
            q.view(batch_size, qo_len, num_qo_heads, head_dim_qk).float(),
            k.view(batch_size, kv_len, num_qo_heads, head_dim_qk).float(),
        )
        * sm_scale
    )

    if causal:
        mask = (
            torch.arange(kv_len - qo_len, kv_len).unsqueeze(1)
            >= torch.arange(0, kv_len).unsqueeze(0)
        ).to(q.device)
    else:
        mask = torch.ones(qo_len, kv_len).to(q.device)

    logits = logits.masked_fill(mask.unsqueeze(0).unsqueeze(0) == 0, float("-inf"))
    lse_ref = torch.logsumexp(logits, -1).transpose(-1, -2)
    p = torch.softmax(logits, dim=-1)
    o_ref = (
        torch.einsum(
            "bhmn,bnhd->bmhd",
            p,
            v.view(batch_size, kv_len, num_qo_heads, head_dim_vo).float(),
        )
        .contiguous()
        .view(batch_size * qo_len, num_qo_heads, head_dim_vo)
        .to(q)
    )

    return o_ref, lse_ref * math.log2(math.e)


@pytest.mark.parametrize("kv_len", [5532, 7563])
@pytest.mark.parametrize("qo_len", [1832, 3928])
@pytest.mark.parametrize("num_heads", [4, 32, 128])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("backend", ["fa2", "fa3"])
def test_single_prefill_with_kv_cache(
    kv_len,
    qo_len,
    num_heads,
    causal,
    backend,
):
    torch.manual_seed(42)
    head_dim_qk = 192
    head_dim_vo = 128
    q = torch.randn(qo_len, num_heads, head_dim_qk).to(0).half()
    k = torch.randn(kv_len, num_heads, head_dim_qk).to(0).half()
    v = torch.randn(kv_len, num_heads, head_dim_vo).to(0).half()
    o, lse = flashinfer.single_prefill_with_kv_cache(
        q, k, v, causal=causal, backend=backend, return_lse=True
    )
    sm_scale = 1.0 / (head_dim_qk**0.5)

    o_ref, lse_ref = attention_ref(1, q, k, v, causal, sm_scale)
    torch.testing.assert_close(o, o_ref, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(lse, lse_ref.squeeze(0), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("batch_size", [12, 17])
@pytest.mark.parametrize("kv_len", [544, 977])
@pytest.mark.parametrize("qo_len", [377, 177])
@pytest.mark.parametrize("num_heads", [4, 32, 128])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("backend", ["fa2", "fa3"])
def test_batch_prefill_with_ragged_kv_cache(
    batch_size,
    kv_len,
    qo_len,
    num_heads,
    causal,
    backend,
):
    torch.manual_seed(42)
    kv_layout = "NHD"
    head_dim_qk = 192
    head_dim_vo = 128
    q = torch.randn(batch_size * qo_len, num_heads, head_dim_qk).to(0).half()
    q_indptr = torch.arange(0, batch_size + 1).to(0).int() * qo_len

    k = torch.zeros(batch_size * kv_len, num_heads, head_dim_qk).to(0).half()
    v = torch.randn(batch_size * kv_len, num_heads, head_dim_vo).to(0).half()
    kv_indptr = torch.arange(0, batch_size + 1).to(0).int() * kv_len

    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8).to(0)
    wrapper = flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper(
        workspace_buffer, kv_layout, backend=backend
    )
    wrapper.plan(
        q_indptr,
        kv_indptr,
        num_heads,
        num_heads,
        head_dim_qk,
        head_dim_vo=head_dim_vo,
        causal=causal,
    )
    o, lse = wrapper.run_return_lse(q, k, v)

    sm_scale = 1.0 / (head_dim_qk**0.5)
    o_ref, lse_ref = attention_ref(batch_size, q, k, v, causal, sm_scale)

    lse_ref = lse_ref.flatten(0, 1)
    torch.testing.assert_close(o, o_ref, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(lse, lse_ref, rtol=1e-3, atol=1e-3)

    # test with pre-allocated output
    o_buffer = torch.empty_like(o)
    lse_buffer = torch.empty_like(lse)
    wrapper.run(q, k, v, out=o_buffer, lse=lse_buffer)
    torch.testing.assert_close(o, o_buffer, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(lse, lse_buffer, rtol=1e-3, atol=1e-3)

def generate_kv_from_cache(ckv, kpe, kv_len, batch_size, num_heads):
    bs_page_num, page_size, ckv_dim = ckv.shape
    page_num = bs_page_num // batch_size
    _,_,kpe_dim = kpe.shape
    ckv = ckv.view(batch_size, page_num * page_size, ckv_dim)
    kpe = kpe.view(batch_size, page_num * page_size, kpe_dim)
    ckv = ckv[:, :kv_len, :]
    kpe = kpe[:, :kv_len, :]
    k = (torch.cat([ckv, kpe], dim = -1).view(-1, 1, ckv_dim + kpe_dim).repeat_interleave(num_heads, dim = 1))
    v = ckv.repeat_interleave(num_heads, dim = 1)

    return k, v

@pytest.mark.parametrize("batch_size", [1, 17, 37])
@pytest.mark.parametrize("kv_len", [17, 33, 96, 97, 114, 514, 1024])
@pytest.mark.parametrize("qo_len", [1, 17, 37, 77])
@pytest.mark.parametrize("num_heads", [4, 32, 128])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("page_size", [1, 2, 16])
@pytest.mark.parametrize("backend", ["fa2"])
def test_batch_mla_page_attention(
    batch_size,
    kv_len,
    qo_len,
    num_heads,
    causal,
    page_size,
    backend,
):
    if causal and qo_len > kv_len:
        pytest.skip("qo_len > kv_len not supported for causal attention")
    torch.manual_seed(42)
    head_dim_ckv = 512
    head_dim_kpe = 64
    q_nope = torch.randn(
        batch_size * qo_len, num_heads, head_dim_ckv, dtype=torch.half, device="cuda"
    )
    q_pe = torch.randn(
        batch_size * qo_len, num_heads, head_dim_kpe, dtype=torch.half, device="cuda"
    )   
    pages_num = math.ceil(kv_len // page_size)
    ckv = torch.randn(
        batch_size * pages_num,
        page_size,
        head_dim_ckv,
        dtype=torch.half,
        device="cuda",
    )
    kpe = torch.randn(
        batch_size * pages_num,
        page_size,
        head_dim_kpe,
        dtype=torch.half,
        device="cuda",
    )
    sm_scale = 1.0 / ((128 + 64) ** 0.5)  # use head dimension before matrix absorption
    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8).to(0)
    wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(
        workspace_buffer, backend=backend
    )
    q_indptr = torch.arange(0, batch_size + 1).to(0).int() * qo_len
    kv_indptr = torch.arange(0, batch_size + 1).to(0).int() * pages_num
    kv_indices = torch.arange(0, batch_size * pages_num).to(0).int()
    kv_lens = torch.full((batch_size,), kv_len, dtype=torch.int32).to(0)
    wrapper.plan(
        q_indptr,
        kv_indptr,
        kv_indices,
        kv_lens,
        num_heads,
        head_dim_ckv,
        head_dim_kpe,
        page_size,
        causal,
        sm_scale,
        q_nope.dtype,
        ckv.dtype,
    )
    o, lse = wrapper.run(q_nope, q_pe, ckv, kpe, return_lse=True)

    k, v = generate_kv_from_cache(ckv, kpe, kv_len, batch_size, num_heads)

    q = torch.cat([q_nope, q_pe], dim=-1)
    o_ref, lse_ref = attention_ref(batch_size, q, k, v, causal, sm_scale)
    lse_ref = lse_ref.flatten(0, 1)
    torch.testing.assert_close(o, o_ref, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(lse, lse_ref, rtol=1e-3, atol=1e-3)

    # test with pre-allocated output
    o_buffer = torch.empty_like(o)
    lse_buffer = torch.empty_like(lse)
    wrapper.run(q_nope, q_pe, ckv, kpe, out=o_buffer, lse=lse_buffer)
    torch.testing.assert_close(o, o_buffer, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(lse, lse_buffer, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    # test_single_prefill_with_kv_cache(54, 37, 4, 32, False, "fa2")
    # test_batch_prefill_with_ragged_kv_cache(12, 54, 37, 4, 4, False, "fa2")
    # test_batch_mla_page_attention(12, 54, 37, 128, False, "fa2")
    # test_batch_mla_page_attention(1, 320, 1, 1, False, 1, "fa2")
    # test_batch_mla_page_attention(3, 32, 1, 4, False, 1, "fa2")
    # test_batch_mla_page_attention(37, 33, 1, 128, True, 1, "fa2")
    # test_batch_mla_page_attention(33, 2, 1, 1, False, 1, "fa2")
    # test_batch_mla_page_attention(17, 64, 17, 128, False, 1, "fa2")
    test_batch_mla_page_attention(4, 64, 17, 32, False, 1, "fa2")
