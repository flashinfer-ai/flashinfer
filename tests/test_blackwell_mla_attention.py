"""
Tests for the modular MLA decode attention kernel.

Ported from test_deepseek_mla.py::test_batch_mla_page_attention_cute_dsl,
importing from the modular attention package.
"""

import math

import pytest
import torch

from flashinfer.utils import is_sm100a_supported


def attention_ref(batch_size, q, k, v, causal, sm_scale):
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
        mask = torch.arange(kv_len - qo_len, kv_len, device=q.device).unsqueeze(
            1
        ) >= torch.arange(0, kv_len, device=q.device).unsqueeze(0)
    else:
        mask = torch.ones(qo_len, kv_len, device=q.device)

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


def generate_kv_from_cache(ckv, kpe, kv_len, batch_size, num_heads):
    bs_page_num, page_size, ckv_dim = ckv.shape
    page_num = bs_page_num // batch_size
    _, _, kpe_dim = kpe.shape
    ckv = ckv.view(batch_size, page_num * page_size, ckv_dim)
    kpe = kpe.view(batch_size, page_num * page_size, kpe_dim)
    ckv = ckv[:, :kv_len, :]
    kpe = kpe[:, :kv_len, :]
    k = (
        torch.cat([ckv, kpe], dim=-1)
        .view(-1, 1, ckv_dim + kpe_dim)
        .repeat_interleave(num_heads, dim=1)
    )
    v = ckv.repeat_interleave(num_heads, dim=1)

    return k, v


@pytest.mark.parametrize("batch_size", [1, 7])
@pytest.mark.parametrize("kv_len", [1, 128, 1024])
@pytest.mark.parametrize("num_heads", [128, 16])
@pytest.mark.parametrize("page_size", [128])
@pytest.mark.parametrize("dtype", [torch.half, torch.bfloat16])
def test_mla_decode(
    batch_size,
    kv_len,
    num_heads,
    page_size,
    dtype,
):
    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("SM100A is required")

    from flashinfer.cute_dsl.attention import BatchMLAPagedAttentionWrapperCuteDSL

    qo_len = 1
    causal = True
    head_dim_ckv = 512
    head_dim_kpe = 64

    torch.manual_seed(42)
    q_nope = torch.randn(
        batch_size * qo_len, num_heads, head_dim_ckv, dtype=dtype, device="cuda"
    )
    q_pe = torch.randn(
        batch_size * qo_len, num_heads, head_dim_kpe, dtype=dtype, device="cuda"
    )
    pages_num = math.ceil(kv_len / page_size)
    ckv = torch.randn(
        batch_size * pages_num,
        page_size,
        head_dim_ckv,
        dtype=dtype,
        device="cuda",
    )
    kpe = torch.randn(
        batch_size * pages_num,
        page_size,
        head_dim_kpe,
        dtype=dtype,
        device="cuda",
    )

    sm_scale = 1.0 / ((128 + 64) ** 0.5)
    workspace_buffer = torch.empty(1, dtype=torch.float32, device="cuda")
    kv_lens = torch.full((batch_size,), kv_len, dtype=torch.int32, device="cuda")

    wrapper = BatchMLAPagedAttentionWrapperCuteDSL(workspace_buffer, split_kv=-1)

    qo_indptr = torch.arange(0, batch_size + 1, dtype=torch.int32, device="cuda")
    kv_indptr = torch.arange(
        0, batch_size + 1, dtype=torch.int32, device="cuda"
    ) * math.ceil(kv_len / page_size)
    kv_indices = torch.arange(
        batch_size * math.ceil(kv_len / page_size), dtype=torch.int32, device="cuda"
    )

    wrapper.plan(
        qo_indptr=qo_indptr,
        kv_indptr=kv_indptr,
        kv_indices=kv_indices,
        kv_len_arr=kv_lens,
        num_heads=num_heads,
        head_dim_ckv=head_dim_ckv,
        head_dim_kpe=head_dim_kpe,
        page_size=page_size,
        causal=causal,
        sm_scale=sm_scale,
        q_data_type=q_nope.dtype,
        kv_data_type=ckv.dtype,
    )

    o, lse = wrapper.run(
        q_nope=q_nope, q_pe=q_pe, ckv_cache=ckv, kpe_cache=kpe, return_lse=True
    )

    o = o.permute(2, 0, 1).contiguous()
    lse = lse.permute(1, 0).contiguous()

    k, v = generate_kv_from_cache(ckv, kpe, kv_len, batch_size, num_heads)
    q = torch.cat([q_nope, q_pe], dim=-1)
    o_ref, lse_ref = attention_ref(batch_size, q, k, v, causal, sm_scale)
    lse_ref = lse_ref.flatten(0, 1)
    torch.testing.assert_close(o, o_ref, rtol=1e-3, atol=1e-3)
    if kv_len != 0:
        torch.testing.assert_close(lse, lse_ref, rtol=1e-3, atol=1e-3)
