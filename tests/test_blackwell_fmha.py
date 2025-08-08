import math

import pytest
import torch
from conftest import VARLEN_INDPTR_PARAMS

import flashinfer
from flashinfer.utils import is_sm100a_supported


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


def attention_varlen_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    causal: bool,
    sm_scale: float,
) -> torch.Tensor:
    batch_size = qo_indptr.shape[0] - 1
    nnz_qo = qo_indptr[-1].item()
    o = torch.empty(nnz_qo, *q.shape[1:-1], v.shape[-1], device=q.device, dtype=q.dtype)
    lse = torch.empty(nnz_qo, q.shape[1], device=q.device, dtype=torch.float32)

    for i in range(batch_size):
        o_i, lse_i = attention_ref(
            1,
            q[qo_indptr[i] : qo_indptr[i + 1]],
            k[kv_indptr[i] : kv_indptr[i + 1]],
            v[kv_indptr[i] : kv_indptr[i + 1]],
            causal,
            sm_scale,
        )

        lse_i = lse_i.flatten(0, 1)
        o[qo_indptr[i] : qo_indptr[i + 1]] = o_i
        lse[qo_indptr[i] : qo_indptr[i + 1]] = lse_i

    return o, lse


@pytest.mark.parametrize("batch_size", [1, 2, 3, 9, 17])
@pytest.mark.parametrize("qo_len", [1, 17, 177, 377, 977])
@pytest.mark.parametrize("kv_len", [1, 17, 544, 977, 1999])
@pytest.mark.parametrize("num_qo_heads", [32])
@pytest.mark.parametrize("num_kv_heads", [8, 32])
@pytest.mark.parametrize("head_dim_qk", [192, 128])
@pytest.mark.parametrize("head_dim_vo", [128])
@pytest.mark.parametrize("sm_scale", [1.0, 1.0 / math.sqrt(192), 1.0 / math.sqrt(128)])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_blackwell_cutlass_fmha(
    batch_size,
    qo_len,
    kv_len,
    num_qo_heads,
    num_kv_heads,
    head_dim_qk,
    head_dim_vo,
    sm_scale,
    causal,
    dtype,
):
    if qo_len > kv_len and causal:
        pytest.skip("qo_len > kv_len and causal is not supported")

    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("SM100A is not supported on this device")
    torch.manual_seed(42)
    q = torch.randn(
        batch_size * qo_len, num_qo_heads, head_dim_qk, dtype=dtype, device="cuda"
    )
    qo_indptr = (
        torch.arange(0, batch_size + 1, device="cuda", dtype=torch.int32) * qo_len
    )
    k = torch.randn(
        batch_size * kv_len, num_kv_heads, head_dim_qk, dtype=dtype, device="cuda"
    )
    v = torch.randn(
        batch_size * kv_len, num_kv_heads, head_dim_vo, dtype=dtype, device="cuda"
    )
    kv_indptr = (
        torch.arange(0, batch_size + 1, device="cuda", dtype=torch.int32) * kv_len
    )

    wrapper = flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper(
        torch.empty(128 * 1024 * 1024, device="cuda", dtype=torch.uint8),
        kv_layout="NHD",
        backend="cutlass",
    )
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        num_qo_heads,
        num_kv_heads,
        head_dim_qk,
        head_dim_vo=head_dim_vo,
        causal=causal,
        sm_scale=sm_scale,
        q_data_type=dtype,
        kv_data_type=dtype,
    )
    o, lse = wrapper.run(q, k, v, return_lse=True)

    gqa_group_ratio = num_qo_heads // num_kv_heads
    k_repeated = torch.repeat_interleave(k, gqa_group_ratio, dim=1)
    v_repeated = torch.repeat_interleave(v, gqa_group_ratio, dim=1)
    o_ref, lse_ref = attention_ref(
        batch_size, q, k_repeated, v_repeated, causal, sm_scale
    )

    lse_ref = lse_ref.flatten(0, 1)
    if dtype == torch.half:
        torch.testing.assert_close(o, o_ref, rtol=1e-3, atol=1e-3)
    else:
        torch.testing.assert_close(o, o_ref, rtol=1e-2, atol=1e-2)

    torch.testing.assert_close(lse, lse_ref, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("indptr", VARLEN_INDPTR_PARAMS)
@pytest.mark.parametrize("num_qo_heads", [32])
@pytest.mark.parametrize("num_kv_heads", [8, 32])
@pytest.mark.parametrize("head_dim_qk", [192, 128])
@pytest.mark.parametrize("head_dim_vo", [128])
@pytest.mark.parametrize("sm_scale", [1.0 / math.sqrt(128)])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_blackwell_cutlass_varlen(
    indptr,
    num_qo_heads,
    num_kv_heads,
    head_dim_qk,
    head_dim_vo,
    sm_scale,
    causal,
    dtype,
):
    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("SM100A is not supported on this device")
    torch.manual_seed(42)
    qkv = torch.randn(
        indptr[-1],
        (
            num_qo_heads * head_dim_qk
            + num_kv_heads * head_dim_qk
            + num_kv_heads * head_dim_vo
        ),
        dtype=dtype,
        device="cuda",
    )
    q = qkv[:, : num_qo_heads * head_dim_qk].view(indptr[-1], num_qo_heads, head_dim_qk)
    k = qkv[
        :,
        num_qo_heads * head_dim_qk : num_qo_heads * head_dim_qk
        + num_kv_heads * head_dim_qk,
    ].view(indptr[-1], num_kv_heads, head_dim_qk)
    v = qkv[:, num_qo_heads * head_dim_qk + num_kv_heads * head_dim_qk :].view(
        indptr[-1], num_kv_heads, head_dim_vo
    )
    qo_indptr = torch.tensor(indptr, device="cuda", dtype=torch.int32)
    kv_indptr = qo_indptr

    wrapper = flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper(
        torch.empty(128 * 1024 * 1024, device="cuda", dtype=torch.uint8),
        kv_layout="NHD",
        backend="cutlass",
    )

    wrapper.plan(
        qo_indptr,
        kv_indptr,
        num_qo_heads,
        num_kv_heads,
        head_dim_qk,
        head_dim_vo=head_dim_vo,
        causal=causal,
        sm_scale=sm_scale,
        q_data_type=dtype,
        kv_data_type=dtype,
    )
    o, lse = wrapper.run(q, k, v, return_lse=True)

    gqa_group_ratio = num_qo_heads // num_kv_heads
    k_repeated = torch.repeat_interleave(k, gqa_group_ratio, dim=1)
    v_repeated = torch.repeat_interleave(v, gqa_group_ratio, dim=1)

    o_ref, lse_ref = attention_varlen_ref(
        q, k_repeated, v_repeated, qo_indptr, kv_indptr, causal, sm_scale
    )

    if dtype == torch.half:
        torch.testing.assert_close(o, o_ref, rtol=1e-3, atol=1e-3)
    else:
        torch.testing.assert_close(o, o_ref, rtol=1e-2, atol=1e-2)

    torch.testing.assert_close(lse, lse_ref, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("qo_indptr_list", [[0, 10, 20, 30, 40, 50, 60, 100]])
@pytest.mark.parametrize("kv_indptr_list", [[0, 50, 50, 50, 50, 50, 50, 50]])
@pytest.mark.parametrize("num_qo_heads", [32])
@pytest.mark.parametrize("num_kv_heads", [8, 32])
@pytest.mark.parametrize("head_dim_qk", [192, 128])
@pytest.mark.parametrize("head_dim_vo", [128])
@pytest.mark.parametrize("sm_scale", [1.0 / math.sqrt(128)])
@pytest.mark.parametrize("dtype", [torch.half, torch.bfloat16])
def test_blackwell_cutlass_qo_kv_varlen(
    qo_indptr_list,
    kv_indptr_list,
    num_qo_heads,
    num_kv_heads,
    head_dim_qk,
    head_dim_vo,
    sm_scale,
    dtype,
):
    causal = False
    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("SM100A is not supported on this device")
    torch.manual_seed(42)
    q = torch.randn(
        qo_indptr_list[-1],
        num_qo_heads,
        head_dim_qk,
        dtype=dtype,
        device="cuda",
    )
    k = torch.randn(
        kv_indptr_list[-1],
        num_kv_heads,
        head_dim_qk,
        dtype=dtype,
        device="cuda",
    )
    v = torch.randn(
        kv_indptr_list[-1],
        num_kv_heads,
        head_dim_vo,
        dtype=dtype,
        device="cuda",
    )

    qo_indptr = torch.tensor(qo_indptr_list, device="cuda", dtype=torch.int32)
    kv_indptr = torch.tensor(kv_indptr_list, device="cuda", dtype=torch.int32)

    wrapper = flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper(
        torch.empty(128 * 1024 * 1024, device="cuda", dtype=torch.uint8),
        kv_layout="NHD",
        backend="cutlass",
    )

    wrapper.plan(
        qo_indptr,
        kv_indptr,
        num_qo_heads,
        num_kv_heads,
        head_dim_qk,
        head_dim_vo=head_dim_vo,
        causal=causal,
        sm_scale=sm_scale,
        q_data_type=dtype,
        kv_data_type=dtype,
    )
    o, lse = wrapper.run(q, k, v, return_lse=True)

    gqa_group_ratio = num_qo_heads // num_kv_heads
    k_repeated = torch.repeat_interleave(k, gqa_group_ratio, dim=1)
    v_repeated = torch.repeat_interleave(v, gqa_group_ratio, dim=1)

    o_ref, lse_ref = attention_varlen_ref(
        q, k_repeated, v_repeated, qo_indptr, kv_indptr, causal, sm_scale
    )

    if dtype == torch.half:
        torch.testing.assert_close(o[10:60], o_ref[10:60], rtol=1e-3, atol=1e-3)
    else:
        torch.testing.assert_close(o[10:60], o_ref[10:60], rtol=1e-2, atol=1e-2)

    torch.testing.assert_close(lse, lse_ref, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    test_blackwell_cutlass_fmha(
        9,
        377,
        977,
        1,
        1,
        192,
        128,
        1,
        False,
        torch.bfloat16,
    )

    test_blackwell_cutlass_varlen(
        [0, 1274, 2568, 3915, 5194, 6498, 7839, 8192],
        32,
        4,
        128,
        128,
        1,
        True,
        torch.bfloat16,
    )

    test_blackwell_cutlass_qo_kv_varlen(
        [0, 10, 20, 30, 40, 50, 60, 100],
        [0, 50, 50, 50, 50, 50, 50, 50],
        32,
        8,
        128,
        128,
        1,
        torch.bfloat16,
    )
