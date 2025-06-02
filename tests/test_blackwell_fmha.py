import math

import pytest
import torch

import flashinfer
import flashinfer.triton
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


@pytest.mark.parametrize("batch_size", [1, 2, 3, 9, 17])
@pytest.mark.parametrize("qo_len", [1, 17, 177, 377, 977])
@pytest.mark.parametrize("kv_len", [1, 17, 544, 977, 1999])
@pytest.mark.parametrize("num_qo_heads", [32])
@pytest.mark.parametrize("num_kv_heads", [4, 32])
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

    # wrapper = flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper(
    #     torch.empty(128 * 1024 * 1024, device="cuda", dtype=torch.uint8),
    #     kv_layout="NHD",
    #     backend="cutlass",
    # )
    # wrapper.plan(
    #     qo_indptr,
    #     kv_indptr,
    #     num_qo_heads,
    #     num_kv_heads,
    #     head_dim_qk,
    #     head_dim_vo=head_dim_vo,
    #     causal=causal,
    #     sm_scale=sm_scale,
    #     q_data_type=dtype,
    #     kv_data_type=dtype,
    # )
    # o, lse = wrapper.run(q, k, v, return_lse=True)

    module = flashinfer.prefill.get_fmha_module(
        q.dtype,
        k.dtype,
        v.dtype,
        torch.int32,
        q.shape[2],
        v.shape[2],
        0,
        False,
        False,
    )
    plan_info = flashinfer.prefill.fmha_varlen_plan(
        module,
        qo_indptr,
        kv_indptr,
        q.shape[1],
        causal,
    )

    o, lse = flashinfer.prefill.fmha_varlen(
        q,
        k,
        v,
        qo_indptr,
        kv_indptr,
        plan_info=plan_info,
        causal=causal,
        sm_scale=sm_scale,
        max_qo_len=qo_len,
    )

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


@pytest.mark.parametrize("indptr", [[0, 1274, 2568, 3915, 5194, 6498, 7839, 8192]])
@pytest.mark.parametrize("num_qo_heads", [32])
@pytest.mark.parametrize("num_kv_heads", [4, 32])
@pytest.mark.parametrize("head_dim_qk", [192, 128])
@pytest.mark.parametrize("head_dim_vo", [128])
@pytest.mark.parametrize("sm_scale", [1.0, 1.0 / math.sqrt(192), 1.0 / math.sqrt(128)])
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
    batch_size = len(indptr) - 1

    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("SM100A is not supported on this device")
    torch.manual_seed(42)
    q = torch.randn(indptr[-1], num_qo_heads, head_dim_qk, dtype=dtype, device="cuda")

    k = torch.randn(indptr[-1], num_kv_heads, head_dim_qk, dtype=dtype, device="cuda")
    v = torch.randn(indptr[-1], num_kv_heads, head_dim_vo, dtype=dtype, device="cuda")
    qo_indptr = torch.tensor(indptr, device="cuda", dtype=torch.int32)
    kv_indptr = qo_indptr

    # wrapper = flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper(
    #     torch.empty(128 * 1024 * 1024, device="cuda", dtype=torch.uint8),
    #     kv_layout="NHD",
    #     backend="cutlass",
    # )
    # wrapper.plan(
    #     qo_indptr,
    #     kv_indptr,
    #     num_qo_heads,
    #     num_kv_heads,
    #     head_dim_qk,
    #     head_dim_vo=head_dim_vo,
    #     causal=causal,
    #     sm_scale=sm_scale,
    #     q_data_type=dtype,
    #     kv_data_type=dtype,
    # )
    # o, lse = wrapper.run(q, k, v, return_lse=True)
    # print(o)
    module = flashinfer.prefill.get_fmha_module(
        q.dtype,
        k.dtype,
        v.dtype,
        torch.int32,
        q.shape[2],
        v.shape[2],
        0,
        False,
        False,
    )
    plan_info = flashinfer.prefill.fmha_varlen_plan(
        module,
        qo_indptr,
        kv_indptr,
        q.shape[1],
        causal,
    )

    o, lse = flashinfer.prefill.fmha_varlen(
        q,
        k,
        v,
        qo_indptr,
        kv_indptr,
        plan_info=plan_info,
        causal=causal,
        sm_scale=sm_scale,
        max_qo_len=max(indptr),
    )

    gqa_group_ratio = num_qo_heads // num_kv_heads
    k_repeated = torch.repeat_interleave(k, gqa_group_ratio, dim=1)
    v_repeated = torch.repeat_interleave(v, gqa_group_ratio, dim=1)

    for i in range(batch_size):
        q_slice = q[indptr[i] : indptr[i + 1]]
        k_slice = k_repeated[indptr[i] : indptr[i + 1]]
        v_slice = v_repeated[indptr[i] : indptr[i + 1]]
        o_ref, lse_ref = attention_ref(1, q_slice, k_slice, v_slice, causal, sm_scale)

        lse_ref = lse_ref.flatten(0, 1)
        if dtype == torch.half:
            torch.testing.assert_close(
                o[indptr[i] : indptr[i + 1]], o_ref, rtol=1e-3, atol=1e-3
            )
        else:
            torch.testing.assert_close(
                o[indptr[i] : indptr[i + 1]], o_ref, rtol=1e-2, atol=1e-2
            )

        torch.testing.assert_close(
            lse[indptr[i] : indptr[i + 1]], lse_ref, rtol=1e-3, atol=1e-3
        )


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
        192,
        128,
        1,
        False,
        torch.bfloat16,
    )
