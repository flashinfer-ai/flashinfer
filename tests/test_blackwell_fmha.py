import math

import pytest
import torch
from conftest import VARLEN_INDPTR_PARAMS

import flashinfer
import flashinfer.triton
from flashinfer.utils import is_sm100a_supported

from sink_attention_reference import sink_softmax
from types import SimpleNamespace
import cutlass.cute as cute

from flashinfer.cute_dsl.prefill import BatchPrefillCuteDSLWrapper


def attention_ref(
    batch_size,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
    sm_scale: float,
    sink: torch.Tensor | None = None,
) -> torch.Tensor:
    qo_len = q.shape[0] // batch_size
    kv_len = k.shape[0] // batch_size
    num_qo_heads = q.shape[1]
    num_kv_heads = k.shape[1]
    head_dim_qk = q.shape[2]
    head_dim_vo = v.shape[2]
    
    # Handle GQA: if num_qo_heads > num_kv_heads, repeat K and V to match Q shape
    if num_qo_heads > num_kv_heads:
        assert num_qo_heads % num_kv_heads == 0, f"num_qo_heads ({num_qo_heads}) must be divisible by num_kv_heads ({num_kv_heads}) for GQA"
        group_size = num_qo_heads // num_kv_heads
        k = torch.repeat_interleave(k, group_size, dim=1)  # (batch * kv_len, num_qo_heads, head_dim_qk)
        v = torch.repeat_interleave(v, group_size, dim=1)  # (batch * kv_len, num_qo_heads, head_dim_vo)
    
    # Now all tensors have the same number of heads, use standard attention logic
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
    if sink is not None:
        p = sink_softmax(logits, sink)
    else:
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


def attention_sigmoid_ref(
    batch_size,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
    sigmoid_scale: float,
    sigmoid_bias: float = 0.0,
) -> torch.Tensor:
    qo_len = q.shape[0] // batch_size
    kv_len = k.shape[0] // batch_size
    num_qo_heads = q.shape[1]
    num_kv_heads = k.shape[1]
    head_dim_qk = q.shape[2]
    head_dim_vo = v.shape[2]
    
    # Handle GQA: if num_qo_heads > num_kv_heads, repeat K and V to match Q shape
    if num_qo_heads > num_kv_heads:
        assert num_qo_heads % num_kv_heads == 0, f"num_qo_heads ({num_qo_heads}) must be divisible by num_kv_heads ({num_kv_heads}) for GQA"
        group_size = num_qo_heads // num_kv_heads
        k = torch.repeat_interleave(k, group_size, dim=1)  # (batch * kv_len, num_qo_heads, head_dim_qk)
        v = torch.repeat_interleave(v, group_size, dim=1)  # (batch * kv_len, num_qo_heads, head_dim_vo)
    
    # Now all tensors have the same number of heads, use standard attention logic
    logits = (
        torch.einsum(
            "bmhd,bnhd->bhmn",
            q.view(batch_size, qo_len, num_qo_heads, head_dim_qk).float(),
            k.view(batch_size, kv_len, num_qo_heads, head_dim_qk).float(),
        )
        * sigmoid_scale
    )

    if causal:
        mask = torch.arange(kv_len - qo_len, kv_len, device=q.device).unsqueeze(
            1
        ) >= torch.arange(0, kv_len, device=q.device).unsqueeze(0)
    else:
        mask = torch.ones(qo_len, kv_len, device=q.device)

    logits = logits.masked_fill(mask.unsqueeze(0).unsqueeze(0) == 0, float("-inf"))
    
    # Use sigmoid instead of softmax
    p = torch.sigmoid(logits + sigmoid_bias)
    
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

    return o_ref


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


@pytest.mark.parametrize("batch_size", [1, 9, 17])
@pytest.mark.parametrize("qo_len", [256, 1024])
@pytest.mark.parametrize("kv_len", [256, 1024])
@pytest.mark.parametrize("num_qo_heads", [32])
@pytest.mark.parametrize("num_kv_heads", [8, 32])
@pytest.mark.parametrize("head_dim_qk", [128])
@pytest.mark.parametrize("head_dim_vo", [128])
@pytest.mark.parametrize("sm_scale", [1.0, 1.0 / math.sqrt(128)])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_blackwell_cutedsl_fmha(
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

    wrapper = flashinfer.cute_dsl.prefill.BatchPrefillCuteDSLWrapper(
        torch.empty(128 * 1024 * 1024, device="cuda", dtype=torch.uint8),
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
    o = wrapper.run(q, k, v)
    o_ref, lse_ref = attention_ref(
        batch_size, q, k, v, causal, sm_scale
    )

    if dtype == torch.half:
        torch.testing.assert_close(o, o_ref, rtol=1e-2, atol=1e-2)
    else:
        torch.testing.assert_close(o, o_ref, rtol=1e-2, atol=1e-2)

@pytest.mark.parametrize("indptr", [[0, 256, 1024, 2048, 2560]])
@pytest.mark.parametrize("num_qo_heads", [32])
@pytest.mark.parametrize("num_kv_heads", [8, 32])
@pytest.mark.parametrize("head_dim_qk", [128])
@pytest.mark.parametrize("head_dim_vo", [128])
@pytest.mark.parametrize("sm_scale", [1.0 / math.sqrt(128)])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_blackwell_cutedsl_fmha_varlen(
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

    q = torch.randn(indptr[-1], num_qo_heads, head_dim_qk, dtype=dtype, device="cuda")
    k = torch.randn(indptr[-1], num_kv_heads, head_dim_qk, dtype=dtype, device="cuda")
    v = torch.randn(indptr[-1], num_kv_heads, head_dim_vo, dtype=dtype, device="cuda")

    qo_indptr = torch.tensor(indptr, device="cuda", dtype=torch.int32)
    kv_indptr = qo_indptr

    wrapper = flashinfer.cute_dsl.prefill.BatchPrefillCuteDSLWrapper(
        torch.empty(1, device="cuda", dtype=torch.uint8),
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
    o = wrapper.run(q, k, v)

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



# @pytest.mark.parametrize("batch_size", [1, 2, 3, 9, 17])
# @pytest.mark.parametrize("qo_len", [1, 17, 177, 377, 977])
# @pytest.mark.parametrize("kv_len", [1, 17, 544, 977, 1999])
# @pytest.mark.parametrize("num_qo_heads", [32])
# @pytest.mark.parametrize("num_kv_heads", [8, 32])
# @pytest.mark.parametrize("head_dim_qk", [192, 128])
# @pytest.mark.parametrize("head_dim_vo", [128])
# @pytest.mark.parametrize("causal", [False, True])
# @pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_blackwell_cutedsl_fmha_logits_transform(
    batch_size,
    qo_len,
    kv_len,
    num_qo_heads,
    num_kv_heads,
    head_dim_qk,
    head_dim_vo,
    causal,
    dtype,
):

    params = SimpleNamespace(
        scale=1.0 * math.log2(math.exp(1.0)),
        bias=0.0,
    )
    @cute.jit
    def sigmoid_logits_transform(params, x, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx):
        scale = params.scale
        bias = params.bias
        return cute.arch.rcp_approx(1 + cute.arch.exp2(-(x * scale + bias)))

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

    wrapper = flashinfer.cute_dsl.prefill.BatchPrefillCuteDSLWrapper(
        torch.empty(128 * 1024 * 1024, device="cuda", dtype=torch.uint8),
    )
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        num_qo_heads,
        num_kv_heads,
        head_dim_qk,
        head_dim_vo=head_dim_vo,
        causal=causal,
        sm_scale=1.0,
        q_data_type=dtype,
        kv_data_type=dtype,
        custom_params=params,
        logits_transform=sigmoid_logits_transform,
    )
    o = wrapper.run(q, k, v)

    gqa_group_ratio = num_qo_heads // num_kv_heads
    k_repeat = k.repeat(1, gqa_group_ratio, 1)
    v_repeat = v.repeat(1, gqa_group_ratio, 1)
    
    # Use sigmoid-based attention reference instead of softmax
    o_ref = attention_sigmoid_ref(
        batch_size, q, k_repeat, v_repeat, causal, 1.0, 0.0
    )

    if dtype == torch.half:
        torch.testing.assert_close(o, o_ref, rtol=1e-2, atol=1e-2)
    else:
        torch.testing.assert_close(o, o_ref, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("batch_size", [1, 2, 3, 9, 17])
@pytest.mark.parametrize("qo_len", [1, 17, 177, 377, 977])
@pytest.mark.parametrize("kv_len", [1, 17, 544, 977, 1999])
@pytest.mark.parametrize("num_qo_heads", [32])
@pytest.mark.parametrize("num_kv_heads", [8, 32])
@pytest.mark.parametrize("head_dim_qk", [192, 128])
@pytest.mark.parametrize("head_dim_vo", [128])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_blackwell_cutedsl_fmha_output_transform(
    batch_size,
    qo_len,
    kv_len,
    num_qo_heads,
    num_kv_heads,
    head_dim_qk,
    head_dim_vo,
    causal,
    dtype,
):

    @cute.jit
    def dumb_output_transform(params, output, batch_idx, qo_idx, qo_head_idx, m, rcp_d, scale):
        return output * scale * 2.0 * rcp_d

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

    wrapper = flashinfer.cute_dsl.prefill.BatchPrefillCuteDSLWrapper(
        torch.empty(128 * 1024 * 1024, device="cuda", dtype=torch.uint8),
    )
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        num_qo_heads,
        num_kv_heads,
        head_dim_qk,
        head_dim_vo=head_dim_vo,
        causal=causal,
        sm_scale=1.0,
        q_data_type=dtype,
        kv_data_type=dtype,
        output_transform=dumb_output_transform,
    )
    o = wrapper.run(q, k, v)

    gqa_group_ratio = num_qo_heads // num_kv_heads
    k_repeat = k.repeat(1, gqa_group_ratio, 1)
    v_repeat = v.repeat(1, gqa_group_ratio, 1)
    
    # Use sigmoid-based attention reference instead of softmax
    o_ref, _ = attention_ref(
        batch_size, q, k_repeat, v_repeat, causal, 1.0
    )
    o_ref_transform = o_ref * 2.0

    if dtype == torch.half:
        torch.testing.assert_close(o, o_ref_transform, rtol=1e-2, atol=1e-2)
    else:
        torch.testing.assert_close(o, o_ref_transform, rtol=1e-2, atol=1e-2)

# @pytest.mark.parametrize("batch_size", [1, 2, 3, 9, 17])
# @pytest.mark.parametrize("qo_len", [1, 17, 177, 377, 977])
# @pytest.mark.parametrize("kv_len", [1, 17, 544, 977, 1999])
# @pytest.mark.parametrize("num_qo_heads", [32])
# @pytest.mark.parametrize("num_kv_heads", [8, 32])
# @pytest.mark.parametrize("head_dim_qk", [192, 128])
# @pytest.mark.parametrize("head_dim_vo", [128])
# @pytest.mark.parametrize("causal", [False, True])
# @pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_blackwell_cutedsl_fmha_attention_sink(
    batch_size,
    qo_len,
    kv_len,
    num_qo_heads,
    num_kv_heads,
    head_dim_qk,
    head_dim_vo,
    causal,
    dtype,
):
    import cutlass.cute as cute

    @cute.jit
    def sink_M_D_update(params, kv_tile_idx, qo_head_idx, m, d, scale):
        log_sink = params.sink[qo_head_idx] * math.log2(math.exp(1.0)) if (kv_tile_idx == 0 and qo_head_idx < num_qo_heads) else -math.inf
        m_new = log_sink if log_sink > m else m
        scale = cute.arch.exp2(m - m_new)
        d_new = cute.arch.exp2(log_sink - m_new) + d * scale
        return m_new, d_new
    
    @cute.jit
    def sink_output_transform(params, output, batch_idx, qo_idx, qo_head_idx, m, rcp_d, scale):
        return output * scale * rcp_d

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
    sink = torch.randn((num_qo_heads,), dtype=dtype, device="cuda")

    wrapper = flashinfer.cute_dsl.prefill.BatchPrefillCuteDSLWrapper(
        torch.empty(128 * 1024 * 1024, device="cuda", dtype=torch.uint8),
    )
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        num_qo_heads,
        num_kv_heads,
        head_dim_qk,
        head_dim_vo=head_dim_vo,
        causal=causal,
        sm_scale=1.0,
        q_data_type=dtype,
        kv_data_type=dtype,
        output_transform=sink_output_transform,
        M_D_update=sink_M_D_update,
        use_attention_sink=True,
    )
    o = wrapper.run(q, k, v, sink=sink)
    
    # Use sigmoid-based attention reference instead of softmax
    o_ref, _ = attention_ref(
        batch_size, q, k, v, causal, 1.0, sink=sink
    )

    if dtype == torch.half:
        torch.testing.assert_close(o, o_ref, rtol=1e-2, atol=1e-2)
    else:
        torch.testing.assert_close(o, o_ref, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    test_blackwell_cutedsl_fmha(
        4,
        1024,
        1024,
        32,
        8,
        128,
        128,
        1,
        True,
        torch.bfloat16,
    )
    test_blackwell_cutedsl_fmha_varlen(
        [0, 256, 1024, 2048, 2560],
        32,
        32,
        128,
        128,
        1.0,
        True,
        torch.bfloat16,
    )
    # test_blackwell_cutedsl_fmha_logits_transform(
    #     4,
    #     1024,
    #     1024,
    #     32,
    #     32,
    #     128,
    #     128,
    #     True,
    #     torch.bfloat16,
    # )
    # test_blackwell_cutedsl_fmha_output_transform(
    #     4,
    #     1024,
    #     1024,
    #     32,
    #     32,
    #     128,
    #     128,
    #     True,
    #     torch.bfloat16,
    # )
    # test_blackwell_cutedsl_fmha_attention_sink(
    #     4,
    #     1024,
    #     1024,
    #     32,
    #     8,
    #     128,
    #     128,
    #     True,
    #     torch.bfloat16,
    # )
    # test_blackwell_cutlass_fmha(
    #     9,
    #     377,
    #     977,
    #     1,
    #     1,
    #     192,
    #     128,
    #     1,
    #     False,
    #     torch.bfloat16,
    # )

    # test_blackwell_cutlass_varlen(
    #     [0, 1274, 2568, 3915, 5194, 6498, 7839, 8192],
    #     32,
    #     4,
    #     128,
    #     128,
    #     1,
    #     True,
    #     torch.bfloat16,
    # )

    # test_blackwell_cutlass_qo_kv_varlen(
    #     [0, 10, 20, 30, 40, 50, 60, 100],
    #     [0, 50, 50, 50, 50, 50, 50, 50],
    #     32,
    #     8,
    #     128,
    #     128,
    #     1,
    #     torch.bfloat16,
    # )
