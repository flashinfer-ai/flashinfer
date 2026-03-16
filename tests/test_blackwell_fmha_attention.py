"""Tests for the refactored flashinfer.cute_dsl.attention package.

Mirrors the CuTe DSL tests from test_blackwell_fmha.py but imports from
the new modular attention/ package instead of the monolithic prefill.py.
"""

import math

import pytest
import torch
from tests.test_helpers.params import VARLEN_INDPTR_PARAMS

from flashinfer.utils import is_sm100a_supported

from tests.test_helpers.sink_attention_reference import sink_softmax
from types import SimpleNamespace
import cutlass.cute as cute

# Key change: import from the new attention package
from flashinfer.cute_dsl.attention import BatchPrefillCuteDSLWrapper


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

    if num_qo_heads > num_kv_heads:
        assert num_qo_heads % num_kv_heads == 0, (
            f"num_qo_heads ({num_qo_heads}) must be divisible by num_kv_heads ({num_kv_heads}) for GQA"
        )
        group_size = num_qo_heads // num_kv_heads
        k = torch.repeat_interleave(k, group_size, dim=1)
        v = torch.repeat_interleave(v, group_size, dim=1)

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


@pytest.mark.parametrize("batch_size", [1, 9, 17])
@pytest.mark.parametrize("qo_len", [256, 1024])
@pytest.mark.parametrize("num_qo_heads", [32])
@pytest.mark.parametrize("num_kv_heads", [8, 32])
@pytest.mark.parametrize("head_dim_qk", [128])
@pytest.mark.parametrize("head_dim_vo", [128])
@pytest.mark.parametrize("sm_scale", [1.0, 1.0 / math.sqrt(128)])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_attention_prefill(
    batch_size,
    qo_len,
    num_qo_heads,
    num_kv_heads,
    head_dim_qk,
    head_dim_vo,
    sm_scale,
    causal,
    dtype,
):
    kv_len = qo_len

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

    wrapper = BatchPrefillCuteDSLWrapper(
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
    o_ref, lse_ref = attention_ref(batch_size, q, k, v, causal, sm_scale)

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
def test_attention_prefill_varlen(
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

    wrapper = BatchPrefillCuteDSLWrapper(
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


@pytest.mark.parametrize("batch_size", [1, 2, 3, 9, 17])
@pytest.mark.parametrize("qo_len", [1, 17, 177, 377, 977])
@pytest.mark.parametrize("num_qo_heads", [32])
@pytest.mark.parametrize("num_kv_heads", [8, 32])
@pytest.mark.parametrize("head_dim_qk", [192, 128])
@pytest.mark.parametrize("head_dim_vo", [128])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_attention_prefill_output_transform(
    batch_size,
    qo_len,
    num_qo_heads,
    num_kv_heads,
    head_dim_qk,
    head_dim_vo,
    causal,
    dtype,
):
    kv_len = qo_len

    @cute.jit
    def dumb_output_transform(
        params, output, batch_idx, qo_idx, qo_head_idx, m, rcp_d, scale
    ):
        return output * scale * 2.0 * rcp_d

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

    wrapper = BatchPrefillCuteDSLWrapper(
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

    o_ref, _ = attention_ref(batch_size, q, k_repeat, v_repeat, causal, 1.0)
    o_ref_transform = o_ref * 2.0

    if dtype == torch.half:
        torch.testing.assert_close(o, o_ref_transform, rtol=1e-2, atol=1e-2)
    else:
        torch.testing.assert_close(o, o_ref_transform, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    test_attention_prefill(
        4,
        1024,
        32,
        8,
        128,
        128,
        1,
        True,
        torch.bfloat16,
    )
    test_attention_prefill_varlen(
        [0, 256, 1024, 2048, 2560],
        32,
        32,
        128,
        128,
        1.0,
        True,
        torch.bfloat16,
    )
