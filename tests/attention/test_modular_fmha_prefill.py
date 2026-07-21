# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the refactored flashinfer.cute_dsl.attention package.

Covers: basic prefill (various q/kv/batch combos, GQA vs MHA, causal),
variable-length sequences, output transform, logits transform (sigmoid),
attention sink, and band masks (causal sliding window, symmetric window,
left-bound-only window).

Each unique (mask_type, fusion) combination triggers one JIT compilation
(~30s). The test matrix is designed to reuse compiled kernels across
many runtime configurations so the full suite runs in a few minutes.
"""

import math

import pytest
import torch

from flashinfer.cute_dsl import is_cute_dsl_available
from flashinfer.utils import is_sm100a_supported

if not is_cute_dsl_available():
    pytest.skip("CuTe DSL not available", allow_module_level=True)

from tests.test_helpers.sink_attention_reference import sink_softmax
import cutlass.cute as cute

from flashinfer.cute_dsl.attention import (
    BatchPrefillCuteDSLWrapper,
    AttentionVariant,
    AttentionWithSink,
    SigmoidAttention,
    SigmoidTanhAttention,
    ALiBiAttention,
    RPEAttention,
)


# ---------------------------------------------------------------------------
#  Reference implementations
# ---------------------------------------------------------------------------


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
        assert num_qo_heads % num_kv_heads == 0
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

    if num_qo_heads > num_kv_heads:
        assert num_qo_heads % num_kv_heads == 0
        group_size = num_qo_heads // num_kv_heads
        k = torch.repeat_interleave(k, group_size, dim=1)
        v = torch.repeat_interleave(v, group_size, dim=1)

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


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------


def _skip_if_unsupported(qo_len, kv_len, causal):
    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("SM100A is not supported on this device")
    if qo_len > kv_len and causal:
        pytest.skip("qo_len > kv_len and causal is not supported")


HEAD_DIM = 128
NUM_QO_HEADS = 32
SM_SCALE = 1.0 / math.sqrt(HEAD_DIM)
DTYPE = torch.bfloat16
ATOL = 1e-2
RTOL = 1e-2


# ---------------------------------------------------------------------------
#  1. Basic prefill — curated (batch, qo, kv) combos covering tile boundaries
# ---------------------------------------------------------------------------

BASIC_SHAPE_PARAMS = [
    # (batch, qo_len, kv_len)
    (1, 64, 64),  # single batch, sub-tile
    (1, 128, 128),  # single batch, exact tile
    (1, 177, 977),  # single batch, multi-tile
    (2, 256, 256),  # small batch, multi-tile
    (9, 1, 1),  # many batches, minimal sizes
    (9, 64, 1),  # many batches, sub-tile Q, minimal KV
    (9, 177, 1),  # multi-tile Q, minimal KV (regression for accumulate bug)
    (9, 177, 64),  # multi-tile Q, sub-tile KV
    (9, 177, 128),  # multi-tile Q, exact-tile KV boundary
    (9, 177, 129),  # multi-tile Q, just-over-tile KV
    (9, 256, 17),  # multi-tile Q, small non-aligned KV
    (9, 256, 256),  # multi-tile both, aligned
    (9, 177, 544),  # multi-tile, causal_offset=367 not tile-aligned (regression)
    (9, 256, 544),  # multi-tile, causal_offset=288 not tile-aligned
    (9, 177, 1999),  # multi-tile both, large non-aligned KV
    (17, 177, 977),  # large batch, multi-tile
]


@pytest.mark.parametrize("batch_size,qo_len,kv_len", BASIC_SHAPE_PARAMS)
@pytest.mark.parametrize("num_kv_heads", [8, 32])
@pytest.mark.parametrize("causal", [False, True])
def test_attention_prefill(
    batch_size,
    qo_len,
    kv_len,
    num_kv_heads,
    causal,
):
    _skip_if_unsupported(qo_len, kv_len, causal)

    torch.manual_seed(42)
    q = torch.randn(
        batch_size * qo_len, NUM_QO_HEADS, HEAD_DIM, dtype=DTYPE, device="cuda"
    )
    k = torch.randn(
        batch_size * kv_len, num_kv_heads, HEAD_DIM, dtype=DTYPE, device="cuda"
    )
    v = torch.randn(
        batch_size * kv_len, num_kv_heads, HEAD_DIM, dtype=DTYPE, device="cuda"
    )
    qo_indptr = (
        torch.arange(0, batch_size + 1, device="cuda", dtype=torch.int32) * qo_len
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
        NUM_QO_HEADS,
        num_kv_heads,
        HEAD_DIM,
        head_dim_vo=HEAD_DIM,
        causal=causal,
        sm_scale=SM_SCALE,
        q_data_type=DTYPE,
        kv_data_type=DTYPE,
    )
    o = wrapper.run(q, k, v)
    o_ref, _ = attention_ref(batch_size, q, k, v, causal, SM_SCALE)

    torch.testing.assert_close(o, o_ref, rtol=RTOL, atol=ATOL)

    # Verify the out= in-place contract: the user's pre-allocated tensor
    # must be populated with the results and returned as-is.
    o_buffer = torch.empty_like(o)
    o_ret = wrapper.run(q, k, v, out=o_buffer)
    assert o_ret.data_ptr() == o_buffer.data_ptr(), (
        "run(out=...) must return the same tensor object"
    )
    torch.testing.assert_close(o_buffer, o_ref, rtol=RTOL, atol=ATOL)


# ---------------------------------------------------------------------------
#  2. Variable-length sequences — a few representative indptr patterns
# ---------------------------------------------------------------------------

VARLEN_INDPTR_PARAMS_COMPACT = [
    [0, 7],  # single very short seq
    [0, 1284],  # single long seq
    [0, 1298, 2638],  # 2 seqs
    [0, 1350, 2667, 4003, 5347, 6631, 7919, 9208, 10524],  # 8 seqs
    [0, 1300, 2614, 3924],  # 3 seqs, short
    [
        0,
        1536,
        3061,
        4578,
        6177,
        7774,
        9378,
        10958,
        12636,
        14292,
        15954,
    ],  # 10 seqs, varied
]


@pytest.mark.parametrize("indptr", VARLEN_INDPTR_PARAMS_COMPACT)
@pytest.mark.parametrize("num_kv_heads", [8, 32])
@pytest.mark.parametrize("causal", [False, True])
def test_attention_prefill_varlen(
    indptr,
    num_kv_heads,
    causal,
):
    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("SM100A is not supported on this device")

    torch.manual_seed(42)
    sm_scale = SM_SCALE

    q = torch.randn(indptr[-1], NUM_QO_HEADS, HEAD_DIM, dtype=DTYPE, device="cuda")
    k = torch.randn(indptr[-1], num_kv_heads, HEAD_DIM, dtype=DTYPE, device="cuda")
    v = torch.randn(indptr[-1], num_kv_heads, HEAD_DIM, dtype=DTYPE, device="cuda")

    qo_indptr = torch.tensor(indptr, device="cuda", dtype=torch.int32)
    kv_indptr = qo_indptr

    wrapper = BatchPrefillCuteDSLWrapper(
        torch.empty(1, device="cuda", dtype=torch.uint8),
    )
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        NUM_QO_HEADS,
        num_kv_heads,
        HEAD_DIM,
        head_dim_vo=HEAD_DIM,
        causal=causal,
        sm_scale=sm_scale,
        q_data_type=DTYPE,
        kv_data_type=DTYPE,
    )
    o = wrapper.run(q, k, v)

    gqa_group_ratio = NUM_QO_HEADS // num_kv_heads
    k_repeated = torch.repeat_interleave(k, gqa_group_ratio, dim=1)
    v_repeated = torch.repeat_interleave(v, gqa_group_ratio, dim=1)

    o_ref, _ = attention_varlen_ref(
        q, k_repeated, v_repeated, qo_indptr, kv_indptr, causal, sm_scale
    )

    torch.testing.assert_close(o, o_ref, rtol=RTOL, atol=ATOL)


# ---------------------------------------------------------------------------
#  3. Output transform
# ---------------------------------------------------------------------------

FUSION_SHAPE_PARAMS = [
    # (batch, qo_len, kv_len) — smaller matrix, covers key boundaries
    (1, 177, 977),
    (9, 177, 64),
    (9, 256, 256),
    (9, 177, 128),
]


class _ScaleBy2TestVariant(AttentionVariant):
    """Test-only: multiply output by 2x to verify output_transform hook."""

    has_output_transform = True

    @cute.jit
    def transform_output(self, output, batch_idx, qo_idx, qo_head_idx, m, rcp_d, scale):
        return output * scale * 2.0 * rcp_d


@pytest.mark.parametrize("batch_size,qo_len,kv_len", FUSION_SHAPE_PARAMS)
@pytest.mark.parametrize("causal", [False, True])
def test_attention_prefill_output_transform(
    batch_size,
    qo_len,
    kv_len,
    causal,
):
    _skip_if_unsupported(qo_len, kv_len, causal)
    num_kv_heads = 8

    torch.manual_seed(42)
    q = torch.randn(
        batch_size * qo_len, NUM_QO_HEADS, HEAD_DIM, dtype=DTYPE, device="cuda"
    )
    k = torch.randn(
        batch_size * kv_len, num_kv_heads, HEAD_DIM, dtype=DTYPE, device="cuda"
    )
    v = torch.randn(
        batch_size * kv_len, num_kv_heads, HEAD_DIM, dtype=DTYPE, device="cuda"
    )
    qo_indptr = (
        torch.arange(0, batch_size + 1, device="cuda", dtype=torch.int32) * qo_len
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
        NUM_QO_HEADS,
        num_kv_heads,
        HEAD_DIM,
        head_dim_vo=HEAD_DIM,
        causal=causal,
        sm_scale=1.0,
        q_data_type=DTYPE,
        kv_data_type=DTYPE,
        variant=_ScaleBy2TestVariant(),
    )
    o = wrapper.run(q, k, v)

    o_ref, _ = attention_ref(batch_size, q, k, v, causal, 1.0)
    o_ref_transform = o_ref * 2.0

    torch.testing.assert_close(o, o_ref_transform, rtol=RTOL, atol=ATOL)


# ---------------------------------------------------------------------------
#  4. Logits transform (sigmoid)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("batch_size,qo_len,kv_len", FUSION_SHAPE_PARAMS)
@pytest.mark.parametrize("causal", [False, True])
def test_attention_prefill_logits_transform(
    batch_size,
    qo_len,
    kv_len,
    causal,
):
    _skip_if_unsupported(qo_len, kv_len, causal)
    num_kv_heads = 8

    torch.manual_seed(42)
    q = torch.randn(
        batch_size * qo_len, NUM_QO_HEADS, HEAD_DIM, dtype=DTYPE, device="cuda"
    )
    k = torch.randn(
        batch_size * kv_len, num_kv_heads, HEAD_DIM, dtype=DTYPE, device="cuda"
    )
    v = torch.randn(
        batch_size * kv_len, num_kv_heads, HEAD_DIM, dtype=DTYPE, device="cuda"
    )
    qo_indptr = (
        torch.arange(0, batch_size + 1, device="cuda", dtype=torch.int32) * qo_len
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
        NUM_QO_HEADS,
        num_kv_heads,
        HEAD_DIM,
        head_dim_vo=HEAD_DIM,
        causal=causal,
        sm_scale=1.0,
        q_data_type=DTYPE,
        kv_data_type=DTYPE,
        variant=SigmoidAttention(scale=1.0, bias=0.0),
    )
    o = wrapper.run(q, k, v)

    o_ref = attention_sigmoid_ref(batch_size, q, k, v, causal, 1.0, 0.0)

    # Sigmoid logits transform has known ~1% element accuracy limitations
    # (documented in PR #1549)
    torch.testing.assert_close(o, o_ref, rtol=0.15, atol=0.15)


@pytest.mark.parametrize(
    "batch_size,qo_len,kv_len",
    [
        (1, 128, 128),
        (9, 256, 256),
    ],
)
@pytest.mark.parametrize("bias", [-5.0, -2.0, 1.0])
def test_attention_prefill_sigmoid_bias(batch_size, qo_len, kv_len, bias):
    """Regression test: SigmoidAttention bias must match torch.sigmoid semantics.

    The bias parameter should produce σ(score * scale + bias), matching
    the C++ FlashSigmoid which converts both scale and bias to log-base-2
    via multiplication by log2(e).  A previous implementation only converted
    scale, effectively attenuating the bias by ln(2) ≈ 0.693.
    """
    _skip_if_unsupported(qo_len, kv_len, causal=False)
    num_kv_heads = 8
    scale = 1.0

    torch.manual_seed(42)
    q = torch.randn(
        batch_size * qo_len, NUM_QO_HEADS, HEAD_DIM, dtype=DTYPE, device="cuda"
    )
    k = torch.randn(
        batch_size * kv_len, num_kv_heads, HEAD_DIM, dtype=DTYPE, device="cuda"
    )
    v = torch.randn(
        batch_size * kv_len, num_kv_heads, HEAD_DIM, dtype=DTYPE, device="cuda"
    )
    qo_indptr = (
        torch.arange(0, batch_size + 1, device="cuda", dtype=torch.int32) * qo_len
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
        NUM_QO_HEADS,
        num_kv_heads,
        HEAD_DIM,
        head_dim_vo=HEAD_DIM,
        causal=False,
        sm_scale=1.0,
        q_data_type=DTYPE,
        kv_data_type=DTYPE,
        variant=SigmoidAttention(scale=scale, bias=bias),
    )
    o = wrapper.run(q, k, v)

    o_ref = attention_sigmoid_ref(batch_size, q, k, v, False, scale, bias)

    torch.testing.assert_close(o, o_ref, rtol=0.15, atol=0.15)


# ---------------------------------------------------------------------------
#  4b. Logits transform (sigmoid via tanh)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("batch_size,qo_len,kv_len", FUSION_SHAPE_PARAMS)
@pytest.mark.parametrize("causal", [False, True])
def test_attention_prefill_sigmoid_tanh(
    batch_size,
    qo_len,
    kv_len,
    causal,
):
    _skip_if_unsupported(qo_len, kv_len, causal)
    num_kv_heads = 8

    torch.manual_seed(42)
    q = torch.randn(
        batch_size * qo_len, NUM_QO_HEADS, HEAD_DIM, dtype=DTYPE, device="cuda"
    )
    k = torch.randn(
        batch_size * kv_len, num_kv_heads, HEAD_DIM, dtype=DTYPE, device="cuda"
    )
    v = torch.randn(
        batch_size * kv_len, num_kv_heads, HEAD_DIM, dtype=DTYPE, device="cuda"
    )
    qo_indptr = (
        torch.arange(0, batch_size + 1, device="cuda", dtype=torch.int32) * qo_len
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
        NUM_QO_HEADS,
        num_kv_heads,
        HEAD_DIM,
        head_dim_vo=HEAD_DIM,
        causal=causal,
        sm_scale=1.0,
        q_data_type=DTYPE,
        kv_data_type=DTYPE,
        variant=SigmoidTanhAttention(scale=1.0, bias=0.0),
    )
    o = wrapper.run(q, k, v)

    o_ref = attention_sigmoid_ref(batch_size, q, k, v, causal, 1.0, 0.0)

    torch.testing.assert_close(o, o_ref, rtol=0.15, atol=0.15)


@pytest.mark.parametrize(
    "batch_size,qo_len,kv_len",
    [
        (1, 128, 128),
        (9, 256, 256),
    ],
)
@pytest.mark.parametrize("bias", [-5.0, -2.0, 1.0])
def test_attention_prefill_sigmoid_tanh_bias(batch_size, qo_len, kv_len, bias):
    """Regression test: SigmoidTanhAttention bias must match torch.sigmoid semantics."""
    _skip_if_unsupported(qo_len, kv_len, causal=False)
    num_kv_heads = 8
    scale = 1.0

    torch.manual_seed(42)
    q = torch.randn(
        batch_size * qo_len, NUM_QO_HEADS, HEAD_DIM, dtype=DTYPE, device="cuda"
    )
    k = torch.randn(
        batch_size * kv_len, num_kv_heads, HEAD_DIM, dtype=DTYPE, device="cuda"
    )
    v = torch.randn(
        batch_size * kv_len, num_kv_heads, HEAD_DIM, dtype=DTYPE, device="cuda"
    )
    qo_indptr = (
        torch.arange(0, batch_size + 1, device="cuda", dtype=torch.int32) * qo_len
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
        NUM_QO_HEADS,
        num_kv_heads,
        HEAD_DIM,
        head_dim_vo=HEAD_DIM,
        causal=False,
        sm_scale=1.0,
        q_data_type=DTYPE,
        kv_data_type=DTYPE,
        variant=SigmoidTanhAttention(scale=scale, bias=bias),
    )
    o = wrapper.run(q, k, v)

    o_ref = attention_sigmoid_ref(batch_size, q, k, v, False, scale, bias)

    torch.testing.assert_close(o, o_ref, rtol=0.15, atol=0.15)


# ---------------------------------------------------------------------------
#  5. Attention sink
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("batch_size,qo_len,kv_len", FUSION_SHAPE_PARAMS)
@pytest.mark.parametrize("causal", [False, True])
def test_attention_prefill_attention_sink(
    batch_size,
    qo_len,
    kv_len,
    causal,
):
    _skip_if_unsupported(qo_len, kv_len, causal)
    num_kv_heads = 8

    torch.manual_seed(42)
    q = torch.randn(
        batch_size * qo_len, NUM_QO_HEADS, HEAD_DIM, dtype=DTYPE, device="cuda"
    )
    k = torch.randn(
        batch_size * kv_len, num_kv_heads, HEAD_DIM, dtype=DTYPE, device="cuda"
    )
    v = torch.randn(
        batch_size * kv_len, num_kv_heads, HEAD_DIM, dtype=DTYPE, device="cuda"
    )
    qo_indptr = (
        torch.arange(0, batch_size + 1, device="cuda", dtype=torch.int32) * qo_len
    )
    kv_indptr = (
        torch.arange(0, batch_size + 1, device="cuda", dtype=torch.int32) * kv_len
    )
    sink = torch.randn((NUM_QO_HEADS,), dtype=DTYPE, device="cuda")

    wrapper = BatchPrefillCuteDSLWrapper(
        torch.empty(128 * 1024 * 1024, device="cuda", dtype=torch.uint8),
    )
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        NUM_QO_HEADS,
        num_kv_heads,
        HEAD_DIM,
        head_dim_vo=HEAD_DIM,
        causal=causal,
        sm_scale=SM_SCALE,
        q_data_type=DTYPE,
        kv_data_type=DTYPE,
        variant=AttentionWithSink(sink),
    )
    o = wrapper.run(q, k, v)

    o_ref, _ = attention_ref(batch_size, q, k, v, causal, SM_SCALE, sink=sink)

    torch.testing.assert_close(o, o_ref, rtol=RTOL, atol=ATOL)


# ---------------------------------------------------------------------------
#  6. Float16 dtype
# ---------------------------------------------------------------------------

FP16_SHAPE_PARAMS = [
    (1, 128, 128),
    (9, 256, 256),
    (1, 177, 977),
]


@pytest.mark.parametrize(
    "batch_size,qo_len,kv_len,causal,window_left",
    [
        (1, 2048, 2048, True, -1),
        (1, 2048, 2048, True, 127),
        (3, 512, 512, False, -1),
    ],
)
def test_attention_prefill_fp8(batch_size, qo_len, kv_len, causal, window_left):
    """Uniform fp8 (e4m3) inputs on the modular kernel.

    The tolerance reflects fp8's inherent error — P (the softmax weights)
    is stored in e4m3 for the PV GEMM — and was calibrated against the
    trtllm CuTe DSL FMHA kernel's fp8 output on identical inputs (both
    kernels reach max_err ~0.066 vs an f32 reference at these shapes).
    """
    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("SM100A is not supported on this device")
    num_kv_heads = 8

    torch.manual_seed(42)
    q = torch.randn(
        batch_size * qo_len, NUM_QO_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda"
    ).to(torch.float8_e4m3fn)
    k = torch.randn(
        batch_size * kv_len, num_kv_heads, HEAD_DIM, dtype=torch.bfloat16, device="cuda"
    ).to(torch.float8_e4m3fn)
    v = torch.randn(
        batch_size * kv_len, num_kv_heads, HEAD_DIM, dtype=torch.bfloat16, device="cuda"
    ).to(torch.float8_e4m3fn)
    qo_indptr = (
        torch.arange(0, batch_size + 1, device="cuda", dtype=torch.int32) * qo_len
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
        NUM_QO_HEADS,
        num_kv_heads,
        HEAD_DIM,
        head_dim_vo=HEAD_DIM,
        causal=causal,
        sm_scale=SM_SCALE,
        q_data_type=torch.float8_e4m3fn,
        kv_data_type=torch.float8_e4m3fn,
        window_left=window_left,
    )
    o = wrapper.run(q, k, v)

    # f32 reference over the dequantized fp8 inputs.
    o_ref = attention_band_mask_ref(
        batch_size,
        q.to(torch.float32),
        k.to(torch.float32),
        v.to(torch.float32),
        SM_SCALE,
        causal,
        window_left,
        window_right=-1,
    )
    torch.testing.assert_close(o.float(), o_ref.float(), rtol=1e-2, atol=8e-2)


def _band_mask_lse_ref(batch_size, q, k, causal, window_left, sm_scale):
    """Log2-domain LSE reference over band-masked logits (per uniform batch)."""
    qo_len = q.shape[0] // batch_size
    kv_len = k.shape[0] // batch_size
    group_size = q.shape[1] // k.shape[1]
    kk = torch.repeat_interleave(k, group_size, dim=1)
    logits = (
        torch.einsum(
            "bmhd,bnhd->bhmn",
            q.view(batch_size, qo_len, q.shape[1], -1).float(),
            kk.view(batch_size, kv_len, q.shape[1], -1).float(),
        )
        * sm_scale
    )
    qk_offset = kv_len - qo_len
    q_idx = torch.arange(qo_len, device=q.device).unsqueeze(1)
    k_idx = torch.arange(kv_len, device=q.device).unsqueeze(0)
    mask = torch.ones(qo_len, kv_len, dtype=torch.bool, device=q.device)
    if causal:
        mask &= k_idx <= q_idx + qk_offset
    if window_left >= 0:
        mask &= (q_idx + qk_offset) - k_idx <= window_left
    logits = logits.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    # (b, h, m) -> (b*m, h), log2 domain
    lse = torch.logsumexp(logits, dim=-1) * math.log2(math.e)
    return lse.transpose(1, 2).reshape(batch_size * qo_len, q.shape[1])


@pytest.mark.parametrize(
    "batch_size,qo_len,kv_len,causal,window_left",
    [
        (1, 2048, 2048, True, -1),
        (1, 2048, 2048, True, 127),
        (3, 512, 512, False, -1),
        (1, 128, 512, True, 100),
    ],
)
def test_attention_prefill_lse(batch_size, qo_len, kv_len, causal, window_left):
    """return_lse on the standard path: log2-domain (total_q, h_q) f32."""
    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("SM100A is not supported on this device")
    num_kv_heads = 8

    torch.manual_seed(42)
    q = torch.randn(
        batch_size * qo_len, NUM_QO_HEADS, HEAD_DIM, dtype=DTYPE, device="cuda"
    )
    k = torch.randn(
        batch_size * kv_len, num_kv_heads, HEAD_DIM, dtype=DTYPE, device="cuda"
    )
    v = torch.randn_like(k)
    qo_indptr = (
        torch.arange(0, batch_size + 1, device="cuda", dtype=torch.int32) * qo_len
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
        NUM_QO_HEADS,
        num_kv_heads,
        HEAD_DIM,
        head_dim_vo=HEAD_DIM,
        causal=causal,
        sm_scale=SM_SCALE,
        q_data_type=DTYPE,
        kv_data_type=DTYPE,
        window_left=window_left,
    )
    o_plain = wrapper.run(q, k, v)
    o, lse = wrapper.run(q, k, v, return_lse=True)

    # The LSE variant must not perturb the attention output.
    torch.testing.assert_close(o, o_plain, rtol=0, atol=0)
    o_ref = attention_band_mask_ref(
        batch_size, q, k, v, SM_SCALE, causal, window_left, window_right=-1
    )
    torch.testing.assert_close(o, o_ref, rtol=RTOL, atol=ATOL)
    lse_ref = _band_mask_lse_ref(batch_size, q, k, causal, window_left, SM_SCALE)
    torch.testing.assert_close(lse, lse_ref, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("indptr", [[0, 7, 1291, 1547, 3083]])
def test_attention_prefill_lse_varlen(indptr):
    """LSE row indexing over a mixed-length ragged batch (causal+window)."""
    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("SM100A is not supported on this device")
    num_kv_heads = 8
    window_left = 100

    torch.manual_seed(42)
    q = torch.randn(indptr[-1], NUM_QO_HEADS, HEAD_DIM, dtype=DTYPE, device="cuda")
    k = torch.randn(indptr[-1], num_kv_heads, HEAD_DIM, dtype=DTYPE, device="cuda")
    v = torch.randn_like(k)
    qo_indptr = torch.tensor(indptr, device="cuda", dtype=torch.int32)

    wrapper = BatchPrefillCuteDSLWrapper(
        torch.empty(128 * 1024 * 1024, device="cuda", dtype=torch.uint8),
    )
    wrapper.plan(
        qo_indptr,
        qo_indptr,
        NUM_QO_HEADS,
        num_kv_heads,
        HEAD_DIM,
        head_dim_vo=HEAD_DIM,
        causal=True,
        sm_scale=SM_SCALE,
        q_data_type=DTYPE,
        kv_data_type=DTYPE,
        window_left=window_left,
    )
    o, lse = wrapper.run(q, k, v, return_lse=True)

    for i in range(len(indptr) - 1):
        lo, hi = indptr[i], indptr[i + 1]
        o_ref = attention_band_mask_ref(
            1, q[lo:hi], k[lo:hi], v[lo:hi], SM_SCALE, True, window_left, -1
        )
        lse_ref = _band_mask_lse_ref(1, q[lo:hi], k[lo:hi], True, window_left, SM_SCALE)
        torch.testing.assert_close(o[lo:hi], o_ref, rtol=RTOL, atol=ATOL)
        torch.testing.assert_close(lse[lo:hi], lse_ref, rtol=1e-3, atol=1e-3)


def test_attention_prefill_lse_alibi():
    """LSE with a score_mod variant (standard path): LSE over ALiBi logits."""
    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("SM100A is not supported on this device")
    batch_size, qo_len, kv_len = 9, 256, 256
    num_kv_heads = 8

    torch.manual_seed(42)
    q = torch.randn(
        batch_size * qo_len, NUM_QO_HEADS, HEAD_DIM, dtype=DTYPE, device="cuda"
    )
    k = torch.randn(
        batch_size * kv_len, num_kv_heads, HEAD_DIM, dtype=DTYPE, device="cuda"
    )
    v = torch.randn_like(k)
    indptr = torch.arange(0, batch_size + 1, device="cuda", dtype=torch.int32) * qo_len
    alibi_slopes = ALiBiAttention.get_slopes(NUM_QO_HEADS).cuda()

    wrapper = BatchPrefillCuteDSLWrapper(
        torch.empty(128 * 1024 * 1024, device="cuda", dtype=torch.uint8),
    )
    wrapper.plan(
        indptr,
        indptr.clone(),
        NUM_QO_HEADS,
        num_kv_heads,
        HEAD_DIM,
        head_dim_vo=HEAD_DIM,
        causal=True,
        sm_scale=SM_SCALE,
        q_data_type=DTYPE,
        kv_data_type=DTYPE,
        variant=ALiBiAttention(alibi_slopes),
    )
    o, lse = wrapper.run(q, k, v, return_lse=True)

    o_ref = attention_alibi_ref(batch_size, q, k, v, True, SM_SCALE, alibi_slopes)
    torch.testing.assert_close(o, o_ref, rtol=RTOL, atol=ATOL)

    # LSE over the ALiBi-modified logits (mirrors attention_alibi_ref).
    kk = torch.repeat_interleave(k, NUM_QO_HEADS // num_kv_heads, dim=1)
    logits = torch.einsum(
        "bmhd,bnhd->bhmn",
        q.view(batch_size, qo_len, NUM_QO_HEADS, HEAD_DIM).float(),
        kk.view(batch_size, kv_len, NUM_QO_HEADS, HEAD_DIM).float(),
    )
    qo_pos = torch.arange(qo_len, device=q.device).view(1, 1, -1, 1)
    kv_pos = torch.arange(kv_len, device=q.device).view(1, 1, 1, -1)
    logits = (logits + alibi_slopes.view(1, -1, 1, 1) * (kv_pos - qo_pos)) * SM_SCALE
    mask = torch.arange(kv_len - qo_len, kv_len, device=q.device).unsqueeze(
        1
    ) >= torch.arange(0, kv_len, device=q.device).unsqueeze(0)
    logits = logits.masked_fill(mask.unsqueeze(0).unsqueeze(0) == 0, float("-inf"))
    lse_ref = (
        (torch.logsumexp(logits, dim=-1) * math.log2(math.e))
        .transpose(1, 2)
        .reshape(batch_size * qo_len, NUM_QO_HEADS)
    )
    torch.testing.assert_close(lse, lse_ref, rtol=1e-3, atol=1e-3)


def test_attention_prefill_lse_sink():
    """LSE with attention sinks: the sink term is part of the denominator."""
    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("SM100A is not supported on this device")
    batch_size, qo_len, kv_len = 1, 256, 256
    num_kv_heads = 8

    torch.manual_seed(42)
    q = torch.randn(
        batch_size * qo_len, NUM_QO_HEADS, HEAD_DIM, dtype=DTYPE, device="cuda"
    )
    k = torch.randn(
        batch_size * kv_len, num_kv_heads, HEAD_DIM, dtype=DTYPE, device="cuda"
    )
    v = torch.randn_like(k)
    indptr = torch.arange(0, batch_size + 1, device="cuda", dtype=torch.int32) * qo_len
    sink = torch.randn((NUM_QO_HEADS,), dtype=DTYPE, device="cuda")

    wrapper = BatchPrefillCuteDSLWrapper(
        torch.empty(128 * 1024 * 1024, device="cuda", dtype=torch.uint8),
    )
    wrapper.plan(
        indptr,
        indptr.clone(),
        NUM_QO_HEADS,
        num_kv_heads,
        HEAD_DIM,
        head_dim_vo=HEAD_DIM,
        causal=True,
        sm_scale=SM_SCALE,
        q_data_type=DTYPE,
        kv_data_type=DTYPE,
        variant=AttentionWithSink(sink),
    )
    o, lse = wrapper.run(q, k, v, return_lse=True)

    o_ref, _ = attention_ref(batch_size, q, k, v, True, SM_SCALE, sink=sink)
    torch.testing.assert_close(o, o_ref, rtol=RTOL, atol=ATOL)

    # With-sink LSE: log of the actual normalization denominator,
    # log2(sum_k exp(s_k) + exp(sink)) — attention_ref's lse is sink-less.
    kk = torch.repeat_interleave(k, NUM_QO_HEADS // num_kv_heads, dim=1)
    logits = (
        torch.einsum(
            "bmhd,bnhd->bhmn",
            q.view(batch_size, qo_len, NUM_QO_HEADS, HEAD_DIM).float(),
            kk.view(batch_size, kv_len, NUM_QO_HEADS, HEAD_DIM).float(),
        )
        * SM_SCALE
    )
    mask = torch.arange(kv_len - qo_len, kv_len, device=q.device).unsqueeze(
        1
    ) >= torch.arange(0, kv_len, device=q.device).unsqueeze(0)
    logits = logits.masked_fill(mask.unsqueeze(0).unsqueeze(0) == 0, float("-inf"))
    logits_with_sink = torch.cat(
        [logits, sink.float().view(1, -1, 1, 1).expand(batch_size, -1, qo_len, 1)],
        dim=-1,
    )
    lse_ref = (
        (torch.logsumexp(logits_with_sink, dim=-1) * math.log2(math.e))
        .transpose(1, 2)
        .reshape(batch_size * qo_len, NUM_QO_HEADS)
    )
    torch.testing.assert_close(lse, lse_ref, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("batch_size,qo_len,kv_len", FP16_SHAPE_PARAMS)
@pytest.mark.parametrize("causal", [False, True])
def test_attention_prefill_fp16(
    batch_size,
    qo_len,
    kv_len,
    causal,
):
    _skip_if_unsupported(qo_len, kv_len, causal)
    num_kv_heads = 8
    dtype = torch.float16

    torch.manual_seed(42)
    q = torch.randn(
        batch_size * qo_len, NUM_QO_HEADS, HEAD_DIM, dtype=dtype, device="cuda"
    )
    k = torch.randn(
        batch_size * kv_len, num_kv_heads, HEAD_DIM, dtype=dtype, device="cuda"
    )
    v = torch.randn(
        batch_size * kv_len, num_kv_heads, HEAD_DIM, dtype=dtype, device="cuda"
    )
    qo_indptr = (
        torch.arange(0, batch_size + 1, device="cuda", dtype=torch.int32) * qo_len
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
        NUM_QO_HEADS,
        num_kv_heads,
        HEAD_DIM,
        head_dim_vo=HEAD_DIM,
        causal=causal,
        sm_scale=SM_SCALE,
        q_data_type=dtype,
        kv_data_type=dtype,
    )
    o = wrapper.run(q, k, v)
    o_ref, _ = attention_ref(batch_size, q, k, v, causal, SM_SCALE)

    torch.testing.assert_close(o, o_ref, rtol=RTOL, atol=ATOL)


# ---------------------------------------------------------------------------
#  7. Sliding window mask
# ---------------------------------------------------------------------------


def attention_band_mask_ref(
    batch_size,
    q,
    k,
    v,
    sm_scale,
    causal,
    window_left,
    window_right,
):
    """Band-mask reference. With offset = kv_len - qo_len (Q right-aligned
    to KV), row q sees k in::

        max(0, q + offset - window_left) <= k <= q + offset            (causal)
        max(0, q + offset - window_left) <= k <= q + offset + window_right

    where a window value of -1 means unbounded on that side.
    """
    qo_len = q.shape[0] // batch_size
    kv_len = k.shape[0] // batch_size
    num_qo_heads = q.shape[1]
    num_kv_heads = k.shape[1]
    head_dim_qk = q.shape[2]
    head_dim_vo = v.shape[2]

    if num_qo_heads > num_kv_heads:
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

    qk_offset = kv_len - qo_len
    q_idx = torch.arange(qo_len, device=q.device).unsqueeze(1)
    k_idx = torch.arange(kv_len, device=q.device).unsqueeze(0)
    mask = torch.ones(qo_len, kv_len, dtype=torch.bool, device=q.device)
    # Causal is a right bound of 0 (the kernel's own folding); it must come
    # from the flag because callers pass causal=True with window_right=-1.
    wr = 0 if causal else window_right
    if wr >= 0:
        mask &= k_idx - (q_idx + qk_offset) <= wr
    if window_left >= 0:
        mask &= (q_idx + qk_offset) - k_idx <= window_left
    logits = logits.masked_fill(mask.unsqueeze(0).unsqueeze(0) == 0, float("-inf"))

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
    return o_ref


def _run_band_mask_case(batch_size, qo_len, kv_len, causal, window_left, window_right):
    num_kv_heads = 8

    torch.manual_seed(42)
    q = torch.randn(
        batch_size * qo_len, NUM_QO_HEADS, HEAD_DIM, dtype=DTYPE, device="cuda"
    )
    k = torch.randn(
        batch_size * kv_len, num_kv_heads, HEAD_DIM, dtype=DTYPE, device="cuda"
    )
    v = torch.randn(
        batch_size * kv_len, num_kv_heads, HEAD_DIM, dtype=DTYPE, device="cuda"
    )
    qo_indptr = (
        torch.arange(0, batch_size + 1, device="cuda", dtype=torch.int32) * qo_len
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
        NUM_QO_HEADS,
        num_kv_heads,
        HEAD_DIM,
        head_dim_vo=HEAD_DIM,
        causal=causal,
        sm_scale=SM_SCALE,
        q_data_type=DTYPE,
        kv_data_type=DTYPE,
        window_left=window_left,
        window_right=window_right,
    )
    o = wrapper.run(q, k, v)
    o_ref = attention_band_mask_ref(
        batch_size, q, k, v, SM_SCALE, causal, window_left, window_right
    )

    torch.testing.assert_close(o, o_ref, rtol=RTOL, atol=ATOL)


SLIDING_WINDOW_PARAMS = [
    # (batch, qo_len, kv_len, window_left)
    (1, 256, 256, 64),
    (1, 256, 256, 128),
    (9, 256, 256, 100),
    (1, 512, 512, 200),
    # qo_len != kv_len (Q right-aligned to KV, as in append/prefill-with-cache)
    (1, 128, 256, 64),
    (1, 128, 512, 100),
    (1, 256, 512, 128),
    (3, 128, 384, 80),
]

# Causal sliding window (the serving configuration, e.g. Mistral/Gemma SWA).
# Window values reuse the symmetric list's to share compiled kernels where
# possible; includes a non-tile-aligned kv_len to exercise the seqlen_k tail
# combined with the window left edge.
CAUSAL_SLIDING_WINDOW_PARAMS = [
    # (batch, qo_len, kv_len, window_left)
    (1, 256, 256, 0),  # attend-self-only: exercises the head-borrow path
    (1, 256, 256, 64),
    (1, 512, 512, 128),
    (9, 256, 256, 100),
    (1, 128, 512, 100),  # qo_len != kv_len
    (1, 256, 300, 64),  # kv_len not tile-aligned
]


@pytest.mark.parametrize("batch_size,qo_len,kv_len,window_left", SLIDING_WINDOW_PARAMS)
def test_attention_prefill_sliding_window_symmetric(
    batch_size,
    qo_len,
    kv_len,
    window_left,
):
    """Non-causal symmetric window: k in [q+off-w, q+off+w]."""
    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("SM100A is not supported on this device")
    _run_band_mask_case(
        batch_size,
        qo_len,
        kv_len,
        causal=False,
        window_left=window_left,
        window_right=window_left,
    )


@pytest.mark.parametrize(
    "batch_size,qo_len,kv_len,window_left", CAUSAL_SLIDING_WINDOW_PARAMS
)
def test_attention_prefill_causal_sliding_window(
    batch_size,
    qo_len,
    kv_len,
    window_left,
):
    """Causal + window_left: k in [q+off-w, q+off] (regression: the window
    used to be silently ignored when causal=True)."""
    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("SM100A is not supported on this device")
    _run_band_mask_case(
        batch_size,
        qo_len,
        kv_len,
        causal=True,
        window_left=window_left,
        window_right=-1,
    )


@pytest.mark.parametrize(
    "batch_size,qo_len,kv_len,window_left",
    [
        (1, 256, 256, 64),
        (1, 128, 512, 100),
    ],
)
def test_attention_prefill_left_window_only(
    batch_size,
    qo_len,
    kv_len,
    window_left,
):
    """Non-causal left-bound-only window: k in [q+off-w, kv_len), matching
    the FlashInfer window_left convention (variants.cuh)."""
    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("SM100A is not supported on this device")
    _run_band_mask_case(
        batch_size,
        qo_len,
        kv_len,
        causal=False,
        window_left=window_left,
        window_right=-1,
    )


@pytest.mark.parametrize("causal", [False, True])
def test_attention_prefill_sliding_window_top_level_wrapper(causal):
    """Windowed plans through the public ragged wrapper (backend="cute-dsl").

    Guards the backend routing in flashinfer/prefill.py: windowed plans
    must reach the modular cute-dsl path.  The trtllm CuTe DSL FMHA route
    (taken for variant-less head-128 plans) measures slower on sliding
    windows, and its prebuilt artifact matrix has no windowed variants —
    a mis-routed windowed plan either JIT-compiles a slower kernel or, on
    older glue, fails the FFI signature check outright.
    """
    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("SM100A is not supported on this device")
    import flashinfer

    batch_size, seq_len, window_left = 1, 512, 128
    num_kv_heads = 8
    torch.manual_seed(42)
    q = torch.randn(
        batch_size * seq_len, NUM_QO_HEADS, HEAD_DIM, dtype=DTYPE, device="cuda"
    )
    k = torch.randn(
        batch_size * seq_len, num_kv_heads, HEAD_DIM, dtype=DTYPE, device="cuda"
    )
    v = torch.randn_like(k)
    indptr = torch.arange(0, batch_size + 1, device="cuda", dtype=torch.int32) * seq_len

    wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        torch.empty(128 * 1024 * 1024, device="cuda", dtype=torch.uint8),
        backend="cute-dsl",
    )
    wrapper.plan(
        indptr,
        indptr.clone(),
        NUM_QO_HEADS,
        num_kv_heads,
        HEAD_DIM,
        causal=causal,
        sm_scale=SM_SCALE,
        window_left=window_left,
        q_data_type=DTYPE,
        kv_data_type=DTYPE,
    )
    o = wrapper.run(q, k, v)
    o_ref = attention_band_mask_ref(
        batch_size, q, k, v, SM_SCALE, causal, window_left, window_right=-1
    )
    torch.testing.assert_close(o, o_ref, rtol=RTOL, atol=ATOL)


# Mixed per-sequence lengths: every sequence in a batch gets its own band
# geometry (kv_start, head/main/tail peel counts, borrow rule) and short
# sequences exercise the item-skip path — none of which uniform batches
# cover.  Patterns include a tiny sequence, uneven multi-sequence batches,
# and non-tile-aligned lengths.
WINDOW_VARLEN_INDPTR_PARAMS = [
    [0, 7, 1291, 1547, 3083],  # tiny seq + uneven lengths
    [0, 1350, 2667, 4003, 5347, 6631, 7919, 9208, 10524],  # 8 uneven seqs
    [0, 300, 556, 2604],  # non-tile-aligned mix
]


@pytest.mark.parametrize("indptr", WINDOW_VARLEN_INDPTR_PARAMS)
@pytest.mark.parametrize(
    "causal,window_left,window_right",
    [
        (True, 100, -1),  # causal + sliding window (serving SWA)
        (False, 64, -1),  # left-bound-only window
    ],
)
def test_attention_prefill_sliding_window_varlen(
    indptr,
    causal,
    window_left,
    window_right,
):
    """Windowed masks over mixed-length ragged batches."""
    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("SM100A is not supported on this device")
    num_kv_heads = 8

    torch.manual_seed(42)
    q = torch.randn(indptr[-1], NUM_QO_HEADS, HEAD_DIM, dtype=DTYPE, device="cuda")
    k = torch.randn(indptr[-1], num_kv_heads, HEAD_DIM, dtype=DTYPE, device="cuda")
    v = torch.randn_like(k)
    qo_indptr = torch.tensor(indptr, device="cuda", dtype=torch.int32)
    kv_indptr = qo_indptr

    wrapper = BatchPrefillCuteDSLWrapper(
        torch.empty(128 * 1024 * 1024, device="cuda", dtype=torch.uint8),
    )
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        NUM_QO_HEADS,
        num_kv_heads,
        HEAD_DIM,
        head_dim_vo=HEAD_DIM,
        causal=causal,
        sm_scale=SM_SCALE,
        q_data_type=DTYPE,
        kv_data_type=DTYPE,
        window_left=window_left,
        window_right=window_right,
    )
    o = wrapper.run(q, k, v)

    # Per-sequence reference: the band mask depends only on each
    # sequence's own lengths, so apply the uniform-batch reference to
    # each slice with batch_size=1.
    o_ref = torch.empty_like(o)
    for i in range(len(indptr) - 1):
        lo, hi = indptr[i], indptr[i + 1]
        o_ref[lo:hi] = attention_band_mask_ref(
            1,
            q[lo:hi],
            k[lo:hi],
            v[lo:hi],
            SM_SCALE,
            causal,
            window_left,
            window_right,
        )

    torch.testing.assert_close(o, o_ref, rtol=RTOL, atol=ATOL)


# ---------------------------------------------------------------------------
#  8. Head dimension 64
# ---------------------------------------------------------------------------

HEAD64_SHAPE_PARAMS = [
    (1, 128, 128),
    (9, 256, 256),
    (1, 177, 977),
]


@pytest.mark.parametrize("batch_size,qo_len,kv_len", HEAD64_SHAPE_PARAMS)
@pytest.mark.parametrize("causal", [False, True])
def test_attention_prefill_head_dim_64(
    batch_size,
    qo_len,
    kv_len,
    causal,
):
    _skip_if_unsupported(qo_len, kv_len, causal)
    head_dim = 64
    num_kv_heads = 8
    sm_scale = 1.0 / math.sqrt(head_dim)

    torch.manual_seed(42)
    q = torch.randn(
        batch_size * qo_len, NUM_QO_HEADS, head_dim, dtype=DTYPE, device="cuda"
    )
    k = torch.randn(
        batch_size * kv_len, num_kv_heads, head_dim, dtype=DTYPE, device="cuda"
    )
    v = torch.randn(
        batch_size * kv_len, num_kv_heads, head_dim, dtype=DTYPE, device="cuda"
    )
    qo_indptr = (
        torch.arange(0, batch_size + 1, device="cuda", dtype=torch.int32) * qo_len
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
        NUM_QO_HEADS,
        num_kv_heads,
        head_dim,
        head_dim_vo=head_dim,
        causal=causal,
        sm_scale=sm_scale,
        q_data_type=DTYPE,
        kv_data_type=DTYPE,
    )
    o = wrapper.run(q, k, v)
    o_ref, _ = attention_ref(batch_size, q, k, v, causal, sm_scale)

    torch.testing.assert_close(o, o_ref, rtol=RTOL, atol=ATOL)


# ---------------------------------------------------------------------------
#  9. Variable-length + logits transform (sigmoid)
# ---------------------------------------------------------------------------

VARLEN_FUSION_INDPTRS = [
    [0, 1298, 2638],
    [0, 1350, 2667, 4003, 5347, 6631, 7919, 9208, 10524],
]


def attention_varlen_sigmoid_ref(
    q,
    k,
    v,
    qo_indptr,
    kv_indptr,
    causal,
    sigmoid_scale,
    sigmoid_bias=0.0,
):
    batch_size = qo_indptr.shape[0] - 1
    nnz_qo = qo_indptr[-1].item()
    o = torch.empty(nnz_qo, *q.shape[1:-1], v.shape[-1], device=q.device, dtype=q.dtype)
    for i in range(batch_size):
        o_i = attention_sigmoid_ref(
            1,
            q[qo_indptr[i] : qo_indptr[i + 1]],
            k[kv_indptr[i] : kv_indptr[i + 1]],
            v[kv_indptr[i] : kv_indptr[i + 1]],
            causal,
            sigmoid_scale,
            sigmoid_bias,
        )
        o[qo_indptr[i] : qo_indptr[i + 1]] = o_i
    return o


@pytest.mark.parametrize("indptr", VARLEN_FUSION_INDPTRS)
def test_attention_prefill_varlen_logits_transform(indptr):
    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("SM100A is not supported on this device")
    num_kv_heads = 8

    torch.manual_seed(42)
    q = torch.randn(indptr[-1], NUM_QO_HEADS, HEAD_DIM, dtype=DTYPE, device="cuda")
    k = torch.randn(indptr[-1], num_kv_heads, HEAD_DIM, dtype=DTYPE, device="cuda")
    v = torch.randn(indptr[-1], num_kv_heads, HEAD_DIM, dtype=DTYPE, device="cuda")

    qo_indptr = torch.tensor(indptr, device="cuda", dtype=torch.int32)
    kv_indptr = qo_indptr

    wrapper = BatchPrefillCuteDSLWrapper(
        torch.empty(128 * 1024 * 1024, device="cuda", dtype=torch.uint8),
    )
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        NUM_QO_HEADS,
        num_kv_heads,
        HEAD_DIM,
        head_dim_vo=HEAD_DIM,
        causal=False,
        sm_scale=1.0,
        q_data_type=DTYPE,
        kv_data_type=DTYPE,
        variant=SigmoidAttention(scale=1.0, bias=0.0),
    )
    o = wrapper.run(q, k, v)

    gqa_group_ratio = NUM_QO_HEADS // num_kv_heads
    k_repeated = torch.repeat_interleave(k, gqa_group_ratio, dim=1)
    v_repeated = torch.repeat_interleave(v, gqa_group_ratio, dim=1)
    o_ref = attention_varlen_sigmoid_ref(
        q,
        k_repeated,
        v_repeated,
        qo_indptr,
        kv_indptr,
        False,
        1.0,
        0.0,
    )

    torch.testing.assert_close(o, o_ref, rtol=0.15, atol=0.15)


# ---------------------------------------------------------------------------
#  9b. Variable-length + logits transform (sigmoid via tanh)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("indptr", VARLEN_FUSION_INDPTRS)
def test_attention_prefill_varlen_sigmoid_tanh(indptr):
    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("SM100A is not supported on this device")
    num_kv_heads = 8

    torch.manual_seed(42)
    q = torch.randn(indptr[-1], NUM_QO_HEADS, HEAD_DIM, dtype=DTYPE, device="cuda")
    k = torch.randn(indptr[-1], num_kv_heads, HEAD_DIM, dtype=DTYPE, device="cuda")
    v = torch.randn(indptr[-1], num_kv_heads, HEAD_DIM, dtype=DTYPE, device="cuda")

    qo_indptr = torch.tensor(indptr, device="cuda", dtype=torch.int32)
    kv_indptr = qo_indptr

    wrapper = BatchPrefillCuteDSLWrapper(
        torch.empty(128 * 1024 * 1024, device="cuda", dtype=torch.uint8),
    )
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        NUM_QO_HEADS,
        num_kv_heads,
        HEAD_DIM,
        head_dim_vo=HEAD_DIM,
        causal=False,
        sm_scale=1.0,
        q_data_type=DTYPE,
        kv_data_type=DTYPE,
        variant=SigmoidTanhAttention(scale=1.0, bias=0.0),
    )
    o = wrapper.run(q, k, v)

    gqa_group_ratio = NUM_QO_HEADS // num_kv_heads
    k_repeated = torch.repeat_interleave(k, gqa_group_ratio, dim=1)
    v_repeated = torch.repeat_interleave(v, gqa_group_ratio, dim=1)
    o_ref = attention_varlen_sigmoid_ref(
        q,
        k_repeated,
        v_repeated,
        qo_indptr,
        kv_indptr,
        False,
        1.0,
        0.0,
    )

    torch.testing.assert_close(o, o_ref, rtol=0.15, atol=0.15)


# ---------------------------------------------------------------------------
#  10. Variable-length + attention sink
# ---------------------------------------------------------------------------


def attention_varlen_sink_ref(
    q,
    k,
    v,
    qo_indptr,
    kv_indptr,
    causal,
    sm_scale,
    sink,
):
    batch_size = qo_indptr.shape[0] - 1
    nnz_qo = qo_indptr[-1].item()
    o = torch.empty(nnz_qo, *q.shape[1:-1], v.shape[-1], device=q.device, dtype=q.dtype)
    for i in range(batch_size):
        o_i, _ = attention_ref(
            1,
            q[qo_indptr[i] : qo_indptr[i + 1]],
            k[kv_indptr[i] : kv_indptr[i + 1]],
            v[kv_indptr[i] : kv_indptr[i + 1]],
            causal,
            sm_scale,
            sink=sink,
        )
        o[qo_indptr[i] : qo_indptr[i + 1]] = o_i
    return o


@pytest.mark.parametrize("indptr", VARLEN_FUSION_INDPTRS)
def test_attention_prefill_varlen_attention_sink(indptr):
    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("SM100A is not supported on this device")
    num_kv_heads = 8

    torch.manual_seed(42)
    q = torch.randn(indptr[-1], NUM_QO_HEADS, HEAD_DIM, dtype=DTYPE, device="cuda")
    k = torch.randn(indptr[-1], num_kv_heads, HEAD_DIM, dtype=DTYPE, device="cuda")
    v = torch.randn(indptr[-1], num_kv_heads, HEAD_DIM, dtype=DTYPE, device="cuda")
    sink = torch.randn((NUM_QO_HEADS,), dtype=DTYPE, device="cuda")

    qo_indptr = torch.tensor(indptr, device="cuda", dtype=torch.int32)
    kv_indptr = qo_indptr

    wrapper = BatchPrefillCuteDSLWrapper(
        torch.empty(128 * 1024 * 1024, device="cuda", dtype=torch.uint8),
    )
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        NUM_QO_HEADS,
        num_kv_heads,
        HEAD_DIM,
        head_dim_vo=HEAD_DIM,
        causal=False,
        sm_scale=SM_SCALE,
        q_data_type=DTYPE,
        kv_data_type=DTYPE,
        variant=AttentionWithSink(sink),
    )
    o = wrapper.run(q, k, v)

    gqa_group_ratio = NUM_QO_HEADS // num_kv_heads
    k_repeated = torch.repeat_interleave(k, gqa_group_ratio, dim=1)
    v_repeated = torch.repeat_interleave(v, gqa_group_ratio, dim=1)
    o_ref = attention_varlen_sink_ref(
        q,
        k_repeated,
        v_repeated,
        qo_indptr,
        kv_indptr,
        False,
        SM_SCALE,
        sink,
    )

    torch.testing.assert_close(o, o_ref, rtol=RTOL, atol=ATOL)


# ---------------------------------------------------------------------------
#  11. Attention sink with MHA (num_kv_heads == NUM_QO_HEADS)
# ---------------------------------------------------------------------------

SINK_MHA_SHAPE_PARAMS = [
    (9, 256, 256),
    (1, 177, 977),
]


@pytest.mark.parametrize("batch_size,qo_len,kv_len", SINK_MHA_SHAPE_PARAMS)
@pytest.mark.parametrize("causal", [False, True])
def test_attention_prefill_attention_sink_mha(
    batch_size,
    qo_len,
    kv_len,
    causal,
):
    _skip_if_unsupported(qo_len, kv_len, causal)
    num_kv_heads = NUM_QO_HEADS

    torch.manual_seed(42)
    q = torch.randn(
        batch_size * qo_len, NUM_QO_HEADS, HEAD_DIM, dtype=DTYPE, device="cuda"
    )
    k = torch.randn(
        batch_size * kv_len, num_kv_heads, HEAD_DIM, dtype=DTYPE, device="cuda"
    )
    v = torch.randn(
        batch_size * kv_len, num_kv_heads, HEAD_DIM, dtype=DTYPE, device="cuda"
    )
    qo_indptr = (
        torch.arange(0, batch_size + 1, device="cuda", dtype=torch.int32) * qo_len
    )
    kv_indptr = (
        torch.arange(0, batch_size + 1, device="cuda", dtype=torch.int32) * kv_len
    )
    sink = torch.randn((NUM_QO_HEADS,), dtype=DTYPE, device="cuda")

    wrapper = BatchPrefillCuteDSLWrapper(
        torch.empty(128 * 1024 * 1024, device="cuda", dtype=torch.uint8),
    )
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        NUM_QO_HEADS,
        num_kv_heads,
        HEAD_DIM,
        head_dim_vo=HEAD_DIM,
        causal=causal,
        sm_scale=SM_SCALE,
        q_data_type=DTYPE,
        kv_data_type=DTYPE,
        variant=AttentionWithSink(sink),
    )
    o = wrapper.run(q, k, v)

    o_ref, _ = attention_ref(batch_size, q, k, v, causal, SM_SCALE, sink=sink)

    torch.testing.assert_close(o, o_ref, rtol=RTOL, atol=ATOL)


# ---------------------------------------------------------------------------
#  12. ALiBi (score_mod hook)
# ---------------------------------------------------------------------------


def attention_alibi_ref(
    batch_size,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
    sm_scale: float,
    alibi_slopes: torch.Tensor,
) -> torch.Tensor:
    """Reference ALiBi attention: adds slope * (kv_pos - qo_pos) before softmax."""
    qo_len = q.shape[0] // batch_size
    kv_len = k.shape[0] // batch_size
    num_qo_heads = q.shape[1]
    num_kv_heads = k.shape[1]
    head_dim_qk = q.shape[2]
    head_dim_vo = v.shape[2]

    if num_qo_heads > num_kv_heads:
        group_size = num_qo_heads // num_kv_heads
        k = torch.repeat_interleave(k, group_size, dim=1)
        v = torch.repeat_interleave(v, group_size, dim=1)

    logits = torch.einsum(
        "bmhd,bnhd->bhmn",
        q.view(batch_size, qo_len, num_qo_heads, head_dim_qk).float(),
        k.view(batch_size, kv_len, num_qo_heads, head_dim_qk).float(),
    )

    qo_pos = torch.arange(qo_len, device=q.device).view(1, 1, -1, 1)
    kv_pos = torch.arange(kv_len, device=q.device).view(1, 1, 1, -1)
    alibi_bias = alibi_slopes.view(1, -1, 1, 1) * (kv_pos - qo_pos)
    logits = (logits + alibi_bias) * sm_scale

    if causal:
        mask = torch.arange(kv_len - qo_len, kv_len, device=q.device).unsqueeze(
            1
        ) >= torch.arange(0, kv_len, device=q.device).unsqueeze(0)
    else:
        mask = torch.ones(qo_len, kv_len, device=q.device)

    logits = logits.masked_fill(mask.unsqueeze(0).unsqueeze(0) == 0, float("-inf"))

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

    return o_ref


ALIBI_SHAPE_PARAMS = [
    (1, 128, 128),
    (1, 177, 977),
    (9, 256, 256),
    (9, 177, 544),
]


@pytest.mark.parametrize("batch_size,qo_len,kv_len", ALIBI_SHAPE_PARAMS)
@pytest.mark.parametrize("causal", [False, True])
def test_attention_prefill_alibi(
    batch_size,
    qo_len,
    kv_len,
    causal,
):
    _skip_if_unsupported(qo_len, kv_len, causal)
    num_kv_heads = 8

    torch.manual_seed(42)
    q = torch.randn(
        batch_size * qo_len, NUM_QO_HEADS, HEAD_DIM, dtype=DTYPE, device="cuda"
    )
    k = torch.randn(
        batch_size * kv_len, num_kv_heads, HEAD_DIM, dtype=DTYPE, device="cuda"
    )
    v = torch.randn(
        batch_size * kv_len, num_kv_heads, HEAD_DIM, dtype=DTYPE, device="cuda"
    )
    qo_indptr = (
        torch.arange(0, batch_size + 1, device="cuda", dtype=torch.int32) * qo_len
    )
    kv_indptr = (
        torch.arange(0, batch_size + 1, device="cuda", dtype=torch.int32) * kv_len
    )

    alibi_slopes = ALiBiAttention.get_slopes(NUM_QO_HEADS).cuda()

    wrapper = BatchPrefillCuteDSLWrapper(
        torch.empty(128 * 1024 * 1024, device="cuda", dtype=torch.uint8),
    )
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        NUM_QO_HEADS,
        num_kv_heads,
        HEAD_DIM,
        head_dim_vo=HEAD_DIM,
        causal=causal,
        sm_scale=SM_SCALE,
        q_data_type=DTYPE,
        kv_data_type=DTYPE,
        variant=ALiBiAttention(alibi_slopes),
    )
    o = wrapper.run(q, k, v)

    o_ref = attention_alibi_ref(
        batch_size,
        q,
        k,
        v,
        causal,
        SM_SCALE,
        alibi_slopes,
    )

    torch.testing.assert_close(o, o_ref, rtol=RTOL, atol=ATOL)


# ---------------------------------------------------------------------------
#  13. RPE (Relative Positional Encoding — 2-D params)
# ---------------------------------------------------------------------------


def attention_rpe_ref(
    batch_size,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
    sm_scale: float,
    rpe_table: torch.Tensor,
    max_rel_dist: int,
) -> torch.Tensor:
    """Reference RPE attention: adds rpe_table[head, clamp(kv-qo+offset)] before softmax."""
    qo_len = q.shape[0] // batch_size
    kv_len = k.shape[0] // batch_size
    num_qo_heads = q.shape[1]
    num_kv_heads = k.shape[1]
    head_dim_qk = q.shape[2]
    head_dim_vo = v.shape[2]

    if num_qo_heads > num_kv_heads:
        group_size = num_qo_heads // num_kv_heads
        k = torch.repeat_interleave(k, group_size, dim=1)
        v = torch.repeat_interleave(v, group_size, dim=1)

    logits = torch.einsum(
        "bmhd,bnhd->bhmn",
        q.view(batch_size, qo_len, num_qo_heads, head_dim_qk).float(),
        k.view(batch_size, kv_len, num_qo_heads, head_dim_qk).float(),
    )

    qo_pos = torch.arange(qo_len, device=q.device).view(1, 1, -1, 1)
    kv_pos = torch.arange(kv_len, device=q.device).view(1, 1, 1, -1)
    rel_pos = (kv_pos - qo_pos + max_rel_dist).clamp(0, 2 * max_rel_dist)
    rpe_bias = rpe_table[:, rel_pos.squeeze(0).squeeze(0).long()].unsqueeze(0)
    logits = (logits + rpe_bias) * sm_scale

    if causal:
        mask = torch.arange(kv_len - qo_len, kv_len, device=q.device).unsqueeze(
            1
        ) >= torch.arange(0, kv_len, device=q.device).unsqueeze(0)
    else:
        mask = torch.ones(qo_len, kv_len, device=q.device)

    logits = logits.masked_fill(mask.unsqueeze(0).unsqueeze(0) == 0, float("-inf"))

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

    return o_ref


RPE_SHAPE_PARAMS = [
    (1, 128, 128),
    (1, 177, 977),
    (9, 256, 256),
]


@pytest.mark.parametrize("batch_size,qo_len,kv_len", RPE_SHAPE_PARAMS)
@pytest.mark.parametrize("causal", [False, True])
def test_attention_prefill_rpe(
    batch_size,
    qo_len,
    kv_len,
    causal,
):
    _skip_if_unsupported(qo_len, kv_len, causal)
    num_kv_heads = 8
    max_rel_dist = 64

    torch.manual_seed(42)
    q = torch.randn(
        batch_size * qo_len, NUM_QO_HEADS, HEAD_DIM, dtype=DTYPE, device="cuda"
    )
    k = torch.randn(
        batch_size * kv_len, num_kv_heads, HEAD_DIM, dtype=DTYPE, device="cuda"
    )
    v = torch.randn(
        batch_size * kv_len, num_kv_heads, HEAD_DIM, dtype=DTYPE, device="cuda"
    )
    qo_indptr = (
        torch.arange(0, batch_size + 1, device="cuda", dtype=torch.int32) * qo_len
    )
    kv_indptr = (
        torch.arange(0, batch_size + 1, device="cuda", dtype=torch.int32) * kv_len
    )

    rpe_table = (
        torch.randn(
            NUM_QO_HEADS, 2 * max_rel_dist + 1, dtype=torch.float32, device="cuda"
        )
        * 0.1
    )

    wrapper = BatchPrefillCuteDSLWrapper(
        torch.empty(128 * 1024 * 1024, device="cuda", dtype=torch.uint8),
    )
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        NUM_QO_HEADS,
        num_kv_heads,
        HEAD_DIM,
        head_dim_vo=HEAD_DIM,
        causal=causal,
        sm_scale=SM_SCALE,
        q_data_type=DTYPE,
        kv_data_type=DTYPE,
        variant=RPEAttention(rpe_table, max_rel_dist),
    )
    o = wrapper.run(q, k, v)

    o_ref = attention_rpe_ref(
        batch_size,
        q,
        k,
        v,
        causal,
        SM_SCALE,
        rpe_table,
        max_rel_dist,
    )

    torch.testing.assert_close(o, o_ref, rtol=RTOL, atol=ATOL)


# ---------------------------------------------------------------------------
#  14. SoftCapping regression: non-tile-aligned kv_len
# ---------------------------------------------------------------------------


def attention_soft_capping_ref(
    batch_size,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
    sm_scale: float,
    cap: float,
) -> torch.Tensor:
    """Reference SoftCapping attention: cap * tanh(score / cap) before softmax."""
    qo_len = q.shape[0] // batch_size
    kv_len = k.shape[0] // batch_size
    num_qo_heads = q.shape[1]
    num_kv_heads = k.shape[1]
    head_dim_qk = q.shape[2]
    head_dim_vo = v.shape[2]

    if num_qo_heads > num_kv_heads:
        group_size = num_qo_heads // num_kv_heads
        k = torch.repeat_interleave(k, group_size, dim=1)
        v = torch.repeat_interleave(v, group_size, dim=1)

    logits = torch.einsum(
        "bmhd,bnhd->bhmn",
        q.view(batch_size, qo_len, num_qo_heads, head_dim_qk).float(),
        k.view(batch_size, kv_len, num_qo_heads, head_dim_qk).float(),
    )

    logits = cap * torch.tanh(logits / cap)
    logits = logits * sm_scale

    if causal:
        mask = torch.arange(kv_len - qo_len, kv_len, device=q.device).unsqueeze(
            1
        ) >= torch.arange(0, kv_len, device=q.device).unsqueeze(0)
        logits = logits.masked_fill(mask.unsqueeze(0).unsqueeze(0) == 0, float("-inf"))

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

    return o_ref


SOFTCAP_SHAPE_PARAMS = [
    (1, 128, 200),
    (1, 200, 200),
    (2, 128, 300),
]


@pytest.mark.parametrize("batch_size,qo_len,kv_len", SOFTCAP_SHAPE_PARAMS)
@pytest.mark.parametrize("causal", [False])
def test_attention_prefill_soft_capping_small_cap(
    batch_size,
    qo_len,
    kv_len,
    causal,
):
    """SoftCapping with small cap and non-tile-aligned kv_len.

    Regression test: score_mod transforms masked -inf to -cap.  With cap=1.0
    and kv_len not divisible by 128, masked positions leak into softmax.
    """
    _skip_if_unsupported(qo_len, kv_len, causal)
    num_kv_heads = 8
    cap = 1.0

    torch.manual_seed(42)
    q = torch.randn(
        batch_size * qo_len, NUM_QO_HEADS, HEAD_DIM, dtype=DTYPE, device="cuda"
    )
    k = torch.randn(
        batch_size * kv_len, num_kv_heads, HEAD_DIM, dtype=DTYPE, device="cuda"
    )
    v = torch.randn(
        batch_size * kv_len, num_kv_heads, HEAD_DIM, dtype=DTYPE, device="cuda"
    )
    qo_indptr = (
        torch.arange(0, batch_size + 1, device="cuda", dtype=torch.int32) * qo_len
    )
    kv_indptr = (
        torch.arange(0, batch_size + 1, device="cuda", dtype=torch.int32) * kv_len
    )

    from flashinfer.cute_dsl.attention import SoftCappingAttention

    wrapper = BatchPrefillCuteDSLWrapper(
        torch.empty(128 * 1024 * 1024, device="cuda", dtype=torch.uint8),
    )
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        NUM_QO_HEADS,
        num_kv_heads,
        HEAD_DIM,
        head_dim_vo=HEAD_DIM,
        causal=causal,
        sm_scale=SM_SCALE,
        q_data_type=DTYPE,
        kv_data_type=DTYPE,
        variant=SoftCappingAttention(cap=cap),
    )
    o = wrapper.run(q, k, v)

    o_ref = attention_soft_capping_ref(
        batch_size,
        q,
        k,
        v,
        causal,
        SM_SCALE,
        cap,
    )

    torch.testing.assert_close(o, o_ref, rtol=RTOL, atol=ATOL)


if __name__ == "__main__":
    test_attention_prefill(4, 1024, 1024, 8, True)
    test_attention_prefill_varlen(
        [0, 256, 1024, 2048, 2560],
        32,
        True,
    )
