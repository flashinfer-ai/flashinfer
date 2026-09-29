# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Binding-level guard tests for the SM120 sparse-MLA standalone entries.

The Python wrappers keep these invariants by construction; these tests call
the compiled modules directly so a caller that bypasses them still gets loud
failures instead of silent chunk dropping, truncated integer overrides, or
out-of-range row reads.
"""

import pytest
import torch

from flashinfer.mla import nvfp4_quantize_pack_sparse_mla_cache
from flashinfer.mla._sparse_mla_sm120 import _get_sparse_mla_sm120_decode_module
from flashinfer.mla._sparse_mla_sm120._dsv4_nvfp4 import (
    get_sparse_mla_nvfp4_sm120_module,
)
from tests.attention.sparse_mla_test_utils import (
    _dequantize_nvfp4_cache,
    _dequantize_nvfp4_query,
    _ref_sparse_attn,
    _reference_sparse_attention,
    dequantize_kv_dsv4,
    quantize_kv_dsv4,
    quantize_kv_glm53_nope,
    require_sm12x,
)


@pytest.fixture
def sm12x():
    require_sm12x()


def _decode_dsv4(module, q, cache, idx, mid, mlse, out, lse, splits, cpb):
    module.sparse_mla_sm120_decode_dsv4(
        q,
        cache,
        idx,
        mid,
        mlse,
        out,
        lse,
        splits,
        512**-0.5,
        None,
        None,
        None,
        None,
        None,
        1,  # DSV4
        cpb,
        False,
        1.0,
    )


def _dsv4_decode_case(tokens, heads, topk, splits):
    q = torch.zeros(tokens, heads, 512, device="cuda", dtype=torch.bfloat16)
    cache = quantize_kv_dsv4(
        torch.full((2, 64, 1, 512), 0.5, device="cuda", dtype=torch.bfloat16)
    )
    idx = torch.zeros(tokens, topk, device="cuda", dtype=torch.int32)
    mid = torch.empty(tokens, heads, splits, 512, device="cuda", dtype=torch.bfloat16)
    mlse = torch.empty(tokens, heads, splits, device="cuda")
    out, lse = torch.empty_like(q), torch.empty(tokens, heads, device="cuda")
    return q, cache, idx, mid, mlse, out, lse


def _glm53_decode_case(tokens, heads, topk, splits):
    q = torch.zeros(tokens, heads, 512, device="cuda", dtype=torch.bfloat16)
    cache = quantize_kv_glm53_nope(
        torch.full((1, 64, 1, 512), 0.5, device="cuda", dtype=torch.bfloat16)
    )[..., :528].contiguous()
    idx = torch.zeros(tokens, topk, device="cuda", dtype=torch.int32)
    mid = torch.empty(tokens, heads, splits, 512, device="cuda", dtype=torch.bfloat16)
    mlse = torch.empty(tokens, heads, splits, device="cuda")
    out, lse = torch.empty_like(q), torch.empty(tokens, heads, device="cuda")
    return q, cache, idx, mid, mlse, out, lse


@pytest.mark.usefixtures("sm12x")
@pytest.mark.parametrize("cpb", [2**31 + 8, -(2**33)])
def test_decode_dsv4_cpb_override_out_of_int_range_rejected(cpb):
    """int64 -> int narrowing of the CPB override must fail loudly."""
    module = _get_sparse_mla_sm120_decode_module()
    with pytest.raises(RuntimeError, match="chunks_per_block_override"):
        _decode_dsv4(module, *_dsv4_decode_case(1, 16, 128, 2), 2, cpb)


@pytest.mark.usefixtures("sm12x")
def test_decode_dsv3_2_cpb_override_out_of_int_range_rejected():
    module = _get_sparse_mla_sm120_decode_module()
    q, cache, idx, mid, mlse, out, lse = _glm53_decode_case(1, 8, 2176, 34)
    with pytest.raises(RuntimeError, match="chunks_per_block_override"):
        module.sparse_mla_sm120_decode_dsv3_2(
            q,
            cache,
            idx,
            mid,
            mlse,
            out,
            lse,
            34,
            512**-0.5,
            None,
            None,
            3,
            2**31 + 8,
            1.0,
        )


@pytest.mark.usefixtures("sm12x")
def test_nvfp4_decode_cpb_out_of_int_range_rejected():
    tokens, heads, topk, splits = 1, 16, 128, 2
    q = torch.zeros(tokens, heads, 512, device="cuda", dtype=torch.bfloat16)
    cache = nvfp4_quantize_pack_sparse_mla_cache(
        torch.zeros(2, 64, 512, device="cuda", dtype=torch.bfloat16)
    )
    idx = torch.zeros(tokens, topk, device="cuda", dtype=torch.int32)
    mid = torch.empty(tokens, heads, splits, 512, device="cuda", dtype=torch.bfloat16)
    mlse = torch.empty(tokens, heads, splits, device="cuda")
    out, lse = torch.empty_like(q), torch.empty(tokens, heads, device="cuda")
    with pytest.raises(RuntimeError, match="cpb"):
        get_sparse_mla_nvfp4_sm120_module().sparse_mla_sm120_nvfp4_decode(
            q,
            cache,
            idx,
            mid,
            mlse,
            out,
            lse,
            splits,
            512**-0.5,
            None,
            None,
            None,
            None,
            None,
            2**31 + 8,
            False,
            1.0,
        )


@pytest.mark.usefixtures("sm12x")
def test_decode_dsv4_num_splits_must_cover_chunk_plan():
    """topk=128 is two 64-wide chunks; cpb=1 keeps both active, so a
    num_splits=1 grid would silently drop the tail chunk."""
    module = _get_sparse_mla_sm120_decode_module()
    with pytest.raises(RuntimeError, match="num_splits"):
        _decode_dsv4(module, *_dsv4_decode_case(1, 16, 128, 1), 1, 1)

    # Exact coverage (num_splits == chunk capacity) stays legal and correct,
    # including an in-range override past the chunk count (auto-resolved).
    torch.manual_seed(20260915)
    tokens, heads, topk, splits = 2, 16, 128, 2
    kv_bf16 = (
        torch.randn(2, 64, 1, 512, device="cuda", dtype=torch.bfloat16) / 10
    ).clamp(-1, 1)
    cache = quantize_kv_dsv4(kv_bf16)
    q = (
        torch.randn(tokens, heads, 512, device="cuda", dtype=torch.bfloat16) / 10
    ).clamp(-1, 1)
    idx = torch.randint(0, 128, (tokens, topk), device="cuda", dtype=torch.int32)

    def run(cpb):
        mid = torch.empty(
            tokens, heads, splits, 512, device="cuda", dtype=torch.bfloat16
        )
        mlse = torch.empty(tokens, heads, splits, device="cuda")
        out, lse = torch.empty_like(q), torch.empty(tokens, heads, device="cuda")
        _decode_dsv4(module, q, cache, idx, mid, mlse, out, lse, splits, cpb)
        return out, lse

    out, lse = run(1)
    auto_out, auto_lse = run(100)
    assert torch.equal(out, auto_out) and torch.equal(lse, auto_lse)
    ref_out, ref_lse = _ref_sparse_attn(
        q, dequantize_kv_dsv4(cache), idx, 512**-0.5, 512
    )
    torch.testing.assert_close(out, ref_out, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(lse, ref_lse, atol=5e-2, rtol=5e-2)


@pytest.mark.usefixtures("sm12x")
def test_decode_dsv3_2_num_splits_must_cover_chunk_plan():
    """topk=2176 is 34 chunks; num_splits=4 with cpb=1 would drop the tail."""
    module = _get_sparse_mla_sm120_decode_module()
    q, cache, idx, mid, mlse, out, lse = _glm53_decode_case(1, 8, 2176, 4)
    with pytest.raises(RuntimeError, match="num_splits"):
        module.sparse_mla_sm120_decode_dsv3_2(
            q, cache, idx, mid, mlse, out, lse, 4, 512**-0.5, None, None, 3, 1, 1.0
        )


@pytest.mark.usefixtures("sm12x")
def test_nvfp4_decode_two_split_merge_epilogue():
    """H=16/topk=128 with cpb=1 resolves to two active splits, i.e. the
    dedicated merge2 epilogue kernel."""
    torch.manual_seed(20260915)
    tokens, heads, topk, splits = 2, 16, 128, 2
    kv_bf16 = (
        torch.randn(4, 64, 1, 512, device="cuda", dtype=torch.bfloat16) / 10
    ).clamp(-1, 1)
    q = (
        torch.randn(tokens, heads, 512, device="cuda", dtype=torch.bfloat16) / 10
    ).clamp(-1, 1)
    indices = torch.randint(0, 4 * 64, (tokens, topk), device="cuda", dtype=torch.int32)
    cache = nvfp4_quantize_pack_sparse_mla_cache(kv_bf16)
    mid = torch.empty(tokens, heads, splits, 512, device="cuda", dtype=torch.bfloat16)
    mlse = torch.empty(tokens, heads, splits, device="cuda")
    out, lse = torch.empty_like(q), torch.empty(tokens, heads, device="cuda")
    get_sparse_mla_nvfp4_sm120_module().sparse_mla_sm120_nvfp4_decode(
        q,
        cache,
        indices,
        mid,
        mlse,
        out,
        lse,
        splits,
        512**-0.5,
        None,
        None,
        None,
        None,
        None,
        1,
        False,
        1.0,
    )
    reference, reference_lse = _reference_sparse_attention(
        _dequantize_nvfp4_query(q), _dequantize_nvfp4_cache(cache), indices, 512**-0.5
    )
    torch.testing.assert_close(out, reference, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(lse, reference_lse, atol=2e-2, rtol=2e-2)


@pytest.mark.usefixtures("sm12x")
@pytest.mark.parametrize("extra_pbs", [2, 64])
def test_prefill_mg_dual_unaligned_extra_topk(extra_pbs):
    """extra_topk % 64 != 0 leaves the last extra tile partial; its masked
    index reads must stay inside the declared extra-indices row."""
    torch.manual_seed(20260915 + extra_pbs)
    tokens, heads, topk, extra_topk = 3, 32, 128, 72
    main_blocks, main_pbs = 8, 64
    extra_blocks = max((extra_topk + extra_pbs - 1) // extra_pbs * 2, 16)
    main_s_kv = main_blocks * main_pbs
    extra_s_kv = extra_blocks * extra_pbs

    main_bf16 = (
        torch.randn(main_blocks, main_pbs, 1, 512, device="cuda", dtype=torch.bfloat16)
        / 10
    ).clamp(-1, 1)
    extra_bf16 = (
        torch.randn(
            extra_blocks, extra_pbs, 1, 512, device="cuda", dtype=torch.bfloat16
        )
        / 10
    ).clamp(-1, 1)
    main_packed = quantize_kv_dsv4(main_bf16)
    extra_packed = quantize_kv_dsv4(extra_bf16)

    q = (
        torch.randn(tokens, heads, 512, device="cuda", dtype=torch.bfloat16) / 10
    ).clamp(-1, 1)
    idx = torch.randint(0, main_s_kv, (tokens, topk), device="cuda", dtype=torch.int32)
    exidx = torch.randint(
        0, extra_s_kv, (tokens, extra_topk), device="cuda", dtype=torch.int32
    )
    idx[:, topk // 2 :] = -1
    exidx[:, extra_topk // 2 :] = -1

    out = torch.empty(tokens, heads, 512, device="cuda", dtype=torch.bfloat16)
    lse = torch.empty(tokens, heads, device="cuda")
    _get_sparse_mla_sm120_decode_module().sparse_mla_sm120_paged_attention(
        q,
        main_packed,
        idx,
        out,
        lse,
        512**-0.5,
        1,  # DSV4
        3,  # PREFILL_MG_DUAL
        None,
        None,
        extra_packed,
        exidx,
        None,
        False,
    )

    virtual_kv = torch.cat(
        [
            dequantize_kv_dsv4(main_packed).reshape(-1, 512),
            dequantize_kv_dsv4(extra_packed).reshape(-1, 512),
        ]
    ).reshape(-1, 1, 1, 512)
    virtual_idx = torch.cat(
        [idx, torch.where(exidx < 0, exidx, exidx + main_s_kv)], dim=-1
    )
    ref_out, ref_lse = _ref_sparse_attn(q, virtual_kv, virtual_idx, 512**-0.5, 512)
    torch.testing.assert_close(out, ref_out, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(lse, ref_lse, atol=5e-2, rtol=5e-2)
