# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Independent global-scale signatures and live device values for MLA."""

from dataclasses import replace
from itertools import product

import pytest
import torch

pytest.importorskip("cutlass", minversion="4.7.0")

from flashinfer.attention.prims_ts import (
    BatchSparseMLADecodePagedTSWrapper,
    MLAAttentionTuning,
    batch_mla_decode_with_paged_kv_cache,
    batch_sparse_mla_decode_with_paged_kv_cache,
)
from flashinfer.attention.prims_ts import mla_decode as mla
from flashinfer.attention.prims_ts.kernels.mla_decode.sparse_policy import (
    SparseMlaProfile,
)
from tests.attention.test_attention_ts_mla_decode import (
    _assert_case_correct,
    _make_mla_case,
    _pack_mla_case,
    _plan_case,
    _policy_dict,
)
from tests.attention.test_attention_ts_sparse_mla import (
    _reference,
    make_nvfp4_cache,
    make_tensor_fp8_cache,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="Requires SM100/SM103",
)


def _scale(value, tensor):
    return (
        torch.tensor([value], device="cuda", dtype=torch.float32) if tensor else value
    )


@pytest.mark.parametrize("kinds", list(product((False, True), repeat=2)))
@pytest.mark.parametrize("family", ["1cta", "2cta"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_dense_mla_global_scales(kinds, family, dtype):
    """Float/device operands and packed Q retain the existing BF16/FP8 references."""
    case = _make_mla_case(
        batch_size=4 if family == "1cta" else 8,
        num_qo_heads=16 if family == "1cta" else 128,
        max_kv_len=4097 if family == "1cta" else 2048,
        qkv_dtype=dtype,
        seq_len_q=1,
        seed=32144,
    )
    qo_indptr = None
    if family == "2cta":
        case, qo_indptr = _pack_mla_case(case, [1] * 8)
    values = (case.softmax_scale, case.output_scale)
    scales = tuple(
        _scale(value, kind) for value, kind in zip(values, kinds, strict=True)
    )
    wrapper = _plan_case(
        case,
        qo_indptr=qo_indptr,
        max_q_len=1,
        softmax_scale=scales[0],
        output_scale=scales[1],
        lse_mode="base2",
    )
    policy = _policy_dict(wrapper)
    assert policy["kernel"] == (
        "throughput_latency_1cta" if family == "1cta" else "throughput_2cta"
    )
    out = torch.empty(
        (*case.query.shape[:-1], 512), device="cuda", dtype=torch.bfloat16
    )
    lse = torch.empty(case.query.shape[:-1], device="cuda", dtype=torch.float32)

    def run():
        return wrapper.run(
            case.query,
            case.kv_cache,
            case.block_tables,
            case.seq_lens_kv,
            qo_indptr=qo_indptr,
            out=out,
            lse=lse,
            validate=False,
        )

    run()
    _assert_case_correct(out, case, policy, qo_indptr=qo_indptr)
    graph = torch.cuda.CUDAGraph()
    previous_sync_mode = torch.cuda.get_sync_debug_mode()
    try:
        torch.cuda.set_sync_debug_mode("error")
        with torch.cuda.graph(graph):
            run()
    finally:
        torch.cuda.set_sync_debug_mode(previous_sync_mode)
    changed = tuple(
        value * (1.5 if kind else 1.0)
        for value, kind in zip(values, kinds, strict=True)
    )
    for scale, kind, value in zip(scales, kinds, changed, strict=True):
        if kind:
            scale.fill_(value)
    graph.replay()
    changed_case = replace(case, softmax_scale=changed[0], output_scale=changed[1])
    _assert_case_correct(out, changed_case, policy, qo_indptr=qo_indptr)
    reference_wrapper = _plan_case(
        changed_case, qo_indptr=qo_indptr, max_q_len=1, lse_mode="base2"
    )
    expected, expected_lse = reference_wrapper.run(
        case.query,
        case.kv_cache,
        case.block_tables,
        case.seq_lens_kv,
        qo_indptr=qo_indptr,
        softmax_scale=changed[0],
        output_scale=changed[1],
        return_lse=True,
    )
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    torch.testing.assert_close(lse, expected_lse, rtol=1e-6, atol=1e-6)
    for index, name in enumerate(("softmax_scale", "output_scale")):
        with pytest.raises(TypeError, match="same scalar/tensor type"):
            wrapper.run(
                case.query,
                case.kv_cache,
                case.block_tables,
                case.seq_lens_kv,
                qo_indptr=qo_indptr,
                **{name: _scale(values[index], not kinds[index])},
            )
    eager = batch_mla_decode_with_paged_kv_cache(
        case.query,
        case.kv_cache,
        case.block_tables,
        case.seq_lens_kv,
        max_q_len=1,
        max_kv_len=case.max_kv_len,
        qo_indptr=qo_indptr,
        softmax_scale=scales[0],
        output_scale=scales[1],
    )
    torch.testing.assert_close(eager, expected, rtol=0, atol=0)
    # Fresh tensor addresses and changed values reuse this exact ABI specialization.
    other = _plan_case(
        changed_case,
        qo_indptr=qo_indptr,
        max_q_len=1,
        lse_mode="base2",
        softmax_scale=_scale(changed[0], kinds[0]),
        output_scale=_scale(changed[1], kinds[1]),
    )
    assert other._plan_state.compiled is wrapper._plan_state.compiled
    torch.cuda.synchronize()


@pytest.mark.parametrize("kinds", list(product((False, True), repeat=3)))
@pytest.mark.parametrize("family,split", [("keep", 1), ("2cta", 2)])
def test_sparse_mla_global_scales(kinds, family, split, monkeypatch):
    """All eight signatures preserve independent FP8/NVFP4 source scaling."""
    monkeypatch.setattr(
        mla,
        "select_sparse_mla_profile",
        lambda **_: SparseMlaProfile(
            family=family,
            tile_size_q=64 if family == "keep" else 128,
            split_kv=split,
            gather_issue_warps=1 if family == "2cta" else 4,
            balanced_registers=family == "2cta",
        ),
    )
    torch.manual_seed(127)
    rows, heads, topk = 2, 128, 129
    query = (torch.randn(rows, heads, 512, device="cuda") * 0.2).to(torch.float8_e4m3fn)
    primary, primary_ref = make_tensor_fp8_cache(2, 64)
    extra, extra_ref = make_nvfp4_cache(2, 32)
    indices = torch.randint(64, (rows, topk), device="cuda", dtype=torch.int32)
    extra_indices = torch.randint(32, (rows, topk), device="cuda", dtype=torch.int32)
    lengths = torch.tensor([97, topk], device="cuda", dtype=torch.int32)
    extra_lengths = torch.tensor([topk, 71], device="cuda", dtype=torch.int32)
    indices[:, 3] = -1
    extra_indices[:, 7] = -1
    offsets = torch.tensor([0, rows], device="cuda", dtype=torch.int32)
    sinks = torch.linspace(-2, 1, heads, device="cuda")
    values = (0.17, 0.75, 0.5)
    scales = tuple(
        _scale(value, kind) for value, kind in zip(values, kinds, strict=True)
    )
    plan = dict(
        device=query.device,
        batch_size=1,
        num_qo_heads=heads,
        max_topk=topk,
        max_extra_topk=topk,
        packed_query=True,
        max_q_len=rows,
        q_dtype=query.dtype,
        kv_cache_format="fp8",
        extra_kv_cache_format="nvfp4",
        return_lse=True,
        use_attention_sinks=True,
        tuning=MLAAttentionTuning(0.0),
    )
    wrapper = BatchSparseMLADecodePagedTSWrapper()
    wrapper.plan(
        **plan,
        softmax_scale=scales[0],
        output_scale=scales[1],
        kv_cache_scale=scales[2],
    )
    out = torch.empty(rows, heads, 512, device="cuda", dtype=torch.bfloat16)
    lse = torch.empty(rows, heads, device="cuda", dtype=torch.float32)
    inputs = dict(
        extra_kv_cache=extra,
        extra_indices=extra_indices,
        extra_lengths=extra_lengths,
        qo_indptr=offsets,
        attention_sinks=sinks,
    )

    def run():
        wrapper.run(
            query, primary, indices, lengths, out=out, lse=lse, validate=False, **inputs
        )

    def check(current):
        ref, ref_lse, bound = _reference(
            query,
            primary_ref * current[2],
            extra_ref,
            indices,
            extra_indices,
            lengths=lengths,
            extra_lengths=extra_lengths,
            attention_sinks=sinks,
            softmax_scale=current[0],
            output_scale=current[1],
        )
        assert ((out.double() - ref).abs() <= bound).all()
        torch.testing.assert_close(lse.double(), ref_lse, atol=2e-4, rtol=1e-4)

    run()
    check(values)
    graph = torch.cuda.CUDAGraph()
    previous_sync_mode = torch.cuda.get_sync_debug_mode()
    try:
        torch.cuda.set_sync_debug_mode("error")
        with torch.cuda.graph(graph):
            run()
    finally:
        torch.cuda.set_sync_debug_mode(previous_sync_mode)
    changed = tuple(
        value * (1.5 if kind else 1.0)
        for value, kind in zip(values, kinds, strict=True)
    )
    for scale, kind, value in zip(scales, kinds, changed, strict=True):
        if kind:
            scale.fill_(value)
    graph.replay()
    check(changed)
    eager, eager_lse = batch_sparse_mla_decode_with_paged_kv_cache(
        query,
        primary,
        indices,
        lengths,
        max_q_len=rows,
        softmax_scale=scales[0],
        output_scale=scales[1],
        kv_cache_scale=scales[2],
        kv_cache_format="fp8",
        extra_kv_cache_format="nvfp4",
        return_lse=True,
        tuning=MLAAttentionTuning(0.0),
        **inputs,
    )
    torch.testing.assert_close(out, eager, rtol=0, atol=0)
    torch.testing.assert_close(lse, eager_lse, rtol=0, atol=0)
    torch.cuda.synchronize()


def test_sparse_mla_tensor_cache_scale_requires_fp8():
    with pytest.raises(ValueError, match="FP8 primary pool"):
        BatchSparseMLADecodePagedTSWrapper().plan(
            "cuda", 1, 8, max_topk=1, kv_cache_scale=_scale(1.0, True)
        )


@pytest.mark.parametrize(
    "family,split", [("keep", 1), ("keep", 2), ("2cta", 1), ("2cta", 2)]
)
def test_sparse_mla_global_scales_fused_output(family, split, monkeypatch):
    """Device multipliers precede fused output quantization in stores and reducers."""
    from tests.attention.test_attention_ts_sparse_mla import (
        _dequant_output,
        _inverse_output_rope,
    )

    monkeypatch.setattr(
        mla,
        "select_sparse_mla_profile",
        lambda **_: SparseMlaProfile(
            family=family,
            tile_size_q=64 if family == "keep" else 128,
            split_kv=split,
            gather_issue_warps=1 if family == "2cta" else 4,
            balanced_registers=family == "2cta",
        ),
    )
    torch.manual_seed(129)
    rows, heads, topk = 2, 128, 129
    query = (torch.randn(rows, 1, heads, 512, device="cuda") * 0.2).to(
        torch.float8_e4m3fn
    )
    primary, primary_ref = make_tensor_fp8_cache(2, 64)
    indices = torch.randint(64, (rows, topk), device="cuda", dtype=torch.int32)
    lengths = torch.tensor([97, topk], device="cuda", dtype=torch.int32)
    positions = torch.tensor([[0], [1]], device="cuda", dtype=torch.int32)
    theta = torch.randn(2, 32, device="cuda")
    cos_sin = torch.cat((theta.cos(), theta.sin()), -1)
    values = (0.17, 0.75, 0.5)
    tensors = tuple(_scale(value, True) for value in values)
    plan = dict(
        device=query.device,
        batch_size=rows,
        num_qo_heads=heads,
        max_topk=topk,
        q_dtype=query.dtype,
        kv_cache_format="fp8",
        return_lse=True,
        output_quant_block_size=128,
        out_scale_format="fp32",
        num_output_groups=2,
        tuning=MLAAttentionTuning(0.0),
    )
    wrapper = BatchSparseMLADecodePagedTSWrapper()
    wrapper.plan(
        **plan,
        softmax_scale=tensors[0],
        output_scale=tensors[1],
        kv_cache_scale=tensors[2],
    )
    inputs = dict(token_positions=positions, cos_sin_cache=cos_sin)
    out, out_scale, lse = wrapper.run(query, primary, indices, lengths, **inputs)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        wrapper.run(
            query,
            primary,
            indices,
            lengths,
            out=out,
            out_scale=out_scale,
            lse=lse,
            validate=False,
            **inputs,
        )
    changed = (0.26, 1.125, 0.75)
    for tensor, value in zip(tensors, changed, strict=True):
        tensor.fill_(value)
    graph.replay()
    # Scale-output buffers receive quantization metadata; they are not global inputs.
    scalar = BatchSparseMLADecodePagedTSWrapper()
    scalar.plan(
        **plan,
        softmax_scale=changed[0],
        output_scale=changed[1],
        kv_cache_scale=changed[2],
    )
    expected, expected_scale, expected_lse = scalar.run(
        query, primary, indices, lengths, **inputs
    )
    torch.testing.assert_close(out.float(), expected.float(), rtol=0, atol=0)
    torch.testing.assert_close(out_scale, expected_scale, rtol=0, atol=0)
    torch.testing.assert_close(lse, expected_lse, rtol=0, atol=0)
    ref, ref_lse, bound = _reference(
        query,
        primary_ref * changed[2],
        None,
        indices,
        lengths=lengths,
        softmax_scale=changed[0],
        output_scale=changed[1],
    )
    ref = _inverse_output_rope(ref, positions, cos_sin)
    bound = bound.reshape(rows, heads, 512)
    c = cos_sin[positions.flatten(), :32].abs()[:, None]
    s = cos_sin[positions.flatten(), 32:].abs()[:, None]
    bx, by = bound[..., 448::2].clone(), bound[..., 449::2].clone()
    bound[..., 448::2] = bx * c + by * s
    bound[..., 449::2] = by * c + bx * s
    actual, quant_scales = _dequant_output(out, out_scale, 128)
    actual = actual.reshape(rows, heads, 512)
    rounding = quant_scales.repeat_interleave(128, dim=-1).view_as(actual) / 1024
    allowed = bound + (ref.abs() + bound) / 16 + rounding + 1e-6
    assert torch.all((actual - ref).abs() <= allowed)
    torch.testing.assert_close(lse.double(), ref_lse, atol=0.03, rtol=0.01)
    torch.cuda.synchronize()


@pytest.mark.parametrize("reload", [False, True])
def test_sparse_mla_d576_device_scales(reload, monkeypatch):
    """The native D576 reload fallback and direct sparse route both load scales."""
    from tests.attention.test_attention_ts_sparse_mla import _d576_inputs

    monkeypatch.setattr(mla, "prefer_sparse_d576_reload", lambda **_: reload)
    monkeypatch.setattr(
        mla,
        "select_sparse_mla_profile",
        lambda **_: SparseMlaProfile(family="keep", tile_size_q=64, split_kv=2),
    )
    query, cache, indices, lengths = _d576_inputs(2, 64, 513, pool=1024, seed=43)
    scales = (_scale(0.03, True), _scale(0.75, True))
    wrapper = BatchSparseMLADecodePagedTSWrapper()
    wrapper.plan(
        query.device,
        2,
        64,
        max_topk=513,
        q_dtype=query.dtype,
        qk_rope_head_dim=64,
        return_lse=True,
        tuning=MLAAttentionTuning(0.0),
        softmax_scale=scales[0],
        output_scale=scales[1],
    )
    assert wrapper._state["reload_d576"] is reload
    out, lse = wrapper.run(query, cache, indices, lengths)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        wrapper.run(query, cache, indices, lengths, out=out, lse=lse, validate=False)
    scales[0].fill_(0.05)
    scales[1].fill_(1.25)
    graph.replay()
    ref, ref_lse, bound = _reference(
        query,
        cache.reshape(-1, 1, 576),
        None,
        indices,
        lengths=lengths,
        softmax_scale=0.05,
        output_scale=1.25,
    )
    assert ((out.double() - ref).abs() <= bound).all()
    torch.testing.assert_close(lse.double(), ref_lse, atol=2e-4, rtol=1e-4)
    torch.cuda.synchronize()
