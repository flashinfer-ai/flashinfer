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

"""Independent context global-scale signatures and live graph operands."""

import itertools

import pytest
import torch

pytest.importorskip("cutlass", minversion="4.7.0a0")

from flashinfer.attention.prims_ts import (
    BatchPrefillPagedTSWrapper,
    BatchPrefillTSWrapper,
    batch_prefill,
    batch_prefill_with_paged_kv_cache,
)
import flashinfer.attention.prims_ts.context as context_module
from flashinfer.utils import get_compute_capability
from tests.attention.sage_test_utils import ieee_float32_references  # noqa: F401

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or get_compute_capability(torch.device("cuda")) not in ((10, 0), (10, 3), (10, 7)),
    reason="PrimTS context requires an SM100-class GPU",
)

_SCALE_KINDS = tuple(itertools.product((False, True), repeat=2))


def _scale(value, tensor):
    return (
        torch.tensor([value], device="cuda", dtype=torch.float32) if tensor else value
    )


def _inputs(recipe="bf16"):
    generator = torch.Generator(device="cuda").manual_seed(17)
    q = torch.randn((1, 64, 2, 128), device="cuda", generator=generator) * 0.5
    k = torch.randn((1, 128, 2, 128), device="cuda", generator=generator) * 0.5
    v = torch.randn((1, 128, 2, 128), device="cuda", generator=generator)
    qk_dtype = torch.float8_e4m3fn if recipe == "fp8_qkv" else torch.bfloat16
    v_dtype = torch.bfloat16 if recipe == "bf16" else torch.float8_e4m3fn
    return q.to(qk_dtype), k.to(qk_dtype), v.to(v_dtype)


def _arguments(q, k, v, paged):
    if not paged:
        return q, k, v
    # Convert BSHD to native page/head/token/dimension storage.
    caches = tuple(
        t.reshape(4, 32, 2, 128).transpose(1, 2).contiguous() for t in (k, v)
    )
    return (
        q[0],
        *caches,
        torch.tensor([0, 64], dtype=torch.int32, device="cuda"),
        torch.arange(4, dtype=torch.int32, device="cuda").view(1, 4),
        torch.tensor([128], dtype=torch.int32, device="cuda"),
    )


def _plan(q, k, v, paged, scales):
    wrapper = BatchPrefillPagedTSWrapper() if paged else BatchPrefillTSWrapper()
    wrapper.plan(
        device=q.device,
        batch_size=1,
        max_q_len=64,
        max_kv_len=128,
        num_qo_heads=2,
        num_kv_heads=2,
        head_dim=128,
        q_dtype=q.dtype,
        k_dtype=k.dtype,
        v_dtype=v.dtype,
        out_dtype=torch.bfloat16,
        softmax_scale=scales[0],
        output_scale=scales[1],
    )
    return wrapper


def _reference(q, k, v, scales, paged):
    scores = torch.einsum("bqhd,bkhd->bhqk", q.float(), k.float()) * scales[0]
    result = torch.einsum("bhqk,bkhd->bqhd", scores.softmax(-1), v.float()) * scales[1]
    return result[0] if paged else result


@pytest.mark.parametrize("paged", (False, True))
@pytest.mark.parametrize("kinds", _SCALE_KINDS)
@pytest.mark.parametrize("recipe", ("bf16", "fp8_v", "fp8_qkv"))
def test_context_scale_kinds_and_graph_replay(paged, kinds, recipe):
    q, k, v = _inputs(recipe)
    values = (0.125, 0.75)
    scales = tuple(
        _scale(value, kind) for value, kind in zip(values, kinds, strict=False)
    )
    wrapper = _plan(q, k, v, paged, scales)
    args = _arguments(q, k, v, paged)
    out = torch.empty_like(args[0], dtype=torch.bfloat16)
    wrapper.run(*args, out=out, validate=False)
    torch.testing.assert_close(
        out.float(), _reference(q, k, v, values, paged), atol=2e-2, rtol=5e-2
    )

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        wrapper.run(*args, out=out, validate=False)
    # Change only device operands: floats captured by value intentionally stay fixed.
    changed = tuple(
        value * 1.5 if kind else value
        for value, kind in zip(values, kinds, strict=False)
    )
    for operand, value, kind in zip(scales, changed, kinds, strict=False):
        if kind:
            operand.fill_(value)
    graph.replay()
    torch.testing.assert_close(
        out.float(), _reference(q, k, v, changed, paged), atol=2e-2, rtol=5e-2
    )

    # Overrides bind current addresses/values without mutating the planned defaults.
    overrides = tuple(
        _scale(value * 0.5, kind) for value, kind in zip(values, kinds, strict=False)
    )
    wrapper.run(
        *args,
        out=out,
        softmax_scale=overrides[0],
        output_scale=overrides[1],
        validate=False,
    )
    torch.testing.assert_close(
        out.float(),
        _reference(q, k, v, tuple(value * 0.5 for value in values), paged),
        atol=2e-2,
        rtol=5e-2,
    )


@pytest.mark.parametrize("paged", (False, True))
@pytest.mark.parametrize("kinds", _SCALE_KINDS)
def test_context_one_shot_global_scales(paged, kinds):
    q, k, v = _inputs()
    values = (0.125, 0.75)
    scales = tuple(
        _scale(value, kind) for value, kind in zip(values, kinds, strict=False)
    )
    kwargs = dict(softmax_scale=scales[0], output_scale=scales[1])
    if paged:
        actual = batch_prefill_with_paged_kv_cache(*_arguments(q, k, v, True), **kwargs)
    else:
        actual = batch_prefill(q, k, v, **kwargs)
    torch.testing.assert_close(
        actual.float(), _reference(q, k, v, values, paged), atol=2e-2, rtol=5e-2
    )


@pytest.mark.parametrize("paged", (False, True))
@pytest.mark.parametrize("kinds", _SCALE_KINDS)
def test_context_scale_type_guard_without_validation(paged, kinds):
    q, k, v = _inputs()
    scales = tuple(
        _scale(value, kind) for value, kind in zip((0.125, 0.75), kinds, strict=False)
    )
    wrapper = _plan(q, k, v, paged, scales)
    args = _arguments(q, k, v, paged)
    for name, kind in zip(("softmax_scale", "output_scale"), kinds, strict=False):
        with pytest.raises(TypeError, match="same scalar/tensor type"):
            wrapper.run(*args, **{name: _scale(0.5, not kind)}, validate=False)
        for invalid in (
            torch.ones(1),
            torch.ones(1, dtype=torch.float64, device="cuda"),
            torch.ones(2, device="cuda"),
            torch.ones((), device="cuda"),
        ):
            with pytest.raises(ValueError, match="CUDA float32 tensor of shape"):
                wrapper.run(*args, **{name: invalid}, validate=False)


@pytest.mark.parametrize("paged", (False, True))
def test_context_compile_identity_contains_scale_kinds_only(monkeypatch, paged):
    q, k, v = _inputs()
    specs = []

    def fake_compile(spec):
        specs.append(spec)
        return lambda *_args: None, ()

    monkeypatch.setattr(
        context_module,
        "_get_compiled_paged_context" if paged else "_get_compiled_context",
        fake_compile,
    )
    for kinds in _SCALE_KINDS:
        _plan(q, k, v, paged, tuple(_scale(0.5, kind) for kind in kinds))
        _plan(q, k, v, paged, tuple(_scale(0.75, kind) for kind in kinds))
        assert specs[-1] == specs[-2]
        assert specs[-1].scale_types == kinds
    assert len(set(specs)) == 4


@pytest.mark.parametrize("kinds", _SCALE_KINDS)
def test_context_sage_global_scale_replay(kinds):
    from flashinfer.attention.prims_ts import SageAttentionConfig
    from tests.attention.test_attention_ts_context_sage import (
        _plan as plan_sage,
        _random_params,
        _reference as sage_reference,
    )

    torch.manual_seed(19)
    config = SageAttentionConfig(q_block_size=1, k_block_size=16, v_mean=True)
    shape = dict(
        batch_size=1, seq_len_q=64, seq_len_kv=256, num_qo_heads=2, num_kv_heads=2
    )
    values = (0.125, 0.75)
    scales = tuple(
        _scale(value, kind) for value, kind in zip(values, kinds, strict=False)
    )
    wrapper = plan_sage(
        sage=config, softmax_scale=scales[0], output_scale=scales[1], **shape
    )
    params = _random_params(config, **shape)
    q = torch.randn((1, 64, 2, 128), device="cuda").to(torch.float8_e4m3fn)
    k = torch.randn((1, 256, 2, 128), device="cuda").to(torch.float8_e4m3fn)
    v = torch.randn_like(k.float()).to(torch.float8_e4m3fn)
    out = wrapper.run(q, k, v, sage=params)
    expected = sage_reference(q, k, v, params, config=config, softmax_scale=values[0])
    torch.testing.assert_close(out.float(), expected * values[1], rtol=8e-3, atol=2e-3)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        wrapper.run(q, k, v, sage=params, out=out, validate=False)
    changed = tuple(
        value * 1.5 if kind else value
        for value, kind in zip(values, kinds, strict=False)
    )
    for scale, value, kind in zip(scales, changed, kinds, strict=False):
        if kind:
            scale.fill_(value)
    graph.replay()
    expected = sage_reference(q, k, v, params, config=config, softmax_scale=changed[0])
    torch.testing.assert_close(out.float(), expected * changed[1], rtol=8e-3, atol=2e-3)


@pytest.mark.parametrize("kinds", _SCALE_KINDS)
def test_context_vc_global_scale_replay(kinds):
    from flashinfer.attention.prims_ts import VCAttentionConfig
    from flashinfer.attention.prims_ts.vc_attention import vc_reference
    from tests.attention.test_attention_ts_vc import _reference_quantize

    if get_compute_capability(torch.device("cuda")) == (10, 7):
        pytest.skip("VC attention is not supported on SM107")
    q, k, v = _inputs()
    # Identity ordering isolates scale behavior from preprocessing policy.
    perm = torch.arange(k.shape[1], device="cuda").view(1, 1, -1).expand(1, 2, -1)
    ops = _reference_quantize(k, v, perm=perm)
    values = (0.125, 0.75)
    scales = tuple(
        _scale(value, kind) for value, kind in zip(values, kinds, strict=False)
    )
    wrapper = BatchPrefillTSWrapper()
    wrapper.plan(
        device=q.device,
        batch_size=1,
        max_q_len=64,
        max_kv_len=128,
        num_qo_heads=2,
        num_kv_heads=2,
        head_dim=128,
        q_dtype=q.dtype,
        k_dtype=k.dtype,
        v_dtype=ops.v.dtype,
        out_dtype=torch.bfloat16,
        softmax_scale=scales[0],
        output_scale=scales[1],
        vc_config=VCAttentionConfig(),
    )
    out = wrapper.run(q, ops.k, ops.v, vc=ops.params)
    expected = vc_reference(q, ops, sm_scale=values[0]) * values[1]
    torch.testing.assert_close(out.float(), expected, rtol=1e-1, atol=1e-1)
    assert torch.linalg.vector_norm(out.float() - expected) / expected.norm() < 6e-2
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        wrapper.run(q, ops.k, ops.v, vc=ops.params, out=out, validate=False)
    changed = tuple(
        value * 1.5 if kind else value
        for value, kind in zip(values, kinds, strict=False)
    )
    for scale, value, kind in zip(scales, changed, kinds, strict=False):
        if kind:
            scale.fill_(value)
    graph.replay()
    expected = vc_reference(q, ops, sm_scale=changed[0]) * changed[1]
    torch.testing.assert_close(out.float(), expected, rtol=1e-1, atol=1e-1)
    assert torch.linalg.vector_norm(out.float() - expected) / expected.norm() < 6e-2
