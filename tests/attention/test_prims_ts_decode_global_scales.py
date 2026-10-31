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

"""Live independent global scale operands across paged decode entry points."""

import itertools
from dataclasses import replace

import pytest
import torch

pytest.importorskip("cutlass", minversion="4.7.0a0")

from flashinfer.attention.prims_ts import decode
from flashinfer.attention.prims_ts._compile_cache import _compile_cache_key
from flashinfer.attention.prims_ts.kernels.fmha_decode import fmha_decode_config
from flashinfer.utils import get_compute_capability
from tests.attention.sage_test_utils import ieee_float32_references  # noqa: F401

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or get_compute_capability(torch.device("cuda")) not in ((10, 0), (10, 3), (10, 7)),
    reason="PrimTS decode requires an SM100-class GPU",
)

_KINDS = tuple(itertools.product((False, True), repeat=2))


def _scale(value, tensor):
    return torch.tensor([value], device="cuda") if tensor else value


def _inputs(recipe="bf16", length=1024):
    generator = torch.Generator(device="cuda").manual_seed(81)
    dtype = torch.float16 if recipe == "fp16" else torch.bfloat16
    qk_dtype = torch.float8_e4m3fn if recipe == "fp8_qkv" else dtype
    v_dtype = torch.float8_e4m3fn if recipe in ("fp8_qkv", "fp8_v") else dtype
    q = (torch.randn((1, 8, 128), device="cuda", generator=generator) * 0.5).to(
        qk_dtype
    )
    shape = (length // 16, 1, 16, 128)
    k = (torch.randn(shape, device="cuda", generator=generator) * 0.5).to(qk_dtype)
    v = torch.randn(shape, device="cuda", generator=generator).to(v_dtype)
    lengths = torch.tensor([length], dtype=torch.int32, device="cuda")
    table = torch.arange(length // 16, dtype=torch.int32, device="cuda").view(1, -1)
    out = torch.empty(q.shape, dtype=dtype, device=q.device)
    return q, (k, v), lengths, table, out


def _plan(args, scales, **kwargs):
    q, (k, v), lengths, _, out = args
    wrapper = decode.BatchDecodePagedTSWrapper()
    wrapper.plan(
        q.device,
        1,
        8,
        1,
        128,
        16,
        k.shape[0] * 16,
        q_dtype=q.dtype,
        k_dtype=k.dtype,
        v_dtype=v.dtype,
        out_dtype=out.dtype,
        softmax_scale=scales[0],
        output_scale=scales[1],
        **kwargs,
    )
    return wrapper


def _reference(args, values):
    q, (k, v), _, _, _ = args
    scores = q[0].float() @ k[:, 0].reshape(-1, 128).float().T * values[0]
    return (scores.softmax(-1) @ v[:, 0].reshape(-1, 128).float() * values[1])[None]


def _run(wrapper, args, **kwargs):
    q, cache, lengths, table, out = args
    return wrapper.run(q, cache, lengths, table, out=out, validate=False, **kwargs)


def _assert_result(args, values):
    torch.testing.assert_close(
        args[-1].float(), _reference(args, values), atol=0.02, rtol=0.05
    )


@pytest.mark.parametrize("kinds", _KINDS)
@pytest.mark.parametrize("recipe", ("fp16", "bf16", "fp8_v", "fp8_qkv"))
def test_decode_scale_types_graph_replay_and_one_shot(kinds, recipe):
    args = _inputs(recipe)
    values = (0.125, 0.75)
    scales = tuple(
        _scale(value, kind) for value, kind in zip(values, kinds, strict=False)
    )
    wrapper = _plan(args, scales)
    _run(wrapper, args)
    _assert_result(args, values)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _run(wrapper, args)
    changed = tuple(
        value * 1.5 if kind else value
        for value, kind in zip(values, kinds, strict=False)
    )
    for scale, value, kind in zip(scales, changed, kinds, strict=False):
        if kind:
            scale.fill_(value)
    graph.replay()
    _assert_result(args, changed)

    # Current tensor bindings must not be replaced by the plan's original pointers.
    overrides = tuple(
        _scale(value * 0.5, kind) for value, kind in zip(values, kinds, strict=False)
    )
    _run(wrapper, args, softmax_scale=overrides[0], output_scale=overrides[1])
    _assert_result(args, tuple(value * 0.5 for value in values))
    _run(wrapper, args)
    _assert_result(args, changed)

    q, cache, lengths, table, out = args
    decode.batch_decode_with_paged_kv_cache(
        q,
        cache,
        table,
        lengths,
        max_kv_len=cache[0].shape[0] * 16,
        softmax_scale=scales[0],
        output_scale=scales[1],
        out=out,
        workspace_buffer=wrapper._plan_state.workspace_buffer,
        validate=False,
    )
    _assert_result(args, changed)
    for index, name in enumerate(("softmax_scale", "output_scale")):
        with pytest.raises(TypeError, match="same scalar/tensor type"):
            _run(wrapper, args, **{name: _scale(values[index], not kinds[index])})


@pytest.mark.parametrize("kinds", _KINDS)
@pytest.mark.parametrize(
    "mode",
    (
        "disabled",
        "gmem_reduction",
        "gmem_reduction_with_separate_kernel",
        "cluster_smem_reduction",
    ),
)
def test_decode_global_scales_reduction_modes(monkeypatch, kinds, mode):
    original = fmha_decode_config.make_decode_config

    def configured(*args, **kwargs):
        profile = dict(
            use_keeps_mma_ab=False,
            tile_size_q=16 if mode == "gmem_reduction_with_separate_kernel" else 8,
            tile_size_kv=128,
            use_persistent_scheduler=False,
        )
        kwargs["args"] = (kwargs["args"], profile) if kwargs.get("args") else profile
        splits = 1 if mode == "disabled" else 2
        kwargs.update(
            split_kv_mode=mode, splits_kv=splits, max_splits_kv=splits, auto_tuner=False
        )
        return original(*args, **kwargs)

    monkeypatch.setattr(fmha_decode_config, "make_decode_config", configured)
    decode._resolve_decode_launch_spec.cache_clear()
    try:
        args = _inputs(length=2048)
        values = (0.1875, 1.25)
        scales = tuple(
            _scale(value, kind) for value, kind in zip(values, kinds, strict=False)
        )
        wrapper = _plan(args, scales)
        cfg = wrapper._plan_state.config
        assert cfg.use_split_kv == (mode != "disabled")
        assert cfg.use_separate_reduction_kernel == (
            mode == "gmem_reduction_with_separate_kernel"
        )
        assert cfg.use_cluster_smem_reduction == (mode == "cluster_smem_reduction")
        _run(wrapper, args)
        _assert_result(args, values)
    finally:
        decode._resolve_decode_launch_spec.cache_clear()


@pytest.mark.parametrize("kinds", _KINDS)
def test_prepared_decode_inherits_global_scales_and_replays(kinds):
    args = _inputs()
    q, cache, lengths, table, out = args
    values = (0.0625, 1.5)
    scales = tuple(
        _scale(value, kind) for value, kind in zip(values, kinds, strict=False)
    )
    wrapper = _plan(args, scales)
    workspace = torch.empty_like(wrapper._plan_state.workspace_buffer)
    plan = decode.prepare_prims_ts_batch_decode_with_kv_cache(
        q,
        cache,
        workspace,
        table,
        lengths,
        cache[0].shape[0] * 16,
        out=out,
        softmax_scale=scales[0],
        output_scale=scales[1],
    )
    plan.run(q, out=out)
    _assert_result(args, values)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        plan.run(q, out=out, validate=False)
    changed = tuple(
        value * 1.5 if kind else value
        for value, kind in zip(values, kinds, strict=False)
    )
    for scale, value, kind in zip(scales, changed, kinds, strict=False):
        if kind:
            scale.fill_(value)
    graph.replay()
    _assert_result(args, changed)


def test_decode_compile_and_persistent_keys_distinguish_scale_types(monkeypatch):
    specs = []
    original = decode._get_compiled_decode

    def record(spec):
        specs.append(spec)
        return original(spec)

    monkeypatch.setattr(decode, "_get_compiled_decode", record)
    args = _inputs()
    for kinds in _KINDS:
        for values in ((0.125, 0.75), (0.25, 1.5)):
            scales = tuple(
                _scale(value, kind) for value, kind in zip(values, kinds, strict=False)
            )
            _plan(args, scales)
    assert all(specs[i] == specs[i + 1] for i in range(0, 8, 2))
    assert len(set(specs)) == 4
    keys = {_compile_cache_key(spec, decode._COMPILE_OPTIONS) for spec in specs}
    assert None not in keys and len(keys) == 4
    assert replace(specs[0], scale_types=(True, True)) == specs[-1]
