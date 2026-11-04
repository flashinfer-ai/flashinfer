# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU contracts for scale schemas; meta scales cannot be read back as scalars."""

from types import SimpleNamespace
import inspect

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from flashinfer.attention.prims_ts._scales import normalize_scale, resolve_scale
from flashinfer.attention.prims_ts.block_sparse import (
    BlockSparsePagedTSWrapper,
    BlockSparseTSWrapper,
    block_sparse_attention,
    block_sparse_attention_with_paged_kv_cache,
)
from flashinfer.attention.prims_ts.decode import (
    BatchDecodePagedTSWrapper,
    batch_decode_with_paged_kv_cache,
)
from flashinfer.attention.prims_ts.mla_decode import (
    BatchMLADecodePagedTSWrapper,
    batch_mla_decode_with_paged_kv_cache,
)
from flashinfer.fi_trace import fi_trace
from flashinfer.api_logging import _TRACE_DISPATCHERS, _TRACE_REGISTRY
from flashinfer.trace.templates.attention import (
    _prims_ts_scale_trace_variant,
    _prims_ts_trace_scale_values,
)


_INVALID_HOST_SCALES = [
    (0.0, ValueError),
    (-1.0, ValueError),
    (float("nan"), ValueError),
    (float("inf"), ValueError),
    (float("-inf"), ValueError),
    (True, TypeError),
    (1e40, ValueError),
    (1e-50, ValueError),
    ("0.25", TypeError),
]


def _trace_case(family, bound, scales):
    q = torch.empty(2, 1, 8, 128, dtype=torch.bfloat16)
    k = torch.empty(2, 2, 32, 128, dtype=q.dtype)
    tables = torch.arange(2, dtype=torch.int32).reshape(2, 1)
    lengths = torch.full((2,), 32, dtype=torch.int32)
    common = dict(block_tables=tables, seq_lens_kv=lengths)
    state = dict(softmax_scale=scales[0], output_scale=scales[1])
    if family == "decode":
        kwargs = dict(q=q, paged_kv_cache=(k, k), **common)
        api = batch_decode_with_paged_kv_cache
        wrapper = BatchDecodePagedTSWrapper()
        state.update(
            page_size=32,
            storage_page_size=32,
            packed_query=False,
            seq_len_q=1,
            out_dtype=q.dtype,
            mask_type="dense",
            window_left=-1,
            max_kv_len=32,
            kv_prefix_mode="dynamic",
            kv_lengths_mode="dynamic",
            planned_seq_lens_kv_device=None,
        )
    elif family == "mla":
        kwargs = dict(
            query=torch.empty(2, 1, 8, 576, dtype=q.dtype),
            kv_cache=torch.empty(2, 32, 576, dtype=q.dtype),
            **common,
        )
        api = batch_mla_decode_with_paged_kv_cache
        wrapper = BatchMLADecodePagedTSWrapper()
        state.update(
            packed_query=False,
            mask_type="dense",
            max_q_len=1,
            max_kv_len=32,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
        )
        if not bound:
            kwargs.update(kv_lora_rank=512, qk_rope_head_dim=64)
    else:
        state = dict(sm_scale=scales[0], output_scale=scales[1])
        if family == "block_sparse":
            api = block_sparse_attention
            wrapper = BlockSparseTSWrapper()
            state.update(use_block_sparse=False)
            kv = torch.empty(2, 32, 2, 128, dtype=q.dtype)
            kwargs = dict(q=q, k=kv, v=kv)
            if not bound:
                kwargs.update(
                    use_block_sparse=False,
                    block_indptr=None,
                    block_indices=None,
                    q_block_size=2,
                    kv_block_size=32,
                )
        else:
            api = block_sparse_attention_with_paged_kv_cache
            wrapper = BlockSparsePagedTSWrapper()
            kwargs = dict(q=q, paged_kv_cache=(k, k), **common)
            kwargs.update(
                block_indptr=torch.arange(5, dtype=torch.int32)
                .unfold(0, 2, 1)
                .reshape(2, 2, 2),
                block_indices=torch.zeros(4, dtype=torch.int32),
            )
            if not bound:
                kwargs.update(q_block_size=2, kv_block_size=32, max_seq_len_kv=32)
        names = ("sm_scale", "output_scale")
    if family in ("decode", "mla"):
        names = ("softmax_scale", "output_scale")
    if bound:
        wrapper._plan_state = SimpleNamespace(**state)
        return wrapper.run, kwargs, names
    kwargs.update(zip(names, scales, strict=True))
    return api, kwargs, names


@pytest.mark.parametrize(
    "scale_types", [(False, False), (False, True), (True, False), (True, True)]
)
def test_prims_ts_global_scale_trace_types(scale_types):
    scales = tuple(
        torch.empty((), device="meta") if tensor else 0.25 for tensor in scale_types
    )
    api, kwargs, names = _trace_case("decode", True, scales)
    definition = fi_trace(api, **kwargs)
    for name, tensor in zip(names, scale_types, strict=True):
        descriptor = definition["inputs"][name]
        assert descriptor["dtype"] == "float32"
        assert descriptor["shape"] == (["global_scale_size"] if tensor else None)
    if any(scale_types):
        assert definition["name"].endswith(
            "_scales_" + "".join("t" if tensor else "f" for tensor in scale_types)
        )
        assert definition["axes"]["global_scale_size"]["value"] == 1
    # Inherited scalar-shaped tensors and explicit vector replacements share
    # the same schema, regardless of address or host scale value.
    overrides = {
        name: torch.empty(1, device="meta") if tensor else 0.75
        for name, tensor in zip(names, scale_types, strict=True)
    }
    assert fi_trace(api, **kwargs, **overrides) == definition
    for name, tensor in zip(names, scale_types, strict=True):
        with pytest.raises(ValueError, match="planned float or tensor type"):
            fi_trace(
                api,
                **kwargs,
                **{name: 1.0 if tensor else torch.empty(1, device="meta")},
            )


def test_prims_ts_float_scale_trace_keeps_values_runtime():
    api, kwargs, names = _trace_case("decode", True, (0.25, 1.5))
    inherited = fi_trace(api, **kwargs)
    explicit = fi_trace(api, **kwargs, **dict(zip(names, (0.25, 1.5), strict=True)))
    assert inherited == explicit
    assert "global_scale_size" not in inherited["axes"]
    assert "_scales_" not in inherited["name"]
    for name in names:
        assert inherited["inputs"][name] == {
            "shape": None,
            "dtype": "float32",
            "optional": True,
        }
    api.__self__._plan_state.output_scale = 2.5
    assert fi_trace(api, **kwargs) == inherited


@pytest.mark.parametrize("invalid,error_type", _INVALID_HOST_SCALES)
def test_prims_ts_trace_rejects_invalid_host_scales(invalid, error_type):
    with pytest.raises(error_type, match="softmax_scale"):
        _prims_ts_trace_scale_values(
            {"softmax_scale": invalid}, ("softmax_scale", "output_scale"), bound=False
        )


@pytest.mark.parametrize(
    "shape,dtype",
    [
        ((0,), torch.float32),
        ((2,), torch.float32),
        ((1, 1), torch.float32),
        ((1,), torch.float64),
    ],
)
def test_global_scale_invalid_tensor_metadata(shape, dtype):
    # Schema-only traces and runtime CUDA bindings enforce the same tensor ABI.
    invalid = torch.empty(shape, dtype=dtype, device="meta")
    with pytest.raises(ValueError, match="softmax_scale .* float32 tensor"):
        _prims_ts_trace_scale_values(
            {"softmax_scale": invalid}, ("softmax_scale",), bound=False
        )
    with FakeTensorMode():
        invalid = torch.empty(shape, dtype=dtype, device="cuda:0")
    with pytest.raises(ValueError, match="CUDA float32 tensor"):
        normalize_scale(invalid, "softmax_scale", torch.device("cuda:0"))


@pytest.mark.parametrize("scale_device", ["cpu", "meta", "cuda:1"])
def test_global_scale_wrong_device(scale_device):
    with FakeTensorMode():
        query = torch.empty(1, device="cuda:0")
        invalid = torch.empty((), device=scale_device)
    with pytest.raises(ValueError, match="softmax_scale .* on cuda:0"):
        _prims_ts_trace_scale_values(
            {"query": query, "softmax_scale": invalid}, ("softmax_scale",), bound=False
        )
    with pytest.raises(ValueError, match="CUDA float32 tensor"):
        normalize_scale(invalid, "softmax_scale", query.device)


def test_prims_ts_trace_prefers_planned_device_and_scale():
    with FakeTensorMode():
        planned = torch.empty((), device="cuda:0")
        replacement = torch.empty(1, device="cuda:0")
        other_device = torch.empty(1, device="cuda:1")
    state = SimpleNamespace(device=planned.device, softmax_scale=planned)
    kwargs = {"self": SimpleNamespace(_plan_state=state), "q": other_device}
    inherited = _prims_ts_trace_scale_values(kwargs, ("softmax_scale",), bound=True)
    assert inherited["softmax_scale"]._base is planned
    explicit = _prims_ts_trace_scale_values(
        {**kwargs, "softmax_scale": replacement}, ("softmax_scale",), bound=True
    )
    assert explicit["softmax_scale"] is replacement
    with pytest.raises(ValueError, match="softmax_scale .* on cuda:0"):
        _prims_ts_trace_scale_values(
            {**kwargs, "softmax_scale": other_device}, ("softmax_scale",), bound=True
        )
    state.softmax_scale = other_device
    with pytest.raises(ValueError, match="softmax_scale .* on cuda:0"):
        _prims_ts_trace_scale_values(kwargs, ("softmax_scale",), bound=True)


def test_prims_ts_trace_omitted_scale_defaults():
    names = ("softmax_scale", "output_scale")
    assert _prims_ts_trace_scale_values({}, names, bound=False) == dict.fromkeys(names)
    state = SimpleNamespace(softmax_scale=0.25, output_scale=1.5)
    kwargs = {"self": SimpleNamespace(_plan_state=state), "softmax_scale": None}
    assert _prims_ts_trace_scale_values(kwargs, names, bound=True) == {
        "softmax_scale": 0.25,
        "output_scale": 1.5,
    }


@pytest.mark.parametrize("shape", [(), (1,)])
def test_global_scale_tensor_normalization_and_resolution(shape):
    with FakeTensorMode():
        scalar = torch.empty(shape, device="cuda:0")
        replacement = torch.empty(1, device="cuda:0")
    normalized = normalize_scale(scalar, "softmax_scale", scalar.device)
    assert normalized.shape == (1,)
    assert normalized is scalar or normalized._base is scalar
    assert resolve_scale(None, normalized, "softmax_scale", scalar.device) is normalized
    assert (
        resolve_scale(replacement, normalized, "softmax_scale", scalar.device)
        is replacement
    )
    with pytest.raises(TypeError, match="same scalar/tensor type"):
        resolve_scale(0.5, normalized, "softmax_scale", scalar.device)
    with pytest.raises(TypeError, match="same scalar/tensor type"):
        resolve_scale(normalized, 0.5, "softmax_scale", scalar.device)


def test_global_scale_host_normalization_and_resolution():
    device = torch.device("cuda:0")
    expected = torch.tensor(0.1, dtype=torch.float32).item()
    assert normalize_scale(0.1, "softmax_scale", device) == expected
    assert resolve_scale(None, expected, "softmax_scale", device) == expected
    assert resolve_scale(0.25, expected, "softmax_scale", device) == 0.25


@pytest.mark.parametrize(
    "family,bound,scale_types",
    [
        ("decode", False, (True, False)),
        ("decode", True, (False, True)),
        ("mla", False, (False, True)),
        ("mla", True, (True, True)),
        ("block_sparse", False, (False, False)),
        ("block_sparse", True, (True, False)),
        ("paged_block_sparse", False, (True, True)),
        ("paged_block_sparse", True, (False, False)),
    ],
)
def test_prims_ts_scale_variants_are_registered(family, bound, scale_types):
    scales = tuple(
        torch.empty((), device="meta") if tensor else 0.25 for tensor in scale_types
    )
    api, kwargs, names = _trace_case(family, bound, scales)
    original = inspect.unwrap(api)
    if inspect.ismethod(original):
        original = original.__func__
    dispatch = _TRACE_DISPATCHERS[original]
    templates = dispatch.templates
    registered = [template for func, template, _ in _TRACE_REGISTRY if func is original]
    assert len(templates) == len({id(template) for template in templates})
    assert len(templates) == len({template.name_prefix for template in templates})
    assert [id(template) for template in registered] == [
        id(template) for template in templates
    ]
    bases = [
        template for template in templates if "_scales_" not in template.name_prefix
    ]
    assert len(templates) == 4 * len(bases)
    assert {template.name_prefix for template in templates} == {
        base.name_prefix + suffix
        for base in bases
        for suffix in ("", "_scales_ft", "_scales_tf", "_scales_tt")
    }
    for base in bases:
        assert (
            _prims_ts_scale_trace_variant(base, names, (False, False), bound=bound)
            is base
        )
    for template in templates:
        if "_scales_" in template.name_prefix:
            assert template._scale_names == names
            assert template._bound is bound
    if bound:
        kwargs["self"] = api.__self__
    selected = dispatch(**kwargs)
    if bound and family in ("decode", "mla"):
        # Dynamic plan schemas retain TraceApply's deliberate fallback behavior.
        assert all(template is not selected for template in templates)
    else:
        assert any(template is selected for template in templates)
