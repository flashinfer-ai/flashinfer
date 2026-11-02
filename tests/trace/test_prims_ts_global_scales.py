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
from flashinfer.trace.templates.attention import _prims_ts_scale_trace_variant


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
    "family", ["decode", "mla", "block_sparse", "paged_block_sparse"]
)
@pytest.mark.parametrize("bound", [False, True])
@pytest.mark.parametrize(
    "scale_types", [(False, False), (False, True), (True, False), (True, True)]
)
def test_prims_ts_global_scale_trace_types(family, bound, scale_types):
    scales = tuple(
        torch.empty(1, device="meta") if tensor else 0.25 for tensor in scale_types
    )
    api, kwargs, names = _trace_case(family, bound, scales)
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
    if bound:
        # Supplying the same operand kinds explicitly gives the same schema as
        # inheriting defaults. Rebinding an address never changes trace identity.
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


@pytest.mark.parametrize(
    "family", ["decode", "mla", "block_sparse", "paged_block_sparse"]
)
def test_prims_ts_float_scale_trace_keeps_values_runtime(family):
    """Float defaults/overrides share one schema, just as explicit float inputs do."""
    api, kwargs, names = _trace_case(family, True, (0.25, 1.5))
    inherited = fi_trace(api, **kwargs)
    explicit = fi_trace(api, **kwargs, **dict(zip(names, (0.25, 1.5), strict=True)))
    assert inherited == explicit
    assert "global_scale_size" not in inherited["axes"]
    assert "_scales_" not in inherited["name"]
    for name in names:
        # Scalar inputs describe their ABI, not the value of one invocation.
        assert inherited["inputs"][name] == {
            "shape": None,
            "dtype": "float32",
            "optional": True,
        }
    api.__self__._plan_state.output_scale = 2.5
    assert fi_trace(api, **kwargs) == inherited


@pytest.mark.parametrize("bound", [False, True])
def test_prims_ts_scalar_tensor_trace(bound):
    """Scalar-shaped scales retain the normalized device-operand schema."""
    scalar = torch.empty((), device="meta", dtype=torch.float32)
    api, kwargs, names = _trace_case("decode", bound, (scalar, 1.0))
    definition = fi_trace(api, **kwargs)
    assert definition["inputs"][names[0]]["shape"] == ["global_scale_size"]
    assert definition["axes"]["global_scale_size"]["value"] == 1
    assert definition == fi_trace(api, **{**kwargs, names[0]: scalar.view(1)})


@pytest.mark.parametrize(
    "family", ["decode", "mla", "block_sparse", "paged_block_sparse"]
)
@pytest.mark.parametrize("bound", [False, True])
@pytest.mark.parametrize("scale_index", [0, 1])
@pytest.mark.parametrize(
    "shape,dtype",
    [
        ((0,), torch.float32),
        ((2,), torch.float32),
        ((1, 1), torch.float32),
        ((1,), torch.float16),
    ],
)
def test_prims_ts_trace_rejects_invalid_scale_metadata(
    family, bound, scale_index, shape, dtype
):
    scales = [0.25, 1.0]
    invalid = torch.empty(shape, dtype=dtype, device="meta")
    scales[scale_index] = invalid
    api, kwargs, names = _trace_case(family, bound, scales)
    with pytest.raises(
        ValueError, match=f"{names[scale_index]} must be a contiguous float32 tensor"
    ):
        fi_trace(api, **kwargs)
    if bound:
        # Both an inherited operand and an explicit run override must satisfy
        # the fixed one-element tensor schema before TraceApply can select it.
        scales[scale_index] = torch.empty(1, device="meta")
        api, kwargs, _ = _trace_case(family, True, scales)
        with pytest.raises(
            ValueError,
            match=f"{names[scale_index]} must be a contiguous float32 tensor",
        ):
            fi_trace(api, **kwargs, **{names[scale_index]: invalid})


@pytest.mark.parametrize(
    "family", ["decode", "mla", "block_sparse", "paged_block_sparse"]
)
@pytest.mark.parametrize("bound", [False, True])
def test_prims_ts_scale_variants_are_registered(family, bound):
    api, _, names = _trace_case(family, bound, (0.25, 1.0))
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
    for base in bases:
        assert (
            _prims_ts_scale_trace_variant(base, names, (False, False), bound=bound)
            is base
        )
        for kinds in ((False, True), (True, False), (True, True)):
            variant = _prims_ts_scale_trace_variant(base, names, kinds, bound=bound)
            assert any(template is variant for template in templates)
    for kinds in ((False, False), (False, True), (True, False), (True, True)):
        scales = tuple(
            torch.empty((), device="meta") if tensor else 0.25 for tensor in kinds
        )
        api, kwargs, _ = _trace_case(family, bound, scales)
        if bound:
            kwargs["self"] = api.__self__
        selected = dispatch(**kwargs)
        if bound and family in ("decode", "mla"):
            # These wrappers deliberately synthesize plan-specific schemas;
            # TraceApply must continue falling back for dynamic identities.
            assert all(template is not selected for template in templates)
        else:
            assert any(template is selected for template in templates)
