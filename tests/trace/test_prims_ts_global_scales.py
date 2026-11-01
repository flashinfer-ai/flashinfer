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
                kwargs["use_block_sparse"] = False
        else:
            api = block_sparse_attention_with_paged_kv_cache
            wrapper = BlockSparsePagedTSWrapper()
            kwargs = dict(q=q, paged_kv_cache=(k, k), **common)
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
