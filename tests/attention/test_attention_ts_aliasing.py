# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU coverage for Attention-TS storage-alias launch guards."""

from dataclasses import replace

import pytest
import torch

pytest.importorskip(
    "cutlass",
    minversion="4.7.0",
    reason="PrimTS attention tests require nvidia-cutlass-dsl>=4.7.0",
)

from flashinfer.attention.prims_ts._tensor_aliasing import (
    _tensor_byte_span,
    _tensors_overlap,
    _validate_out_does_not_overlap_inputs,
)
from flashinfer.attention.prims_ts.decode import (
    _DecodeRuntime,
    _validate_decode_output_aliasing,
)
from flashinfer.attention.prims_ts.mla_decode import (
    _MLARuntime,
    _validate_mla_output_aliasing,
)


def test_tensor_byte_span_includes_stride_holes_and_storage_offset() -> None:
    storage = torch.empty(64, dtype=torch.bfloat16)
    tensor = storage.as_strided((2, 3), (10, 2), storage_offset=3)

    assert _tensor_byte_span(tensor) == (
        tensor.data_ptr(),
        tensor.data_ptr() + 15 * tensor.element_size(),
    )


def test_strided_paged_views_are_conservatively_bounded() -> None:
    combined_cache = torch.empty((3, 2, 2, 4), dtype=torch.uint8)
    k_cache = combined_cache[:, 0]
    v_cache = combined_cache[:, 1]

    # The views select disjoint elements, but their outer-stride bounding spans
    # overlap. False-positive rejection is safer than under-bounding a cache.
    assert _tensors_overlap(k_cache, v_cache)


def test_empty_tensor_has_no_overlap() -> None:
    storage = torch.empty(16, dtype=torch.float32)

    assert not _tensors_overlap(storage[4:4], storage)


def test_disjoint_slices_of_one_storage_do_not_overlap() -> None:
    storage = torch.empty(16, dtype=torch.float32)

    assert not _tensors_overlap(storage[:4], storage[8:12])


def test_named_overlap_error_identifies_the_input() -> None:
    storage = torch.empty(16, dtype=torch.float32)

    with pytest.raises(ValueError, match="out must not overlap metadata storage"):
        _validate_out_does_not_overlap_inputs(
            storage[4:8],
            ("metadata", storage[:6]),
        )


def _decode_runtime() -> _DecodeRuntime:
    return _DecodeRuntime(
        q=torch.empty(8),
        k_cache=torch.empty(8),
        v_cache=torch.empty(8),
        out=torch.empty(8),
        num_physical_pages=1,
        k_page_stride=8,
        v_page_stride=8,
        bmm1_scale=1.0,
        bmm2_scale=1.0,
    )


def test_fmha_decode_guard_covers_every_live_allocation() -> None:
    for aliased_name in (
        "query",
        "k_cache",
        "v_cache",
        "seq_lens",
        "qo_indptr",
        "paged_kv_indptr",
        "paged_kv_indices",
        "paged_kv_last_page_len",
        "workspace_buffer",
    ):
        runtime = _decode_runtime()
        inputs = {
            "seq_lens": torch.empty(8),
            "qo_indptr": torch.empty(8),
            "paged_kv_indptr": torch.empty(8),
            "paged_kv_indices": torch.empty(8),
            "paged_kv_last_page_len": torch.empty(8),
            "workspace_buffer": torch.empty(8),
        }
        if aliased_name == "query":
            runtime = replace(runtime, q=runtime.out)
        elif aliased_name in ("k_cache", "v_cache"):
            runtime = replace(runtime, **{aliased_name: runtime.out})
        else:
            inputs[aliased_name] = runtime.out

        with pytest.raises(
            ValueError,
            match=rf"out must not overlap {aliased_name} storage",
        ):
            _validate_decode_output_aliasing(runtime, **inputs)


def _mla_runtime() -> _MLARuntime:
    return _MLARuntime(
        query=torch.empty(8),
        normalized_cache=torch.empty(8),
        out=torch.empty(8),
        bmm1_scale=1.0,
        bmm2_scale=1.0,
    )


def test_mla_decode_guard_covers_every_live_allocation() -> None:
    for aliased_name in (
        "query",
        "kv_cache",
        "block_tables",
        "seq_lens",
        "qo_indptr",
        "workspace_buffer",
    ):
        runtime = _mla_runtime()
        inputs = {
            "block_tables": torch.empty(8),
            "seq_lens": torch.empty(8),
            "qo_indptr": torch.empty(8),
            "workspace_buffer": torch.empty(8),
        }
        if aliased_name == "query":
            runtime = replace(runtime, query=runtime.out)
        elif aliased_name == "kv_cache":
            runtime = replace(runtime, normalized_cache=runtime.out)
        else:
            inputs[aliased_name] = runtime.out

        with pytest.raises(
            ValueError,
            match=rf"out must not overlap {aliased_name} storage",
        ):
            _validate_mla_output_aliasing(runtime, **inputs)
