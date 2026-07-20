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

"""Narrow public-surface guard for PrimTS attention APIs."""

from __future__ import annotations

import inspect

import pytest

pytest.importorskip(
    "cutlass",
    minversion="4.7.0",
    reason="PrimTS attention tests require nvidia-cutlass-dsl>=4.7.0",
)

from flashinfer.attention.prims_ts import (
    BatchDecodePagedTSWrapper,
    BatchMLADecodePagedTSWrapper,
    BatchPrefillPagedTSWrapper,
    BatchPrefillTSWrapper,
    batch_prefill,
    batch_decode_mla_with_paged_kv_cache,
    batch_decode_with_paged_kv_cache,
    batch_prefill_with_paged_kv_cache,
)
from flashinfer.decode import (
    get_prims_ts_batch_decode_workspace_size,
    prims_ts_batch_decode_with_kv_cache,
)
from flashinfer.mla import (
    get_prims_ts_batch_decode_mla_workspace_size,
    prims_ts_batch_decode_with_kv_cache_mla,
)


def test_attention_ts_public_surfaces_have_no_tuning_knobs():
    surfaces = (
        BatchPrefillTSWrapper.plan,
        BatchPrefillTSWrapper.run,
        batch_prefill,
        BatchPrefillPagedTSWrapper.plan,
        BatchPrefillPagedTSWrapper.run,
        batch_prefill_with_paged_kv_cache,
        BatchDecodePagedTSWrapper.plan,
        BatchDecodePagedTSWrapper.run,
        batch_decode_with_paged_kv_cache,
        get_prims_ts_batch_decode_workspace_size,
        prims_ts_batch_decode_with_kv_cache,
        BatchMLADecodePagedTSWrapper.plan,
        BatchMLADecodePagedTSWrapper.run,
        batch_decode_mla_with_paged_kv_cache,
        get_prims_ts_batch_decode_mla_workspace_size,
        prims_ts_batch_decode_with_kv_cache_mla,
    )
    forbidden = (
        "autotuner",
        "clc",
        "config",
        "persistent",
        "profile",
        "reduction",
        "schedule",
        "single_kv",
        "split",
        "stage",
        "tile",
        "warp",
    )
    violations = [
        f"{surface.__qualname__}.{parameter}"
        for surface in surfaces
        for parameter in inspect.signature(surface).parameters
        if any(part in parameter for part in forbidden)
    ]
    assert violations == []
