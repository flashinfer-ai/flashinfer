# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch

from flashinfer import fi_trace
from flashinfer.attention.prims_ts import BatchSparseMLADecodePagedTSWrapper
from flashinfer.attention.prims_ts import SparseMLAPreparedMetadata
from flashinfer.trace.templates.sparse_mla import sparse_mla_wrapper_trace_dispatch
from tests.trace.test_fi_trace_template_consistency import (
    assert_template_axes_covered,
    assert_template_signature_consistency,
)


@pytest.mark.parametrize(
    "packed,layout,assume_valid_prefix", [(False, "NHD", False), (True, "HND", True)]
)
def test_sparse_trace_binds_both_sources_and_device_scalars(
    packed, layout, assume_valid_prefix
):
    wrapper = BatchSparseMLADecodePagedTSWrapper()
    wrapper._impl._state = dict(
        batch=2,
        max_q=3,
        heads=16,
        packed=packed,
        swa_page=16,
        compressed_page=2,
        capacity=256,
        ks=128,
        kc=64,
        kv_layout=layout,
        has_sinks=True,
        return_lse=True,
        assume_valid_prefix=assume_valid_prefix,
    )
    prefix = (4,) if packed else (2, 3)
    q = torch.empty(*prefix, 16, 512, dtype=torch.bfloat16)
    s = torch.empty(
        (5, 1, 16, 512) if layout == "HND" else (5, 16, 1, 512), dtype=torch.bfloat16
    )
    c = torch.empty(
        (7, 1, 2, 512) if layout == "HND" else (7, 2, 1, 512), dtype=torch.bfloat16
    )
    rows = q.numel() // (16 * 512)
    metadata = SparseMLAPreparedMetadata(
        torch.zeros(rows, 128, dtype=torch.int32),
        torch.full((rows,), 128, dtype=torch.int32),
        torch.zeros(rows, 64, dtype=torch.int32),
        torch.full((rows,), 64, dtype=torch.int32),
    )
    kwargs = dict(
        query=q,
        kv_cache=s,
        extra_kv_cache=c,
        metadata=metadata,
        q_scale=torch.tensor(1.0),
        kv_scale=torch.ones(1),
        sinks=torch.zeros(16),
        qo_indptr=torch.tensor([0, 1, 4], dtype=torch.int32) if packed else None,
    )
    template = sparse_mla_wrapper_trace_dispatch(self=wrapper, **kwargs)
    assert_template_axes_covered(template, func=wrapper.run)
    assert_template_signature_consistency(wrapper.run, template)
    result = fi_trace(wrapper.run, **kwargs)
    assert result["inputs"]["indices"]["dtype"] == "int32"
    assert result["inputs"]["extra_kv_cache"]["dtype"] == "bfloat16"
    assert result["outputs"]["lse"]["dtype"] == "float32"
    assert result["axes"]["head_dim"]["value"] == 512

    assert ("indices:valid-prefix" in result["tags"]) == assume_valid_prefix
    assert ("valid_prefix" in result["name"]) == assume_valid_prefix


@pytest.mark.parametrize("extra,packed", [(False, False), (True, True)])
def test_prepared_trace_binds_source_neutral_metadata(extra, packed):
    w = BatchSparseMLADecodePagedTSWrapper()
    w._impl._state = dict(
        batch=1,
        max_q=2,
        heads=16,
        packed=packed,
        ks=16,
        kc=8 if extra else 0,
        capacity=256,
        kv_layout="NHD",
        has_sinks=False,
        return_lse=True,
    )
    meta = SparseMLAPreparedMetadata(
        torch.zeros(2, 16, dtype=torch.int32),
        torch.full((2,), 16, dtype=torch.int32),
        torch.zeros(2, 8, dtype=torch.int32) if extra else None,
        torch.full((2,), 8, dtype=torch.int32) if extra else None,
        torch.zeros(1, 2, 256, dtype=torch.int32),
        torch.ones(1, 2, dtype=torch.int32),
        torch.ones(1, 2, dtype=torch.int32),
        torch.ones(1, 20),
    )
    kwargs = dict(
        query=torch.empty(
            (2, 16, 512) if packed else (1, 2, 16, 512), dtype=torch.bfloat16
        ),
        kv_cache=torch.empty(8, 16, 512, dtype=torch.bfloat16),
        metadata=meta,
        extra_kv_cache=torch.empty(4, 2, 512, dtype=torch.bfloat16) if extra else None,
        qo_indptr=torch.tensor([0, 2], dtype=torch.int32) if packed else None,
        kv_scale=torch.tensor(1.0),
    )
    template = sparse_mla_wrapper_trace_dispatch(self=w, **kwargs)
    assert_template_axes_covered(template, func=w.run)
    assert_template_signature_consistency(w.run, template)
    result = fi_trace(w.run, **kwargs)
    assert result["inputs"]["indices"]["dtype"] == "int32"
    assert result["inputs"]["routes"]["dtype"] == "int32"
    assert result["axes"]["primary_page_size"]["value"] == 16
    assert ("two-source" if extra else "single-source") in result["tags"]
