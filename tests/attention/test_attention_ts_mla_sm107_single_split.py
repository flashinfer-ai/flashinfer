# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Original-capacity numerical/graph coverage for the bounded SM107 MLA policy."""

from dataclasses import replace

import pytest
import torch

pytest.importorskip("cutlass", minversion="4.7.0")

from tests.attention.test_attention_ts_mla_decode import (
    _CUDA_GRAPH_ANCHOR_OWNERS,
    _FP8,
    _make_mla_case,
    _policy_dict,
)
from tests.attention.test_attention_ts_mla_kv_bounds import (
    _assert_bounded_result,
    _plan_bounds,
    _run_bounds,
)


@pytest.mark.parametrize("packed_query", (False, True), ids=("fixed", "packed"))
def test_attention_ts_mla_sm107_single_split_boundaries(packed_query: bool) -> None:
    """Reuse Q-capacity eight and K-capacity 1024 across all page boundaries."""

    if (
        not torch.cuda.is_available()
        or torch.cuda.get_device_capability() != (10, 7)
        or torch.cuda.get_device_properties("cuda").multi_processor_count != 212
    ):
        pytest.skip("This bounded policy requires SM107 with 212 SMs")

    storage = _make_mla_case(
        batch_size=1,
        num_qo_heads=96,
        max_kv_len=1024,
        qkv_dtype=_FP8,
        seq_len_q=8,
        page_size=128,
        kv_seq_lens=(1024,),
        seed=73128,
    )
    case = replace(storage, query=storage.query[0]) if packed_query else storage
    wrapper = _plan_bounds(case, max_q_len=8, packed_query=packed_query)
    plan = wrapper._plan_state
    assert plan is not None
    policy = _policy_dict(wrapper)
    assert policy["source"] == "auto"
    assert policy["kernel"] == "throughput_2cta"
    assert policy["split_kv"] == 1
    assert policy["split_capacity"] == 1
    assert policy["separate_reducer_impl"] == "none"
    assert plan.max_kv_len == 1024
    assert plan.max_q_len == 8
    assert plan.workspace_layout.kernel_workspace.byte_size == 0
    assert plan.workspace_layout.total_bytes == 0

    original_pages = storage.block_tables.clone()
    bounds_storage = torch.empty(8, dtype=torch.int32, device="cuda")
    qo_indptr = (
        torch.tensor([0, 8], dtype=torch.int32, device="cuda") if packed_query else None
    )
    page_boundaries = sorted(
        {0, 1}
        | {
            boundary + delta
            for boundary in range(128, 1025, 128)
            for delta in (-1, 0, 1)
            if 0 <= boundary + delta <= 1024
        }
    )
    # Explicitly contract, empty, grow, and restore the same captured buffers.
    kv_lengths = page_boundaries + [129, 0, 1024, 129, 1024]
    for active_q in (8, 1, 0, 8):
        # Fixed plans retain all eight physical rows; a zero bound makes a
        # row neutral. Packed plans instead vary their physical row extent.
        actual_q = active_q if packed_query else 8
        case = (
            replace(storage, query=storage.query[0, 8 - actual_q :])
            if packed_query
            else storage
        )
        if qo_indptr is not None:
            qo_indptr[1] = actual_q
        bounds = bounds_storage[:actual_q]
        bounds.fill_(1024)
        bounds[: actual_q - active_q].zero_()
        storage.seq_lens_kv.fill_(1024)
        storage.block_tables.copy_(original_pages)
        eager_out, eager_stats = _run_bounds(wrapper, case, bounds, qo_indptr=qo_indptr)
        graph_out = torch.empty_like(eager_out)
        graph_stats = torch.empty_like(eager_stats)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _run_bounds(
                wrapper,
                case,
                bounds,
                qo_indptr=qo_indptr,
                out=graph_out,
                stats=graph_stats,
                validate=False,
            )
        for kv_len in kv_lengths:
            storage.seq_lens_kv.fill_(kv_len)
            storage.block_tables.copy_(original_pages)
            storage.block_tables[:, (kv_len + 127) // 128 :].fill_(-1)
            values = (
                sorted(
                    min(kv_len, value)
                    for value in (0, 1, 127, 128, 129, 511, 1023, 1024)
                )
                if active_q == 8
                else [0] * (actual_q - active_q) + [kv_len] * active_q
            )
            bounds.copy_(torch.tensor(values, dtype=torch.int32, device="cuda"))
            _run_bounds(
                wrapper,
                case,
                bounds,
                qo_indptr=qo_indptr,
                out=eager_out,
                stats=eager_stats,
            )
            _assert_bounded_result(
                case,
                bounds,
                eager_out,
                eager_stats,
                qo_indptr=qo_indptr,
                policy=policy,
            )
            graph_out.fill_(float("nan"))
            graph_stats.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(graph_out, eager_out, rtol=0, atol=0)
            torch.testing.assert_close(graph_stats, eager_stats, rtol=0, atol=0)
            assert wrapper._plan_state is plan
        _CUDA_GRAPH_ANCHOR_OWNERS.append(
            (graph, wrapper, case, bounds, qo_indptr, graph_out, graph_stats)
        )
