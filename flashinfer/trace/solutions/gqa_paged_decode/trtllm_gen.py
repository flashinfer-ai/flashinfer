# Copyright (c) 2025 by FlashInfer team.
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

"""FlashInfer trtllm-gen solution for gqa_paged_decode."""

from types import SimpleNamespace

import torch

from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper
from flashinfer.trace.solutions._helpers import (
    solution_autotune,
    default_paged_metadata,
    full_last_page_len,
    workspace,
)

definition = "gqa_paged_decode"
api = "flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper.run"
backend = "trtllm-gen"
inputs = ("q", "k_cache", "v_cache", "kv_indptr", "kv_indices", "sm_scale")
outputs = ("output", "lse")
api_kwargs = {"q": "q", "paged_kv_cache": ("k_cache", "v_cache")}
requires_setup = True

_state = None


def _require_state():
    if _state is None:
        raise RuntimeError("Call setup(...) before benchmarking run(...).")
    return _state


def setup(q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
    global _state
    batch_size, num_qo_heads, head_dim = q.shape
    num_pages, page_size, num_kv_heads, _ = k_cache.shape
    if kv_indptr is None or kv_indices is None:
        kv_indptr, kv_indices = default_paged_metadata(batch_size, num_pages, q.device)
    wrapper = BatchDecodeWithPagedKVCacheWrapper(
        workspace(q.device), "NHD", backend=backend
    )
    plan_kwargs = {"q_data_type": q.dtype, "kv_data_type": k_cache.dtype}
    if sm_scale is not None:
        plan_kwargs["sm_scale"] = float(sm_scale)
    wrapper.plan(
        kv_indptr,
        kv_indices,
        full_last_page_len(kv_indptr, page_size),
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        **plan_kwargs,
    )
    _state = SimpleNamespace(
        wrapper=wrapper,
        out=torch.empty_like(q),
        lse=torch.empty(
            (batch_size, num_qo_heads), dtype=torch.float32, device=q.device
        ),
    )


def run(q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
    with solution_autotune(
        definition,
        backend,
        q,
        k_cache,
        v_cache,
        kv_indptr,
        kv_indices,
        sm_scale,
    ):
        state = _require_state()
        return state.wrapper.run(
            q,
            (k_cache, v_cache),
            out=state.out,
            lse=state.lse,
            return_lse=True,
        )
