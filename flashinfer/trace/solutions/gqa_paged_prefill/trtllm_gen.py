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

"""FlashInfer trtllm-gen solution for gqa_paged_prefill."""

from types import SimpleNamespace

import torch

from flashinfer.prefill import BatchPrefillWithPagedKVCacheWrapper
from flashinfer.trace.solutions._helpers import (
    solution_autotune,
    default_paged_metadata,
    full_last_page_len,
    workspace,
)

definition = "gqa_paged_prefill"
api = "flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper.run"
backend = "trtllm-gen"
inputs = ("q", "k_cache", "v_cache", "qo_indptr", "kv_indptr", "kv_indices", "sm_scale")
outputs = ("output", "lse")
api_kwargs = {"q": "q", "paged_kv_cache": ("k_cache", "v_cache")}
constants = {"num_qo_heads": 32, "num_kv_heads": 8, "head_dim": 128, "page_size": 16}
requires_setup = True

_state = None


def _require_state():
    if _state is None:
        raise RuntimeError("Call setup(...) before benchmarking run(...).")
    return _state


def setup(q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
    global _state
    total_q, num_qo_heads, head_dim = q.shape
    num_pages, page_size, num_kv_heads, _ = k_cache.shape
    if kv_indptr is None or kv_indices is None:
        kv_indptr, kv_indices = default_paged_metadata(1, num_pages, q.device)
    if qo_indptr is None:
        qo_indptr = torch.tensor([0, total_q], dtype=torch.int32, device=q.device)
    wrapper = BatchPrefillWithPagedKVCacheWrapper(
        workspace(q.device), "NHD", backend=backend
    )
    plan_kwargs = {"q_data_type": q.dtype, "kv_data_type": k_cache.dtype}
    if sm_scale is not None:
        plan_kwargs["sm_scale"] = float(sm_scale)
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        full_last_page_len(kv_indptr, page_size),
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        causal=True,
        **plan_kwargs,
    )
    _state = SimpleNamespace(
        wrapper=wrapper,
        out=torch.empty_like(q),
        lse=torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=q.device),
    )


def run(q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
    with solution_autotune(
        definition,
        backend,
        q,
        k_cache,
        v_cache,
        qo_indptr,
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
