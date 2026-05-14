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

"""FlashInfer fa3 solution for gqa_ragged."""

from types import SimpleNamespace

import torch

from flashinfer.prefill import BatchPrefillWithRaggedKVCacheWrapper
from flashinfer.trace.solutions._helpers import workspace

definition = "gqa_ragged"
api = "flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper.run"
backend = "fa3"
inputs = ("q", "k", "v", "qo_indptr", "kv_indptr", "sm_scale")
outputs = ("output", "lse")
api_kwargs = {"q": "q", "k": "k", "v": "v"}
constants = {"num_qo_heads": 32, "num_kv_heads": 8, "head_dim": 128}
requires_setup = True

_state = None


def _require_state():
    if _state is None:
        raise RuntimeError("Call setup(...) before benchmarking run(...).")
    return _state


def setup(q, k, v, qo_indptr, kv_indptr, sm_scale):
    global _state
    total_q, num_qo_heads, head_dim = q.shape
    total_kv, num_kv_heads, _ = k.shape
    if qo_indptr is None:
        qo_indptr = torch.tensor([0, total_q], dtype=torch.int32, device=q.device)
    if kv_indptr is None:
        kv_indptr = torch.tensor([0, total_kv], dtype=torch.int32, device=q.device)
    wrapper = BatchPrefillWithRaggedKVCacheWrapper(
        workspace(q.device), "NHD", backend=backend
    )
    plan_kwargs = {"q_data_type": q.dtype, "kv_data_type": k.dtype}
    if sm_scale is not None:
        plan_kwargs["sm_scale"] = float(sm_scale)
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        causal=True,
        **plan_kwargs,
    )
    _state = SimpleNamespace(
        wrapper=wrapper,
        out=torch.empty_like(q),
        lse=torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=q.device),
    )


def run(q, k, v, qo_indptr, kv_indptr, sm_scale):
    state = _require_state()
    return state.wrapper.run(
        q,
        k,
        v,
        out=state.out,
        lse=state.lse,
        return_lse=True,
    )
