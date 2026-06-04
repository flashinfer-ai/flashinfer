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

"""FlashInfer fa3 solution for mla_paged_decode."""

from types import SimpleNamespace

import torch

from flashinfer.mla import BatchMLAPagedAttentionWrapper
from flashinfer.trace.solutions._helpers import (
    default_paged_metadata,
    workspace,
    solution_autotune,
)

definition = "mla_paged_decode"
api = "flashinfer.mla._core.BatchMLAPagedAttentionWrapper.run"
backend = "fa3"
inputs = (
    "q_nope",
    "q_pe",
    "ckv_cache",
    "kpe_cache",
    "kv_indptr",
    "kv_indices",
    "sm_scale",
)
outputs = ("output", "lse")
api_kwargs = {
    "q_nope": "q_nope",
    "q_pe": "q_pe",
    "ckv_cache": "ckv_cache",
    "kpe_cache": "kpe_cache",
}
requires_setup = True

_state = None


def _require_state():
    if _state is None:
        raise RuntimeError("Call setup(...) before benchmarking run(...).")
    return _state


def setup(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
    global _state
    batch_size, num_heads, head_dim_ckv = q_nope.shape
    _, page_size, head_dim_kpe = kpe_cache.shape
    num_pages = ckv_cache.shape[0]
    if kv_indptr is None or kv_indices is None:
        kv_indptr, kv_indices = default_paged_metadata(
            batch_size, num_pages, q_nope.device
        )
    qo_indptr = torch.arange(batch_size + 1, dtype=torch.int32, device=q_nope.device)
    kv_len_arr = (kv_indptr[1:] - kv_indptr[:-1]) * page_size
    wrapper = BatchMLAPagedAttentionWrapper(workspace(q_nope.device), backend=backend)
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_len_arr,
        num_heads,
        head_dim_ckv,
        head_dim_kpe,
        page_size,
        False,
        float(sm_scale)
        if sm_scale is not None
        else 1.0 / ((head_dim_ckv + head_dim_kpe) ** 0.5),
        q_nope.dtype,
        ckv_cache.dtype,
    )
    _state = SimpleNamespace(
        wrapper=wrapper,
        out=torch.empty_like(q_nope),
        lse=torch.empty(
            (batch_size, num_heads), dtype=torch.float32, device=q_nope.device
        ),
    )


def run(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
    with solution_autotune(
        definition,
        backend,
        q_nope,
        q_pe,
        ckv_cache,
        kpe_cache,
        kv_indptr,
        kv_indices,
        sm_scale,
    ):
        state = _require_state()
        return state.wrapper.run(
            q_nope,
            q_pe,
            ckv_cache,
            kpe_cache,
            out=state.out,
            lse=state.lse,
            return_lse=True,
        )
