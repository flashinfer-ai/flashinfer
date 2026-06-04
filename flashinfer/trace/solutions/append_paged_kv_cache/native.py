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

"""FlashInfer flashinfer solution for append_paged_kv_cache."""

from flashinfer.page import append_paged_kv_cache as _api
from flashinfer.trace.solutions._helpers import solution_autotune

definition = "append_paged_kv_cache"
api = "flashinfer.page.append_paged_kv_cache"
backend = "flashinfer"
inputs = (
    "append_key",
    "append_value",
    "batch_indices",
    "positions",
    "paged_kv_cache",
    "kv_indices",
    "kv_indptr",
    "kv_last_page_len",
)
outputs = ("paged_kv_cache",)
api_kwargs = {
    "append_key": "append_key",
    "append_value": "append_value",
    "batch_indices": "batch_indices",
    "positions": "positions",
    "paged_kv_cache": "paged_kv_cache",
    "kv_indices": "kv_indices",
    "kv_indptr": "kv_indptr",
    "kv_last_page_len": "kv_last_page_len",
}


def run(
    append_key,
    append_value,
    batch_indices,
    positions,
    paged_kv_cache,
    kv_indices,
    kv_indptr,
    kv_last_page_len,
):
    with solution_autotune(
        definition,
        backend,
        append_key,
        append_value,
        batch_indices,
        positions,
        paged_kv_cache,
        kv_indices,
        kv_indptr,
        kv_last_page_len,
    ):
        result = _api(
            append_key=append_key,
            append_value=append_value,
            batch_indices=batch_indices,
            positions=positions,
            paged_kv_cache=paged_kv_cache,
            kv_indices=kv_indices,
            kv_indptr=kv_indptr,
            kv_last_page_len=kv_last_page_len,
        )
        if result is not None:
            return result
        return paged_kv_cache
