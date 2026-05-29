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

"""FlashInfer fa3 solution for single_prefill."""

from flashinfer.prefill import single_prefill_with_kv_cache as _api
from flashinfer.trace.solutions._helpers import solution_autotune

definition = "single_prefill"
api = "flashinfer.prefill.single_prefill_with_kv_cache"
backend = "fa3"
inputs = ("q", "k", "v")
outputs = ("output",)
api_kwargs = {"q": "q", "k": "k", "v": "v"}
constants = {"num_qo_heads": 32, "num_kv_heads": 8, "head_dim": 128}


def run(q, k, v):
    with solution_autotune(
        definition,
        backend,
        q,
        k,
        v,
    ):
        result = _api(
            q=q,
            k=k,
            v=v,
            backend=backend,
        )
        if result is not None:
            return result
        raise RuntimeError(
            "single_prefill" + " returned None without mutating declared outputs"
        )
