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

"""FlashInfer flashinfer solution for gdn_prefill."""

from flashinfer.gdn_prefill import chunk_gated_delta_rule as _api
from flashinfer.trace.solutions._helpers import solution_autotune

definition = "gdn_prefill"
api = "flashinfer.gdn_prefill.chunk_gated_delta_rule"
backend = "flashinfer"
inputs = ("q", "k", "v", "state", "A_log", "a", "dt_bias", "b", "cu_seqlens", "scale")
outputs = ("output", "new_state")
api_kwargs = {
    "q": "q",
    "k": "k",
    "v": "v",
    "initial_state": "state",
    "g": "a",
    "beta": "b",
    "cu_seqlens": "cu_seqlens",
    "scale": "scale",
}


def run(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
    with solution_autotune(
        definition,
        backend,
        q,
        k,
        v,
        state,
        A_log,
        a,
        dt_bias,
        b,
        cu_seqlens,
        scale,
    ):
        result = _api(
            q=q,
            k=k,
            v=v,
            g=a,
            beta=b,
            scale=scale,
            initial_state=state,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
        )
        if result is not None:
            return result
        raise RuntimeError(
            "gdn_prefill" + " returned None without mutating declared outputs"
        )
