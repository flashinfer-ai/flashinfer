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

"""FlashInfer flashinfer solution for gdn_mtp."""

from flashinfer.gdn_decode import gated_delta_rule_mtp as _api
from flashinfer.trace.solutions._helpers import solution_autotune

definition = "gdn_mtp"
api = "flashinfer.gdn_decode.gated_delta_rule_mtp"
backend = "flashinfer"
inputs = (
    "q",
    "k",
    "v",
    "initial_state",
    "initial_state_indices",
    "A_log",
    "a",
    "dt_bias",
    "b",
    "scale",
    "intermediate_states_buffer",
)
outputs = ("output", "final_state")
api_kwargs = {
    "q": "q",
    "k": "k",
    "v": "v",
    "initial_state": "initial_state",
    "initial_state_indices": "initial_state_indices",
    "A_log": "A_log",
    "a": "a",
    "dt_bias": "dt_bias",
    "b": "b",
    "scale": "scale",
    "intermediate_states_buffer": "intermediate_states_buffer",
}
constants = {"num_q_heads": 4, "num_k_heads": 4, "num_v_heads": 8, "head_size": 128}


def run(
    q,
    k,
    v,
    initial_state,
    initial_state_indices,
    A_log,
    a,
    dt_bias,
    b,
    scale,
    intermediate_states_buffer,
):
    with solution_autotune(
        definition,
        backend,
        q,
        k,
        v,
        initial_state,
        initial_state_indices,
        A_log,
        a,
        dt_bias,
        b,
        scale,
        intermediate_states_buffer,
    ):
        result = _api(
            q=q,
            k=k,
            v=v,
            initial_state=initial_state,
            initial_state_indices=initial_state_indices,
            A_log=A_log,
            a=a,
            dt_bias=dt_bias,
            b=b,
            scale=scale,
            intermediate_states_buffer=intermediate_states_buffer,
        )
        if result is not None:
            return result
        raise RuntimeError(
            "gdn_mtp" + " returned None without mutating declared outputs"
        )
