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

"""FlashInfer flashinfer solution for gdn_decode."""

from flashinfer.gdn_decode import gated_delta_rule_decode as _api

definition = "gdn_decode"
api = "flashinfer.gdn_decode.gated_delta_rule_decode"
backend = "flashinfer"
inputs = ("q", "k", "v", "state", "A_log", "a", "dt_bias", "b", "scale")
outputs = ("output", "new_state")
api_kwargs = {
    "q": "q",
    "k": "k",
    "v": "v",
    "state": "state",
    "A_log": "A_log",
    "a": "a",
    "dt_bias": "dt_bias",
    "b": "b",
    "scale": "scale",
}
constants = {
    "seq_len": 1,
    "num_q_heads": 4,
    "num_k_heads": 4,
    "num_v_heads": 8,
    "head_size": 128,
}


def run(q, k, v, state, A_log, a, dt_bias, b, scale):
    result = _api(
        q=q,
        k=k,
        v=v,
        state=state,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b,
        scale=scale,
    )
    if result is not None:
        return result
    raise RuntimeError(
        "gdn_decode" + " returned None without mutating declared outputs"
    )
