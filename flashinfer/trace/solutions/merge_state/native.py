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

"""FlashInfer flashinfer solution for merge_state."""

from flashinfer.cascade import merge_state as _api

definition = "merge_state"
api = "flashinfer.cascade.merge_state"
backend = "flashinfer"
inputs = ("v_a", "s_a", "v_b", "s_b")
outputs = ("v_merged", "s_merged")
api_kwargs = {"v_a": "v_a", "s_a": "s_a", "v_b": "v_b", "s_b": "s_b"}
constants = {"num_heads": 32, "head_dim": 128}


def run(v_a, s_a, v_b, s_b):
    result = _api(
        v_a=v_a,
        s_a=s_a,
        v_b=v_b,
        s_b=s_b,
    )
    if result is not None:
        return result
    raise RuntimeError(
        "merge_state" + " returned None without mutating declared outputs"
    )
