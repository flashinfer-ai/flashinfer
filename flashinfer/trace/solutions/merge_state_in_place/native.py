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

"""FlashInfer flashinfer solution for merge_state_in_place."""

from flashinfer.cascade import merge_state_in_place as _api

definition = "merge_state_in_place"
api = "flashinfer.cascade.merge_state_in_place"
backend = "flashinfer"
inputs = ("v", "s", "v_other", "s_other", "mask")
outputs = ("v", "s")
api_kwargs = {
    "v": "v",
    "s": "s",
    "v_other": "v_other",
    "s_other": "s_other",
    "mask": "mask",
}
constants = {"num_heads": 32, "head_dim": 128}


def run(v, s, v_other, s_other, mask):
    result = _api(
        v=v,
        s=s,
        v_other=v_other,
        s_other=s_other,
        mask=mask,
    )
    if result is not None:
        return result
    return v, s
