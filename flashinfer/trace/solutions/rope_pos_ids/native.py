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

"""FlashInfer flashinfer solution for rope_pos_ids."""

from flashinfer.rope import apply_rope_pos_ids as _api
from flashinfer.trace.solutions._helpers import solution_autotune

definition = "rope_pos_ids"
api = "flashinfer.rope.apply_rope_pos_ids"
backend = "flashinfer"
inputs = ("q", "k", "pos_ids", "rotary_dim", "interleave", "rope_scale", "rope_theta")
outputs = ("q_rope", "k_rope")
api_kwargs = {
    "q": "q",
    "k": "k",
    "pos_ids": "pos_ids",
    "rotary_dim": "rotary_dim",
    "interleave": "interleave",
    "rope_scale": "rope_scale",
    "rope_theta": "rope_theta",
}


def run(q, k, pos_ids, rotary_dim, interleave, rope_scale, rope_theta):
    with solution_autotune(
        definition,
        backend,
        q,
        k,
        pos_ids,
        rotary_dim,
        interleave,
        rope_scale,
        rope_theta,
    ):
        result = _api(
            q=q,
            k=k,
            pos_ids=pos_ids,
            rotary_dim=rotary_dim,
            interleave=interleave,
            rope_scale=rope_scale,
            rope_theta=rope_theta,
        )
        if result is not None:
            return result
        raise RuntimeError(
            "rope_pos_ids" + " returned None without mutating declared outputs"
        )
