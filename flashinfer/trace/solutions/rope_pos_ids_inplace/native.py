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

"""FlashInfer flashinfer solution for rope_pos_ids_inplace."""

from flashinfer.rope import apply_rope_pos_ids_inplace as _api
from flashinfer.trace.solutions._helpers import solution_autotune

definition = "rope_pos_ids_inplace"
api = "flashinfer.rope.apply_rope_pos_ids_inplace"
backend = "flashinfer"
inputs = ("q", "k", "pos_ids", "rotary_dim", "interleave", "rope_scale", "rope_theta")
outputs = ("q", "k")
api_kwargs = {
    "q": "q",
    "k": "k",
    "pos_ids": "pos_ids",
    "rotary_dim": "rotary_dim",
    "interleave": "interleave",
    "rope_scale": "rope_scale",
    "rope_theta": "rope_theta",
}
constants = {"num_q_heads": 32, "num_k_heads": 8, "head_dim": 128}


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
        return q, k
