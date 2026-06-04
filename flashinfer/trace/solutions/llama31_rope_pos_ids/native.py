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

"""FlashInfer flashinfer solution for llama31_rope_pos_ids."""

from flashinfer.rope import apply_llama31_rope_pos_ids as _api
from flashinfer.trace.solutions._helpers import solution_autotune

definition = "llama31_rope_pos_ids"
api = "flashinfer.rope.apply_llama31_rope_pos_ids"
backend = "flashinfer"
inputs = (
    "q",
    "k",
    "pos_ids",
    "rotary_dim",
    "interleave",
    "rope_scale",
    "rope_theta",
    "low_freq_factor",
    "high_freq_factor",
    "old_context_len",
)
outputs = ("q_rope", "k_rope")
api_kwargs = {
    "q": "q",
    "k": "k",
    "pos_ids": "pos_ids",
    "rotary_dim": "rotary_dim",
    "interleave": "interleave",
    "rope_scale": "rope_scale",
    "rope_theta": "rope_theta",
    "low_freq_factor": "low_freq_factor",
    "high_freq_factor": "high_freq_factor",
    "old_context_len": "old_context_len",
}


def run(
    q,
    k,
    pos_ids,
    rotary_dim,
    interleave,
    rope_scale,
    rope_theta,
    low_freq_factor,
    high_freq_factor,
    old_context_len,
):
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
        low_freq_factor,
        high_freq_factor,
        old_context_len,
    ):
        result = _api(
            q=q,
            k=k,
            pos_ids=pos_ids,
            rotary_dim=rotary_dim,
            interleave=interleave,
            rope_scale=rope_scale,
            rope_theta=rope_theta,
            low_freq_factor=low_freq_factor,
            high_freq_factor=high_freq_factor,
            old_context_len=old_context_len,
        )
        if result is not None:
            return result
        raise RuntimeError(
            "llama31_rope_pos_ids" + " returned None without mutating declared outputs"
        )
