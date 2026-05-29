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

"""FlashInfer flashinfer solution for rope_cos_sin_cache."""

from flashinfer.rope import apply_rope_with_cos_sin_cache as _api
from flashinfer.trace.solutions._helpers import solution_autotune

definition = "rope_cos_sin_cache"
api = "flashinfer.rope.apply_rope_with_cos_sin_cache"
backend = "flashinfer"
inputs = ("positions", "query", "key", "head_size", "cos_sin_cache", "is_neox")
outputs = ("query_out", "key_out")
api_kwargs = {
    "positions": "positions",
    "query": "query",
    "key": "key",
    "head_size": "head_size",
    "cos_sin_cache": "cos_sin_cache",
    "is_neox": "is_neox",
}
constants = {
    "num_q_heads_x_head_size": 4096,
    "num_k_heads_x_head_size": 1024,
    "head_size": 128,
    "rotary_dim": 128,
}


def run(positions, query, key, head_size, cos_sin_cache, is_neox):
    with solution_autotune(
        definition,
        backend,
        positions,
        query,
        key,
        head_size,
        cos_sin_cache,
        is_neox,
    ):
        result = _api(
            positions=positions,
            query=query,
            key=key,
            head_size=head_size,
            cos_sin_cache=cos_sin_cache,
            is_neox=is_neox,
        )
        if result is not None:
            return result
        raise RuntimeError(
            "rope_cos_sin_cache" + " returned None without mutating declared outputs"
        )
