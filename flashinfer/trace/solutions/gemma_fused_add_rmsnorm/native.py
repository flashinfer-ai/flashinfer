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

"""FlashInfer flashinfer solution for gemma_fused_add_rmsnorm."""

from flashinfer.norm import gemma_fused_add_rmsnorm as _api
from flashinfer.trace.solutions._helpers import solution_autotune

definition = "gemma_fused_add_rmsnorm"
api = "flashinfer.norm.gemma_fused_add_rmsnorm"
backend = "flashinfer"
inputs = ("hidden_states", "residual", "weight")
outputs = ("output", "residual")
api_kwargs = {"input": "hidden_states", "residual": "residual", "weight": "weight"}


def run(hidden_states, residual, weight):
    with solution_autotune(
        definition,
        backend,
        hidden_states,
        residual,
        weight,
    ):
        result = _api(
            input=hidden_states,
            residual=residual,
            weight=weight,
        )
        if result is not None:
            return result
        return hidden_states, residual
