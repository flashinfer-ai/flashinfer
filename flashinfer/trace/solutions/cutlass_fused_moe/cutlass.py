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

"""FlashInfer cutlass solution for cutlass_fused_moe."""

import torch

from flashinfer.fused_moe.core import cutlass_fused_moe as _api
from flashinfer.trace.solutions._helpers import first_output, solution_autotune

definition = "cutlass_fused_moe"
api = "flashinfer.fused_moe.core.cutlass_fused_moe"
backend = "cutlass"
inputs = (
    "input",
    "token_selected_experts",
    "token_final_scales",
    "fc1_expert_weights",
    "fc2_expert_weights",
)
outputs = ("output",)
api_kwargs = {
    "input": "input",
    "token_selected_experts": "token_selected_experts",
    "token_final_scales": "token_final_scales",
    "fc1_expert_weights": "fc1_expert_weights",
    "fc2_expert_weights": "fc2_expert_weights",
}


def _output_dtype(fc2_expert_weights):
    if fc2_expert_weights.dtype in (torch.float16, torch.bfloat16, torch.float32):
        return fc2_expert_weights.dtype
    return torch.bfloat16


def run(
    input,
    token_selected_experts,
    token_final_scales,
    fc1_expert_weights,
    fc2_expert_weights,
):
    with solution_autotune(
        definition,
        backend,
        input,
        token_selected_experts,
        token_final_scales,
        fc1_expert_weights,
        fc2_expert_weights,
    ):
        result = _api(
            input=input,
            token_selected_experts=token_selected_experts,
            token_final_scales=token_final_scales,
            fc1_expert_weights=fc1_expert_weights,
            fc2_expert_weights=fc2_expert_weights,
            output_dtype=_output_dtype(fc2_expert_weights),
            quant_scales=None,
        )
    return first_output(result, definition)
