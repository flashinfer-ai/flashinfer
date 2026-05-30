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

"""FlashInfer cute-dsl solution for cute_dsl_fused_moe_nvfp4."""

import torch

from flashinfer.fused_moe.cute_dsl.fused_moe import (
    cute_dsl_fused_moe_nvfp4 as _api,
)
from flashinfer.trace.solutions._helpers import first_output, solution_autotune

definition = "cute_dsl_fused_moe_nvfp4"
api = "flashinfer.fused_moe.cute_dsl.fused_moe.cute_dsl_fused_moe_nvfp4"
backend = "cute-dsl"
inputs = (
    "x",
    "x_sf",
    "token_selected_experts",
    "token_final_scales",
    "w1_weight",
    "w1_weight_sf",
    "w1_alpha",
    "fc2_input_scale",
    "w2_weight",
    "w2_weight_sf",
    "w2_alpha",
    "num_experts",
    "top_k",
    "local_expert_offset",
)
outputs = ("output",)
api_kwargs = {
    "x": "x",
    "x_sf": "x_sf",
    "token_selected_experts": "token_selected_experts",
    "token_final_scales": "token_final_scales",
    "w1_weight": "w1_weight",
    "w1_weight_sf": "w1_weight_sf",
    "w1_alpha": "w1_alpha",
    "fc2_input_scale": "fc2_input_scale",
    "w2_weight": "w2_weight",
    "w2_weight_sf": "w2_weight_sf",
    "w2_alpha": "w2_alpha",
    "num_experts": "num_experts",
    "top_k": "top_k",
    "local_expert_offset": "local_expert_offset",
}


def run(
    x,
    x_sf,
    token_selected_experts,
    token_final_scales,
    w1_weight,
    w1_weight_sf,
    w1_alpha,
    fc2_input_scale,
    w2_weight,
    w2_weight_sf,
    w2_alpha,
    num_experts,
    top_k,
    local_expert_offset,
):
    with solution_autotune(
        definition,
        backend,
        x,
        x_sf,
        token_selected_experts,
        token_final_scales,
        w1_weight,
        w1_weight_sf,
        w1_alpha,
        fc2_input_scale,
        w2_weight,
        w2_weight_sf,
        w2_alpha,
        num_experts,
        top_k,
        local_expert_offset,
    ):
        result = _api(
            x=x,
            x_sf=x_sf,
            token_selected_experts=token_selected_experts,
            token_final_scales=token_final_scales,
            w1_weight=w1_weight,
            w1_weight_sf=w1_weight_sf,
            w1_alpha=w1_alpha,
            fc2_input_scale=fc2_input_scale,
            w2_weight=w2_weight,
            w2_weight_sf=w2_weight_sf,
            w2_alpha=w2_alpha,
            num_experts=num_experts,
            top_k=top_k,
            num_local_experts=w1_weight.shape[0],
            local_expert_offset=0
            if local_expert_offset is None
            else local_expert_offset,
            output_dtype=torch.bfloat16,
        )
    return first_output(result, definition)
