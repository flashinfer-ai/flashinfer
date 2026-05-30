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

"""FlashInfer b12x solution for b12x_fused_moe."""

import torch

from flashinfer.fused_moe.cute_dsl.b12x_moe import b12x_fused_moe as _api
from flashinfer.trace.solutions._helpers import first_output, solution_autotune

definition = "b12x_fused_moe"
api = "flashinfer.fused_moe.cute_dsl.b12x_moe.b12x_fused_moe"
backend = "b12x"
inputs = (
    "x",
    "w1_weight",
    "w1_weight_sf",
    "w2_weight",
    "w2_weight_sf",
    "token_selected_experts",
    "token_final_scales",
    "num_experts",
    "top_k",
    "w1_alpha",
    "w2_alpha",
    "fc2_input_scale",
    "activation_precision",
    "quant_mode",
    "source_format",
)
outputs = ("output",)
api_kwargs = {
    "x": "x",
    "w1_weight": "w1_weight",
    "w1_weight_sf": "w1_weight_sf",
    "w2_weight": "w2_weight",
    "w2_weight_sf": "w2_weight_sf",
    "token_selected_experts": "token_selected_experts",
    "token_final_scales": "token_final_scales",
    "num_experts": "num_experts",
    "top_k": "top_k",
    "w1_alpha": "w1_alpha",
    "w2_alpha": "w2_alpha",
    "fc2_input_scale": "fc2_input_scale",
    "activation_precision": "activation_precision",
    "quant_mode": "quant_mode",
    "source_format": "source_format",
}


def _activation_precision(value):
    return "fp4" if value is None else value


def _source_format(value):
    return "modelopt" if value is None else value


def run(
    x,
    w1_weight,
    w1_weight_sf,
    w2_weight,
    w2_weight_sf,
    token_selected_experts,
    token_final_scales,
    num_experts,
    top_k,
    w1_alpha,
    w2_alpha,
    fc2_input_scale,
    activation_precision,
    quant_mode,
    source_format,
):
    with solution_autotune(
        definition,
        backend,
        x,
        w1_weight,
        w1_weight_sf,
        w2_weight,
        w2_weight_sf,
        token_selected_experts,
        token_final_scales,
        num_experts,
        top_k,
        w1_alpha,
        w2_alpha,
        fc2_input_scale,
        activation_precision,
        quant_mode,
        source_format,
    ):
        result = _api(
            x=x,
            w1_weight=w1_weight,
            w1_weight_sf=w1_weight_sf,
            w2_weight=w2_weight,
            w2_weight_sf=w2_weight_sf,
            token_selected_experts=token_selected_experts,
            token_final_scales=token_final_scales,
            num_experts=num_experts,
            top_k=top_k,
            w1_alpha=w1_alpha,
            w2_alpha=w2_alpha,
            fc2_input_scale=fc2_input_scale,
            num_local_experts=w1_weight.shape[0],
            output_dtype=torch.bfloat16,
            activation_precision=_activation_precision(activation_precision),
            quant_mode=quant_mode,
            source_format=_source_format(source_format),
        )
    return first_output(result, definition)
