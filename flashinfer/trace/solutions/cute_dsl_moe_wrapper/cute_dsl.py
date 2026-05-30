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

"""FlashInfer cute-dsl solution for cute_dsl_moe_wrapper."""

from types import SimpleNamespace

import torch

from flashinfer.fused_moe.cute_dsl.fused_moe import CuteDslMoEWrapper
from flashinfer.trace.solutions._helpers import first_output, solution_autotune

definition = "cute_dsl_moe_wrapper"
api = "flashinfer.fused_moe.cute_dsl.fused_moe.CuteDslMoEWrapper.run"
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
}
requires_setup = True

_state = None


def _require_state():
    if _state is None:
        raise RuntimeError("Call setup(...) before benchmarking run(...).")
    return _state


def setup(
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
    del x_sf, token_selected_experts, token_final_scales
    del w1_weight_sf, w1_alpha, fc2_input_scale, w2_weight_sf, w2_alpha
    global _state
    local_offset = 0 if local_expert_offset is None else local_expert_offset
    wrapper = CuteDslMoEWrapper(
        num_experts=num_experts,
        top_k=top_k,
        hidden_size=x.shape[1] * 2,
        intermediate_size=w2_weight.shape[2] * 2,
        use_cuda_graph=False,
        max_num_tokens=x.shape[0],
        num_local_experts=w1_weight.shape[0],
        local_expert_offset=local_offset,
        output_dtype=torch.bfloat16,
        device=str(x.device),
    )
    _state = SimpleNamespace(wrapper=wrapper)


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
    del num_experts, top_k, local_expert_offset
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
    ):
        state = _require_state()
        result = state.wrapper.run(
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
        )
    return first_output(result, definition)
