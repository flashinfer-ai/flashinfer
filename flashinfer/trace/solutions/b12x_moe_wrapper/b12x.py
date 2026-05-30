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

"""FlashInfer b12x solution for b12x_moe_wrapper."""

from types import SimpleNamespace

import torch

from flashinfer.fused_moe.cute_dsl.b12x_moe import B12xMoEWrapper
from flashinfer.trace.solutions._helpers import first_output, solution_autotune

definition = "b12x_moe_wrapper"
api = "flashinfer.fused_moe.cute_dsl.b12x_moe.B12xMoEWrapper.run"
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
    "w1_alpha": "w1_alpha",
    "w2_alpha": "w2_alpha",
    "fc2_input_scale": "fc2_input_scale",
}
requires_setup = True

_state = None


def _require_state():
    if _state is None:
        raise RuntimeError("Call setup(...) before benchmarking run(...).")
    return _state


def _source_format(value):
    return "modelopt" if value is None else value


def setup(
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
    quant_mode,
    source_format,
):
    del w1_weight_sf, w2_weight_sf, token_selected_experts, token_final_scales
    del w1_alpha, w2_alpha, fc2_input_scale
    global _state
    wrapper = B12xMoEWrapper(
        num_experts=num_experts,
        top_k=top_k,
        hidden_size=x.shape[1],
        intermediate_size=w2_weight.shape[2] * 2,
        use_cuda_graph=False,
        max_num_tokens=x.shape[0],
        num_local_experts=w1_weight.shape[0],
        output_dtype=torch.bfloat16,
        device=str(x.device),
        quant_mode=quant_mode,
        source_format=_source_format(source_format),
    )
    _state = SimpleNamespace(wrapper=wrapper)


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
    quant_mode,
    source_format,
):
    del num_experts, top_k, quant_mode, source_format
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
        w1_alpha,
        w2_alpha,
        fc2_input_scale,
    ):
        state = _require_state()
        result = state.wrapper.run(
            x=x,
            w1_weight=w1_weight,
            w1_weight_sf=w1_weight_sf,
            w2_weight=w2_weight,
            w2_weight_sf=w2_weight_sf,
            token_selected_experts=token_selected_experts,
            token_final_scales=token_final_scales,
            w1_alpha=w1_alpha,
            w2_alpha=w2_alpha,
            fc2_input_scale=fc2_input_scale,
        )
    return first_output(result, definition)
