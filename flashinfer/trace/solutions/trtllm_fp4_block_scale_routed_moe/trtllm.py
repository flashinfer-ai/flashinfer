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

"""FlashInfer trtllm solution for trtllm_fp4_block_scale_routed_moe."""

from flashinfer.fused_moe.core import trtllm_fp4_block_scale_routed_moe as _api
from flashinfer.trace.solutions._helpers import first_output, solution_autotune

definition = "trtllm_fp4_block_scale_routed_moe"
api = "flashinfer.fused_moe.core.trtllm_fp4_block_scale_routed_moe"
backend = "trtllm"
inputs = (
    "topk_ids",
    "routing_bias",
    "hidden_states",
    "hidden_states_scale",
    "gemm1_weights",
    "gemm1_weights_scale",
    "gemm2_weights",
    "gemm2_weights_scale",
    "num_experts",
    "top_k",
    "local_expert_offset",
    "routed_scaling_factor",
)
outputs = ("output",)
api_kwargs = {
    "topk_ids": "topk_ids",
    "routing_bias": "routing_bias",
    "hidden_states": "hidden_states",
    "hidden_states_scale": "hidden_states_scale",
    "gemm1_weights": "gemm1_weights",
    "gemm1_weights_scale": "gemm1_weights_scale",
    "gemm2_weights": "gemm2_weights",
    "gemm2_weights_scale": "gemm2_weights_scale",
    "num_experts": "num_experts",
    "top_k": "top_k",
    "local_expert_offset": "local_expert_offset",
    "routed_scaling_factor": "routed_scaling_factor",
}


def run(
    topk_ids,
    routing_bias,
    hidden_states,
    hidden_states_scale,
    gemm1_weights,
    gemm1_weights_scale,
    gemm2_weights,
    gemm2_weights_scale,
    num_experts,
    top_k,
    local_expert_offset,
    routed_scaling_factor,
):
    with solution_autotune(
        definition,
        backend,
        topk_ids,
        routing_bias,
        hidden_states,
        hidden_states_scale,
        gemm1_weights,
        gemm1_weights_scale,
        gemm2_weights,
        gemm2_weights_scale,
        num_experts,
        top_k,
        local_expert_offset,
        routed_scaling_factor,
    ):
        result = _api(
            topk_ids=topk_ids,
            routing_bias=routing_bias,
            hidden_states=hidden_states,
            hidden_states_scale=hidden_states_scale,
            gemm1_weights=gemm1_weights,
            gemm1_weights_scale=gemm1_weights_scale,
            gemm1_bias=None,
            gemm1_alpha=None,
            gemm1_beta=None,
            gemm1_clamp_limit=None,
            gemm2_weights=gemm2_weights,
            gemm2_weights_scale=gemm2_weights_scale,
            gemm2_bias=None,
            output1_scale_scalar=None,
            output1_scale_gate_scalar=None,
            output2_scale_scalar=None,
            num_experts=num_experts,
            top_k=top_k,
            n_group=None,
            topk_group=None,
            intermediate_size=gemm2_weights.shape[2] * 2,
            local_expert_offset=local_expert_offset,
            local_num_experts=gemm1_weights.shape[0],
            routed_scaling_factor=routed_scaling_factor,
        )
    return first_output(result, definition)
