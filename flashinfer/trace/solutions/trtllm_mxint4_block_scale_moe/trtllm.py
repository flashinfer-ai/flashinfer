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

"""FlashInfer trtllm solution for trtllm_mxint4_block_scale_moe."""

from flashinfer.fused_moe.core import trtllm_mxint4_block_scale_moe as _api
from flashinfer.trace.solutions._helpers import first_output, solution_autotune

definition = "trtllm_mxint4_block_scale_moe"
api = "flashinfer.fused_moe.core.trtllm_mxint4_block_scale_moe"
backend = "trtllm"
inputs = (
    "routing_logits",
    "routing_bias",
    "hidden_states",
    "gemm1_weights",
    "gemm1_weights_scale",
    "gemm2_weights",
    "gemm2_weights_scale",
    "top_k",
    "n_group",
    "topk_group",
    "local_expert_offset",
    "routed_scaling_factor",
    "routing_method_type",
)
outputs = ("output",)
api_kwargs = {
    "routing_logits": "routing_logits",
    "routing_bias": "routing_bias",
    "hidden_states": "hidden_states",
    "gemm1_weights": "gemm1_weights",
    "gemm1_weights_scale": "gemm1_weights_scale",
    "gemm2_weights": "gemm2_weights",
    "gemm2_weights_scale": "gemm2_weights_scale",
    "top_k": "top_k",
    "n_group": "n_group",
    "topk_group": "topk_group",
    "local_expert_offset": "local_expert_offset",
    "routed_scaling_factor": "routed_scaling_factor",
    "routing_method_type": "routing_method_type",
}


def run(
    routing_logits,
    routing_bias,
    hidden_states,
    gemm1_weights,
    gemm1_weights_scale,
    gemm2_weights,
    gemm2_weights_scale,
    top_k,
    n_group,
    topk_group,
    local_expert_offset,
    routed_scaling_factor,
    routing_method_type,
):
    routing_method = 0 if routing_method_type is None else routing_method_type
    with solution_autotune(
        definition,
        backend,
        routing_logits,
        routing_bias,
        hidden_states,
        gemm1_weights,
        gemm1_weights_scale,
        gemm2_weights,
        gemm2_weights_scale,
        top_k,
        n_group,
        topk_group,
        local_expert_offset,
        routed_scaling_factor,
        routing_method,
    ):
        result = _api(
            routing_logits=routing_logits,
            routing_bias=routing_bias,
            hidden_states=hidden_states,
            gemm1_weights=gemm1_weights,
            gemm1_weights_scale=gemm1_weights_scale,
            gemm1_alpha=None,
            gemm1_beta=None,
            gemm1_clamp_limit=None,
            gemm2_weights=gemm2_weights,
            gemm2_weights_scale=gemm2_weights_scale,
            num_experts=routing_logits.shape[1],
            top_k=top_k,
            n_group=n_group,
            topk_group=topk_group,
            intermediate_size=gemm2_weights.shape[2] * 2,
            local_expert_offset=local_expert_offset,
            local_num_experts=gemm1_weights.shape[0],
            routed_scaling_factor=routed_scaling_factor,
            routing_method_type=routing_method,
        )
    return first_output(result, definition)
