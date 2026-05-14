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

"""FlashInfer flashinfer solution for moe_fp8_block_scale_renormalize_naive_routing."""

from flashinfer.fused_moe.core import trtllm_fp8_block_scale_moe as _api

definition = "moe_fp8_block_scale_renormalize_naive_routing"
api = "flashinfer.fused_moe.core.trtllm_fp8_block_scale_moe"
backend = "flashinfer"
inputs = (
    "routing_logits",
    "routing_bias",
    "hidden_states",
    "hidden_states_scale",
    "gemm1_weights",
    "gemm1_weights_scale",
    "gemm2_weights",
    "gemm2_weights_scale",
    "top_k",
    "local_expert_offset",
    "routed_scaling_factor",
)
outputs = ("output",)
api_kwargs = {
    "routing_logits": "routing_logits",
    "routing_bias": "routing_bias",
    "hidden_states": "hidden_states",
    "hidden_states_scale": "hidden_states_scale",
    "gemm1_weights": "gemm1_weights",
    "gemm1_weights_scale": "gemm1_weights_scale",
    "gemm2_weights": "gemm2_weights",
    "gemm2_weights_scale": "gemm2_weights_scale",
    "top_k": "top_k",
    "local_expert_offset": "local_expert_offset",
    "routed_scaling_factor": "routed_scaling_factor",
}
constants = {
    "num_experts": 256,
    "top_k": 8,
    "num_local_experts": 32,
    "hidden_size": 7168,
    "intermediate_size": 2048,
    "gemm1_out_size": 4096,
    "num_hidden_blocks": 56,
    "num_intermediate_blocks": 16,
    "num_gemm1_out_blocks": 32,
}


def run(
    routing_logits,
    routing_bias,
    hidden_states,
    hidden_states_scale,
    gemm1_weights,
    gemm1_weights_scale,
    gemm2_weights,
    gemm2_weights_scale,
    top_k,
    local_expert_offset,
    routed_scaling_factor,
):
    result = _api(
        routing_logits=routing_logits,
        routing_bias=routing_bias,
        hidden_states=hidden_states,
        hidden_states_scale=hidden_states_scale,
        gemm1_weights=gemm1_weights,
        gemm1_weights_scale=gemm1_weights_scale,
        gemm2_weights=gemm2_weights,
        gemm2_weights_scale=gemm2_weights_scale,
        num_experts=constants["num_experts"],
        top_k=top_k,
        n_group=None,
        topk_group=None,
        intermediate_size=constants["intermediate_size"],
        local_expert_offset=local_expert_offset,
        local_num_experts=constants["num_local_experts"],
        routed_scaling_factor=routed_scaling_factor,
    )
    if result is not None:
        return result
    raise RuntimeError(
        "moe_fp8_block_scale_renormalize_naive_routing"
        + " returned None without mutating declared outputs"
    )
