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

"""FlashInfer trtllm solution for moe_fp4_block_scale_ds_routing."""

from flashinfer.fused_moe.core import trtllm_fp4_block_scale_moe as _api
from flashinfer.trace.solutions._helpers import solution_autotune

definition = "moe_fp4_block_scale_ds_routing"
api = "flashinfer.fused_moe.core.trtllm_fp4_block_scale_moe"
backend = "trtllm"
inputs = (
    "routing_logits",
    "routing_bias",
    "hidden_states",
    "hidden_states_scale",
    "gemm1_weights",
    "gemm1_weights_scale",
    "gemm1_bias",
    "gemm1_alpha",
    "gemm1_beta",
    "gemm1_clamp_limit",
    "gemm2_weights",
    "gemm2_weights_scale",
    "gemm2_bias",
    "output1_scale_scalar",
    "output1_scale_gate_scalar",
    "output2_scale_scalar",
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
    "gemm1_bias": "gemm1_bias",
    "gemm1_alpha": "gemm1_alpha",
    "gemm1_beta": "gemm1_beta",
    "gemm1_clamp_limit": "gemm1_clamp_limit",
    "gemm2_weights": "gemm2_weights",
    "gemm2_weights_scale": "gemm2_weights_scale",
    "gemm2_bias": "gemm2_bias",
    "output1_scale_scalar": "output1_scale_scalar",
    "output1_scale_gate_scalar": "output1_scale_gate_scalar",
    "output2_scale_scalar": "output2_scale_scalar",
    "local_expert_offset": "local_expert_offset",
    "routed_scaling_factor": "routed_scaling_factor",
}
constants = {"top_k": 8, "n_group": 8, "topk_group": 4}


def run(
    routing_logits,
    routing_bias,
    hidden_states,
    hidden_states_scale,
    gemm1_weights,
    gemm1_weights_scale,
    gemm1_bias,
    gemm1_alpha,
    gemm1_beta,
    gemm1_clamp_limit,
    gemm2_weights,
    gemm2_weights_scale,
    gemm2_bias,
    output1_scale_scalar,
    output1_scale_gate_scalar,
    output2_scale_scalar,
    local_expert_offset,
    routed_scaling_factor,
):
    with solution_autotune(
        definition,
        backend,
        routing_logits,
        routing_bias,
        hidden_states,
        hidden_states_scale,
        gemm1_weights,
        gemm1_weights_scale,
        gemm1_bias,
        gemm1_alpha,
        gemm1_beta,
        gemm1_clamp_limit,
        gemm2_weights,
        gemm2_weights_scale,
        gemm2_bias,
        output1_scale_scalar,
        output1_scale_gate_scalar,
        output2_scale_scalar,
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
            gemm1_bias=gemm1_bias,
            gemm1_alpha=gemm1_alpha,
            gemm1_beta=gemm1_beta,
            gemm1_clamp_limit=gemm1_clamp_limit,
            gemm2_weights=gemm2_weights,
            gemm2_weights_scale=gemm2_weights_scale,
            gemm2_bias=gemm2_bias,
            output1_scale_scalar=output1_scale_scalar,
            output1_scale_gate_scalar=output1_scale_gate_scalar,
            output2_scale_scalar=output2_scale_scalar,
            num_experts=routing_logits.shape[1],
            top_k=constants["top_k"],
            n_group=constants["n_group"],
            topk_group=constants["topk_group"],
            intermediate_size=gemm2_weights.shape[2] * 2,
            local_expert_offset=local_expert_offset,
            local_num_experts=gemm1_weights.shape[0],
            routed_scaling_factor=routed_scaling_factor,
            routing_method_type=2,
        )
        if result is not None:
            return result
        raise RuntimeError(
            "moe_fp4_block_scale_ds_routing"
            + " returned None without mutating declared outputs"
        )
