"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""Generate padding-mask trace examples without executing CUDA kernels."""

from pathlib import Path

import torch

from flashinfer.fused_moe import trtllm_bf16_moe, trtllm_fp4_block_scale_moe


def generate_moe_token_mask_traces(save_dir):
    n, h, inter, experts = 17, 256, 128, 8
    common = dict(
        routing_logits=torch.empty(n, experts),
        routing_bias=None,
        num_experts=experts,
        top_k=2,
        n_group=None,
        topk_group=None,
        intermediate_size=inter,
        local_expert_offset=0,
        local_num_experts=experts,
        routing_method_type=0,
        token_mask=torch.arange(n) % 2 == 1,
        save_dir=save_dir,
    )
    trtllm_bf16_moe.fi_trace(
        **common,
        hidden_states=torch.empty(n, h, dtype=torch.bfloat16),
        gemm1_weights=torch.empty(experts, 2 * inter, h, dtype=torch.bfloat16),
        gemm2_weights=torch.empty(experts, h, inter, dtype=torch.bfloat16),
        name="trtllm_bf16_moe_token_mask_topk2_e8_h256_i128",
    )
    fp8 = torch.float8_e4m3fn
    trtllm_fp4_block_scale_moe.fi_trace(
        **common,
        hidden_states=torch.empty(n, h // 2, dtype=torch.uint8),
        hidden_states_scale=torch.empty(n, h // 16, dtype=fp8),
        gemm1_weights=torch.empty(experts, 2 * inter, h // 2, dtype=torch.uint8),
        gemm1_weights_scale=torch.empty(experts, 2 * inter, h // 16, dtype=fp8),
        gemm2_weights=torch.empty(experts, h, inter // 2, dtype=torch.uint8),
        gemm2_weights_scale=torch.empty(experts, h, inter // 16, dtype=fp8),
        gemm1_bias=None,
        gemm2_bias=None,
        gemm1_alpha=None,
        gemm1_beta=None,
        gemm1_clamp_limit=None,
        output1_scale_scalar=None,
        output1_scale_gate_scalar=None,
        output2_scale_scalar=None,
        name="moe_fp4_block_scale_token_mask_topk2_e8_h256_i128",
    )


if __name__ == "__main__":
    generate_moe_token_mask_traces(Path(__file__).with_name("fi_trace_out"))
