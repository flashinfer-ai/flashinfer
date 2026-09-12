# Copyright (c) 2026 by FlashInfer team.
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

"""Trace metadata for AlphaMoE W8A8 and its offline weight-layout helper."""

from ..template import Const, Scalar, Tensor, TraceTemplate, Var


_WEIGHT_AXES = {
    "num_experts": Const(description="Number of local experts."),
    "hidden_size": Const(description="Input and output hidden width."),
    "gated_size": Const(description="Combined gate and up width."),
    "hidden_blocks": Const(description="128-column weight-scale blocks."),
    "gated_blocks": Const(description="128-row gate/up weight-scale blocks."),
}

_WEIGHT_INPUTS = {
    "gemm1_weights": Tensor(["num_experts", "gated_size", "hidden_size"]),
    "gemm1_weights_scale": Tensor(["num_experts", "gated_blocks", "hidden_blocks"]),
}

_WEIGHT_CONSTRAINTS = [
    "hidden_size == hidden_blocks * 128",
    "gated_size == gated_blocks * 128",
    "gated_size % 256 == 0",
]

alphamoe_interleave_gated_weights_trace = TraceTemplate(
    op_type="moe_preprocess",
    name_prefix="alphamoe_interleave_gated_weights",
    description="Interleave logical [gate; up] rows for the AlphaMoE W8A8 layout.",
    axes=_WEIGHT_AXES,
    inputs=_WEIGHT_INPUTS,
    outputs={
        "interleaved_weights": Tensor(
            ["num_experts", "gated_size", "hidden_size"],
            dtype_from="gemm1_weights",
        ),
        "interleaved_scales": Tensor(
            ["num_experts", "gated_blocks", "hidden_blocks"],
            dtype_from="gemm1_weights_scale",
        ),
    },
    constraints=_WEIGHT_CONSTRAINTS,
    tags=["layout:gate_up_interleaved", "quantization:fp8_block_scale"],
)

alphamoe_fp8_block_scale_aligned_moe_trace = TraceTemplate(
    op_type="moe",
    name_prefix="alphamoe_fp8_block_scale_aligned_moe",
    description="Fused W8A8 MoE over an aligned route plan, accumulating into BF16 output.",
    axes={
        **_WEIGHT_AXES,
        "num_tokens": Var(description="Number of input token rows."),
        "intermediate_size": Const(description="Per-expert intermediate width."),
        "intermediate_blocks": Const(description="128-column down-scale blocks."),
        "plan_capacity": Var(description="Allocated aligned token-id entries."),
        "route_blocks": Var(description="Allocated expert-id blocks."),
        "top_k": Const(description="Routed contributions per token."),
        "one": Const(value=1),
    },
    inputs={
        "hidden_states": Tensor(["num_tokens", "hidden_size"]),
        "hidden_states_scale": Tensor(["num_tokens", "hidden_blocks"]),
        **_WEIGHT_INPUTS,
        "gemm2_weights": Tensor(["num_experts", "hidden_size", "intermediate_size"]),
        "gemm2_weights_scale": Tensor(
            ["num_experts", "hidden_blocks", "intermediate_blocks"]
        ),
        "sorted_token_ids": Tensor(["plan_capacity"]),
        "expert_ids": Tensor(["route_blocks"]),
        "num_tokens_post_padded": Tensor(
            ["one"], description="Device-side valid plan extent."
        ),
        "topk_weights": Tensor(["num_tokens", "top_k"]),
        "top_k": Scalar("int32"),
        "block_m": Scalar("int32", optional=True),
        "routed_scaling_factor": Scalar("float32", optional=True),
        "out": Tensor(
            ["num_tokens", "hidden_size"],
            optional=True,
            description="Initial BF16 accumulator; omitted output starts at zero.",
        ),
    },
    outputs={"output": Tensor(["num_tokens", "hidden_size"], dtype="bfloat16")},
    constraints=[
        *_WEIGHT_CONSTRAINTS,
        "gated_size == 2 * intermediate_size",
        "intermediate_size == intermediate_blocks * 128",
        "block_m >= 8",
        "block_m % 8 == 0",
        "plan_capacity >= route_blocks * block_m",
    ],
    tags=["moe:aligned_plan", "quantization:fp8_block_scale", "arch:sm100_sm103"],
)
