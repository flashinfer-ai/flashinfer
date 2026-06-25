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

"""TraceTemplate for the contributed FuseMoE Blackwell kernel.

The op_type matches the existing ``moe`` family (so this slots into the
same benchmark suite as ``trtllm_fp8_block_scale_moe``); ``name_prefix``
distinguishes the contributed kernel from the trtllm-gen reference.

The reference implementation is the DSV3 routing reference shared by
``trtllm_fp8_block_scale_moe_ds_routing_trace``.
"""

from ..template import Const, Scalar, Tensor, TraceTemplate, Var
from .moe import _trtllm_fp8_block_scale_moe_ds_routing_reference


fusemoe_blackwell_fp8_dsv3_trace = TraceTemplate(
    op_type="moe",
    name_prefix="fusemoe_blackwell_fp8_dsv3",
    description=(
        "Contributed FuseMoE Blackwell FP8 block-scale kernel with "
        "DeepSeek-V3 routing. Hand-tuned for SM100; shape constants are "
        "JIT-rendered per call site via Jinja."
    ),
    axes={
        "seq_len": Var(description="Sequence length (number of tokens)."),
        "num_experts": Const(description="Total number of experts.", abbrev=""),
        "top_k": Const(description="Experts routed to per token.", abbrev="topk"),
        "n_group": Const(description="Number of expert groups.", abbrev="ng"),
        "topk_group": Const(description="Groups kept after group top-k.", abbrev="kg"),
        "num_local_experts": Const(
            description="Local experts on this rank.", abbrev="e"
        ),
        "hidden_size": Const(description="Hidden dimension size.", abbrev="h"),
        "intermediate_size": Const(
            description="MoE intermediate layer size.", abbrev="i"
        ),
        "gemm1_out_size": Const(
            description="Output size of GEMM1 (W13). Equals 2 * intermediate_size.",
            abbrev="",
        ),
        "num_hidden_blocks": Const(
            description="Number of FP8 quant blocks along hidden_size (block_size=128).",
            abbrev="",
        ),
        "num_intermediate_blocks": Const(
            description="Number of FP8 quant blocks along intermediate_size.",
            abbrev="",
        ),
        "num_gemm1_out_blocks": Const(
            description="Number of FP8 quant blocks along gemm1_out_size.",
            abbrev="",
        ),
    },
    inputs={
        "routing_logits": Tensor(
            ["seq_len", "num_experts"],
            description="Routing logits for expert selection.",
        ),
        "routing_bias": Tensor(
            ["num_experts"],
            description="Per-expert routing bias.",
        ),
        "hidden_states": Tensor(
            ["seq_len", "hidden_size"],
            description="Input hidden states (FP8 quantised).",
        ),
        "hidden_states_scale": Tensor(
            ["num_hidden_blocks", "seq_len"],
            description="Block-wise scaling factors for hidden states.",
        ),
        "gemm1_weights": Tensor(
            ["num_local_experts", "gemm1_out_size", "hidden_size"],
            description="GEMM1 weights (gate + up).",
        ),
        "gemm1_weights_scale": Tensor(
            ["num_local_experts", "num_gemm1_out_blocks", "num_hidden_blocks"],
            description="Block-wise scaling factors for GEMM1 weights.",
        ),
        "gemm2_weights": Tensor(
            ["num_local_experts", "hidden_size", "intermediate_size"],
            description="GEMM2 weights (down projection).",
        ),
        "gemm2_weights_scale": Tensor(
            ["num_local_experts", "num_hidden_blocks", "num_intermediate_blocks"],
            description="Block-wise scaling factors for GEMM2 weights.",
        ),
        "top_k": Scalar("int32", description="Number of experts per token."),
        "n_group": Scalar("int32", description="Number of expert groups."),
        "topk_group": Scalar("int32", description="Groups kept after group top-k."),
        "local_expert_offset": Scalar(
            "int32", description="Offset of local experts in global space."
        ),
        "routed_scaling_factor": Scalar(
            "float32", description="DSV3 routed scaling factor."
        ),
    },
    outputs={
        "output": Tensor(
            ["seq_len", "hidden_size"],
            dtype="bfloat16",
            description="Final MoE output.",
        ),
    },
    tags=["status:verified", "quantization:float8_e4m3fn"],
    reference=_trtllm_fp8_block_scale_moe_ds_routing_reference,
)
