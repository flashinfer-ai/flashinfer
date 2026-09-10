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

"""Trace definition for MiniMax-H3 MXFP8 pre-attention."""

from ..template import Const, Tensor, TraceTemplate, Var


minimax_h3_mxfp8_pre_attention_trace = TraceTemplate(
    op_type="minimax_h3_mxfp8_pre_attention",
    name_prefix="minimax_h3_mxfp8_pre_attention",
    description=(
        "BF16 RMSNorm and indexed AdaLN, prepacked MXFP8 QKV GEMM, Q/K "
        "RMSNorm and split-half NeoX RoPE, followed by destination-major "
        "E4M3 and UE8M0 packing."
    ),
    axes={
        "M": Var(description="Token rows."),
        "hidden_size": Const(value=5376, abbrev=""),
        "qkv_width": Const(value=21504, abbrev=""),
        "adaln_rows": Const(value=9, abbrev=""),
        "rope_width": Const(value=96, abbrev=""),
        "num_heads": Const(value=56, abbrev=""),
        "P": Const(abbrev="p", description="Destination partitions."),
        "heads_per_destination": Const(abbrev="hdst"),
        "qkv_kinds": Const(value=3, abbrev=""),
        "head_dim": Const(value=128, abbrev="d"),
        "weight_scale_elements": Var(),
        "output_scale_stride": Var(),
    },
    inputs={
        "x": Tensor(["M", "hidden_size"]),
        "x_norm_weight": Tensor(["hidden_size"]),
        "adaln_scale": Tensor(["adaln_rows", "hidden_size"]),
        "adaln_shift": Tensor(["adaln_rows", "hidden_size"]),
        "adaln_index": Tensor(["M"]),
        "qkv_weight_q": Tensor(["qkv_width", "hidden_size"]),
        "qkv_weight_sf": Tensor(["weight_scale_elements"]),
        "q_norm_weight": Tensor(["head_dim"]),
        "k_norm_weight": Tensor(["head_dim"]),
        "rope_cos_sin": Tensor(["M", "rope_width"]),
        "out_q": Tensor(
            ["P", "M", "heads_per_destination", "qkv_kinds", "head_dim"],
            description="Caller-owned destination-major E4M3 output.",
        ),
        "out_sf": Tensor(
            ["P", "output_scale_stride"],
            description="Caller-owned destination-major UE8M0 scale output.",
        ),
        "debug_q_bf16": Tensor(
            ["M", "num_heads", "head_dim"],
            dtype="bfloat16",
            optional=True,
        ),
        "debug_k_bf16": Tensor(
            ["M", "num_heads", "head_dim"],
            dtype="bfloat16",
            optional=True,
        ),
    },
    outputs={
        "out_q": Tensor(
            ["P", "M", "heads_per_destination", "qkv_kinds", "head_dim"],
            param="out_q",
            dtype_from="out_q",
        ),
        "out_sf": Tensor(
            ["P", "output_scale_stride"],
            param="out_sf",
            dtype_from="out_sf",
        ),
    },
    constraints=["P * heads_per_destination == 56"],
    tags=["stage:pre_attention", "status:verified", "dtype:mxfp8"],
)


__all__ = ["minimax_h3_mxfp8_pre_attention_trace"]
