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

"""Trace definitions for MiniMax-H3 MXFP8 / NVFP4 pre-attention and QKV quantize-and-pack."""

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


minimax_h3_nvfp4_pre_attention_trace = TraceTemplate(
    op_type="minimax_h3_nvfp4_pre_attention",
    name_prefix="minimax_h3_nvfp4_pre_attention",
    description=(
        "BF16 RMSNorm and indexed AdaLN with dynamic NVFP4 activation "
        "quantization, prepacked NVFP4 (W4A4) QKV GEMM, Q/K RMSNorm and "
        "split-half NeoX RoPE, followed by destination-major E2M1 and "
        "swizzled E4M3 block-16 packing."
    ),
    axes={
        "M": Var(description="Token rows."),
        "hidden_size": Const(value=5376, abbrev=""),
        "packed_hidden_size": Const(
            value=2688, abbrev="", description="E2M1 nibble-pair bytes per row."
        ),
        "qkv_width": Const(value=21504, abbrev=""),
        "adaln_rows": Const(value=9, abbrev=""),
        "rope_width": Const(value=96, abbrev=""),
        "num_heads": Const(value=56, abbrev=""),
        "P": Const(abbrev="p", description="Destination partitions."),
        "heads_per_destination": Const(abbrev="hdst"),
        "qkv_kinds": Const(value=3, abbrev=""),
        "head_dim": Const(value=128, abbrev="d"),
        "packed_head_dim": Const(
            value=64, abbrev="", description="E2M1 nibble-pair bytes per head row."
        ),
        "weight_scale_elements": Var(),
        "output_scale_stride": Var(),
    },
    inputs={
        "x": Tensor(["M", "hidden_size"]),
        "x_norm_weight": Tensor(["hidden_size"]),
        "adaln_scale": Tensor(["adaln_rows", "hidden_size"]),
        "adaln_shift": Tensor(["adaln_rows", "hidden_size"]),
        "adaln_index": Tensor(["M"]),
        "x_global_scale": Tensor(
            ["1"], description="Float32 NVFP4 global encode scale for x."
        ),
        "qkv_weight_q": Tensor(
            ["qkv_width", "packed_hidden_size"],
            description="Prepacked E2M1 nibble-pair QKV weight.",
        ),
        "qkv_weight_sf": Tensor(
            ["weight_scale_elements"],
            description="Swizzled-128x4 E4M3 block-16 weight scales.",
        ),
        "w_global_scale": Tensor(
            ["1"], description="Float32 NVFP4 global encode scale for the weight."
        ),
        "q_norm_weight": Tensor(["head_dim"]),
        "k_norm_weight": Tensor(["head_dim"]),
        "rope_cos_sin": Tensor(["M", "rope_width"]),
        "out_global_scale": Tensor(
            ["1"],
            description="Float32 NVFP4 global encode scale shared by Q/K/V outputs.",
        ),
        "out_q": Tensor(
            ["P", "M", "heads_per_destination", "qkv_kinds", "packed_head_dim"],
            description="Caller-owned destination-major E2M1 nibble-pair output.",
        ),
        "out_sf": Tensor(
            ["P", "output_scale_stride"],
            description=(
                "Caller-owned destination-major swizzled-128x4 E4M3 scale output."
            ),
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
        "debug_adaln_bf16": Tensor(
            ["M", "hidden_size"],
            dtype="bfloat16",
            optional=True,
            description=(
                "Optional exact BF16 AdaLN intermediate (stage-1 debug output)."
            ),
        ),
    },
    outputs={
        "out_q": Tensor(
            ["P", "M", "heads_per_destination", "qkv_kinds", "packed_head_dim"],
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
    tags=["stage:pre_attention", "status:verified", "dtype:nvfp4"],
)


minimax_h3_qkv_quantize_pack_trace = TraceTemplate(
    op_type="minimax_h3_qkv_quantize_pack",
    name_prefix="minimax_h3_qkv_quantize_pack",
    description=(
        "One-pass destination-major routing of BF16 Q/K/V into the Ulysses "
        "send buffer with per-destination block quantization: NVFP4 (E2M1 "
        "nibble pairs, swizzled E4M3 block-16 scales, static global scale) or "
        "MXFP8 (E4M3 values, swizzled UE8M0 block-32 scales)."
    ),
    axes={
        "M": Var(description="Token rows."),
        "num_heads": Const(value=56, abbrev=""),
        "P": Const(abbrev="p", description="Destination partitions."),
        "heads_per_destination": Const(abbrev="hdst"),
        "qkv_kinds": Const(value=3, abbrev=""),
        "head_dim": Const(value=128, abbrev="d"),
        "packed_head_dim": Const(
            abbrev="pk",
            description=(
                "Bytes per packed head row: 64 (NVFP4 nibble pairs) or 128 (MXFP8)."
            ),
        ),
        "output_scale_stride": Var(),
    },
    inputs={
        "q": Tensor(["M", "num_heads", "head_dim"]),
        "k": Tensor(["M", "num_heads", "head_dim"]),
        "v": Tensor(["M", "num_heads", "head_dim"]),
        "out_global_scale": Tensor(
            ["1"],
            dtype="float32",
            optional=True,
            description=(
                "Float32 static NVFP4 global encode scale (448 * 6 / amax); "
                "absent for MXFP8."
            ),
        ),
        "out_q": Tensor(
            ["P", "M", "heads_per_destination", "qkv_kinds", "packed_head_dim"],
            description=(
                "Caller-owned destination-major packed output: uint8 E2M1 nibble "
                "pairs (NVFP4) or float8_e4m3fn (MXFP8)."
            ),
        ),
        "out_sf": Tensor(
            ["P", "output_scale_stride"],
            description=(
                "Caller-owned per-destination swizzled-128x4 scale tile (E4M3 for "
                "NVFP4, UE8M0 for MXFP8) including zero padding rows."
            ),
        ),
    },
    outputs={
        "out_q": Tensor(
            ["P", "M", "heads_per_destination", "qkv_kinds", "packed_head_dim"],
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
    tags=[
        "stage:pre_attention",
        "status:verified",
        "quantization:fp4",
        "quantization:mxfp8",
    ],
)


__all__ = [
    "minimax_h3_mxfp8_pre_attention_trace",
    "minimax_h3_nvfp4_pre_attention_trace",
    "minimax_h3_qkv_quantize_pack_trace",
]
