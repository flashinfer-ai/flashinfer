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

"""TraceTemplates for diffusion-model operations."""

from ..template import Const, Scalar, Tensor, TraceTemplate, Var


minimax_h3_bf16_pre_attention_trace = TraceTemplate(
    op_type="minimax_h3_bf16_pre_attention",
    name_prefix="minimax_h3_bf16_pre_attention",
    description=(
        "MiniMax-H3 fused BF16 RMSNorm, indexed AdaLN, QKV projection, "
        "Q/K RMSNorm, partial 3-D RoPE, and destination-major packing."
    ),
    axes={
        "num_tokens": Var(description="Local token count M."),
        "hidden_size": Const(value=5376, abbrev="h"),
        "adaln_rows": Const(value=9, abbrev="ar"),
        "qkv_kinds": Const(value=3, abbrev="qkv"),
        "head_dim": Const(value=128, abbrev="d"),
        "qkv_width": Const(value=21504, abbrev="w"),
        "rope_dim": Const(value=96, abbrev="rope"),
        "ulysses_degree": Var(description="Ulysses destination count."),
        "heads_per_destination": Var(description="Attention heads per destination."),
    },
    inputs={
        "x": Tensor(["num_tokens", "hidden_size"]),
        "x_norm_weight": Tensor(["hidden_size"]),
        "adaln_scale": Tensor(["adaln_rows", "hidden_size"]),
        "adaln_shift": Tensor(["adaln_rows", "hidden_size"]),
        "adaln_index": Tensor(["num_tokens"]),
        "qkv_weight": Tensor(["qkv_width", "hidden_size"]),
        "q_norm_weight": Tensor(["head_dim"]),
        "k_norm_weight": Tensor(["head_dim"]),
        "rope_cos_sin": Tensor(["num_tokens", "rope_dim"]),
        "out": Tensor(
            [
                "ulysses_degree",
                "num_tokens",
                "heads_per_destination",
                "qkv_kinds",
                "head_dim",
            ],
            description="Caller-owned destination buffer (mutated in place).",
        ),
        "ulysses_degree": Scalar("int32"),
        "eps": Scalar("float32", optional=True),
    },
    outputs={
        "out": Tensor(
            [
                "ulysses_degree",
                "num_tokens",
                "heads_per_destination",
                "qkv_kinds",
                "head_dim",
            ],
            dtype_from="out",
        ),
    },
    constraints=[
        "ulysses_degree in (1, 2, 4, 8)",
        "qkv_width == ulysses_degree * heads_per_destination * qkv_kinds * head_dim",
        "rope_dim <= head_dim",
    ],
    tags=["stage:pre-attention", "dtype:bf16", "status:experimental"],
)
