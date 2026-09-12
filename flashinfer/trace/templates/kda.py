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

"""TraceTemplate for recurrent Key-Driven Attention (KDA) decode."""

from ..template import Const, Scalar, Tensor, TraceTemplate, Var


recurrent_kda_trace = TraceTemplate(
    op_type="kda",
    name_prefix="recurrent_kda",
    description=(
        "Recurrent Key-Driven Attention decode with per-key-dimension gating "
        "and an optional read-only committed-state source."
    ),
    axes={
        "batch_size": Var(description="Number of input batch rows."),
        "seq_len": Var(description="Tokens carried by each input batch row."),
        "num_q_heads": Const(description="Number of query and key heads.", abbrev="q"),
        "num_v_heads": Const(description="Number of value heads.", abbrev="v"),
        "head_dim": Const(
            description="Query, key, and value head dimension.", abbrev="d"
        ),
        "state_pool_size": Var(description="Number of writable state slots."),
        "source_pool_size": Var(description="Number of committed-state slots."),
        "num_sequences": Var(description="Number of state-source indices."),
        "num_checkpoints": Var(description="Number of packed prefill checkpoints."),
        "num_checkpoint_offsets": Var(
            description="Number of packed checkpoint cumulative offsets."
        ),
        "kg_dim": Var(description="Key/gate cache row width (2 * head_dim)."),
    },
    inputs={
        "q": Tensor(["batch_size", "seq_len", "num_q_heads", "head_dim"]),
        "k": Tensor(["batch_size", "seq_len", "num_q_heads", "head_dim"]),
        "v": Tensor(["batch_size", "seq_len", "num_v_heads", "head_dim"]),
        "g": Tensor(["batch_size", "seq_len", "num_v_heads", "head_dim"]),
        "beta": Tensor(["batch_size", "seq_len", "num_v_heads"]),
        "initial_state": Tensor(
            ["state_pool_size", "num_v_heads", "head_dim", "head_dim"],
            optional=True,
        ),
        "initial_state_source": Tensor(
            ["source_pool_size", "num_v_heads", "head_dim", "head_dim"],
            optional=True,
            description="Read-only committed-state pool.",
        ),
        "initial_state_indices": Tensor(
            ["num_sequences"],
            optional=True,
            description="Committed-state slot selected for each sequence.",
        ),
        "ssm_state_indices": Tensor(
            ["num_sequences"],
            optional=True,
            description="Writable state-pool slot selected for each prefill sequence.",
        ),
        "state_checkpoints": Tensor(
            ["num_checkpoints", "num_v_heads", "head_dim", "head_dim"],
            optional=True,
            description="Caller-owned packed KDA pre-block state output.",
        ),
        "checkpoint_cu_starts": Tensor(
            ["num_checkpoint_offsets"],
            optional=True,
            description="Per-sequence cumulative packed checkpoint counts.",
        ),
        "scale": Scalar("float32", optional=True),
        "output_final_state": Scalar("int32", optional=True),
        "use_qk_l2norm_in_kernel": Scalar("int32", optional=True),
        "use_gate_in_kernel": Scalar("int32", optional=True),
        "lower_bound": Scalar("float32", optional=True),
        "num_spec_tokens": Scalar("int32", optional=True),
        "beta_is_logit": Scalar("int32", optional=True),
        "disable_state_update": Scalar("int32", optional=True),
        "correction_cache": Tensor(
            ["source_pool_size", "num_v_heads", "seq_len", "head_dim"],
            optional=True,
            description=(
                "Frozen-verify only: slot-indexed float32 per-token "
                "delta-rule corrections for a commit/recovery kernel."
            ),
        ),
        "kg_cache": Tensor(
            ["source_pool_size", "num_v_heads", "seq_len", "kg_dim"],
            optional=True,
            description=(
                "Frozen-verify only: slot-indexed (normalized key | raw "
                "gate) cache; kg_dim == 2 * head_dim."
            ),
        ),
        "checkpoint_every_n_tokens": Scalar("int32", optional=True),
    },
    outputs={
        "output": Tensor(
            ["batch_size", "seq_len", "num_v_heads", "head_dim"],
            dtype_from="q",
        ),
        "final_state": Tensor(
            ["state_pool_size", "num_v_heads", "head_dim", "head_dim"],
            dtype="bfloat16",
            optional=True,
        ),
        "state_checkpoints": Tensor(
            ["num_checkpoints", "num_v_heads", "head_dim", "head_dim"],
            dtype="bfloat16",
            optional=True,
            param="state_checkpoints",
        ),
    },
    constraints=[
        "num_v_heads % num_q_heads == 0",
        "head_dim in (64, 128)",
        "num_checkpoint_offsets == num_sequences + 1",
    ],
    tags=["stage:decode", "status:verified"],
)


packed_kda_decode_trace = TraceTemplate(
    op_type="kda",
    name_prefix="packed_kda_decode",
    description=(
        "Serving-native Kimi K3 T=1 recurrent decode from a packed, "
        "post-convolution QKV row. The recurrent state pool is mutated in place."
    ),
    axes={
        "batch_size": Var(description="Number of one-token decode rows."),
        # The public output is rank four, but its optional caller-owned buffer
        # is the only tensor that carries this dimension.
        "singleton": Const(
            description="Fixed single-token output dimension.", abbrev="", value=1
        ),
        "num_heads": Const(description="Number of KDA heads.", abbrev="h"),
        "head_dim": Const(description="KDA head dimension.", abbrev="d"),
        "mixed_width": Const(description="Width of the packed QKV row.", abbrev=""),
        "gate_width": Const(
            description="Width of the raw per-channel gate.", abbrev=""
        ),
        "state_pool_size": Var(description="Number of writable recurrent-state slots."),
    },
    inputs={
        "mixed_qkv": Tensor(["batch_size", "mixed_width"]),
        "raw_gate": Tensor(["batch_size", "gate_width"]),
        "raw_beta": Tensor(["batch_size", "num_heads"]),
        "A_log": Tensor(["num_heads"]),
        "dt_bias": Tensor(["gate_width"]),
        "state": Tensor(["state_pool_size", "num_heads", "head_dim", "head_dim"]),
        "state_indices": Tensor(["batch_size"]),
        "output": Tensor(
            ["batch_size", "singleton", "num_heads", "head_dim"],
            optional=True,
            description="Optional caller-owned output allocation.",
        ),
    },
    outputs={
        "output": Tensor(
            ["batch_size", "singleton", "num_heads", "head_dim"],
            dtype_from="mixed_qkv",
        ),
        "state": Tensor(
            ["state_pool_size", "num_heads", "head_dim", "head_dim"],
            dtype_from="state",
            param="state",
            description="Updated recurrent-state pool (in place).",
        ),
    },
    constraints=[
        "num_heads == 12",
        "head_dim == 128",
        "singleton == 1",
        "mixed_width == 3 * num_heads * head_dim",
        "gate_width == num_heads * head_dim",
    ],
    tags=["stage:decode", "status:verified"],
)


fused_kda_decode_trace = TraceTemplate(
    op_type="kda",
    name_prefix="fused_kda_decode",
    description=(
        "Kimi width-four causal convolution, recurrent KDA update, and "
        "gated RMSNorm fused into one decode kernel. The convolution and "
        "recurrent state pools are mutated in-place."
    ),
    axes={
        "num_rows": Var(description="Number of decode rows."),
        "num_heads": Const(description="Number of KDA heads.", abbrev="h"),
        "head_dim": Const(description="KDA head dimension.", abbrev="d"),
        "singleton": Const(description="Leading singleton dimension.", abbrev=""),
        "projection_groups": Const(
            description="Packed QKV projection groups.", abbrev=""
        ),
        "hidden_size": Const(
            description="Channels in one Q, K, or V projection.", abbrev=""
        ),
        "qkv_width": Const(
            description="Width of the packed QKV projection.", abbrev=""
        ),
        "conv_width": Const(description="Depthwise convolution width.", abbrev=""),
        "conv_history": Const(
            description="Cached convolution history length.", abbrev=""
        ),
        "num_slots": Var(description="Number of cache slots."),
    },
    inputs={
        "x": Tensor(["num_rows", "qkv_width"]),
        "weight": Tensor(["projection_groups", "conv_width", "hidden_size"]),
        "conv_state": Tensor(["num_slots", "qkv_width", "conv_history"]),
        "raw_gate": Tensor(["singleton", "num_rows", "num_heads", "head_dim"]),
        "raw_beta": Tensor(["singleton", "num_rows", "num_heads"]),
        "A_log": Tensor(["num_heads"]),
        "dt_bias": Tensor(["hidden_size"]),
        "state_indices": Tensor(["num_rows"]),
        "state": Tensor(["num_slots", "num_heads", "head_dim", "head_dim"]),
        "output_gate": Tensor(["num_rows", "num_heads", "head_dim"]),
        "norm_weight": Tensor(["head_dim"]),
        "lower_bound": Scalar("float32", optional=True),
        "norm_eps": Scalar("float32", optional=True),
    },
    outputs={
        "output": Tensor(
            ["singleton", "num_rows", "num_heads", "head_dim"], dtype_from="x"
        ),
    },
    constraints=[
        "qkv_width == 3 * num_heads * head_dim",
        "hidden_size == num_heads * head_dim",
        "singleton == 1",
        "projection_groups == 3",
        "head_dim == 128",
        "num_heads in (12, 24, 32, 48, 96)",
        "conv_width == 4",
        "conv_history == 3",
    ],
    tags=["stage:decode", "status:verified"],
)
