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

"""TraceTemplate for recurrent Key-Driven Attention (KDA)."""

from ..template import Const, Scalar, Tensor, TraceTemplate, Var


recurrent_kda_trace = TraceTemplate(
    op_type="kda",
    name_prefix="recurrent_kda",
    description=(
        "Recurrent Key-Driven Attention decode/spec-decode plus exact "
        "FlashKDA-compatible ordinary prefill on B200, with "
        "per-key-dimension gating and recurrent BF16 state."
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
        "num_sequences_plus_one": Var(
            description="Number of packed cumulative-length entries."
        ),
    },
    inputs={
        "q": Tensor(["batch_size", "seq_len", "num_q_heads", "head_dim"]),
        "k": Tensor(["batch_size", "seq_len", "num_q_heads", "head_dim"]),
        "v": Tensor(["batch_size", "seq_len", "num_v_heads", "head_dim"]),
        "g": Tensor(["batch_size", "seq_len", "num_v_heads", "head_dim"]),
        "beta": Tensor(["batch_size", "seq_len", "num_v_heads"]),
        "A_log": Tensor(
            ["num_q_heads"],
            dtype="float32",
            optional=True,
            description="FP32 per-query-head log decay rate.",
        ),
        "dt_bias": Tensor(
            ["num_q_heads", "head_dim"],
            dtype="float32",
            optional=True,
            description=(
                "FP32 per-head/key decay bias; flattened [H*D] or [H,D] "
                "storage is accepted by the API."
            ),
        ),
        "initial_state": Tensor(
            ["state_pool_size", "num_v_heads", "head_dim", "head_dim"],
            dtype="bfloat16",
            optional=True,
        ),
        "initial_state_source": Tensor(
            ["source_pool_size", "num_v_heads", "head_dim", "head_dim"],
            dtype="bfloat16",
            optional=True,
            description="Read-only committed-state pool.",
        ),
        "initial_state_indices": Tensor(
            ["num_sequences"],
            dtype="int32",
            optional=True,
            description="Committed-state slot selected for each sequence.",
        ),
        "cu_seqlens": Tensor(
            ["num_sequences_plus_one"],
            optional=True,
            description="Packed cumulative sequence lengths (int32 or int64).",
        ),
        "num_accepted_tokens": Tensor(
            ["num_sequences"],
            dtype="int32",
            optional=True,
            description="Accepted-token counts for speculative decode.",
        ),
        "seq_order": Tensor(
            ["num_sequences"],
            dtype="int32",
            optional=True,
            description=(
                "Packed-prefill sequence permutation, normally descending "
                "by sequence length."
            ),
        ),
        "scale": Scalar("float32", optional=True),
        "output_final_state": Scalar("int32", optional=True),
        "use_qk_l2norm_in_kernel": Scalar("int32", optional=True),
        "use_gate_in_kernel": Scalar("int32", optional=True),
        "lower_bound": Scalar("float32", optional=True),
        "num_spec_tokens": Scalar("int32", optional=True),
        "beta_is_logit": Scalar("int32", optional=True),
    },
    outputs={
        "output": Tensor(
            ["batch_size", "seq_len", "num_v_heads", "head_dim"],
            dtype_from="q",
            param="output",
        ),
        "final_state": Tensor(
            ["state_pool_size", "num_v_heads", "head_dim", "head_dim"],
            dtype="bfloat16",
            optional=True,
        ),
    },
    constraints=[
        "num_v_heads % num_q_heads == 0",
        "head_dim in (64, 128)",
    ],
    tags=["stage:decode", "stage:prefill", "status:verified"],
)
