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

from ..template import Const, Scalar, Tensor, TraceTemplate, Var


mxfp8_attention_sm120_fwd_trace = TraceTemplate(
    op_type="attention",
    name_prefix="mxfp8_attention_sm120_fwd",
    description=(
        "Ragged varlen per-tensor-FP8 prefill attention on SM120/SM121 "
        "(warp-specialized persistent MXFP8 kernel, kUniformFp8 mode)."
    ),
    axes={
        "batch_plus_one": Var(),
        "total_q": Var(),
        "total_kv": Var(),
        "num_qo_heads": Var(),
        "num_kv_heads": Var(),
        "head_dim": Const(),
    },
    inputs={
        "q": Tensor(["total_q", "num_qo_heads", "head_dim"]),
        "k": Tensor(["total_kv", "num_kv_heads", "head_dim"]),
        "v": Tensor(["total_kv", "num_kv_heads", "head_dim"]),
        "qo_indptr": Tensor(["batch_plus_one"]),
        "kv_indptr": Tensor(["batch_plus_one"]),
        "sm_scale": Scalar("float"),
        "q_scale": Scalar("float"),
        "k_scale": Scalar("float"),
        "v_scale": Scalar("float"),
        "causal": Scalar("bool"),
    },
    outputs={
        "out": Tensor(["total_q", "num_qo_heads", "head_dim"]),
        "lse": Tensor(["total_q", "num_qo_heads"]),
    },
    constraints=[
        "num_qo_heads % num_kv_heads == 0",
        "head_dim == 128",
    ],
    tags=["sm120", "mxfp8", "fp8", "ragged"],
)


mxfp8_attention_sm120_run_trace = TraceTemplate(
    op_type="attention",
    name_prefix="mxfp8_attention_sm120_run",
    description=(
        "Run step of the SM120/SM121 ragged per-tensor-FP8 prefill wrapper "
        "(MXFP8AttentionSM120Wrapper.run; plan() amortizes the host-side "
        "padding layout and LPT work-list build across layers)."
    ),
    axes={
        "total_q": Var(),
        "total_kv": Var(),
        "num_qo_heads": Var(),
        "num_kv_heads": Var(),
        "head_dim": Const(),
    },
    inputs={
        "q": Tensor(["total_q", "num_qo_heads", "head_dim"]),
        "k": Tensor(["total_kv", "num_kv_heads", "head_dim"]),
        "v": Tensor(["total_kv", "num_kv_heads", "head_dim"]),
        "sm_scale": Scalar("float"),
        "q_scale": Scalar("float"),
        "k_scale": Scalar("float"),
        "v_scale": Scalar("float"),
    },
    outputs={
        "out": Tensor(["total_q", "num_qo_heads", "head_dim"]),
        "lse": Tensor(["total_q", "num_qo_heads"]),
    },
    constraints=[
        "num_qo_heads % num_kv_heads == 0",
        "head_dim == 128",
    ],
    tags=["sm120", "mxfp8", "fp8", "ragged"],
)
