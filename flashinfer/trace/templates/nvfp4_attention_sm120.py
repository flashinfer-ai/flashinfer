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


nvfp4_attention_sm120_quantize_qkv_trace = TraceTemplate(
    op_type="attention",
    name_prefix="nvfp4_attention_sm120_quantize_qkv",
    description="Preprocess and quantize dense Q/K/V tensors for the SM120 NVFP4 attention kernel.",
    axes={
        "batch_size": Var(),
        "num_heads": Var(),
        "seq_len": Var(),
        "head_dim": Const(),
        "packed_head_dim": Var(),
        "scale_head_dim": Var(),
        "packed_seq_len": Var(),
        "scale_seq_len": Var(),
        "correction_seq_len": Var(),
    },
    inputs={
        "q": Tensor(["batch_size", "num_heads", "seq_len", "head_dim"]),
        "k": Tensor(["batch_size", "num_heads", "seq_len", "head_dim"]),
        "v": Tensor(["batch_size", "num_heads", "seq_len", "head_dim"]),
        "per_block_mean": Scalar("bool"),
    },
    outputs={
        "q_fp4": Tensor(["batch_size", "num_heads", "seq_len", "packed_head_dim"]),
        "k_fp4": Tensor(["batch_size", "num_heads", "seq_len", "packed_head_dim"]),
        "v_fp4_t": Tensor(["batch_size", "num_heads", "head_dim", "packed_seq_len"]),
        "q_scale": Tensor(["batch_size", "num_heads", "seq_len", "scale_head_dim"]),
        "k_scale": Tensor(["batch_size", "num_heads", "seq_len", "scale_head_dim"]),
        "v_scale_t": Tensor(["batch_size", "num_heads", "head_dim", "scale_seq_len"]),
        "qk_correction": Tensor(
            ["batch_size", "num_heads", "correction_seq_len", "seq_len"]
        ),
    },
    constraints=[
        "head_dim == 2 * packed_head_dim",
        "head_dim == 16 * scale_head_dim",
    ],
    tags=["sm120", "nvfp4"],
)


nvfp4_attention_sm120_fwd_trace = TraceTemplate(
    op_type="attention",
    name_prefix="nvfp4_attention_sm120_fwd",
    description="Run the SM120 NVFP4 attention forward kernel on pre-quantized Q/K/V tensors.",
    axes={
        "batch_size": Var(),
        "num_heads": Var(),
        "seq_len": Var(),
        "head_dim": Const(),
        "packed_head_dim": Const(),
        "scale_head_dim": Const(),
        "packed_seq_len": Var(),
        "scale_seq_len": Var(),
        "correction_seq_len": Var(),
    },
    inputs={
        "q_fp4": Tensor(["batch_size", "num_heads", "seq_len", "packed_head_dim"]),
        "k_fp4": Tensor(["batch_size", "num_heads", "seq_len", "packed_head_dim"]),
        "v_fp4_t": Tensor(["batch_size", "num_heads", "head_dim", "packed_seq_len"]),
        "q_scale": Tensor(["batch_size", "num_heads", "seq_len", "scale_head_dim"]),
        "k_scale": Tensor(["batch_size", "num_heads", "seq_len", "scale_head_dim"]),
        "v_scale_t": Tensor(["batch_size", "num_heads", "head_dim", "scale_seq_len"]),
        "qk_correction": Tensor(
            ["batch_size", "num_heads", "correction_seq_len", "seq_len"]
        ),
        "sm_scale": Scalar("float32"),
        "causal": Scalar("bool"),
        "per_block_mean": Scalar("bool"),
    },
    outputs={
        "out": Tensor(["batch_size", "num_heads", "seq_len", "head_dim"]),
        "lse": Tensor(["batch_size", "num_heads", "seq_len"]),
    },
    tags=["sm120", "nvfp4"],
)
