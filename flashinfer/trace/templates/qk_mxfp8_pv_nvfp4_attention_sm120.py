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


qk_mxfp8_pv_nvfp4_attention_sm120_quantize_qkv_trace = TraceTemplate(
    op_type="attention",
    name_prefix="qk_mxfp8_pv_nvfp4_attention_sm120_quantize_qkv",
    description="Preprocess and quantize dense Q/K/V tensors for the SM120 QK MXFP8/PV NVFP4 attention kernel.",
    axes={
        "batch_size": Var(),
        "num_qo_heads": Var(),
        "num_kv_heads": Var(),
        "qo_len": Var(),
        "kv_len": Var(),
        "padded_qo_len": Var(),
        "padded_kv_len": Var(),
        "head_dim": Const(),
        "scale_head_dim": Var(),
        "packed_kv_len": Var(),
        "scale_kv_len": Var(),
    },
    inputs={
        "q": Tensor(["batch_size", "num_qo_heads", "qo_len", "head_dim"]),
        "k": Tensor(["batch_size", "num_kv_heads", "kv_len", "head_dim"]),
        "v": Tensor(["batch_size", "num_kv_heads", "kv_len", "head_dim"]),
    },
    outputs={
        "q_fp8": Tensor(
            ["batch_size", "num_qo_heads", "padded_qo_len", "head_dim"],
            dtype="float8_e4m3fn",
        ),
        "k_fp8": Tensor(
            ["batch_size", "num_kv_heads", "padded_kv_len", "head_dim"],
            dtype="float8_e4m3fn",
        ),
        "v_fp4_t": Tensor(
            ["batch_size", "num_kv_heads", "head_dim", "packed_kv_len"],
            dtype="uint8",
        ),
        "q_scale": Tensor(
            ["batch_size", "num_qo_heads", "padded_qo_len", "scale_head_dim"],
            dtype="uint8",
        ),
        "k_scale": Tensor(
            ["batch_size", "num_kv_heads", "padded_kv_len", "scale_head_dim"],
            dtype="uint8",
        ),
        "v_scale_t": Tensor(
            ["batch_size", "num_kv_heads", "head_dim", "scale_kv_len"],
            dtype="float8_e4m3fn",
        ),
    },
    constraints=[
        "head_dim == 32 * scale_head_dim",
        "padded_qo_len % 128 == 0",
        "padded_kv_len % 128 == 0",
        "padded_qo_len >= qo_len",
        "padded_qo_len - qo_len < 128",
        "padded_kv_len >= kv_len",
        "padded_kv_len - kv_len < 128",
        "padded_kv_len == 2 * packed_kv_len",
        "padded_kv_len == 16 * scale_kv_len",
        "num_kv_heads > 0",
        "num_qo_heads >= num_kv_heads",
        "num_qo_heads % num_kv_heads == 0",
    ],
    tags=["sm120", "mxfp8", "nvfp4"],
)


qk_mxfp8_pv_nvfp4_attention_sm120_fwd_trace = TraceTemplate(
    op_type="attention",
    name_prefix="qk_mxfp8_pv_nvfp4_attention_sm120_fwd",
    description="Run the SM120 QK MXFP8/PV NVFP4 attention forward kernel on pre-quantized Q/K/V tensors.",
    axes={
        "batch_size": Var(),
        "num_qo_heads": Var(),
        "num_kv_heads": Var(),
        "padded_qo_len": Var(),
        "padded_kv_len": Var(),
        "head_dim": Const(),
        "scale_head_dim": Const(),
        "packed_kv_len": Var(),
        "scale_kv_len": Var(),
    },
    inputs={
        "q_fp8": Tensor(["batch_size", "num_qo_heads", "padded_qo_len", "head_dim"]),
        "k_fp8": Tensor(["batch_size", "num_kv_heads", "padded_kv_len", "head_dim"]),
        "v_fp4_t": Tensor(["batch_size", "num_kv_heads", "head_dim", "packed_kv_len"]),
        "q_scale": Tensor(
            ["batch_size", "num_qo_heads", "padded_qo_len", "scale_head_dim"]
        ),
        "k_scale": Tensor(
            ["batch_size", "num_kv_heads", "padded_kv_len", "scale_head_dim"]
        ),
        "v_scale_t": Tensor(["batch_size", "num_kv_heads", "head_dim", "scale_kv_len"]),
        "sm_scale": Scalar("float32"),
        "causal": Scalar("bool"),
        "out": Tensor(
            ["batch_size", "num_qo_heads", "padded_qo_len", "head_dim"],
            optional=True,
        ),
        "lse": Tensor(
            ["batch_size", "num_qo_heads", "padded_qo_len"],
            optional=True,
        ),
        "out_dtype": Scalar("dtype", optional=True),
        "unpadded_q_len": Scalar("int32", optional=True),
        "unpadded_k_len": Scalar("int32", optional=True),
        "return_lse": Scalar(
            "bool", optional=True, description="Bool: also compute and return LSE."
        ),
    },
    outputs={
        "out": Tensor(
            ["batch_size", "num_qo_heads", "padded_qo_len", "head_dim"],
            param="out",
            dtype="bfloat16",
            dtype_from="out",
            dtype_from_scalar="out_dtype",
        ),
        "lse": Tensor(
            ["batch_size", "num_qo_heads", "padded_qo_len"],
            param="lse",
            dtype="float32",
            optional=True,
        ),
    },
    constraints=[
        "head_dim == 128",
        "head_dim == 32 * scale_head_dim",
        "padded_qo_len % 128 == 0",
        "padded_kv_len % 128 == 0",
        "padded_kv_len == 2 * packed_kv_len",
        "padded_kv_len == 16 * scale_kv_len",
        "num_kv_heads > 0",
        "num_qo_heads >= num_kv_heads",
        "num_qo_heads % num_kv_heads == 0",
    ],
    tags=["sm120", "mxfp8", "nvfp4"],
)
