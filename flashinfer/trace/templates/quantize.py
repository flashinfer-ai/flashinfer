# Copyright (c) 2025 by FlashInfer team.
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

"""TraceTemplates for FP4 / FP8 quantization APIs."""

from ..template import Const, Scalar, Tensor, TraceTemplate, Var

# ── FP4 quantization (generic) ───────────────────────────────────────────────
# input [M, K]  →  (quantized [M, K/2] uint8 packed,  scales [variable])

_FP4_AXES = {
    "M": Var(description="Number of rows."),
    "K": Const(abbrev="k", description="Number of input columns."),
    "K_packed": Var(
        description="Packed column dimension (K/2 for FP4, two values per uint8).",
    ),
    "num_scale_elems": Var(
        description="Total number of scale factor elements (layout-dependent)."
    ),
    "one": Var(description="Placeholder for shape [1] scalar tensors."),
}

fp4_quantize_trace = TraceTemplate(
    op_type="quantization",
    name_prefix="fp4_quantize",
    description="Generic FP4 quantization: bf16/fp16 input → packed FP4 e2m1fn + block scales.",
    axes=_FP4_AXES,
    inputs={
        "input": Tensor(
            ["M", "K"],
            param="input",
            description="Input tensor, fp16/bf16/fp8_e4m3fn.",
        ),
        "global_scale": Tensor(
            ["one"],
            dtype="float32",
            optional=True,
            description="Optional per-tensor global scale (shape [1]).",
        ),
        "sf_vec_size": Scalar(
            "int32",
            optional=True,
            description="Scale-factor vector size (16 for NVFP4, 32 for MXFP4).",
        ),
    },
    outputs={
        "quantized": Tensor(
            ["M", "K_packed"],
            dtype="uint8",
            description="Packed FP4 output (two e2m1fn values per byte).",
        ),
        "scales": Tensor(
            ["num_scale_elems"],
            dtype="uint8",
            description="Block scale factors packed as uint8 bytes (layout-dependent shape).",
        ),
    },
    constraints=["K_packed == K // 2"],
    tags=["status:verified", "quantization:fp4"],
)

# ── NVFP4 quantization ────────────────────────────────────────────────────────
nvfp4_quantize_trace = TraceTemplate(
    op_type="quantization",
    name_prefix="nvfp4_quantize",
    description="NVFP4 quantization (sf_vec_size=16). Requires a per-tensor global scale.",
    axes=_FP4_AXES,
    inputs={
        "a": Tensor(["M", "K"], description="Input tensor, fp16/bf16/fp8_e4m3fn."),
        "a_global_sf": Tensor(
            ["one"],
            dtype="float32",
            description="Global scale factor, shape [1].",
        ),
        "sf_vec_size": Scalar(
            "int32",
            optional=True,
            description="Scale-factor vector size (fixed at 16 for NVFP4).",
        ),
    },
    outputs={
        "quantized": Tensor(
            ["M", "K_packed"],
            dtype="uint8",
            description="Packed FP4 output.",
        ),
        "scales": Tensor(
            ["num_scale_elems"],
            dtype="uint8",
            description="Block scale factors packed as uint8 bytes (layout-dependent shape).",
        ),
    },
    constraints=["K_packed == K // 2"],
    tags=["status:verified", "quantization:nvfp4"],
)

# ── MXFP4 quantization ────────────────────────────────────────────────────────
mxfp4_quantize_trace = TraceTemplate(
    op_type="quantization",
    name_prefix="mxfp4_quantize",
    description="MXFP4 quantization (sf_vec_size=32, UE8M0 scales). No global scale.",
    axes=_FP4_AXES,
    inputs={
        "a": Tensor(["M", "K"], description="Input tensor, fp16/bf16."),
    },
    outputs={
        "quantized": Tensor(
            ["M", "K_packed"],
            dtype="uint8",
            description="Packed FP4 output.",
        ),
        "scales": Tensor(
            ["num_scale_elems"],
            dtype="uint8",
            description="UE8M0 block scale factors (1 byte per 32-element block).",
        ),
    },
    constraints=["K_packed == K // 2"],
    tags=["status:verified", "quantization:mxfp4"],
)

# ── MXFP8 quantization ────────────────────────────────────────────────────────

mxfp8_quantize_trace = TraceTemplate(
    op_type="quantization",
    name_prefix="mxfp8_quantize",
    description="MXFP8 quantization (block size 32, UE8M0 scales). Output is fp8_e4m3fn.",
    axes={
        "M": Var(description="Number of rows."),
        "K": Const(abbrev="k", description="Number of input columns."),
        "num_scale_elems": Var(
            description="Total number of scale factor elements (layout-dependent)."
        ),
    },
    inputs={
        "input": Tensor(
            ["M", "K"],
            param="input",
            description="Input tensor, fp16/bf16.",
        ),
    },
    outputs={
        "quantized": Tensor(
            ["M", "K"],
            dtype="float8_e4m3fn",
            description="MXFP8 quantized output.",
        ),
        "scales": Tensor(
            ["num_scale_elems"],
            dtype="uint8",
            description="UE8M0 block scale factors (1 byte per 32-element block).",
        ),
    },
    tags=["status:verified", "quantization:mxfp8"],
)
