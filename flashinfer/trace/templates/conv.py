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

"""Trace templates for convolution operations."""

from ..template import Const, Scalar, Tensor, TraceTemplate, Var


prepare_nvfp4_conv3d_weight_trace = TraceTemplate(
    op_type="prepare_nvfp4_conv3d_weight",
    name_prefix="prepare_nvfp4_conv3d_weight_sm120",
    description=(
        "Prepare a BF16 KCTRS Conv3d weight as packed E2M1 data and "
        "128x4-swizzled E4M3 scales for SM120."
    ),
    axes={
        "output_channels": Const(abbrev="k"),
        "input_channels": Const(abbrev="c"),
        "filter_t": Const(abbrev="t"),
        "filter_r": Const(abbrev="r"),
        "filter_s": Const(abbrev="s"),
        "packed_input_channels": Var(),
        "weight_scale_bytes": Var(),
        "global_scale_elements": Var(),
    },
    inputs={
        "weight": Tensor(
            [
                "output_channels",
                "input_channels",
                "filter_t",
                "filter_r",
                "filter_s",
            ]
        ),
        "weight_global_scale": Tensor(
            ["global_scale_elements"],
            optional=True,
            dtype="float32",
        ),
    },
    outputs={
        "packed_weight": Tensor(
            [
                "output_channels",
                "filter_t",
                "filter_r",
                "filter_s",
                "packed_input_channels",
            ],
            dtype="uint8",
        ),
        "weight_scale": Tensor(["weight_scale_bytes"], dtype="uint8"),
        "weight_global_scale": Tensor(
            ["global_scale_elements"],
            dtype="float32",
        ),
    },
    constraints=[
        "packed_input_channels * 2 == input_channels",
        "input_channels % 128 == 0",
        "output_channels % 128 == 0",
        "filter_t == 3",
        "filter_r == 3",
        "filter_s == 3",
        "weight_scale_bytes == ((output_channels + 127) // 128) * "
        "(((input_channels * 27 // 16) + 3) // 4) * 512",
        "global_scale_elements == 1",
    ],
    tags=["sm120", "nvfp4", "weight-preparation", "status:verified"],
)


conv3d_nvfp4_trace = TraceTemplate(
    op_type="conv3d_nvfp4",
    name_prefix="conv3d_nvfp4_sm120",
    description=(
        "SM120 3x3x3 block-scaled NVFP4 Conv3d with dynamic activation "
        "quantization, FP32 accumulation, and BF16 output."
    ),
    axes={
        "batch": Const(abbrev="n"),
        "input_channels": Const(abbrev="c"),
        "input_depth": Var(),
        "input_height": Var(),
        "input_width": Var(),
        "output_channels": Const(abbrev="k"),
        "filter_t": Const(abbrev="t"),
        "filter_r": Const(abbrev="r"),
        "filter_s": Const(abbrev="s"),
        "packed_input_channels": Var(),
        "weight_scale_bytes": Var(),
        "scale_elements": Const(abbrev=""),
        "output_depth": Var(),
        "output_height": Var(),
        "output_width": Var(),
    },
    inputs={
        "input": Tensor(
            [
                "batch",
                "input_channels",
                "input_depth",
                "input_height",
                "input_width",
            ]
        ),
        "packed_weight": Tensor(
            [
                "output_channels",
                "filter_t",
                "filter_r",
                "filter_s",
                "packed_input_channels",
            ]
        ),
        "weight_scale": Tensor(["weight_scale_bytes"]),
        "input_global_scale": Tensor(["scale_elements"]),
        "weight_global_scale": Tensor(["scale_elements"]),
        "bias": Tensor(["output_channels"], optional=True),
        "stride": Scalar("int32[3]"),
        "padding": Scalar("int32[3]"),
        "dilation": Scalar("int32[3]"),
        "groups": Scalar("int32"),
        "out": Tensor(
            [
                "batch",
                "output_channels",
                "output_depth",
                "output_height",
                "output_width",
            ],
            optional=True,
        ),
    },
    outputs={
        "output": Tensor(
            [
                "batch",
                "output_channels",
                "output_depth",
                "output_height",
                "output_width",
            ],
            dtype_from="input",
        )
    },
    constraints=[
        "batch == 1",
        "input_channels == 2 * packed_input_channels",
        "input_channels % 128 == 0",
        "output_channels % 128 == 0",
        "filter_t == 3",
        "filter_r == 3",
        "filter_s == 3",
        "weight_scale_bytes == ((output_channels + 127) // 128) * "
        "(((input_channels * 27 // 16) + 3) // 4) * 512",
        "scale_elements == 1",
        "output_depth == input_depth - 2",
    ],
    tags=["sm120", "nvfp4", "status:verified"],
)


__all__ = ["conv3d_nvfp4_trace", "prepare_nvfp4_conv3d_weight_trace"]
