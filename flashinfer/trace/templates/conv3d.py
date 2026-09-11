"""
Copyright (c) 2026 by the PatchShift Conv3d contributors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""TraceTemplate for the SM100a PatchShift Conv3d compute primitive."""

from ..template import Const, Scalar, Tensor, TraceTemplate, Var


patchshift_conv3d_trace = TraceTemplate(
    op_type="conv3d",
    name_prefix="patchshift_conv3d",
    description=(
        "BF16 3x3x3 Conv3d with padding 1, stride 1, dilation 1, groups 1; "
        "input and output use NDHWC layout and weights are prepacked."
    ),
    axes={
        "batch_size": Var(),
        "depth": Var(),
        "height": Var(),
        "width": Var(),
        "in_channels": Const(abbrev="c"),
        "out_channels": Const(abbrev="k"),
        "packed_size": Var(),
        "workspace_size": Var(),
    },
    inputs={
        "input": Tensor(
            ["batch_size", "depth", "height", "width", "in_channels"],
            dtype="bfloat16",
        ),
        "packed_weight": Tensor(["packed_size"], dtype="bfloat16"),
        "workspace": Tensor(["workspace_size"], dtype="uint8"),
        "out_channels": Scalar("int64"),
    },
    outputs={
        "output": Tensor(
            ["batch_size", "depth", "height", "width", "out_channels"],
            dtype_from="input",
        ),
    },
    constraints=["in_channels % 8 == 0", "out_channels > 0"],
    tags=["status:experimental", "architecture:sm100a"],
)
