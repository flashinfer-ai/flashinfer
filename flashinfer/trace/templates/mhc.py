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

"""TraceTemplates for mHC operations."""

from ..template import Const, Scalar, Tensor, TraceTemplate, Var


_MHC_AXES: dict[str, Var | Const] = {
    "num_tokens": Var(description="Flattened outer token count."),
    "hc": Const(abbrev="hc", description="mHC residual channel count."),
    "hidden_size": Const(abbrev="h", description="Hidden size."),
}

_MHC_PRE_AXES: dict[str, Var | Const] = {
    **_MHC_AXES,
    "mhc_mix": Const(abbrev="mix", description="mHC mix vector width."),
    "scale_size": Const(abbrev="", description="mHC scale vector length."),
    "one": Var(description="Placeholder for shape [1] output tensors."),
}

_MHC_PRE_SCALARS = {
    "rms_eps": Scalar("float32", optional=True),
    "mhc_pre_eps": Scalar("float32", optional=True),
    "mhc_sinkhorn_eps": Scalar("float32", optional=True),
    "mhc_post_mult_value": Scalar("float32", optional=True),
    "sinkhorn_repeat": Scalar("int32", optional=True),
    "block_size": Scalar("int32", optional=True),
}


mhc_post_trace = TraceTemplate(
    op_type="mhc_post",
    name_prefix="mhc_post",
    description="mHC post mapping for HC=4.",
    axes=_MHC_AXES,
    inputs={
        "x": Tensor(["num_tokens", "hidden_size"], dtype="bfloat16"),
        "residual": Tensor(["num_tokens", "hc", "hidden_size"], dtype="bfloat16"),
        "post_layer_mix": Tensor(["num_tokens", "hc"], dtype="float32"),
        "comb_res_mix": Tensor(["num_tokens", "hc", "hc"], dtype="float32"),
    },
    outputs={
        "out": Tensor(["num_tokens", "hc", "hidden_size"], dtype_from="residual"),
    },
    tags=["fused"],
)


mhc_pre_big_fuse_trace = TraceTemplate(
    op_type="mhc_pre_big_fuse",
    name_prefix="mhc_pre_big_fuse",
    description="mHC pre-map big-fuse using external projection and sqrsum.",
    axes=_MHC_PRE_AXES,
    inputs={
        "dot_mix": Tensor(["num_tokens", "mhc_mix"], dtype="float32"),
        "sqrsum": Tensor(["num_tokens"], dtype="float32"),
        "residual": Tensor(["num_tokens", "hc", "hidden_size"], dtype="bfloat16"),
        "mhc_scale": Tensor(["scale_size"], dtype="float32"),
        "mhc_base": Tensor(["mhc_mix"], dtype="float32"),
        "k": Scalar("int32"),
        **_MHC_PRE_SCALARS,
        "num_splits": Scalar("int32", optional=True),
    },
    outputs={
        "post_mix": Tensor(["num_tokens", "hc", "one"], dtype="float32"),
        "comb_mix": Tensor(["num_tokens", "hc", "hc"], dtype="float32"),
        "layer_input": Tensor(["num_tokens", "hidden_size"], dtype_from="residual"),
    },
    tags=["fused"],
)


mhc_pre_big_fuse_with_prenorm_trace = TraceTemplate(
    op_type="mhc_pre_big_fuse",
    name_prefix="mhc_pre_big_fuse_with_prenorm",
    description="mHC pre-map big-fuse that computes RMS sqrsum from residual.",
    axes=_MHC_PRE_AXES,
    inputs={
        "dot_mix": Tensor(["num_tokens", "mhc_mix"], dtype="float32"),
        "residual": Tensor(["num_tokens", "hc", "hidden_size"], dtype="bfloat16"),
        "mhc_scale": Tensor(["scale_size"], dtype="float32"),
        "mhc_base": Tensor(["mhc_mix"], dtype="float32"),
        **_MHC_PRE_SCALARS,
    },
    outputs={
        "post_mix": Tensor(["num_tokens", "hc", "one"], dtype="float32"),
        "comb_mix": Tensor(["num_tokens", "hc", "hc"], dtype="float32"),
        "layer_input": Tensor(["num_tokens", "hidden_size"], dtype_from="residual"),
    },
    tags=["fused", "prenorm"],
)
