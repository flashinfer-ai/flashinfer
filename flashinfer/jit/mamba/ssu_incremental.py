"""
Copyright (c) 2025 by FlashInfer team.

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

import os
from typing import Optional

import jinja2
import torch

from .. import env as jit_env
from ..core import JitSpec, gen_jit_spec
from ..utils import write_if_different

# Map torch dtypes to C++ type names
_dtype_map = {
    torch.float16: "half",
    torch.bfloat16: "nv_bfloat16",
    torch.float32: "float",
    torch.int16: "int16_t",
    torch.int32: "int32_t",
    torch.int64: "int64_t",
}

# Map torch dtypes to filename-safe strings
_filename_safe_dtype_map = {
    torch.float16: "f16",
    torch.bfloat16: "bf16",
    torch.float32: "f32",
    torch.int16: "i16",
    torch.int32: "i32",
    torch.int64: "i64",
}


def get_ssu_incremental_uri(
    state_dtype: torch.dtype,
    input_dtype: torch.dtype,
    dt_dtype: torch.dtype,
    weight_dtype: torch.dtype,
    matrixA_dtype: torch.dtype,
    stateIndex_dtype: torch.dtype,
    state_scale_dtype: Optional[torch.dtype],
    dim: int,
    dstate: int,
    ntokens_mtp: int,
    philox_rounds: int = 0,
) -> str:
    s = _filename_safe_dtype_map
    uri = (
        f"ssu_incremental_"
        f"s_{s[state_dtype]}_i_{s[input_dtype]}_dt_{s[dt_dtype]}_w_{s[weight_dtype]}_"
        f"a_{s[matrixA_dtype]}_si_{s[stateIndex_dtype]}_"
        f"d_{dim}_ds_{dstate}_nt_{ntokens_mtp}"
    )
    if state_scale_dtype is not None:
        uri += f"_sc_{s[state_scale_dtype]}"
    if philox_rounds > 0:
        uri += f"_pr_{philox_rounds}"
    return uri


def gen_ssu_incremental_module(
    state_dtype: torch.dtype,
    input_dtype: torch.dtype,
    dt_dtype: torch.dtype,
    weight_dtype: torch.dtype,
    matrixA_dtype: torch.dtype,
    stateIndex_dtype: torch.dtype,
    state_scale_dtype: Optional[torch.dtype],
    dim: int,
    dstate: int,
    ntokens_mtp: int,
    philox_rounds: int = 0,
    extra_cuda_cflags: list = None,
) -> JitSpec:
    uri = get_ssu_incremental_uri(
        state_dtype,
        input_dtype,
        dt_dtype,
        weight_dtype,
        matrixA_dtype,
        stateIndex_dtype,
        state_scale_dtype,
        dim,
        dstate,
        ntokens_mtp,
        philox_rounds,
    )
    gen_directory = jit_env.FLASHINFER_GEN_SRC_DIR / uri
    os.makedirs(gen_directory, exist_ok=True)

    # Render the config .inc
    with open(
        jit_env.FLASHINFER_CSRC_DIR / "ssu_incremental_customize_config.jinja"
    ) as f:
        config_templ = jinja2.Template(f.read())

    state_scale_type = (
        _dtype_map[state_scale_dtype] if state_scale_dtype is not None else "void"
    )
    config_str = config_templ.render(
        state_dtype=_dtype_map[state_dtype],
        input_dtype=_dtype_map[input_dtype],
        dt_dtype=_dtype_map[dt_dtype],
        weight_dtype=_dtype_map[weight_dtype],
        matrixA_dtype=_dtype_map[matrixA_dtype],
        stateIndex_dtype=_dtype_map[stateIndex_dtype],
        dim=dim,
        dstate=dstate,
        ntokens_mtp=ntokens_mtp,
        state_scale_type=state_scale_type,
        philox_rounds=philox_rounds,
    )
    write_if_different(gen_directory / "ssu_incremental_config.inc", config_str)

    # Copy source files to gen directory
    source_paths = []
    for filename in [
        "ssu_incremental.cu",
        "ssu_incremental_kernel_inst.cu",
        "ssu_incremental_jit_binding.cu",
    ]:
        src_path = jit_env.FLASHINFER_CSRC_DIR / filename
        dest_path = gen_directory / filename
        source_paths.append(dest_path)
        with open(src_path, "r") as f:
            source = f.read()
        write_if_different(dest_path, source)

    return gen_jit_spec(
        uri,
        source_paths,
        extra_cuda_cflags=extra_cuda_cflags or [],
    )
