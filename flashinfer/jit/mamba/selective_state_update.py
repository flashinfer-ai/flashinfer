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

import jinja2
import torch

from ...compilation_context import CompilationContext
from .. import env as jit_env
from ..core import JitSpec, gen_jit_spec
from ..utils import write_if_different

# Map torch dtypes to C++ type names
_dtype_map = {
    torch.float16: "half",
    torch.bfloat16: "nv_bfloat16",
    torch.float32: "float",
    torch.int32: "int32_t",
    torch.int64: "int64_t",
}

# Map torch dtypes to filename-safe strings
_filename_safe_dtype_map = {
    torch.float16: "f16",
    torch.bfloat16: "bf16",
    torch.float32: "f32",
    torch.int32: "i32",
    torch.int64: "i64",
}


def get_selective_state_update_uri(
    state_dtype: torch.dtype,
    input_dtype: torch.dtype,
    weight_dtype: torch.dtype,
    matrixA_dtype: torch.dtype,
    stateIndex_dtype: torch.dtype,
    dim: int,
    dstate: int,
    ntokens_mtp: int,
) -> str:
    s = _filename_safe_dtype_map
    return (
        f"selective_state_update_"
        f"s_{s[state_dtype]}_i_{s[input_dtype]}_w_{s[weight_dtype]}_"
        f"a_{s[matrixA_dtype]}_si_{s[stateIndex_dtype]}_"
        f"d_{dim}_ds_{dstate}_nt_{ntokens_mtp}"
    )


def _gen_module(
    uri: str,
    state_dtype: torch.dtype,
    input_dtype: torch.dtype,
    weight_dtype: torch.dtype,
    matrixA_dtype: torch.dtype,
    stateIndex_dtype: torch.dtype,
    dim: int,
    dstate: int,
    ntokens_mtp: int,
    extra_cuda_cflags: list = None,
) -> JitSpec:
    gen_directory = jit_env.FLASHINFER_GEN_SRC_DIR / uri
    os.makedirs(gen_directory, exist_ok=True)

    # Render the config .inc
    with open(
        jit_env.FLASHINFER_CSRC_DIR / "selective_state_update_customize_config.jinja"
    ) as f:
        config_templ = jinja2.Template(f.read())

    config_str = config_templ.render(
        state_dtype=_dtype_map[state_dtype],
        input_dtype=_dtype_map[input_dtype],
        weight_dtype=_dtype_map[weight_dtype],
        matrixA_dtype=_dtype_map[matrixA_dtype],
        stateIndex_dtype=_dtype_map[stateIndex_dtype],
        dim=dim,
        dstate=dstate,
        ntokens_mtp=ntokens_mtp,
    )
    write_if_different(gen_directory / "selective_state_update_config.inc", config_str)

    # Copy source files to gen directory (so they can #include the config.inc)
    source_paths = []
    for filename in [
        "selective_state_update.cu",
        "selective_state_update_kernel_inst.cu",
        "flashinfer_mamba_binding.cu",
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


def gen_selective_state_update_module(
    state_dtype: torch.dtype,
    input_dtype: torch.dtype,
    weight_dtype: torch.dtype,
    matrixA_dtype: torch.dtype,
    stateIndex_dtype: torch.dtype,
    dim: int,
    dstate: int,
    ntokens_mtp: int,
) -> JitSpec:
    uri = get_selective_state_update_uri(
        state_dtype,
        input_dtype,
        weight_dtype,
        matrixA_dtype,
        stateIndex_dtype,
        dim,
        dstate,
        ntokens_mtp,
    )
    return _gen_module(
        uri,
        state_dtype,
        input_dtype,
        weight_dtype,
        matrixA_dtype,
        stateIndex_dtype,
        dim,
        dstate,
        ntokens_mtp,
    )


def gen_selective_state_update_sm90_module(
    state_dtype: torch.dtype,
    input_dtype: torch.dtype,
    weight_dtype: torch.dtype,
    matrixA_dtype: torch.dtype,
    stateIndex_dtype: torch.dtype,
    dim: int,
    dstate: int,
    ntokens_mtp: int,
) -> JitSpec:
    uri = (
        get_selective_state_update_uri(
            state_dtype,
            input_dtype,
            weight_dtype,
            matrixA_dtype,
            stateIndex_dtype,
            dim,
            dstate,
            ntokens_mtp,
        )
        + "_sm90"
    )
    compilation_context = CompilationContext()
    nvcc_flags = compilation_context.get_nvcc_flags_list(
        supported_major_versions=[9, 10, 11, 12]
    )
    nvcc_flags += ["-DFLASHINFER_MAMBA_ENABLE_SM90"]
    return _gen_module(
        uri,
        state_dtype,
        input_dtype,
        weight_dtype,
        matrixA_dtype,
        stateIndex_dtype,
        dim,
        dstate,
        ntokens_mtp,
        extra_cuda_cflags=nvcc_flags,
    )
