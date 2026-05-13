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

from ...compilation_context import CompilationContext
from .. import env as jit_env
from ..core import JitSpec, gen_jit_spec
from ..utils import write_if_different


# Arch-gating constants.  All kernel variants need cp.async +
# mma.sync.m16n8k16 → SM80+ (Ampere baseline).  The fp8 e4m3fn state
# path additionally needs the `__nv_fp8_e4m3(x)` constructor which
# compiles to `cvt.rn.satfinite.e4m3.f32` PTX (SM89+, Ada/Hopper/
# Blackwell — *not* SM80/SM86 Ampere).
#
# Philox SR PTX (`cvt.rs.f16x2.f32`, `cvt.rs.satfinite.e4m3x4.f32`) is
# already `#ifdef __CUDA_ARCH__ >= 1000`-guarded in the kernel source,
# so no JIT-level gating is needed for it; the wrapper asserts SM100+
# at runtime when `rand_seed` is provided with fp16 / fp8 state.
_SUPPORTED_MAJORS = {8, 9, 10, 11, 12}
_FP8_MIN_CC = (8, 9)


def _arch_flags_for_state_dtype(state_dtype: torch.dtype) -> list[str]:
    """Build ``-gencode`` flags filtered for this kernel's arch requirements.

    Uses ``CompilationContext.TARGET_CUDA_ARCHS`` (structured ``(major,
    minor_str)`` tuples) directly — the public ``get_nvcc_flags_list``
    only takes ``supported_major_versions``, but for fp8 we need a
    finer ``(major, minor) >= (8, 9)`` bound to keep SM89 (Ada
    Lovelace) while excluding SM80 / SM86 (Ampere).
    """
    ctx = CompilationContext()
    archs = {
        (maj, mn) for (maj, mn) in ctx.TARGET_CUDA_ARCHS if maj in _SUPPORTED_MAJORS
    }
    if state_dtype == torch.float8_e4m3fn:
        # Strip the suffix from the minor str ("9" / "0a" / "0f" → numeric prefix).
        def _numeric_minor(mn: str) -> int:
            n = ""
            for ch in mn:
                if not ch.isdigit():
                    break
                n += ch
            return int(n) if n else 0

        archs = {
            (maj, mn) for (maj, mn) in archs if (maj, _numeric_minor(mn)) >= _FP8_MIN_CC
        }
        if not archs:
            raise RuntimeError(
                "fp8_e4m3fn state requires SM 89+ (Ada Lovelace / Hopper / "
                "Blackwell) for `cvt.rn.satfinite.e4m3.f32` PTX; no supported "
                "CUDA archs in the current FLASHINFER_CUDA_ARCH_LIST / "
                "auto-detected device list satisfy this."
            )
    elif not archs:
        raise RuntimeError(
            "checkpointing_ssu requires SM 80+ (Ampere or newer); no supported "
            "CUDA archs in the current FLASHINFER_CUDA_ARCH_LIST / auto-detected "
            "device list satisfy this."
        )
    return [
        f"-gencode=arch=compute_{maj}{mn},code=sm_{maj}{mn}"
        for maj, mn in sorted(archs)
    ] + ctx.COMMON_NVCC_FLAGS


# Map torch dtypes to C++ type names
_dtype_map = {
    torch.float16: "half",
    torch.bfloat16: "nv_bfloat16",
    torch.float32: "float",
    torch.int8: "int8_t",
    torch.int16: "int16_t",
    torch.int32: "int32_t",
    torch.int64: "int64_t",
    torch.float8_e4m3fn: "__nv_fp8_e4m3",
}

# Map torch dtypes to filename-safe strings
_filename_safe_dtype_map = {
    torch.float16: "f16",
    torch.bfloat16: "bf16",
    torch.float32: "f32",
    torch.int8: "i8",
    torch.int16: "i16",
    torch.int32: "i32",
    torch.int64: "i64",
    torch.float8_e4m3fn: "e4m3",
}


def get_checkpointing_ssu_uri(
    state_dtype: torch.dtype,
    input_dtype: torch.dtype,
    dt_dtype: torch.dtype,
    weight_dtype: torch.dtype,
    matrixA_dtype: torch.dtype,
    stateIndex_dtype: torch.dtype,
    state_scale_dtype: Optional[torch.dtype],
    dim: int,
    dstate: int,
    npredicted: int,
    max_window: int,
    heads_per_group: int,
    philox_rounds: int = 0,
) -> str:
    s = _filename_safe_dtype_map
    uri = (
        f"checkpointing_ssu_"
        f"s_{s[state_dtype]}_i_{s[input_dtype]}_dt_{s[dt_dtype]}_w_{s[weight_dtype]}_"
        f"a_{s[matrixA_dtype]}_si_{s[stateIndex_dtype]}_"
        f"d_{dim}_ds_{dstate}_np_{npredicted}_mw_{max_window}_hpg_{heads_per_group}"
    )
    if state_scale_dtype is not None:
        uri += f"_sc_{s[state_scale_dtype]}"
    if philox_rounds > 0:
        uri += f"_pr_{philox_rounds}"
    return uri


def gen_checkpointing_ssu_module(
    state_dtype: torch.dtype,
    input_dtype: torch.dtype,
    dt_dtype: torch.dtype,
    weight_dtype: torch.dtype,
    matrixA_dtype: torch.dtype,
    stateIndex_dtype: torch.dtype,
    state_scale_dtype: Optional[torch.dtype],
    dim: int,
    dstate: int,
    npredicted: int,
    max_window: int,
    heads_per_group: int,
    philox_rounds: int = 0,
    extra_cuda_cflags: list = None,
) -> JitSpec:
    uri = get_checkpointing_ssu_uri(
        state_dtype,
        input_dtype,
        dt_dtype,
        weight_dtype,
        matrixA_dtype,
        stateIndex_dtype,
        state_scale_dtype,
        dim,
        dstate,
        npredicted,
        max_window,
        heads_per_group,
        philox_rounds,
    )
    gen_directory = jit_env.FLASHINFER_GEN_SRC_DIR / uri
    os.makedirs(gen_directory, exist_ok=True)

    # Render the config .inc
    with open(
        jit_env.FLASHINFER_CSRC_DIR / "checkpointing_ssu_customize_config.jinja"
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
        npredicted=npredicted,
        max_window=max_window,
        heads_per_group=heads_per_group,
        state_scale_type=state_scale_type,
        philox_rounds=philox_rounds,
    )
    write_if_different(gen_directory / "checkpointing_ssu_config.inc", config_str)

    # Copy source files to gen directory
    source_paths = []
    for filename in [
        "checkpointing_ssu.cu",
        "checkpointing_ssu_kernel_inst.cu",
        "checkpointing_ssu_jit_binding.cu",
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
        extra_cuda_cflags=_arch_flags_for_state_dtype(state_dtype)
        + (extra_cuda_cflags or []),
    )
