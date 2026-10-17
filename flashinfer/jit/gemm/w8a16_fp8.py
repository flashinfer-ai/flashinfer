"""
Copyright (c) 2026 by FlashInfer team.

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

import shutil

import jinja2
import torch

from .. import env as jit_env
from ..core import JitSpec, current_compilation_context, gen_jit_spec
from ..utils import dtype_map, filename_safe_dtype_map, write_if_different

# DLPack dtype-code constant (csrc/tvm_ffi_utils.h) the launcher checks act.dtype against.
_ACT_DTYPE_CODE = {torch.float16: "float16_code", torch.bfloat16: "bfloat16_code"}


def _gen_sources(uri: str, act_dtype: torch.dtype, cu_files: list[str]) -> list:
    """Render the per-dtype config header next to copies of the launcher sources.

    One module per activation dtype: the launcher is a template over the activation
    type, so fixing it here instantiates half the kernel variants per module and
    halves the compile time of a first call (the other dtype compiles only if used).
    """
    if act_dtype not in _ACT_DTYPE_CODE:
        raise ValueError(
            f"w8a16_fp8_gemm activation must be fp16 or bf16, got {act_dtype}"
        )
    gen_directory = jit_env.FLASHINFER_GEN_SRC_DIR / uri
    gen_directory.mkdir(parents=True, exist_ok=True)
    with open(jit_env.FLASHINFER_CSRC_DIR / "w8a16_fp8_gemm_config.jinja") as f:
        config = jinja2.Template(f.read()).render(
            act_ctype=dtype_map[act_dtype], act_dtype_code=_ACT_DTYPE_CODE[act_dtype]
        )
    write_if_different(gen_directory / "w8a16_fp8_gemm_config.inc", config)
    sources = []
    for fname in cu_files:
        # copy2 keeps source mtimes; a fresh mtime forces a full ninja rebuild (#4782)
        shutil.copy2(jit_env.FLASHINFER_CSRC_DIR / fname, gen_directory / fname)
        sources.append(gen_directory / fname)
    return sources


def gen_w8a16_fp8_gemm_module(act_dtype: torch.dtype = torch.bfloat16) -> JitSpec:
    """W8A16 GEMM, Hopper (sm_90a): FP8 e4m3 weight (per-tensor scale) x one 16-bit
    activation dtype (``act_dtype``, fp16 or bf16; one module per dtype)."""
    nvcc_flags = current_compilation_context.get_nvcc_flags_list(
        supported_major_versions=[9]
    )
    uri = f"w8a16_fp8_gemm_{filename_safe_dtype_map[act_dtype]}"
    return gen_jit_spec(
        uri,
        _gen_sources(
            uri, act_dtype, ["w8a16_fp8_gemm.cu", "w8a16_fp8_gemm_jit_binding.cu"]
        ),
        extra_cuda_cflags=nvcc_flags + ["-DENABLE_BF16", "-DENABLE_FP8"],
    )


def gen_w8a16_fp8_gemm_sm100_module(
    act_dtype: torch.dtype = torch.bfloat16,
) -> JitSpec:
    """W8A16 GEMM, Blackwell (sm_100a): CUTLASS mixed-input mainloop, same contract."""
    nvcc_flags = current_compilation_context.get_nvcc_flags_list(
        supported_major_versions=[10]
    )
    uri = f"w8a16_fp8_gemm_sm100_{filename_safe_dtype_map[act_dtype]}"
    return gen_jit_spec(
        uri,
        _gen_sources(
            uri,
            act_dtype,
            ["w8a16_fp8_gemm_sm100.cu", "w8a16_fp8_gemm_sm100_jit_binding.cu"],
        ),
        extra_cuda_cflags=nvcc_flags
        + ["-DENABLE_BF16", "-DENABLE_FP8", "-DCUTLASS_ENABLE_GDC_FOR_SM100=1"],
        extra_cflags=["-DFAST_BUILD"],
    )
