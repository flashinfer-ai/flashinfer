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

from typing import List

from .core import JitSpec, gen_jit_spec, current_compilation_context, sm90a_nvcc_flags
from . import env as jit_env


def gen_mxfp8_quantization_module(nvcc_flags: List[str], device_arch: str) -> JitSpec:
    return gen_jit_spec(
        f"mxfp8_quantization_sm{device_arch}",
        [
            jit_env.FLASHINFER_CSRC_DIR
            / "nv_internal/tensorrt_llm/thop/fp8Quantize.cpp",
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal/cpp/kernels/quantization.cu",
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal/cpp/common/envUtils.cpp",
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal/cpp/common/logger.cpp",
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal/cpp/common/stringUtils.cpp",
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal/cpp/common/tllmException.cpp",
        ],
        extra_cuda_cflags=nvcc_flags
        + [
            "-DENABLE_BF16",
            "-DENABLE_FP8",
            "-DENABLE_FP4",
        ],
        extra_cflags=[
            "-DENABLE_BF16",
            "-DENABLE_FP8",
            "-DENABLE_FP4",
        ],
        extra_include_paths=[
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal",
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal" / "include",
        ],
    )


def gen_mxfp8_quantization_sm100_module() -> JitSpec:
    return gen_mxfp8_quantization_module(
        current_compilation_context.get_nvcc_flags_list(
            supported_major_versions=None, map_sm107_to_100f=True
        ),
        "100",
    )


def gen_mxfp8_quantization_sm80_module() -> JitSpec:
    return gen_mxfp8_quantization_module(
        current_compilation_context.get_nvcc_flags_list(supported_major_versions=[8])
        + ["-DFLASHINFER_FP4_SW_CVT", "-DFLASHINFER_FP4_QUANT_NO_TMA"],
        "80",
    )


def gen_mxfp8_quantization_sm90_module() -> JitSpec:
    return gen_mxfp8_quantization_module(
        sm90a_nvcc_flags + ["-DFLASHINFER_FP4_SW_CVT"], "90"
    )
