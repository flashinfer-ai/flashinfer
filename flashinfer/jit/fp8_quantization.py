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

from .core import JitSpec, gen_jit_spec, current_compilation_context
from . import env as jit_env


def gen_mxfp8_quantization_sm100_module() -> JitSpec:
    return gen_jit_spec(
        "mxfp8_quantization_sm100",
        [
            jit_env.FLASHINFER_CSRC_DIR
            / "nv_internal/tensorrt_llm/thop/fp8Quantize.cpp",
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal/cpp/kernels/quantization.cu",
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal/cpp/common/envUtils.cpp",
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal/cpp/common/logger.cpp",
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal/cpp/common/stringUtils.cpp",
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal/cpp/common/tllmException.cpp",
        ],
        extra_cuda_cflags=current_compilation_context.get_nvcc_flags_list(
            supported_major_versions=None
        )
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
