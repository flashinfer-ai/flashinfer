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

from .. import env as jit_env
from ..core import JitSpec, current_compilation_context, gen_jit_spec


def gen_w8a16_fp8_gemm_module() -> JitSpec:
    """W8A16 GEMM, Hopper (sm_90a): FP8 e4m3 weight (per-tensor scale) x fp16/bf16 activation."""
    nvcc_flags = current_compilation_context.get_nvcc_flags_list(
        supported_major_versions=[9]
    )
    return gen_jit_spec(
        "w8a16_fp8_gemm",
        [
            jit_env.FLASHINFER_CSRC_DIR / "w8a16_fp8_gemm.cu",
            jit_env.FLASHINFER_CSRC_DIR / "w8a16_fp8_gemm_jit_binding.cu",
        ],
        extra_cuda_cflags=nvcc_flags + ["-DENABLE_BF16", "-DENABLE_FP8"],
    )


def gen_w8a16_fp8_gemm_sm100_module() -> JitSpec:
    """W8A16 GEMM, Blackwell (sm_100a): CUTLASS mixed-input mainloop, same contract."""
    nvcc_flags = current_compilation_context.get_nvcc_flags_list(
        supported_major_versions=[10]
    )
    return gen_jit_spec(
        "w8a16_fp8_gemm_sm100",
        [
            jit_env.FLASHINFER_CSRC_DIR / "w8a16_fp8_gemm_sm100.cu",
            jit_env.FLASHINFER_CSRC_DIR / "w8a16_fp8_gemm_sm100_jit_binding.cu",
        ],
        extra_cuda_cflags=nvcc_flags
        + ["-DENABLE_BF16", "-DENABLE_FP8", "-DCUTLASS_ENABLE_GDC_FOR_SM100=1"],
        extra_cflags=["-DFAST_BUILD"],
    )
