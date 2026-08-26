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
from ..core import JitSpec, gen_jit_spec, sm100a_nvcc_flags


def gen_dual_bf16_weight_gemm_sm100_module() -> JitSpec:
    """Build the exact-SM100 dual-BF16 weight GEMM module."""

    return gen_jit_spec(
        "dual_bf16_weight_gemm_sm100",
        [
            jit_env.FLASHINFER_CSRC_DIR / "dual_bf16_weight_gemm_sm100.cu",
        ],
        extra_cuda_cflags=sm100a_nvcc_flags
        + [
            "-DENABLE_BF16",
            "-DCUTLASS_ENABLE_GDC_FOR_SM100=1",
        ],
        extra_cflags=["-DFAST_BUILD"],
    )
