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
from ..core import JitSpec, gen_jit_spec


def gen_w8a16_fp8_gemm_module() -> JitSpec:
    """W8A16 GEMM: FP8 e4m3 weight (per-tensor scale) x fp16/bf16 activation."""
    return gen_jit_spec(
        "w8a16_fp8_gemm",
        [
            jit_env.FLASHINFER_CSRC_DIR / "w8a16_fp8_gemm.cu",
            jit_env.FLASHINFER_CSRC_DIR / "w8a16_fp8_gemm_jit_binding.cu",
        ],
        extra_cuda_cflags=["-DENABLE_BF16", "-DENABLE_FP8"],
    )
