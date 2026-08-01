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

import functools

from . import env as jit_env
from .core import JitSpec, current_compilation_context, gen_jit_spec


@functools.cache
def gen_sm120_direct_fused_moe_module() -> JitSpec:
    """Build the JIT spec for the low-token SM120 BF16 fused MoE kernel."""
    nvcc_flags = current_compilation_context.get_nvcc_flags_list(
        supported_major_versions=[12]
    )
    return gen_jit_spec(
        "sm120_direct_fused_moe",
        [
            jit_env.FLASHINFER_CSRC_DIR / "fused_moe" / "sm120_direct_fused_moe.cu",
            jit_env.FLASHINFER_CSRC_DIR
            / "fused_moe"
            / "sm120_direct_fused_moe_jit_binding.cu",
        ],
        extra_cuda_cflags=[
            *nvcc_flags,
            "-DFLASHINFER_ENABLE_BF16",
            "--use_fast_math",
            "-lineinfo",
        ],
    )
