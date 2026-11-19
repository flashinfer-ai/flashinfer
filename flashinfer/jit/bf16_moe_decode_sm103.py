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

from . import env as jit_env
from .core import JitSpec, gen_jit_spec, sm103a_nvcc_flags


def gen_bf16_moe_decode_sm103_module() -> JitSpec:
    """JIT spec of the fused SM103 BF16 MoE decode kernels of the specialized
    MoELayer backend (``fused_moe/backends/specialized/kernels/bf16_decode_sm103_glm52``).
    The source is the TVM-FFI entry point; the kernels are in the framework-agnostic
    header ``include/flashinfer/fused_moe/bf16_moe_decode_sm103.cuh`` (on the default
    include path). SM103 only; exact math (no fast-math), since the kernels
    reproduce the trtllm-gen data path."""
    return gen_jit_spec(
        "bf16_moe_decode_sm103",
        [jit_env.FLASHINFER_CSRC_DIR / "fused_moe" / "bf16_moe_decode_sm103.cu"],
        extra_cuda_cflags=sm103a_nvcc_flags + ["-lineinfo"],
        extra_ldflags=["-lcuda"],
        use_fast_math=False,
    )
