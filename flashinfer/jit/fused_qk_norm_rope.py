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

from . import env as jit_env
from .core import JitSpec, gen_jit_spec


def gen_fused_qk_norm_rope_module() -> JitSpec:
    """JIT spec for the fused QK RMSNorm + RoPE op.

    The kernel uses ``__nv_bfloat16`` / ``__half`` intrinsics directly, so no
    extra preprocessor gates are required beyond the FP16/BF16 flags already
    added to every JIT build by :mod:`flashinfer.jit.core`.
    """
    return gen_jit_spec(
        "fused_qk_norm_rope",
        [
            jit_env.FLASHINFER_CSRC_DIR / "fused_qk_norm_rope.cu",
            jit_env.FLASHINFER_CSRC_DIR / "fused_qk_norm_rope_jit_binding.cu",
        ],
    )
