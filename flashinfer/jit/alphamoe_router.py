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
from .core import JitSpec, gen_jit_spec


@functools.cache
def gen_alphamoe_router_module() -> JitSpec:
    """Build the JIT spec for the fused AlphaMoE gating router module."""
    return gen_jit_spec(
        "alphamoe_router",
        [
            jit_env.FLASHINFER_CSRC_DIR / "alphamoe_router" / "alphamoe_router.cu",
            jit_env.FLASHINFER_CSRC_DIR
            / "alphamoe_router"
            / "alphamoe_router_jit_binding.cu",
        ],
        extra_cuda_cflags=["-lineinfo"],
    )
