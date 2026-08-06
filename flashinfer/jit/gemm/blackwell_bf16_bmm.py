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
from ..core import JitSpec, gen_jit_spec, sm103a_nvcc_flags


def gen_blackwell_bf16_bmm_module() -> JitSpec:
    """Build the frozen SM103a CAKE-generated BF16 BMM dispatcher."""

    return gen_jit_spec(
        "blackwell_bf16_bmm_cake_sm103a",
        [
            jit_env.FLASHINFER_CSRC_DIR / "blackwell_bf16_bmm.cu",
            jit_env.FLASHINFER_CSRC_DIR / "blackwell_bf16_bmm_sm103.cu",
        ],
        extra_cuda_cflags=sm103a_nvcc_flags + ["--use_fast_math"],
    )


__all__ = ["gen_blackwell_bf16_bmm_module"]
